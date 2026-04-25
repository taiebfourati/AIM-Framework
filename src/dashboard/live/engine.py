"""
engine.py
=========

Live streaming engine that drives the real AIMP / RTP framework with real
Simu5G rows, at a user-controlled wall-clock rate, with pluggable drift
injections applied in-line before each sample reaches ``rtp.observe()``.

Scope implemented in this version
---------------------------------
* Replay-with-injection of ``simu5g_real_simulation_results.csv`` (same
  clipping + phase-shuffle approach as the integration test).
* Real AIMP / RTP / ATM wiring. The three MTP variants (MTP-L, MTP-E,
  MTP-C) are all wired and can be driven manually from the UI, or left
  on "auto" which follows the ATMPolicy's severity-based logic.
* MTP-E uses a sqlite-backed MLflow tracking URI so it works without a
  separate MLflow server.
* MTP-C trains on a historical corpus built from the full CSV (all 15
  runs, z-scored), with GradientBoosting + GridSearch as the heavier
  algorithm that distinguishes it from MTP-L's quick RF fine-tune.
* Golden-holdout NDT: disjoint Simu5G runs (stable-3/4 + drift-4) act as
  the NDT validation set with *ground-truth* labels derived from raw
  network state. NDT now computes a pseudo-label score (LIB/LOB) and a
  ground-truth score (golden), makes the pass/fail decision on the
  ground-truth score, and emits both so the dashboard can show the
  +0.357 bias collapsing.
* Closed-loop simulation toggle: when enabled, after ATM deploys a new
  model the engine linearly decays the injection sliders to zero over
  ``CLOSED_LOOP_DECAY_S`` seconds — simulating what MTP-C/Policy Engine
  would do in a real network. The user can always override by moving a
  slider during decay, which cancels the animation.
* Honest reference refit: on every successful deploy the detector
  reference window is replaced with a fresh sample from the current
  (post-retrain) LIB snapshot, so DDD/CDD stop firing on the new
  steady-state rather than being perpetually compared to the ancient
  stable-phase reference.

Design
------
* ``SampleSource`` is an abstract iterator over KPI rows. Today's impl
  (``Simu5GCsvSource``) reads ``simu5g_real_simulation_results.csv``.
* ``InjectionState`` is a single mutable dict the UI writes to via REST.
  Each tick the engine reads it, perturbs the row, then calls observe().
* A single ``threading.Thread`` runs the observe loop. Events are pushed
  into a ``queue.Queue`` that the FastAPI server drains and relays to
  connected WebSockets.

Thread-safety
-------------
* ``InjectionState`` is guarded by a lock; UI writers and the engine
  reader both acquire it.
* The event queue is thread-safe (``queue.Queue``).
* ``EngineState`` snapshots are produced inside the engine thread and
  shipped as plain dicts — consumers never touch engine internals.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Local imports — repo root is injected by server before first import
from aimp import AIMP, AIMPPolicy, RTPComposer, MTPComposer
from atm.atm import ATMPolicy, MTPVariant
from atm.mtp_l import MTPLocal
from atm.mtp_c import MTPCloud

# Closed-loop RAN (Level-2) — actuator + live physics simulator
from dashboard.live.ran_simulator import RANSimulator
from dashboard.live.ran_actuator  import RANActuator

log = logging.getLogger("dashboard.live.engine")

FEATURE_COLS = ["rsrp_dbm", "sinr_db", "throughput_mbps", "delay_ms"]

# 3GPP clipping ranges, identical to test_aimp_real_simu5g.py
RSRP_RANGE = (-156.0, -31.0)
SINR_RANGE = (-23.0, 40.0)
TPUT_RANGE = (0.0, 1000.0)
LAT_RANGE  = (1.0, 100.0)

# Runs reserved as NDT golden holdout — never seen during training
# (these are index values of the "run" column; each is a complete
# Simu5G seed with its own channel trajectory).
GOLDEN_RUN_INDICES = {"stable": (3, 4), "drift": (4,)}

CLOSED_LOOP_DECAY_S = 3.0   # wall-clock seconds to decay sliders to 0

# Warmup window: how many stream samples before we let automatic
# detector-triggered retrains actually fire.  Under-filled LIB/LOB +
# early-life detector instabilities (DPD Isolation Forest seeing the
# first batch as anomalous, CPD's shadow correlation returning NaN)
# otherwise push degenerate candidates through NDT and replace the
# freshly-trained MLIN.  Manual `/api/control force_retrain` always
# bypasses this guard (the operator knows what they're doing).
RETRAIN_WARMUP_SAMPLES = 500


# ---------------------------------------------------------------------------
# Label rule — derive ground truth from raw network state
# ---------------------------------------------------------------------------

# Default thresholds, calibrated to the OMNeT-CSV distribution
# (rsrp_p50 ≈ -38 dBm, sinr_p50 ≈ 50 dB, delay_p50 ≈ 5 ms) so that
# CSV-replay mode produces ~15-20% positives at baseline.
#
# The Live RAN simulator emits realistic 5G NR KPIs (sinr_p50 ≈ 17 dB,
# delay_p50 ≈ 16 ms) and would saturate to 100% positives under these
# constants, so the engine *re-calibrates* the thresholds at pipeline-build
# time from its reference-frame percentiles (see ``LiveEngine._calibrate_label_thresholds``).
# Constants below are used only as fallbacks when the engine has not yet
# calibrated (e.g. unit tests, ``MTPCloud`` invocations during boot).
SINR_DEGRADED_DB  = 40.0
DELAY_DEGRADED_MS = 10.0

# Calibration floors / ceilings to keep the re-derived thresholds inside
# physically-meaningful 5G NR ranges even when the reference distribution
# is pathological (single-cell, no fading, etc).
_SINR_THRESH_FLOOR_DB  = 5.0    # never call SINR>5 dB "degraded"
_SINR_THRESH_CEIL_DB   = 45.0   # never let CSV inflation push it absurdly high
_DELAY_THRESH_FLOOR_MS = 5.0
_DELAY_THRESH_CEIL_MS  = 60.0


def derive_label(
    df: pd.DataFrame,
    *,
    sinr_thresh_db: float  = SINR_DEGRADED_DB,
    delay_thresh_ms: float = DELAY_DEGRADED_MS,
) -> np.ndarray:
    """
    Ground-truth label: 1 if the sample is in a *degraded network state*
    warranting a policy reaction (handover / model switch).

    Thresholds default to the CSV-calibrated constants but the engine
    overrides them with reference-derived percentiles so the rule stays
    sensible across both CSV-replay and Live-RAN modes.
    """
    handover = df["handover_flag"].fillna(0).astype(bool)
    sinr_bad  = df["sinr_db"]  < sinr_thresh_db
    delay_bad = df["delay_ms"] > delay_thresh_ms
    return (handover | sinr_bad | delay_bad).astype(int).to_numpy()


def derive_label_scalar(
    rsrp_dbm: float, sinr_db: float, tput_mbps: float, delay_ms: float,
    handover_flag: int = 0,
    *,
    sinr_thresh_db: float  = SINR_DEGRADED_DB,
    delay_thresh_ms: float = DELAY_DEGRADED_MS,
) -> int:
    """Single-sample version of :func:`derive_label` used at stream time
    so that y_true reflects the *injected* (post-perturbation) feature
    vector — not the pristine CSV row — giving honest rolling accuracy."""
    if bool(handover_flag):
        return 1
    if sinr_db  < sinr_thresh_db:  return 1
    if delay_ms > delay_thresh_ms: return 1
    return 0


def calibrate_label_thresholds(
    ref_df: pd.DataFrame,
    *,
    sinr_quantile:  float = 0.20,
    delay_quantile: float = 0.80,
) -> tuple[float, float]:
    """
    Derive ``(sinr_thresh_db, delay_thresh_ms)`` from a reference frame so
    that ~``sinr_quantile`` of the reference falls below the SINR threshold
    and ~``1 - delay_quantile`` falls above the delay threshold, giving a
    baseline positive rate of roughly 20-30 % regardless of the source's
    absolute KPI scale.  Bounded to physically meaningful 5G NR ranges.
    """
    if ref_df is None or len(ref_df) == 0 or "sinr_db" not in ref_df.columns:
        return SINR_DEGRADED_DB, DELAY_DEGRADED_MS
    sinr_p = float(ref_df["sinr_db"].quantile(sinr_quantile))
    delay_p = float(ref_df["delay_ms"].quantile(delay_quantile))
    sinr_t  = min(_SINR_THRESH_CEIL_DB, max(_SINR_THRESH_FLOOR_DB, sinr_p))
    delay_t = min(_DELAY_THRESH_CEIL_MS, max(_DELAY_THRESH_FLOOR_MS, delay_p))
    return sinr_t, delay_t


# ---------------------------------------------------------------------------
# Sample source abstraction
# ---------------------------------------------------------------------------

class SampleSource(Iterable[dict]):
    def reset(self) -> None: ...
    def __iter__(self) -> Iterator[dict]: ...
    def __len__(self) -> int: ...


class Simu5GCsvSource(SampleSource):
    """Streams rows from the pre-built real-Simu5G CSV, cycling forever."""

    def __init__(self, csv_path: Path):
        log.info("loading Simu5G CSV from %s", csv_path)
        df = pd.read_csv(csv_path)
        need = {"phase", "run", "ue", "t",
                "rsrp_dbm", "sinr_db", "throughput_mbps", "delay_ms",
                "handover_flag"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        df["rsrp_dbm"]        = df["rsrp_dbm"].clip(*RSRP_RANGE)
        df["sinr_db"]         = df["sinr_db"].clip(*SINR_RANGE)
        df["throughput_mbps"] = df["throughput_mbps"].clip(*TPUT_RANGE)
        df = df.dropna(subset=["delay_ms"]).reset_index(drop=True)
        df["delay_ms"] = df["delay_ms"].clip(*LAT_RANGE)

        df["label"] = derive_label(df)

        # Assign a run_index within each phase so we can cleanly exclude
        # specific runs (stable-3/4, drift-4) for the golden holdout.
        df["run_idx"] = df.groupby("phase")["run"].transform(
            lambda s: pd.Categorical(s, categories=sorted(s.unique())).codes
        )

        # Split: training stream vs golden holdout. The training stream
        # drops the golden runs entirely so RTP / ATM never learn on
        # them; the golden holdout is stashed in `self.golden_df` for
        # the NDT to pull from.
        golden_mask = np.zeros(len(df), dtype=bool)
        for phase, idxs in GOLDEN_RUN_INDICES.items():
            for idx in idxs:
                golden_mask |= (df["phase"] == phase) & (df["run_idx"] == idx)
        self.golden_df = df.loc[golden_mask].reset_index(drop=True).copy()
        train_df = df.loc[~golden_mask].reset_index(drop=True).copy()

        # Order: stable → drift → poison, shuffled within each phase so
        # every recent window is UE-representative.
        phase_order = {"stable": 0, "drift": 1, "poison": 2}
        train_df["_po"] = train_df["phase"].map(phase_order).fillna(99).astype(int)
        parts = []
        for po in sorted(train_df["_po"].unique()):
            chunk = train_df[train_df["_po"] == po].sample(frac=1.0, random_state=42)
            parts.append(chunk)
        train_df = pd.concat(parts, ignore_index=True).drop(columns="_po")

        self._df = train_df
        self._full_df = df      # full CSV for MTP-C historical corpus
        self._idx = 0
        log.info(
            "Simu5GCsvSource: %d train rows, %d golden rows, phases=%s",
            len(train_df), len(self.golden_df),
            sorted(train_df["phase"].unique().tolist()),
        )

    def reset(self) -> None:
        self._idx = 0

    def __iter__(self) -> Iterator[dict]:
        while True:
            if self._idx >= len(self._df):
                self._idx = 0
            row = self._df.iloc[self._idx].to_dict()
            self._idx += 1
            yield row

    def __len__(self) -> int:
        return len(self._df)

    @property
    def reference_frame(self) -> pd.DataFrame:
        """First 60% of stable-phase training rows — used to train the
        initial AIF. Excludes golden runs by construction."""
        stable = self._df[self._df["phase"] == "stable"]
        split = max(1, int(0.6 * len(stable)))
        return stable.iloc[:split]

    @property
    def full_frame(self) -> pd.DataFrame:
        """Full CSV (including golden runs) — for MTP-C historical corpus.
        MTP-C's job is to train from the archive; exclusion of golden runs
        is the NDT's responsibility, not MTP-C's."""
        return self._full_df


# ---------------------------------------------------------------------------
# Live RAN sample source (Level-2 closed-loop mode)
# ---------------------------------------------------------------------------

class RANLiveSource(SampleSource):
    """
    Live, controller-mutable sample source backed by a :class:`RANSimulator`.

    The engine pipeline expects more than just an iterator — it reads
    ``reference_frame`` for detector stats, ``golden_df`` for the NDT
    holdout, and ``full_frame`` for the class-balance archive used by the
    train-time augment.  We satisfy all three by *pre-generating* a
    diverse synthetic corpus at construction time (sweeping distance, TX
    power offset, and interference offset) and writing it to disk so
    ``MTPCloud(historical_corpus_path=...)`` can consume it identically
    to the offline CSV.

    ``__iter__``, however, delegates to the *live* simulator's
    :meth:`RANSimulator.step`, so each sample yielded reflects whatever
    actions the actuator has applied to the RAN state since the last
    tick.  This is what makes the loop *closed*: detector → actuator →
    simulator state → next sample.
    """

    # Corpus-generation grid — enough variety to give both label classes
    # without making boot too slow.
    _SWEEP_DISTANCES_M     = (50.0, 100.0, 200.0, 350.0, 500.0, 700.0)
    _SWEEP_TX_POWER_DB     = (-6.0, -3.0, 0.0, +3.0, +6.0)
    _SWEEP_INTERF_DB       = (-6.0, 0.0, +6.0, +12.0)
    _SAMPLES_PER_CELL      = 6     # n samples per (d, tx, interf) tuple
    _GOLDEN_FRACTION       = 0.10
    _CSV_FILENAME          = "live_ran_corpus.csv"

    def __init__(
        self,
        ran_simulator: RANSimulator,
        cache_dir:     Path,
        regenerate:    bool = False,
        dt_s:          float = 0.1,
        ue_id:         int   = 0,
    ) -> None:
        self.ran    = ran_simulator
        self.dt_s   = float(dt_s)
        self.ue_id  = int(ue_id)
        self._step_count = 0

        cache_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = cache_dir / self._CSV_FILENAME

        if regenerate or not self.csv_path.exists():
            self._generate_corpus(self.csv_path)
            log.info("RANLiveSource: corpus written to %s", self.csv_path)
        else:
            log.info("RANLiveSource: reusing cached corpus %s", self.csv_path)

        # Internal CSV-source provides reference_frame / golden_df / full_frame
        # using the synthetic corpus we just generated.
        self._inner = Simu5GCsvSource(self.csv_path)

    # ── corpus pre-generation ────────────────────────────────────────────

    def _generate_corpus(self, out: Path) -> None:
        """
        Sweep the simulator across (distance × TX power × interference) to
        build a corpus rich in *both* label classes.  Saved as a CSV with
        the same columns as the offline Simu5G dump so all downstream
        consumers (MTPCloud, derive_label, the augment) just work.
        """
        # Use a *separate* simulator instance so we don't disturb the live
        # one's RNG / state.  We hand it the same radio cfg though.
        from dashboard.live.ran_simulator import RANState
        probe = RANSimulator(
            seed         = 1234,
            radio_cfg    = self.ran.cfg,
            ue_speed_mps = 0.0,           # static probes — controlled physics
            initial_distance_m = 200.0,
            ue_id        = self.ue_id,
        )

        rows: list[dict] = []
        run_idx = 0
        for d in self._SWEEP_DISTANCES_M:
            for tx in self._SWEEP_TX_POWER_DB:
                for interf in self._SWEEP_INTERF_DB:
                    # Reset probe to clean state, then set the cell-specific
                    # static parameters before stepping.
                    probe.reset()
                    with probe._lock:
                        probe.state.ue_distance_m       = d
                        probe.state.tx_power_offset_db  = tx
                        probe.state.interference_offset_db = interf
                    for _ in range(self._SAMPLES_PER_CELL):
                        s = probe.step(dt=0.1)
                        rows.append({
                            "phase":           "stable" if interf <= 0 else "drift",
                            "run":             run_idx,
                            "ue":              self.ue_id,
                            "t":               s["t"],
                            "rsrp_dbm":        s["rsrp_dbm"],
                            "sinr_db":         s["sinr_db"],
                            "throughput_mbps": s["throughput_mbps"],
                            "delay_ms":        s["delay_ms"],
                            "handover_flag":   s["handover_flag"],
                        })
                    run_idx += 1

        df = pd.DataFrame(rows)
        df.to_csv(out, index=False)
        log.info(
            "RANLiveSource: synthetic corpus = %d rows over %d cells "
            "(label rate=%.1f%% by derive_label)",
            len(df), run_idx,
            100.0 * float(derive_label(df).mean()),
        )

    # ── SampleSource interface ───────────────────────────────────────────

    def reset(self) -> None:
        """Reset both the live simulator and the inner CSV index."""
        self.ran.reset()
        self._inner.reset()
        self._step_count = 0

    def __iter__(self) -> Iterator[dict]:
        # Live: every yield is a fresh physics tick that reflects whatever
        # actions the actuator has applied since the last call.
        while True:
            s = self.ran.step(dt=self.dt_s)
            # The corpus uses 'phase' for downstream label/MTP-C visibility;
            # live samples carry "live" so the UI can colour them differently.
            self._step_count += 1
            yield {
                "phase":           s.get("phase", "live"),
                "run":             0,
                "ue":              self.ue_id,
                "t":               float(s["t"]),
                "rsrp_dbm":        float(s["rsrp_dbm"]),
                "sinr_db":         float(s["sinr_db"]),
                "throughput_mbps": float(s["throughput_mbps"]),
                "delay_ms":        float(s["delay_ms"]),
                "handover_flag":   int(s.get("handover_flag", 0) or 0),
            }

    def __len__(self) -> int:
        # The "length" reported is the synthetic corpus length, since that's
        # what bounds resets / progress; live mode is genuinely unbounded.
        return len(self._inner)

    # ── delegated frames (for engine pipeline construction) ──────────────

    @property
    def reference_frame(self) -> pd.DataFrame:
        return self._inner.reference_frame

    @property
    def golden_df(self) -> pd.DataFrame:
        return self._inner.golden_df

    @property
    def full_frame(self) -> pd.DataFrame:
        return self._inner.full_frame

    @property
    def _df(self) -> pd.DataFrame:           # used by _build_pipeline at L543
        return self._inner._df


# ---------------------------------------------------------------------------
# Injection layer
# ---------------------------------------------------------------------------

@dataclass
class InjectionState:
    """User-controllable drift knobs + variant selection + modes."""
    # Continuous biases (applied every sample while non-zero)
    sinr_bias_db:   float = 0.0
    rsrp_bias_db:   float = 0.0
    delay_bias_ms:  float = 0.0
    tput_scale:     float = 1.0

    # Transient modes
    poison_mode:    bool  = False
    noise_scale:    float = 1.0

    # Operator controls
    preferred_variant:     Optional[str] = None   # None=auto, or "LOCAL"/"EXTERNAL"/"CLOUD"
    closed_loop_enabled:   bool = False
    use_golden_ndt:        bool = True

    # One-shot flags
    force_retrain_pending: bool = False
    reset_pending:         bool = False

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Cancel hook — engine registers a callback that cancels any active
    # closed-loop decay when the UI writes to an injection slider mid-decay.
    # Set externally; default is a no-op so unit tests don't need wiring.
    _on_user_injection_change: Optional[Callable[[], None]] = field(
        default=None, repr=False
    )

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "sinr_bias_db":        self.sinr_bias_db,
                "rsrp_bias_db":        self.rsrp_bias_db,
                "delay_bias_ms":       self.delay_bias_ms,
                "tput_scale":          self.tput_scale,
                "poison_mode":         self.poison_mode,
                "noise_scale":         self.noise_scale,
                "preferred_variant":   self.preferred_variant,
                "closed_loop_enabled": self.closed_loop_enabled,
                "use_golden_ndt":      self.use_golden_ndt,
            }

    # Keys that count as "user touched an injection slider". Changing any
    # of these mid-decay must cancel the closed-loop decay animation so the
    # user's override actually takes effect instead of being stomped on the
    # next tick.
    _INJECTION_KEYS = frozenset({
        "sinr_bias_db", "rsrp_bias_db", "delay_bias_ms",
        "tput_scale", "noise_scale", "poison_mode",
    })

    def update(self, patch: dict, _source: str = "user") -> None:
        with self._lock:
            for k, v in patch.items():
                if   k == "sinr_bias_db":        self.sinr_bias_db        = float(v)
                elif k == "rsrp_bias_db":        self.rsrp_bias_db        = float(v)
                elif k == "delay_bias_ms":       self.delay_bias_ms       = float(v)
                elif k == "tput_scale":          self.tput_scale          = float(v)
                elif k == "poison_mode":         self.poison_mode         = bool(v)
                elif k == "noise_scale":         self.noise_scale         = float(v)
                elif k == "preferred_variant":   self.preferred_variant   = (
                    None if v in (None, "", "AUTO", "auto") else str(v).upper()
                )
                elif k == "closed_loop_enabled": self.closed_loop_enabled = bool(v)
                elif k == "use_golden_ndt":      self.use_golden_ndt      = bool(v)
            cb = self._on_user_injection_change if _source == "user" else None
            touched_injection = bool(self._INJECTION_KEYS & set(patch.keys()))
        # Fire the cancel hook *outside* the lock: the engine callback
        # acquires its own decay lock and we must not invert the order.
        if cb is not None and touched_injection:
            try:
                cb()
            except Exception:
                pass

    def trigger_retrain(self) -> None:
        with self._lock:
            self.force_retrain_pending = True

    def trigger_reset(self) -> None:
        with self._lock:
            self.reset_pending = True

    def consume_retrain_flag(self) -> bool:
        with self._lock:
            v, self.force_retrain_pending = self.force_retrain_pending, False
            return v

    def consume_reset_flag(self) -> bool:
        with self._lock:
            v, self.reset_pending = self.reset_pending, False
            return v


def apply_injections(row: dict, inj: InjectionState, rng: np.random.Generator) -> np.ndarray:
    # Read injection sliders without taking the lock — CPython attribute
    # reads of int/float/bool are GIL-atomic and the worst-case race here
    # (one slider already updated, the next still old) is invisible at
    # 200 Hz; it self-corrects on the very next tick.  Calling snapshot()
    # acquires InjectionState._lock once per sample which is the single
    # cheapest hot-path lock to remove (see bottleneck audit).
    rsrp_bias  = inj.rsrp_bias_db
    sinr_bias  = inj.sinr_bias_db
    delay_bias = inj.delay_bias_ms
    tput_scale = inj.tput_scale
    noise_scl  = inj.noise_scale
    poison     = inj.poison_mode

    rsrp = row["rsrp_dbm"] + rsrp_bias
    sinr = row["sinr_db"]  + sinr_bias
    tput = row["throughput_mbps"] * tput_scale
    dly  = row["delay_ms"]  + delay_bias

    if noise_scl > 1.0:
        s = noise_scl
        rsrp += rng.normal(0, 3.0 * (s - 1.0))
        sinr += rng.normal(0, 1.5 * (s - 1.0))
        tput += rng.normal(0, 20.0 * (s - 1.0))
        dly  += rng.normal(0, 2.0 * (s - 1.0))

    if poison and rng.random() < 0.15:
        kind = rng.integers(0, 4)
        if   kind == 0: rsrp -= rng.uniform(20, 40)
        elif kind == 1: sinr -= rng.uniform(15, 30)
        elif kind == 2: dly  += rng.uniform(30, 60)
        else:           tput *= rng.uniform(0.05, 0.2)

    rsrp = float(np.clip(rsrp, *RSRP_RANGE))
    sinr = float(np.clip(sinr, *SINR_RANGE))
    tput = float(np.clip(tput, *TPUT_RANGE))
    dly  = float(np.clip(dly,  *LAT_RANGE))
    return np.array([rsrp, sinr, tput, dly], dtype=np.float64)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

@dataclass
class EngineConfig:
    csv_path: Path
    seed:     int   = 42
    rate_hz:  float = 30.0
    max_queue: int  = 5000
    mlflow_db: Optional[Path] = None   # sqlite file for MLflow (None = auto)

    # ── Level-2 closed-loop RAN (live physics + actuator) ───────────────
    # When ``live_mode`` is True the engine builds a ``RANLiveSource``
    # backed by an in-memory ``RANSimulator`` instead of the CSV replay.
    # An optional ``RANActuator`` then receives every detector snapshot
    # and may issue actions that mutate the simulator's state in place.
    live_mode:        bool = False
    actuator_enabled: bool = True   # only applies when live_mode=True
    live_dt_s:        float = 0.1   # per-tick simulator advance


class LiveEngine:
    """Background streaming engine — owns RTP/AIMP + pushes events to a queue."""

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.injection = InjectionState()
        self.events: queue.Queue[dict] = queue.Queue(maxsize=cfg.max_queue)

        self._rate_hz  = cfg.rate_hz
        self._paused   = False
        self._stop     = threading.Event()
        self._thread:  Optional[threading.Thread] = None
        self._lock     = threading.Lock()

        self.source:  Optional[SampleSource] = None
        self.aimp:    Optional[AIMP] = None
        self.rtp            = None
        self.atm            = None
        self.aif            = None

        # Level-2 closed-loop RAN handles (None when live_mode=False)
        self.ran:        Optional[RANSimulator] = None
        self.actuator:   Optional[RANActuator]  = None
        # Most-recent KPI tail — passed to the actuator alongside the
        # detector event so the rule table can do symptom diagnosis.
        self._last_kpi_tail: dict = {}
        self.rng            = np.random.default_rng(cfg.seed)
        self._step          = 0
        self._correct       = 0
        self._total         = 0

        # Standardization stats (populated in _build_pipeline)
        self._ref_mean: Optional[np.ndarray] = None
        self._ref_std:  Optional[np.ndarray] = None

        # Label-rule thresholds (recalibrated from the source's reference
        # frame in _build_pipeline so the rule stays sensible across both
        # CSV-replay (sinr_p50≈50 dB) and Live RAN (sinr_p50≈17 dB) modes).
        # Defaults match the CSV calibration so behaviour is unchanged
        # if calibration is somehow skipped.
        self._sinr_thresh_db:  float = SINR_DEGRADED_DB
        self._delay_thresh_ms: float = DELAY_DEGRADED_MS

        # Golden holdout (populated in _build_pipeline)
        self._golden_X: Optional[np.ndarray] = None      # standardized
        self._golden_y: Optional[np.ndarray] = None
        self._golden_n: int = 0

        # Rolling LIB snapshot for honest reference refit.
        # ``deque(maxlen=N)`` evicts on append in O(1) — replaces a list +
        # ``pop(0)`` per tick (which is O(n) and shifts ~600 elements/tick
        # at 200 Hz once the buffer is full).  See bottleneck audit.
        self._recent_x_cap = 600
        self._recent_x_raw: deque[np.ndarray] = deque(maxlen=self._recent_x_cap)

        # One-shot flag for the post-boot detector reference rebase.  In
        # Live mode the cached corpus distribution (used to fit the initial
        # detector reference at build time) does NOT match the live RAN
        # stream — DDD/CDD then false-fire on legitimate idle data,
        # confusing users who haven't injected anything.  After ~250 live
        # samples we re-snapshot the reference from what we're actually
        # observing; subsequent fires are then real drift, not stale-corpus
        # mismatch.  CSV-replay mode keeps the original behaviour.
        self._initial_ref_rebased: bool = False

        # Sidecar ground-truth buffer (1-for-1 with LIB, derived from
        # injected features at stream time). Feeds MTP training via the
        # install_spies monkey-patch so candidates learn from real labels
        # instead of the current MLIN's self-consistent pseudo-labels.
        # Same O(1)-eviction reason as ``_recent_x_raw``.
        self._gt_buffer_cap = 2000
        self._gt_buffer: deque[int] = deque(maxlen=self._gt_buffer_cap)

        # Last NDT dual scores (for status snapshot)
        self._last_ndt_pseudo: Optional[float] = None
        self._last_ndt_gt:     Optional[float] = None
        self._last_ndt_bias:   Optional[float] = None

        # Closed-loop decay state (thread-safe snapshot pattern)
        self._decay_lock = threading.Lock()
        self._decay_active = False
        self._decay_t0 = 0.0
        self._decay_from: dict = {}   # snapshot of injection at decay start

        # Register decay-cancel hook so a user slider move mid-decay stops
        # the animation (the module docstring promises this behaviour).
        self.injection._on_user_injection_change = self._cancel_closed_loop_decay

        # Throttle detector snapshots to ~20 Hz regardless of stream rate
        self._last_det_emit = 0.0

        # Pipeline-ready barrier — set at end of ``_build_pipeline`` so the
        # server can pre-warm in a background thread at startup, then
        # ``start()`` (called from the user's "Start" button) skips the
        # heavy build and just kicks the loop thread (instant).  Without
        # this, the first /api/control click freezes the UI for the full
        # CSV-load + sklearn-fit + MTPCloud + MLflow init duration.
        self._pipeline_ready: threading.Event = threading.Event()
        # Separate lock for ``_build_pipeline`` so the pre-warm thread and
        # ``start()`` (which holds ``self._lock``) can't race into a double
        # build.  Held only during the ~2 s construction; uses an RLock so
        # ``start()`` can call ``_build_pipeline`` while still owning the
        # outer lock if the user clicks Start before the pre-warm fires.
        self._build_lock: threading.RLock = threading.RLock()

        # ── Retrain worker (UI-freeze fix) ─────────────────────────────
        # ATM retrains take 1-15 s (sklearn fit + NDT + MLflow round-trip).
        # When run synchronously inside the engine loop they freeze sample
        # generation, so the UI's KPI charts pause for the whole duration
        # — the user clicks Force retrain and the dashboard "stops".
        # The retrain worker decouples this: ``_wrapped_handle`` snapshots
        # the signal + UI variant, hands the job to ``_retrain_queue``,
        # and returns a synthetic ATMResult immediately.  The dedicated
        # worker thread drains the queue, runs the slow body (LOB swap,
        # class-balance augment, ``orig_handle``, unwind, ``_on_deploy``),
        # and emits ``retrain_done`` / ``model_history`` from its own
        # context.  Sample generation never pauses.
        # Queue is single-slot — concurrent triggers (user spam-clicks
        # Force retrain, or detector chatter) become "skipped, already in
        # flight" emits so the timeline stays honest.
        self._retrain_queue: queue.Queue = queue.Queue(maxsize=1)
        self._retrain_inflight: threading.Event = threading.Event()
        self._retrain_thread: Optional[threading.Thread] = None
        self._retrain_meta: dict = {
            "step":       None,
            "started_at": None,
            "variant":    None,
        }

    # ── public ────────────────────────────────────────────────────────────

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict:
        # Retrain worker snapshot — surfaced so the UI can render a
        # "Retraining…" badge with elapsed-seconds during the 1-15 s
        # async cycle, and so the operator can see *why* the timeline
        # isn't logging new retrains (worker busy → SKIPPED triggers).
        rt_started = self._retrain_meta.get("started_at")
        retrain_status = {
            "in_flight":  self._retrain_inflight.is_set(),
            "step":       self._retrain_meta.get("step"),
            "started_at": rt_started,
            "elapsed_s":  (time.time() - rt_started) if rt_started else None,
            "variant":    self._retrain_meta.get("variant"),
        }
        out = {
            "running":  self.is_running(),
            "paused":   self._paused,
            "rate_hz":  self._rate_hz,
            "step":     self._step,
            "total":    self._total,
            "correct":  self._correct,
            "accuracy": (self._correct / self._total) if self._total else 0.0,
            "injection": self.injection.snapshot(),
            "csv_path": str(self.cfg.csv_path),
            "source_len": len(self.source) if self.source is not None else 0,
            "golden_n":  self._golden_n,
            "ndt_last": {
                "pseudo": self._last_ndt_pseudo,
                "gt":     self._last_ndt_gt,
                "bias":   self._last_ndt_bias,
            },
            "live_mode": bool(self.cfg.live_mode),
            "pipeline_ready": self._pipeline_ready.is_set(),
            "pipeline_building": (not self._pipeline_ready.is_set()) and (self.aimp is not None or self.source is not None),
            "retrain": retrain_status,
        }
        if self.ran is not None:
            out["ran_state"] = self.ran.snapshot_state()
        if self.actuator is not None:
            out["actuator"] = self.actuator.snapshot()
        return out

    def set_rate(self, hz: float) -> None:
        self._rate_hz = max(0.5, min(500.0, float(hz)))

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def stop(self) -> None:
        self._stop.set()
        # Push the shutdown sentinel BEFORE joining the engine loop so the
        # retrain worker (which blocks on get(timeout=0.5)) wakes promptly.
        # If the queue happens to be full (a job in flight), put_nowait
        # would raise — fall back to put with a short timeout.
        try:
            self._retrain_queue.put_nowait(None)
        except queue.Full:
            try:
                self._retrain_queue.put(None, timeout=1.0)
            except queue.Full:
                log.warning("stop: could not enqueue retrain sentinel")
        rt = self._retrain_thread
        if rt is not None and rt.is_alive():
            # Allow up to 20 s for an in-flight ATM retrain to wind down
            # naturally — sklearn fit + NDT + MLflow round-trip can take
            # ~15 s in worst-case CLOUD GridSearch paths.  After that we
            # leave the daemon thread to die with the process; the worker
            # never holds external resources so this is safe.
            rt.join(timeout=20.0)
            if rt.is_alive():
                log.warning("stop: retrain worker did not exit within 20s")
        self._retrain_thread = None
        t = self._thread
        if t is not None:
            t.join(timeout=5.0)
        self._thread = None

    def start(self) -> None:
        with self._lock:
            if self.is_running():
                self._paused = False
                return
            self._stop.clear()
            self._paused = False
            if self.aimp is None:
                self._build_pipeline()
            # Start (or restart) the retrain worker before the engine loop
            # so the very first detector trigger has somewhere to dispatch.
            # Drain any leftover sentinel/jobs from a previous stop()/start()
            # cycle so we begin with an empty queue.
            while True:
                try:
                    self._retrain_queue.get_nowait()
                    self._retrain_queue.task_done()
                except queue.Empty:
                    break
                except ValueError:
                    break
            self._retrain_inflight.clear()
            self._retrain_meta = {
                "step": None, "started_at": None, "variant": None,
            }
            if self._retrain_thread is None or not self._retrain_thread.is_alive():
                self._retrain_thread = threading.Thread(
                    target=self._retrain_worker_loop,
                    name="RetrainWorker",
                    daemon=True,
                )
                self._retrain_thread.start()
            self._thread = threading.Thread(target=self._loop, name="LiveEngine", daemon=True)
            self._thread.start()
            self._emit({"type": "started", "status": self.status()})

    # ── pipeline construction ─────────────────────────────────────────────

    def _build_pipeline(self) -> None:
        # Idempotent: pre-warm thread (server startup) and start() (user
        # click) can both call this; serialize via _build_lock and bail
        # if the pipeline is already up.  The inexpensive event check
        # outside the lock is a fast path for the common case where the
        # pre-warm completed before the user pressed Start.
        if self._pipeline_ready.is_set():
            return
        with self._build_lock:
            if self._pipeline_ready.is_set():
                return
            self._build_pipeline_locked()

    def _build_pipeline_locked(self) -> None:
        if self.cfg.live_mode:
            # Level-2: spin up the live RAN simulator, wrap it in
            # RANLiveSource (which also pre-generates a synthetic CSV
            # corpus so MTPCloud + golden holdout work identically),
            # and optionally attach the actuator.
            self.ran = RANSimulator(
                seed       = self.cfg.seed,
                ue_speed_mps = 5.0,
                initial_distance_m = 200.0,
            )
            cache_dir = self.cfg.csv_path.parent
            self.source = RANLiveSource(
                ran_simulator = self.ran,
                cache_dir     = cache_dir,
                regenerate    = False,
                dt_s          = self.cfg.live_dt_s,
            )
            # Re-point cfg.csv_path at the synthetic corpus so MTPCloud
            # and any other csv-aware downstream component picks it up.
            self.cfg = EngineConfig(
                **{**self.cfg.__dict__, "csv_path": self.source.csv_path}
            )
            self.actuator = RANActuator(
                ran      = self.ran,
                enabled  = self.cfg.actuator_enabled,
            )
            log.info(
                "engine: LIVE mode — RANSimulator + RANActuator(enabled=%s)",
                self.cfg.actuator_enabled,
            )
        else:
            self.source = Simu5GCsvSource(self.cfg.csv_path)

        # ── Reference stats (detector baseline: stable-phase only) ────
        ref = self.source.reference_frame
        REF_CAP = 2000
        if len(ref) > REF_CAP:
            ref = ref.sample(n=REF_CAP, random_state=self.cfg.seed).reset_index(drop=True)
        X_ref_raw = ref[FEATURE_COLS].to_numpy(dtype=np.float64)

        # ── Calibrate label-rule thresholds from this reference ──────
        # Default constants are tuned for CSV (sinr_p50≈50 dB) but the
        # Live RAN simulator emits realistic 5G NR (sinr_p50≈17 dB), so
        # the constants would saturate every Live sample as "degraded"
        # and CPD would fire on tick 1 with no injection.  Re-deriving
        # from the reference's 20th/80th percentiles produces ~20-30%
        # baseline positives in *both* modes.
        self._sinr_thresh_db, self._delay_thresh_ms = calibrate_label_thresholds(ref)
        log.info(
            "engine: label thresholds calibrated — sinr<%.2f dB or delay>%.2f ms = degraded "
            "(ref: sinr_p20=%.2f dB, delay_p80=%.2f ms)",
            self._sinr_thresh_db, self._delay_thresh_ms,
            float(ref["sinr_db"].quantile(0.20)) if len(ref) else float("nan"),
            float(ref["delay_ms"].quantile(0.80)) if len(ref) else float("nan"),
        )

        # Re-label the source frames in place using the calibrated thresholds
        # so reference y_ref, MLIN init training set, and golden holdout all
        # speak the same language as the streaming derive_label_scalar call.
        def _relabel(df: pd.DataFrame) -> None:
            if df is None or len(df) == 0 or "label" not in df.columns:
                return
            df["label"] = derive_label(
                df,
                sinr_thresh_db  = self._sinr_thresh_db,
                delay_thresh_ms = self._delay_thresh_ms,
            )

        _relabel(getattr(self.source, "_df",       None))
        _relabel(getattr(self.source, "_full_df",  None))
        _relabel(getattr(self.source, "golden_df", None))
        # Reference is a *view* into _df (stable rows, first 60%); re-pull
        # to pick up the recalibrated label column.
        ref = self.source.reference_frame
        if len(ref) > REF_CAP:
            ref = ref.sample(n=REF_CAP, random_state=self.cfg.seed).reset_index(drop=True)
        y_ref = ref["label"].to_numpy(dtype=np.int64)

        STD_FLOOR = np.array([2.5, 1.25, 20.0, 2.0], dtype=np.float64)
        self._ref_mean = X_ref_raw.mean(axis=0)
        raw_std        = X_ref_raw.std(axis=0)
        self._ref_std  = np.maximum(raw_std, STD_FLOOR)
        X_ref = (X_ref_raw - self._ref_mean) / self._ref_std
        log.info(
            "engine: detector reference mean=%s std_clamped=%s ref_pos_rate=%.2f%%",
            np.round(self._ref_mean, 2).tolist(),
            np.round(self._ref_std,  2).tolist(),
            100.0 * float(y_ref.mean()) if len(y_ref) else 0.0,
        )

        # Ensure RTP's reference y carries at least a couple of positives
        # so the detector baseline doesn't see a degenerate one-class view
        # (only used for detector internals — MLIN training uses a
        # separate, properly stratified set below).
        y_ref_for_rtp = y_ref.copy()
        if y_ref_for_rtp.sum() == 0 and len(y_ref_for_rtp):
            y_ref_for_rtp[: max(5, len(y_ref_for_rtp) // 40)] = 1

        # ── Stratified initial MLIN training set ─────────────────────
        # Stable-phase alone is ~100% label=0 on this corpus (sinr_p50
        # clips to 40 dB, delay_p50 ≈ 5 ms), so a RandomForest trained
        # on it cannot meaningfully distinguish degraded states — it
        # collapses to a near-constant predictor.  We instead build a
        # stratified sample from *all non-golden* rows so both classes
        # are well represented, which gives the dashboard a realistic
        # pre-drift baseline accuracy to degrade from.
        train_df = self.source._df   # already excludes golden runs
        positives = train_df[train_df["label"] == 1]
        negatives = train_df[train_df["label"] == 0]
        target_each = min(1500, len(positives), len(negatives))
        if target_each < 50:
            # Degenerate corpus — fall back to the old stable-only path
            X_train = X_ref
            y_train = y_ref_for_rtp
            log.warning(
                "engine: insufficient positives for stratified init "
                "(pos=%d, neg=%d) — training MLIN on stable-only ref",
                len(positives), len(negatives),
            )
        else:
            pos_samp = positives.sample(n=target_each, random_state=self.cfg.seed)
            neg_samp = negatives.sample(n=target_each, random_state=self.cfg.seed)
            strat = pd.concat([pos_samp, neg_samp], ignore_index=True).sample(
                frac=1.0, random_state=self.cfg.seed,
            )
            X_train_raw = strat[FEATURE_COLS].to_numpy(dtype=np.float64)
            X_train = (X_train_raw - self._ref_mean) / self._ref_std
            y_train = strat["label"].to_numpy(dtype=np.int64)
            log.info(
                "engine: MLIN init training — stratified %d rows "
                "(%d pos / %d neg) from %d non-golden",
                len(strat), int(y_train.sum()), int((1 - y_train).sum()),
                len(train_df),
            )

        clf = RandomForestClassifier(n_estimators=80, random_state=self.cfg.seed)
        clf.fit(X_train, y_train)

        # ── Golden holdout ────────────────────────────────────────────
        golden = self.source.golden_df
        if len(golden) > 0:
            X_golden_raw = golden[FEATURE_COLS].to_numpy(dtype=np.float64)
            self._golden_X = (X_golden_raw - self._ref_mean) / self._ref_std
            self._golden_y = golden["label"].to_numpy(dtype=np.int64)
            self._golden_n = len(golden)
            # Probe the freshly-trained MLIN against the golden holdout
            # so we can see, at boot time, whether the initial model is
            # actually competent — otherwise all NDT decisions downstream
            # are measured against a degenerate base.
            try:
                init_golden = float(
                    (clf.predict(self._golden_X) == self._golden_y).mean()
                )
            except Exception as exc:
                log.warning("engine: initial-MLIN golden probe failed: %s", exc)
                init_golden = float("nan")
            log.info(
                "engine: golden holdout — %d rows, positive rate=%.2f%%, "
                "initial MLIN acc=%.4f",
                self._golden_n, 100.0 * float(self._golden_y.mean()), init_golden,
            )
        else:
            log.warning("engine: golden holdout is empty — NDT will run pseudo-only")

        # ── Build MTP-C with standardizing historical corpus ──────────
        def _standardize(X_raw: np.ndarray) -> np.ndarray:
            return (X_raw - self._ref_mean) / self._ref_std

        # Capture calibrated thresholds in the lambda closure so MTPCloud
        # labels its historical corpus consistently with the streaming rule.
        _sinr_t  = self._sinr_thresh_db
        _delay_t = self._delay_thresh_ms
        mtp_c = MTPCloud(
            historical_corpus_path=self.cfg.csv_path,   # use the same CSV as archive
            standardize_fn=_standardize,
            feature_cols=tuple(FEATURE_COLS),
            label_rule=lambda df: derive_label(
                df, sinr_thresh_db=_sinr_t, delay_thresh_ms=_delay_t,
            ),
            corpus_sample_size=5000,
            use_grid_search=False,     # keep demo snappy; flip True for heavier runs
            n_splits=3,
            random_state=self.cfg.seed,
            slow_factor_s=2.0,         # enforce visible cloud latency
        )

        # ── Build MTP-E with sqlite-backed MLflow (no external server) ──
        mlflow_db = self.cfg.mlflow_db or (
            self.cfg.csv_path.parent / "mlflow_dashboard.db"
        )
        mlflow_uri = f"sqlite:///{mlflow_db.as_posix()}"
        # Separate artifact root so file:// logs work alongside sqlite metadata
        import os as _os
        _os.environ.setdefault(
            "MLFLOW_ARTIFACT_ROOT",
            str(self.cfg.csv_path.parent / "mlruns_artifacts"),
        )
        try:
            from atm.mtp_e import MTPExternal
            mtp_e = MTPExternal(
                experiment_name="aimp_dashboard",
                model_name="aif_classifier",
                mlflow_uri=mlflow_uri,
                tune_hyperparams=False,
                n_splits=3,
                tags={"component": "MTP-E", "origin": "dashboard"},
            )
            log.info("engine: MTP-E configured with MLflow uri=%s", mlflow_uri)
        except Exception as exc:
            log.warning("engine: MTP-E setup failed (%s); EXTERNAL will fall back", exc)
            mtp_e = None

        mtpc = MTPComposer(
            mtp_l=MTPLocal(n_splits=3, fine_tune_first=True,
                           random_state=self.cfg.seed),
            mtp_e=mtp_e,
            mtp_c=mtp_c,
        )

        policy = AIMPPolicy(
            atm_policy=ATMPolicy(
                prefer_variant=None,       # ← auto-select; UI may override
                use_ndt=True,
                ndt_min_accuracy=0.70,
                auto_deploy=True,
                critical_always_local_first=True,
                local_max_samples=500,
            ),
            rtp_profile_name="classifier_default",
            cost_limit=1.5,
            reconfigure_rtp_on_model_change=True,
        )
        self.aimp = AIMP(policy=policy, rtpc=RTPComposer(), mtpc=mtpc)
        # Pass the stable-phase `X_ref` + class-guarded `y_ref_for_rtp` as
        # the detector reference; the MLIN itself has already been fit on
        # the stratified training set above.
        self.aif, self.rtp, self.atm = self.aimp.register_aif(
            estimator=clf, X_ref=X_ref, y_ref=y_ref_for_rtp,
        )

        # ── Tame AIF's DPP so it doesn't double-standardize ────────────
        # The AIF wraps the classifier in a DPP (StandardScaler + z-clip)
        # that fits *lazily on the first call to `aif.predict`*.  Since
        # we already z-score every feature upstream in the engine loop,
        # an unattended DPP would fit its scaler on the first sample
        # (mean=that_sample, var=0 → scale=1 by sklearn convention) and
        # then subtract that single sample's values from every
        # subsequent input — giving a shifted feature space the
        # classifier has never seen and producing a degenerate
        # always-predict-positive-class model.  Force DPP into a true
        # identity transform by hand-populating its scaler with mean=0
        # and scale=1 so `DPP.transform(z)` returns `z` verbatim (modulo
        # the 5-σ clip, which is a no-op on already-standardized data).
        try:
            n_feat = len(FEATURE_COLS)
            scaler = self.aif.dpp._scaler
            scaler.mean_  = np.zeros(n_feat, dtype=np.float64)
            scaler.scale_ = np.ones(n_feat, dtype=np.float64)
            scaler.var_   = np.ones(n_feat, dtype=np.float64)
            scaler.n_features_in_ = n_feat
            scaler.n_samples_seen_ = 1
            self.aif.dpp._fitted = True
            log.info("engine: AIF DPP pinned to identity "
                     "(mean=0, scale=1) in z-space")
        except Exception as exc:
            log.warning("engine: AIF DPP identity-pin failed: %s", exc)

        # ── RTP detector thresholds (unchanged — tuned previously) ────
        self.rtp.ddd.mmd_threshold           = 0.3
        self.rtp.ddd.min_drifted_features    = 2
        self.rtp.ddd.ks_alpha                = 0.01
        self.rtp.dpd.contamination_threshold = 0.20
        self.rtp.dpd.mahal_threshold         = 15.0

        # ── NDT tuning ────────────────────────────────────────────────
        # Default `min_improvement=0` rejects any candidate that doesn't
        # *strictly* exceed the current MLIN.  With a well-fitted base
        # model (golden accuracy ≈ 0.98-0.99 on this Simu5G corpus) the
        # candidate is often effectively tied, which is a valid reason to
        # deploy — a fresh model trained on newer data is preferable even
        # when accuracies match.  Allow ties and very minor regressions
        # (≤0.5 pp) so drift-triggered retrains can actually ship.
        try:
            self.atm.ndt.min_improvement = -0.005
            self.atm.ndt.min_score       = 0.60
        except Exception as exc:
            log.debug("NDT threshold tune failed: %s", exc)

        self._install_spies()

        self._emit({
            "type": "init",
            "rows":         len(self.source),
            "golden_rows":  self._golden_n,
            "features":     FEATURE_COLS,
            "ref_samples":  len(X_ref),
            "mlflow_uri":   mlflow_uri,
            "variants":     ["AUTO", "LOCAL", "EXTERNAL", "CLOUD"],
        })

        # Signal: all heavy lifting (CSV load, sklearn fit, MTPCloud spin-up,
        # MLflow init, label calibration) is done.  /api/control can now call
        # self.start() without blocking the request thread for ~2s.  The flag
        # is also surfaced via status() so the UI can hide the Start button
        # behind a "warming up…" spinner during cold boot.
        self._pipeline_ready.set()

    def _install_spies(self) -> None:
        # ── MToUT spy ────────────────────────────────────────────────
        original_mtout = self.rtp._on_mtout

        def _on_mtout(signal):
            try:
                self._emit({
                    "type":     "mtout",
                    "step":     int(signal.step),
                    "severity": signal.severity(),
                    "reasons":  [r.name for r in signal.reasons],
                })
            except Exception as exc:
                log.debug("mtout emit failed: %s", exc)
            if original_mtout is not None:
                original_mtout(signal)

        self.rtp._on_mtout = _on_mtout

        # ── NDT spy: compute pseudo + gt scores, decide on gt ────────
        ndt = self.atm.ndt
        orig_score = ndt._score
        orig_get_current = ndt._get_current

        def _patched_validate(
            candidate, X_val, y_val,
            min_score=None, run_id=None, y_val_gt=None, **_kw
        ):
            # X_val / y_val are LIB / LOB (pseudo-labels).
            X_val = np.atleast_2d(np.asarray(X_val, dtype=float))
            y_val = np.asarray(y_val, dtype=float).ravel()
            threshold = min_score if min_score is not None else ndt.min_score

            # 1) Pseudo score (self-referential — the biased one)
            pseudo_cand = orig_score(candidate, X_val, y_val)
            current = orig_get_current()
            try:
                pseudo_base = orig_score(current, X_val, y_val) if current is not None else 0.0
            except Exception:
                pseudo_base = 0.0

            # 2) Ground-truth score on disjoint golden holdout
            use_golden = (
                self.injection.snapshot()["use_golden_ndt"]
                and self._golden_X is not None and self._golden_y is not None
            )
            gt_cand = gt_base = None
            if use_golden:
                try:
                    gt_cand = orig_score(candidate, self._golden_X, self._golden_y)
                    if current is not None:
                        gt_base = orig_score(current, self._golden_X, self._golden_y)
                    else:
                        gt_base = 0.0
                except Exception as exc:
                    log.warning("NDT golden scoring failed: %s", exc)
                    gt_cand = gt_base = None

            # 3) Pick decision score — gt if available, pseudo otherwise
            decision_cand = gt_cand if gt_cand is not None else pseudo_cand
            decision_base = gt_base if gt_base is not None else pseudo_base
            passed = (
                decision_cand >= threshold
                and (decision_cand - decision_base) >= ndt.min_improvement
            )

            # 4) Bookkeeping + emit NDT dual event
            bias = (pseudo_cand - gt_cand) if gt_cand is not None else None
            self._last_ndt_pseudo = float(pseudo_cand)
            self._last_ndt_gt     = float(gt_cand) if gt_cand is not None else None
            self._last_ndt_bias   = float(bias)    if bias     is not None else None

            ndt.history.append({
                "candidate_score":    float(decision_cand),
                "baseline_score":     float(decision_base),
                "improvement":        float(decision_cand - decision_base),
                "min_score":          float(threshold),
                "min_improvement":    float(ndt.min_improvement),
                "passed":             bool(passed),
                "run_id":             run_id,
                "candidate_gt_score": float(gt_cand) if gt_cand is not None else None,
                "baseline_gt_score":  float(gt_base) if gt_base is not None else None,
                "pseudo_score":       float(pseudo_cand),
                "validation_mode":    "golden" if use_golden else "pseudo",
            })

            self._emit({
                "type":        "ndt_dual",
                "step":        self._step,
                # Field names intentionally match the frontend's
                # applyNdtDual() contract: {pseudo,gt}_{base,cand}.
                "pseudo_base": float(pseudo_base),
                "pseudo_cand": float(pseudo_cand),
                "gt_base":     float(gt_base) if gt_base is not None else None,
                "gt_cand":     float(gt_cand) if gt_cand is not None else None,
                "bias":        float(bias) if bias is not None else None,
                "passed":      bool(passed),
                "threshold":   float(threshold),
                "golden_n":    self._golden_n,
                "mode":        "golden" if use_golden else "pseudo",
            })
            return bool(passed)

        ndt.validate = _patched_validate

        # ── ATM spy: enforce UI variant override + post-deploy hooks ──
        # The wrapped handle no longer runs the heavy retrain inline —
        # it captures the signal + UI snapshot and hands the work to a
        # dedicated worker thread (see ``_retrain_worker_loop``), so
        # sample generation never freezes during the 1-15 s sklearn fit
        # + NDT + MLflow round-trip.  This used to manifest as the
        # KPI/severity charts pausing for the entire retrain — now they
        # keep streaming and the user sees a "Retraining…" badge.
        atm = self.atm
        orig_handle = atm.handle
        # Stash for the worker thread (closures can't be called from
        # outside the spy without a handle to the original).
        self._orig_atm_handle = orig_handle

        from atm.atm import ATMResult, TrainStatus  # late import: atm already loaded

        def _wrapped_handle(signal):
            # ── Warmup guard ──────────────────────────────────────────
            # Early-life detector chatter (DPD IF rate=100%, CPD
            # shadow_div=1 with corr_delta=nan) otherwise fires a burst
            # of retrains before LIB/LOB have 50 samples each.  The
            # resulting candidates train on a handful of rows and,
            # even though NDT-on-golden *should* reject them, they can
            # tie the base (min_improvement=-0.005) and ship.  We skip
            # those, but always honour a manual force-retrain.
            ctx = getattr(signal, "kpi_context", None) or {}
            is_manual = bool(ctx.get("manual")) or ctx.get("source") == "dashboard"
            if not is_manual and self._step < RETRAIN_WARMUP_SAMPLES:
                log.info(
                    "engine: retrain skipped — warmup (step=%d < %d)",
                    self._step, RETRAIN_WARMUP_SAMPLES,
                )
                self._emit({
                    "type":       "retrain_done",
                    "step":       int(signal.step),
                    "status":     "SKIPPED",
                    "variant":    None,
                    "ndt_passed": None,
                    "deployed":   False,
                    "duration_s": 0.0,
                    "message":    f"warmup guard (step<{RETRAIN_WARMUP_SAMPLES})",
                    "ndt_pseudo": None,
                    "ndt_gt":     None,
                    "ndt_bias":   None,
                })
                return ATMResult(
                    status=TrainStatus.SKIPPED,
                    variant_used=None, ndt_passed=None, deployed=False,
                    attempts=0, duration_s=0.0,
                    message=f"warmup guard (step<{RETRAIN_WARMUP_SAMPLES})",
                )

            # ── In-flight guard ───────────────────────────────────────
            # Single-trainer policy: while a retrain is running, swallow
            # additional triggers.  The detector can fire repeatedly
            # during a long ATM cycle (every drift signal queues a job),
            # and the user can spam Force retrain — both must be no-ops
            # rather than backlog stacking.  We emit a SKIPPED so the
            # timeline stays honest about what was dropped.
            if self._retrain_inflight.is_set():
                log.info(
                    "engine: retrain skipped — worker busy "
                    "(in-flight retrain at step=%s, new request at step=%d)",
                    self._retrain_meta.get("step"), int(signal.step),
                )
                self._emit({
                    "type":       "retrain_done",
                    "step":       int(signal.step),
                    "status":     "SKIPPED",
                    "variant":    None,
                    "ndt_passed": None,
                    "deployed":   False,
                    "duration_s": 0.0,
                    "message":    "retrain worker busy (already in flight)",
                    "ndt_pseudo": None,
                    "ndt_gt":     None,
                    "ndt_bias":   None,
                })
                return ATMResult(
                    status=TrainStatus.SKIPPED,
                    variant_used=None, ndt_passed=None, deployed=False,
                    attempts=0, duration_s=0.0,
                    message="retrain worker busy (already in flight)",
                )

            # ── Capture UI variant snapshot synchronously ─────────────
            # Read injection.preferred_variant on the engine thread so
            # the worker uses the value that was active at the moment
            # of the trigger, not whatever the user toggles to during
            # the 1-15 s training window.
            preferred = self.injection.snapshot()["preferred_variant"]

            # Emit retrain_start now (cheap, ~µs) so the UI shows the
            # banner immediately rather than after the worker dequeues.
            self._emit({"type": "retrain_start", "step": int(signal.step)})

            # ── Dispatch to worker thread ─────────────────────────────
            # Mark in-flight + record meta BEFORE put_nowait so a racing
            # caller observing the meta sees consistent state.  Queue
            # is single-slot (maxsize=1); since the in-flight check
            # above guarantees the slot is empty, put_nowait should
            # never raise — but guard for safety.
            job = {
                "signal":    signal,
                "preferred": preferred,
                "step":      int(signal.step),
            }
            self._retrain_meta = {
                "step":       int(signal.step),
                "started_at": time.time(),
                "variant":    preferred or "AUTO",
            }
            self._retrain_inflight.set()
            try:
                self._retrain_queue.put_nowait(job)
            except queue.Full:
                # Defensive: should not happen given the in-flight
                # check + maxsize=1, but if it does, unwind cleanly.
                self._retrain_inflight.clear()
                self._retrain_meta = {
                    "step": None, "started_at": None, "variant": None,
                }
                log.warning(
                    "engine: retrain dispatch failed — queue full at step=%d",
                    int(signal.step),
                )
                self._emit({
                    "type":       "retrain_done",
                    "step":       int(signal.step),
                    "status":     "SKIPPED",
                    "variant":    None,
                    "ndt_passed": None,
                    "deployed":   False,
                    "duration_s": 0.0,
                    "message":    "retrain queue full (race)",
                    "ndt_pseudo": None,
                    "ndt_gt":     None,
                    "ndt_bias":   None,
                })
                return ATMResult(
                    status=TrainStatus.SKIPPED,
                    variant_used=None, ndt_passed=None, deployed=False,
                    attempts=0, duration_s=0.0,
                    message="retrain queue full (race)",
                )

            # Synthetic "deferred" result.  AIMP's caller discards the
            # return value (verified at aimp/aimp.py:505), so the only
            # contract is to satisfy the type — SUCCESS with a clear
            # message documents intent without polluting timeline
            # statistics that key off SKIPPED/FAILED.
            return ATMResult(
                status=TrainStatus.SUCCESS,
                variant_used=None, ndt_passed=None, deployed=False,
                attempts=0, duration_s=0.0,
                message="deferred to retrain worker thread",
            )

        atm.handle = _wrapped_handle

    # ── retrain worker (UI-freeze fix) ────────────────────────────────────

    def _retrain_worker_loop(self) -> None:
        """Long-lived consumer that drains ``self._retrain_queue`` and runs
        ATM retrains off the engine loop, so sample generation never freezes
        during the 1-15 s sklearn fit + NDT + MLflow round-trip.

        Uses a 0.5 s ``get`` timeout so ``stop()`` can wake the worker even
        when no jobs are queued (the sentinel ``None`` is the explicit
        shutdown path).  Exceptions in ``_retrain_run_one`` are logged and
        swallowed — one bad retrain must not crash the worker.
        """
        log.info("retrain worker: started")
        while not self._stop.is_set():
            try:
                job = self._retrain_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if job is None:
                # Sentinel from ``stop()``.
                log.info("retrain worker: shutdown sentinel received")
                self._retrain_queue.task_done()
                break
            try:
                self._retrain_run_one(job)
            except Exception as exc:
                log.exception("retrain worker: job failed: %s", exc)
                # Best-effort emit so the dashboard doesn't get stuck on
                # the "Retraining…" badge.
                try:
                    self._emit({
                        "type":       "retrain_done",
                        "step":       int(job.get("step", -1)),
                        "status":     "FAILED",
                        "variant":    None,
                        "ndt_passed": None,
                        "deployed":   False,
                        "duration_s": 0.0,
                        "message":    f"worker exception: {exc}"[:200],
                        "ndt_pseudo": None,
                        "ndt_gt":     None,
                        "ndt_bias":   None,
                    })
                except Exception:
                    pass
            finally:
                # Always release in-flight + clear meta so the next
                # trigger can dispatch.  The queue is single-slot so
                # task_done() is balanced against the get() above.
                self._retrain_inflight.clear()
                self._retrain_meta = {
                    "step": None, "started_at": None, "variant": None,
                }
                try:
                    self._retrain_queue.task_done()
                except ValueError:
                    # Already marked done in some error path.
                    pass
        log.info("retrain worker: exited")

    def _retrain_run_one(self, job: dict) -> None:
        """Heavy body of one retrain cycle.  Runs on the worker thread.

        Replicates the previous inline code path:
        1) Apply UI variant override on the worker thread (mirror at end).
        2) LOB ground-truth swap (pseudo-label bias fix, training side).
        3) Class-balance augment (label-collapse guard).
        4) Call ``self._orig_atm_handle(signal)`` — the blocking sklearn
           fit + NDT + MLflow round-trip.
        5) Unwind LOB swap + buffer patches + variant override.
        6) Emit ``retrain_done`` + ``model_history``; on deploy run
           ``_on_deploy`` (which may trigger detector refit + decay).

        All emits go through ``self._emit`` which is thread-safe (queue.Queue),
        so the engine loop and worker can both write to the websocket stream.
        """
        from atm.atm import ATMResult, TrainStatus, MTPVariant

        signal    = job["signal"]
        preferred = job["preferred"]

        atm = self.atm
        orig_handle = self._orig_atm_handle

        # 1) UI variant override (mirror the synchronous behaviour the
        # closure used to do — set before orig_handle, restore after).
        prev_override = atm.policy.prefer_variant
        ui_forced_variant = True
        if preferred == "LOCAL":
            atm.policy.prefer_variant = MTPVariant.LOCAL
        elif preferred == "EXTERNAL":
            atm.policy.prefer_variant = MTPVariant.EXTERNAL
        elif preferred == "CLOUD":
            atm.policy.prefer_variant = MTPVariant.CLOUD
        else:
            # AUTO: leave whatever AIMP's MTPC has already chosen in place.
            ui_forced_variant = False

        # 2) LOB ground-truth swap (pseudo-label bias fix, training side).
        # ATM pulls training data via ``lob.get_flat_values()`` inside
        # ``orig_handle``.  To let candidates learn from *real* labels while
        # keeping CPD's correlation monitor untouched in the steady state,
        # snapshot the LOB, overwrite only the *last-N* entries (aligned
        # with the sidecar ground-truth buffer), let orig_handle train,
        # then restore.  Symmetric with the NDT-golden fix — both sides of
        # the pseudo-label loop now see ground truth during a retrain
        # cycle, and nowhere else.
        lob = self.rtp.buffers.lob
        n_swap = min(len(lob), len(self._gt_buffer))
        snapshot_vals = []
        gt_tail = []
        if n_swap > 0:
            # ``deque`` doesn't support slicing — convert once.  Cold path
            # so the O(n) materialisation is fine.
            gt_tail = list(self._gt_buffer)[-n_swap:]
            for idx in range(n_swap):
                buf_i = len(lob) - n_swap + idx
                snapshot_vals.append(lob._buf[buf_i].value)
                lob._buf[buf_i].value = np.atleast_1d(
                    np.asarray(gt_tail[idx], dtype=float)
                )

        # 3) Class-balance augment (label-collapse guard).
        # When derive_label_scalar saturates under an extreme SINR shock
        # (every sample SINR<40 → every label=1), the recent gt_tail is
        # single-class and MTP candidates trained on it collapse to
        # constant predictors that NDT correctly rejects — freezing the
        # loop into "no candidate ever ships".  Patch ATM's three buffer
        # reads to prepend a stratified sample of the missing class from
        # the historical corpus, only for this train call, so candidates
        # always have both classes.  The deploy gate (golden-holdout NDT
        # score) reads ``self._golden_X`` directly and is unaffected by
        # this patch — augmentation cannot game the validation.
        patched_methods: list[tuple[Any, str]] = []
        augment_n = 0
        if n_swap > 0:
            gt_tail_arr = np.asarray(gt_tail, dtype=int)
            if len(np.unique(gt_tail_arr)) < 2:
                minority_class = 1 - int(gt_tail_arr[0])
                archive = self.source.full_frame
                pool = archive[archive["label"] == minority_class]
                if len(pool) > 0:
                    augment_n = min(200, len(pool))
                    diverse = pool.sample(
                        n=augment_n, random_state=self.cfg.seed,
                    )
                    X_aug_raw = diverse[FEATURE_COLS].to_numpy(
                        dtype=np.float64
                    )
                    X_aug_std = (
                        X_aug_raw - self._ref_mean
                    ) / self._ref_std
                    y_aug = diverse["label"].to_numpy(dtype=np.float64)

                    lib_obj = self.rtp.buffers.lib
                    ygt_obj = self.rtp.buffers.ygt
                    orig_lib = lib_obj.get_values
                    orig_lob = lob.get_flat_values
                    orig_ygt = ygt_obj.get_flat_values

                    def _aug_lib(n=None, _o=orig_lib, _a=X_aug_std):
                        real = _o(n)
                        if _a.size == 0:
                            return real
                        if real.ndim == 1 or real.size == 0:
                            return _a.copy()
                        return np.vstack([_a, real])

                    def _aug_lob(n=None, _o=orig_lob, _a=y_aug):
                        real = _o(n)
                        if _a.size == 0:
                            return real
                        return np.concatenate([_a, real])

                    def _aug_ygt(n=None, _o=orig_ygt, _a=y_aug):
                        real = _o(n)
                        if _a.size == 0:
                            return real
                        return np.concatenate([_a, real])

                    lib_obj.get_values      = _aug_lib
                    lob.get_flat_values     = _aug_lob
                    ygt_obj.get_flat_values = _aug_ygt
                    patched_methods = [
                        (lib_obj, "get_values"),
                        (lob,     "get_flat_values"),
                        (ygt_obj, "get_flat_values"),
                    ]
                    log.info(
                        "engine: class-balance augment — patched "
                        "buffers to prepend %d minority(class=%d) "
                        "rows from archive (label-collapse guard)",
                        augment_n, minority_class,
                    )
                    try:
                        self._emit({
                            "type":           "class_balance_augment",
                            "step":           int(signal.step),
                            "n_augmented":    int(augment_n),
                            "minority_class": int(minority_class),
                        })
                    except Exception:
                        pass

        # 4) Heavy call.
        try:
            res = orig_handle(signal)
        finally:
            # 5) Unwind, mirroring the original closure's finally block.
            # Only roll back the override we actually set.  When UI is AUTO
            # we never wrote to prefer_variant, so AIMP's finally block
            # will restore the pre-MTPC value correctly.
            if ui_forced_variant:
                atm.policy.prefer_variant = prev_override
            # Restore LOB pseudo-labels so CPD keeps seeing the natural
            # prediction-vs-feature correlation between retrains.
            if n_swap > 0:
                for idx in range(n_swap):
                    buf_i = len(lob) - n_swap + idx
                    if 0 <= buf_i < len(lob):
                        lob._buf[buf_i].value = snapshot_vals[idx]
            # Restore patched buffer reads so detectors / next-tick observe()
            # see the real LIB/LOB/YGT.  delattr removes the instance shadow,
            # exposing the class methods again.
            for obj, name in patched_methods:
                try:
                    delattr(obj, name)
                except AttributeError:
                    pass

        # 6) Emit retrain_done + model_history; on deploy run _on_deploy.
        try:
            self._emit({
                "type":       "retrain_done",
                "step":       int(signal.step),
                "status":     getattr(res.status, "name", str(res.status)),
                "variant":    res.variant_used.value if res.variant_used else None,
                "ndt_passed": bool(res.ndt_passed) if res.ndt_passed is not None else None,
                "deployed":   bool(res.deployed),
                "duration_s": float(res.duration_s),
                "message":    str(res.message)[:200],
                "ndt_pseudo": self._last_ndt_pseudo,
                "ndt_gt":     self._last_ndt_gt,
                "ndt_bias":   self._last_ndt_bias,
            })
            self._emit_model_history()

            # Post-deploy actions
            if res.deployed:
                self._on_deploy(step=int(signal.step))
        except Exception as exc:
            log.debug("retrain emit failed: %s", exc)

    # ── post-deploy hooks ─────────────────────────────────────────────────

    def _on_deploy(self, step: int) -> None:
        """Run after a model is successfully deployed."""
        # A) Honest reference refit — DDD/CDD stop firing on the stale ref
        self._refit_detector_reference()
        # B) Closed-loop simulation — decay injection sliders to zero
        if self.injection.snapshot()["closed_loop_enabled"]:
            self._start_closed_loop_decay(step=step)
        # C) Emit a marker for the accuracy chart
        self._emit({"type": "retrain_marker", "step": step})

    def _refit_detector_reference(self) -> None:
        """Replace the RTP reference buffer with the recent standardized LIB.

        Matches the thesis-documented refit-reference fix: after the model has
        learned a new regime, the detectors must also forget the ancient
        reference or they will keep firing on the new steady-state.
        """
        if len(self._recent_x_raw) < 50:
            return
        try:
            X_raw = np.asarray(self._recent_x_raw, dtype=np.float64)
            X_std = (X_raw - self._ref_mean) / self._ref_std
            # We reuse the current LOB predictions as pseudo-y for the ref,
            # which is fine because these are detector references, not NDT.
            y_dummy = np.zeros(len(X_std), dtype=np.int64)
            # Ensure both classes present so DDD internals don't barf
            y_dummy[: max(5, len(y_dummy) // 20)] = 1
            self.rtp.set_reference(X_std, y_dummy)
            log.info("engine: detector reference refitted on %d recent samples",
                     len(X_std))
            self._emit({
                "type": "reference_refit",
                "step": self._step,
                "n":    int(len(X_std)),
            })
        except Exception as exc:
            log.warning("engine: reference refit failed: %s", exc)

    def _start_closed_loop_decay(self, step: int) -> None:
        snap = self.injection.snapshot()
        with self._decay_lock:
            self._decay_active = True
            self._decay_t0 = time.monotonic()
            self._decay_from = {
                "sinr_bias_db":  snap["sinr_bias_db"],
                "rsrp_bias_db":  snap["rsrp_bias_db"],
                "delay_bias_ms": snap["delay_bias_ms"],
                "tput_scale":    snap["tput_scale"],
                "noise_scale":   snap["noise_scale"],
            }
        self._emit({
            "type": "closed_loop_start",
            "step": step,
            "duration_s": CLOSED_LOOP_DECAY_S,
            "from": self._decay_from,
        })

    def _tick_closed_loop_decay(self) -> None:
        """Linearly move injection sliders toward neutral values."""
        with self._decay_lock:
            if not self._decay_active:
                return
            elapsed = time.monotonic() - self._decay_t0
            if elapsed >= CLOSED_LOOP_DECAY_S:
                self._decay_active = False
                self.injection.update({
                    "sinr_bias_db":  0.0,
                    "rsrp_bias_db":  0.0,
                    "delay_bias_ms": 0.0,
                    "tput_scale":    1.0,
                    "noise_scale":   1.0,
                }, _source="decay")
                self._emit({"type": "closed_loop_end", "step": self._step})
                return
            frac = elapsed / CLOSED_LOOP_DECAY_S
            f = self._decay_from
            # Ease-out (cubic) toward neutral
            k = 1.0 - (1.0 - frac) ** 3
            self.injection.update({
                "sinr_bias_db":  f["sinr_bias_db"]  * (1 - k),
                "rsrp_bias_db":  f["rsrp_bias_db"]  * (1 - k),
                "delay_bias_ms": f["delay_bias_ms"] * (1 - k),
                "tput_scale":    f["tput_scale"]    + (1.0 - f["tput_scale"])  * k,
                "noise_scale":   f["noise_scale"]   + (1.0 - f["noise_scale"]) * k,
            }, _source="decay")

    def _cancel_closed_loop_decay(self) -> None:
        """Stop any in-flight decay — called when the user nudges a slider.

        Runs *outside* any InjectionState lock (see InjectionState.update) so
        it is safe to take the decay lock here without risking inversion.
        """
        with self._decay_lock:
            if not self._decay_active:
                return
            self._decay_active = False
        # Emit outside the lock to avoid holding it across queue.put().
        try:
            self._emit({
                "type": "closed_loop_end",
                "step": self._step,
                "reason": "user_override",
            })
        except Exception:
            pass

    # ── main loop ────────────────────────────────────────────────────────

    def _loop(self) -> None:
        assert self.source is not None and self.rtp is not None
        it = iter(self.source)
        next_tick = time.monotonic()

        try:
            while not self._stop.is_set():
                if self.injection.consume_reset_flag():
                    self.source.reset()
                    if self.ran is not None:
                        self.ran.reset()
                    self._step = 0
                    self._correct = 0
                    self._total = 0
                    self._recent_x_raw.clear()
                    self._gt_buffer.clear()
                    self._emit({"type": "reset"})

                if self.injection.consume_retrain_flag():
                    self._force_retrain_once()

                if self._paused:
                    time.sleep(0.05)
                    continue

                # Closed-loop decay (runs regardless of tick pacing)
                self._tick_closed_loop_decay()

                interval = 1.0 / max(0.5, self._rate_hz)
                now = time.monotonic()
                if now < next_tick:
                    time.sleep(min(0.05, next_tick - now))
                    continue
                next_tick = now + interval

                try:
                    row = next(it)
                except StopIteration:
                    self.source.reset()
                    it = iter(self.source)
                    continue

                x_raw = apply_injections(row, self.injection, self.rng)
                # Re-derive y_true from the *injected* KPIs so drift actually
                # produces wrong-answers that the MLIN must adapt to; using
                # the pre-injection row["label"] would hide all injected drift.
                y_true = derive_label_scalar(
                    rsrp_dbm=float(x_raw[0]),
                    sinr_db=float(x_raw[1]),
                    tput_mbps=float(x_raw[2]),
                    delay_ms=float(x_raw[3]),
                    handover_flag=int(row.get("handover_flag", 0) or 0),
                    sinr_thresh_db  = self._sinr_thresh_db,
                    delay_thresh_ms = self._delay_thresh_ms,
                )

                if self._ref_mean is not None and self._ref_std is not None:
                    x = (x_raw - self._ref_mean) / self._ref_std
                else:
                    x = x_raw

                try:
                    pred = self.rtp.observe(x, y_true=y_true)
                    y_pred = int(np.asarray(pred).ravel()[0])
                except Exception as exc:
                    log.error("rtp.observe failed at step=%d: %s", self._step, exc)
                    continue

                # Keep a rolling LIB snapshot for reference refit
                self._recent_x_raw.append(x_raw)
                # Sidecar ground-truth buffer, aligned 1:1 with the RTP's LOB.
                # Training spies fetch from this (instead of the pseudo-label
                # LOB) so MTP candidates learn from real labels without
                # disturbing the RTP's natural pseudo-label flow — which CPD
                # legitimately monitors for *prediction* drift.
                # Both buffers are bounded ``deque``s — eviction is O(1).
                self._gt_buffer.append(int(y_true))

                correct = int(y_pred == y_true)
                self._step += 1
                self._total += 1
                self._correct += correct

                # One-shot detector-reference rebase on live data.  The
                # initial reference came from the cached corpus snapshot
                # (or RANLiveSource's pre-generated corpus) which subtly
                # differs from the actual streaming distribution — DDD/CDD
                # see that mismatch as drift and false-fire.  After 250
                # live samples we have enough to characterize the live
                # distribution; rebase once and let real drift take over.
                # Skipped in CSV mode: the corpus IS the stream, so the
                # original reference is already correct.
                if (
                    self.cfg.live_mode
                    and not self._initial_ref_rebased
                    and self._step >= 250
                    and len(self._recent_x_raw) >= 200
                ):
                    self._initial_ref_rebased = True
                    log.info(
                        "engine: rebasing detector reference on %d live samples "
                        "(corpus→live distribution shift)",
                        len(self._recent_x_raw),
                    )
                    self._refit_detector_reference()

                self._emit({
                    "type":   "sample",
                    "step":   self._step,
                    "phase":  row.get("phase"),
                    "ue":     int(row.get("ue", 0)),
                    "t":      float(row.get("t", 0.0)),
                    "x":      [float(v) for v in x_raw],
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "correct": correct,
                })

                # Cache the latest KPI tail for the actuator's symptom
                # diagnosis (rsrp_low → handover, delay_high → URLLC, etc.).
                self._last_kpi_tail = {
                    "rsrp_dbm":        float(x_raw[0]),
                    "sinr_db":         float(x_raw[1]),
                    "throughput_mbps": float(x_raw[2]),
                    "delay_ms":        float(x_raw[3]),
                }

                # Cache ``now`` once — avoid double monotonic() at 200 Hz.
                _now = time.monotonic()
                if _now - self._last_det_emit > 0.05:
                    det_ev = self._emit_detector_snapshot()
                    self._last_det_emit = _now

                    # ── Closed-loop RAN: detector → actuator → simulator ──
                    # Suppress the actuator until the live-distribution
                    # rebase has happened (Live mode only).  Otherwise the
                    # cached-corpus reference triggers DDD on the very first
                    # sample, the actuator slams in an interferer-null /
                    # handover, and the resulting KPI shift contaminates the
                    # samples we'd use for the rebase — leaving the detectors
                    # noisy forever.  Pre-rebase fires are still emitted as
                    # observability events; they just don't drive the RAN.
                    actuator_armed = (
                        not self.cfg.live_mode
                        or self._initial_ref_rebased
                    )
                    if (
                        self.actuator is not None
                        and self.ran is not None
                        and det_ev is not None
                        and actuator_armed
                    ):
                        try:
                            action = self.actuator.on_detector_event(
                                det_ev, self._last_kpi_tail,
                            )
                        except Exception as exc:
                            log.warning("actuator failed: %s", exc)
                            action = None
                        if action is not None:
                            self._emit({
                                "type":   "ran_action",
                                "step":   self._step,
                                "action": {
                                    "type":           action.type.value,
                                    "delta":          float(action.delta),
                                    "duration_s":     float(action.duration_s),
                                    "reason":         action.reason,
                                    "issued_at_step": int(action.issued_at_step),
                                },
                                "ran_state": self.ran.snapshot_state(),
                            })

        except Exception as exc:
            log.exception("engine loop crashed: %s", exc)
            self._emit({"type": "error", "message": str(exc)})
        finally:
            self._emit({"type": "stopped"})

    def _force_retrain_once(self) -> None:
        try:
            from rtp.rtp import MToUTSignal, TriggerReason
            sig = MToUTSignal(
                reasons=[TriggerReason.CONCEPT_DRIFT, TriggerReason.DATA_DRIFT],
                step=self._step,
                kpi_context={"manual": True, "source": "dashboard"},
            )
            self._emit({"type": "mtout", "step": self._step,
                        "severity": "MANUAL", "reasons": ["MANUAL_FORCE"]})
            self.atm.handle(sig)
        except Exception as exc:
            log.error("force_retrain failed: %s", exc)
            self._emit({"type": "error", "message": f"force_retrain: {exc}"})

    # ── event helpers ────────────────────────────────────────────────────

    def _emit(self, ev: dict) -> None:
        try:
            self.events.put_nowait(ev)
        except queue.Full:
            try:
                self.events.get_nowait()
                self.events.put_nowait(ev)
            except Exception:
                pass

    def _emit_detector_snapshot(self) -> Optional[dict]:
        rtp = self.rtp
        if rtp is None:
            return None

        def _ddd(r):
            if r is None: return None
            return {
                "triggered": bool(r.drift_detected),
                "ks_max_p":  float(max(r.ks_pvalues)) if len(r.ks_pvalues) > 0 else None,
                "mmd":       float(r.mmd_statistic) if r.mmd_statistic is not None else None,
            }
        def _dpd(r):
            if r is None: return None
            return {
                "triggered": bool(r.poisoning_detected),
                "if_rate":   float(r.if_anomaly_rate),
                "mahal_max": float(r.mahal_max) if r.mahal_max is not None else None,
            }
        def _cdd(r):
            if r is None: return None
            return {
                "triggered": bool(r.drift_detected),
                "ph_stat":   float(r.ph_statistic),
                "perf_drop": float(r.perf_drop) if r.perf_drop is not None else None,
            }
        def _cpd(r):
            if r is None: return None
            return {
                "triggered":        bool(r.poisoning_detected),
                "shadow_divergence": float(r.shadow_divergence)
                                       if r.shadow_divergence is not None else None,
            }

        ev = {
            "type": "detector",
            "step": self._step,
            "ddd":  _ddd(rtp.last_ddd),
            "dpd":  _dpd(rtp.last_dpd),
            "cdd":  _cdd(rtp.last_cdd),
            "cpd":  _cpd(rtp.last_cpd),
        }
        self._emit(ev)
        return ev

    def _emit_model_history(self) -> None:
        try:
            hist = self.aimp.get_model_history(aif_id=1)
            self._emit({
                "type": "model_history",
                "versions": [
                    {
                        "version":  int(e["version"]),
                        "source":   str(e["source"]),
                        "accuracy": float(e["metadata"].get("accuracy", 0.0))
                                    if isinstance(e.get("metadata"), dict) else 0.0,
                    }
                    for e in hist
                ],
            })
        except Exception as exc:
            log.debug("model_history emit failed: %s", exc)
