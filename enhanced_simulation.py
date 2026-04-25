"""
enhanced_simulation.py — UERANSIM-aligned multi-seed simulation for 5G NR networks.

Data generation follows the 3GPP NR measurement report model used by UERANSIM
(https://github.com/aligungr/UERANSIM), an open-source 5G UE/gNB simulator.
The feature ranges are derived from 3GPP TS 38.215 / TS 38.133 and the
UERANSIM ASN.1 definitions (ASN_RRC_RSRP_Range 0-127, ASN_RRC_SINR_Range
0-127), which map to the following physical quantities:

    RSRP (SS-RSRP):  -156 to -31 dBm   (3GPP TS 38.215 §5.1.1)
    SINR (SS-SINR):  -23  to  40 dB     (3GPP TS 38.215 §5.1.5)
    Throughput:        0   to 1000 Mbps  (gNB-reported, UE-aggregated)
    Latency (RTT):     1   to 100 ms     (N3/GTP-U measured RTT)

The classification task is 5G NR handover prediction: given a UE measurement
report, decide whether to trigger Xn/NG handover (label=1) or stay (label=0).
The decision boundary models the UERANSIM A3 event + latency budget:
    handover = (RSRP < -100 AND SINR < 5)  OR  latency > 50

Features
--------
1. UERANSIM-aligned 5G NR KPI features with 3GPP-compliant value ranges.
2. Multi-seed analysis (30 seeds) for statistically robust evaluation.
3. NDT ground-truth validation to expose the self-referential pseudo-label issue.
4. Enhanced attack scenarios: subtle (3σ contamination) vs aggressive poisoning.
5. Check-interval sensitivity analysis: detection latency vs overhead trade-off.
6. Publication-quality figures saved to thesis/figures/.
7. CSV result tables saved to thesis/results/.

Run:
    python enhanced_simulation.py
    powershell.exe -Command "py enhanced_simulation.py"
"""

from __future__ import annotations

import copy
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

# Suppress sklearn/scipy deprecation noise during large batch runs.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Matplotlib setup — attempt seaborn grid style, fall back gracefully
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe on Windows CI)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_STYLE_CANDIDATES = [
    "seaborn-v0_8-whitegrid",   # matplotlib >= 3.6
    "seaborn-whitegrid",        # matplotlib < 3.6
    "ggplot",
    "default",
]
for _style in _STYLE_CANDIDATES:
    try:
        plt.style.use(_style)
        break
    except OSError:
        continue

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import is_classifier

from aif.aif    import AIF
from rtp.rtp    import RTP, RTPConfig, MToUTSignal, RTPEvent
from atm.atm    import ATM, ATMPolicy, MTPVariant
from atm.mtp_l  import MTPLocal
from atm.mtp_e  import MTPExternal
from ndt.ndt    import NDT

# ---------------------------------------------------------------------------
# Logging — keep WARNING-level to avoid flooding during 30-seed runs
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)-8s %(name)-20s — %(message)s",
)
log = logging.getLogger("ENH_SIM")
log.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Directory setup — must exist before any file write
# ---------------------------------------------------------------------------
FIGURES_DIR = os.path.join("thesis", "figures")
RESULTS_DIR = os.path.join("thesis", "results")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Simulation constants (matching main.py phase structure)
# ---------------------------------------------------------------------------
N_SEEDS           = 30
PHASE1_STEPS      = 400
PHASE2_STEPS      = 200
PHASE3A_STEPS     = 50    # subtle poisoning (new)
PHASE3B_STEPS     = 50    # aggressive poisoning
PHASE4_STEPS      = 200

TOTAL_STEPS = PHASE1_STEPS + PHASE2_STEPS + PHASE3A_STEPS + PHASE3B_STEPS + PHASE4_STEPS

CHECK_INTERVALS   = [10, 25, 50, 100]   # for sensitivity analysis

# 5G NR KPI ranges (3GPP TS 38.215 / TS 38.133, UERANSIM ASN.1 mappings)
# ASN_RRC_RSRP_Range [0..127] maps to -156..-31 dBm (3GPP TS 38.133 Table 10.1.6.1-1)
# ASN_RRC_SINR_Range [0..127] maps to -23..40 dB    (3GPP TS 38.133 Table 10.1.16.1-1)
RSRP_RANGE        = (-156.0, -31.0)    # dBm  (SS-RSRP, 3GPP TS 38.215 §5.1.1)
SINR_RANGE        = (-23.0, 40.0)      # dB   (SS-SINR, 3GPP TS 38.215 §5.1.5)
THROUGHPUT_RANGE   = (0.0, 1000.0)     # Mbps (gNB DL throughput per UE)
LATENCY_RANGE      = (1.0, 100.0)     # ms   (N3 GTP-U RTT)

# UERANSIM gNB/UE configuration references (from config/*.yaml)
UERANSIM_MCC = '999'          # Mobile Country Code (test network)
UERANSIM_MNC = '70'           # Mobile Network Code (test network)
UERANSIM_SST = 1              # Network Slice Selection (eMBB)
UERANSIM_TAC = 1              # Tracking Area Code


# ===========================================================================
# Section 1 — UERANSIM-aligned 5G NR KPI dataset generator
# ===========================================================================

def _handover_label(rsrp: np.ndarray, sinr: np.ndarray, latency: np.ndarray) -> np.ndarray:
    """
    5G NR handover decision boundary modelling UERANSIM A3 event trigger.

    In 3GPP TS 38.331, the A3 event fires when a neighbour cell's RSRP
    exceeds the serving cell's by an offset. We simplify this to an absolute
    threshold model consistent with UERANSIM's simulation abstraction:

    Handover needed (1) when:
        (SS-RSRP < -100 dBm  AND  SS-SINR < 5 dB)  — weak signal + poor quality
        OR
        N3 RTT > 50 ms                               — latency budget exhausted

    The first condition captures radio-layer degradation (A3-like); the second
    captures transport-layer congestion that UERANSIM models via UDP RTT.

    Returns binary label array of shape (n,).
    """
    weak_signal = (rsrp < -100.0) & (sinr < 5.0)
    high_latency = latency > 50.0
    return (weak_signal | high_latency).astype(int)


def make_kpi_data(
    n: int,
    rng: np.random.Generator,
    *,
    # Phase-specific distribution shifts
    rsrp_shift:       float = 0.0,    # dBm offset to entire distribution
    sinr_shift:       float = 0.0,    # dB offset
    latency_shift:    float = 0.0,    # ms offset
    throughput_scale: float = 1.0,    # multiplicative scale
    label_noise:      float = 0.03,   # fraction of labels flipped (concept noise)
    # Poisoning parameters
    poison_fraction:  float = 0.0,    # fraction of samples to corrupt
    poison_mode:      str   = "none", # "none" | "subtle" | "aggressive"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate UERANSIM-aligned 5G NR measurement report data.

    Simulates UE measurement reports as defined in 3GPP TS 38.331 §5.5.5
    (MeasurementReport), with features derived from UERANSIM's gNB radio
    layer abstraction and N3/GTP-U transport metrics.

    Features (columns):
        0: SS-RSRP     dBm   [-156, -31]  (3GPP TS 38.215 §5.1.1)
        1: SS-SINR     dB    [-23,  40]   (3GPP TS 38.215 §5.1.5)
        2: DL throughput Mbps [0, 1000]   (gNB RLC-layer reported)
        3: N3 RTT       ms   [1, 100]    (GTP-U round-trip time)

    Correlation structure:
        RSRP → SINR  (positive: signal strength drives quality)
        SINR → throughput (positive: better channel → higher rate)
        RSRP → latency (negative: weaker signal → more retransmissions)

    Returns
    -------
    X : np.ndarray, shape (n, 4)
    y : np.ndarray, shape (n,)   — binary handover label (1=handover, 0=stay)
    """
    # --- Generate correlated KPI base distributions -------------------------
    # Physical model: RSRP drives SINR (path loss correlation), SINR drives
    # achievable throughput (Shannon capacity), RSRP inversely drives latency
    # (weaker signals cause HARQ retransmissions). These correlations mirror
    # the radio channel model abstracted by UERANSIM's RLS (Radio Link
    # Simulation) layer over UDP.

    rsrp_raw  = rng.normal(-90.0 + rsrp_shift, 18.0, n)
    sinr_raw  = rng.normal(10.0  + sinr_shift + 0.12 * (rsrp_raw - (-90.0)), 5.0, n)
    tput_raw  = throughput_scale * np.clip(
        rng.normal(10.0 * sinr_raw + 150.0, 80.0, n), 0.0, 1000.0
    )
    latency_raw = np.clip(
        rng.normal(25.0 - 0.15 * rsrp_raw + latency_shift, 12.0, n), 1.0, 500.0
    )

    # Clip to realistic KPI ranges
    rsrp     = np.clip(rsrp_raw,     RSRP_RANGE[0],       RSRP_RANGE[1])
    sinr     = np.clip(sinr_raw,     SINR_RANGE[0],       SINR_RANGE[1])
    tput     = np.clip(tput_raw,     THROUGHPUT_RANGE[0], THROUGHPUT_RANGE[1])
    latency  = np.clip(latency_raw,  LATENCY_RANGE[0],    LATENCY_RANGE[1])

    X = np.column_stack([rsrp, sinr, tput, latency])
    y = _handover_label(rsrp, sinr, latency)

    # --- Concept noise (label flip) ----------------------------------------
    if label_noise > 0:
        flip_idx = rng.random(n) < label_noise
        y[flip_idx] = 1 - y[flip_idx]

    # --- Data poisoning injection ------------------------------------------
    if poison_fraction > 0 and poison_mode != "none":
        n_poison = max(1, int(n * poison_fraction))
        poison_idx = rng.choice(n, size=n_poison, replace=False)

        if poison_mode == "subtle":
            # Values within 3-sigma of normal — hard to detect
            feature_stds = np.array([18.0, 5.0, 80.0, 12.0])
            feature_means = np.array([-90.0, 10.0, 300.0, 25.0])
            for fi in poison_idx:
                noise = rng.normal(0, 2.5 * feature_stds)
                X[fi] = np.clip(feature_means + noise,
                                [RSRP_RANGE[0], SINR_RANGE[0], THROUGHPUT_RANGE[0], LATENCY_RANGE[0]],
                                [RSRP_RANGE[1], SINR_RANGE[1], THROUGHPUT_RANGE[1], LATENCY_RANGE[1]])
                # Subtly flip the label with 70% probability
                if rng.random() < 0.70:
                    y[fi] = 1 - y[fi]

        elif poison_mode == "aggressive":
            # Extreme outliers: 30-50x outside normal ranges
            scale_factors = rng.uniform(30.0, 50.0, (n_poison, 4))
            sign_flip = rng.choice([-1, 1], size=(n_poison, 4))
            base_means = np.array([-90.0, 10.0, 300.0, 25.0])
            for i, fi in enumerate(poison_idx):
                X[fi] = base_means + sign_flip[i] * scale_factors[i] * np.array([1.0, 1.0, 10.0, 1.0])
            y[poison_idx] = rng.integers(0, 2, size=n_poison)

    return X, y


# ===========================================================================
# Section 2 — RTPConfig matching main.py
# ===========================================================================

def build_rtpconfig(check_interval: int = 50) -> RTPConfig:
    """Build the RTPConfig with the same parameters as main.py."""
    return RTPConfig(
        buffer_maxlen=2000,
        check_interval=check_interval,
        cdd_task="classifier",
        ddd_reference_size=200,
        ddd_recent_size=100,
        dpd_reference_size=200,
        dpd_recent_size=50,
        dpd_contamination_threshold=0.08,
        dpd_mahal_threshold=5.0,
        cdd_reference_window=150,
        cdd_recent_window=50,
        cdd_perf_drop_threshold=0.12,
        cdd_ph_lambda=40.0,
        cpd_reference_size=200,
        cpd_recent_size=100,
        cpd_shadow_threshold=0.38,
        cpd_output_ks_alpha=0.0001,
        cpd_corr_threshold=0.60,
        mtout_cooldown_steps=150,
    )


# ===========================================================================
# Section 3 — Ground-truth NDT validation wrapper
# ===========================================================================

@dataclass
class NDTDualResult:
    """Side-by-side NDT scores: pseudo-label (LOB) vs ground-truth."""
    pseudo_label_score:   float
    ground_truth_score:   float
    pseudo_label_passed:  bool
    ground_truth_passed:  bool
    improvement_pseudo:   float
    improvement_gt:       float


class NDTGroundTruth:
    """
    Wraps the standard NDT and adds a parallel ground-truth evaluation path.

    The standard NDT validates against LOB (pseudo-labels from MLIN).
    This wrapper also scores the candidate against the TRUE y values,
    exposing any self-referential bias in the pseudo-label approach.

    Parameters
    ----------
    ndt : NDT
        The standard NDT instance (uses LOB pseudo-labels).
    min_score : float
        Minimum acceptable accuracy on ground-truth labels.
    """

    def __init__(self, ndt: NDT, min_score: float = 0.65) -> None:
        self._ndt = ndt
        self.min_score = min_score
        self.dual_history: list[NDTDualResult] = []

    def validate_dual(
        self,
        candidate,
        X_val: np.ndarray,
        y_pseudo: np.ndarray,   # LOB outputs (pseudo-labels)
        y_true: np.ndarray,     # ground-truth labels
    ) -> NDTDualResult:
        """
        Evaluate candidate model against both pseudo-labels and ground truth.

        Returns
        -------
        NDTDualResult
        """
        # Standard NDT call — uses LOB pseudo-labels
        pseudo_passed = self._ndt.validate(
            candidate, X_val=X_val, y_val=y_pseudo,
            min_score=self.min_score,
        )
        pseudo_score = self._ndt.last_result()["candidate_score"]

        # Ground-truth evaluation — direct accuracy computation
        preds = candidate.predict(X_val)
        gt_score = float(np.mean(preds == y_true.ravel()))

        # Baseline for improvement comparison
        current = self._ndt._get_current()
        if current is not None:
            try:
                baseline_gt = float(np.mean(current.predict(X_val) == y_true.ravel()))
            except Exception:
                baseline_gt = 0.0
        else:
            baseline_gt = 0.0

        gt_passed = gt_score >= self.min_score

        result = NDTDualResult(
            pseudo_label_score   = pseudo_score,
            ground_truth_score   = gt_score,
            pseudo_label_passed  = pseudo_passed,
            ground_truth_passed  = gt_passed,
            improvement_pseudo   = self._ndt.last_result()["improvement"],
            improvement_gt       = gt_score - baseline_gt,
        )
        self.dual_history.append(result)
        return result


# ===========================================================================
# Section 4 — One full simulation run
# ===========================================================================

@dataclass
class SeedResult:
    """All metrics collected from a single seed's simulation run."""
    seed: int
    check_interval: int

    # Detector precision/recall (TP/FP/FN tracked per phase)
    ddd_tp: int = 0
    ddd_fp: int = 0
    ddd_fn: int = 0
    dpd_tp: int = 0
    dpd_fp: int = 0
    dpd_fn: int = 0
    cdd_tp: int = 0
    cdd_fp: int = 0
    cdd_fn: int = 0
    cpd_tp: int = 0
    cpd_fp: int = 0
    cpd_fn: int = 0

    # Detection latency (steps from anomaly start to first alert)
    ddd_latency: Optional[float] = None
    dpd_latency: Optional[float] = None
    cdd_latency: Optional[float] = None
    cpd_latency: Optional[float] = None

    # Accuracy by phase
    phase1_accuracy: float = 0.0
    phase2_accuracy: float = 0.0
    phase4_accuracy: float = 0.0

    # NDT scores
    ndt_pseudo_scores:  list[float] = field(default_factory=list)
    ndt_gt_scores:      list[float] = field(default_factory=list)

    # ATM training time (seconds)
    atm_training_times: list[float] = field(default_factory=list)

    # Rolling accuracy trace for confidence band plots (one value per step)
    rolling_accuracy: list[float] = field(default_factory=list)

    # Event counts
    total_mtout_signals: int = 0
    total_security_alerts: int = 0
    atm_deployments: int = 0

    def precision(self, tp: int, fp: int) -> float:
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def recall(self, tp: int, fn: int) -> float:
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @property
    def ddd_precision(self) -> float: return self.precision(self.ddd_tp, self.ddd_fp)
    @property
    def ddd_recall(self)    -> float: return self.recall(self.ddd_tp, self.ddd_fn)
    @property
    def dpd_precision(self) -> float: return self.precision(self.dpd_tp, self.dpd_fp)
    @property
    def dpd_recall(self)    -> float: return self.recall(self.dpd_tp, self.dpd_fn)
    @property
    def cdd_precision(self) -> float: return self.precision(self.cdd_tp, self.cdd_fp)
    @property
    def cdd_recall(self)    -> float: return self.recall(self.cdd_tp, self.cdd_fn)
    @property
    def cpd_precision(self) -> float: return self.precision(self.cpd_tp, self.cpd_fp)
    @property
    def cpd_recall(self)    -> float: return self.recall(self.cpd_tp, self.cpd_fn)


def run_single_seed(
    seed: int,
    check_interval: int = 50,
    verbose: bool = False,
) -> SeedResult:
    """
    Execute the full 5-phase simulation for a single random seed.

    Phase structure
    ---------------
    Phase 1 (steps   1 -  400): Stable 5G/6G KPIs — no alerts expected.
    Phase 2 (steps 401 -  600): Concept drift (RSRP degrades, latency spikes).
    Phase 3a(steps 601 -  650): Subtle poisoning (10% contamination, 3-sigma).
    Phase 3b(steps 651 -  700): Aggressive poisoning (extreme outliers).
    Phase 4 (steps 701 -  900): Recovery — clean KPIs.

    Ground-truth labels are tracked throughout for NDT dual validation.

    Returns
    -------
    SeedResult
    """
    rng = np.random.default_rng(seed)
    result = SeedResult(seed=seed, check_interval=check_interval)

    # ── Initial training data ─────────────────────────────────────────────
    X_train, y_train = make_kpi_data(500, rng, label_noise=0.03)
    clf = RandomForestClassifier(n_estimators=50, random_state=seed).fit(X_train, y_train)
    aif = AIF(clf)

    # ── RTP configuration ─────────────────────────────────────────────────
    cfg = build_rtpconfig(check_interval=check_interval)

    received_signals: list[MToUTSignal] = []
    security_alerts:  list[RTPEvent]    = []

    def on_mtout(signal: MToUTSignal) -> None:
        received_signals.append(signal)

    def on_security(event: RTPEvent) -> None:
        security_alerts.append(event)

    rtp = RTP(aif, config=cfg, on_mtout=on_mtout, on_security_alert=on_security)

    X_ref, y_ref = make_kpi_data(300, rng, label_noise=0.03)
    lob_ref = clf.predict(X_ref)
    rtp.set_reference(X_ref, y_ref, lob_ref)

    # ── ATM / MTP components ──────────────────────────────────────────────
    mtp_local = MTPLocal(n_splits=3, fine_tune_first=True, random_state=seed)

    ndt_inner = NDT(
        current_model_getter=lambda: rtp.aif.active_estimator,
        min_score=0.65,
        min_improvement=-0.05,
    )
    ndt_dual = NDTGroundTruth(ndt_inner, min_score=0.65)

    policy = ATMPolicy(
        prefer_variant=MTPVariant.LOCAL,    # always local for speed in multi-seed
        local_max_samples=600,
        use_ndt=True,
        ndt_min_accuracy=0.65,
        auto_deploy=True,
        max_retrain_attempts=2,
    )

    atm_results_local: list = []

    def on_atm_result(r) -> None:
        atm_results_local.append(r)
        if r.duration_s > 0:
            result.atm_training_times.append(r.duration_s)
        if r.deployed:
            result.atm_deployments += 1

    atm = ATM(
        rtp=rtp,
        mtp_l=mtp_local,
        mtp_e=None,     # MTP-E skipped — no MLflow server required in multi-seed
        ndt=ndt_inner,
        policy=policy,
        on_result=on_atm_result,
    )

    # Wire ATM into RTP callback — also collect NDT dual results during each ATM cycle
    def _full_mtout_handler(signal: MToUTSignal) -> None:
        on_mtout(signal)
        t0 = time.time()

        # Run ATM training
        atm_result = atm.handle(signal)

        # Compute NDT dual score using the newly installed model (after deploy)
        # X_val comes from current LIB, y_pseudo from LOB, y_true from ground truth.
        lib = rtp.buffers.lib
        lob = rtp.buffers.lob
        n_val = min(200, len(lib))
        if n_val >= 10:
            X_val = lib.get_values(n_val)
            y_pseudo = lob.get_flat_values(n_val)
            # Re-derive ground truth for the same LIB window using the decision boundary.
            # This is valid because the KPI data generator is deterministic given the inputs.
            rsrp_v   = X_val[:, 0]
            sinr_v   = X_val[:, 1]
            lat_v    = X_val[:, 3]
            y_gt_val = _handover_label(rsrp_v, sinr_v, lat_v)

            if atm_result.deployed:
                dual = ndt_dual.validate_dual(
                    candidate=rtp.aif.active_estimator,
                    X_val=X_val,
                    y_pseudo=y_pseudo,
                    y_true=y_gt_val,
                )
                result.ndt_pseudo_scores.append(dual.pseudo_label_score)
                result.ndt_gt_scores.append(dual.ground_truth_score)

    rtp._on_mtout = _full_mtout_handler

    # ── Detection tracking helpers ────────────────────────────────────────
    # Per-check-cycle TP/FP/FN tracking (NOT binary per-phase).
    # At each check cycle:
    #   - Detector fires during anomaly phase → TP
    #   - Detector fires during benign phase  → FP
    #   - Detector silent during anomaly phase → FN
    #   - Detector silent during benign phase  → TN (not tracked)
    # This gives continuous precision/recall rather than trivial binary 0/1.

    # Phase boundaries (1-indexed steps)
    DRIFT_START    = PHASE1_STEPS + 1
    DRIFT_END      = PHASE1_STEPS + PHASE2_STEPS
    POISON_START   = DRIFT_END + 1
    POISON_END     = DRIFT_END + PHASE3A_STEPS + PHASE3B_STEPS

    # Track first-alert steps for latency computation
    ddd_first_alert_step: Optional[int] = None
    dpd_first_alert_step: Optional[int] = None
    cdd_first_alert_step: Optional[int] = None
    cpd_first_alert_step: Optional[int] = None

    def _classify_alert(step: int) -> str:
        """Return 'drift', 'poison', 'stable', or 'recovery'."""
        if DRIFT_START <= step <= DRIFT_END:
            return "drift"
        if POISON_START <= step <= POISON_END:
            return "poison"
        if step > POISON_END:
            return "recovery"
        return "stable"

    # DDD/CDD target drift+poison; DPD/CPD target poison only
    drift_targets = {"drift", "poison"}
    poison_targets = {"poison"}
    benign_tags = {"stable", "recovery"}

    # ── Rolling accuracy tracking ─────────────────────────────────────────
    correct_window: list[int] = []
    ROLLING_WIN = 50

    def _update_rolling_accuracy(y_pred_val: int, y_true_val: int) -> None:
        correct_window.append(int(y_pred_val == y_true_val))
        if len(correct_window) > ROLLING_WIN:
            correct_window.pop(0)
        result.rolling_accuracy.append(float(np.mean(correct_window)))

    # ═══════════════════════════════════════════════════════════════════
    # Phase 1 — Stable operation (steps 1-400)
    # ═══════════════════════════════════════════════════════════════════
    X1, y1 = make_kpi_data(PHASE1_STEPS, rng, label_noise=0.03)
    phase1_correct = 0
    for i in range(PHASE1_STEPS):
        step = i + 1
        pred = rtp.observe(X1[i], y_true=y1[i])
        pred_val = int(pred.ravel()[0])
        _update_rolling_accuracy(pred_val, y1[i])

        # Accumulate accuracy
        phase1_correct += int(pred_val == y1[i])

        # Per-check-cycle TP/FP/FN classification
        if step % check_interval == 0 and rtp.last_ddd is not None:
            phase_tag = _classify_alert(step)

            # DDD
            ddd_fired = rtp.last_ddd.drift_detected
            if ddd_fired and phase_tag in drift_targets:
                result.ddd_tp += 1
                if ddd_first_alert_step is None:
                    ddd_first_alert_step = step
            elif ddd_fired and phase_tag in benign_tags:
                result.ddd_fp += 1
            elif not ddd_fired and phase_tag in drift_targets:
                result.ddd_fn += 1

            # DPD
            dpd_fired = (rtp.last_dpd is not None and rtp.last_dpd.poisoning_detected)
            if dpd_fired and phase_tag in poison_targets:
                result.dpd_tp += 1
                if dpd_first_alert_step is None:
                    dpd_first_alert_step = step
            elif dpd_fired and phase_tag in benign_tags:
                result.dpd_fp += 1
            elif not dpd_fired and phase_tag in poison_targets:
                result.dpd_fn += 1

            # CDD
            cdd_fired = (rtp.last_cdd is not None and rtp.last_cdd.drift_detected)
            if cdd_fired and phase_tag in drift_targets:
                result.cdd_tp += 1
                if cdd_first_alert_step is None:
                    cdd_first_alert_step = step
            elif cdd_fired and phase_tag in benign_tags:
                result.cdd_fp += 1
            elif not cdd_fired and phase_tag in drift_targets:
                result.cdd_fn += 1

            # CPD
            cpd_fired = (rtp.last_cpd is not None and rtp.last_cpd.poisoning_detected)
            if cpd_fired and phase_tag in poison_targets:
                result.cpd_tp += 1
                if cpd_first_alert_step is None:
                    cpd_first_alert_step = step
            elif cpd_fired and phase_tag in benign_tags:
                result.cpd_fp += 1
            elif not cpd_fired and phase_tag in poison_targets:
                result.cpd_fn += 1

    result.phase1_accuracy = phase1_correct / PHASE1_STEPS

    # ═══════════════════════════════════════════════════════════════════
    # Phase 2 — Concept drift (steps 401-600)
    # RSRP degrades by -15 dBm, latency spikes +20 ms → boundary shift
    # ═══════════════════════════════════════════════════════════════════
    X2, y2 = make_kpi_data(
        PHASE2_STEPS, rng,
        rsrp_shift=-15.0,    # signal strength degrades
        latency_shift=20.0,  # latency budget tighter
        label_noise=0.08,    # more labelling noise under drift
    )
    phase2_correct = 0
    for i in range(PHASE2_STEPS):
        step = PHASE1_STEPS + i + 1
        pred = rtp.observe(X2[i], y_true=y2[i])
        pred_val = int(pred.ravel()[0])
        _update_rolling_accuracy(pred_val, y2[i])
        phase2_correct += int(pred_val == y2[i])

        if step % check_interval == 0 and rtp.last_ddd is not None:
            phase_tag = _classify_alert(step)

            # DDD — per-check-cycle
            ddd_fired = rtp.last_ddd.drift_detected
            if ddd_fired and phase_tag in drift_targets:
                result.ddd_tp += 1
                if ddd_first_alert_step is None:
                    ddd_first_alert_step = step
            elif ddd_fired and phase_tag in benign_tags:
                result.ddd_fp += 1
            elif not ddd_fired and phase_tag in drift_targets:
                result.ddd_fn += 1

            # DPD — per-check-cycle
            dpd_fired = (rtp.last_dpd is not None and rtp.last_dpd.poisoning_detected)
            if dpd_fired and phase_tag in poison_targets:
                result.dpd_tp += 1
                if dpd_first_alert_step is None:
                    dpd_first_alert_step = step
            elif dpd_fired and phase_tag in benign_tags:
                result.dpd_fp += 1
            elif not dpd_fired and phase_tag in poison_targets:
                result.dpd_fn += 1

            # CDD — per-check-cycle
            cdd_fired = (rtp.last_cdd is not None and rtp.last_cdd.drift_detected)
            if cdd_fired and phase_tag in drift_targets:
                result.cdd_tp += 1
                if cdd_first_alert_step is None:
                    cdd_first_alert_step = step
            elif cdd_fired and phase_tag in benign_tags:
                result.cdd_fp += 1
            elif not cdd_fired and phase_tag in drift_targets:
                result.cdd_fn += 1

            # CPD — per-check-cycle
            cpd_fired = (rtp.last_cpd is not None and rtp.last_cpd.poisoning_detected)
            if cpd_fired and phase_tag in poison_targets:
                result.cpd_tp += 1
                if cpd_first_alert_step is None:
                    cpd_first_alert_step = step
            elif cpd_fired and phase_tag in benign_tags:
                result.cpd_fp += 1
            elif not cpd_fired and phase_tag in poison_targets:
                result.cpd_fn += 1

    result.phase2_accuracy = phase2_correct / PHASE2_STEPS

    # ═══════════════════════════════════════════════════════════════════
    # Phase 3a — Subtle poisoning (steps 601-650)
    # 10% contamination, values within 3-sigma of normal distribution
    # ═══════════════════════════════════════════════════════════════════
    X3a, y3a = make_kpi_data(
        PHASE3A_STEPS, rng,
        poison_fraction=0.10,
        poison_mode="subtle",
    )
    for i in range(PHASE3A_STEPS):
        step = PHASE1_STEPS + PHASE2_STEPS + i + 1
        pred = rtp.observe(X3a[i], y_true=y3a[i])
        pred_val = int(pred.ravel()[0])
        _update_rolling_accuracy(pred_val, y3a[i])

        if step % check_interval == 0 and rtp.last_ddd is not None:
            phase_tag = _classify_alert(step)

            # DDD — per-check-cycle
            ddd_fired = rtp.last_ddd.drift_detected
            if ddd_fired and phase_tag in drift_targets:
                result.ddd_tp += 1
                if ddd_first_alert_step is None:
                    ddd_first_alert_step = step
            elif ddd_fired and phase_tag in benign_tags:
                result.ddd_fp += 1
            elif not ddd_fired and phase_tag in drift_targets:
                result.ddd_fn += 1

            # DPD — per-check-cycle
            dpd_fired = (rtp.last_dpd is not None and rtp.last_dpd.poisoning_detected)
            if dpd_fired and phase_tag in poison_targets:
                result.dpd_tp += 1
                if dpd_first_alert_step is None:
                    dpd_first_alert_step = step
            elif dpd_fired and phase_tag in benign_tags:
                result.dpd_fp += 1
            elif not dpd_fired and phase_tag in poison_targets:
                result.dpd_fn += 1

            # CDD — per-check-cycle
            cdd_fired = (rtp.last_cdd is not None and rtp.last_cdd.drift_detected)
            if cdd_fired and phase_tag in drift_targets:
                result.cdd_tp += 1
                if cdd_first_alert_step is None:
                    cdd_first_alert_step = step
            elif cdd_fired and phase_tag in benign_tags:
                result.cdd_fp += 1
            elif not cdd_fired and phase_tag in drift_targets:
                result.cdd_fn += 1

            # CPD — per-check-cycle
            cpd_fired = (rtp.last_cpd is not None and rtp.last_cpd.poisoning_detected)
            if cpd_fired and phase_tag in poison_targets:
                result.cpd_tp += 1
                if cpd_first_alert_step is None:
                    cpd_first_alert_step = step
            elif cpd_fired and phase_tag in benign_tags:
                result.cpd_fp += 1
            elif not cpd_fired and phase_tag in poison_targets:
                result.cpd_fn += 1

    # ═══════════════════════════════════════════════════════════════════
    # Phase 3b — Aggressive poisoning (steps 651-700)
    # Extreme outliers: 30-50x outside normal KPI ranges
    # ═══════════════════════════════════════════════════════════════════
    X3b_clean, y3b_clean = make_kpi_data(45, rng, label_noise=0.03)
    X3b_inject = np.column_stack([
        rng.uniform(-140.0 * 40, -44.0 * 40, 5),   # extreme RSRP
        rng.uniform(-5.0 * 40,   30.0 * 40, 5),     # extreme SINR
        rng.uniform(0.0,         1000.0 * 40, 5),   # extreme throughput
        rng.uniform(1.0,         100.0 * 40, 5),    # extreme latency
    ])
    y3b_inject = rng.integers(0, 2, size=5)
    X3b = np.vstack([X3b_clean, X3b_inject])
    y3b = np.concatenate([y3b_clean, y3b_inject])
    idx3b = rng.permutation(len(X3b))
    X3b, y3b = X3b[idx3b], y3b[idx3b]

    for i in range(PHASE3B_STEPS):
        step = PHASE1_STEPS + PHASE2_STEPS + PHASE3A_STEPS + i + 1
        pred = rtp.observe(X3b[i], y_true=y3b[i])
        pred_val = int(pred.ravel()[0])
        _update_rolling_accuracy(pred_val, y3b[i])

        if step % check_interval == 0 and rtp.last_ddd is not None:
            phase_tag = _classify_alert(step)

            # DDD — per-check-cycle
            ddd_fired = rtp.last_ddd.drift_detected
            if ddd_fired and phase_tag in drift_targets:
                result.ddd_tp += 1
                if ddd_first_alert_step is None:
                    ddd_first_alert_step = step
            elif ddd_fired and phase_tag in benign_tags:
                result.ddd_fp += 1
            elif not ddd_fired and phase_tag in drift_targets:
                result.ddd_fn += 1

            # DPD — per-check-cycle
            dpd_fired = (rtp.last_dpd is not None and rtp.last_dpd.poisoning_detected)
            if dpd_fired and phase_tag in poison_targets:
                result.dpd_tp += 1
                if dpd_first_alert_step is None:
                    dpd_first_alert_step = step
            elif dpd_fired and phase_tag in benign_tags:
                result.dpd_fp += 1
            elif not dpd_fired and phase_tag in poison_targets:
                result.dpd_fn += 1

            # CDD — per-check-cycle
            cdd_fired = (rtp.last_cdd is not None and rtp.last_cdd.drift_detected)
            if cdd_fired and phase_tag in drift_targets:
                result.cdd_tp += 1
                if cdd_first_alert_step is None:
                    cdd_first_alert_step = step
            elif cdd_fired and phase_tag in benign_tags:
                result.cdd_fp += 1
            elif not cdd_fired and phase_tag in drift_targets:
                result.cdd_fn += 1

            # CPD — per-check-cycle
            cpd_fired = (rtp.last_cpd is not None and rtp.last_cpd.poisoning_detected)
            if cpd_fired and phase_tag in poison_targets:
                result.cpd_tp += 1
                if cpd_first_alert_step is None:
                    cpd_first_alert_step = step
            elif cpd_fired and phase_tag in benign_tags:
                result.cpd_fp += 1
            elif not cpd_fired and phase_tag in poison_targets:
                result.cpd_fn += 1

    # ═══════════════════════════════════════════════════════════════════
    # Phase 4 — Recovery (steps 701-900)
    # ═══════════════════════════════════════════════════════════════════
    X4, y4 = make_kpi_data(PHASE4_STEPS, rng, label_noise=0.03)
    phase4_correct = 0
    for i in range(PHASE4_STEPS):
        step = PHASE1_STEPS + PHASE2_STEPS + PHASE3A_STEPS + PHASE3B_STEPS + i + 1
        pred = rtp.observe(X4[i], y_true=y4[i])
        pred_val = int(pred.ravel()[0])
        _update_rolling_accuracy(pred_val, y4[i])
        phase4_correct += int(pred_val == y4[i])

        # Any poison/drift alert in Phase 4 is a false positive
        if step % check_interval == 0:
            if rtp.last_ddd is not None and rtp.last_ddd.drift_detected:
                result.ddd_fp += 1
            if rtp.last_dpd is not None and rtp.last_dpd.poisoning_detected:
                result.dpd_fp += 1
            if rtp.last_cdd is not None and rtp.last_cdd.drift_detected:
                result.cdd_fp += 1
            if rtp.last_cpd is not None and rtp.last_cpd.poisoning_detected:
                result.cpd_fp += 1

    result.phase4_accuracy = phase4_correct / PHASE4_STEPS

    # ── Detection latency (steps from phase start to first alert) ─────────
    if ddd_first_alert_step is not None:
        result.ddd_latency = float(ddd_first_alert_step - DRIFT_START)
    if dpd_first_alert_step is not None:
        result.dpd_latency = float(dpd_first_alert_step - POISON_START)
    if cdd_first_alert_step is not None:
        result.cdd_latency = float(cdd_first_alert_step - DRIFT_START)
    if cpd_first_alert_step is not None:
        result.cpd_latency = float(cpd_first_alert_step - POISON_START)

    # ── Summary counts ────────────────────────────────────────────────────
    result.total_mtout_signals   = len(received_signals)
    result.total_security_alerts = len(security_alerts)

    if verbose:
        log.info(
            "Seed %3d  ci=%3d  P1=%.3f P2=%.3f P4=%.3f  "
            "DDD_P=%.2f DDD_R=%.2f  CPD_P=%.2f CPD_R=%.2f  "
            "MToUT=%d  Deploys=%d",
            seed, check_interval,
            result.phase1_accuracy, result.phase2_accuracy, result.phase4_accuracy,
            result.ddd_precision, result.ddd_recall,
            result.cpd_precision, result.cpd_recall,
            result.total_mtout_signals, result.atm_deployments,
        )

    return result


# ===========================================================================
# Section 5 — Multi-seed runner
# ===========================================================================

def run_multi_seed(
    n_seeds: int = N_SEEDS,
    check_interval: int = 50,
) -> list[SeedResult]:
    """
    Run the full simulation across multiple seeds and return all results.

    Parameters
    ----------
    n_seeds : int
        Number of random seeds to evaluate.
    check_interval : int
        Detector check interval to use for all seeds.

    Returns
    -------
    list[SeedResult]
    """
    log.info("Running %d seeds with check_interval=%d…", n_seeds, check_interval)
    results = []
    for s in range(n_seeds):
        seed_result = run_single_seed(
            seed=s,
            check_interval=check_interval,
            verbose=(s % 10 == 0),
        )
        results.append(seed_result)
        # Print progress dot every seed
        print(f"\r  Seed {s+1:2d}/{n_seeds}", end="", flush=True)
    print()  # newline after progress dots
    return results


# ===========================================================================
# Section 6 — Aggregation & CSV export
# ===========================================================================

def results_to_dataframe(results: list[SeedResult]) -> pd.DataFrame:
    """Convert SeedResult objects to a flat DataFrame for CSV export."""
    rows = []
    for r in results:
        row = {
            "seed":            r.seed,
            "check_interval":  r.check_interval,
            "ddd_precision":   r.ddd_precision,
            "ddd_recall":      r.ddd_recall,
            "dpd_precision":   r.dpd_precision,
            "dpd_recall":      r.dpd_recall,
            "cdd_precision":   r.cdd_precision,
            "cdd_recall":      r.cdd_recall,
            "cpd_precision":   r.cpd_precision,
            "cpd_recall":      r.cpd_recall,
            "ddd_latency":     r.ddd_latency if r.ddd_latency is not None else float("nan"),
            "dpd_latency":     r.dpd_latency if r.dpd_latency is not None else float("nan"),
            "cdd_latency":     r.cdd_latency if r.cdd_latency is not None else float("nan"),
            "cpd_latency":     r.cpd_latency if r.cpd_latency is not None else float("nan"),
            "phase1_accuracy": r.phase1_accuracy,
            "phase2_accuracy": r.phase2_accuracy,
            "phase4_accuracy": r.phase4_accuracy,
            "ndt_pseudo_mean": float(np.mean(r.ndt_pseudo_scores)) if r.ndt_pseudo_scores else float("nan"),
            "ndt_gt_mean":     float(np.mean(r.ndt_gt_scores))     if r.ndt_gt_scores     else float("nan"),
            "atm_training_time_mean": float(np.mean(r.atm_training_times)) if r.atm_training_times else float("nan"),
            "total_mtout":     r.total_mtout_signals,
            "total_security":  r.total_security_alerts,
            "atm_deployments": r.atm_deployments,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ± std for all numeric columns (grouped by check_interval)."""
    numeric_cols = [c for c in df.columns if c not in ("seed",)]
    grouped = df[numeric_cols].groupby("check_interval")
    mean_df = grouped.mean().add_suffix("_mean")
    std_df  = grouped.std().add_suffix("_std")
    summary = pd.concat([mean_df, std_df], axis=1).sort_index(axis=1)
    return summary.reset_index()


# ===========================================================================
# Section 7 — Figures
# ===========================================================================

_FIG_DPI = 300
_DETECTOR_COLORS = {
    "DDD": "#2196F3",   # blue
    "DPD": "#F44336",   # red
    "CDD": "#4CAF50",   # green
    "CPD": "#FF9800",   # orange
}


def fig_boxplot_precision_recall(
    df: pd.DataFrame,
    save_path: str,
) -> None:
    """
    Figure (a): Box plots of precision and recall for each detector across seeds.
    """
    detectors = ["DDD", "DPD", "CDD", "CPD"]
    metrics   = ["precision", "recall"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Detector Precision and Recall across 30 Seeds\n(UERANSIM-aligned 5G NR KPI Dataset, check_interval=50)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    for ax, metric in zip(axes, metrics):
        data_per_detector = [
            df[f"{det.lower()}_{metric}"].dropna().values
            for det in detectors
        ]
        # 'tick_labels' is the new name in matplotlib >= 3.9; fall back to
        # 'labels' on older versions to stay broadly compatible.
        import matplotlib
        _mpl_ver = tuple(int(x) for x in matplotlib.__version__.split(".")[:2])
        _bp_label_kwarg = "tick_labels" if _mpl_ver >= (3, 9) else "labels"
        bp = ax.boxplot(
            data_per_detector,
            **{_bp_label_kwarg: detectors},
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker="o", markersize=4, alpha=0.5),
        )
        for patch, det in zip(bp["boxes"], detectors):
            patch.set_facecolor(_DETECTOR_COLORS[det])
            patch.set_alpha(0.75)

        ax.set_title(metric.capitalize(), fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=10)
        ax.set_xlabel("Detector", fontsize=10)
        ax.set_ylim(-0.05, 1.15)
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.axhline(0.5, color="gray", linestyle=":",  alpha=0.4, linewidth=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", save_path)


def fig_latency_tradeoff(
    sensitivity_df: pd.DataFrame,
    save_path: str,
) -> None:
    """
    Figure (b): Detection latency vs check_interval for all four detectors.
    """
    detectors = ["DDD", "DPD", "CDD", "CPD"]
    col_map = {
        "DDD": "ddd_latency",
        "DPD": "dpd_latency",
        "CDD": "cdd_latency",
        "CPD": "cpd_latency",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Check-Interval Sensitivity Analysis\nDetection Latency vs. Detector Overhead",
        fontsize=13,
        fontweight="bold",
    )

    ax_lat, ax_calls = axes

    # Latency vs check_interval
    for det in detectors:
        col_mean = f"{col_map[det]}_mean"
        col_std  = f"{col_map[det]}_std"
        if col_mean not in sensitivity_df.columns:
            continue
        cis    = sensitivity_df["check_interval"].values
        means  = sensitivity_df[col_mean].values
        stds   = sensitivity_df[col_std].fillna(0).values

        ax_lat.plot(cis, means, marker="o", label=det,
                    color=_DETECTOR_COLORS[det], linewidth=1.8)
        ax_lat.fill_between(cis, means - stds, means + stds,
                             alpha=0.18, color=_DETECTOR_COLORS[det])

    ax_lat.set_xlabel("Check Interval (steps)", fontsize=10)
    ax_lat.set_ylabel("Detection Latency (steps)", fontsize=10)
    ax_lat.set_title("Detection Latency", fontsize=11)
    ax_lat.legend(fontsize=9)
    ax_lat.set_xticks(CHECK_INTERVALS)

    # Overhead proxy: number of MToUT signals / steps
    # Approximate: detector calls per simulation = TOTAL_STEPS / check_interval
    overhead = [TOTAL_STEPS / ci for ci in CHECK_INTERVALS]
    ax_calls.bar(
        [str(ci) for ci in CHECK_INTERVALS],
        overhead,
        color="#607D8B",
        alpha=0.75,
        edgecolor="black",
        linewidth=0.8,
    )
    ax_calls.set_xlabel("Check Interval (steps)", fontsize=10)
    ax_calls.set_ylabel("Detector Invocations per Run", fontsize=10)
    ax_calls.set_title("Computational Overhead Proxy", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", save_path)


def fig_ndt_comparison(
    df: pd.DataFrame,
    save_path: str,
) -> None:
    """
    Figure (c): NDT pseudo-label vs ground-truth score comparison bar chart.
    """
    # Collect per-seed NDT scores — filter seeds that had at least one ATM deployment
    pseudo_scores = df["ndt_pseudo_mean"].dropna().values
    gt_scores     = df["ndt_gt_mean"].dropna().values

    if len(pseudo_scores) == 0 or len(gt_scores) == 0:
        log.warning("NDT comparison: no data available. Skipping figure.")
        return

    # Align on seeds that have both scores
    valid_mask = ~(np.isnan(pseudo_scores) | np.isnan(gt_scores))
    pseudo_scores = pseudo_scores[valid_mask]
    gt_scores     = gt_scores[valid_mask]
    seeds_with_data = np.where(valid_mask)[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "NDT Validation: Pseudo-Label vs Ground-Truth Scores\n"
        "(Self-referential bias analysis across 30 seeds)",
        fontsize=13,
        fontweight="bold",
    )

    # Left: bar chart per seed
    ax_bar = axes[0]
    x = np.arange(len(pseudo_scores))
    width = 0.38
    ax_bar.bar(x - width/2, pseudo_scores, width, label="Pseudo-label (LOB)",
               color="#2196F3", alpha=0.78, edgecolor="black", linewidth=0.6)
    ax_bar.bar(x + width/2, gt_scores,     width, label="Ground truth",
               color="#4CAF50", alpha=0.78, edgecolor="black", linewidth=0.6)
    ax_bar.axhline(0.65, color="red", linestyle="--", linewidth=1.0,
                   label="NDT floor (0.65)")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"S{seeds_with_data[i]}" for i in x],
                            fontsize=7, rotation=45)
    ax_bar.set_xlabel("Seed", fontsize=10)
    ax_bar.set_ylabel("NDT Score (Accuracy)", fontsize=10)
    ax_bar.set_title("Per-Seed NDT Scores", fontsize=11)
    ax_bar.set_ylim(0, 1.1)
    ax_bar.legend(fontsize=9)

    # Right: scatter plot pseudo vs ground truth + bias diagonal
    ax_sc = axes[1]
    ax_sc.scatter(pseudo_scores, gt_scores, color="#FF9800", edgecolors="black",
                  linewidths=0.6, s=55, alpha=0.85, zorder=3)
    _lo = min(pseudo_scores.min(), gt_scores.min()) - 0.05
    _hi = max(pseudo_scores.max(), gt_scores.max()) + 0.05
    ax_sc.plot([_lo, _hi], [_lo, _hi], "k--", linewidth=1.0, label="Perfect agreement")
    ax_sc.axhline(0.65, color="red", linestyle=":", linewidth=0.9, alpha=0.7)
    ax_sc.axvline(0.65, color="blue", linestyle=":", linewidth=0.9, alpha=0.7)

    # Annotate bias direction
    bias = float(np.mean(pseudo_scores - gt_scores))
    ax_sc.set_title(
        f"Pseudo vs Ground-Truth\nMean bias: {bias:+.4f}", fontsize=11
    )
    ax_sc.set_xlabel("Pseudo-label score (LOB)", fontsize=10)
    ax_sc.set_ylabel("Ground-truth score", fontsize=10)
    ax_sc.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", save_path)


def fig_accuracy_band(
    all_results: list[SeedResult],
    save_path: str,
) -> None:
    """
    Figure (d): Rolling accuracy across 30 seeds with 95% confidence band.

    Each seed contributes a rolling_accuracy trace of length TOTAL_STEPS.
    We compute the mean and +/- 1.96*std across seeds at each step.
    """
    # Align traces — pad shorter traces with their last value
    traces = [r.rolling_accuracy for r in all_results]
    max_len = max(len(t) for t in traces)
    padded = np.array([
        t + [t[-1]] * (max_len - len(t)) if t else [0.0] * max_len
        for t in traces
    ])

    mean_acc = padded.mean(axis=0)
    std_acc  = padded.std(axis=0)
    steps    = np.arange(1, max_len + 1)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(steps, mean_acc, color="#2196F3", linewidth=1.8, label="Mean accuracy")
    ax.fill_between(
        steps,
        mean_acc - 1.96 * std_acc,
        mean_acc + 1.96 * std_acc,
        alpha=0.22,
        color="#2196F3",
        label="95% confidence band",
    )

    # Phase boundary markers
    phase_boundaries = [
        (PHASE1_STEPS,                             "End P1",  "#4CAF50"),
        (PHASE1_STEPS + PHASE2_STEPS,              "End P2",  "#FF9800"),
        (PHASE1_STEPS + PHASE2_STEPS + PHASE3A_STEPS, "P3a/3b", "#F44336"),
        (PHASE1_STEPS + PHASE2_STEPS + PHASE3A_STEPS + PHASE3B_STEPS, "End P3", "#9C27B0"),
    ]
    for boundary_step, label, color in phase_boundaries:
        ax.axvline(boundary_step, color=color, linestyle="--", linewidth=1.2, alpha=0.8)
        ax.text(boundary_step + 2, 0.35, label, color=color,
                fontsize=8, rotation=90, va="bottom")

    # Shaded phase regions
    ax.axvspan(0, PHASE1_STEPS, alpha=0.04, color="green",  label="Phase 1 (stable)")
    ax.axvspan(PHASE1_STEPS, PHASE1_STEPS + PHASE2_STEPS, alpha=0.06,
               color="orange", label="Phase 2 (drift)")
    ax.axvspan(PHASE1_STEPS + PHASE2_STEPS,
               PHASE1_STEPS + PHASE2_STEPS + PHASE3A_STEPS + PHASE3B_STEPS,
               alpha=0.06, color="red", label="Phase 3 (poisoning)")
    ax.axvspan(PHASE1_STEPS + PHASE2_STEPS + PHASE3A_STEPS + PHASE3B_STEPS,
               max_len, alpha=0.04, color="purple", label="Phase 4 (recovery)")

    ax.set_xlabel("Simulation Step", fontsize=11)
    ax.set_ylabel("Rolling Accuracy (50-step window)", fontsize=11)
    ax.set_title(
        "Rolling Classification Accuracy across 30 Seeds\n"
        "UERANSIM-aligned 5G NR Handover Prediction — Mean ± 95% CI",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=9, loc="lower left", ncol=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", save_path)


# ===========================================================================
# Section 8 — Console summary table
# ===========================================================================

def print_summary_table(summary_df: pd.DataFrame) -> None:
    """Print a formatted summary table to stdout."""
    print()
    print("=" * 78)
    print("  MULTI-SEED SIMULATION SUMMARY  (mean ± std across 30 seeds)")
    print("=" * 78)

    # Pull out the ci=50 row (main results)
    main = summary_df[summary_df["check_interval"] == 50]
    if main.empty:
        main = summary_df.iloc[[0]]

    def _ms(col: str) -> str:
        """Format mean ± std."""
        mc = f"{col}_mean"
        sc = f"{col}_std"
        if mc not in main.columns:
            return "N/A"
        m = main[mc].values[0]
        s = main[sc].values[0] if sc in main.columns else float("nan")
        if np.isnan(m):
            return " N/A"
        if np.isnan(s):
            return f"{m:.3f}"
        return f"{m:.3f} ± {s:.3f}"

    rows = [
        ("Detector",         "Precision",                          "Recall"),
        ("DDD",              _ms("ddd_precision"),                 _ms("ddd_recall")),
        ("DPD",              _ms("dpd_precision"),                 _ms("dpd_recall")),
        ("CDD",              _ms("cdd_precision"),                 _ms("cdd_recall")),
        ("CPD",              _ms("cpd_precision"),                 _ms("cpd_recall")),
    ]
    header = f"  {'Detector':<8}  {'Precision':>22}  {'Recall':>22}"
    print(header)
    print("  " + "-" * 56)
    for det, prec, rec in rows[1:]:
        print(f"  {det:<8}  {prec:>22}  {rec:>22}")

    print()
    print("  Detection Latency (steps from anomaly start to first alert):")
    print(f"    DDD latency:  {_ms('ddd_latency')}")
    print(f"    DPD latency:  {_ms('dpd_latency')}")
    print(f"    CDD latency:  {_ms('cdd_latency')}")
    print(f"    CPD latency:  {_ms('cpd_latency')}")

    print()
    print("  Phase Accuracy:")
    print(f"    Phase 1 (stable):   {_ms('phase1_accuracy')}")
    print(f"    Phase 2 (drift):    {_ms('phase2_accuracy')}")
    print(f"    Phase 4 (recovery): {_ms('phase4_accuracy')}")

    print()
    print("  NDT Validation (mean score over ATM-triggered cycles):")
    print(f"    Pseudo-label score: {_ms('ndt_pseudo_mean')}")
    print(f"    Ground-truth score: {_ms('ndt_gt_mean')}")

    # Compute bias
    pm_col = "ndt_pseudo_mean_mean"
    gm_col = "ndt_gt_mean_mean"
    if pm_col in main.columns and gm_col in main.columns:
        pm = main[pm_col].values[0]
        gm = main[gm_col].values[0]
        if not (np.isnan(pm) or np.isnan(gm)):
            print(f"    Pseudo-label bias:  {pm - gm:+.4f}  "
                  f"({'over-estimates' if pm > gm else 'under-estimates'} ground truth)")

    print()
    print("  ATM Training:")
    print(f"    Mean training time: {_ms('atm_training_time_mean')} s/cycle")
    print(f"    Deployments:        {_ms('atm_deployments')}")
    print(f"    MToUT signals:      {_ms('total_mtout')}")

    print()
    print("  Check-Interval Sensitivity (mean DDD latency by interval):")
    print(f"    {'Check Interval':<20}  {'DDD Latency':>18}  {'CDD Latency':>18}")
    print("    " + "-" * 58)
    for _, row in summary_df.iterrows():
        ci   = int(row["check_interval"])
        d_l  = row.get("ddd_latency_mean", float("nan"))
        c_l  = row.get("cdd_latency_mean", float("nan"))
        d_s  = row.get("ddd_latency_std",  float("nan"))
        c_s  = row.get("cdd_latency_std",  float("nan"))
        d_str = f"{d_l:.1f} ± {d_s:.1f}" if not (np.isnan(d_l) or np.isnan(d_s)) else "N/A"
        c_str = f"{c_l:.1f} ± {c_s:.1f}" if not (np.isnan(c_l) or np.isnan(c_s)) else "N/A"
        print(f"    {ci:<20}  {d_str:>18}  {c_str:>18}")

    print("=" * 78)
    print()


# ===========================================================================
# Section 9 — Main entry point
# ===========================================================================

def main() -> None:
    """
    Orchestrate the full enhanced simulation pipeline:

    1. Multi-seed analysis (30 seeds, check_interval=50).
    2. Sensitivity analysis (4 check_interval values, 10 seeds each for speed).
    3. Save all CSV results.
    4. Generate all four publication figures.
    5. Print summary table.
    """
    t_start = time.time()
    print()
    print("=" * 78)
    print("  UERANSIM-aligned 5G NR Simulation — Multi-Seed Evaluation")
    print(f"  Seeds: {N_SEEDS}   Total steps/run: {TOTAL_STEPS}")
    print(f"  Data model: 3GPP TS 38.215 (SS-RSRP/SS-SINR) + GTP-U RTT")
    print(f"  Reference: UERANSIM (github.com/aligungr/UERANSIM)")
    print("=" * 78)

    # ── Step 1: Multi-seed run at default check_interval ─────────────────
    print()
    print(f"[1/4] Multi-seed analysis ({N_SEEDS} seeds, check_interval=50)…")
    main_results = run_multi_seed(n_seeds=N_SEEDS, check_interval=50)

    # Save per-seed CSV
    main_df = results_to_dataframe(main_results)
    per_seed_path = os.path.join(RESULTS_DIR, "multi_seed_results.csv")
    main_df.to_csv(per_seed_path, index=False)
    print(f"  Saved: {per_seed_path}")

    # ── Step 2: Sensitivity analysis ─────────────────────────────────────
    print()
    print(f"[2/4] Check-interval sensitivity analysis {CHECK_INTERVALS}…")
    sensitivity_seeds = 10  # fewer seeds for speed; 10 is enough for trend

    sensitivity_results: list[SeedResult] = []
    for ci in CHECK_INTERVALS:
        print(f"  check_interval = {ci:>3} …", end="", flush=True)
        ci_results = run_multi_seed(n_seeds=sensitivity_seeds, check_interval=ci)
        sensitivity_results.extend(ci_results)
        print(f" done ({sensitivity_seeds} seeds)")

    sensitivity_df = results_to_dataframe(sensitivity_results)
    sensitivity_path = os.path.join(RESULTS_DIR, "sensitivity_results.csv")
    sensitivity_df.to_csv(sensitivity_path, index=False)
    print(f"  Saved: {sensitivity_path}")

    # Compute summary statistics across all runs (main + sensitivity)
    all_results_df = pd.concat([main_df, sensitivity_df], ignore_index=True)
    summary_df = compute_summary(all_results_df)
    summary_path = os.path.join(RESULTS_DIR, "summary_statistics.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")

    # ── Step 3: Generate figures ──────────────────────────────────────────
    print()
    print("[3/4] Generating publication figures…")

    # (a) Box plots of precision/recall
    fig_a_path = os.path.join(FIGURES_DIR, "multi_seed_boxplot.png")
    fig_boxplot_precision_recall(main_df, fig_a_path)
    print(f"  (a) {fig_a_path}")

    # (b) Detection latency vs check_interval
    fig_b_path = os.path.join(FIGURES_DIR, "latency_tradeoff.png")
    fig_latency_tradeoff(summary_df, fig_b_path)
    print(f"  (b) {fig_b_path}")

    # (c) NDT comparison
    fig_c_path = os.path.join(FIGURES_DIR, "ndt_comparison.png")
    fig_ndt_comparison(main_df, fig_c_path)
    print(f"  (c) {fig_c_path}")

    # (d) Rolling accuracy confidence band
    fig_d_path = os.path.join(FIGURES_DIR, "accuracy_band.png")
    fig_accuracy_band(main_results, fig_d_path)
    print(f"  (d) {fig_d_path}")

    # ── Step 4: Console summary ───────────────────────────────────────────
    print()
    print("[4/4] Computing and printing summary table…")
    print_summary_table(summary_df)

    elapsed = time.time() - t_start
    print(f"Total wall-clock time: {elapsed:.1f} s  ({elapsed/60:.1f} min)")
    print()
    print("Output files:")
    print(f"  CSV:     {per_seed_path}")
    print(f"  CSV:     {sensitivity_path}")
    print(f"  CSV:     {summary_path}")
    print(f"  Figure:  {fig_a_path}")
    print(f"  Figure:  {fig_b_path}")
    print(f"  Figure:  {fig_c_path}")
    print(f"  Figure:  {fig_d_path}")
    print()


if __name__ == "__main__":
    main()
