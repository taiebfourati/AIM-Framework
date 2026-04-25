"""
rtp.py — Runtime Pipeline (RTP)

The RTP is the central observer of the AIF. It wraps every inference call,
feeds data into the LIB/LOB buffers, runs all four detectors, and fires the
Model Training or Update Trigger (MToUT) when anomalies are detected.

Paper reference: Section IV-C
  "The first AIF pipeline, called the Runtime Pipeline (RTP), supports the
   runtime operations of an AIF. It observes the input data, detects data
   and concept drift and poisoning, and analyses network indicators,
   including KPIs. On that basis, or on the network operator's request,
   it triggers the MToU process."

  "The MToUT may quickly trigger from MLIN to MLIO if the behaviour of
   MLIO yields worse results or in case of MLIN poisoning."

Architecture
------------

        ┌─────────────────────────────────────────────────────┐
        │                      RTP                            │
        │                                                     │
  x ──► │ AIF.predict() ──► LIB / LOB                        │
        │                       │                             │
        │          ┌────────────┼────────────┐                │
        │          ▼            ▼            ▼                │
        │         DDD          DPD          CDD               │
        │          │            │            │                │
        │          └────────────┴──────┬─────┘                │
        │                        CPD ◄─┘                      │
        │                         │                           │
        │                      MToUT ──► ATM (controller)     │
        └─────────────────────────────────────────────────────┘

Key behaviours
--------------
1. Every call to rtp.observe(x, y_true) runs a full inference + detector cycle.
2. Detectors are only queried every `check_interval` steps to avoid overhead.
3. On poisoning detection, the AIF is immediately rolled back to MLIO and
   the security subsystem is notified via the on_security_alert callback.
4. On drift detection, MToUT fires and the on_mtout callback is invoked —
   the controller (ATM) decides which MTP variant to launch.
5. All events are recorded in a structured event log for post-hoc analysis.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

import numpy as np

from aif.aif import AIF, AIFEventType
from aif.buffers import BufferPair
from aif.event_log import EventLog
from aif.golden_corpus import GoldenCorpus
from aif.input_validation import ValidationStats, is_finite_obs
from detectors.cdd import CDDResult, CDD
from detectors.cpd import CPDResult, CPD
from detectors.ddd import DDDResult, DDD
from detectors.dpd import DPDResult, DPD
from detectors.reset import DetectorResetCoordinator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class EventType(Enum):
    DATA_DRIFT        = auto()
    DATA_POISONING    = auto()
    CONCEPT_DRIFT     = auto()
    CONCEPT_POISONING = auto()
    MTOUT_FIRED       = auto()
    ROLLBACK          = auto()
    SECURITY_ALERT    = auto()
    MODEL_UPDATED     = auto()
    # Emitted whenever the CPD shadow classifier is refit. The event's
    # ``details`` carry {source, corpus_hash, n_rows} so auditors can
    # verify which trusted corpus (or unsafe fallback) produced the
    # shadow whose divergence decisions later appear in the log.
    SHADOW_REFIT      = auto()
    # Emitted after ``aif.rollback()`` when the detector-snapshot
    # coordinator successfully rolled all four detectors back to the
    # state captured at the restored slot's deploy time.  Payload:
    # {step, slot_id, source: "rollback"|"refit", detectors_restored}.
    DETECTOR_RESET    = auto()
    # Emitted when a rollback fires but no detector snapshot exists for
    # the slot being re-activated (edge case: rollback before any
    # successful deploy).  Payload carries slot_id, source and reason so
    # downstream can schedule a full re-fit on the next clean window.
    DETECTOR_RESET_FAILED = auto()
    # Emitted when the ingress validator drops a single observation
    # at ``rtp.observe`` because x or y contained NaN, inf, or an
    # extreme magnitude.  Rate-limited to avoid log flood under attack.
    INVALID_OBSERVATION = auto()
    # Emitted when the ingress validator drops rows from a batch push
    # (``BufferPair.push_batch``).  Carries drop counts per reason so
    # auditors can tell a sensor hiccup apart from a targeted flood.
    INVALID_BATCH     = auto()
    # Emitted when DPD's cumulative (EWMA) Mahalanobis tracker exceeds
    # its slow-poisoning threshold WITHOUT the per-batch DPD arm
    # firing on the same window.  Targets the "below-threshold slow
    # drip" attack: each batch stays just under per-batch cutoffs, but
    # the cumulative effect still shifts the distribution.  Payload:
    # ``{step, ewma, threshold, n_batches_above}``.
    SLOW_POISONING_SUSPECTED = auto()
    # ── AIF two-phase commit notify_model_updated outcomes ──────────
    # Emitted when the AIF's prepare phase failed: at least one
    # subscriber refused or raised, candidate slot stays STANDBY, no
    # commit attempted.  Payload mirrors the AIF's emit dict (offending
    # subscriber index, reason, etc.).  Routed from AIFEventType via
    # the RTP's event-callback subscription so a single audit query
    # over rtp.event_log surfaces every model-update outcome.
    MODEL_NOTIFY_ABORTED = auto()
    # Emitted when prepare succeeded but the commit phase failed at
    # subscriber index k > 0; the AIF auto-rolls back so the OLD model
    # remains active.  Carries committed_subscribers / uncommitted_
    # subscribers so an operator can see the partial-commit blast
    # radius before manual intervention.
    MODEL_NOTIFY_PARTIAL = auto()
    # Emitted when prepare AND commit succeeded for every registered
    # subscriber and the new slot is now ACTIVE.  This is the success
    # path; counted in event_summary so deployments can be tracked.
    MODEL_NOTIFY_OK = auto()
    # Emitted when a candidate slot is rejected (NDT failure, manual
    # operator decision, etc.) and transitions to the terminal FAILED
    # state.  The slot will never be promoted; AIMP will replace it on
    # the next attempt.  Payload includes slot_id and rejection reason.
    SLOT_FAILED = auto()
    # ── notify_model_updated detector-baseline operations ────────────
    # Emitted once per detector that has its reference baseline refit
    # during ``rtp.notify_model_updated``.  Distinct from SHADOW_REFIT
    # (which is CPD-specific): REFERENCE_REFIT covers DDD and DPD
    # (and any future detector with a refittable baseline).  Payload:
    # ``{detector, n_rows, source: "live_tail"|"caller_batch"|"golden",
    #   step}`` — auditors trace which population each baseline tracks.
    REFERENCE_REFIT = auto()
    # Emitted after the LOB tail re-stamp in ``notify_model_updated``.
    # The LOB carries the OLD model's predictions on the most recent
    # observations; without re-stamping, the next CPD.check compares
    # the freshly-fit shadow against stale outputs and false-fires.
    # Event payload {n_rows_restamped, step} lets auditors tell a
    # successful self-heal from a deferred-CPD-fire path apart.
    LOB_RESTAMPED = auto()
    # Emitted whenever the cooldown gate suppresses a would-be MToUT
    # signal (``steps_since_last < mtout_cooldown_steps``).  Carries
    # the suppressed reasons so an investigator can tell "the system
    # detected an anomaly but throttled the response" apart from "the
    # system saw nothing", which previously looked identical from the
    # event log.  Payload: ``{step, reasons, steps_since_last,
    # cooldown_steps}``.
    MTOUT_SUPPRESSED = auto()


@dataclass
class RTPEvent:
    """A single timestamped event emitted by the RTP."""
    event_type: EventType
    step: int
    timestamp: float = field(default_factory=time.time)
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"[step={self.step}] {self.event_type.name} — {self.details}"
        )


# ---------------------------------------------------------------------------
# MToUT result — what the trigger sends to the controller (ATM)
# ---------------------------------------------------------------------------

class TriggerReason(Enum):
    DATA_DRIFT        = auto()
    CONCEPT_DRIFT     = auto()
    DATA_POISONING    = auto()
    CONCEPT_POISONING = auto()
    OPERATOR_REQUEST  = auto()


@dataclass
class MToUTSignal:
    """
    Signal fired by MToUT toward the ATM (controller).

    The controller uses this to decide:
      - Whether to retrain (fine-tune, retrain, or model change)
      - Which MTP variant to use (MTP-L, MTP-C, MTP-E)
    """
    reasons: list[TriggerReason]
    step: int
    timestamp: float = field(default_factory=time.time)
    # Detector snapshots for the controller to inspect
    ddd_result: Optional[DDDResult] = None
    dpd_result: Optional[DPDResult] = None
    cdd_result: Optional[CDDResult] = None
    cpd_result: Optional[CPDResult] = None
    # KPI context from the management plane
    kpi_context: dict = field(default_factory=dict)
    # True when the DATA_POISONING reason (if present) was raised
    # solely by the cumulative/slow-poisoning arm and NOT by the
    # per-batch DPD detectors.  Slow-poisoning is a patience-limited
    # patient attack — it warrants urgent investigation but does not
    # justify the CRITICAL auto-rollback semantics of per-batch DPD.
    # The severity clamp in :meth:`severity` uses this flag.
    slow_poisoning_only: bool = False

    def severity(self) -> str:
        """
        A simple severity rating based on which detectors fired.

        CRITICAL  — poisoning confirmed (immediate rollback needed)
        HIGH      — both drift types detected
        MEDIUM    — one drift type detected (or slow-poisoning only)
        LOW       — single detector, operator-initiated
        """
        if TriggerReason.CONCEPT_POISONING in self.reasons:
            return "CRITICAL"
        if TriggerReason.DATA_POISONING in self.reasons:
            # Slow-poisoning is cumulative, not an immediate concern
            # — clamp to MEDIUM.  The per-batch DPD path (immediate
            # injection) stays at CRITICAL.
            if self.slow_poisoning_only:
                return "MEDIUM"
            return "CRITICAL"
        if TriggerReason.DATA_DRIFT in self.reasons and \
           TriggerReason.CONCEPT_DRIFT in self.reasons:
            return "HIGH"
        if self.reasons:
            return "MEDIUM"
        return "LOW"

    def __str__(self) -> str:
        return (
            f"MToUTSignal(step={self.step}, severity={self.severity()}, "
            f"reasons={[r.name for r in self.reasons]})"
        )


# ---------------------------------------------------------------------------
# RTP configuration
# ---------------------------------------------------------------------------

@dataclass
class RTPConfig:
    """
    All tunable parameters for the RTP and its detectors.
    Pass a customised instance to RTP() to override defaults.
    """
    # Buffer sizes
    buffer_maxlen: int = 2000

    # How often to run the detector battery (every N inference steps).
    # Lower = more reactive but higher CPU overhead.
    check_interval: int = 50

    # DDD settings
    ddd_reference_size: int = 300
    ddd_recent_size: int = 100
    ddd_ks_alpha: float = 0.05
    ddd_min_drifted_features: int = 1
    ddd_mmd_threshold: float = 0.05
    ddd_use_mmd: bool = True

    # DPD settings
    dpd_reference_size: int = 300
    dpd_recent_size: int = 50
    dpd_contamination: float = 0.02
    dpd_contamination_threshold: float = 0.10
    dpd_mahal_threshold: float = 4.0
    # Soft-rule: require N samples past mahal_threshold to trigger
    # (suppresses single-4σ noise hits on clean gaussian traffic).
    dpd_min_mahal_outliers: int = 3
    # Hard-rule: any single sample past this triggers immediately,
    # unless the fraction of recent samples that cleared it exceeds
    # ``dpd_mahal_hard_max_fraction`` — that indicates a population-wide
    # distribution shift (drift), not a sparse poisoning injection.
    dpd_mahal_hard_threshold: float = 8.0
    dpd_mahal_hard_max_fraction: float = 0.30
    # Cumulative drift / slow-poisoning arm — EWMA of per-batch mahal_max.
    # Defaults (α=0.05 ≈ 20-batch memory, threshold=7.0 between the 4σ
    # soft and 8σ hard per-batch cutoffs) close the "stay just-below the
    # per-batch threshold forever" attack surface: a sustained 3.9σ
    # every batch grows the EWMA toward 3.9, which is well below 7.0,
    # so the attacker cannot both evade the per-batch arm AND keep the
    # EWMA silent.
    dpd_slow_poisoning_alpha: float = 0.05
    dpd_slow_poisoning_threshold: float = 7.0

    # CDD settings
    cdd_task: str = "classifier"          # "classifier" or "regressor"
    cdd_reference_window: int = 200
    cdd_recent_window: int = 50
    cdd_perf_drop_threshold: float = 0.10
    cdd_ph_delta: float = 0.005
    # Page-Hinkley detection threshold — tuned so that a sustained
    # loss=1 stream (classifier always wrong, regressor always off by 1)
    # trips the detector within one ``check_interval`` of 50 samples
    # once the baseline has been frozen by warmup.  With ``λ=50``,
    # cumulative deviation reaches 49.4 at step 50 (1 - x_mean - δ per
    # step, with x_mean pinned near 0.02 for a clean reference corpus)
    # — literally one sample short of λ.  A default of 40 restores the
    # ``detection ≤ check_interval`` property that the integration
    # pipeline relies on without making the detector noticeably more
    # sensitive to noise on clean streams (the false-alarm rate remains
    # negligible for sub-ε drifts).
    cdd_ph_lambda: float = 40.0

    # CPD settings
    cpd_reference_size: int = 300
    cpd_recent_size: int = 100
    cpd_shadow_threshold: float = 0.25
    cpd_output_ks_alpha: float = 0.01
    cpd_corr_threshold: float = 0.40
    # Fisher-z standardized |Δr| threshold — gate against sampling-noise
    # false positives on small recent windows. p ≈ 6·10⁻⁵ per feature at 4.0.
    cpd_corr_z_threshold: float = 4.0

    # Cooldown: minimum steps between two consecutive MToUT fires
    # (avoids flooding the ATM during a single sustained drift event)
    mtout_cooldown_steps: int = 200

    # ── Detector reset on rollback ─────────────────────────────────────
    # When True, disables the per-slot snapshot/restore machinery that
    # keeps the detectors in sync with the currently ACTIVE model. This
    # flag exists so a Tier-2 regression test can pin the OLD broken
    # behaviour in place and assert that CPD re-fires within a few
    # check intervals after a poisoning-induced rollback. Production
    # deployments MUST keep this at its default of False; disabling it
    # reintroduces the forward-biased-detectors loop observed in
    # ``dashboard_live.log`` (79 rollbacks in 30 s).
    disable_detector_reset: bool = False
    # Number of detector-state snapshots the coordinator retains. Covers
    # the deploy→rollback chain without unbounded memory growth on
    # long-running processes. 3 is enough for the "deploy → poison →
    # rollback → deploy → poison → rollback → deploy" sequence.
    detector_reset_cache_size: int = 3


# ---------------------------------------------------------------------------
# RTP
# ---------------------------------------------------------------------------

class RTP:
    """
    Runtime Pipeline — the main observer component.

    Parameters
    ----------
    aif : AIF
        The AI Function being monitored.
    config : RTPConfig, optional
        Detector and buffer configuration. Uses defaults if not provided.
    on_mtout : Callable[[MToUTSignal], None], optional
        Callback invoked when MToUT fires. The controller (ATM) registers
        here to receive training requests.
    on_security_alert : Callable[[RTPEvent], None], optional
        Callback invoked when poisoning is detected. The network security
        subsystem registers here.

    Example
    -------
    >>> rtp = RTP(aif, config=RTPConfig(cdd_task="classifier"))
    >>> rtp.set_reference(X_ref, y_ref, lob_ref)
    >>>
    >>> for x, y_true in stream:
    ...     result = rtp.observe(x, y_true=y_true)
    """

    def __init__(
        self,
        aif: AIF,
        config: Optional[RTPConfig] = None,
        on_mtout: Optional[Callable[[MToUTSignal], None]] = None,
        on_security_alert: Optional[Callable[[RTPEvent], None]] = None,
        golden_corpus: Optional[GoldenCorpus] = None,
    ) -> None:
        self.aif = aif
        self.cfg = config or RTPConfig()
        # Optional operator-curated clean corpus used to refit the CPD
        # shadow. When ``None`` (the default) the RTP falls back to the
        # legacy buffer-derived refit — which is attacker-influenceable.
        # A one-shot runtime warning is emitted the first time the CPD
        # refit path is traversed without a golden corpus in place.
        self.golden_corpus: Optional[GoldenCorpus] = golden_corpus
        self._warned_missing_golden_corpus: bool = False

        # Buffers
        self.buffers = BufferPair(maxlen=self.cfg.buffer_maxlen)

        # Detectors
        self.ddd = DDD(
            reference_size=self.cfg.ddd_reference_size,
            recent_size=self.cfg.ddd_recent_size,
            ks_alpha=self.cfg.ddd_ks_alpha,
            min_drifted_features=self.cfg.ddd_min_drifted_features,
            mmd_threshold=self.cfg.ddd_mmd_threshold,
            use_mmd=self.cfg.ddd_use_mmd,
        )
        self.dpd = DPD(
            reference_size=self.cfg.dpd_reference_size,
            recent_size=self.cfg.dpd_recent_size,
            if_contamination=self.cfg.dpd_contamination,
            contamination_threshold=self.cfg.dpd_contamination_threshold,
            mahal_threshold=self.cfg.dpd_mahal_threshold,
            min_mahal_outliers=self.cfg.dpd_min_mahal_outliers,
            mahal_hard_threshold=self.cfg.dpd_mahal_hard_threshold,
            mahal_hard_max_fraction=self.cfg.dpd_mahal_hard_max_fraction,
            slow_poisoning_alpha=self.cfg.dpd_slow_poisoning_alpha,
            slow_poisoning_threshold=self.cfg.dpd_slow_poisoning_threshold,
        )
        self.cdd = CDD(
            task=self.cfg.cdd_task,
            reference_window=self.cfg.cdd_reference_window,
            recent_window=self.cfg.cdd_recent_window,
            perf_drop_threshold=self.cfg.cdd_perf_drop_threshold,
            ph_delta=self.cfg.cdd_ph_delta,
            ph_lambda=self.cfg.cdd_ph_lambda,
        )
        self.cpd = CPD(
            task=self.cfg.cdd_task,
            reference_size=self.cfg.cpd_reference_size,
            recent_size=self.cfg.cpd_recent_size,
            shadow_threshold=self.cfg.cpd_shadow_threshold,
            output_ks_alpha=self.cfg.cpd_output_ks_alpha,
            corr_threshold=self.cfg.cpd_corr_threshold,
            corr_z_threshold=self.cfg.cpd_corr_z_threshold,
        )

        # Detector-reset coordinator — captures a per-slot snapshot of
        # every detector's internal state on every successful deploy and
        # restores it after a rollback, so a poisoned MLIN's influence
        # on detector internals does NOT survive the rollback.  Without
        # this, a rollback immediately re-fires CPD/CDD/DPD against the
        # (now-innocent) restored MLIO (see dashboard_live.log — 79
        # rollbacks in 30 s, CPD re-fires 2-5 s after each).
        self._detector_reset = DetectorResetCoordinator(
            ddd=self.ddd, dpd=self.dpd, cdd=self.cdd, cpd=self.cpd,
            cache_size=self.cfg.detector_reset_cache_size,
            event_cb=self._on_detector_reset_event,
        )
        # Tracks whether a rollback occurred but no snapshot was
        # available — the next notify_model_updated must refit from
        # scratch rather than assume the detectors are healthy.
        self._detector_refit_owed: bool = False

        # Callbacks
        self._on_mtout = on_mtout
        self._on_security_alert = on_security_alert

        # State
        self._step: int = 0
        self._last_mtout_step: int = -self.cfg.mtout_cooldown_steps
        # Running count of DPD check batches where the cumulative
        # (EWMA) Mahalanobis tracker exceeded its slow-poisoning
        # threshold.  Surfaced in the SLOW_POISONING_SUSPECTED event
        # payload so auditors can tell "one suspicious window" from
        # "sustained for 50 batches".
        self._slow_poisoning_batches_above: int = 0
        # Tamper-evident append-only event log.  Every ``_log_event``
        # call builds a hash-chained + HMAC-signed entry so a compromised
        # subscriber or buggy path cannot retroactively mutate or delete
        # evidence of a poisoning incident or rollback cascade.  Iteration
        # and indexing still return view objects with the legacy
        # ``event_type`` / ``step`` / ``details`` attributes, so every
        # existing consumer keeps working unchanged; tamper evidence is
        # verified via :meth:`RTP.verify_event_log`.
        self.event_log: EventLog = EventLog()

        # Ingress validation — see ``aif/input_validation.py`` for the
        # decision function.  ``_validation_stats`` accumulates per-reason
        # drop counters over the RTP's lifetime; ``_drops_since_event``
        # implements a cheap rate-limiter so an attacker flooding the
        # ingress with NaNs cannot flood the event_log too (at most one
        # ``INVALID_OBSERVATION`` event is emitted per
        # ``_invalid_event_every_n`` drops).  A matching counter for
        # ``push_batch`` calls is held on the BufferPair via the
        # validation callback wired below.
        self._validation_stats: ValidationStats = ValidationStats()
        self._drops_since_event: int = 0
        self._invalid_event_every_n: int = 100

        # Wire the same ValidationStats + event emitter into the
        # BufferPair so ``push_batch`` validation lands in the same
        # per-reason counter RTP exposes to callers.
        self.buffers.set_validation_hooks(
            stats=self._validation_stats,
            event_callback=self._emit_invalid_batch_event,
        )

        # Subscribe to the AIF's two-phase commit / slot-state events
        # so that MODEL_NOTIFY_OK / MODEL_NOTIFY_PARTIAL / MODEL_NOTIFY_
        # ABORTED / SLOT_FAILED appear in the SAME tamper-evident log
        # as the RTP's own detector and MToUT events.  Without this,
        # an auditor investigating a rollback cascade has to correlate
        # two independent in-memory streams (or worse, only see the
        # RTP-side events).  The mapping table in
        # ``_aif_to_rtp_event_type`` makes the AIF→RTP enum contract
        # explicit and greppable.
        self.aif.set_event_callback(self._on_aif_event)

        # Last detector results (exposed for inspection)
        self.last_ddd: Optional[DDDResult] = None
        self.last_dpd: Optional[DPDResult] = None
        self.last_cdd: Optional[CDDResult] = None
        self.last_cpd: Optional[CPDResult] = None

        logger.info("RTP initialised. Config: check_interval=%d, task=%s",
                    self.cfg.check_interval, self.cfg.cdd_task)

    # ------------------------------------------------------------------
    # Reference initialisation
    # ------------------------------------------------------------------

    def set_reference(
        self,
        X_ref: np.ndarray,
        y_ref: np.ndarray,
        lob_ref: Optional[np.ndarray] = None,
    ) -> "RTP":
        """
        Provide a clean reference dataset to all detectors at once.

        Parameters
        ----------
        X_ref : np.ndarray of shape (n, n_features)
            Clean input samples.
        y_ref : np.ndarray of shape (n,)
            Ground-truth labels or values for those inputs.
        lob_ref : np.ndarray of shape (n,), optional
            MLIN's predictions on X_ref (for CPD shadow cross-check).
            If None, uses y_ref as a proxy (assumes perfect model on reference).
        """
        X_ref = np.atleast_2d(np.asarray(X_ref, dtype=float))
        y_ref = np.asarray(y_ref, dtype=float).ravel()
        lob_ref = y_ref if lob_ref is None else np.asarray(lob_ref, dtype=float).ravel()

        self.ddd.fit_reference(X_ref)
        self.dpd.fit_reference(X_ref)
        # ``set_reference`` is an explicit bootstrap call from the
        # application — the caller has already vouched for (X_ref, y_ref),
        # so flip the authorize_raw gate to silence the security warning
        # while keeping the shadow_source_hash provenance stamped as
        # ``"unsafe_raw"``. If a golden corpus is wired up we ALSO use it
        # right now so that the very first check already benefits from
        # the trusted shadow — otherwise the shadow would stay bound to
        # the bootstrap arrays for the lifetime of the process.
        self.cpd.fit_reference(X_ref, y_ref, lob_ref, authorize_raw=True)
        self._log_event(EventType.SHADOW_REFIT, {
            "step": self._step,
            "source": "bootstrap",
            "corpus_hash": "unsafe_raw",
            "n_rows": int(len(X_ref)),
        })
        if self.golden_corpus is not None and self.golden_corpus.is_ready(
            min_size=min(self.cpd.reference_size, self.cpd.recent_size)
        ):
            snapshot = self.golden_corpus.snapshot(self.cpd.reference_size)
            # Do NOT pass the live MLIN's predictions as the reference
            # LOB here — if the live MLIN has been poisoned, those
            # predictions already encode the attack and would pollute
            # the KS / correlation baselines. Letting the CPD use
            # ``snapshot.y`` (the trusted honest labels) as the
            # reference output distribution is the correct anchor:
            # later checks will catch a poisoned MLIN because its
            # predictions on live inputs will drift away from that
            # honest baseline.
            self.cpd.fit_reference_from_snapshot(snapshot)
            self._log_event(EventType.SHADOW_REFIT, {
                "step": self._step,
                "source": "golden",
                "corpus_hash": snapshot.corpus_hash_hex,
                "n_rows": int(snapshot.n_rows),
            })

        # Warm up CDD's Page-Hinkley baseline with the reference regime.
        # Without this, the first live sample anchors PH's running mean
        # at its own loss, so a stream that starts already-drifted looks
        # flat to PH and drift is never detected.  Feeding the reference
        # corpus first establishes "normal = low loss" so subsequent high
        # losses register as a genuine change-point.  It also primes the
        # window-comparison buffer so CDD can fire from the first full
        # check interval instead of waiting for 250 live samples.
        try:
            self.cdd.warmup(X_ref, y_ref, self.aif.predict)
        except Exception as exc:   # pragma: no cover - defensive
            logger.warning(
                "RTP.set_reference: CDD warmup failed (%s); detector will "
                "start cold and may miss drift on already-drifted streams.",
                exc,
            )

        # Pre-populate buffers with reference data so detectors have
        # context from the start
        self.buffers.push_batch(X_ref, lob_ref)

        # Capture a detector-state snapshot for the INITIAL MLIN slot so
        # that a subsequent rollback (after at least one successful
        # retrain has populated MLIO) finds a clean baseline to restore.
        # Without this, the very first rollback would always hit the
        # DETECTOR_RESET_FAILED branch — safe, but wasteful when we
        # already have a clean snapshot right here.  The feature flag
        # path skips this too so the regression test that pins the
        # pre-fix broken behaviour can run.
        if not self.cfg.disable_detector_reset:
            initial_slot_id = self.aif.mlin.slot_id
            if initial_slot_id is not None:
                self._detector_reset.capture(int(initial_slot_id))

        logger.info(
            "RTP: reference set — %d samples, %d features.",
            len(X_ref), X_ref.shape[1],
        )
        return self

    # ------------------------------------------------------------------
    # Main observe loop — called once per inference step
    # ------------------------------------------------------------------

    def observe(
        self,
        x: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        kpi_context: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Run one full RTP cycle:
          1. AIF inference (DPP → SIB → MLI)
          2. Push (x, prediction) into LIB / LOB
          3. Update CDD online monitor
          4. Every check_interval steps: run DDD, DPD, CDD.check(), CPD
          5. Fire MToUT if any detector raises an alert

        Parameters
        ----------
        x : np.ndarray
            Raw (unscaled) input features.
        y_true : np.ndarray, optional
            Ground-truth label for this step (used by CDD for accuracy tracking).
            Pass None to use CDD's proxy mode.
        kpi_context : dict, optional
            Network KPI indicators from the Management Plane
            (e.g. {"latency_ms": 12.3, "packet_loss": 0.002}).

        Returns
        -------
        np.ndarray
            AIF prediction for this step.
        """
        kpi_context = kpi_context or {}

        # ── 0. Ingress validation — fail closed, do not crash ────────
        # A single NaN / inf / 1e300 in x or y is enough to corrupt
        # Page-Hinkley, destabilise the KS reference window, and make
        # IsolationForest refit raise or return useless scores.  Drop
        # the observation here so detectors never see it, bump the
        # per-reason counter, and emit (at most every Nth drop) a
        # structured INVALID_OBSERVATION event so operators can spot
        # an attack or an upstream fault without flooding the log.
        # Critical: we do NOT advance self._step on rejection — a
        # poisoned sample should not appear in the stream at all.
        ok, reason = is_finite_obs(x, y_true)
        self._validation_stats.record(reason)
        if not ok:
            self._drops_since_event += 1
            # Rate-limit: emit one event per N drops (default 100).
            # Always emit on the very first drop so a test / operator
            # sees the first anomaly immediately rather than waiting
            # for the window to fill.
            if (
                self._drops_since_event == 1
                or self._drops_since_event % self._invalid_event_every_n == 0
            ):
                self._log_event(EventType.INVALID_OBSERVATION, {
                    "step": self._step,
                    "reason": reason,
                    "dropped_fraction_last_1000":
                        self._validation_stats.dropped_fraction_last_1000(),
                    "total_rejected": self._validation_stats.total_rejected,
                })
            # Early return — step not advanced, buffers untouched,
            # detectors never see the bad sample.  Return an empty
            # prediction-shaped array so the caller's ``preds.append``
            # loop does not crash (they can filter on ``.size == 0``).
            return np.empty((0,), dtype=float)

        self._step += 1

        # ── 1. AIF inference ─────────────────────────────────────────
        prediction = self.aif.predict(x)

        # ── 2. Buffer update ─────────────────────────────────────────
        self.buffers.push(
            x=np.asarray(x, dtype=float).ravel(),
            y=prediction.ravel(),
            metadata=kpi_context,
            y_true=y_true,
        )

        # ── 3. CDD online update (every step, lightweight) ───────────
        self.cdd.update(y_pred=prediction, y_true=y_true)

        # ── 4. Full detector battery (every check_interval steps) ────
        if self._step % self.cfg.check_interval == 0:
            self._run_detectors(kpi_context)

        return prediction

    def force_check(self, kpi_context: Optional[dict] = None) -> MToUTSignal | None:
        """
        Immediately run all detectors regardless of check_interval.
        Useful for operator-initiated checks or after injecting test data.
        Returns the MToUTSignal if one was fired, else None.
        """
        return self._run_detectors(kpi_context or {})

    # ------------------------------------------------------------------
    # Operator-initiated trigger
    # ------------------------------------------------------------------

    def operator_request(self, reason: str = "") -> None:
        """
        Manually fire MToUT (e.g. scheduled retraining or operator override).
        """
        signal = MToUTSignal(
            reasons=[TriggerReason.OPERATOR_REQUEST],
            step=self._step,
            kpi_context={"reason": reason},
        )
        self._fire_mtout(signal)

    # ------------------------------------------------------------------
    # Model update notification (called by ATM after retraining)
    # ------------------------------------------------------------------

    def notify_model_updated(
        self,
        new_estimator,
        X_new_ref: Optional[np.ndarray] = None,
        y_new_ref: Optional[np.ndarray] = None,
    ) -> None:
        """
        Called by the ATM after a successful model update.

        1. Installs the new model into the AIF (MLIN slot).
        2. Resets the CDD Page-Hinkley state.
        3. Optionally re-fits detector references on new data.
        """
        self.aif.update_model(new_estimator)

        # Reset concept drift monitor for the new model's baseline.
        # We use full reset() (PH state + rolling perf buffer) rather
        # than reset_ph() alone — the perf buffer still holds pre-update
        # loss entries that would contaminate the sliding-window check
        # (reference=old-bad-losses vs recent=new-good-losses flips the
        # perf_drop sign and fires spurious "recovery" alarms in the
        # wrong direction).  The new model needs a clean slate.
        self.cdd.reset()

        # After a model update, all four detectors need fresh references
        # reflecting the *new* regime.  The LIB/LOB buffers are typically
        # pre-populated by ``set_reference`` with pre-drift data, so we
        # must ignore those rows when computing the new baseline — only
        # the live (post-reference-preload) observations belong to the
        # post-deploy regime.  ``self._step`` counts observe() calls and
        # therefore equals the number of live samples in LIB/LOB.
        live_count = max(self._step, 0)
        lib = self.buffers.lib
        lob = self.buffers.lob

        # Compute the most-recent slice of LIB that is guaranteed to be
        # live data only.  If the caller supplied a training batch we
        # combine the two sources and favour whichever is larger — but
        # never exceed ``live_count`` rows.  This is crucial: dipping
        # into the pre-populated reference corpus pollutes the post-
        # deploy baseline and causes detectors to false-fire.
        def _live_tail(n_wanted: int) -> int:
            """Rows to slice from the tail, clamped to live samples."""
            if live_count == 0:
                # Pre-deploy path (no observations yet): fall back to
                # everything in LIB — there is nothing else to use.
                return min(n_wanted, len(lib))
            return min(n_wanted, live_count, len(lib))

        # ── DDD / DPD baselines ─────────────────────────────────────
        # Each successful refit emits a REFERENCE_REFIT event so an
        # auditor can answer "which population is each detector
        # currently anchored to?" by walking the event log alone.  The
        # ``source`` field documents the slice provenance ("live_tail"
        # = post-deploy live samples, the canonical path).
        ddd_ref_size = getattr(self.ddd, "reference_size", 500)
        ddd_take = _live_tail(ddd_ref_size)
        if ddd_take > 0:
            self.ddd.fit_reference(lib.get_values(ddd_take))
            self._log_event(EventType.REFERENCE_REFIT, {
                "step": self._step,
                "detector": "DDD",
                "n_rows": int(ddd_take),
                "source": "live_tail",
            })

        dpd_ref_size = getattr(self.dpd, "reference_size", 500)
        dpd_take = _live_tail(dpd_ref_size)
        if dpd_take > 0:
            self.dpd.fit_reference(lib.get_values(dpd_take))
            self._log_event(EventType.REFERENCE_REFIT, {
                "step": self._step,
                "detector": "DPD",
                "n_rows": int(dpd_take),
                "source": "live_tail",
            })

        # ── CPD baseline ────────────────────────────────────────────
        # Security-critical: the CPD shadow is the ONLY detector designed
        # for concept-space attacks. Historically the shadow was refit
        # from LIB/LOB/YGT buffers an adversary can influence, so a
        # patient attacker could tutor it onto the poisoned boundary
        # and silence the detector. The hardened path below prefers
        # an operator-curated ``GoldenCorpus`` whenever one is wired
        # up, falling back to the legacy buffer path only behind an
        # explicit ``authorize_raw=True`` flag (see detectors/cpd.py).
        #
        # The caller's labelled batch (X_new_ref, y_new_ref) is still
        # accepted as a last-resort fallback because it IS the (X, y) the
        # new model was trained on — i.e. the operator has already
        # vouched for it — but we record it as ``unsafe_caller_batch``
        # in the SHADOW_REFIT event so auditors can see it was not
        # drawn from the signed golden corpus.
        cpd_ref_size = getattr(self.cpd, "reference_size", 300)
        shadow_refit_source: Optional[str] = None
        shadow_refit_hash: Optional[str] = None
        shadow_refit_n: int = 0
        X_slice: Optional[np.ndarray] = None
        y_slice: Optional[np.ndarray] = None

        if self.golden_corpus is not None and self.golden_corpus.is_ready(
            min_size=min(cpd_ref_size, self.cpd.recent_size)
        ):
            # ── Preferred path: trusted corpus commitment ─────────────
            # Critical: do not seed the output-distribution / correlation
            # reference from the NEW estimator's predictions on the
            # snapshot. If the new estimator has been poisoned (the
            # exact attack we're trying to detect) those predictions
            # already encode the attack and neutralise the KS /
            # correlation arms. Pass no ``lob_outputs`` so the CPD uses
            # ``snapshot.y`` (trusted honest labels) as the reference.
            snapshot = self.golden_corpus.snapshot(cpd_ref_size)
            self.cpd.fit_reference_from_snapshot(snapshot)
            shadow_refit_source = "golden"
            shadow_refit_hash = snapshot.corpus_hash_hex
            shadow_refit_n = snapshot.n_rows
            # Use the snapshot for the CDD re-warmup too so the PH
            # long-run mean tracks the trusted reference regime.
            X_slice = snapshot.X
            y_slice = snapshot.y
        else:
            # ── Unsafe fallback: buffer-driven refit ─────────────────
            if self.golden_corpus is None and not self._warned_missing_golden_corpus:
                logger.warning(
                    "RTP.notify_model_updated: no GoldenCorpus wired; CPD "
                    "shadow will be refit from attacker-influenceable LIB/"
                    "YGT buffers. This silences the concept-poisoning "
                    "detector against patient YGT-poisoning attacks. "
                    "Supply ``golden_corpus=`` to RTP(...) in production."
                )
                self._warned_missing_golden_corpus = True
            cpd_take = _live_tail(cpd_ref_size)
            if X_new_ref is not None and y_new_ref is not None:
                X_arr = np.atleast_2d(np.asarray(X_new_ref, dtype=float))
                y_arr = np.asarray(y_new_ref, dtype=float).ravel()
                n = min(cpd_ref_size, X_arr.shape[0])
                X_slice = X_arr[-n:]
                y_slice = y_arr[-n:]
                shadow_refit_source = "unsafe_caller_batch"
            elif cpd_take > 0:
                # Security-auditor HIGH #5 + QA post-ship Risk 1: use the
                # pair's atomic *triple* snapshot so X / LOB-tail / YGT-tail
                # are all read under the same pair-lock hold.  Without the
                # third-buffer leg, a concurrent ``push_batch`` could trim
                # YGT between the X+LOB read and the YGT read, leaving
                # ``np.where(~isnan(y_gt), y_gt, lob)`` mixing
                # ground-truth row i with prediction row i+1 — a quiet
                # one-row shift in the CPD reference concept.
                X_slice, lob_tail, y_gt_tail = self.buffers.snapshot_triple(
                    cpd_take,
                )
                # Re-derive cpd_take from the actual snapshot width so the
                # downstream slicing uses the true row count (the snapshot
                # may be shorter if the buffer was partially full).
                cpd_take = X_slice.shape[0]
                if cpd_take > 0:
                    # snapshot_triple already guarantees y_gt_tail has the
                    # same length as lob_tail; pick GT where present and
                    # fall back to LOB otherwise.  ``unsafe_ygt`` /
                    # ``unsafe_lob`` retains the same audit-event provenance
                    # the security audit / SHADOW_REFIT history depends on.
                    valid = ~np.isnan(y_gt_tail)
                    if bool(valid.any()):
                        y_slice = np.where(valid, y_gt_tail, lob_tail)
                        shadow_refit_source = "unsafe_ygt"
                    else:
                        y_slice = lob_tail
                        shadow_refit_source = "unsafe_lob"
            if X_slice is not None and y_slice is not None and X_slice.shape[0] > 0:
                try:
                    lob_new = np.asarray(
                        new_estimator.predict(X_slice), dtype=float
                    ).ravel()
                except Exception:   # pragma: no cover - defensive
                    lob_new = y_slice
                # Explicit authorize_raw=True — this IS the unsafe path
                # and the operator has accepted the risk by not wiring
                # a GoldenCorpus. The SHADOW_REFIT event logs it so
                # downstream tooling can alert.
                self.cpd.fit_reference(
                    X_slice, y_slice, lob_new, authorize_raw=True,
                )
                shadow_refit_n = X_slice.shape[0]
                shadow_refit_hash = "unsafe_raw"

        # CDD re-warmup uses whichever reference slice the CPD was fit
        # against, so the PH long-run mean tracks the same regime the
        # shadow was calibrated to.
        if X_slice is not None and y_slice is not None and X_slice.shape[0] > 0:
            try:
                self.cdd.warmup(X_slice, y_slice, new_estimator.predict)
            except Exception as exc:   # pragma: no cover - defensive
                logger.warning(
                    "RTP.notify_model_updated: CDD re-warmup failed (%s).",
                    exc,
                )

        # Emit the SHADOW_REFIT event — only when a refit actually
        # happened. Tests and auditors can key off the ``source`` field
        # to distinguish trusted-corpus refits from fallback refits.
        if shadow_refit_source is not None:
            self._log_event(EventType.SHADOW_REFIT, {
                "step": self._step,
                "source": shadow_refit_source,
                "corpus_hash": shadow_refit_hash,
                "n_rows": int(shadow_refit_n),
            })

        # Re-stamp the live tail of LOB with the NEW model's predictions.
        # The LOB was populated incrementally by observe() using whatever
        # MLIN was active at the time, so post-retrain its tail still
        # carries the OLD MLIN's outputs on the most-recent inputs.  On
        # the next CPD.check() that stale tail is compared against the
        # freshly-fitted shadow and drives shadow_div artificially high
        # (~0.8 in practice), which fires CPD spuriously and breaks the
        # self-healing property.  Re-predicting the live rows in place
        # keeps LOB consistent with the deployed model so subsequent
        # checks measure the true shadow-vs-MLIN divergence.
        if live_count > 0 and len(lob) > 0:
            restamp_ok: bool = False
            restamp_n: int = 0
            restamp_err: Optional[str] = None
            try:
                # QA post-ship Risk 5: hold the BufferPair lock across
                # read-X / predict / write-LOB so a concurrent observe()
                # cannot push a NEW (X, y) row between the LIB read and
                # the LOB write.  Without the lock the new row would
                # land at LOB tail, then the in-place loop would over-
                # write it with a prediction generated from a row it was
                # never paired with — silently corrupting the next CPD
                # check.  The pair lock is an RLock, so the inner
                # ``lib.get_values`` (which takes its own per-buffer
                # lock) is a free re-entry.
                with self.buffers._lock:
                    restamp_n = min(live_count, len(lib), len(lob))
                    X_live = lib.get_values(restamp_n)
                    new_live_preds = np.asarray(
                        new_estimator.predict(X_live), dtype=float
                    ).ravel()
                    # In-place update of LOB samples' values.  deque
                    # supports index assignment; each entry is a Sample
                    # dataclass whose ``value`` field is a np.ndarray
                    # matching the original prediction shape.
                    lob_len = len(lob._buf)
                    for i, pred in enumerate(new_live_preds):
                        buf_idx = lob_len - restamp_n + i
                        if 0 <= buf_idx < lob_len:
                            lob._buf[buf_idx].value = np.atleast_1d(
                                np.asarray(pred, dtype=float)
                            )
                logger.info(
                    "RTP: re-stamped %d LOB rows with new MLIN predictions.",
                    restamp_n,
                )
                restamp_ok = True
            except Exception as exc:  # pragma: no cover - defensive
                restamp_err = repr(exc)
                logger.warning(
                    "RTP: LOB re-stamp failed (%s); CPD may transiently "
                    "report stale shadow divergence until the old rows "
                    "roll out of the buffer.",
                    exc,
                )
            # Always emit LOB_RESTAMPED so an auditor can tell a
            # successful self-heal apart from a deferred-CPD-fire path
            # by the ``ok`` field.  Restamp failures are rare but the
            # event is the only way to surface them at the audit layer.
            self._log_event(EventType.LOB_RESTAMPED, {
                "step": self._step,
                "n_rows_restamped": int(restamp_n),
                "ok": bool(restamp_ok),
                "error": restamp_err,
            })

        self._log_event(EventType.MODEL_UPDATED, {
            "estimator": type(new_estimator).__name__,
            "step": self._step,
        })
        logger.info("RTP: model updated at step %d.", self._step)

        # ── Detector-state snapshot for rollback recovery ───────────────
        # Capture AFTER every detector has been re-baselined against the
        # new MLIN above.  The snapshot is keyed by the new MLIN's
        # slot_id so a future rollback (which re-activates the PREVIOUS
        # slot) cannot accidentally pick up THIS slot's post-deploy
        # state.  The previous slot's snapshot was captured during its
        # own notify_model_updated call — when rollback runs we look up
        # that earlier snapshot.
        #
        # The ``disable_detector_reset`` feature flag below exists for a
        # single regression test that documents the pre-fix broken
        # behaviour; production deployments must keep it False.
        if not self.cfg.disable_detector_reset:
            new_slot_id = self.aif.mlin.slot_id
            if new_slot_id is not None:
                self._detector_reset.capture(int(new_slot_id))
            # Successful fresh capture clears any pending refit debt —
            # the coordinator is now once again holding a valid baseline
            # for the active slot.
            self._detector_refit_owed = False

    # ------------------------------------------------------------------
    # Internal — detector battery
    # ------------------------------------------------------------------

    def _emergency_detector_refit(self) -> None:
        """
        Refit DDD/DPD references from the current LIB tail when a prior
        rollback could not restore detector state from any snapshot.

        Background — QA post-ship Gap 2.  When CONCEPT_POISONING fires the
        AIF rolls back to the previous slot and the RTP asks the
        ``DetectorResetCoordinator`` to restore the snapshot captured the
        last time that slot was active.  If no snapshot exists for the
        restored ``slot_id`` (e.g. very first deploy, or the snapshot was
        evicted by ``cache_size``), :attr:`_detector_refit_owed` is set to
        ``True`` and a ``DETECTOR_RESET_FAILED`` event is emitted — but
        until this method existed nothing READ the flag, so the detectors
        kept their now-stale state, CPD continued reporting near-1.0
        shadow_div on every check, and the system death-spiralled into
        consecutive ATM failures (observed in the operator's poison-preset
        dashboard run: 18 back-to-back NDT_FAIL cycles).

        This emergency path refits DDD and DPD from the active live tail
        of LIB so the next ``check()`` measures fresh-vs-fresh divergence
        instead of fresh-vs-stale.  CPD is intentionally NOT refit here —
        the CPD shadow refit is the security-sensitive path
        (``notify_model_updated`` gates it on the GoldenCorpus) and we do
        not want a rollback to silently re-baseline the only concept-
        space detector against attacker-controlled buffer contents.

        Emits ``REFERENCE_REFIT`` with ``source="rollback_emergency_refit"``
        per detector so an auditor can grep the event log to see exactly
        when an emergency refit ran and what it observed.
        """
        if not self._detector_refit_owed:
            return

        lib = self.buffers.lib
        live_count = max(self._step, 0)

        def _live_tail(n_wanted: int) -> int:
            if live_count == 0:
                return min(n_wanted, len(lib))
            return min(n_wanted, live_count, len(lib))

        any_refit = False

        ddd_ref_size = getattr(self.ddd, "reference_size", 500)
        ddd_take = _live_tail(ddd_ref_size)
        if ddd_take > 0:
            try:
                self.ddd.fit_reference(lib.get_values(ddd_take))
                self._log_event(EventType.REFERENCE_REFIT, {
                    "step": self._step,
                    "detector": "DDD",
                    "n_rows": int(ddd_take),
                    "source": "rollback_emergency_refit",
                })
                any_refit = True
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "RTP: emergency DDD refit failed (%s); detector "
                    "remains stale until next notify_model_updated.",
                    exc,
                )

        dpd_ref_size = getattr(self.dpd, "reference_size", 500)
        dpd_take = _live_tail(dpd_ref_size)
        if dpd_take > 0:
            try:
                self.dpd.fit_reference(lib.get_values(dpd_take))
                self._log_event(EventType.REFERENCE_REFIT, {
                    "step": self._step,
                    "detector": "DPD",
                    "n_rows": int(dpd_take),
                    "source": "rollback_emergency_refit",
                })
                any_refit = True
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "RTP: emergency DPD refit failed (%s); detector "
                    "remains stale until next notify_model_updated.",
                    exc,
                )

        # Reset CDD's PH state too — its long-run mean tracks the regime
        # served by the now-demoted MLIN and would otherwise bias the
        # next perf_drop calculation for tens of check intervals.  Skip
        # warmup (we have no trusted predict() here without locking down
        # which slot's estimator should generate the labels).
        try:
            self.cdd.reset()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("RTP: emergency CDD reset failed (%s).", exc)

        # Clear the debt regardless of per-detector success — the next
        # notify_model_updated will fully re-baseline everything.  Leaving
        # the flag set would re-fire this method on every observe() until
        # a new deploy lands, which spams the event log.
        self._detector_refit_owed = False

        if any_refit:
            logger.info(
                "RTP: emergency detector refit completed at step %d "
                "after rollback snapshot miss.",
                self._step,
            )

    def _run_detectors(self, kpi_context: dict) -> MToUTSignal | None:
        """Run all four detectors and decide whether to fire MToUT."""
        # QA post-ship Gap 2: pay any outstanding refit debt BEFORE the
        # detectors run.  If a rollback at step N-1 missed its snapshot,
        # the flag was set to True; without this call the detectors would
        # report stale-vs-fresh divergence on every check and trigger a
        # death-spiral of MToUTs against the (now-correctly-restored)
        # MLIO.  The check is a no-op (single attribute read) when no
        # debt is owed, so the hot path stays cheap.
        self._emergency_detector_refit()

        lib = self.buffers.lib
        lob = self.buffers.lob

        # Run all detectors
        ddd_r = self.ddd.check(lib)
        dpd_r = self.dpd.check(lib)
        cdd_r = self.cdd.check()
        cpd_r = self.cpd.check(lib, lob)

        # Cache latest results
        self.last_ddd = ddd_r
        self.last_dpd = dpd_r
        self.last_cdd = cdd_r
        self.last_cpd = cpd_r

        # ── Poisoning: immediate rollback + security alert ────────────
        if dpd_r.poisoning_detected:
            self._log_event(EventType.DATA_POISONING, {"step": self._step, "detail": dpd_r.message})
            self._security_alert(EventType.DATA_POISONING, dpd_r.message)

        # ── Slow-poisoning (cumulative arm) ──────────────────────────
        # Targets attackers who keep every batch just-below the
        # per-batch DPD thresholds forever.  The DPD's EWMA of
        # ``mahal_max`` climbs even when no single batch fires, so we
        # emit a dedicated event + MToUT contribution distinct from
        # per-batch DATA_POISONING.  No auto-rollback or security
        # alert: the signal is cumulative and warrants investigation,
        # not the immediate "evict the MLIN now" response.
        if getattr(dpd_r, "slow_poisoning_detected", False):
            self._slow_poisoning_batches_above += 1
            self._log_event(EventType.SLOW_POISONING_SUSPECTED, {
                "step": self._step,
                "ewma": float(getattr(dpd_r, "mahal_ewma", 0.0)),
                "threshold": float(getattr(dpd_r, "slow_poisoning_threshold", 0.0)),
                "n_batches_above": int(self._slow_poisoning_batches_above),
            })
        else:
            # Reset the consecutive-above counter as soon as the EWMA
            # falls back below threshold.  ``n_batches_above`` in the
            # event payload therefore always represents the length of
            # the CURRENT sustained-above run, which is the quantity
            # auditors care about ("how long has this been going on?").
            self._slow_poisoning_batches_above = 0

        if cpd_r.poisoning_detected:
            self._log_event(EventType.CONCEPT_POISONING, {"step": self._step, "detail": cpd_r.message})
            self._security_alert(EventType.CONCEPT_POISONING, cpd_r.message)
            # Immediate rollback to MLIO
            rolled_back = self.aif.rollback()
            if rolled_back:
                self._log_event(EventType.ROLLBACK, {
                    "reason": "CONCEPT_POISONING", "step": self._step
                })
                logger.warning("RTP: immediate rollback to MLIO at step %d.", self._step)
                # ── Detector state restoration ─────────────────────────
                # The detectors' internals were shaped while the now-
                # demoted (poisoned) MLIN was serving predictions; left
                # alone they will re-fire within a few check intervals
                # against the restored MLIO (see dashboard_live.log —
                # 79 rollbacks in 30 s).  Restore each detector to the
                # snapshot captured when the now-active slot was last
                # the ACTIVE one.  Skipped entirely when the feature
                # flag is off (the broken-behaviour regression test
                # uses this branch to assert the old failure mode).
                if not self.cfg.disable_detector_reset:
                    restored_slot_id = self.aif.mlin.slot_id
                    ok = False
                    if restored_slot_id is not None:
                        ok = self._detector_reset.restore(
                            int(restored_slot_id), source="rollback",
                        )
                    if not ok:
                        # No snapshot for the restored slot.  The
                        # coordinator has already emitted
                        # DETECTOR_RESET_FAILED.  Flag that the next
                        # _run_detectors call must refit DDD/DPD from
                        # scratch (Gap 2: paid by
                        # ``_emergency_detector_refit`` at the top of
                        # the next ``_run_detectors`` invocation).
                        self._detector_refit_owed = True

                # QA post-ship Risk 2: reset the MToUT cooldown clock so
                # the post-rollback regime can immediately request a new
                # deploy if drift persists.  Without this, the cooldown
                # window that was started by the MToUT that triggered
                # this rollback continues counting AGAINST the (now
                # rolled-back, healthier) MLIO — meaning genuine drift
                # observed against the restored model is silently
                # suppressed for the remainder of the cooldown window
                # (default 50 steps).  Setting the timer
                # ``cooldown_steps + 1`` in the past guarantees the next
                # ``_run_detectors`` cooldown gate evaluates to "ready".
                self._last_mtout_step = (
                    self._step - self.cfg.mtout_cooldown_steps - 1
                )

        # ── Log drift events ─────────────────────────────────────────
        if ddd_r.drift_detected:
            self._log_event(EventType.DATA_DRIFT, {"step": self._step, "detail": ddd_r.message})
        if cdd_r.drift_detected:
            self._log_event(EventType.CONCEPT_DRIFT, {"step": self._step, "detail": cdd_r.message})

        # ── Assemble trigger reasons ──────────────────────────────────
        reasons: list[TriggerReason] = []
        if ddd_r.drift_detected:
            reasons.append(TriggerReason.DATA_DRIFT)
        if dpd_r.poisoning_detected:
            reasons.append(TriggerReason.DATA_POISONING)
        if cdd_r.drift_detected:
            reasons.append(TriggerReason.CONCEPT_DRIFT)
        if cpd_r.poisoning_detected:
            reasons.append(TriggerReason.CONCEPT_POISONING)
        # Slow-poisoning contributes a DATA_POISONING reason too, but
        # we tag the signal so the severity calculation clamps to
        # MEDIUM rather than auto-escalating to CRITICAL — it's a
        # cumulative warning, not an immediate-rollback trigger.
        slow_only = False
        if getattr(dpd_r, "slow_poisoning_detected", False):
            if TriggerReason.DATA_POISONING not in reasons:
                reasons.append(TriggerReason.DATA_POISONING)
                slow_only = not dpd_r.poisoning_detected

        if not reasons:
            return None

        # ── Cooldown gate — avoid flooding ATM ───────────────────────
        steps_since_last = self._step - self._last_mtout_step
        if steps_since_last < self.cfg.mtout_cooldown_steps:
            logger.debug(
                "RTP: MToUT suppressed by cooldown (%d/%d steps).",
                steps_since_last, self.cfg.mtout_cooldown_steps,
            )
            # Emit MTOUT_SUPPRESSED so the audit log distinguishes
            # "detector fired but throttled" from "detector saw nothing".
            # Previously these two states were indistinguishable on the
            # event log, which made post-incident triage of "did we miss
            # a window?" guesswork.  Carry the suppressed reasons so an
            # operator can decide whether to retune cooldown_steps for
            # the affected reason class.
            self._log_event(EventType.MTOUT_SUPPRESSED, {
                "step": self._step,
                "reasons": [r.name for r in reasons],
                "steps_since_last": int(steps_since_last),
                "cooldown_steps": int(self.cfg.mtout_cooldown_steps),
                "slow_poisoning_only": bool(slow_only),
            })
            return None

        # ── Fire MToUT ────────────────────────────────────────────────
        signal = MToUTSignal(
            reasons=reasons,
            step=self._step,
            ddd_result=ddd_r,
            dpd_result=dpd_r,
            cdd_result=cdd_r,
            cpd_result=cpd_r,
            kpi_context=kpi_context,
            slow_poisoning_only=slow_only,
        )
        self._fire_mtout(signal)
        return signal

    def _fire_mtout(self, signal: MToUTSignal) -> None:
        """Record and dispatch an MToUT signal."""
        self._last_mtout_step = self._step
        self._log_event(EventType.MTOUT_FIRED, {
            "severity": signal.severity(),
            "reasons": [r.name for r in signal.reasons],
            "step": self._step,
        })
        logger.warning("RTP: MToUT fired — %s", signal)
        if self._on_mtout:
            self._on_mtout(signal)

    def _security_alert(self, event_type: EventType, detail: str) -> None:
        """Notify the security subsystem of a poisoning event."""
        event = RTPEvent(
            event_type=EventType.SECURITY_ALERT,
            step=self._step,
            details={"type": event_type.name, "detail": detail},
        )
        self._log_event(EventType.SECURITY_ALERT, event.details)
        logger.critical("RTP SECURITY ALERT [%s]: %s", event_type.name, detail)
        if self._on_security_alert:
            self._on_security_alert(event)

    def _log_event(self, event_type: EventType, details: dict) -> None:
        # Delegate to the tamper-evident EventLog.  The legacy
        # ``RTPEvent`` constructor is no longer on the hot path — the
        # view object synthesised by :meth:`EventLog.append` exposes
        # the same ``event_type`` / ``step`` / ``details`` /
        # ``timestamp`` attributes consumers have always relied on.
        self.event_log.append(event_type, details, step=self._step)

    def verify_event_log(self) -> tuple[bool, Optional[str]]:
        """
        Tamper-detection helper.

        Walks the underlying hash chain and verifies every entry's
        hash, prev-link and (when a signing key is configured) HMAC
        signature.  Returns ``(True, None)`` on a clean chain and
        ``(False, <reason>)`` on the first detected inconsistency,
        pointing at the offending entry.

        Typical operator use after an incident:

        >>> ok, err = rtp.verify_event_log()
        >>> if not ok:
        ...     logger.critical("event log tampered with: %s", err)
        """
        return self.event_log.verify()

    def _on_detector_reset_event(self, event_name: str, payload: dict) -> None:
        """
        Callback wired into :class:`DetectorResetCoordinator` so that
        coordinator-level events land in the RTP's shared event log.
        The coordinator passes ``"DETECTOR_RESET"`` on a successful
        restore and ``"DETECTOR_RESET_FAILED"`` when no snapshot was
        available for the requested slot.  The ``step`` field is
        stamped here from the RTP's live step counter so auditors can
        correlate the reset with the MToUT / ROLLBACK events that
        bracket it.
        """
        if event_name == "DETECTOR_RESET":
            event_type = EventType.DETECTOR_RESET
        elif event_name == "DETECTOR_RESET_FAILED":
            event_type = EventType.DETECTOR_RESET_FAILED
        else:   # pragma: no cover - defensive
            logger.warning(
                "RTP: DetectorResetCoordinator emitted unknown event %r.",
                event_name,
            )
            return
        details = dict(payload)
        details.setdefault("step", self._step)
        self._log_event(event_type, details)

    # ------------------------------------------------------------------
    # AIF event-type mapping — keep the AIF→RTP enum table greppable
    # ------------------------------------------------------------------
    _AIF_EVENT_MAP: dict = {
        AIFEventType.MODEL_NOTIFY_OK:      EventType.MODEL_NOTIFY_OK,
        AIFEventType.MODEL_NOTIFY_PARTIAL: EventType.MODEL_NOTIFY_PARTIAL,
        AIFEventType.MODEL_NOTIFY_ABORTED: EventType.MODEL_NOTIFY_ABORTED,
        AIFEventType.SLOT_FAILED:          EventType.SLOT_FAILED,
    }

    def _on_aif_event(self, event_type: AIFEventType, details: dict) -> None:
        """
        Callback wired into :meth:`aif.aif.AIF.set_event_callback`.

        Translates an :class:`AIFEventType` into the matching
        :class:`EventType` and emits via :meth:`_log_event` so that the
        AIF's notify_model_updated / mark_slot_failed events land in
        the RTP's tamper-evident hash-chained log alongside detector
        and MToUT events.  An auditor querying ``rtp.event_log`` after
        an incident sees a complete causal trace without having to
        join two independent in-memory streams.

        Unknown event types (i.e. AIFEventType values added later
        without a corresponding RTP EventType) are logged at WARNING
        but NOT dropped silently — a single line in the operator log
        is enough to surface the gap during the next code review.
        """
        rtp_event_type = self._AIF_EVENT_MAP.get(event_type)
        if rtp_event_type is None:   # pragma: no cover - defensive
            logger.warning(
                "RTP: unmapped AIFEventType %r — extend "
                "RTP._AIF_EVENT_MAP to surface this event in the "
                "tamper-evident log. payload=%r",
                event_type, details,
            )
            return
        payload = dict(details)
        payload.setdefault("step", self._step)
        self._log_event(rtp_event_type, payload)

    def _emit_invalid_batch_event(self, details: dict) -> None:
        """
        Callback wired into :class:`aif.buffers.BufferPair` so that the
        batch-level validator can emit an :attr:`EventType.INVALID_BATCH`
        event through the same event log as every other RTP event.  Kept
        as a dedicated method (rather than exposing ``_log_event``
        directly) so the BufferPair does not gain a reference to the
        EventType enum.
        """
        self._log_event(EventType.INVALID_BATCH, details)

    # ------------------------------------------------------------------
    # Ingress validation — public accessor
    # ------------------------------------------------------------------

    def validation_stats(self) -> ValidationStats:
        """
        Return the running :class:`ValidationStats` counter.

        Callers (tests, dashboards) get a **live** reference to the
        counter — mutating it from the outside will affect subsequent
        ``observe()`` behaviour, so treat the return value as read-only.
        Use ``ValidationStats.as_dict()`` for a point-in-time snapshot.
        """
        return self._validation_stats

    # ------------------------------------------------------------------
    # Status & reporting
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """
        Return a concise status snapshot — useful for logging and the
        Management Plane interface.
        """
        return {
            "step": self._step,
            "buffer_len": len(self.buffers),
            "last_mtout_step": self._last_mtout_step,
            "active_model": type(self.aif.active_estimator).__name__
                            if self.aif.active_estimator else "None",
            "mlio_available": self.aif.mlio.state.name,
            "events_total": len(self.event_log),
            "last_ddd": self.last_ddd.drift_detected if self.last_ddd else None,
            "last_dpd": self.last_dpd.poisoning_detected if self.last_dpd else None,
            "last_cdd": self.last_cdd.drift_detected if self.last_cdd else None,
            "last_cpd": self.last_cpd.poisoning_detected if self.last_cpd else None,
        }

    def event_summary(self) -> dict:
        """Count events by type — useful for thesis result tables."""
        summary: dict[str, int] = {}
        for e in self.event_log:
            summary[e.event_type.name] = summary.get(e.event_type.name, 0) + 1
        return summary

    def __repr__(self) -> str:
        return (
            f"RTP(step={self._step}, "
            f"buffer={len(self.buffers)}/{self.cfg.buffer_maxlen}, "
            f"check_interval={self.cfg.check_interval})"
        )
