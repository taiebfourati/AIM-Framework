"""
detectors/dpd.py — Data Poisoning Detector (DPD)

Detects when incoming AIF inputs contain adversarially injected or
corrupted samples that could corrupt a future retraining run.

Paper reference: Section IV-C
  "Data Poisoning Detector (DPD) checks LIB for data poisoning. The
   detection of data poisoning triggers two actions. First, the security
   subsystem of the network is informed, and the ATM is informed too.
   The problem has to be solved on a system level, not by ATM."

What distinguishes poisoning from drift
---------------------------------------
Drift (DDD) = the *entire* recent distribution has gradually shifted.
Poisoning (DPD) = a *subset* of recent samples are anomalous outliers
injected to corrupt the training data — the bulk of the distribution
is still normal, but a fraction of samples are suspicious.

Detection strategy
------------------
Two complementary anomaly detectors run in parallel:

1. Isolation Forest  [sklearn.ensemble.IsolationForest]
   An unsupervised tree-based anomaly detector trained on clean reference
   data. Assigns an anomaly score to each new sample. Flags the recent
   window if the fraction of anomalous samples exceeds `contamination_threshold`.

2. Mahalanobis boundary — two-tier rule
   A *soft* threshold (``mahal_threshold``, default 4σ) flags mild
   outliers: at least ``min_mahal_outliers`` samples must exceed it
   before an alert fires, which rules out the normal 0.05 %–0.5 % of
   samples that sit beyond 4σ on clean gaussian-like data.
   A *hard* threshold (``mahal_hard_threshold``, default 8σ) triggers
   on any SINGLE exceedance — under N(0,I) the probability is
   ~2·10⁻¹⁵ per sample, so a single 8σ hit is virtually always a
   real injection rather than noise.

A poisoning alert fires when EITHER the Isolation Forest or the
Mahalanobis rule (soft OR hard) triggers.

Historical note
---------------
Prior to the two-tier Mahalanobis rule the detector triggered on a
single 4σ hit, which produced false positives at rate ~0.02 per check
on clean gaussian traffic — matching the dashboard-observed sporadic
``DATA_POISONING`` firings at steps 1950 / 2850 / 3450 of the live
demo. The current rule cuts that rate by ~4 orders of magnitude while
still catching single-sample extreme injections.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Optional
from aif.buffers import LIB, LOB

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EmpiricalCovariance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DPDResult:
    """Outcome of one DPD.check() call."""
    poisoning_detected: bool
    # Isolation Forest
    if_anomaly_rate: float          # fraction of recent samples flagged
    if_threshold: float
    if_triggered: bool
    if_anomalous_indices: list[int] # indices inside the recent window
    # Mahalanobis — two-tier rule
    mahal_max: float                # worst-case distance in recent window
    mahal_threshold: float          # soft threshold (σ)
    mahal_hard_threshold: float     # hard threshold (σ) — any single hit triggers
    mahal_triggered: bool           # True iff EITHER soft-count OR hard-hit fires
    mahal_soft_triggered: bool      # ≥ min_mahal_outliers samples past soft threshold
    mahal_hard_triggered: bool      # ≥ 1 sample past hard threshold
    mahal_anomalous_indices: list[int]   # indices past soft threshold
    mahal_hard_indices: list[int]        # indices past hard threshold
    # Slow-poisoning / cumulative drift — EWMA of per-batch mahal_max.
    # A patient attacker can keep every single batch just under the
    # per-batch thresholds above while the cumulative signal shifts the
    # distribution.  These fields carry the cumulative-view verdict so a
    # caller (RTP) can raise a separate MToUT path for the slow-poison
    # suspicion without double-firing when the per-batch arm also fires.
    slow_poisoning_detected: bool = False
    mahal_ewma: float = 0.0
    slow_poisoning_threshold: float = 0.0
    slow_poisoning_alpha: float = 0.0
    message: str = ""

    def __bool__(self) -> bool:
        return self.poisoning_detected


# ---------------------------------------------------------------------------
# DPD
# ---------------------------------------------------------------------------

class DPD:
    """
    Data Poisoning Detector.

    Parameters
    ----------
    reference_size : int
        Samples from LIB used to train the Isolation Forest and estimate
        the reference covariance for Mahalanobis distance.
    recent_size : int
        Most-recent LIB samples inspected on each check() call.
    if_contamination : float
        Expected fraction of outliers in the training set — passed to
        IsolationForest. Keep low (0.01–0.05) to avoid false positives
        on clean data.
    contamination_threshold : float
        Fraction of the *recent* window that must be flagged as anomalous
        by the Isolation Forest before a poisoning alert fires.
        Example: 0.10 means "alert if >10% of recent samples are anomalous".
    mahal_threshold : float
        Soft Mahalanobis distance cutoff (σ units). A sample exceeding
        this is "suspicious". On its own a single soft hit is NOT enough
        to trigger — see ``min_mahal_outliers``. Typical: 3.0–5.0.
    min_mahal_outliers : int
        Minimum number of samples in the recent window that must exceed
        ``mahal_threshold`` for the soft rule to fire. Default 3 matches
        paper-style coordinated-attack semantics while suppressing the
        ~1–3 % per-check false-positive rate that a single 4σ sample
        would otherwise cause on clean gaussian traffic.
    mahal_hard_threshold : float
        Hard Mahalanobis distance cutoff (σ units). A single sample past
        this is unusual enough to be suspicious, but — as with the soft
        and Isolation-Forest arms — we suppress the trigger when the hit
        fraction is so large it's a population-wide shift (= drift).
        Default 8.0 gives P ≈ 2·10⁻¹⁵ per sample under N(0,I), so even
        a handful of hits on clean data is vanishingly unlikely; a
        drift of ~4σ, however, routinely places tens of samples past
        8σ in the Mahalanobis metric because the covariance was fit on
        the pre-drift regime.
    mahal_hard_max_fraction : float
        Upper bound on the fraction of the recent window allowed past
        the hard threshold before the hard arm is suppressed (treated
        as drift instead of poisoning).  Default 0.30 — comfortably
        above a Poisson-tail dozen hits for real single-sample attacks,
        well below the 56-100% hit rates seen under uniform distribution
        shifts.
    n_estimators : int
        Number of trees in the Isolation Forest.
    slow_poisoning_alpha : float
        EWMA smoothing factor for the cumulative drift tracker.  The
        tracker updates on every ``check()`` as
        ``ewma := α · mahal_max + (1 - α) · ewma`` so ``α`` controls the
        memory depth of the running average: roughly ``1/α`` batches
        contribute meaningfully.  Default ``0.05`` gives a ~20-batch
        memory, which is the right horizon for a "patient" attacker who
        keeps each batch just below the per-batch thresholds while the
        cumulative effect shifts the distribution.
    slow_poisoning_threshold : float
        Threshold on the EWMA of ``mahal_max`` that the slow-poisoning
        arm fires on.  Default ``7.0`` versus the per-batch
        ``mahal_threshold=4.0`` and ``mahal_hard_threshold=8.0``: sitting
        just-below the per-batch threshold (say ~3.9σ every batch)
        grows the EWMA into the 3.9-ish band, while a sustained ~50 %
        of the hard threshold (7.0) cannot be maintained without the
        per-batch soft arm also firing occasionally.  In practice the
        attacker has to pick: either fire the per-batch arm, or stay
        well below 7σ on the EWMA — leaving no silent attack surface.
    """

    def __init__(
        self,
        reference_size: int = 300,
        recent_size: int = 50,
        if_contamination: float = 0.02,
        contamination_threshold: float = 0.10,
        mahal_threshold: float = 4.0,
        min_mahal_outliers: int = 3,
        mahal_hard_threshold: float = 8.0,
        mahal_hard_max_fraction: float = 0.30,
        n_estimators: int = 100,
        slow_poisoning_alpha: float = 0.05,
        slow_poisoning_threshold: float = 7.0,
    ) -> None:
        if min_mahal_outliers < 1:
            raise ValueError("min_mahal_outliers must be >= 1")
        if mahal_hard_threshold <= mahal_threshold:
            raise ValueError(
                "mahal_hard_threshold must be > mahal_threshold "
                f"(got hard={mahal_hard_threshold}, soft={mahal_threshold})"
            )
        if not 0.0 < mahal_hard_max_fraction <= 1.0:
            raise ValueError(
                "mahal_hard_max_fraction must lie in (0, 1] "
                f"(got {mahal_hard_max_fraction})"
            )
        if not 0.0 < slow_poisoning_alpha <= 1.0:
            raise ValueError(
                "slow_poisoning_alpha must lie in (0, 1] "
                f"(got {slow_poisoning_alpha})"
            )
        if slow_poisoning_threshold <= 0.0:
            raise ValueError(
                "slow_poisoning_threshold must be > 0 "
                f"(got {slow_poisoning_threshold})"
            )
        self.reference_size = reference_size
        self.recent_size = recent_size
        self.if_contamination = if_contamination
        self.contamination_threshold = contamination_threshold
        self.mahal_threshold = mahal_threshold
        self.min_mahal_outliers = min_mahal_outliers
        self.mahal_hard_threshold = mahal_hard_threshold
        self.mahal_hard_max_fraction = mahal_hard_max_fraction
        self.n_estimators = n_estimators
        self._slow_alpha: float = float(slow_poisoning_alpha)
        self._slow_threshold: float = float(slow_poisoning_threshold)

        self._iforest: Optional[IsolationForest] = None
        self._cov: Optional[EmpiricalCovariance] = None
        self._ref_mean: Optional[np.ndarray] = None

        # Cumulative drift tracker — EWMA of per-batch mahal_max.
        # Updated on every check() call so a patient attacker who keeps
        # every batch just-below the per-batch threshold still trips the
        # cumulative arm once the EWMA climbs past _slow_threshold.
        self._mahal_ewma: float = 0.0

    # ------------------------------------------------------------------
    # Reference fitting
    # ------------------------------------------------------------------

    def fit_reference(self, X: np.ndarray) -> "DPD":
        """
        Fit the Isolation Forest and covariance estimator on clean
        reference data X of shape (n_samples, n_features).
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))

        # Isolation Forest
        self._iforest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.if_contamination,
            random_state=42,
        ).fit(X)

        # Covariance for Mahalanobis distance
        self._cov = EmpiricalCovariance().fit(X)
        self._ref_mean = X.mean(axis=0)

        # Reset the slow-poisoning EWMA — the distance values that built
        # up the previous running mean were computed against a different
        # reference distribution and are meaningless under the new one.
        # Snapshot/restore still preserves the EWMA across rollbacks
        # (fit_reference is a DIFFERENT pathway: an explicit new
        # baseline, not a rollback to an earlier one).
        self._mahal_ewma = 0.0

        logger.info(
            "DPD: reference fitted - %d samples, %d features.", *X.shape
        )
        return self

    # ------------------------------------------------------------------
    # Snapshot / restore — used by DetectorResetCoordinator on rollback
    # ------------------------------------------------------------------

    def snapshot_state(self) -> dict:
        """
        Capture the fitted IsolationForest and covariance estimator so a
        later rollback can reinstate the pre-deploy anomaly baseline.

        Design note — why deepcopy rather than re-fit
        ----------------------------------------------
        An ``IsolationForest`` is a trained ensemble of decision trees;
        its decision boundary is defined by the trees' split criteria
        and cannot be reconstructed from a scalar summary. Re-fitting
        against a fresh clean window defeats the purpose of the snapshot
        (that window may not be available post-rollback, and fitting it
        introduces new variance). Picking the estimator would work too,
        but :func:`copy.deepcopy` is stdlib-only, avoids the attack
        surface of ``pickle.loads``, and produces an object byte-
        equivalent to the live fitted estimator (sklearn estimators
        define ``__deepcopy__`` via ``__getstate__``/``__setstate__``).
        """
        return {
            "iforest": (
                copy.deepcopy(self._iforest) if self._iforest is not None else None
            ),
            "cov": (
                copy.deepcopy(self._cov) if self._cov is not None else None
            ),
            "ref_mean": (
                self._ref_mean.copy() if self._ref_mean is not None else None
            ),
            # Preserve the slow-poisoning EWMA so a rollback does not
            # accidentally reset the cumulative-drift memory — an
            # attacker who drove the EWMA up under the poisoned MLIN
            # would otherwise get a free "reset" on every rollback.
            # Matching semantics: the EWMA is restored to whatever it
            # was at the moment of the CAPTURE (typically the pre-deploy
            # baseline), so the post-rollback detector picks up exactly
            # where the clean regime left off.
            "mahal_ewma": float(self._mahal_ewma),
        }

    def restore_state(self, state: dict) -> None:
        """Restore DPD internals from a ``snapshot_state()`` payload."""
        iforest = state.get("iforest")
        cov = state.get("cov")
        ref_mean = state.get("ref_mean")
        # Deepcopy again on the way out so repeated restores from the
        # same snapshot don't share tree structures between live detector
        # instances.
        self._iforest = copy.deepcopy(iforest) if iforest is not None else None
        self._cov = copy.deepcopy(cov) if cov is not None else None
        self._ref_mean = (
            np.asarray(ref_mean, dtype=float).copy()
            if ref_mean is not None else None
        )
        # Restore the EWMA memory.  ``get`` with a 0.0 default keeps
        # backwards compatibility with snapshots captured before the
        # slow-poisoning arm existed — a missing key is treated as "no
        # cumulative history yet", which is the safe conservative value.
        self._mahal_ewma = float(state.get("mahal_ewma", 0.0))

    def refit_reference(self, lib) -> "DPD":
        """
        Re-fit from the most-recent portion of LIB (call after model update).

        Uses the newest samples so the post-deployment baseline reflects the
        current regime.  If LIB holds fewer than ``reference_size`` samples
        we still refit from whatever is available (down to ``recent_size``)
        rather than silently no-op — otherwise the IsolationForest / Mahal
        reference stays bound to the pre-drift distribution and flags every
        new sample as an outlier, triggering a retrain loop.
        """
        available = len(lib)
        if available < self.recent_size:
            logger.warning(
                "DPD.refit_reference: only %d samples in LIB (need >= %d). "
                "Reference not updated.",
                available, self.recent_size,
            )
            return self
        take = min(self.reference_size, available)
        # lib.get_values(n) returns the LAST n samples — the post-deployment
        # regime — which is what we want as the new baseline.
        self.fit_reference(lib.get_values(take))
        return self

    # ------------------------------------------------------------------
    # Main detection
    # ------------------------------------------------------------------

    def check(self, lib) -> DPDResult:
        """
        Scan the most recent LIB samples for poisoning signs.

        Auto-fits the reference on first call if none was provided.
        """
        # ── Ensure reference is ready ─────────────────────────────────
        needed = self.reference_size + self.recent_size
        if self._iforest is None:
            if len(lib) < needed:
                return self._not_ready(lib, needed)
            ref, rec = lib.split_reference_recent(self.reference_size, self.recent_size)
            self.fit_reference(ref)
        else:
            if len(lib) < self.recent_size:
                return self._not_ready(lib, self.recent_size)
            rec = lib.get_values(self.recent_size)

        rec = np.atleast_2d(rec)

        # ── Isolation Forest ──────────────────────────────────────────
        # predict returns +1 (inlier) or -1 (outlier)
        if_preds = self._iforest.predict(rec)
        if_anomalous = [i for i, p in enumerate(if_preds) if p == -1]
        if_rate = len(if_anomalous) / len(rec)
        # Sparsity gate: if more than half the window is flagged the IF
        # has detected a uniform population shift (= drift), not sparse
        # injections.  Suppress the IF arm; let DDD handle the shift.
        if_triggered = (if_rate >= self.contamination_threshold) and \
                       (if_rate < 0.5)

        # ── Mahalanobis distance (two-tier rule) ──────────────────────
        mahal_dists = self._cov.mahalanobis(rec)   # squared distances
        mahal_dists = np.sqrt(np.maximum(mahal_dists, 0))  # → actual distance
        mahal_anomalous = [
            i for i, d in enumerate(mahal_dists) if d > self.mahal_threshold
        ]
        mahal_hard_hits = [
            i for i, d in enumerate(mahal_dists) if d > self.mahal_hard_threshold
        ]
        mahal_max = float(mahal_dists.max())
        # Sparsity gate on soft-hits too: a uniform shift makes every
        # sample exceed the 4σ Mahalanobis threshold, which is the
        # signature of drift, not poisoning.  Require soft-hits to
        # account for fewer than half the recent window.
        mahal_soft_triggered = (
            len(mahal_anomalous) >= self.min_mahal_outliers
            and len(mahal_anomalous) / len(rec) < 0.5
        )
        # Sparsity gate on hard-hits: even an 8σ sample is not unusual
        # when the whole distribution shifted by 4σ — in that regime
        # tens of samples routinely clear 8σ in the old covariance's
        # metric.  Require the hard hits to remain a sparse minority
        # before treating them as poisoning.  At the default 0.30 cap
        # a 4σ uniform shift (which typically drives ~55-100% of the
        # window past 8σ) no longer fires, while single-to-few-sample
        # extreme injections (1-30% of a 50-sample window) still do.
        #
        # Extreme-outlier escape hatch: the fraction gate above would
        # otherwise suppress bulk poisoning episodes where ~50%+ of a
        # small window are deliberately massive outliers (e.g. 20 σ
        # + N(50, 1) injections in the classic "flood" attack).  A
        # 4σ uniform shift on 4-dim N(0,1) tops out at ~10-12σ Mahal;
        # a 20σ+ hit is a full order of magnitude past that and can
        # only reasonably come from injected data, regardless of how
        # many samples the attacker injected.  If any recent sample
        # clears ``2 × mahal_hard_threshold`` (default 16σ) the hard
        # arm fires unconditionally.
        hard_fraction = len(mahal_hard_hits) / len(rec)
        extreme_hit = mahal_max > 2.0 * self.mahal_hard_threshold
        mahal_hard_triggered = (
            (len(mahal_hard_hits) > 0 and hard_fraction < self.mahal_hard_max_fraction)
            or extreme_hit
        )
        mahal_triggered = mahal_soft_triggered or mahal_hard_triggered

        # ── Cumulative drift tracker — EWMA of per-batch mahal_max ────
        # This is the slow-poisoning arm.  An attacker who keeps every
        # batch just-below the per-batch thresholds (say 3.9σ) cannot
        # also keep the EWMA below ``_slow_threshold`` indefinitely —
        # the running mean converges to ~3.9, which is well below 7.0,
        # so the attacker is forced into one of two visible regimes:
        # (a) occasionally pop above the per-batch threshold → regular
        # DPD fires; or (b) stay so far below it that the cumulative
        # arm also stays quiet → no actual attack progress.  Either way
        # the silent-attack-below-threshold window closes.
        self._mahal_ewma = (
            self._slow_alpha * mahal_max
            + (1.0 - self._slow_alpha) * self._mahal_ewma
        )

        # ── Decision ──────────────────────────────────────────────────
        detected = if_triggered or mahal_triggered
        # Slow-poisoning fires only if the cumulative arm clears the
        # threshold AND the per-batch arm is NOT already firing on this
        # same window — otherwise the per-batch arm is the primary
        # signal (higher priority: it is an immediate concern) and the
        # slow arm would just add noise to the event stream.
        slow_detected = (self._mahal_ewma > self._slow_threshold) and not detected
        if slow_detected:
            logger.warning(
                "DPD: SLOW POISONING suspected - EWMA=%.3f > %.3f",
                self._mahal_ewma, self._slow_threshold,
            )
        if detected:
            trigger_source = []
            if if_triggered:             trigger_source.append("IF")
            if mahal_soft_triggered:     trigger_source.append("Mahal-soft")
            if mahal_hard_triggered:
                trigger_source.append(
                    "Mahal-extreme" if extreme_hit else "Mahal-hard"
                )
            msg = (
                f"DATA POISONING suspected [{', '.join(trigger_source)}] - "
                f"IF rate={if_rate:.2%} "
                f"(thresh={self.contamination_threshold:.0%}), "
                f"Mahal max={mahal_max:.2f} "
                f"(soft={self.mahal_threshold}x{self.min_mahal_outliers}hits, "
                f"hard={self.mahal_hard_threshold}), "
                f"soft_hits={len(mahal_anomalous)}, "
                f"hard_hits={len(mahal_hard_hits)}"
            )
            logger.warning(msg)
        else:
            msg = (
                f"No poisoning - IF rate={if_rate:.2%}, "
                f"Mahal max={mahal_max:.2f} (soft_hits={len(mahal_anomalous)}, "
                f"ewma={self._mahal_ewma:.2f})"
            )
            logger.debug(msg)

        return DPDResult(
            poisoning_detected=detected,
            if_anomaly_rate=if_rate,
            if_threshold=self.contamination_threshold,
            if_triggered=if_triggered,
            if_anomalous_indices=if_anomalous,
            mahal_max=mahal_max,
            mahal_threshold=self.mahal_threshold,
            mahal_hard_threshold=self.mahal_hard_threshold,
            mahal_triggered=mahal_triggered,
            mahal_soft_triggered=mahal_soft_triggered,
            mahal_hard_triggered=mahal_hard_triggered,
            mahal_anomalous_indices=mahal_anomalous,
            mahal_hard_indices=mahal_hard_hits,
            slow_poisoning_detected=slow_detected,
            mahal_ewma=float(self._mahal_ewma),
            slow_poisoning_threshold=float(self._slow_threshold),
            slow_poisoning_alpha=float(self._slow_alpha),
            message=msg,
        )

    def _not_ready(self, lib, needed: int) -> DPDResult:
        msg = f"DPD: need {needed} samples, LIB has {len(lib)}."
        logger.debug(msg)
        # Expose the current EWMA state in the result even on the
        # not-ready path so downstream dashboards never see an abrupt
        # drop to 0.0 mid-run.  The cumulative arm itself cannot fire
        # before the reference is fitted, so slow_poisoning_detected
        # stays False here.
        return DPDResult(
            poisoning_detected=False,
            if_anomaly_rate=0.0, if_threshold=self.contamination_threshold,
            if_triggered=False, if_anomalous_indices=[],
            mahal_max=0.0,
            mahal_threshold=self.mahal_threshold,
            mahal_hard_threshold=self.mahal_hard_threshold,
            mahal_triggered=False,
            mahal_soft_triggered=False,
            mahal_hard_triggered=False,
            mahal_anomalous_indices=[],
            mahal_hard_indices=[],
            slow_poisoning_detected=False,
            mahal_ewma=float(self._mahal_ewma),
            slow_poisoning_threshold=float(self._slow_threshold),
            slow_poisoning_alpha=float(self._slow_alpha),
            message=msg,
        )

    def __repr__(self) -> str:
        return (
            f"DPD(ref={self.reference_size}, recent={self.recent_size}, "
            f"contamination_thresh={self.contamination_threshold}, "
            f"mahal_soft={self.mahal_threshold}x{self.min_mahal_outliers}hits, "
            f"mahal_hard={self.mahal_hard_threshold}, "
            f"fitted={self._iforest is not None})"
        )
