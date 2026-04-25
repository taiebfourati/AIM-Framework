"""
detectors/ddd.py — Data Drift Detector (DDD)

Detects when the distribution of recent AIF inputs has shifted away from
a reference distribution — a signal that the model may need retraining.

Paper reference: Section IV-C
  "Data Drift Detector (DDD) is an entity that analyses the LIB data for
   data drift detection. Such detection may initiate the MToU process
   through Model Training or Update Trigger (MToUT)."

Detection strategy
------------------
Two complementary tests run on every check() call:

1. Per-feature Kolmogorov-Smirnov (KS) test  [scipy.stats.ks_2samp]
   Compares the marginal distribution of each feature independently.
   Fast, interpretable — tells you *which* features drifted.

2. Maximum Mean Discrepancy (MMD)  [closed-form RBF kernel, no extra library]
   A multivariate test that catches joint distribution shifts that
   per-feature tests miss (e.g. correlated features rotating together).

Drift is flagged when EITHER:
  - At least `min_drifted_features` features fail the KS test after
    Bonferroni correction, OR
  - The MMD² statistic exceeds `mmd_threshold`.

The reference window is fixed at construction time (fitted on the first
`reference_size` samples in LIB) and does NOT shift — this mirrors a
"stable deployment baseline". Call refit_reference() after a successful
model update to reset the baseline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional
from aif.buffers import LIB, LOB

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DDDResult:
    """Outcome of one DDD.check() call."""
    drift_detected: bool
    ks_pvalues: np.ndarray          # p-value per feature
    ks_drifted_features: list[int]  # indices of features that failed KS
    mmd_statistic: float
    mmd_threshold: float
    mmd_triggered: bool
    reference_size: int
    recent_size: int
    message: str = ""

    def __bool__(self) -> bool:
        return self.drift_detected


# ---------------------------------------------------------------------------
# MMD helper — unbiased estimator with RBF kernel
# ---------------------------------------------------------------------------

def _rbf_mmd(X: np.ndarray, Y: np.ndarray, sigma: Optional[float] = None) -> float:
    """
    Unbiased Maximum Mean Discrepancy between sample sets X and Y.

    MMD² = E[k(x,x')] - 2·E[k(x,y)] + E[k(y,y')]
    where k is an RBF kernel: k(a,b) = exp(-||a-b||² / 2σ²)

    sigma defaults to the median-heuristic bandwidth.
    """
    def _K(A: np.ndarray, B: np.ndarray, s: float) -> np.ndarray:
        diff = A[:, None, :] - B[None, :, :]      # (n, m, d)
        return np.exp(-(diff ** 2).sum(-1) / (2 * s ** 2))

    if sigma is None:
        sample = np.vstack([X[:200], Y[:200]])
        diffs = sample[:, None, :] - sample[None, :, :]
        sq = (diffs ** 2).sum(-1)
        sigma = float(np.sqrt(np.median(sq[sq > 0]) / 2))
        sigma = max(sigma, 1e-6)

    n, m = len(X), len(Y)
    Kxx = _K(X, X, sigma); np.fill_diagonal(Kxx, 0)
    Kyy = _K(Y, Y, sigma); np.fill_diagonal(Kyy, 0)
    Kxy = _K(X, Y, sigma)

    mmd2 = Kxx.sum() / (n * (n - 1)) - 2 * Kxy.mean() + Kyy.sum() / (m * (m - 1))
    return float(max(mmd2, 0.0))


# ---------------------------------------------------------------------------
# DDD
# ---------------------------------------------------------------------------

class DDD:
    """
    Data Drift Detector.

    Parameters
    ----------
    reference_size : int
        Samples to use as fixed reference baseline.
    recent_size : int
        Most-recent samples to compare against the reference.
    ks_alpha : float
        Family-wise significance level (Bonferroni-corrected per feature).
    min_drifted_features : int
        How many features must drift for KS to trigger an alert.
    mmd_threshold : float
        MMD² value above which multivariate drift is flagged.
    use_mmd : bool
        Disable on resource-constrained edge nodes if needed.
    """

    def __init__(
        self,
        reference_size: int = 300,
        recent_size: int = 100,
        ks_alpha: float = 0.05,
        min_drifted_features: int = 1,
        mmd_threshold: float = 0.05,
        use_mmd: bool = True,
    ) -> None:
        self.reference_size = reference_size
        self.recent_size = recent_size
        self.ks_alpha = ks_alpha
        self.min_drifted_features = min_drifted_features
        self.mmd_threshold = mmd_threshold
        self.use_mmd = use_mmd
        self._reference: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Reference management
    # ------------------------------------------------------------------

    def fit_reference(self, X: np.ndarray) -> "DDD":
        """Fix the reference distribution from a clean baseline array."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        self._reference = X.copy()
        logger.info("DDD: reference fitted - %d samples, %d features.", *X.shape)
        return self

    # ------------------------------------------------------------------
    # Snapshot / restore — used by DetectorResetCoordinator on rollback
    # ------------------------------------------------------------------

    def snapshot_state(self) -> dict:
        """
        Capture the reference window so a later rollback can re-point
        the KS/MMD baseline at the pre-deploy distribution. DDD has no
        online accumulator to preserve — just the reference array.
        """
        return {
            "reference": (
                self._reference.copy() if self._reference is not None else None
            ),
        }

    def restore_state(self, state: dict) -> None:
        """Restore DDD reference from a ``snapshot_state()`` payload."""
        ref = state.get("reference")
        self._reference = None if ref is None else np.asarray(ref, dtype=float).copy()

    def refit_reference(self, lib) -> "DDD":
        """
        Re-fit from the most-recent portion of LIB (call after model update).

        Uses the newest samples so the post-deployment baseline reflects the
        current regime.  If LIB holds fewer than ``reference_size`` samples
        we still refit from whatever is available (down to ``recent_size``)
        rather than silently no-op — otherwise the detector keeps firing on
        the stale pre-drift reference and triggers a retrain loop.
        """
        available = len(lib)
        if available < self.recent_size:
            logger.warning(
                "DDD.refit_reference: only %d samples in LIB (need >= %d). "
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

    def check(self, lib) -> DDDResult:
        """
        Run drift detection against the current LIB.

        Auto-fits the reference on first call if none was provided.
        """
        # ── Resolve windows ──────────────────────────────────────────
        if self._reference is None:
            needed = self.reference_size + self.recent_size
            if len(lib) < needed:
                return self._not_ready(lib, needed)
            ref, rec = lib.split_reference_recent(self.reference_size, self.recent_size)
            self.fit_reference(ref)
        else:
            if len(lib) < self.recent_size:
                return self._not_ready(lib, self.recent_size)
            rec = lib.get_values(self.recent_size)
            ref = self._reference

        ref = np.atleast_2d(ref)
        rec = np.atleast_2d(rec)
        n_features = ref.shape[1]

        # ── KS test (Bonferroni correction) ───────────────────────────
        alpha_corrected = self.ks_alpha / n_features
        ks_pvalues = np.zeros(n_features)
        drifted: list[int] = []

        for f in range(n_features):
            _, pval = stats.ks_2samp(ref[:, f], rec[:, f])
            ks_pvalues[f] = pval
            if pval < alpha_corrected:
                drifted.append(f)

        ks_triggered = len(drifted) >= self.min_drifted_features

        # ── MMD test ──────────────────────────────────────────────────
        mmd_val, mmd_triggered = 0.0, False
        if self.use_mmd:
            mmd_val = _rbf_mmd(ref, rec)
            mmd_triggered = mmd_val > self.mmd_threshold

        # ── Decision ──────────────────────────────────────────────────
        detected = ks_triggered or mmd_triggered
        if detected:
            msg = (
                f"DATA DRIFT — KS drifted features={drifted} "
                f"(α={alpha_corrected:.4f}), MMD²={mmd_val:.4f} "
                f"(thresh={self.mmd_threshold})"
            )
            logger.warning(msg)
        else:
            msg = f"No drift — min p={ks_pvalues.min():.4f}, MMD²={mmd_val:.4f}"
            logger.debug(msg)

        return DDDResult(
            drift_detected=detected,
            ks_pvalues=ks_pvalues,
            ks_drifted_features=drifted,
            mmd_statistic=mmd_val,
            mmd_threshold=self.mmd_threshold,
            mmd_triggered=mmd_triggered,
            reference_size=len(ref),
            recent_size=len(rec),
            message=msg,
        )

    def _not_ready(self, lib, needed: int) -> DDDResult:
        msg = f"DDD: need {needed} samples, LIB has {len(lib)}."
        logger.debug(msg)
        return DDDResult(
            drift_detected=False, ks_pvalues=np.array([]),
            ks_drifted_features=[], mmd_statistic=0.0,
            mmd_threshold=self.mmd_threshold, mmd_triggered=False,
            reference_size=0, recent_size=len(lib), message=msg,
        )

    def __repr__(self) -> str:
        return (
            f"DDD(ref={self.reference_size}, recent={self.recent_size}, "
            f"ks_alpha={self.ks_alpha}, mmd_thresh={self.mmd_threshold}, "
            f"fitted={self._reference is not None})"
        )
