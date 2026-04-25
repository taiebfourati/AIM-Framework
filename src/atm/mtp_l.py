"""
mtp_l.py — Local Model Training Pipeline (MTP-L)

The simplest training variant: everything runs in-situ, collocated with
the AIF. No data transfer, no external dependency. Uses sklearn's
partial_fit (if available) for fine-tuning, otherwise falls back to a
full refit on the buffered LIB/LOB data.

Paper reference: Section IV-D-1
  "All MTP-L components are placed close to the AIF and reuse the LIB
   and LOB buffers for their operations. Due to its limited computational
   capabilities and simple training engine, the approach is well-suited
   for fine-tuning and retraining, but not for model updates."

When to use (ATM selection logic, Section V)
--------------------------------------------
  - Severity CRITICAL: fastest path — start here, escalate to MTP-E if needed
  - Severity MEDIUM: single drift type + small dataset (<= local_max_samples)
  - When data transfer cost to cloud/external platform is prohibitive
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


class MTPLocal:
    """
    Local Model Training Pipeline.

    Parameters
    ----------
    n_splits : int
        Number of cross-validation folds used to evaluate the candidate
        model before returning it. Set to 0 to skip CV (faster).
    fine_tune_first : bool
        If True and the base model supports partial_fit, attempt
        incremental learning before falling back to full refit.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_splits: int = 3,
        fine_tune_first: bool = True,
        random_state: int = 42,
    ) -> None:
        self.n_splits = n_splits
        self.fine_tune_first = fine_tune_first
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Main training entry point
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        base_model: Optional[BaseEstimator] = None,
    ) -> BaseEstimator:
        """
        Train or fine-tune a model on (X, y).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training inputs (from LIB).
        y : np.ndarray of shape (n_samples,)
            Training targets (from LOB or ground truth).
        base_model : BaseEstimator, optional
            The current MLIN. If provided, MTP-L attempts fine-tuning
            before falling back to a full refit on a clone.

        Returns
        -------
        BaseEstimator
            Newly trained (or fine-tuned) sklearn estimator, fitted on X, y.
        """
        t0 = time.time()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()

        logger.info(
            "MTP-L: starting training — %d samples, %d features.",
            len(X), X.shape[1],
        )

        candidate = None

        # ── Attempt 1: partial_fit (incremental fine-tune) ────────────
        if self.fine_tune_first and base_model is not None:
            candidate = self._try_partial_fit(base_model, X, y)

        # ── Attempt 2: full refit on a clone ─────────────────────────
        if candidate is None:
            candidate = self._full_refit(base_model, X, y)

        # ── Optional cross-validation quality check ───────────────────
        if self.n_splits > 1 and len(X) >= self.n_splits * 10:
            score = self._cv_score(candidate, X, y)
            logger.info("MTP-L: CV score = %.4f", score)

        elapsed = time.time() - t0
        logger.info("MTP-L: training complete in %.1fs.", elapsed)
        return candidate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_partial_fit(
        self, base_model: BaseEstimator, X: np.ndarray, y: np.ndarray
    ) -> Optional[BaseEstimator]:
        """
        Attempt incremental learning via partial_fit on a deep copy.
        Returns None if the model doesn't support it.
        """
        if not hasattr(base_model, "partial_fit"):
            logger.debug("MTP-L: base model has no partial_fit, skipping.")
            return None

        import copy
        candidate = copy.deepcopy(base_model)
        try:
            if is_classifier(candidate):
                classes = np.unique(y)
                candidate.partial_fit(X, y, classes=classes)
            else:
                candidate.partial_fit(X, y)
            logger.info("MTP-L: fine-tuned via partial_fit.")
            return candidate
        except Exception as exc:
            logger.warning("MTP-L: partial_fit failed — %s. Falling back.", exc)
            return None

    def _full_refit(
        self, base_model: Optional[BaseEstimator], X: np.ndarray, y: np.ndarray
    ) -> BaseEstimator:
        """
        Full refit: clone the base model (preserving hyperparameters)
        and call fit() on the full (X, y) dataset.
        """
        if base_model is not None:
            candidate = clone(base_model)
            logger.info(
                "MTP-L: full refit on clone of %s.",
                type(base_model).__name__,
            )
        else:
            # No base model — use a sensible default
            from sklearn.ensemble import RandomForestClassifier
            candidate = RandomForestClassifier(
                n_estimators=50, random_state=self.random_state
            )
            logger.warning(
                "MTP-L: no base model provided, using RandomForestClassifier."
            )

        candidate.fit(X, y)
        return candidate

    def _cv_score(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> float:
        """Quick cross-validated score (accuracy or R²)."""
        scoring = "accuracy" if is_classifier(model) else "r2"
        scores = cross_val_score(model, X, y, cv=self.n_splits, scoring=scoring)
        return float(scores.mean())

    def __repr__(self) -> str:
        return (
            f"MTPLocal(n_splits={self.n_splits}, "
            f"fine_tune_first={self.fine_tune_first})"
        )
