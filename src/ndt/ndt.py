"""
ndt.py — Network Digital Twin (NDT) validator

The NDT is the gate between "trained" and "deployed". Before a new model
replaces the active MLIN, it must pass NDT validation — ensuring the
candidate actually performs better (or at least not worse) than the
current model on a holdout set.

Paper reference: Section IV-D
  "The deployment of a model is preceded by testing using a testing data
   set and eventually by the Network Digital Twin (NDT). The first test
   is used for the evaluation of MLIN AI-QoS, while the second one is
   used for the estimation of MLIN's impact on network indicators."

This is MLflow Point 2 in the architecture: after validation passes,
the NDT logs its result back to the MLflow run and the MTP-E pipeline
promotes the model from Staging → Production.

In a real 6G network, the NDT would replay the validation dataset through
a network simulator to check KPI impact. Here we implement:

  1. AI-QoS check — is the candidate's accuracy/R² above min_score?
  2. Improvement check — is the candidate better than the current MLIN?
  3. MLflow logging — tag the run with NDT results for audit trail.

The NDT operates on the same holdout slice that the RTP passed to ATM
(the most recent 200 samples from LIB/LOB), which the current MLIN has
already seen during inference — a realistic online evaluation setting.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, is_classifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy MLflow import
# ---------------------------------------------------------------------------

def _mlflow_client():
    try:
        from mlflow.tracking import MlflowClient
        return MlflowClient
    except ImportError:
        return None          # graceful degradation if MLflow not available


# ---------------------------------------------------------------------------
# NDT
# ---------------------------------------------------------------------------

class NDT:
    """
    Network Digital Twin — model validation gate.

    Parameters
    ----------
    current_model_getter : callable
        Zero-argument callable that returns the AIF's current active
        estimator. Used to compute the baseline score for comparison.
        Typically: lambda: rtp.aif.active_estimator
    min_score : float
        Absolute minimum AI-QoS score the candidate must achieve.
        Below this, the model is rejected regardless of comparison.
    min_improvement : float
        The candidate must beat the current model by at least this margin.
        Set to 0.0 to allow equal-or-better models through.
        Set negative (e.g. -0.02) to tolerate small regressions
        (useful when retraining after concept drift where both models
        may perform poorly until more data is collected).
    mlflow_uri : str, optional
        MLflow tracking server URI for logging NDT results.
        Falls back to MLFLOW_TRACKING_URI env var.
    """

    def __init__(
        self,
        current_model_getter,
        min_score: float = 0.70,
        min_improvement: float = 0.0,
        mlflow_uri: Optional[str] = None,
    ) -> None:
        self._get_current = current_model_getter
        self.min_score = min_score
        self.min_improvement = min_improvement
        self.mlflow_uri = mlflow_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"
        )

        self.history: list[dict] = []

        # MLflow availability cache.  None = unchecked; True = reachable;
        # False = unreachable (skip all subsequent log attempts for this
        # NDT's lifetime).  Without this latch a down tracking server
        # triggers urllib3's default 7× retry-with-exponential-backoff on
        # every validation, blocking the retraining loop indefinitely.
        self._mlflow_available: Optional[bool] = None

    # ------------------------------------------------------------------
    # Main validation entry point (called by ATM)
    # ------------------------------------------------------------------

    def validate(
        self,
        candidate: BaseEstimator,
        X_val: np.ndarray,
        y_val: np.ndarray,
        min_score: Optional[float] = None,
        run_id: Optional[str] = None,
        y_val_gt: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Validate a candidate model against the holdout set.

        Pass/fail decisions are always driven by ``y_val`` (LOB pseudo-labels)
        to preserve backward compatibility.  When ``y_val_gt`` is supplied the
        method additionally computes honest ground-truth scores and stores them
        in the history record and MLflow, making the self-referential nature of
        the pseudo-label evaluation visible in the audit trail.

        Parameters
        ----------
        candidate : BaseEstimator
            The newly trained model to evaluate.
        X_val : np.ndarray of shape (n_samples, n_features)
            Holdout inputs (from LIB).
        y_val : np.ndarray of shape (n_samples,)
            Holdout targets derived from LOB (pseudo-labels).  Used for the
            pass/fail decision so that existing callers are unaffected.
        min_score : float, optional
            Override instance-level min_score for this call.
        run_id : str, optional
            MLflow run ID to attach NDT results to.
            If None, results are only logged locally.
        y_val_gt : np.ndarray of shape (n_samples,), optional
            Actual ground-truth labels for the same holdout slice.  When
            provided, ``candidate_gt_score`` and ``baseline_gt_score`` are
            computed and included in the history record and MLflow logs.
            These scores do *not* affect the pass/fail decision.

        Returns
        -------
        bool
            True if the candidate passes all checks and can be deployed.
        """
        X_val = np.atleast_2d(np.asarray(X_val, dtype=float))
        y_val = np.asarray(y_val, dtype=float).ravel()
        threshold = min_score if min_score is not None else self.min_score

        # ── Score candidate ───────────────────────────────────────────
        candidate_score = self._score(candidate, X_val, y_val)

        # ── Score current model (baseline) ────────────────────────────
        current_model = self._get_current()
        if current_model is not None:
            try:
                baseline_score = self._score(current_model, X_val, y_val)
            except Exception:
                baseline_score = 0.0
                logger.warning("NDT: could not score current model, baseline=0.")
        else:
            baseline_score = 0.0

        # ── Check 1: absolute AI-QoS floor ────────────────────────────
        passes_floor = candidate_score >= threshold

        # ── Check 2: improvement over current model ───────────────────
        improvement = candidate_score - baseline_score
        passes_improvement = improvement >= self.min_improvement

        passed = passes_floor and passes_improvement

        # ── Ground-truth scores (honest evaluation, no pass/fail impact) ──
        candidate_gt_score: Optional[float] = None
        baseline_gt_score: Optional[float] = None
        if y_val_gt is not None:
            y_gt = np.asarray(y_val_gt, dtype=float).ravel()
            candidate_gt_score = self._score(candidate, X_val, y_gt)
            if current_model is not None:
                try:
                    baseline_gt_score = self._score(current_model, X_val, y_gt)
                except Exception:
                    baseline_gt_score = 0.0
                    logger.warning(
                        "NDT: could not compute ground-truth score for current "
                        "model, baseline_gt=0."
                    )
            else:
                baseline_gt_score = 0.0
            logger.info(
                "NDT: ground-truth scores - candidate_gt=%.4f, baseline_gt=%.4f "
                "(pseudo-label scores: candidate=%.4f, baseline=%.4f)",
                candidate_gt_score, baseline_gt_score,
                candidate_score, baseline_score,
            )

        # ── Log result ────────────────────────────────────────────────
        record = {
            "candidate_score":    candidate_score,
            "baseline_score":     baseline_score,
            "improvement":        improvement,
            "min_score":          threshold,
            "min_improvement":    self.min_improvement,
            "passed":             passed,
            "run_id":             run_id,
            "candidate_gt_score": candidate_gt_score,
            "baseline_gt_score":  baseline_gt_score,
        }
        self.history.append(record)

        if passed:
            logger.info(
                "NDT: PASSED - candidate=%.4f, baseline=%.4f, "
                "improvement=%.4f (min=%.4f)",
                candidate_score, baseline_score,
                improvement, self.min_improvement,
            )
        else:
            reasons = []
            if not passes_floor:
                reasons.append(
                    f"score {candidate_score:.4f} < floor {threshold:.4f}"
                )
            if not passes_improvement:
                reasons.append(
                    f"improvement {improvement:.4f} < min {self.min_improvement:.4f}"
                )
            logger.warning(
                "NDT: FAILED - %s", " | ".join(reasons)
            )

        # ── Log to MLflow if run_id provided ──────────────────────────
        if run_id is not None:
            self._log_to_mlflow(run_id, record)

        return passed

    # ------------------------------------------------------------------
    # Ground-truth validation (standalone entry point)
    # ------------------------------------------------------------------

    def validate_with_ground_truth(
        self,
        candidate: BaseEstimator,
        X_val: np.ndarray,
        y_true_gt: np.ndarray,
        min_score: Optional[float] = None,
        run_id: Optional[str] = None,
    ) -> bool:
        """
        Validate a candidate model using actual ground-truth labels only.

        This method is identical in structure to :meth:`validate` but scores
        both the candidate and the current model against ``y_true_gt`` rather
        than LOB pseudo-labels.  It therefore avoids the self-referential
        evaluation that causes trivially perfect pseudo-label scores and
        provides an honest measure of model quality against real targets.

        History records produced by this method carry the key
        ``"validation_mode": "ground_truth"`` so they can be distinguished
        from records produced by :meth:`validate`.

        Parameters
        ----------
        candidate : BaseEstimator
            The newly trained model to evaluate.
        X_val : np.ndarray of shape (n_samples, n_features)
            Holdout inputs (from LIB).
        y_true_gt : np.ndarray of shape (n_samples,)
            Actual ground-truth labels for the holdout slice.  These drive
            both the pass/fail decision and all logged metrics.
        min_score : float, optional
            Override instance-level min_score for this call.
        run_id : str, optional
            MLflow run ID to attach NDT results to.
            If None, results are only logged locally.

        Returns
        -------
        bool
            True if the candidate passes all checks and can be deployed.
        """
        X_val = np.atleast_2d(np.asarray(X_val, dtype=float))
        y_gt = np.asarray(y_true_gt, dtype=float).ravel()
        threshold = min_score if min_score is not None else self.min_score

        # ── Score candidate against ground truth ──────────────────────
        candidate_gt_score = self._score(candidate, X_val, y_gt)

        # ── Score current model (baseline) against ground truth ───────
        current_model = self._get_current()
        if current_model is not None:
            try:
                baseline_gt_score = self._score(current_model, X_val, y_gt)
            except Exception:
                baseline_gt_score = 0.0
                logger.warning(
                    "NDT(gt): could not score current model, baseline_gt=0."
                )
        else:
            baseline_gt_score = 0.0

        # ── Check 1: absolute AI-QoS floor ────────────────────────────
        passes_floor = candidate_gt_score >= threshold

        # ── Check 2: improvement over current model ───────────────────
        improvement = candidate_gt_score - baseline_gt_score
        passes_improvement = improvement >= self.min_improvement

        passed = passes_floor and passes_improvement

        # ── Log result ────────────────────────────────────────────────
        record = {
            "candidate_score":    candidate_gt_score,   # primary score = gt
            "baseline_score":     baseline_gt_score,    # primary baseline = gt
            "improvement":        improvement,
            "min_score":          threshold,
            "min_improvement":    self.min_improvement,
            "passed":             passed,
            "run_id":             run_id,
            "candidate_gt_score": candidate_gt_score,
            "baseline_gt_score":  baseline_gt_score,
            "validation_mode":    "ground_truth",
        }
        self.history.append(record)

        if passed:
            logger.info(
                "NDT(gt): PASSED - candidate_gt=%.4f, baseline_gt=%.4f, "
                "improvement=%.4f (min=%.4f)",
                candidate_gt_score, baseline_gt_score,
                improvement, self.min_improvement,
            )
        else:
            reasons = []
            if not passes_floor:
                reasons.append(
                    f"score {candidate_gt_score:.4f} < floor {threshold:.4f}"
                )
            if not passes_improvement:
                reasons.append(
                    f"improvement {improvement:.4f} < min {self.min_improvement:.4f}"
                )
            logger.warning(
                "NDT(gt): FAILED - %s", " | ".join(reasons)
            )

        # ── Log to MLflow if run_id provided ──────────────────────────
        if run_id is not None:
            self._log_to_mlflow(run_id, record)

        return passed

    # ------------------------------------------------------------------
    # MLflow logging (Point 2)
    # ------------------------------------------------------------------

    def _log_to_mlflow(self, run_id: str, record: dict) -> None:
        """
        Attach NDT results back to the MLflow run.
        This is the second MLflow integration point:
        the run now carries the full audit trail from training -> validation.

        Fast-fail semantics: if the MLflow server proves unreachable on any
        call, the per-instance ``_mlflow_available`` latch flips to ``False``
        and every subsequent call returns immediately.  This keeps retraining
        moving when the tracking server is down; audit logs can be
        reconstructed post-hoc from ``self.history``.
        """
        if self._mlflow_available is False:
            return                       # MLflow known-bad for this instance

        MlflowClient = _mlflow_client()
        if MlflowClient is None:
            logger.debug("NDT: MLflow not available, skipping remote logging.")
            self._mlflow_available = False
            return

        # Bound the HTTP timeout for this and all subsequent MLflow calls in
        # the current process.  urllib3's default retry budget (7 retries
        # with exponential backoff) can stall the ATM pipeline for tens of
        # seconds when the server is down, so we override it unless the
        # caller has pinned a value.
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "2")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")

        try:
            client = MlflowClient(tracking_uri=self.mlflow_uri)
            client.log_metric(run_id, "ndt_candidate_score", record["candidate_score"])
            client.log_metric(run_id, "ndt_baseline_score",  record["baseline_score"])
            client.log_metric(run_id, "ndt_improvement",     record["improvement"])
            client.set_tag(run_id, "ndt_passed", str(record["passed"]).lower())

            # Log ground-truth scores when available (honest evaluation metrics).
            if record.get("candidate_gt_score") is not None:
                client.log_metric(
                    run_id, "ndt_candidate_gt_score", record["candidate_gt_score"]
                )
            if record.get("baseline_gt_score") is not None:
                client.log_metric(
                    run_id, "ndt_baseline_gt_score", record["baseline_gt_score"]
                )
            if record.get("validation_mode") is not None:
                client.set_tag(run_id, "ndt_validation_mode", record["validation_mode"])

            self._mlflow_available = True
            logger.info(
                "NDT: results logged to MLflow run %s (passed=%s).",
                run_id, record["passed"],
            )
        except Exception as exc:
            # First failure: emit a single warning and latch the instance
            # off so we don't block retraining on every future validation.
            self._mlflow_available = False
            logger.warning(
                "NDT: MLflow logging unavailable (%s) - continuing without "
                "remote audit trail.",
                exc,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _score(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy for classifiers, R² for regressors."""
        if is_classifier(model):
            preds = model.predict(X)
            return float(np.mean(preds == y))
        return float(model.score(X, y))

    def last_result(self) -> Optional[dict]:
        return self.history[-1] if self.history else None

    def __repr__(self) -> str:
        return (
            f"NDT(min_score={self.min_score}, "
            f"min_improvement={self.min_improvement}, "
            f"validations={len(self.history)})"
        )
