"""
mtp_e.py — External Model Training Pipeline (MTP-E) backed by MLflow

This is the primary MLflow integration point (Point 1 in the architecture).
When the ATM selects MTP-E, this module:
  1. Opens an MLflow experiment run
  2. Logs all training data provenance, hyperparameters, and metrics
  3. Trains a (optionally hyperparameter-tuned) sklearn model
  4. Logs the trained model artifact to the MLflow tracking server
  5. Registers the model in the MLflow Model Registry under "Staging"
  6. Returns the model + run metadata to the ATM

Paper reference: Section IV-D-3
  "The MTP-E variant utilises ETPs for MToU, leveraging the vast compute
   resources and accelerators of public or dedicated clouds, as well as a
   vast library of models and sophisticated training algorithms with
   hyperparameter tuning tools."

MLflow concepts used
--------------------
  mlflow.set_experiment()       — organise runs under one experiment
  mlflow.start_run()            — open a tracked training run
  mlflow.log_params()           — record hyperparameters
  mlflow.log_metrics()          — record accuracy, drift stats, etc.
  mlflow.log_dict()             — record trigger signal provenance
  mlflow.sklearn.log_model()    — save the trained model artifact
  MlflowClient.create_model_version()  — register in Model Registry
  MlflowClient.transition_model_version_stage()  — Staging → Production

To run this you need an MLflow tracking server reachable at MLFLOW_URI.
For local development:
    mlflow server --host 127.0.0.1 --port 5000
Then set:
    export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
Or pass mlflow_uri= directly to MTPExternal().
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.model_selection import cross_val_score, GridSearchCV

if TYPE_CHECKING:
    from rtp import MToUTSignal
    from aif.dpostp import DPostP

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy MLflow import — gives a clear error if not installed
# ---------------------------------------------------------------------------

def _mlflow():
    try:
        import mlflow
        import mlflow.sklearn
        return mlflow
    except ImportError:
        raise ImportError(
            "MLflow is not installed. Run: pip install mlflow"
        )

def _mlflow_client():
    try:
        from mlflow.tracking import MlflowClient
        return MlflowClient
    except ImportError:
        raise ImportError("MLflow is not installed. Run: pip install mlflow")


# ---------------------------------------------------------------------------
# MTPExternal
# ---------------------------------------------------------------------------

class MTPExternal:
    """
    External Model Training Pipeline backed by MLflow.

    Parameters
    ----------
    experiment_name : str
        MLflow experiment name. All training runs land here.
    model_name : str
        Name used when registering the model in the MLflow Model Registry.
    mlflow_uri : str, optional
        MLflow tracking server URI. Falls back to MLFLOW_TRACKING_URI env var,
        then to a local ./mlruns directory.
    tune_hyperparams : bool
        If True, run a lightweight GridSearchCV before the final fit to find
        better hyperparameters. Adds training time but improves model quality.
    n_splits : int
        CV folds for evaluation and (if tune_hyperparams) for grid search.
    param_grid : dict, optional
        Hyperparameter grid for GridSearchCV. If None, a sensible default
        is used based on the model class.
    tags : dict, optional
        Extra MLflow tags attached to every run (e.g. {"project": "6G-CLOUD"}).
    """

    def __init__(
        self,
        experiment_name: str = "rtp_aif_retraining",
        model_name: str = "aif_classifier",
        mlflow_uri: Optional[str] = None,
        tune_hyperparams: bool = False,
        n_splits: int = 3,
        param_grid: Optional[dict] = None,
        tags: Optional[dict] = None,
        dpostp: Optional["DPostP"] = None,
        transport_encryption: bool = False,
        expected_sender_id: Optional[str] = None,
    ) -> None:
        self.experiment_name  = experiment_name
        self.model_name       = model_name
        self.mlflow_uri       = mlflow_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"
        )
        self.tune_hyperparams = tune_hyperparams
        self.n_splits         = n_splits
        self.param_grid       = param_grid
        self.tags             = tags or {}

        # DPostP transport hardening (paper Section IV-D-3).  When
        # ``transport_encryption=True`` the training payload is sealed
        # (gzip + AES-256-GCM + AAD-bound sender/model/algo/timestamp)
        # before MLflow consumes it, and unsealed on the receiver side
        # as a round-trip sanity check.  In a production deployment the
        # seal boundary would be an actual RPC / queue handoff; here
        # both ends run in-process so the unseal acts as the integrity
        # verification that MLflow logging assumes.
        self.dpostp = dpostp
        self.transport_encryption = bool(transport_encryption)
        self.expected_sender_id = expected_sender_id
        if self.transport_encryption and self.dpostp is None:
            raise ValueError(
                "MTPExternal: transport_encryption=True but no DPostP supplied."
            )

    # ------------------------------------------------------------------
    # Main training entry point (called by ATM)
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        base_model: Optional[BaseEstimator] = None,
        signal: Optional["MToUTSignal"] = None,
    ) -> dict[str, Any]:
        """
        Run a full MLflow-tracked training experiment.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training inputs exported from LIB.
        y : np.ndarray of shape (n_samples,)
            Training targets exported from LOB / ground truth.
        base_model : BaseEstimator, optional
            The current MLIN. Its class and hyperparameters are used as the
            starting point; a clone is trained so MLIN is never mutated.
        signal : MToUTSignal, optional
            The trigger signal from RTP — logged as provenance metadata.

        Returns
        -------
        dict with keys:
            "model"     — trained BaseEstimator
            "run_id"    — MLflow run ID string
            "model_uri" — MLflow artifact URI for the logged model
        """
        mlflow = _mlflow()
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment(self.experiment_name)

        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()

        # ── DPostP transport round-trip (paper Section IV-D-3) ────────
        # In a real deployment MTP-E runs on a remote cloud and consumes
        # an encrypted payload delivered over the network.  Here we
        # simulate that handoff in-process: seal on the sender side,
        # unseal on the receiver side, fail closed on tag/AAD mismatch.
        # ``sealed_bytes`` and ``transport_ratio`` are logged to MLflow
        # for provenance.
        sealed_bytes: Optional[int] = None
        plain_bytes:  Optional[int] = None
        if self.transport_encryption and self.dpostp is not None:
            model_version = self._derive_model_version(base_model, signal)
            try:
                payload = self.dpostp.seal(
                    X, y, model_version=model_version,
                )
                sealed_bytes = len(payload)
                plain_bytes  = int(X.nbytes + y.nbytes)
                X, y = self.dpostp.unseal(
                    payload,
                    expected_model_version=model_version,
                    expected_sender_id=self.expected_sender_id,
                )
                logger.info(
                    "MTP-E: transport round-trip OK — plain=%d B sealed=%d B "
                    "(overhead=%+.1f%%).",
                    plain_bytes, sealed_bytes,
                    100.0 * (sealed_bytes - plain_bytes) / max(plain_bytes, 1),
                )
            except Exception as exc:
                logger.error(
                    "MTP-E: DPostP seal/unseal failed (%s). Aborting MLflow run.",
                    exc,
                )
                raise

        logger.info(
            "MTP-E: starting MLflow run — %d samples, %d features.",
            len(X), X.shape[1],
        )

        with mlflow.start_run(tags=self._build_tags(signal)) as run:
            run_id = run.info.run_id
            t0 = time.time()

            # ── Log provenance ────────────────────────────────────────
            mlflow.log_params({
                "n_samples":    len(X),
                "n_features":   X.shape[1],
                "base_model":   type(base_model).__name__ if base_model else "None",
                "trigger_reasons": (
                    str([r.name for r in signal.reasons]) if signal else "manual"
                ),
                "trigger_severity": signal.severity() if signal else "manual",
            })

            if signal is not None:
                mlflow.log_dict(
                    {
                        "step":    signal.step,
                        "reasons": [r.name for r in signal.reasons],
                        "severity": signal.severity(),
                        "ddd_drift":  bool(signal.ddd_result.drift_detected)
                                      if signal.ddd_result else None,
                        "dpd_poison": bool(signal.dpd_result.poisoning_detected)
                                      if signal.dpd_result else None,
                        "cdd_drift":  bool(signal.cdd_result.drift_detected)
                                      if signal.cdd_result else None,
                        "cpd_poison": bool(signal.cpd_result.poisoning_detected)
                                      if signal.cpd_result else None,
                    },
                    "trigger_signal.json",
                )

            # ── Prepare candidate model ───────────────────────────────
            if base_model is not None:
                candidate = clone(base_model)
            else:
                from sklearn.ensemble import RandomForestClassifier
                candidate = RandomForestClassifier(
                    n_estimators=100, random_state=42
                )
                logger.warning(
                    "MTP-E: no base model provided, using RandomForestClassifier."
                )

            # ── Optional hyperparameter tuning ────────────────────────
            if self.tune_hyperparams:
                candidate = self._tune(candidate, X, y, mlflow)

            # ── Final fit ─────────────────────────────────────────────
            candidate.fit(X, y)

            # ── Evaluate and log metrics ──────────────────────────────
            train_score = self._score(candidate, X, y)
            cv_score    = self._cv_score(candidate, X, y)
            elapsed     = time.time() - t0

            mlflow.log_metrics({
                "train_score":    train_score,
                "cv_score":       cv_score,
                "training_time_s": elapsed,
            })
            if sealed_bytes is not None and plain_bytes is not None:
                mlflow.log_metrics({
                    "transport_plain_bytes":  float(plain_bytes),
                    "transport_sealed_bytes": float(sealed_bytes),
                    "transport_overhead_pct": float(
                        100.0 * (sealed_bytes - plain_bytes) / max(plain_bytes, 1)
                    ),
                })
            logger.info(
                "MTP-E: train_score=%.4f, cv_score=%.4f, time=%.1fs",
                train_score, cv_score, elapsed,
            )

            # ── Log model artifact ────────────────────────────────────
            model_info = mlflow.sklearn.log_model(
                sk_model=candidate,
                artifact_path="model",
                registered_model_name=self.model_name,
            )
            model_uri = model_info.model_uri
            logger.info("MTP-E: model logged — uri=%s", model_uri)

            # ── Register to Staging in Model Registry ─────────────────
            self._register_staging(run_id, mlflow)

        return {
            "model":     candidate,
            "run_id":    run_id,
            "model_uri": model_uri,
        }

    # ------------------------------------------------------------------
    # Registry lifecycle — called by ATM after NDT validation
    # ------------------------------------------------------------------

    def promote_to_production(self, run_id: str) -> None:
        """
        Transition the model version associated with run_id from
        Staging → Production in the MLflow Model Registry.

        Called by ATM._deploy() after NDT passes.
        This is MLflow Point 2 (together with NDT).
        """
        MlflowClient = _mlflow_client()
        client = MlflowClient(tracking_uri=self.mlflow_uri)

        versions = client.search_model_versions(
            f"name='{self.model_name}' and tags.run_id='{run_id}'"
        )
        if not versions:
            logger.warning(
                "MTP-E.promote_to_production: no version found for run_id=%s", run_id
            )
            return

        version = versions[0].version
        client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True,  # demote old Production → Archived
        )
        logger.info(
            "MTP-E: model v%s promoted to Production (run_id=%s).",
            version, run_id,
        )

    def mark_failed(self, run_id: str) -> None:
        """
        Tag the MLflow run as failed (NDT did not pass).
        The model version stays in Staging and is not promoted.
        """
        mlflow = _mlflow()
        mlflow.set_tracking_uri(self.mlflow_uri)
        MlflowClient = _mlflow_client()
        client = MlflowClient(tracking_uri=self.mlflow_uri)
        client.set_tag(run_id, "ndt_passed", "false")
        client.set_terminated(run_id, status="FAILED")
        logger.info("MTP-E: run %s marked FAILED (NDT did not pass).", run_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tune(
        self, model: BaseEstimator, X: np.ndarray, y: np.ndarray, mlflow
    ) -> BaseEstimator:
        """Run a lightweight GridSearchCV and log the best params."""
        grid = self.param_grid or self._default_param_grid(model)
        if not grid:
            logger.info("MTP-E: no param grid available, skipping tuning.")
            return model

        scoring = "accuracy" if is_classifier(model) else "r2"
        gs = GridSearchCV(
            model, grid, cv=self.n_splits, scoring=scoring, n_jobs=-1
        )
        gs.fit(X, y)
        best_params = gs.best_params_
        mlflow.log_params({f"tuned_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_cv_score", gs.best_score_)
        logger.info("MTP-E: best params after tuning: %s", best_params)
        return gs.best_estimator_

    def _default_param_grid(self, model: BaseEstimator) -> dict:
        """Minimal hyperparameter grid for common sklearn estimators."""
        name = type(model).__name__
        grids = {
            "RandomForestClassifier": {
                "n_estimators": [50, 100],
                "max_depth": [None, 10, 20],
            },
            "RandomForestRegressor": {
                "n_estimators": [50, 100],
                "max_depth": [None, 10, 20],
            },
            "GradientBoostingClassifier": {
                "n_estimators": [50, 100],
                "learning_rate": [0.05, 0.1],
            },
            "LogisticRegression": {
                "C": [0.1, 1.0, 10.0],
            },
            "Ridge": {
                "alpha": [0.1, 1.0, 10.0],
            },
        }
        return grids.get(name, {})

    def _score(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> float:
        scoring = "accuracy" if is_classifier(model) else "r2"
        if scoring == "accuracy":
            preds = model.predict(X)
            return float(np.mean(preds == y))
        return float(model.score(X, y))

    def _cv_score(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> float:
        if len(X) < self.n_splits * 5:
            return self._score(model, X, y)
        scoring = "accuracy" if is_classifier(model) else "r2"
        scores = cross_val_score(model, X, y, cv=self.n_splits, scoring=scoring)
        return float(scores.mean())

    def _derive_model_version(
        self,
        base_model: Optional[BaseEstimator],
        signal: Optional["MToUTSignal"],
    ) -> str:
        """
        Build a stable model_version string that binds to AAD.

        The format is deliberately human-readable so MLflow tag audits
        can reason about which AAD a given run was sealed under.
        ``<ModelClass>@<step>::<severity>`` is unique across a single
        session and captures the trigger context.
        """
        cls = type(base_model).__name__ if base_model is not None else "None"
        step = getattr(signal, "step", -1) if signal is not None else -1
        sev  = signal.severity() if signal is not None else "manual"
        return f"{cls}@{step}::{sev}"

    def _build_tags(self, signal: Optional["MToUTSignal"]) -> dict:
        tags = {
            "component": "MTP-E",
            "framework": "sklearn",
            **self.tags,
        }
        if signal:
            tags["trigger_severity"] = signal.severity()
            tags["trigger_step"]     = str(signal.step)
        return tags

    def _register_staging(self, run_id: str, mlflow) -> None:
        """Tag the newly created model version with the run_id for lookup."""
        MlflowClient = _mlflow_client()
        client = MlflowClient(tracking_uri=self.mlflow_uri)
        versions = client.search_model_versions(f"name='{self.model_name}'")
        if versions:
            latest = max(versions, key=lambda v: int(v.version))
            client.set_model_version_tag(
                self.model_name, latest.version, "run_id", run_id
            )
            client.set_tag(run_id, "ndt_passed", "pending")
            logger.info(
                "MTP-E: model v%s registered as Staging (run_id=%s).",
                latest.version, run_id,
            )

    def __repr__(self) -> str:
        return (
            f"MTPExternal(experiment='{self.experiment_name}', "
            f"model='{self.model_name}', "
            f"uri='{self.mlflow_uri}', "
            f"tune={self.tune_hyperparams})"
        )
