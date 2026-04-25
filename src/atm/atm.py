"""
atm.py — AIF Training Manager (ATM)

The ATM is the controller component. It receives an MToUTSignal from the
RTP observer and decides:
  1. Whether to act now or defer (based on severity and operator policy)
  2. Which MTP variant to use (MTP-L local, MTP-C cloud, MTP-E MLflow)
  3. Whether to invoke NDT validation before deploying the new model
  4. Whether to deploy or rollback if validation fails

Paper reference: Section IV-D
  "AIF Training Manager (ATM). The entity performs a key role in the
   training process. It obtains the MToU request from RTP... The ATM
   selects one of the three MTP pipelines described below for the MToU
   process and controls the operation until existing model parameters
   or a new model are deployed."

Selection logic (Section V)
---------------------------
The ATM picks the MTP variant based on four factors ranked in priority:

  1. Severity — CRITICAL (poisoning) always routes to the fastest path.
  2. Training type — fine-tune vs full retrain vs model change.
  3. Available resources — local compute budget vs cloud availability.
  4. Operator policy — e.g. prefer_local, prefer_external, cost_limit.

Decision table (simplified from paper Section V):

  Severity    | Trigger reason        | Default MTP
  ------------|----------------------|--------------
  CRITICAL    | poisoning            | MTP-L (speed) then MTP-E if L fails
  HIGH        | drift + drift        | MTP-C or MTP-E
  MEDIUM      | single drift         | MTP-L (fine-tune) or MTP-C
  LOW/MANUAL  | operator request     | per policy

The ATM also manages the full lifecycle:
  train → NDT validate → deploy (or rollback on failure)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, Optional
import numpy as np
from sklearn.base import BaseEstimator, clone

if TYPE_CHECKING:
    from rtp.rtp import RTP, MToUTSignal, TriggerReason
    from aif.aif import AIF
    from aif.dpostp import DPostP
    from ndt.ndt import NDT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MTP variant enum
# ---------------------------------------------------------------------------

class MTPVariant(Enum):
    LOCAL    = "MTP-L"   # fine-tune in-situ, no data transfer
    CLOUD    = "MTP-C"   # operator centralised training platform
    EXTERNAL = "MTP-E"   # external MLflow platform


# ---------------------------------------------------------------------------
# ATM policy — operator-configurable knobs
# ---------------------------------------------------------------------------

@dataclass
class ATMPolicy:
    """
    Operator policy that guides ATM variant selection and lifecycle decisions.

    Parameters
    ----------
    prefer_variant : MTPVariant or None
        Force a specific MTP variant regardless of severity.
        None = automatic selection.
    local_max_samples : int
        If the LIB has more than this many samples, MTP-L is considered
        resource-constrained and the ATM escalates to MTP-C/E.
    critical_always_local_first : bool
        On CRITICAL severity, always try MTP-L first for speed,
        then fall back to MTP-E if local training fails.
    use_ndt : bool
        Whether to run NDT validation before deploying a new model.
        Set False to skip in resource-constrained environments.
    ndt_min_accuracy : float
        Minimum accuracy (or 1 - MAE_normalised for regressors) the
        candidate model must achieve on the NDT holdout set.
    auto_deploy : bool
        Automatically deploy the model if NDT passes.
        False = emit a signal and wait for operator confirmation.
    max_retrain_attempts : int
        How many times to retry training before giving up and keeping
        the current MLIO.
    """
    prefer_variant: Optional[MTPVariant] = None
    local_max_samples: int = 500
    critical_always_local_first: bool = True
    use_ndt: bool = True
    ndt_min_accuracy: float = 0.70
    auto_deploy: bool = True
    max_retrain_attempts: int = 2


# ---------------------------------------------------------------------------
# Training result
# ---------------------------------------------------------------------------

class TrainStatus(Enum):
    SUCCESS  = auto()
    FAILED   = auto()
    SKIPPED  = auto()


@dataclass
class ATMResult:
    """Full outcome of one ATM training cycle."""
    status: TrainStatus
    variant_used: Optional[MTPVariant]
    ndt_passed: Optional[bool]           # None if NDT was skipped
    deployed: bool
    attempts: int
    duration_s: float
    run_id: Optional[str] = None         # MLflow run ID (MTP-E only)
    model_uri: Optional[str] = None      # MLflow model URI (MTP-E only)
    message: str = ""

    def __str__(self) -> str:
        return (
            f"ATMResult(status={self.status.name}, "
            f"variant={self.variant_used.value if self.variant_used else 'none'}, "
            f"ndt={'pass' if self.ndt_passed else 'fail' if self.ndt_passed is False else 'skip'}, "
            f"deployed={self.deployed}, "
            f"duration={self.duration_s:.1f}s)"
        )


# ---------------------------------------------------------------------------
# ATM
# ---------------------------------------------------------------------------

class ATM:
    """
    AIF Training Manager — the controller component.

    Parameters
    ----------
    rtp : RTP
        The Runtime Pipeline being managed. ATM calls
        rtp.notify_model_updated() after a successful deploy.
    mtp_l : MTPLocal
        Local training pipeline instance (mtp_l.py).
    mtp_e : MTPExternal
        MLflow-backed training pipeline instance (mtp_e.py).
    ndt : NDT
        Network Digital Twin validator instance (ndt.py).
    policy : ATMPolicy
        Operator configuration.
    on_result : Callable[[ATMResult], None], optional
        Callback fired after each training cycle completes.
        Useful for logging to the Management Plane.
    """

    def __init__(
        self,
        rtp: "RTP",
        mtp_l,
        mtp_e,
        ndt,
        policy: Optional[ATMPolicy] = None,
        on_result: Optional[Callable[[ATMResult], None]] = None,
        mtp_c=None,
        dpostp: Optional["DPostP"] = None,
    ) -> None:
        self.rtp     = rtp
        self.mtp_l   = mtp_l
        self.mtp_e   = mtp_e
        self.mtp_c   = mtp_c   # MTPCloud — optional, drives MTPVariant.CLOUD
        self.ndt     = ndt
        self.policy  = policy or ATMPolicy()
        self._on_result = on_result

        # DPostP is optional but strongly recommended.  When absent the
        # ATM behaves exactly as before (training on the raw GT slice,
        # no reference padding, no transport encryption).  When present
        # every training slice is sanitised and reference-padded before
        # it reaches the detectors, and MTP-E calls may additionally
        # seal the payload across the trust boundary.
        if dpostp is None:
            try:
                from aif.dpostp import DPostP as _DPostP
                dpostp = _DPostP()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "ATM: DPostP default construction failed (%s); "
                    "running without post-processor.",
                    exc,
                )
        self.dpostp = dpostp

        self.history: list[ATMResult] = []
        logger.info(
            "ATM initialised. Policy: %s. MTP-C: %s. DPostP: %s",
            self.policy,
            "available" if self.mtp_c is not None else "not configured",
            "enabled" if self.dpostp is not None else "disabled",
        )

    # ------------------------------------------------------------------
    # Main entry point — called by RTP's on_mtout callback
    # ------------------------------------------------------------------

    def handle(self, signal: "MToUTSignal") -> ATMResult:
        """
        Process an MToUTSignal end-to-end:
          select variant → train → NDT validate → deploy or rollback.

        Parameters
        ----------
        signal : MToUTSignal
            The trigger signal from RTP's MToUT component.

        Returns
        -------
        ATMResult
        """
        t0 = time.time()
        logger.warning(
            "ATM: received signal — %s", signal
        )

        # ── 1. Collect training data from buffers ─────────────────────
        lib = self.rtp.buffers.lib
        lob = self.rtp.buffers.lob
        ygt = getattr(self.rtp.buffers, "ygt", None)

        if len(lib) < 50:
            msg = f"ATM: insufficient buffer data ({len(lib)} samples). Skipping."
            logger.warning(msg)
            return self._record(ATMResult(
                status=TrainStatus.SKIPPED, variant_used=None,
                ndt_passed=None, deployed=False, attempts=0,
                duration_s=time.time() - t0, message=msg,
            ))

        X_train = lib.get_values()
        y_pseudo = lob.get_flat_values()

        # Prefer ground-truth labels when enough are available.  Pseudo-
        # labels distil the current MLIN — training on them cannot recover
        # from a label-preserving regime shift because the new model just
        # mimics the old one.  YGT holds real labels (NaN where absent);
        # if most recent samples carry labels we swap those in for the
        # training targets, falling back to LOB pseudo-labels elsewhere.
        y_train = y_pseudo
        if ygt is not None and len(ygt) == len(lob):
            y_gt = ygt.get_flat_values()
            valid = ~np.isnan(y_gt)
            # As soon as we have a statistically meaningful GT sample
            # (>= 20 rows), prefer training ONLY on those rows.  Mixing
            # GT with pseudo-labels keeps the candidate anchored to the
            # old MLIN's decision boundary on the pseudo-rows, which
            # defeats the purpose of retraining under a regime shift.
            if valid.sum() >= 20:
                X_gt_only = X_train[valid]
                y_gt_only = y_gt[valid]
                # Classifier guard: if the GT-only slice happens to be
                # single-class (possible when shifted inputs land on one
                # side of the decision boundary), mix with pseudo-labels
                # so the candidate still trains on both classes.
                is_cls = hasattr(self.rtp.aif.active_estimator, "classes_")
                unique_gt = np.unique(y_gt_only)
                if is_cls and unique_gt.size < 2:
                    y_train = np.where(valid, y_gt, y_pseudo)
                    logger.info(
                        "ATM: GT-only slice is single-class; falling back "
                        "to GT/pseudo mix for %d/%d training samples.",
                        int(valid.sum()), len(y_gt),
                    )
                else:
                    X_train = X_gt_only
                    y_train = y_gt_only
                    logger.info(
                        "ATM: training on %d ground-truth samples "
                        "(pseudo-label rows dropped).",
                        int(valid.sum()),
                    )

        # ── 1b. DPostP cleaning pass (paper Section IV-D-3) ───────────
        # Sanitise the training batch before it reaches the MTP: drop
        # NaN/inf rows, z-clip column outliers past ±5σ, and de-dup.
        # This is the paper's "cleaning and filtering LOB outputs before
        # retraining" stage — implemented here because ATM owns the
        # batch-assembly step.  A degenerate empty-after-clean batch
        # falls through to the existing min-samples guard below.
        if self.dpostp is not None and len(X_train) > 0:
            try:
                X_train, y_train = self.dpostp.process_training_batch(
                    X_train, y_train,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "ATM: DPostP.process_training_batch failed (%s); "
                    "continuing with raw batch.",
                    exc,
                )

        if len(X_train) < 20:
            msg = (
                f"ATM: training batch too thin after DPostP cleaning "
                f"({len(X_train)} rows < 20). Skipping."
            )
            logger.warning(msg)
            return self._record(ATMResult(
                status=TrainStatus.SKIPPED, variant_used=None,
                ndt_passed=None, deployed=False, attempts=0,
                duration_s=time.time() - t0, message=msg,
            ))

        # ── 2. Select MTP variant ─────────────────────────────────────
        variant = self._select_variant(signal, len(X_train))
        logger.info("ATM: selected variant=%s", variant.value)

        # ── 3. Training loop with retries ─────────────────────────────
        candidate = None
        run_id = None
        model_uri = None
        last_error = ""

        for attempt in range(1, self.policy.max_retrain_attempts + 1):
            logger.info("ATM: training attempt %d/%d via %s",
                        attempt, self.policy.max_retrain_attempts, variant.value)
            try:
                if variant == MTPVariant.LOCAL:
                    candidate = self.mtp_l.train(
                        X_train, y_train,
                        base_model=self.rtp.aif.active_estimator,
                    )
                elif variant == MTPVariant.EXTERNAL:
                    result = self.mtp_e.train(
                        X_train, y_train,
                        base_model=self.rtp.aif.active_estimator,
                        signal=signal,
                    )
                    candidate = result["model"]
                    run_id    = result["run_id"]
                    model_uri = result["model_uri"]
                elif variant == MTPVariant.CLOUD:
                    if self.mtp_c is not None:
                        candidate = self.mtp_c.train(
                            X_train, y_train,
                            base_model=self.rtp.aif.active_estimator,
                            signal=signal,
                        )
                    else:
                        logger.warning(
                            "ATM: MTP-C requested but not configured; "
                            "falling back to MTP-L."
                        )
                        variant = MTPVariant.LOCAL
                        candidate = self.mtp_l.train(
                            X_train, y_train,
                            base_model=self.rtp.aif.active_estimator,
                        )
                else:
                    logger.warning("ATM: unknown variant %s, falling back to MTP-L.",
                                   variant)
                    variant = MTPVariant.LOCAL
                    candidate = self.mtp_l.train(
                        X_train, y_train,
                        base_model=self.rtp.aif.active_estimator,
                    )

                if candidate is not None:
                    break

            except Exception as exc:
                last_error = str(exc)
                logger.error("ATM: training attempt %d failed — %s", attempt, exc)

                # On CRITICAL + LOCAL failure, escalate to MTP-E
                if (variant == MTPVariant.LOCAL
                        and self.policy.critical_always_local_first
                        and signal.severity() == "CRITICAL"
                        and attempt == 1):
                    logger.warning("ATM: escalating to MTP-E after local failure.")
                    variant = MTPVariant.EXTERNAL

        if candidate is None:
            msg = f"ATM: all training attempts failed. Last error: {last_error}"
            logger.error(msg)
            return self._record(ATMResult(
                status=TrainStatus.FAILED, variant_used=variant,
                ndt_passed=None, deployed=False,
                attempts=self.policy.max_retrain_attempts,
                duration_s=time.time() - t0, message=msg,
                run_id=run_id,
            ))

        # ── 4. NDT validation ─────────────────────────────────────────
        ndt_passed: Optional[bool] = None

        if self.policy.use_ndt:
            n_val = min(200, len(lib))
            X_val = lib.get_values(n_val)
            y_val_pseudo = lob.get_flat_values(n_val)

            # Prefer ground-truth for the pass/fail decision when enough of
            # the holdout carries real labels.  Mixing GT with pseudo-labels
            # is wrong — a candidate correctly retrained on the new regime
            # scores well against GT rows but terribly against the old
            # MLIN's pseudo-labels on drifted inputs, and the combined
            # score is dragged negative.  Instead, when enough GT is
            # available, subset *both* X_val and y_val to GT-only rows so
            # NDT evaluates the candidate against the true target only.
            y_val_for_decision = y_val_pseudo
            y_val_gt_arg = None
            if ygt is not None and len(ygt) >= n_val:
                y_val_gt = ygt.get_flat_values(n_val)
                valid = ~np.isnan(y_val_gt)
                # Use GT-only as soon as we have a statistically meaningful
                # sample (20 rows gives a ~20% accuracy margin at 95% CI).
                # Any honest GT signal dominates stale pseudo-labels, which
                # reflect the old MLIN's bias rather than truth.
                if valid.sum() >= 20:
                    X_val = X_val[valid]
                    y_val_for_decision = y_val_gt[valid]
                    y_val_gt_arg = y_val_for_decision
                    logger.info(
                        "ATM: NDT using ground-truth only for %d holdout "
                        "samples (pseudo-label rows dropped).",
                        int(valid.sum()),
                    )

            ndt_passed = self.ndt.validate(
                candidate,
                X_val=X_val,
                y_val=y_val_for_decision,
                min_score=self.policy.ndt_min_accuracy,
                run_id=run_id,           # links NDT result to MLflow run
                y_val_gt=y_val_gt_arg,
            )
            if not ndt_passed:
                msg = (
                    f"ATM: NDT validation failed for candidate model. "
                    f"Keeping current MLIN. run_id={run_id}"
                )
                logger.warning(msg)

                # Promote MLflow run to "failed" stage
                if run_id and hasattr(self.mtp_e, "mark_failed"):
                    self.mtp_e.mark_failed(run_id)

                return self._record(ATMResult(
                    status=TrainStatus.FAILED, variant_used=variant,
                    ndt_passed=False, deployed=False,
                    attempts=attempt,
                    duration_s=time.time() - t0, message=msg,
                    run_id=run_id, model_uri=model_uri,
                ))

        # ── 5. Deploy ─────────────────────────────────────────────────
        if self.policy.auto_deploy:
            self._deploy(candidate, run_id, X_train=X_train, y_train=y_train)
            deployed = True
            msg = (
                f"ATM: model deployed via {variant.value}. "
                f"run_id={run_id}, ndt={'pass' if ndt_passed else 'skip'}"
            )
            logger.warning(msg)
        else:
            deployed = False
            msg = "ATM: auto_deploy=False. Model ready but awaiting operator confirmation."
            logger.info(msg)

        result = ATMResult(
            status=TrainStatus.SUCCESS,
            variant_used=variant,
            ndt_passed=ndt_passed,
            deployed=deployed,
            attempts=attempt,
            duration_s=time.time() - t0,
            run_id=run_id,
            model_uri=model_uri,
            message=msg,
        )
        return self._record(result)

    # ------------------------------------------------------------------
    # Manual operator trigger
    # ------------------------------------------------------------------

    def operator_retrain(
        self,
        variant: Optional[MTPVariant] = None,
        reason: str = "operator request",
    ) -> ATMResult:
        """
        Trigger a manual retraining cycle outside of RTP's MToUT flow.
        Useful for scheduled retraining or thesis demonstrations.

        The ``variant`` parameter, if supplied, temporarily overrides
        ``policy.prefer_variant`` for this single call only.  The
        previous value is restored in a ``finally`` block so subsequent
        automatic retrains are not permanently locked to the manually
        requested variant.
        """
        from rtp.rtp import (MToUTSignal, TriggerReason)
        signal = MToUTSignal(
            reasons=[TriggerReason.OPERATOR_REQUEST],
            step=self.rtp._step,
            kpi_context={"reason": reason},
        )
        prev = self.policy.prefer_variant
        if variant:
            self.policy.prefer_variant = variant
        try:
            return self.handle(signal)
        finally:
            self.policy.prefer_variant = prev

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_variant(
        self, signal: "MToUTSignal", n_samples: int
    ) -> MTPVariant:
        """
        Apply the selection logic from paper Section V.
        """
        # Operator override takes precedence
        if self.policy.prefer_variant is not None:
            return self.policy.prefer_variant

        from rtp.rtp import TriggerReason
        severity = signal.severity()
        reasons  = {r.name for r in signal.reasons}

        # CRITICAL: poisoning → speed matters most → local first
        if severity == "CRITICAL":
            if (self.policy.critical_always_local_first
                    and n_samples <= self.policy.local_max_samples):
                return MTPVariant.LOCAL
            return MTPVariant.EXTERNAL

        # HIGH: both drift types → full retrain → external for quality
        if severity == "HIGH":
            return MTPVariant.EXTERNAL

        # MEDIUM / LOW: single drift type
        # If data is small enough, local fine-tune is sufficient
        if n_samples <= self.policy.local_max_samples:
            return MTPVariant.LOCAL

        # Too much data for local → use external
        return MTPVariant.EXTERNAL

    def _deploy(
        self,
        model: BaseEstimator,
        run_id: Optional[str],
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
    ) -> None:
        """
        Install the validated model into the AIF and notify RTP.

        ``X_train``/``y_train`` — when supplied — are forwarded as the new
        labelled reference for CPD's shadow model.  Without them the shadow
        stays bound to the pre-retrain label distribution, which causes
        CPD to false-fire on streams whose P(Y|X) has legitimately shifted
        (the model we just deployed was retrained to handle it).

        When a DPostP is attached the training batch is padded with
        recent LIB rows (labelled via ``model.predict``) up to
        ``dpostp.min_ref_rows``, stabilising CPD's correlation baseline
        against the sampling noise inherent to thin GT-only slices.
        """
        X_ref = X_train
        y_ref = y_train
        if (self.dpostp is not None
                and X_train is not None
                and y_train is not None
                and len(X_train) > 0):
            try:
                X_ref, y_ref = self.dpostp.build_reference(
                    X_train, y_train,
                    lib=self.rtp.buffers.lib,
                    new_estimator=model,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "ATM: DPostP.build_reference failed (%s); "
                    "falling back to raw training batch as reference.",
                    exc,
                )
                X_ref, y_ref = X_train, y_train

        self.rtp.notify_model_updated(
            model,
            X_new_ref=X_ref,
            y_new_ref=y_ref,
        )

        # Promote in MLflow registry if applicable
        if run_id and hasattr(self.mtp_e, "promote_to_production"):
            self.mtp_e.promote_to_production(run_id)

    def _record(self, result: ATMResult) -> ATMResult:
        self.history.append(result)
        logger.info("ATM: cycle complete — %s", result)
        if self._on_result:
            self._on_result(result)
        return result

    def summary(self) -> dict:
        """Return a count of outcomes — useful for thesis results tables."""
        counts: dict[str, int] = {}
        for r in self.history:
            counts[r.status.name] = counts.get(r.status.name, 0) + 1
        return counts

    def __repr__(self) -> str:
        return (
            f"ATM(cycles={len(self.history)}, "
            f"policy=prefer_{self.policy.prefer_variant}, "
            f"ndt={self.policy.use_ndt})"
        )
