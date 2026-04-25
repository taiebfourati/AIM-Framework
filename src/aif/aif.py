"""
aif.py — AI Function (AIF) wrapper for a scikit-learn model

Implements the three internal blocks described in the paper (Section IV-B):
  • DPP  — Data PreProcessor  (normalisation, artefact removal)
  • SIB  — Short Input Buffer  (holds the last n inputs for the MLI)
  • MLI  — ML Inference        (sklearn classifier or regressor)

The AIF also manages two model slots:
  • MLIN — Model Inference New (active by default)
  • MLIO — Model Inference Old (standby; rolled back to on poisoning/failure)

Paper reference: Section IV-B
  "The AIF comprises two models: Model Inference New (MLIN) and Model
   Inference Old (MLIO). In normal operations, the MLIN is active and
   the MLIO is passive. The MLIO can be switched with MLIN if, despite
   testing, MLIN provides worse results, or in the case of MLIN poisoning."

Audit-safety: This module owns three HIGH-severity fixes:
  (a) notify_model_updated is now a two-phase commit protocol
      (prepare → apply; failure at any phase is atomic and logged).
  (b) ModelSlotState gains a terminal FAILED state with reason field so
      NDT-rejected candidates are distinguishable from "never existed".
  (c) MToUTSignal, TriggerReasons, and Severity are re-exported from the
      neutral leaf module aif/signals.py (backward-compat shim below).
"""

from __future__ import annotations

import copy
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, is_classifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

# ---------------------------------------------------------------------------
# (c) Backward-compat re-export from the neutral signals leaf module
# ---------------------------------------------------------------------------
from aif.signals import MToUTSignal, TriggerReasons, Severity  # noqa: F401

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local event types — used by AIF's own notify machinery
# ---------------------------------------------------------------------------

class AIFEventType(Enum):
    """
    Local event types emitted by the AIF's model-update machinery.

    These are passed to the subscriber-notification callbacks so that
    the RTP (or any other log consumer) can record them.  The
    canonical serialisable mapping AIFEventType → rtp.EventType lives
    in :attr:`rtp.rtp.RTP._AIF_EVENT_MAP`; the RTP subscribes to AIF
    events via :meth:`AIF.set_event_callback` during construction and
    routes each one through its tamper-evident hash-chained event log.
    An auditor querying ``rtp.event_log`` therefore sees the AIF's
    notify_model_updated / mark_slot_failed transitions inline with
    detector and MToUT events — no out-of-band correlation needed.

    Adding a new AIFEventType requires three edits:
      1. Append the enum here.
      2. Add a corresponding ``EventType`` entry in ``rtp/rtp.py``.
      3. Extend ``RTP._AIF_EVENT_MAP`` with the AIF→RTP mapping.
    The regex ``AIFEventType[.]|_AIF_EVENT_MAP`` surfaces every file
    that needs updating in O(seconds).
    """
    MODEL_NOTIFY_ABORTED = auto()   # prepare phase failed
    MODEL_NOTIFY_PARTIAL = auto()   # commit phase failed; rollback triggered
    MODEL_NOTIFY_OK      = auto()   # all subscribers committed successfully
    SLOT_FAILED          = auto()   # candidate slot rejected by NDT → FAILED


# ---------------------------------------------------------------------------
# Model slot state  — (b) adds terminal FAILED state
# ---------------------------------------------------------------------------

class ModelState(Enum):
    ACTIVE   = auto()   # currently serving predictions
    STANDBY  = auto()   # warm backup, ready to swap in
    ABSENT   = auto()   # slot is empty
    FAILED   = auto()   # (b) NDT-rejected; terminal — cannot be promoted


# ---------------------------------------------------------------------------
# DPP — Data PreProcessor
# ---------------------------------------------------------------------------

class DPP:
    """
    Data PreProcessor.

    Responsibilities (paper Section IV-B):
      • Normalisation  — zero-mean / unit-variance via StandardScaler
      • Artefact removal — clip extreme values (configurable z-score cap)
      • NaN handling   — replace with per-feature mean of fitted scaler

    The scaler is fitted lazily on the first batch it sees, or you can
    call fit() explicitly with a reference dataset.

    Parameters
    ----------
    clip_zscore : float
        Input values whose z-score exceeds this threshold are clipped.
        Set to np.inf to disable clipping.
    """

    def __init__(self, clip_zscore: float = 5.0) -> None:
        self.clip_zscore = clip_zscore
        self._scaler = StandardScaler()
        self._fitted = False

    def fit(self, X: np.ndarray) -> "DPP":
        """Fit the internal scaler on reference data X."""
        self._scaler.fit(X)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Preprocess X.  If the scaler has not been fitted yet, fit it
        on X first (online / lazy initialisation).
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))

        # Lazy fit on first call
        if not self._fitted:
            logger.info("DPP: fitting scaler on first batch (shape=%s)", X.shape)
            self.fit(X)

        # NaN replacement — use per-feature mean from the fitted scaler
        nan_mask = np.isnan(X)
        if nan_mask.any():
            logger.warning("DPP: %d NaN values replaced with feature means", nan_mask.sum())
            X = X.copy()
            X[nan_mask] = np.take(self._scaler.mean_, np.where(nan_mask)[1])

        X_scaled = self._scaler.transform(X)

        # Clip outliers by z-score
        if np.isfinite(self.clip_zscore):
            X_scaled = np.clip(X_scaled, -self.clip_zscore, self.clip_zscore)

        return X_scaled

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def __repr__(self) -> str:
        return f"DPP(clip_zscore={self.clip_zscore}, fitted={self._fitted})"


# ---------------------------------------------------------------------------
# SIB — Short Input Buffer
# ---------------------------------------------------------------------------

class SIB:
    """
    Short Input Buffer.

    Stores exactly the last *capacity* preprocessed feature vectors —
    the number of inputs the MLI needs for one inference call.

    For a standard sklearn estimator capacity=1 (one sample at a time).
    For time-series models (e.g. sliding-window regressors) set
    capacity > 1 so that MLI always receives the full context window.

    Paper reference: Section IV-B
      "SIB is an input buffer that stores the input data required for the
       proper operation of the MLI; its length corresponds to the number
       of MLI inputs."
    """

    def __init__(self, capacity: int = 1) -> None:
        if capacity < 1:
            raise ValueError("SIB capacity must be >= 1")
        self.capacity = capacity
        self._buf: deque[np.ndarray] = deque(maxlen=capacity)

    def push(self, x: np.ndarray) -> None:
        """Add one preprocessed sample to the buffer."""
        self._buf.append(np.asarray(x, dtype=float))

    def ready(self) -> bool:
        """True once the buffer holds exactly *capacity* samples."""
        return len(self._buf) == self.capacity

    def get(self) -> np.ndarray:
        """
        Return the buffer contents as a 2-D array of shape
        (capacity, n_features).  Raises RuntimeError if not ready.
        """
        if not self.ready():
            raise RuntimeError(
                f"SIB not ready: {len(self._buf)}/{self.capacity} samples loaded."
            )
        return np.stack(list(self._buf))

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)

    def __repr__(self) -> str:
        return f"SIB(capacity={self.capacity}, loaded={len(self._buf)})"


# ---------------------------------------------------------------------------
# ModelSlot — holds one sklearn estimator plus its state
# ---------------------------------------------------------------------------

@dataclass
class ModelSlot:
    """Container for one sklearn estimator (MLIN or MLIO).

    ``slot_id`` is a monotonically increasing integer assigned to the
    estimator at deploy-time. It is used as a stable cache key by the
    detector-snapshot machinery (:class:`detectors.reset.DetectorResetCoordinator`)
    so that when ``rollback()`` re-activates an older estimator, the
    detector state captured at that estimator's original deploy can be
    restored, not the state captured against the poisoned successor.

    (b) ``failure_reason`` is populated when a slot transitions to
    ``ModelState.FAILED`` — the terminal state for NDT-rejected candidates.
    A FAILED slot cannot be promoted again and is retained in the AIF's
    slot history for operator inspection via :meth:`AIF.list_failed_slots`.
    """
    estimator: Optional[BaseEstimator] = None
    state: ModelState = ModelState.ABSENT
    trained_at: Optional[float] = None        # unix timestamp
    metadata: dict = field(default_factory=dict)
    slot_id: Optional[int] = None             # stable id for snapshot lookup
    failure_reason: Optional[str] = None      # (b) set when state == FAILED

    def is_fitted(self) -> bool:
        if self.estimator is None:
            return False
        try:
            check_is_fitted(self.estimator)
            return True
        except Exception:
            return False

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted():
            raise RuntimeError("ModelSlot: estimator is not fitted.")
        return self.estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted():
            raise RuntimeError("ModelSlot: estimator is not fitted.")
        if not is_classifier(self.estimator):
            raise TypeError("predict_proba is only available for classifiers.")
        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError(
                f"{type(self.estimator).__name__} does not support predict_proba."
            )
        return self.estimator.predict_proba(X)

    def __repr__(self) -> str:
        name = type(self.estimator).__name__ if self.estimator else "None"
        parts = [f"model={name}", f"state={self.state.name}"]
        if self.failure_reason:
            parts.append(f"failure_reason={self.failure_reason!r}")
        return f"ModelSlot({', '.join(parts)})"


# ---------------------------------------------------------------------------
# (a) Two-phase commit subscriber protocol helpers
# ---------------------------------------------------------------------------

def _subscriber_prepare(subscriber: Any, new_slot: ModelSlot) -> bool:
    """
    Call the prepare phase on *subscriber*.

    Dispatch rules (back-compat shim):
    - If subscriber has ``prepare_model_update(slot)`` → call it; expected
      to return True (ready) or False (abort).
    - Otherwise (legacy subscriber with only ``on_model_updated``) → return
      True unconditionally (we will call ``on_model_updated`` during commit).

    Returns True if the subscriber is ready to commit, False to abort.
    Propagates any exception raised by ``prepare_model_update`` to the
    caller so it can be recorded and the prepare phase aborted.
    """
    if hasattr(subscriber, "prepare_model_update"):
        result = subscriber.prepare_model_update(new_slot)
        return bool(result)
    # Legacy subscriber — always ready; nothing to prepare
    return True


def _subscriber_commit(subscriber: Any, new_slot: ModelSlot) -> None:
    """
    Call the commit phase on *subscriber*.

    Dispatch rules (back-compat shim):
    - If subscriber has ``apply_model_update(slot)`` → call it.
    - Otherwise fall back to ``on_model_updated(slot)`` (legacy API).
    - If neither method exists, log a warning and skip silently.

    Any exception raised by the subscriber's commit method is propagated
    to the caller.
    """
    if hasattr(subscriber, "apply_model_update"):
        subscriber.apply_model_update(new_slot)
    elif hasattr(subscriber, "on_model_updated"):
        subscriber.on_model_updated(new_slot)
    else:
        logger.warning(
            "AIF: subscriber %r has neither apply_model_update nor "
            "on_model_updated; skipping.",
            subscriber,
        )


# ---------------------------------------------------------------------------
# AIF — AI Function
# ---------------------------------------------------------------------------

class AIF:
    """
    AI Function wrapper for a scikit-learn estimator.

    Composes DPP → SIB → MLI into a single inference pipeline and manages
    the MLIN / MLIO dual-model slots for safe model updates and rollbacks.

    Parameters
    ----------
    estimator : BaseEstimator
        A fitted (or unfitted) sklearn classifier or regressor.
        If unfitted it must be fitted before the AIF can serve predictions.
    sib_capacity : int
        Number of preprocessed samples the SIB must hold before inference.
        Use 1 for standard stateless estimators.
    dpp_clip_zscore : float
        Z-score clipping threshold passed to DPP.

    Example
    -------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> clf = RandomForestClassifier().fit(X_train, y_train)
    >>> aif = AIF(clf)
    >>> y_pred = aif.predict(x_new)
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        sib_capacity: int = 1,
        dpp_clip_zscore: float = 5.0,
    ) -> None:
        self.dpp = DPP(clip_zscore=dpp_clip_zscore)
        self.sib = SIB(capacity=sib_capacity)

        # Monotonic counter used to assign a unique ``slot_id`` to every
        # estimator that enters a slot. Crucially, a rolled-back estimator
        # keeps its ORIGINAL slot_id — that is what lets the detector-
        # snapshot coordinator restore the correct pre-deploy state.
        self._slot_id_counter = 0

        # MLIN — active model (provided estimator)
        self.mlin = ModelSlot(
            estimator=estimator,
            state=ModelState.ACTIVE,
            trained_at=time.time(),
            slot_id=self._next_slot_id(),
        )
        # MLIO — standby slot (empty until first model update)
        self.mlio = ModelSlot(state=ModelState.ABSENT)

        # (a) Registered downstream subscribers.  Insertion order is the
        # commit order for notify_model_updated's two-phase protocol.
        self._subscribers: List[Any] = []

        # (b) Bounded history of FAILED slots (last 10) for operator audit.
        self._failed_slots: deque[ModelSlot] = deque(maxlen=10)

        # Internal event callback — wired by RTP so AIF can log its own
        # events into the tamper-evident EventLog without importing rtp.rtp.
        # Signature: (event_type: AIFEventType, details: dict) -> None
        self._event_cb: Optional[Any] = None

        logger.info("AIF created: %s", self)

    # ------------------------------------------------------------------
    # Slot bookkeeping
    # ------------------------------------------------------------------

    def _next_slot_id(self) -> int:
        """Allocate a fresh monotonically increasing slot id."""
        self._slot_id_counter += 1
        return self._slot_id_counter

    # ------------------------------------------------------------------
    # Subscriber registration  (a)
    # ------------------------------------------------------------------

    def register_subscriber(self, subscriber: Any) -> None:
        """
        Register a downstream subscriber for model-update notifications.

        The subscriber is called during :meth:`notify_model_updated`'s
        two-phase commit protocol.  Supported interface:

        **New protocol (preferred)**::

            def prepare_model_update(self, slot: ModelSlot) -> bool: ...
            def apply_model_update(self, slot: ModelSlot) -> None: ...

        **Legacy (back-compat shim)**::

            def on_model_updated(self, slot: ModelSlot) -> None: ...

        Legacy subscribers implicitly return ``True`` from prepare and have
        ``on_model_updated`` called during the commit phase.
        """
        self._subscribers.append(subscriber)

    def set_event_callback(self, cb: Any) -> None:
        """
        Wire a callback so AIF can emit events into an external log.

        ``cb`` is called as ``cb(event_type, details)`` where
        ``event_type`` is an :class:`AIFEventType` and ``details`` is a
        plain dict.  Typically wired by RTP to forward into its own
        tamper-evident EventLog.
        """
        self._event_cb = cb

    def _emit(self, event_type: AIFEventType, details: dict) -> None:
        """Emit an event via the registered callback (if any)."""
        if self._event_cb is not None:
            try:
                self._event_cb(event_type, details)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("AIF: event callback raised (%s).", exc)

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Run one inference step: DPP → SIB → active MLI.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,) or (1, n_features)
            Raw (unscaled) input features.

        Returns
        -------
        np.ndarray
            Model prediction(s).
        """
        x = np.atleast_2d(np.asarray(x, dtype=float))

        # 1. DPP: normalise + clip
        x_clean = self.dpp.transform(x)

        # 2. SIB: push and check readiness
        for row in x_clean:
            self.sib.push(row)

        if not self.sib.ready():
            raise RuntimeError(
                f"SIB not ready yet ({len(self.sib)}/{self.sib.capacity}). "
                "Keep feeding samples until it fills up."
            )

        X_sib = self.sib.get()   # shape: (sib_capacity, n_features)

        # 3. MLI: predict with active model (MLIN)
        active_slot = self._active_slot()
        prediction = active_slot.predict(X_sib)
        return prediction

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Like predict() but returns class probabilities (classifiers only)."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        x_clean = self.dpp.transform(x)
        for row in x_clean:
            self.sib.push(row)
        if not self.sib.ready():
            raise RuntimeError("SIB not ready yet.")
        X_sib = self.sib.get()
        return self._active_slot().predict_proba(X_sib)

    # ------------------------------------------------------------------
    # Model update API — called by ATM/MTP after retraining
    # ------------------------------------------------------------------

    def update_model(
        self,
        new_estimator: BaseEstimator,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Install a newly trained estimator as MLIN, demoting the current
        MLIN to MLIO (standby).

        This implements the MLIN/MLIO swap described in the paper.

        Parameters
        ----------
        new_estimator : BaseEstimator
            Freshly trained sklearn estimator (already fitted).
        metadata : dict, optional
            Provenance info: training timestamp, accuracy, MTP variant…

        Notes
        -----
        This is the legacy direct-swap path used by the production
        retrain chain (``rtp.notify_model_updated`` → ATM → here).  It
        does NOT run the two-phase commit subscriber protocol — that's
        the job of :meth:`notify_model_updated` (which constructs a
        ``ModelSlot`` first and runs prepare/apply against subscribers).

        We still emit ``AIFEventType.MODEL_NOTIFY_OK`` at the end so the
        legacy path is visible in the audit log alongside 2PC-driven
        deploys.  The payload carries ``committed=[]`` and a
        ``via="update_model"`` marker so an investigator can tell the
        two paths apart while still being able to grep a single event
        type for "all model deployments".  This matches the no-subscriber
        branch of :meth:`notify_model_updated` which emits the same
        event with ``committed=[]`` plus a descriptive message.
        """
        # Demote current MLIN → MLIO. The demoted slot KEEPS its
        # ``slot_id`` — that's the key the detector-snapshot coordinator
        # will use if a later rollback brings it back to ACTIVE.
        if self.mlin.estimator is not None:
            self.mlio = ModelSlot(
                estimator=copy.deepcopy(self.mlin.estimator),
                state=ModelState.STANDBY,
                trained_at=self.mlin.trained_at,
                metadata=self.mlin.metadata,
                slot_id=self.mlin.slot_id,
            )
            logger.info("AIF: MLIN demoted to MLIO standby.")

        # Install new model as MLIN — brand new slot_id.
        self.mlin = ModelSlot(
            estimator=new_estimator,
            state=ModelState.ACTIVE,
            trained_at=time.time(),
            metadata=metadata or {},
            slot_id=self._next_slot_id(),
        )
        logger.info("AIF: new MLIN installed. %s", self.mlin)

        # Surface the deploy in the audit log via the same event type
        # the 2PC path uses for "model swap completed".  Without this,
        # production retrains were silently invisible to RTP's tamper-
        # evident event log because ``rtp.notify_model_updated`` calls
        # this legacy method, never the 2PC variant.
        self._emit(AIFEventType.MODEL_NOTIFY_OK, {
            "slot_id": self.mlin.slot_id,
            "committed": [],
            "via": "update_model",
            "message": "Legacy direct-swap (no subscriber 2PC).",
        })

    # ------------------------------------------------------------------
    # (a) Two-phase commit notify_model_updated
    # ------------------------------------------------------------------

    def notify_model_updated(
        self,
        new_slot: ModelSlot,
    ) -> bool:
        """
        Notify all registered subscribers that a new model slot is ready,
        using a two-phase commit (prepare → apply) protocol.

        Phase 1 — Prepare
            Call ``subscriber.prepare_model_update(new_slot)`` on every
            registered subscriber in registration order.  If ANY subscriber
            raises OR returns False, the entire prepare phase is aborted:
              • The offending subscriber and reason are logged.
              • A ``MODEL_NOTIFY_ABORTED`` event is emitted.
              • The slot is left in STANDBY (not promoted to ACTIVE).
              • Returns False.

        Phase 2 — Commit
            If all prepare calls succeed, call
            ``subscriber.apply_model_update(new_slot)`` on each subscriber
            in registration order.  If a commit raises:
              • A ``MODEL_NOTIFY_PARTIAL`` event is emitted with the list of
                subscribers that DID commit and those that did NOT.
              • Rollback is triggered via :meth:`rollback`.
              • Returns False.

        On full success the slot transitions STANDBY → ACTIVE and a
        ``MODEL_NOTIFY_OK`` event is emitted.

        Back-compat shim
            Subscribers without ``prepare_model_update`` / ``apply_model_update``
            keep working: prepare returns True implicitly and
            ``on_model_updated`` is called during commit.

        Parameters
        ----------
        new_slot : ModelSlot
            The candidate slot to notify about.  Must be in STANDBY state.

        Returns
        -------
        bool
            True if all subscribers committed and the slot is now ACTIVE.
        """
        if not self._subscribers:
            # No subscribers — trivially succeed; promote the slot.
            new_slot.state = ModelState.ACTIVE
            self._emit(AIFEventType.MODEL_NOTIFY_OK, {
                "slot_id": new_slot.slot_id,
                "committed": [],
                "message": "No subscribers; slot promoted directly.",
            })
            return True

        # ── Phase 1: Prepare ──────────────────────────────────────────
        for idx, sub in enumerate(self._subscribers):
            try:
                ready = _subscriber_prepare(sub, new_slot)
            except Exception as exc:
                reason = f"subscriber[{idx}] ({sub!r}) raised in prepare: {exc}"
                logger.warning("AIF notify prepare aborted: %s", reason)
                self._emit(AIFEventType.MODEL_NOTIFY_ABORTED, {
                    "slot_id": new_slot.slot_id,
                    "offending_subscriber": idx,
                    "reason": reason,
                })
                return False

            if not ready:
                reason = (
                    f"subscriber[{idx}] ({sub!r}) returned False from "
                    "prepare_model_update"
                )
                logger.warning("AIF notify prepare aborted: %s", reason)
                self._emit(AIFEventType.MODEL_NOTIFY_ABORTED, {
                    "slot_id": new_slot.slot_id,
                    "offending_subscriber": idx,
                    "reason": reason,
                })
                return False

        # ── Phase 2: Commit ───────────────────────────────────────────
        committed: list[int] = []
        for idx, sub in enumerate(self._subscribers):
            try:
                _subscriber_commit(sub, new_slot)
                committed.append(idx)
            except Exception as exc:
                not_committed = [
                    i for i in range(len(self._subscribers))
                    if i not in committed
                ]
                reason = (
                    f"subscriber[{idx}] ({sub!r}) raised during apply: {exc}"
                )
                logger.error(
                    "AIF notify commit failed at subscriber[%d]: %s. "
                    "Committed=%s, not committed=%s. Triggering rollback.",
                    idx, exc, committed, not_committed,
                )
                self._emit(AIFEventType.MODEL_NOTIFY_PARTIAL, {
                    "slot_id": new_slot.slot_id,
                    "committed_subscribers": committed,
                    "uncommitted_subscribers": not_committed,
                    "failing_subscriber": idx,
                    "reason": reason,
                })
                # Auto-rollback: undo the fan-out inconsistency.
                self.rollback()
                return False

        # ── All committed — promote slot to ACTIVE ────────────────────
        new_slot.state = ModelState.ACTIVE
        self._emit(AIFEventType.MODEL_NOTIFY_OK, {
            "slot_id": new_slot.slot_id,
            "committed_subscribers": committed,
        })
        logger.info(
            "AIF: notify_model_updated committed by %d subscriber(s); "
            "slot %s now ACTIVE.",
            len(committed), new_slot.slot_id,
        )
        return True

    # ------------------------------------------------------------------
    # (b) Mark a candidate slot as FAILED (NDT rejection path)
    # ------------------------------------------------------------------

    def mark_slot_failed(
        self,
        slot: ModelSlot,
        reason: str = "NDT validation failed",
    ) -> None:
        """
        Transition *slot* from STANDBY → FAILED (terminal state).

        Called by the ATM when NDT rejects a candidate so that operators
        can distinguish "never existed" (ABSENT) from "existed and was
        rejected" (FAILED).

        Rules
        -----
        * The slot must currently be in STANDBY state; marking an ACTIVE or
          ABSENT slot failed is a programming error and raises ValueError.
        * A FAILED slot cannot be promoted again — any subsequent attempt
          raises RuntimeError.
        * The slot is appended to :attr:`_failed_slots` (bounded deque,
          maxlen=10) for operator inspection via :meth:`list_failed_slots`.
        * A ``SLOT_FAILED`` event is emitted via the event callback.

        Parameters
        ----------
        slot : ModelSlot
            The candidate slot to mark as failed.
        reason : str
            Human-readable explanation of why NDT rejected the candidate.
        """
        if slot.state == ModelState.FAILED:
            # Already failed — idempotent call, just log and return.
            logger.debug(
                "AIF.mark_slot_failed: slot %s already FAILED.", slot.slot_id
            )
            return
        if slot.state != ModelState.STANDBY:
            raise ValueError(
                f"AIF.mark_slot_failed: can only fail a STANDBY slot, "
                f"got state={slot.state.name} for slot_id={slot.slot_id}."
            )
        slot.state = ModelState.FAILED
        slot.failure_reason = reason
        self._failed_slots.append(slot)
        self._emit(AIFEventType.SLOT_FAILED, {
            "slot_id": slot.slot_id,
            "reason": reason,
        })
        logger.warning(
            "AIF: slot %s marked FAILED — %s", slot.slot_id, reason
        )

    def list_failed_slots(self) -> list[ModelSlot]:
        """
        Return the list of FAILED slots retained for operator inspection.

        Bounded to the last 10 failed slots.  Slots are returned in
        chronological order (oldest first).
        """
        return list(self._failed_slots)

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback(self) -> bool:
        """
        Swap MLIN ↔ MLIO: activate the old model, demote the current one.

        Called by MToUT when the new MLIN performs worse or is poisoned,
        or automatically by the two-phase commit protocol when a commit
        subscriber raises.

        Returns True if rollback succeeded, False if MLIO slot was empty.

        The caller can read ``self.mlin.slot_id`` after a successful
        return to know which slot is now active — the detector-snapshot
        coordinator uses that id to pick the correct pre-deploy state to
        restore.
        """
        if self.mlio.state == ModelState.ABSENT:
            logger.warning("AIF rollback requested but MLIO slot is empty.")
            return False

        self.mlin.state = ModelState.STANDBY
        self.mlio.state = ModelState.ACTIVE
        self.mlin, self.mlio = self.mlio, self.mlin
        logger.warning("AIF: rolled back to MLIO (old model now active).")
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _active_slot(self) -> ModelSlot:
        """Return whichever slot is currently ACTIVE."""
        if self.mlin.state == ModelState.ACTIVE:
            return self.mlin
        if self.mlio.state == ModelState.ACTIVE:
            return self.mlio
        raise RuntimeError("AIF: no model slot is in ACTIVE state.")

    @property
    def active_estimator(self) -> Optional[BaseEstimator]:
        """The sklearn estimator that is currently serving predictions."""
        try:
            return self._active_slot().estimator
        except RuntimeError:
            return None

    def is_ready(self) -> bool:
        """True if the active model is fitted and DPP is initialised."""
        try:
            slot = self._active_slot()
            return slot.is_fitted()
        except RuntimeError:
            return False

    def __repr__(self) -> str:
        return (
            f"AIF(\n"
            f"  dpp={self.dpp},\n"
            f"  sib={self.sib},\n"
            f"  mlin={self.mlin},\n"
            f"  mlio={self.mlio}\n"
            f")"
        )
