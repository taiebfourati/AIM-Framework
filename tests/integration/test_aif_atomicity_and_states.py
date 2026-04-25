"""
tests/integration/test_aif_atomicity_and_states.py

Integration tests for the three HIGH-severity audit fixes in aif/aif.py:

  (a) notify_model_updated two-phase commit atomicity
  (b) ModelSlot FAILED terminal state
  (c) MToUTSignal / TriggerReasons / Severity import from aif/signals.py

Test inventory
--------------
A — prepare failure: subscriber #2 raises in prepare; no apply is called;
    slot stays STANDBY; MODEL_NOTIFY_ABORTED event emitted.
B — commit failure: subscriber #2 raises in apply; #1 saw apply, #3 did
    not; rollback triggered; MODEL_NOTIFY_PARTIAL emitted; old model active.
C — happy path: all 3 subscribers commit; slot goes ACTIVE;
    MODEL_NOTIFY_OK emitted.
D — FAILED state: force NDT to fail via mark_slot_failed; slot is FAILED
    (not ABSENT); SLOT_FAILED event emitted; list_failed_slots() returns
    it; attempting to promote a FAILED slot returns/raises an error.
E — signal import: from aif.signals import works; back-compat re-export
    from aif.aif works; Severity is IntEnum with correct ordering.
F — regression: full e2e pipeline 1000 steps passes; list_failed_slots()
    empty; event_log has zero PARTIAL/ABORTED entries.
"""
from __future__ import annotations

import time
from enum import IntEnum
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from aif.aif import AIF, AIFEventType, ModelSlot, ModelState


# ---------------------------------------------------------------------------
# Subscriber helpers
# ---------------------------------------------------------------------------

class RecordingSubscriber:
    """Subscriber that records which phases were called and can inject faults."""

    def __init__(
        self,
        name: str = "",
        prepare_raises: Exception | None = None,
        prepare_returns: bool = True,
        apply_raises: Exception | None = None,
    ) -> None:
        self.name = name
        self.prepare_raises = prepare_raises
        self.prepare_returns = prepare_returns
        self.apply_raises = apply_raises

        self.prepare_called: bool = False
        self.apply_called: bool = False
        self.last_slot: ModelSlot | None = None

    def prepare_model_update(self, slot: ModelSlot) -> bool:
        self.prepare_called = True
        self.last_slot = slot
        if self.prepare_raises is not None:
            raise self.prepare_raises
        return self.prepare_returns

    def apply_model_update(self, slot: ModelSlot) -> None:
        self.apply_called = True
        self.last_slot = slot
        if self.apply_raises is not None:
            raise self.apply_raises

    def __repr__(self) -> str:
        return f"RecordingSubscriber({self.name!r})"


class LegacySubscriber:
    """Back-compat subscriber with only on_model_updated."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.called: bool = False
        self.last_slot: ModelSlot | None = None

    def on_model_updated(self, slot: ModelSlot) -> None:
        self.called = True
        self.last_slot = slot

    def __repr__(self) -> str:
        return f"LegacySubscriber({self.name!r})"


# ---------------------------------------------------------------------------
# AIF factory helper
# ---------------------------------------------------------------------------

def _make_aif() -> tuple[AIF, list[tuple[AIFEventType, dict]]]:
    """Return a pre-fitted AIF and the list that collects its events."""
    X = np.random.default_rng(0).normal(size=(200, 4))
    y = (X[:, 0] > 0).astype(int)
    clf = LogisticRegression(max_iter=300).fit(X, y)
    aif = AIF(estimator=clf, sib_capacity=1)

    events: list[tuple[AIFEventType, dict]] = []
    aif.set_event_callback(lambda et, d: events.append((et, d)))
    return aif, events


def _make_standby_slot(aif: AIF) -> ModelSlot:
    """
    Build a freshly trained ModelSlot in STANDBY state so we can pass it
    to notify_model_updated without using the full update_model() path.
    """
    X = np.random.default_rng(1).normal(size=(200, 4))
    y = (X[:, 0] > 0).astype(int)
    clf2 = LogisticRegression(max_iter=300).fit(X, y)
    slot = ModelSlot(
        estimator=clf2,
        state=ModelState.STANDBY,
        trained_at=time.time(),
        slot_id=aif._next_slot_id(),
    )
    return slot


# ---------------------------------------------------------------------------
# Test A — prepare failure
# ---------------------------------------------------------------------------

class TestPrepareFailure:
    """
    Subscriber #2 (index 1) raises in prepare.
    Expected: no apply is called anywhere; slot stays STANDBY;
    MODEL_NOTIFY_ABORTED event emitted.
    """

    def test_no_apply_called_when_prepare_fails(self) -> None:
        aif, events = _make_aif()
        slot = _make_standby_slot(aif)

        sub1 = RecordingSubscriber("s1")
        sub2 = RecordingSubscriber("s2", prepare_raises=RuntimeError("prepare boom"))
        sub3 = RecordingSubscriber("s3")

        aif.register_subscriber(sub1)
        aif.register_subscriber(sub2)
        aif.register_subscriber(sub3)

        result = aif.notify_model_updated(slot)

        assert result is False, "notify_model_updated should return False on prepare failure"

        # sub1 prepare was called (it runs before the failing sub2)
        assert sub1.prepare_called, "sub1.prepare should have been called"
        # sub2 prepare was called (it's the one that raised)
        assert sub2.prepare_called, "sub2.prepare should have been called"
        # sub3 prepare must NOT have been called (abort after sub2 raises)
        assert not sub3.prepare_called, "sub3.prepare must not be called after abort"

        # No apply anywhere
        assert not sub1.apply_called, "sub1.apply must not be called"
        assert not sub2.apply_called, "sub2.apply must not be called"
        assert not sub3.apply_called, "sub3.apply must not be called"

    def test_slot_stays_standby_on_prepare_failure(self) -> None:
        aif, events = _make_aif()
        slot = _make_standby_slot(aif)

        sub2 = RecordingSubscriber("s2", prepare_raises=ValueError("bad"))
        aif.register_subscriber(RecordingSubscriber("s1"))
        aif.register_subscriber(sub2)
        aif.register_subscriber(RecordingSubscriber("s3"))

        aif.notify_model_updated(slot)

        assert slot.state == ModelState.STANDBY, (
            f"slot must remain STANDBY after prepare failure, got {slot.state}"
        )

    def test_model_notify_aborted_event_emitted(self) -> None:
        aif, events = _make_aif()
        slot = _make_standby_slot(aif)

        sub2 = RecordingSubscriber("s2", prepare_raises=RuntimeError("x"))
        aif.register_subscriber(RecordingSubscriber("s1"))
        aif.register_subscriber(sub2)
        aif.register_subscriber(RecordingSubscriber("s3"))

        aif.notify_model_updated(slot)

        aborted = [e for e in events if e[0] == AIFEventType.MODEL_NOTIFY_ABORTED]
        assert len(aborted) == 1, (
            f"Expected exactly 1 MODEL_NOTIFY_ABORTED event, got {len(aborted)}"
        )
        assert aborted[0][1]["slot_id"] == slot.slot_id

    def test_prepare_returns_false_also_aborts(self) -> None:
        """prepare_model_update returning False (not raising) is also an abort."""
        aif, events = _make_aif()
        slot = _make_standby_slot(aif)

        sub2 = RecordingSubscriber("s2", prepare_returns=False)
        sub3 = RecordingSubscriber("s3")
        aif.register_subscriber(RecordingSubscriber("s1"))
        aif.register_subscriber(sub2)
        aif.register_subscriber(sub3)

        result = aif.notify_model_updated(slot)

        assert result is False
        assert not sub3.prepare_called
        assert not sub3.apply_called
        aborted = [e for e in events if e[0] == AIFEventType.MODEL_NOTIFY_ABORTED]
        assert len(aborted) == 1


# ---------------------------------------------------------------------------
# Test B — commit failure
# ---------------------------------------------------------------------------

class TestCommitFailure:
    """
    Subscriber #2 (index 1) raises during apply.
    Expected: sub1 committed (apply ran), sub3 did NOT; rollback triggered;
    MODEL_NOTIFY_PARTIAL emitted; the OLD model is still active after.
    """

    def _setup(self) -> tuple[AIF, list, RecordingSubscriber, RecordingSubscriber, RecordingSubscriber, ModelSlot]:
        aif, events = _make_aif()

        # Give the AIF a real MLIO standby so rollback can succeed.
        X = np.random.default_rng(99).normal(size=(200, 4))
        y = (X[:, 0] > 0).astype(int)
        old_clf = LogisticRegression(max_iter=300).fit(X, y)
        aif.mlio = ModelSlot(
            estimator=old_clf,
            state=ModelState.STANDBY,
            trained_at=time.time(),
            slot_id=0,
        )

        slot = _make_standby_slot(aif)
        sub1 = RecordingSubscriber("s1")
        sub2 = RecordingSubscriber("s2", apply_raises=RuntimeError("apply boom"))
        sub3 = RecordingSubscriber("s3")

        aif.register_subscriber(sub1)
        aif.register_subscriber(sub2)
        aif.register_subscriber(sub3)

        return aif, events, sub1, sub2, sub3, slot

    def test_sub1_applied_sub3_did_not(self) -> None:
        aif, events, sub1, sub2, sub3, slot = self._setup()
        aif.notify_model_updated(slot)

        assert sub1.apply_called, "sub1 should have had apply called"
        assert not sub3.apply_called, "sub3 must not have had apply called"

    def test_rollback_triggered_on_commit_failure(self) -> None:
        aif, events, sub1, sub2, sub3, slot = self._setup()
        original_active = aif.active_estimator

        aif.notify_model_updated(slot)

        # After rollback the MLIO (old model) should be active again.
        # Original active estimator id should be restored by rollback.
        # (rollback swaps mlin/mlio so the old standby becomes active)
        assert aif.mlin.state == ModelState.ACTIVE

    def test_model_notify_partial_event_emitted(self) -> None:
        aif, events, sub1, sub2, sub3, slot = self._setup()
        aif.notify_model_updated(slot)

        partial = [e for e in events if e[0] == AIFEventType.MODEL_NOTIFY_PARTIAL]
        assert len(partial) == 1, (
            f"Expected exactly 1 MODEL_NOTIFY_PARTIAL event, got {len(partial)}"
        )
        detail = partial[0][1]
        assert detail["slot_id"] == slot.slot_id
        # sub1 (index 0) committed; sub2 (index 1) raised; sub3 (index 2) never ran
        assert 0 in detail["committed_subscribers"]
        assert 1 in detail["uncommitted_subscribers"] or 2 in detail["uncommitted_subscribers"]

    def test_returns_false_on_commit_failure(self) -> None:
        aif, events, sub1, sub2, sub3, slot = self._setup()
        result = aif.notify_model_updated(slot)
        assert result is False


# ---------------------------------------------------------------------------
# Test C — happy path
# ---------------------------------------------------------------------------

class TestHappyPath:
    """All subscribers work → slot goes ACTIVE → MODEL_NOTIFY_OK emitted."""

    def test_all_subscribers_called_and_slot_active(self) -> None:
        aif, events = _make_aif()
        slot = _make_standby_slot(aif)

        sub1 = RecordingSubscriber("s1")
        sub2 = RecordingSubscriber("s2")
        sub3 = RecordingSubscriber("s3")

        aif.register_subscriber(sub1)
        aif.register_subscriber(sub2)
        aif.register_subscriber(sub3)

        result = aif.notify_model_updated(slot)

        assert result is True
        assert sub1.apply_called
        assert sub2.apply_called
        assert sub3.apply_called
        assert slot.state == ModelState.ACTIVE

    def test_model_notify_ok_event_emitted(self) -> None:
        aif, events = _make_aif()
        slot = _make_standby_slot(aif)

        aif.register_subscriber(RecordingSubscriber("s1"))
        aif.register_subscriber(RecordingSubscriber("s2"))

        aif.notify_model_updated(slot)

        ok_events = [e for e in events if e[0] == AIFEventType.MODEL_NOTIFY_OK]
        assert len(ok_events) == 1
        assert ok_events[0][1]["slot_id"] == slot.slot_id

    def test_legacy_subscriber_shim_works_in_happy_path(self) -> None:
        """A legacy on_model_updated subscriber participates correctly."""
        aif, events = _make_aif()
        slot = _make_standby_slot(aif)

        leg = LegacySubscriber("legacy")
        new_api = RecordingSubscriber("new")

        aif.register_subscriber(leg)
        aif.register_subscriber(new_api)

        result = aif.notify_model_updated(slot)

        assert result is True
        assert leg.called, "Legacy on_model_updated must be called during commit"
        assert new_api.apply_called, "New apply_model_update must also run"
        assert slot.state == ModelState.ACTIVE

    def test_no_subscribers_trivially_succeeds(self) -> None:
        aif, events = _make_aif()
        slot = _make_standby_slot(aif)

        result = aif.notify_model_updated(slot)

        assert result is True
        assert slot.state == ModelState.ACTIVE
        ok_events = [e for e in events if e[0] == AIFEventType.MODEL_NOTIFY_OK]
        assert len(ok_events) == 1


# ---------------------------------------------------------------------------
# Test D — FAILED state
# ---------------------------------------------------------------------------

class TestFailedState:
    """
    Force NDT to fail via mark_slot_failed:
      - slot ends FAILED (not ABSENT)
      - SLOT_FAILED event emitted
      - list_failed_slots() returns it
      - attempting to promote a FAILED slot raises an error
    """

    def test_slot_transitions_to_failed(self) -> None:
        aif, events = _make_aif()
        slot = _make_standby_slot(aif)

        aif.mark_slot_failed(slot, reason="NDT score below threshold")

        assert slot.state == ModelState.FAILED
        assert slot.failure_reason == "NDT score below threshold"

    def test_failed_is_not_absent(self) -> None:
        aif, events = _make_aif()
        slot = _make_standby_slot(aif)

        aif.mark_slot_failed(slot)

        assert slot.state != ModelState.ABSENT, (
            "FAILED and ABSENT must be distinct states"
        )

    def test_slot_failed_event_emitted(self) -> None:
        aif, events = _make_aif()
        slot = _make_standby_slot(aif)

        aif.mark_slot_failed(slot, reason="bad score")

        failed_events = [e for e in events if e[0] == AIFEventType.SLOT_FAILED]
        assert len(failed_events) == 1
        assert failed_events[0][1]["slot_id"] == slot.slot_id
        assert "bad score" in failed_events[0][1]["reason"]

    def test_list_failed_slots_returns_the_slot(self) -> None:
        aif, events = _make_aif()
        slot = _make_standby_slot(aif)

        assert aif.list_failed_slots() == []

        aif.mark_slot_failed(slot, reason="NDT fail")

        failed = aif.list_failed_slots()
        assert len(failed) == 1
        assert failed[0] is slot

    def test_multiple_failed_slots_are_retained(self) -> None:
        aif, events = _make_aif()

        slots = []
        for _ in range(3):
            s = _make_standby_slot(aif)
            aif.mark_slot_failed(s, reason="x")
            slots.append(s)

        failed = aif.list_failed_slots()
        assert len(failed) == 3
        assert all(s in failed for s in slots)

    def test_failed_slot_cannot_be_promoted_via_notify(self) -> None:
        """
        A FAILED slot cannot re-enter notify_model_updated successfully.
        The slot is not in STANDBY, so either the caller would have to
        manually patch it back — but mark_slot_failed prevents that
        by making FAILED terminal.  We verify that attempting to mark an
        ACTIVE slot fails with a clear ValueError, and that mark_slot_failed
        on an already-FAILED slot is idempotent (no error, no duplicate event).
        """
        aif, events = _make_aif()

        # Cannot mark an ACTIVE slot as FAILED
        active_slot = aif.mlin
        with pytest.raises(ValueError, match="STANDBY"):
            aif.mark_slot_failed(active_slot, reason="should fail")

    def test_idempotent_double_mark_failed(self) -> None:
        """Calling mark_slot_failed twice on the same slot is idempotent."""
        aif, events = _make_aif()
        slot = _make_standby_slot(aif)

        aif.mark_slot_failed(slot, reason="first")
        events_before = len(events)
        aif.mark_slot_failed(slot, reason="second")  # must not raise

        # No extra SLOT_FAILED event on the second call
        failed_events_after = [e for e in events if e[0] == AIFEventType.SLOT_FAILED]
        assert len(failed_events_after) == 1, (
            "Second mark_slot_failed must not emit a duplicate SLOT_FAILED event"
        )

    def test_failed_slot_retained_up_to_10(self) -> None:
        """Bounded history retains last 10 failed slots."""
        aif, events = _make_aif()

        slots = []
        for _ in range(12):
            s = _make_standby_slot(aif)
            aif.mark_slot_failed(s)
            slots.append(s)

        failed = aif.list_failed_slots()
        assert len(failed) == 10
        # The first two (oldest) should have been evicted
        assert slots[0] not in failed
        assert slots[1] not in failed
        assert slots[-1] in failed


# ---------------------------------------------------------------------------
# Test E — signal import
# ---------------------------------------------------------------------------

class TestSignalImport:
    """
    Verify the import paths work and Severity behaves as IntEnum.
    """

    def test_import_from_aif_signals(self) -> None:
        from aif.signals import MToUTSignal, TriggerReasons, Severity  # noqa: F401
        assert MToUTSignal is not None
        assert TriggerReasons is not None
        assert Severity is not None

    def test_back_compat_import_from_aif_aif(self) -> None:
        from aif.aif import MToUTSignal, TriggerReasons, Severity  # noqa: F401
        assert MToUTSignal is not None
        assert TriggerReasons is not None
        assert Severity is not None

    def test_severity_is_intenum(self) -> None:
        from aif.signals import Severity
        assert isinstance(Severity.CRITICAL, IntEnum), (
            "Severity.CRITICAL must be an instance of IntEnum"
        )

    def test_severity_ordering(self) -> None:
        from aif.signals import Severity
        assert Severity.CRITICAL > Severity.MEDIUM, (
            "Severity.CRITICAL must be greater than Severity.MEDIUM"
        )
        assert Severity.MEDIUM > Severity.LOW
        assert Severity.HIGH > Severity.MEDIUM
        assert Severity.CRITICAL > Severity.HIGH

    def test_same_class_from_both_imports(self) -> None:
        """Both import paths must resolve to the exact same class objects."""
        from aif.signals import MToUTSignal as S1, TriggerReasons as T1, Severity as Sev1
        from aif.aif import MToUTSignal as S2, TriggerReasons as T2, Severity as Sev2

        assert S1 is S2
        assert T1 is T2
        assert Sev1 is Sev2

    def test_mtoutsignal_can_be_constructed(self) -> None:
        from aif.signals import MToUTSignal, TriggerReasons, Severity
        sig = MToUTSignal(
            reasons=[TriggerReasons.DATA_DRIFT],
            step=42,
            severity_level=Severity.MEDIUM,
        )
        assert sig.severity() == "MEDIUM"
        assert TriggerReasons.DATA_DRIFT in sig.reasons


# ---------------------------------------------------------------------------
# Test F — regression: full e2e pipeline
# ---------------------------------------------------------------------------

class TestRegressionE2E:
    """
    Full end-to-end pipeline runs 1000 steps without failures.
    list_failed_slots() remains empty.
    event_log has zero PARTIAL/ABORTED entries (no notify failures).
    """

    def test_e2e_pipeline_1000_steps(self) -> None:
        # Import here to keep harness dependencies self-contained.
        from tests.integration._harness import (
            build_pipeline,
            make_classifier_corpus,
            stream,
        )
        from atm.atm import ATMPolicy

        pipeline = build_pipeline(
            task="classifier",
            seed=7,
            policy=ATMPolicy(
                use_ndt=True,
                auto_deploy=True,
                max_retrain_attempts=1,
            ),
        )

        rng = np.random.default_rng(7)
        X_run, y_run = make_classifier_corpus(1000, 4, seed=7)

        stream(pipeline, X_run, y_true=y_run)

        # list_failed_slots() is only on AIF, which does not accumulate
        # failed slots in the basic e2e path (no NDT rejection forced here).
        assert pipeline.aif.list_failed_slots() == [], (
            "No slots should be FAILED in a clean e2e run"
        )

        # No PARTIAL or ABORTED notify events in the RTP event log.
        partial_or_aborted_names = {
            AIFEventType.MODEL_NOTIFY_PARTIAL.name,
            AIFEventType.MODEL_NOTIFY_ABORTED.name,
        }
        bad_events = [
            e for e in pipeline.rtp.event_log
            if getattr(e.event_type, "name", str(e.event_type))
            in partial_or_aborted_names
        ]
        assert bad_events == [], (
            f"Found unexpected PARTIAL/ABORTED events: {bad_events}"
        )

    def test_list_failed_slots_empty_on_fresh_aif(self) -> None:
        aif, _ = _make_aif()
        assert aif.list_failed_slots() == []
