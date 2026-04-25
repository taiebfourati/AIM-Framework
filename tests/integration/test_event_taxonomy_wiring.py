"""
Tier-2 integration tests — event taxonomy wiring (HIGH #8 + #9).

Verifies that the new EventType entries added in HIGH #8/#9 actually
fire end-to-end through a fully wired pipeline:

* AIF two-phase commit → RTP event log:
    - MODEL_NOTIFY_OK  on every successful subscriber commit
* RTP notify_model_updated detector-baseline ops:
    - REFERENCE_REFIT (DDD)
    - REFERENCE_REFIT (DPD)
    - LOB_RESTAMPED   (with ok=True on the happy path)
* MToUT cooldown gate:
    - MTOUT_SUPPRESSED  when a fired-but-throttled MToUT lands

The test that drives slot FAILED / MODEL_NOTIFY_PARTIAL /
MODEL_NOTIFY_ABORTED is in ``test_aif_atomicity_and_states.py`` —
this file deliberately scopes to the wiring layer (i.e. that the
events arrive at the RTP's tamper-evident hash chain), not the
underlying state-machine semantics, which the atomicity test owns.
"""
from __future__ import annotations

import numpy as np

from rtp.rtp import EventType, RTPConfig

from ._harness import build_pipeline, make_classifier_corpus, stream


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _event_types(pipeline) -> list[EventType]:
    """Return the ordered list of EventType values in the RTP event log."""
    return [e.event_type for e in pipeline.rtp.event_log]


def _events_of(pipeline, et: EventType) -> list:
    return [e for e in pipeline.rtp.event_log if e.event_type == et]


# ---------------------------------------------------------------------------
# Test 1 — successful retrain emits the full new-event tuple
# ---------------------------------------------------------------------------

def test_successful_retrain_emits_aif_and_rtp_events() -> None:
    """
    Drive enough drifted samples to trigger ATM retrain.  After the
    retrain fires AND the AIF promotes the candidate slot, the RTP
    event log must contain ALL of:

      * MODEL_NOTIFY_OK   (from AIFEventType via _on_aif_event)
      * REFERENCE_REFIT × 2  (one for DDD, one for DPD)
      * LOB_RESTAMPED     (with ok=True)
      * MODEL_UPDATED     (the legacy event, still present)
    """
    config = RTPConfig(
        cdd_task="classifier",
        check_interval=50,
        mtout_cooldown_steps=50,
        buffer_maxlen=5000,
    )
    pipeline = build_pipeline(task="classifier", n_features=4, config=config)

    # Drive a strong drift so MToUT fires and ATM retrains.
    rng = np.random.default_rng(11)
    X_drifted, y_drifted = make_classifier_corpus(
        n=600, d=4, seed=11, shift=2.0,
    )
    stream(pipeline, X_drifted, y_true=y_drifted)

    types = _event_types(pipeline)
    # Guard pre-condition: at least one MODEL_UPDATED happened (i.e. the
    # ATM actually retrained — without this the rest of the assertions
    # would be testing the wrong scenario).
    assert EventType.MODEL_UPDATED in types, (
        f"Pre-condition failed: no MODEL_UPDATED in event log. "
        f"Types seen: {sorted({t.name for t in types})}"
    )

    # ── AIF wiring: MODEL_NOTIFY_OK lands in the RTP log ─────────
    ok_events = _events_of(pipeline, EventType.MODEL_NOTIFY_OK)
    assert ok_events, (
        "Expected at least one MODEL_NOTIFY_OK event from AIF after "
        "successful retrain — the AIF→RTP event subscription is broken."
    )

    # ── REFERENCE_REFIT for both DDD and DPD ─────────────────────
    refit_events = _events_of(pipeline, EventType.REFERENCE_REFIT)
    assert refit_events, "no REFERENCE_REFIT events"
    detectors_refit = {e.details["detector"] for e in refit_events}
    assert "DDD" in detectors_refit, (
        f"DDD reference refit not emitted: detectors={detectors_refit}"
    )
    assert "DPD" in detectors_refit, (
        f"DPD reference refit not emitted: detectors={detectors_refit}"
    )

    # ── LOB_RESTAMPED with ok=True ───────────────────────────────
    restamp_events = _events_of(pipeline, EventType.LOB_RESTAMPED)
    assert restamp_events, "no LOB_RESTAMPED events"
    assert any(e.details.get("ok") for e in restamp_events), (
        f"All LOB_RESTAMPED events failed: {restamp_events}"
    )


# ---------------------------------------------------------------------------
# Test 2 — cooldown gate emits MTOUT_SUPPRESSED
# ---------------------------------------------------------------------------

def test_mtout_cooldown_emits_suppressed_event() -> None:
    """
    With a long cooldown, the SECOND MToUT-eligible firing within
    cooldown_steps must emit MTOUT_SUPPRESSED rather than silently
    dropping.  The event payload must carry the suppressed reasons
    so an investigator can see what the throttle hid.
    """
    config = RTPConfig(
        cdd_task="classifier",
        check_interval=50,
        # Set cooldown longer than the test stream so the cooldown
        # gate is guaranteed to fire on the SECOND-and-subsequent
        # would-be MToUTs.
        mtout_cooldown_steps=10_000,
        buffer_maxlen=5000,
    )
    pipeline = build_pipeline(task="classifier", n_features=4, config=config)

    # Bypass ATM retrain — we want the SAME drifted regime to keep
    # firing detectors so the cooldown gate is exercised repeatedly.
    pipeline.rtp._on_mtout = lambda sig: pipeline.events.append(("mtout", sig))

    rng = np.random.default_rng(22)
    X_drifted, y_drifted = make_classifier_corpus(
        n=500, d=4, seed=22, shift=2.5,
    )
    stream(pipeline, X_drifted, y_true=y_drifted)

    suppressed = _events_of(pipeline, EventType.MTOUT_SUPPRESSED)
    assert suppressed, (
        "expected MTOUT_SUPPRESSED events from the long-cooldown gate; "
        "instead the suppressed firings are silently invisible to the "
        "audit log — operators cannot tell 'detector saw nothing' from "
        "'detector fired but throttled'."
    )

    # Spec payload: {step, reasons, steps_since_last, cooldown_steps,
    # slow_poisoning_only}
    sample = suppressed[0]
    for key in (
        "step", "reasons", "steps_since_last",
        "cooldown_steps", "slow_poisoning_only",
    ):
        assert key in sample.details, (
            f"MTOUT_SUPPRESSED payload missing '{key}': {sample.details}"
        )
    assert sample.details["cooldown_steps"] == 10_000
    assert isinstance(sample.details["reasons"], list)
    assert sample.details["reasons"], (
        "reasons list empty in MTOUT_SUPPRESSED — the suppressed "
        "MToUT would have had at least one reason"
    )


# ---------------------------------------------------------------------------
# Test 3 — chain integrity preserved across new events
# ---------------------------------------------------------------------------

def test_event_log_chain_remains_verifiable_with_new_events() -> None:
    """
    The new events go through the SAME tamper-evident hash chain as
    every other event.  Verify the chain ends clean after a typical
    retrain / cooldown / restamp sequence so the new emit sites did
    not accidentally bypass the EventLog.append wrapper.
    """
    config = RTPConfig(
        cdd_task="classifier",
        check_interval=50,
        mtout_cooldown_steps=50,
        buffer_maxlen=5000,
    )
    pipeline = build_pipeline(task="classifier", n_features=4, config=config)

    rng = np.random.default_rng(33)
    X_drifted, y_drifted = make_classifier_corpus(
        n=400, d=4, seed=33, shift=2.0,
    )
    stream(pipeline, X_drifted, y_true=y_drifted)

    ok, err = pipeline.rtp.verify_event_log()
    assert ok, (
        f"Event log chain invalid after retrain run: {err}. "
        f"One of the new emit sites likely bypassed EventLog.append "
        f"(e.g. wrote to event_log directly)."
    )
