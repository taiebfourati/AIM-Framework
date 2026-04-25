"""
Tier-2 integration tests — tamper-evident ``RTP.event_log``
(security-auditor HIGH finding).

The RTP's event log previously was a plain Python ``list`` of
``RTPEvent``.  A compromised subscriber, a buggy detector plug-in or
any other path holding a reference to the RTP could silently mutate,
delete, or re-order entries — erasing evidence of a poisoning incident
or rollback cascade.

The fix wraps the storage in :class:`aif.event_log.EventLog`: every
append builds a hash-chained, HMAC-signed
:class:`aif.audit.ChainedEvent` so retroactive edits are detectable
without a central audit server.  :meth:`rtp.rtp.RTP.verify_event_log`
exposes the tamper check to operators.

These tests exercise each tamper vector against a fully-wired
pipeline built by :func:`tests.integration._harness.build_pipeline`:

* **Test A** — a clean run verifies cleanly.
* **Test B** — mutating a past entry's payload dict fails verification
  at that entry.
* **Test C** — deleting a middle entry breaks the prev-hash chain.
* **Test D** — re-signing a tampered entry without the operator key
  still fails verification (HMAC changes).
* **Test E** — ``event_log.clear()`` refuses in production and the
  dev/test bypass works.
* **Test F** — 100 real events through the pipeline stay verifiable
  and preserve insertion order.
"""
from __future__ import annotations

import numpy as np
import pytest

from aif.audit import ChainedEvent, GENESIS, sign_payload
from aif.event_log import EventLog, EventLogEntry
from rtp.rtp import EventType

from ._harness import build_pipeline, make_classifier_corpus, stream


# ---------------------------------------------------------------------------
# Helpers — drive a small pipeline that emits a handful of events cheaply
# ---------------------------------------------------------------------------

def _small_run(pipeline) -> None:
    """
    Emit a reasonable spread of events without waiting for real drift.

    The harness already triggers SHADOW_REFIT at ``set_reference`` time,
    so every pipeline starts with at least one event.  We pad with an
    operator_request (fires MTOUT) and a synthetic data stream so the
    log is non-trivial (>= 4 entries) and crosses at least one
    ``check_interval`` boundary.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 1.0, size=(200, 4))
    y = (X.sum(axis=1) > 0).astype(int)
    stream(pipeline, X, y)
    pipeline.rtp.operator_request(reason="tamper-test warm-up")


def _pipeline():
    """A cheap classifier pipeline — matches the conventions in other
    integration tests in this directory."""
    return build_pipeline(task="classifier", seed=0)


# ---------------------------------------------------------------------------
# Test A — clean run verifies cleanly
# ---------------------------------------------------------------------------

class TestACleanRunVerifies:
    def test_clean_pipeline_verifies(self) -> None:
        """
        After a normal run, :meth:`RTP.verify_event_log` must return
        ``(True, None)`` — no tampering, no desync between the chain
        and the view mirror.
        """
        pipeline = _pipeline()
        _small_run(pipeline)

        assert len(pipeline.rtp.event_log) > 0, (
            "expected the harness + small run to emit at least one event"
        )
        ok, err = pipeline.rtp.verify_event_log()
        assert ok is True, f"clean-run verify() unexpectedly failed: {err}"
        assert err is None


# ---------------------------------------------------------------------------
# Test B — retroactive payload mutation is detected
# ---------------------------------------------------------------------------

class TestBPayloadMutationDetected:
    def test_mutating_past_payload_fails_verify(self) -> None:
        """
        Bypass the public API and mutate a stored entry's payload dict
        in place.  The stored ``entry_hash`` was computed over the
        original payload, so :meth:`verify` must flag the chain.

        We target index 5 when possible and fall back to the last
        entry when the log is shorter — the contract under test is
        "any retroactive edit is detected", not "exactly index 5".
        """
        pipeline = _pipeline()
        _small_run(pipeline)

        target_idx = min(5, len(pipeline.rtp.event_log._events) - 1)
        assert target_idx >= 0, "no events to tamper with"

        # The "step" key lives inside the hash_payload's ``payload``
        # sub-dict because ``_log_event`` passes ``details`` as the
        # payload and the EventLog wraps it under a ``payload`` key.
        # The important property is that mutating ANY field inside the
        # canonically-hashed envelope invalidates the hash.
        pipeline.rtp.event_log._events[target_idx].payload["step"] = 9999

        ok, err = pipeline.rtp.verify_event_log()
        assert ok is False, (
            "verify() returned True after a payload mutation — the "
            "chain's tamper evidence is not working"
        )
        # Error string should point at the offending index.
        assert err is not None
        assert f"entry {target_idx}" in err, (
            f"err string {err!r} did not mention the tampered "
            f"index {target_idx}"
        )


# ---------------------------------------------------------------------------
# Test C — deleting a middle entry is detected
# ---------------------------------------------------------------------------

class TestCMiddleDeleteDetected:
    def test_deleting_middle_entry_fails_verify(self) -> None:
        """
        An attacker who trims a poisoning event out of the middle of
        the log breaks the prev-hash chain: the next entry's
        ``prev_hash`` no longer matches the preceding entry's
        ``entry_hash``.
        """
        pipeline = _pipeline()
        _small_run(pipeline)

        assert len(pipeline.rtp.event_log) >= 3, (
            "need at least 3 entries to delete a middle one"
        )
        mid = len(pipeline.rtp.event_log._events) // 2
        del pipeline.rtp.event_log._events[mid]

        ok, err = pipeline.rtp.verify_event_log()
        assert ok is False, "deleting a middle entry was not detected"
        assert err is not None
        # Either the index or the prev-hash check must flag it; both
        # legitimately surface the same underlying attack.
        assert ("index" in err) or ("prev_hash" in err) or (
            "mismatch" in err
        ), f"unexpected err string after middle-delete: {err!r}"


# ---------------------------------------------------------------------------
# Test D — re-signing without the key fails
# ---------------------------------------------------------------------------

class TestDResignWithoutKeyFails:
    def test_resign_with_wrong_key_fails(self) -> None:
        """
        An attacker who mutates a payload and re-signs with a key they
        control (but that the log was NOT constructed with) must still
        fail signature verification.
        """
        pipeline = _pipeline()
        _small_run(pipeline)

        target_idx = min(3, len(pipeline.rtp.event_log._events) - 1)
        assert target_idx >= 0
        original = pipeline.rtp.event_log._events[target_idx]

        # Fabricate a tampered envelope with the same index / type but
        # a different payload, sign it with a BOGUS key, and splice
        # it into the chain in place of the original entry.
        bogus_key = b"attacker-chosen-key-xxx"
        tampered_payload = dict(original.payload)
        tampered_payload["step"] = 424242
        tampered_env = ChainedEvent.envelope(
            index=original.index,
            event_type=original.event_type,
            payload=tampered_payload,
        )
        forged_signature = sign_payload(tampered_env, key=bogus_key)
        # Build a new ChainedEvent with the tampered payload, the
        # original prev_hash, and the attacker-signed signature.
        forged = ChainedEvent.make(
            index=original.index,
            event_type=original.event_type,
            payload=tampered_payload,
            prev_hash=original.prev_hash,
            signature=forged_signature,
        )
        pipeline.rtp.event_log._events[target_idx] = forged

        ok, err = pipeline.rtp.verify_event_log()
        assert ok is False, (
            "verify() accepted an entry re-signed with an unknown key "
            "— HMAC validation is broken"
        )
        # Either the signature check or the downstream prev-hash break
        # on subsequent entries (because the tampered entry_hash now
        # differs from the original) flags the chain.
        assert err is not None


# ---------------------------------------------------------------------------
# Test E — clear() is hard-disabled unless explicitly overridden
# ---------------------------------------------------------------------------

class TestEClearHardDisabled:
    def test_clear_raises_permission_error_by_default(self) -> None:
        """
        ``EventLog.clear()`` must raise unconditionally in the normal
        code path — clearing an append-only audit log erases evidence.
        """
        pipeline = _pipeline()
        _small_run(pipeline)
        n_before = len(pipeline.rtp.event_log)
        assert n_before > 0

        with pytest.raises(PermissionError) as excinfo:
            pipeline.rtp.event_log.clear()
        # Operator-visible marker so the error looks unambiguous in
        # production logs.
        assert "!!" in str(excinfo.value)

        # The log must not have been touched.
        assert len(pipeline.rtp.event_log) == n_before
        ok, _ = pipeline.rtp.verify_event_log()
        assert ok is True, (
            "clear() raised as expected, but the chain was nonetheless "
            "left in an inconsistent state"
        )

    def test_clear_with_dev_dangerous_flag_works(self) -> None:
        """
        The explicit dev/test override still works — tamper-test
        fixtures need it to reset an EventLog between phases.
        """
        pipeline = _pipeline()
        _small_run(pipeline)
        assert len(pipeline.rtp.event_log) > 0

        # Kwarg name uses Python's double-leading-underscore mangling
        # on class bodies, but here it is passed via **kwargs so the
        # mangling rules do NOT apply — the EventLog.clear() reads
        # ``kwargs["__DEV_DANGEROUS"]`` verbatim.
        pipeline.rtp.event_log.clear(**{"__DEV_DANGEROUS": True})
        assert len(pipeline.rtp.event_log) == 0
        ok, err = pipeline.rtp.verify_event_log()
        assert ok is True, f"empty log should verify cleanly: {err}"


# ---------------------------------------------------------------------------
# Test F — 100 real events through the pipeline stay verifiable
# ---------------------------------------------------------------------------

class TestFRegressionEndToEnd:
    def test_hundred_events_ordering_and_verify(self) -> None:
        """
        Regression: a realistic pipeline run that produces a large
        number of heterogeneous events (SHADOW_REFIT, DETECTOR_RESET,
        operator_request, MTOUT_FIRED, INVALID_OBSERVATION, ...) must
        still verify cleanly AND preserve insertion order across both
        the ``_events`` chain and the public view.
        """
        pipeline = _pipeline()

        # Drive the pipeline until the event log carries >= 100 entries.
        # Each round mixes 400 clean observations (which trigger
        # detector-battery events on check_interval boundaries), one
        # NaN observation (INVALID_OBSERVATION), one operator_request
        # (MTOUT_FIRED + cascading model-update / shadow-refit events),
        # and a force_check (detector battery on demand).  The upper
        # bound on rounds is generous — the goal is a STABLE chain of
        # >= 100 entries, not a hot-loop.
        rng = np.random.default_rng(42)
        X_clean = rng.normal(0.0, 1.0, size=(400, 4))
        y_clean = (X_clean.sum(axis=1) > 0).astype(int)
        rounds = 0
        while len(pipeline.rtp.event_log) < 100 and rounds < 40:
            stream(pipeline, X_clean, y_clean)
            pipeline.rtp.observe(
                np.array([np.nan, 0.0, 0.0, 0.0]), y_true=np.array([0])
            )
            pipeline.rtp.operator_request(reason=f"round-{rounds}")
            pipeline.rtp.force_check()
            rounds += 1

        assert len(pipeline.rtp.event_log) >= 100, (
            f"failed to generate 100 events, got "
            f"{len(pipeline.rtp.event_log)}"
        )

        # ── Verify the chain ────────────────────────────────────────
        ok, err = pipeline.rtp.verify_event_log()
        assert ok is True, (
            f"large-run verify() unexpectedly failed at: {err}"
        )

        # ── Event ordering: internal chain, view mirror, and step
        # counter must agree ─────────────────────────────────────────
        events = pipeline.rtp.event_log._events
        assert [ev.index for ev in events] == list(range(len(events))), (
            "ChainedEvent indices are not monotonically 0..n-1"
        )
        # Step counters must be monotonically non-decreasing (some
        # events are emitted at the same step — e.g. the pair of
        # SHADOW_REFIT events from ``set_reference``).
        steps = [view.step for view in pipeline.rtp.event_log]
        assert all(s is None or isinstance(s, int) for s in steps)
        # Drop None entries (if any) for the monotonic check.
        int_steps = [s for s in steps if s is not None]
        assert int_steps == sorted(int_steps), (
            "event view ordering broken — step counters went backwards"
        )
        # Chain-side and view-side lengths must match exactly — a
        # desync is itself a tamper signal according to EventLog.verify.
        assert len(pipeline.rtp.event_log) == len(events)
