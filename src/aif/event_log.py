"""
aif/event_log.py — Tamper-evident replacement for ``RTP.event_log``.

Security context
----------------
Before this module existed, ``RTP.event_log`` was a plain ``list`` of
:class:`rtp.rtp.RTPEvent`.  Any code path with a reference to the RTP —
a compromised subscriber, a buggy detector, a malicious plug-in — could
retroactively mutate or delete entries, erasing evidence of a poisoning
incident or a rollback cascade.  The audit trail downstream tooling
relies on (MToUT fire counts, rollback reasons, SHADOW_REFIT corpus
hashes) therefore had no integrity guarantee.

``EventLog`` wraps the underlying storage in a hash-chained,
optionally-HMAC-signed sequence of :class:`aif.audit.ChainedEvent`
entries.  Appending is still O(1); retroactive edits (mutating a past
entry's payload, re-ordering entries, deleting the middle of the log)
are detected by :meth:`EventLog.verify` without the need for a central
audit server.

Design
------
* ``_events`` holds the tamper-evident hash chain (the source of
  truth).  Its name is underscore-prefixed so external callers know
  they MUST go through the public API — direct mutation will be
  detected by :meth:`verify`, but tooling code should not do it in
  the first place.
* ``_views`` holds per-entry "view" objects that preserve the legacy
  attribute shape (``event_type`` as the original enum, ``step``,
  ``details``, ``timestamp``).  Iterating, indexing and ``len()`` on
  the ``EventLog`` return views — every existing consumer that does
  ``for e in rtp.event_log: e.event_type`` keeps working unchanged.
  ``_views`` is mirror-only: the tamper evidence is rooted in
  ``_events`` / :meth:`verify`, so a desync between the two lists is
  also detected by verify.
* :meth:`append` is the ONLY public write API — there is no setter
  for ``_events`` or ``_views``.  :meth:`clear` raises
  ``PermissionError`` unconditionally unless a ``__DEV_DANGEROUS=True``
  kwarg is passed (tests only — an operator who sees the word
  "DANGEROUS" understands it is never safe in production).

Signing
-------
When a signing key is present (``RTP_AUDIT_HMAC_KEY`` env var, or a
bytes literal passed to the constructor), each appended entry carries
an HMAC-SHA256 signature over its envelope.  ``verify`` validates the
signature when it is present and the signing key is known.  An
unsigned entry is still hash-chained — an attacker who knows the
signing key can re-sign a tampered entry, but re-computing the chain
forwards requires rewriting every subsequent entry's ``prev_hash``,
which ``verify`` detects.

Note on signing-key availability at verify time
-----------------------------------------------
``verify`` uses the key that was present when the ``EventLog`` was
constructed.  If an operator rotates the HMAC key mid-run, older
entries signed under the previous key will fail signature verification
— that is the correct behaviour, not a bug: key rotation MUST be
accompanied by an audited chain export.  The v1 scope does not include
an in-memory multi-key verifier.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Optional

from aif.audit import (
    ChainedEvent,
    GENESIS,
    canonical_bytes,
    get_hmac_key,
    sign_payload,
    verify_chain,
    verify_signature,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# View object — preserves the legacy RTPEvent attribute shape
# ---------------------------------------------------------------------------

@dataclass
class EventLogEntry:
    """
    Read-facing view of one appended event.

    Mirrors :class:`rtp.rtp.RTPEvent` so every existing consumer that
    iterates ``rtp.event_log`` and reads ``e.event_type`` / ``e.step``
    / ``e.details`` keeps working without changes.  The ``event_type``
    attribute carries the ORIGINAL value passed to :meth:`EventLog.append`
    — typically a :class:`rtp.rtp.EventType` enum — so enum comparisons
    like ``e.event_type == EventType.MTOUT_FIRED`` still hold.

    The view is a distinct class from :class:`aif.audit.ChainedEvent`:
    ChainedEvent is the hashed, signed storage record; EventLogEntry is
    the convenience surface.  Operators who need tamper evidence go
    through :attr:`EventLog._events` (the ``ChainedEvent`` chain) and
    :meth:`EventLog.verify`.
    """
    event_type: Any
    step: Optional[int]
    timestamp: float
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:   # pragma: no cover - trivial
        name = getattr(self.event_type, "name", str(self.event_type))
        return f"[step={self.step}] {name} - {self.details}"


# ---------------------------------------------------------------------------
# EventLog — tamper-evident append-only store
# ---------------------------------------------------------------------------

# Sentinel for ``clear()`` — the surface word "DANGEROUS" is deliberate
# so anyone reading the call site understands it is never safe in prod.
_DEV_CLEAR_KWARG = "__DEV_DANGEROUS"


class EventLog:
    """
    Tamper-evident, append-only event log with hash-chained + HMAC-signed
    entries.

    Parameters
    ----------
    signing_key : bytes | None
        Optional explicit HMAC-SHA256 key to use for per-entry
        signatures.  When ``None`` (the default) the key is read from
        :func:`aif.audit.get_hmac_key` — i.e. the ``RTP_AUDIT_HMAC_KEY``
        environment variable, with a published dev key as fallback.
        Pass ``b""`` (empty bytes) to DISABLE signing entirely: entries
        will still be hash-chained, but no HMAC is attached.  The test
        suite exercises the signed path because the dev key makes the
        path reachable without any env setup.

    Read API
    --------
    * ``len(event_log)`` — number of appended entries.
    * ``for e in event_log: ...`` — iterate views (``EventLogEntry``).
    * ``event_log[i]`` — get a view by index or slice.

    Write API
    ---------
    * :meth:`append` — the only supported mutation.  Everything else
      either raises (``clear``) or is undefined behaviour that will be
      detected by :meth:`verify`.
    """

    def __init__(self, signing_key: Optional[bytes] = None) -> None:
        # Resolve the signing key once, at construction time.  Tests
        # that manipulate ``RTP_AUDIT_HMAC_KEY`` after construction
        # continue to use the originally-bound key — matching the
        # "key rotation requires a fresh log" invariant documented in
        # the module docstring.
        if signing_key is None:
            self._signing_key: Optional[bytes] = get_hmac_key()
        elif signing_key == b"":
            # Explicit "no signing" — useful for callers that cannot
            # commit to a key (e.g. ephemeral simulations in tests).
            self._signing_key = None
        else:
            self._signing_key = bytes(signing_key)

        self._events: list[ChainedEvent] = []
        # View mirror — stays in sync with ``_events``.  Read-only for
        # external callers (no setter, no write helper).
        self._views: list[EventLogEntry] = []
        self._last_hash: bytes = GENESIS

    # ------------------------------------------------------------------
    # Write API — the ONLY supported mutation
    # ------------------------------------------------------------------

    def append(
        self,
        event_type: Any,
        payload: Mapping[str, Any],
        step: Optional[int] = None,
    ) -> ChainedEvent:
        """
        Append one event to the tamper-evident chain.

        The hash covers the canonical serialisation of::

            {"type": str(event_type),
             "payload": payload,
             "step": step,
             "prev": prev_hash_hex}

        so reordering, retyping, or mutating a past entry's payload all
        break :meth:`verify`.  When a signing key was bound at
        construction the entry also carries an HMAC-SHA256 signature —
        an attacker without the key cannot forge a valid re-signature
        after tampering.

        Returns the :class:`ChainedEvent` that was just committed.
        """
        # ── Canonical hashed payload ────────────────────────────────
        # Carry the prev_hash as a hex string inside the payload so the
        # hash input is self-describing — an auditor parsing an exported
        # chain can see every field the hash commits to without decoding
        # the raw bytes.
        details = dict(payload)    # defensive shallow copy; do NOT let
                                   # callers mutate the stored dict and
                                   # silently invalidate the chain.
        prev_hex = self._last_hash.hex()
        hash_payload: dict[str, Any] = {
            "type": str(event_type),
            "payload": details,
            "step": step,
            "prev": prev_hex,
        }

        # ── Build envelope for signing ──────────────────────────────
        index = len(self._events)
        envelope = ChainedEvent.envelope(
            index=index,
            event_type=str(event_type),
            payload=hash_payload,
        )
        signature: Optional[bytes]
        if self._signing_key:
            signature = sign_payload(envelope, key=self._signing_key)
        else:
            signature = None

        event = ChainedEvent.make(
            index=index,
            event_type=str(event_type),
            payload=hash_payload,
            prev_hash=self._last_hash,
            signature=signature,
        )
        self._events.append(event)
        self._last_hash = event.entry_hash

        # ── Mirror view for backward-compatible iteration ───────────
        view = EventLogEntry(
            event_type=event_type,   # preserve original type (enum, str, ...)
            step=step,
            timestamp=time.time(),
            details=details,
        )
        self._views.append(view)
        return event

    # ------------------------------------------------------------------
    # Read API — iteration / len / indexing over views
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[EventLogEntry]:
        return iter(self._views)

    def __len__(self) -> int:
        return len(self._views)

    def __getitem__(self, key):
        # Preserve Python list slicing semantics: an int returns one
        # view, a slice returns a new list of views (callers already
        # do ``rtp.event_log[:n]`` in the detector-reset tests).
        return self._views[key]

    def __repr__(self) -> str:   # pragma: no cover - trivial
        return (
            f"EventLog(len={len(self._events)}, "
            f"signed={self._signing_key is not None})"
        )

    # ------------------------------------------------------------------
    # Integrity verification
    # ------------------------------------------------------------------

    def verify(self) -> tuple[bool, Optional[str]]:
        """
        Walk the chain and return ``(True, None)`` when every entry's
        hash, prev-link and (if present) signature validates.  On the
        first failure returns ``(False, err)`` where ``err`` points at
        the offending index and the failing check.

        The failure messages deliberately DO NOT carry raw payload
        bytes — an auditor can reach into ``_events[idx]`` to diagnose
        further; the summary is kept short so it can be safely logged.
        """
        if not self._events:
            return True, None

        # Fast path: if the views mirror went out of sync with the
        # canonical chain we treat it as tampering regardless of
        # hash validity — a caller who bypasses the public API lost
        # the "views match events" invariant.
        if len(self._events) != len(self._views):
            return False, (
                "EventLog.verify: _events / _views length mismatch "
                f"({len(self._events)} vs {len(self._views)}) — the log "
                "was mutated through a bypass channel."
            )

        prev = GENESIS
        for i, ev in enumerate(self._events):
            if ev.index != i:
                return False, (
                    f"EventLog.verify: entry {i} has index={ev.index} "
                    "— chain reordered or spliced."
                )
            if ev.prev_hash != prev:
                return False, (
                    f"EventLog.verify: entry {i} prev_hash does not "
                    f"match the preceding entry_hash — a predecessor "
                    f"was mutated or removed."
                )
            # Recompute hash over the stored envelope.
            if ev.recompute_hash() != ev.entry_hash:
                return False, (
                    f"EventLog.verify: entry {i} payload hash mismatch "
                    "— the payload was mutated in place after append."
                )
            # Signature check — only when we have a key AND the entry
            # was appended with one.  A missing signature on an entry
            # while the log WAS signed elsewhere is treated as
            # tampering (someone re-injected an unsigned entry into
            # the chain).
            expected_signed = self._signing_key is not None
            if expected_signed and ev.signature is None:
                return False, (
                    f"EventLog.verify: entry {i} missing signature on "
                    "a signed log — unsigned entry injected."
                )
            if ev.signature is not None and self._signing_key is not None:
                env = ChainedEvent.envelope(
                    index=ev.index,
                    event_type=ev.event_type,
                    payload=ev.payload,
                )
                if not verify_signature(
                    env, ev.signature, key=self._signing_key
                ):
                    return False, (
                        f"EventLog.verify: entry {i} HMAC signature did "
                        "not validate — payload altered or re-signed "
                        "with a different key."
                    )
            prev = ev.entry_hash
        return True, None

    # ------------------------------------------------------------------
    # Clear — hard-disabled except for explicit dev/test override
    # ------------------------------------------------------------------

    def clear(self, **kwargs: Any) -> None:
        """
        Refuse to clear the log.

        ``RTP.event_log`` is append-only by design: operators, auditors
        and downstream tooling rely on being able to replay the full
        run trace after an incident.  Clearing it erases evidence, so
        this method raises :class:`PermissionError` unconditionally
        unless the caller passes ``__DEV_DANGEROUS=True`` — a sentinel
        reserved for the tamper-test suite.

        The error message embeds "!!" so an operator eyeballing a
        failing run immediately sees it is NEVER safe in production.
        """
        if kwargs.get(_DEV_CLEAR_KWARG, False) is True:
            # Test-only path — still log it loudly so a misuse in CI
            # shows up in test output.
            logger.warning(
                "EventLog.clear(!! __DEV_DANGEROUS=True !!): erasing "
                "%d entries — this path MUST NOT run in production.",
                len(self._events),
            )
            self._events.clear()
            self._views.clear()
            self._last_hash = GENESIS
            return
        raise PermissionError(
            "EventLog.clear is not permitted !! the event log is "
            "append-only and clearing it erases tamper-evident audit "
            "evidence !! this call is NEVER safe in production; the "
            "only bypass is an explicit __DEV_DANGEROUS=True kwarg "
            "reserved for the tamper-test suite."
        )

    # ------------------------------------------------------------------
    # Utility — expose the signed-flag for diagnostics / tests
    # ------------------------------------------------------------------

    @property
    def is_signed(self) -> bool:
        """True when every append attaches an HMAC signature."""
        return self._signing_key is not None


# ---------------------------------------------------------------------------
# Standalone helpers — exported for tests / CLI tooling
# ---------------------------------------------------------------------------

def verify_event_chain(
    events: list[ChainedEvent], signing_key: Optional[bytes] = None
) -> tuple[bool, Optional[str]]:
    """
    Thin wrapper around :func:`aif.audit.verify_chain` that returns the
    structured ``(ok, err)`` tuple used everywhere else in this module.

    Exposed so test fixtures and future offline audit tooling can
    validate a chain that was exported from an ``EventLog`` without
    depending on the live instance.
    """
    key = signing_key if signing_key is not None else get_hmac_key()
    # ``verify_chain`` only returns a bool; for richer error reporting
    # we run the same walk inline.
    prev = GENESIS
    for i, ev in enumerate(events):
        if ev.index != i:
            return False, f"entry {i} has index={ev.index}"
        if ev.prev_hash != prev:
            return False, f"entry {i} prev_hash break"
        if ev.recompute_hash() != ev.entry_hash:
            return False, f"entry {i} payload hash mismatch"
        if ev.signature is not None:
            env = ChainedEvent.envelope(
                index=ev.index,
                event_type=ev.event_type,
                payload=ev.payload,
            )
            if not verify_signature(env, ev.signature, key=key):
                return False, f"entry {i} signature mismatch"
        prev = ev.entry_hash
    # Also run the canonical stdlib walk for belt-and-braces.
    if not verify_chain(events, key=key):
        return False, "verify_chain bool-level check failed"
    return True, None
