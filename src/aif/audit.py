"""
aif/audit.py — Tamper-evident append-only event chaining utility.

Security context
----------------
Many subsystems in the RTP Observer (GoldenCorpus additions, CPD shadow
refits, future AIF event logs) need a way to record a sequence of
operator-observable events such that:

  * adding a new entry is cheap (O(1));
  * *any* retroactive edit of a past entry is detectable without a
    central server — every entry carries the hash of the previous
    entry, forming a Merkle-style hash chain;
  * the implementation has no new dependencies (stdlib hashlib + hmac
    are used, matching the project-wide "no new deps" constraint).

This is NOT a substitute for a signed, remote append-only log (e.g.
AWS CloudTrail, Google Cloud Audit Logs, Rekor). For the v1 milestone
it provides in-memory evidence that can be exported to such a log
later.

Design
------
Each event is serialised deterministically (``sorted_keys`` JSON with
NaN-safe float handling). Its ``entry_hash`` is::

    entry_hash = sha256(prev_hash || serialised_payload)

The first entry in a chain uses ``prev_hash = GENESIS`` (the constant
``b"\\x00" * 32``). Subsequent callers pass the most recent entry's
``entry_hash`` as ``prev_hash`` to the next append.

HMAC signing
------------
An optional HMAC-SHA256 signature can be attached via ``sign_payload``
using a shared key loaded from the ``RTP_AUDIT_HMAC_KEY`` environment
variable. When the env var is unset, a hard-coded development key is
used and a warning is logged. Production deployments MUST set the env
var. Key management (rotation, hardware-backed keys, etc.) is out of
scope for v1 — this implementation provides only the verification API.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)


GENESIS: bytes = b"\x00" * 32
"""The canonical "previous" hash for the first entry in any chain."""

_DEV_HMAC_KEY = b"rtp-observer-dev-hmac-key-NOT-FOR-PRODUCTION"
_ENV_HMAC_KEY = "RTP_AUDIT_HMAC_KEY"
_warned_about_dev_key = False


def get_hmac_key() -> bytes:
    """
    Return the HMAC-SHA256 key used for operator-signature verification.

    Reads from the ``RTP_AUDIT_HMAC_KEY`` environment variable. Falls
    back to a published development key and logs one warning when the
    env var is missing — this keeps dev/test flows working without
    silently shipping the dev key to production.
    """
    global _warned_about_dev_key
    raw = os.environ.get(_ENV_HMAC_KEY)
    if raw:
        return raw.encode("utf-8") if isinstance(raw, str) else raw
    if not _warned_about_dev_key:
        logger.warning(
            "aif.audit: %s not set; falling back to the published "
            "development HMAC key. Set %s in production.",
            _ENV_HMAC_KEY, _ENV_HMAC_KEY,
        )
        _warned_about_dev_key = True
    return _DEV_HMAC_KEY


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _canonical_default(obj: Any) -> Any:
    """json.dumps default for objects the stdlib encoder cannot handle."""
    if isinstance(obj, (bytes, bytearray)):
        return obj.hex()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    try:
        import numpy as np  # local import to avoid a hard dep if swapped out
    except Exception:  # pragma: no cover
        np = None  # type: ignore
    if np is not None:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    raise TypeError(f"unable to canonicalise object of type {type(obj)!r}")


def canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    """
    Deterministic, NaN-safe JSON byte serialisation of ``payload``.

    Uses ``sort_keys=True`` and stable float handling so two processes
    that hash the same logical payload get the same bytes.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_canonical_default,
        allow_nan=False,
    ).encode("utf-8")


# ---------------------------------------------------------------------------
# Hash chain primitives
# ---------------------------------------------------------------------------

def chain_hash(prev_hash: bytes, payload: Mapping[str, Any]) -> bytes:
    """
    Compute ``sha256(prev_hash || canonical_bytes(payload))``.

    ``prev_hash`` must be exactly 32 bytes (use :data:`GENESIS` for the
    first entry in a chain). The payload is canonicalised via
    :func:`canonical_bytes` before hashing so equivalent dicts yield
    identical hashes regardless of key order.
    """
    if not isinstance(prev_hash, (bytes, bytearray)) or len(prev_hash) != 32:
        raise ValueError(
            "prev_hash must be 32 raw bytes; use audit.GENESIS for the "
            "first entry in a chain."
        )
    h = hashlib.sha256()
    h.update(bytes(prev_hash))
    h.update(canonical_bytes(payload))
    return h.digest()


def sign_payload(payload: Mapping[str, Any], key: Optional[bytes] = None) -> bytes:
    """
    HMAC-SHA256 sign the canonical serialisation of ``payload``.

    The key is taken from :func:`get_hmac_key` when not supplied.
    """
    k = key if key is not None else get_hmac_key()
    return hmac.new(k, canonical_bytes(payload), hashlib.sha256).digest()


def verify_signature(
    payload: Mapping[str, Any],
    signature: bytes,
    key: Optional[bytes] = None,
) -> bool:
    """Constant-time verification of a payload signature."""
    if not isinstance(signature, (bytes, bytearray)):
        return False
    expected = sign_payload(payload, key=key)
    return hmac.compare_digest(bytes(signature), expected)


# ---------------------------------------------------------------------------
# ChainedEvent — one entry in a hash-chained append-only log
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChainedEvent:
    """
    One entry in a tamper-evident append-only log.

    Fields
    ------
    index : int
        Zero-based position of this event in its chain.
    event_type : str
        Short identifier ("CORPUS_APPEND", "SHADOW_REFIT", ...).
    payload : dict
        Event-specific body. Canonically serialised when hashing.
    prev_hash : bytes
        The ``entry_hash`` of the preceding event, or :data:`GENESIS`
        for the first entry.
    entry_hash : bytes
        ``sha256(prev_hash || canonical_bytes(full_envelope))`` where
        ``full_envelope`` = ``{index, event_type, payload}``.
    signature : bytes | None
        Optional HMAC-SHA256 signature over ``full_envelope``. ``None``
        means the event was appended without operator authorisation
        (dev/test path).
    """
    index: int
    event_type: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    prev_hash: bytes = GENESIS
    entry_hash: bytes = field(default=GENESIS)
    signature: Optional[bytes] = None

    @staticmethod
    def envelope(
        index: int, event_type: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Return the canonical envelope that ``entry_hash`` is taken over."""
        return {
            "index": int(index),
            "event_type": str(event_type),
            "payload": dict(payload),
        }

    @classmethod
    def make(
        cls,
        *,
        index: int,
        event_type: str,
        payload: Mapping[str, Any],
        prev_hash: bytes,
        signature: Optional[bytes] = None,
    ) -> "ChainedEvent":
        """
        Build a new ``ChainedEvent`` and compute its ``entry_hash``.

        The hash covers the full envelope (index + event_type + payload),
        not just the payload, so an attacker cannot reorder or retype
        entries without invalidating the chain.
        """
        env = cls.envelope(index, event_type, payload)
        entry_hash = chain_hash(prev_hash, env)
        return cls(
            index=index,
            event_type=str(event_type),
            payload=dict(payload),
            prev_hash=bytes(prev_hash),
            entry_hash=entry_hash,
            signature=signature,
        )

    # ------------------------------------------------------------------
    # Verification API — callers can audit an exported chain offline
    # ------------------------------------------------------------------

    def recompute_hash(self) -> bytes:
        env = self.envelope(self.index, self.event_type, self.payload)
        return chain_hash(self.prev_hash, env)

    def verify(self, key: Optional[bytes] = None) -> bool:
        """
        True when ``entry_hash`` matches the recomputed value AND, if a
        signature is present, the signature validates under ``key``.
        """
        if self.recompute_hash() != self.entry_hash:
            return False
        if self.signature is not None:
            env = self.envelope(self.index, self.event_type, self.payload)
            return verify_signature(env, self.signature, key=key)
        return True


def verify_chain(
    events: list[ChainedEvent], key: Optional[bytes] = None
) -> bool:
    """
    Validate an entire chain: index sequence, prev-hash links, per-event
    hashes, and optional signatures.
    """
    prev = GENESIS
    for i, ev in enumerate(events):
        if ev.index != i:
            return False
        if ev.prev_hash != prev:
            return False
        if not ev.verify(key=key):
            return False
        prev = ev.entry_hash
    return True
