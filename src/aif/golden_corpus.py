"""
aif/golden_corpus.py — Operator-curated clean reference for the CPD shadow.

Why this module exists
----------------------
The Concept Poisoning Detector (``detectors.cpd.CPD``) relies on a
*shadow* classifier: an independently trained model fitted on data the
operator vouches for as clean. When the shadow's predictions diverge
from the live MLIN's outputs (and one of the two behaviour checks
corroborates), concept poisoning is flagged.

Before this module existed, the shadow was refit on whatever rows
happened to be in the LIB/LOB/YGT buffers — buffers any attacker who
can influence ``rtp.observe()`` inputs can pollute. A patient attacker
could therefore tutor the shadow onto the poisoned boundary and
silence the one detector designed to catch concept-space attacks.

The ``GoldenCorpus`` closes this channel:

* rows are **append-only** — ``snapshot(n)`` reads the most recent ``n``
  rows but never evicts older ones;
* appends require either an HMAC signature from the operator
  (production path) or an explicit ``authorize=True`` kwarg
  (dev/test path that logs the bypass);
* each append is recorded in a tamper-evident hash-chained
  ``event_log`` (``aif.audit.ChainedEvent``); an auditor can detect
  retroactive edits without a central server;
* ``snapshot(n)`` returns a ``GoldenCorpusSnapshot`` carrying a hash
  of the exact rows returned so a downstream consumer (the CPD
  refit) can bind its result to a specific corpus commitment.

The v1 backend is in-memory, but the API is deliberately structured
so a file / SQLite / cloud-object-store backend is a drop-in:

* only ``append`` mutates state;
* ``snapshot`` is pure;
* ``_rows`` is a private list of ``CorpusRow`` dataclasses, not a
  numpy array — persistence layers can iterate over it rather than
  serialising a monolithic ndarray.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional

import numpy as np

from aif.audit import (
    GENESIS,
    ChainedEvent,
    canonical_bytes,
    get_hmac_key,
    sign_payload,
    verify_signature,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Row + snapshot dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CorpusRow:
    """One (X, y) row of the golden corpus.

    ``source`` is a short tag identifying where the row originated
    ("operator_upload", "post_retrain_audit", ...). ``batch_hash`` is
    the hash of the batch ``append`` call this row belonged to — it
    lets an auditor trace a single row back to the signed commitment
    that added it.
    """
    X: np.ndarray                               # shape: (n_features,)
    y: float
    source: str = "operator"
    timestamp: float = field(default_factory=time.time)
    batch_hash: Optional[bytes] = None


@dataclass(frozen=True)
class GoldenCorpusSnapshot:
    """
    Atomic read-only view over a prefix of the corpus.

    The CPD ``fit_reference`` binds its shadow's provenance to
    ``corpus_hash`` — the hash is written into the ``CPDResult`` so a
    downstream auditor can prove WHICH rows produced the shadow that
    fired (or didn't fire) at step N.

    ``corpus_hash`` is a SHA-256 digest over the canonical serialisation
    of ``{n_rows, X_rows, y_rows}``. It is a content hash only: it does
    NOT depend on wall-clock time or row insertion order outside the
    snapshot window, so two callers who request the same tail at the
    same moment get matching hashes.
    """
    X: np.ndarray                               # shape: (n, n_features)
    y: np.ndarray                               # shape: (n,)
    corpus_hash: bytes                          # 32 raw bytes
    n_rows: int
    taken_at: float = field(default_factory=time.time)

    @property
    def corpus_hash_hex(self) -> str:
        """Human-readable hash for log messages and ``CPDResult``."""
        return self.corpus_hash.hex()

    def __len__(self) -> int:
        return int(self.n_rows)


# ---------------------------------------------------------------------------
# GoldenCorpus
# ---------------------------------------------------------------------------

class GoldenCorpus:
    """
    Append-only operator-curated (X, y) store with tamper-evident log.

    Parameters
    ----------
    n_features : int
        Expected width of each X row. Appends are rejected when the
        incoming batch's width does not match — this catches the
        common "operator uploaded the wrong CSV" incident before any
        bad data reaches the shadow.
    allow_unauthorised : bool
        When True, ``append`` accepts ``authorize=True`` without a
        signature. Defaults to True to keep dev/test ergonomic; set
        to False in production environments where only signed
        operator uploads should be admitted.
    """

    def __init__(
        self,
        n_features: int,
        allow_unauthorised: bool = True,
    ) -> None:
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1 (got {n_features})")
        self._n_features = int(n_features)
        self._allow_unauthorised = bool(allow_unauthorised)
        self._rows: list[CorpusRow] = []
        self._event_log: list[ChainedEvent] = []
        self._last_hash: bytes = GENESIS

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def append(
        self,
        X: np.ndarray,
        y: np.ndarray,
        source: str,
        operator_sig: Optional[bytes] = None,
        *,
        authorize: bool = False,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> ChainedEvent:
        """
        Append a batch of ``(X, y)`` rows to the corpus.

        Authorisation — EITHER of:

        * ``operator_sig`` — an HMAC-SHA256 signature produced by
          :func:`aif.audit.sign_payload` over the canonical payload
          ``{source, n, shape}``. The production path.
        * ``authorize=True`` — dev/test-only flag. The append is
          accepted but recorded as ``unauthorised`` in the event log
          so auditors can see the corpus was modified without a
          signature. Rejected entirely when the corpus was constructed
          with ``allow_unauthorised=False``.

        Returns the :class:`ChainedEvent` that was appended to the
        internal event log.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"GoldenCorpus.append: X has {X.shape[0]} rows but y has "
                f"{y.shape[0]}."
            )
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"GoldenCorpus.append: expected {self._n_features} "
                f"features per row, got {X.shape[1]}."
            )
        if X.shape[0] == 0:
            raise ValueError("GoldenCorpus.append: refusing to append 0 rows.")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError(
                "GoldenCorpus.append: non-finite values in X or y — refuse "
                "to admit potentially corrupted rows into the golden corpus."
            )

        # ── Authorisation ──────────────────────────────────────────────
        auth_payload = {
            "source": str(source),
            "n": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            # Bind the signature to the row content — an attacker cannot
            # replay a valid signature with different rows.
            "batch_digest": hashlib.sha256(
                canonical_bytes({"X": X.tolist(), "y": y.tolist()})
            ).hexdigest(),
        }
        auth_mode: str
        if operator_sig is not None:
            if not verify_signature(auth_payload, operator_sig):
                raise PermissionError(
                    "GoldenCorpus.append: operator_sig did not verify "
                    "against the corpus HMAC key."
                )
            auth_mode = "signed"
        elif authorize:
            if not self._allow_unauthorised:
                raise PermissionError(
                    "GoldenCorpus.append: authorize=True bypass is "
                    "disabled on this corpus (production mode). Provide "
                    "operator_sig."
                )
            auth_mode = "unauthorised"
            logger.warning(
                "GoldenCorpus.append: unsigned append of %d rows from "
                "source=%r (dev/test path).", X.shape[0], source,
            )
        else:
            raise PermissionError(
                "GoldenCorpus.append: missing operator_sig and "
                "authorize=True. Refusing to admit rows."
            )

        # ── Hash of this specific batch for provenance ────────────────
        batch_hash = hashlib.sha256(
            canonical_bytes(auth_payload) + self._last_hash
        ).digest()

        # ── Commit rows ───────────────────────────────────────────────
        for row_X, row_y in zip(X, y):
            self._rows.append(CorpusRow(
                X=row_X.copy(),
                y=float(row_y),
                source=str(source),
                batch_hash=batch_hash,
            ))

        # ── Record event ──────────────────────────────────────────────
        event_payload = {
            "action": "CORPUS_APPEND",
            "source": str(source),
            "auth_mode": auth_mode,
            "n_rows": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "batch_digest": auth_payload["batch_digest"],
            "batch_hash": batch_hash.hex(),
            "corpus_size_after": len(self._rows),
            "metadata": dict(metadata or {}),
        }
        envelope = ChainedEvent.envelope(
            index=len(self._event_log),
            event_type="CORPUS_APPEND",
            payload=event_payload,
        )
        signature = sign_payload(envelope) if operator_sig is not None else None
        event = ChainedEvent.make(
            index=len(self._event_log),
            event_type="CORPUS_APPEND",
            payload=event_payload,
            prev_hash=self._last_hash,
            signature=signature,
        )
        self._event_log.append(event)
        self._last_hash = event.entry_hash

        logger.info(
            "GoldenCorpus: +%d rows from source=%r (auth=%s, total=%d).",
            X.shape[0], source, auth_mode, len(self._rows),
        )
        return event

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def n_features(self) -> int:
        return self._n_features

    def is_ready(self, min_size: int) -> bool:
        """True when the corpus has at least ``min_size`` rows."""
        return len(self._rows) >= int(min_size)

    @property
    def event_log(self) -> list[ChainedEvent]:
        """Immutable (shallow-copy) view of the tamper-evident event log."""
        return list(self._event_log)

    def snapshot(self, n: Optional[int] = None) -> GoldenCorpusSnapshot:
        """
        Return an atomic read-only view over the most recent ``n`` rows
        (or all rows when ``n`` is ``None``).

        The returned snapshot carries a SHA-256 ``corpus_hash`` bound
        to the exact ``(X, y)`` content of the slice so callers can
        record the commitment in their own audit trail.
        """
        if not self._rows:
            raise RuntimeError(
                "GoldenCorpus.snapshot: corpus is empty — call is_ready() "
                "before taking a snapshot."
            )
        take = len(self._rows) if n is None else min(int(n), len(self._rows))
        if take <= 0:
            raise ValueError(
                f"GoldenCorpus.snapshot: refusing to return 0 rows (asked {n})."
            )
        tail = self._rows[-take:]
        X = np.stack([row.X for row in tail])
        y = np.asarray([row.y for row in tail], dtype=float)
        corpus_hash = hashlib.sha256(
            canonical_bytes({
                "n_rows": int(take),
                "X": X.tolist(),
                "y": y.tolist(),
            })
        ).digest()
        logger.debug(
            "GoldenCorpus.snapshot: returning %d rows (hash=%s).",
            take, corpus_hash.hex()[:16],
        )
        return GoldenCorpusSnapshot(
            X=X, y=y, corpus_hash=corpus_hash, n_rows=int(take),
        )

    # ------------------------------------------------------------------
    # Helpers for operators and tests
    # ------------------------------------------------------------------

    def sign_append_payload(
        self, source: str, X: np.ndarray, y: np.ndarray
    ) -> bytes:
        """
        Produce the HMAC signature an operator would send alongside
        ``append(..., operator_sig=...)``. Exposed for test fixtures
        and CLI tooling; production signers should use an isolated
        process with access to the private key.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        auth_payload = {
            "source": str(source),
            "n": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "batch_digest": hashlib.sha256(
                canonical_bytes({"X": X.tolist(), "y": y.tolist()})
            ).hexdigest(),
        }
        return sign_payload(auth_payload, key=get_hmac_key())

    def __repr__(self) -> str:
        return (
            f"GoldenCorpus(n_features={self._n_features}, "
            f"len={len(self._rows)}, events={len(self._event_log)}, "
            f"allow_unauthorised={self._allow_unauthorised})"
        )
