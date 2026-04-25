"""
aif/dpostp.py — Data Post-Processor (DPostP)

Implements the post-inference data-processing stage described in the paper
(Section IV-D-3).  Two orthogonal responsibilities:

1.  **Training-corpus cleaning** (local, pre-MTP)
    • z-clip column outliers past ``z_clip``σ against the column's own μ/σ
    • drop rows containing NaN in X or y
    • de-duplicate exact row repeats (common when a deque replays samples)

2.  **Reference stabilisation** (local, pre-``notify_model_updated``)
    The GT-only training slice exported by the ATM is typically only 30-100
    rows, which is too thin for a stable CPD correlation baseline: with
    d=10 features and n=50 rows, pairwise Pearson r has a sampling std of
    ~1/√(n-3) ≈ 0.145, so the 100-row live window routinely drifts by
    |Δr|≈0.4 from the reference through pure noise and CPD false-fires.
    ``build_reference`` pads the thin GT slice with recent LIB rows
    (labelled via the newly-trained estimator so the shadow model's fit
    stays consistent with the post-deploy MLIN) up to ``min_ref_rows``.
    At n=200 the Pearson sampling std drops to ~0.071 — small enough that
    a 4σ Fisher-z false alarm requires |Δr|>0.28 rather than |Δr|>0.56.

3.  **Transport hardening** (cross-trust, pre-MTP-E)
    The paper's "anonymisation, encryption, compression, integrity checks"
    duties collapse into a single AEAD envelope:
        • gzip compress the serialised (X, y)
        • AES-256-GCM encrypt the gzip bytes with a pre-provisioned
          256-bit key loaded from ``DPOSTP_KEY`` (base64)
        • bind an AAD block {sender_id, model_version, algo_id, timestamp}
          so replay and cross-run substitution are rejected on unseal
    Compress-then-encrypt is the correct order — encrypt-then-compress
    exposes no structure in the ciphertext so would be useless.  The GCM
    128-bit tag doubles as the integrity check; no separate HMAC is
    layered on top (defence-in-depth theatre for a prototype).

    Wire format (big-endian where applicable)::

        magic(4) || version(1) || aad_len(2) || aad_json(aad_len)
                 || nonce(12) || ciphertext_and_tag(...)

Anonymisation for numeric feature matrices intentionally stops at metadata
stripping: column names, pandas indices, and dtype annotations are never
written into the envelope.  Gaussian-mechanism DP at eps=1..5 against an
L2-sensitivity of 5√d destroys enough signal that the remote trainer
cannot distinguish classes; DP belongs at training time, not transport.
"""

from __future__ import annotations

import base64
import gzip
import io
import json
import logging
import os
import struct
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator
    from aif.buffers import LIB

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wire-format constants
# ---------------------------------------------------------------------------

_MAGIC = b"DPPT"                       # DPostP Transport payload marker
_VERSION = 1
_ALGO_ID = "AES256GCM-GZIP-v1"         # bound into AAD
_NONCE_LEN = 12                        # 96-bit GCM nonce
_TAG_LEN = 16                          # 128-bit GCM tag
_MAX_AAD_LEN = 65535                   # 2-byte length prefix
_DEFAULT_SKEW_WINDOW_S = 300           # ±5 min replay window
_KEY_LEN = 32                          # 256-bit AES key

# Error sentinels so callers don't have to import hazmat exceptions.
class DPostPError(Exception):
    """Base class for all DPostP transport/reference errors."""


class DPostPAuthenticationError(DPostPError):
    """AEAD tag mismatch, replay/skew violation, or AAD field mismatch."""


class DPostPKeyError(DPostPError):
    """Transport key missing or malformed."""


# ---------------------------------------------------------------------------
# SealedPayload — the opaque export wrapper
# ---------------------------------------------------------------------------

@dataclass
class SealedPayload:
    """
    Opaque container for a cross-trust training export.

    ``data`` is the full wire-format byte string (magic..ciphertext+tag).
    ``aad`` is retained as a dict *only for inspection* — the receiver
    parses AAD from the wire payload and does not trust this field.
    """
    data: bytes
    aad: dict
    n_rows: int
    n_features: int
    algo: str = _ALGO_ID

    def __len__(self) -> int:
        return len(self.data)

    @property
    def sealed_bytes(self) -> int:
        return len(self.data)


# ---------------------------------------------------------------------------
# DPostP
# ---------------------------------------------------------------------------

class DPostP:
    """
    Data Post-Processor.

    Parameters
    ----------
    z_clip : float
        Column-wise z-score cap applied by ``process_training_batch``.
        Set to ``np.inf`` to disable clipping entirely.
    min_ref_rows : int
        Default target size used by ``build_reference`` when the caller
        does not override ``min_rows``.  ~200 gives a 2× sampling-noise
        reduction over a 50-row GT slice.
    sender_id : str
        Identifier written into AAD on ``seal``.  Receiver may pin an
        ``expected_sender_id`` to reject cross-trust source substitution.
    key : bytes or None
        Explicit 32-byte AES-256 key.  If None, falls back to
        ``os.environ[key_env_var]`` (base64-encoded 32 bytes).
    key_env_var : str
        Environment variable consulted when ``key`` is not provided.
    require_transport_key : bool
        When True the constructor raises ``DPostPKeyError`` if no key
        can be resolved.  When False (default) seal/unseal simply raise
        at call time — useful for unit tests that exercise the cleaning
        role without provisioning a key.
    """

    def __init__(
        self,
        *,
        z_clip: float = 5.0,
        min_ref_rows: int = 200,
        sender_id: str = "rtp_observer/local",
        key: Optional[bytes] = None,
        key_env_var: str = "DPOSTP_KEY",
        require_transport_key: bool = False,
    ) -> None:
        if z_clip is not None and not (z_clip > 0):
            raise ValueError(f"z_clip must be > 0 or np.inf (got {z_clip!r})")
        if min_ref_rows < 1:
            raise ValueError(f"min_ref_rows must be >= 1 (got {min_ref_rows})")

        self.z_clip = float(z_clip) if z_clip is not None else float("inf")
        self.min_ref_rows = int(min_ref_rows)
        self.sender_id = str(sender_id)
        self.key_env_var = str(key_env_var)
        self._key: Optional[bytes] = self._resolve_key(key, key_env_var)
        if require_transport_key and self._key is None:
            raise DPostPKeyError(
                f"No transport key provided and {key_env_var!r} is unset."
            )

    # ------------------------------------------------------------------
    # Role 1 — Training-corpus cleaning
    # ------------------------------------------------------------------

    def process_training_batch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        dedup: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sanitise a training batch.

        Steps (in order):
          1. Coerce to 2-D ``float64`` X and 1-D y.
          2. Drop rows whose X or y contains NaN/±inf.
          3. Column-wise z-clip using that batch's own μ/σ (so the
             cleaning is distribution-agnostic and stable on small n).
          4. Optional de-duplication (exact-row repeats).

        Returns
        -------
        (X_clean, y_clean)
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y row-count mismatch: X={X.shape}, y={y.shape}"
            )

        # ── 2. Drop non-finite rows ───────────────────────────────────
        finite = np.isfinite(X).all(axis=1) & np.isfinite(y)
        dropped_nan = int((~finite).sum())
        X = X[finite]
        y = y[finite]

        if X.shape[0] == 0:
            logger.warning(
                "DPostP.process_training_batch: all %d rows dropped as non-finite.",
                dropped_nan,
            )
            return X, y

        # ── 3. Column z-clip ──────────────────────────────────────────
        clipped = 0
        if np.isfinite(self.z_clip):
            mu = X.mean(axis=0)
            sd = X.std(axis=0)
            # Avoid divide-by-zero on constant columns: their z-score is 0.
            safe = np.where(sd > 0, sd, 1.0)
            z = (X - mu) / safe
            mask = np.abs(z) > self.z_clip
            if mask.any():
                clipped = int(mask.sum())
                X_clipped = np.where(
                    mask, np.sign(z) * self.z_clip * safe + mu, X
                )
                X = X_clipped

        # ── 4. Dedup ──────────────────────────────────────────────────
        dropped_dup = 0
        if dedup and X.shape[0] > 1:
            # np.unique on rows is O(n log n) — acceptable at n<=2000.
            combined = np.concatenate([X, y[:, None]], axis=1)
            _, uniq_idx = np.unique(combined, axis=0, return_index=True)
            uniq_idx.sort()
            if uniq_idx.size < X.shape[0]:
                dropped_dup = X.shape[0] - uniq_idx.size
                X = X[uniq_idx]
                y = y[uniq_idx]

        logger.info(
            "DPostP.process_training_batch: kept %d rows "
            "(dropped %d non-finite, %d duplicates, clipped %d cells past %.1fσ).",
            X.shape[0], dropped_nan, dropped_dup, clipped, self.z_clip,
        )
        return X, y

    # ------------------------------------------------------------------
    # Role 2 — Reference stabilisation (CPD shadow + correlation baseline)
    # ------------------------------------------------------------------

    def build_reference(
        self,
        X_gt: np.ndarray,
        y_gt: np.ndarray,
        *,
        lib: Optional["LIB"] = None,
        lib_rows: Optional[np.ndarray] = None,
        new_estimator: Optional["BaseEstimator"] = None,
        min_rows: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Produce a stable (X_ref, y_ref) reference for the detectors.

        The caller should have already run the batch through
        ``process_training_batch``.  When the cleaned GT slice is at
        least ``min_rows`` long the inputs are returned unchanged; when
        it is thinner we pad with recent LIB rows (labelled via
        ``new_estimator.predict`` so the shadow model trained on the
        returned reference stays consistent with the post-deploy MLIN).

        Parameters
        ----------
        X_gt, y_gt : np.ndarray
            Cleaned training slice with real labels.
        lib : LIB, optional
            Source of padding rows.  Either ``lib`` or ``lib_rows`` must
            be provided when padding is required.
        lib_rows : np.ndarray, optional
            Raw LIB row dump already materialised.  Takes precedence
            over ``lib`` when both are supplied.
        new_estimator : BaseEstimator, optional
            Freshly-trained estimator used to label the padding rows.
            If omitted and padding is needed, the function falls back to
            returning just the GT slice with a warning.
        min_rows : int, optional
            Override for the instance-level ``min_ref_rows``.
        """
        X_gt = np.atleast_2d(np.asarray(X_gt, dtype=float))
        y_gt = np.asarray(y_gt, dtype=float).ravel()
        target = int(min_rows if min_rows is not None else self.min_ref_rows)
        n_gt = X_gt.shape[0]

        if n_gt >= target:
            logger.info(
                "DPostP.build_reference: GT slice already %d ≥ %d rows; "
                "no padding required.",
                n_gt, target,
            )
            return X_gt, y_gt

        # ── Materialise padding source ───────────────────────────────
        X_lib: Optional[np.ndarray] = None
        if lib_rows is not None:
            X_lib = np.atleast_2d(np.asarray(lib_rows, dtype=float))
        elif lib is not None:
            try:
                X_lib = np.atleast_2d(np.asarray(lib.get_values(), dtype=float))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "DPostP.build_reference: lib.get_values() failed (%s); "
                    "returning unpadded GT slice.",
                    exc,
                )
                return X_gt, y_gt

        if X_lib is None or X_lib.shape[0] == 0 or new_estimator is None:
            logger.warning(
                "DPostP.build_reference: padding requested but "
                "%s; returning unpadded GT slice (n=%d < target %d).",
                "LIB source is empty" if X_lib is None or X_lib.shape[0] == 0
                    else "no new_estimator supplied",
                n_gt, target,
            )
            return X_gt, y_gt

        if X_lib.shape[1] != X_gt.shape[1]:
            logger.warning(
                "DPostP.build_reference: LIB/X_gt feature-dim mismatch "
                "(LIB=%d, X_gt=%d); returning unpadded GT slice.",
                X_lib.shape[1], X_gt.shape[1],
            )
            return X_gt, y_gt

        # Take the most-recent padding rows.  Some will duplicate X_gt
        # (those rows came from LIB originally); that's fine — the
        # resulting reference just carries a little extra GT weight.
        n_pad = target - n_gt
        take = min(n_pad + n_gt, X_lib.shape[0])
        X_lib = X_lib[-take:]

        try:
            y_lib = np.asarray(
                new_estimator.predict(X_lib), dtype=float
            ).ravel()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "DPostP.build_reference: new_estimator.predict failed (%s); "
                "returning unpadded GT slice.",
                exc,
            )
            return X_gt, y_gt

        # LIB rows first (background), GT rows last (authoritative tail).
        # Downstream CPD.fit_reference trims with X_arr[-n:], so the GT
        # slice is guaranteed to survive even when the combined length
        # overshoots the detector's own reference_size cap.
        X_ref = np.vstack([X_lib, X_gt])
        y_ref = np.concatenate([y_lib, y_gt])

        # Trim to exactly ``target`` rows (GT-heavy tail preserved).
        if X_ref.shape[0] > target:
            X_ref = X_ref[-target:]
            y_ref = y_ref[-target:]

        logger.info(
            "DPostP.build_reference: padded GT slice %d → %d rows "
            "(added %d LIB rows labelled via %s).",
            n_gt, X_ref.shape[0], X_ref.shape[0] - n_gt,
            type(new_estimator).__name__,
        )
        return X_ref, y_ref

    # ------------------------------------------------------------------
    # Role 3 — Transport hardening
    # ------------------------------------------------------------------

    def seal(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        model_version: str,
        extra_aad: Optional[dict] = None,
    ) -> SealedPayload:
        """
        Compress, encrypt, and AAD-bind a training export.

        Raises
        ------
        DPostPKeyError
            No 32-byte transport key was resolved at construction time.
        """
        if self._key is None:
            raise DPostPKeyError(
                "DPostP.seal: transport key unavailable "
                f"(set {self.key_env_var!r} or pass key=... to the constructor)."
            )

        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"seal: X and y row-count mismatch: X={X.shape}, y={y.shape}"
            )

        # ── Serialise (numpy .npz) ────────────────────────────────────
        buf = io.BytesIO()
        # Using savez (not savez_compressed) because we gzip the whole
        # thing below and double-compression adds latency for no gain.
        np.savez(buf, X=X, y=y)
        serialised = buf.getvalue()

        # ── Compress ──────────────────────────────────────────────────
        # gzip level 6 is the Python default and gives ~50% ratio on
        # float64 ML matrices at ~100 MB/s — good enough for a thesis
        # prototype.  (zstandard would be faster but isn't stdlib.)
        compressed = gzip.compress(serialised, compresslevel=6)

        # ── Build AAD and encrypt ─────────────────────────────────────
        aad = {
            "sender_id":     self.sender_id,
            "model_version": str(model_version),
            "algo_id":       _ALGO_ID,
            "timestamp":     int(time.time()),
            "n_rows":        int(X.shape[0]),
            "n_features":    int(X.shape[1]),
        }
        if extra_aad:
            # Disallow key collisions so callers cannot silently override
            # the security-critical fields above.
            bad = set(extra_aad) & set(aad)
            if bad:
                raise ValueError(
                    f"seal: extra_aad keys {sorted(bad)} collide with reserved AAD fields"
                )
            aad.update(extra_aad)

        aad_bytes = self._canonical_json(aad)
        if len(aad_bytes) > _MAX_AAD_LEN:
            raise ValueError(
                f"seal: AAD block {len(aad_bytes)} B exceeds {_MAX_AAD_LEN} B wire limit."
            )

        nonce = os.urandom(_NONCE_LEN)
        ciphertext = self._aesgcm().encrypt(nonce, compressed, aad_bytes)

        # ── Wire format ───────────────────────────────────────────────
        header = _MAGIC + bytes([_VERSION]) + struct.pack(">H", len(aad_bytes))
        wire = header + aad_bytes + nonce + ciphertext

        logger.info(
            "DPostP.seal: %d rows × %d feat → %d B wire "
            "(plain=%d B, gzip=%d B, ratio=%.2f).",
            X.shape[0], X.shape[1], len(wire),
            len(serialised), len(compressed),
            len(compressed) / max(len(serialised), 1),
        )
        return SealedPayload(
            data=wire, aad=aad,
            n_rows=int(X.shape[0]), n_features=int(X.shape[1]),
        )

    def unseal(
        self,
        payload: "SealedPayload | bytes",
        *,
        expected_model_version: Optional[str] = None,
        expected_sender_id: Optional[str] = None,
        skew_window_s: int = _DEFAULT_SKEW_WINDOW_S,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Verify and decrypt a sealed training export.

        All integrity and freshness checks are performed *before* the
        AEAD decrypt returns plaintext; a single malformed byte produces
        ``DPostPAuthenticationError`` rather than corrupted numpy arrays.
        """
        if self._key is None:
            raise DPostPKeyError(
                "DPostP.unseal: transport key unavailable "
                f"(set {self.key_env_var!r} or pass key=... to the constructor)."
            )

        wire = payload.data if isinstance(payload, SealedPayload) else bytes(payload)
        min_len = 4 + 1 + 2 + _NONCE_LEN + _TAG_LEN
        if len(wire) < min_len:
            raise DPostPAuthenticationError(
                f"unseal: payload too short ({len(wire)} < {min_len} B)."
            )

        # ── Parse header ──────────────────────────────────────────────
        if wire[:4] != _MAGIC:
            raise DPostPAuthenticationError("unseal: bad magic.")
        version = wire[4]
        if version != _VERSION:
            raise DPostPAuthenticationError(
                f"unseal: unsupported wire version {version} (expected {_VERSION})."
            )

        aad_len = struct.unpack(">H", wire[5:7])[0]
        aad_end = 7 + aad_len
        if aad_end + _NONCE_LEN + _TAG_LEN > len(wire):
            raise DPostPAuthenticationError("unseal: truncated AAD/nonce.")

        aad_bytes = wire[7:aad_end]
        nonce = wire[aad_end:aad_end + _NONCE_LEN]
        ciphertext = wire[aad_end + _NONCE_LEN:]

        try:
            aad = json.loads(aad_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DPostPAuthenticationError(f"unseal: malformed AAD — {exc}") from exc

        # ── Validate AAD fields BEFORE decrypt (cheap, hands-off) ─────
        if aad.get("algo_id") != _ALGO_ID:
            raise DPostPAuthenticationError(
                f"unseal: algorithm mismatch (payload={aad.get('algo_id')!r}, "
                f"expected={_ALGO_ID!r})."
            )
        if expected_model_version is not None and aad.get("model_version") != expected_model_version:
            raise DPostPAuthenticationError(
                f"unseal: model_version mismatch "
                f"(payload={aad.get('model_version')!r}, "
                f"expected={expected_model_version!r})."
            )
        if expected_sender_id is not None and aad.get("sender_id") != expected_sender_id:
            raise DPostPAuthenticationError(
                f"unseal: sender_id mismatch "
                f"(payload={aad.get('sender_id')!r}, "
                f"expected={expected_sender_id!r})."
            )
        ts = aad.get("timestamp")
        if not isinstance(ts, (int, float)):
            raise DPostPAuthenticationError("unseal: missing/invalid AAD timestamp.")
        now = time.time()
        if abs(now - ts) > skew_window_s:
            raise DPostPAuthenticationError(
                f"unseal: timestamp outside ±{skew_window_s}s window "
                f"(payload={ts}, now={now:.0f})."
            )

        # ── Decrypt ───────────────────────────────────────────────────
        try:
            compressed = self._aesgcm().decrypt(nonce, ciphertext, aad_bytes)
        except Exception as exc:
            # cryptography.exceptions.InvalidTag and friends are thrown
            # from the same inheritance root; we don't need to import
            # them just to re-raise as our own class.
            raise DPostPAuthenticationError(f"unseal: AEAD decrypt failed — {exc}") from exc

        # ── Decompress + deserialise ──────────────────────────────────
        try:
            serialised = gzip.decompress(compressed)
            with np.load(io.BytesIO(serialised)) as npz:
                X = np.asarray(npz["X"], dtype=float)
                y = np.asarray(npz["y"], dtype=float).ravel()
        except Exception as exc:
            raise DPostPAuthenticationError(
                f"unseal: decompress/deserialise failed — {exc}"
            ) from exc

        logger.info(
            "DPostP.unseal: recovered %d rows × %d feat from %d B wire.",
            X.shape[0], X.shape[1], len(wire),
        )
        return X, y

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _canonical_json(obj: dict) -> bytes:
        """Stable JSON bytes — sorted keys, no whitespace, utf-8."""
        return json.dumps(
            obj, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")

    @staticmethod
    def _resolve_key(
        explicit: Optional[bytes], env_var: str
    ) -> Optional[bytes]:
        """
        Prefer the explicit key when supplied; otherwise base64-decode
        the env var.  Returns None if neither source provided a key —
        this is valid in cleaning-only mode.
        """
        if explicit is not None:
            if not isinstance(explicit, (bytes, bytearray)):
                raise DPostPKeyError(
                    f"DPostP key must be bytes (got {type(explicit).__name__})."
                )
            if len(explicit) != _KEY_LEN:
                raise DPostPKeyError(
                    f"DPostP key must be {_KEY_LEN} bytes (got {len(explicit)})."
                )
            return bytes(explicit)

        raw = os.environ.get(env_var)
        if not raw:
            return None
        try:
            decoded = base64.b64decode(raw, validate=True)
        except Exception as exc:
            raise DPostPKeyError(
                f"DPostP: env var {env_var!r} is not valid base64 — {exc}"
            ) from exc
        if len(decoded) != _KEY_LEN:
            raise DPostPKeyError(
                f"DPostP: env var {env_var!r} decodes to {len(decoded)} bytes "
                f"(expected {_KEY_LEN})."
            )
        return decoded

    def _aesgcm(self):
        """
        Lazy import + cache AESGCM instance.  Kept lazy so the module is
        importable on systems where ``cryptography`` isn't installed,
        as long as only cleaning/reference methods are used.
        """
        if self._key is None:
            raise DPostPKeyError(
                "DPostP: internal — _aesgcm called with no key resolved."
            )
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError as exc:
            raise DPostPError(
                "DPostP: cryptography package is required for seal/unseal — "
                "run `pip install cryptography`."
            ) from exc
        # AESGCM objects are cheap; caching one avoids repeated key-schedule
        # setup when a process seals hundreds of batches.
        cached = getattr(self, "_aesgcm_cached", None)
        if cached is None:
            cached = AESGCM(self._key)
            self._aesgcm_cached = cached
        return cached

    @staticmethod
    def generate_key_b64() -> str:
        """
        Helper for scripts / CI to mint a fresh 256-bit key.

        Returns
        -------
        str
            base64-encoded 32-byte key suitable for ``DPOSTP_KEY``.
        """
        return base64.b64encode(os.urandom(_KEY_LEN)).decode("ascii")

    def __repr__(self) -> str:
        return (
            f"DPostP(z_clip={self.z_clip}, min_ref_rows={self.min_ref_rows}, "
            f"sender_id={self.sender_id!r}, "
            f"transport_key={'set' if self._key is not None else 'unset'})"
        )
