"""
buffers.py — Long Input Buffer (LIB) and Long Output Buffer (LOB)

These two buffers are the backbone of the RTP observer. They maintain a
sliding window of recent AIF inputs and outputs, which all four detectors
(DDD, DPD, CDD, CPD) read from.

Paper reference: Section IV-C
  "Long Input Buffer (LIB) is connected to the Short Input Buffer (SIB) of
   AIF, storing the same input data and relevant network indicators for
   longer periods of time (data history)."
  "Long Output Buffer (LOB) stores the output of MLI data. It is
   synchronised with LIB."
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import numpy as np

# ``is_finite_obs`` / ``ValidationStats`` live in ``aif.input_validation``
# and are used by :meth:`BufferPair.push_batch` to drop non-finite rows
# before they reach the detectors.  See the security-auditor HIGH finding
# in ``aif/input_validation.py`` for the full rationale.
from aif.input_validation import ValidationStats, is_finite_obs


# ---------------------------------------------------------------------------
# Stamped sample — every entry is timestamped so detectors can do
# time-windowed analysis (e.g. "last 5 minutes" vs "last 60 seconds").
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    """A single timestamped observation stored inside LIB or LOB."""
    value: np.ndarray          # feature vector (LIB) or prediction (LOB)
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)  # optional KPIs, labels, etc.

    def age(self) -> float:
        """Seconds since this sample was recorded."""
        return time.time() - self.timestamp


# ---------------------------------------------------------------------------
# Base buffer — shared logic for LIB and LOB
# ---------------------------------------------------------------------------

class _RollingBuffer:
    """
    Thread-safe circular buffer backed by collections.deque.

    Concurrency (security-auditor HIGH #4 finding)
    ----------------------------------------------
    The backing ``deque`` is itself atomic for individual append operations,
    but the *logical* invariant callers rely on — "len(LIB) == len(LOB)
    right now, and the last row of each is the same observation" — cannot be
    enforced at the single-deque level.  Readers doing ``get_values(n)``
    followed by ``get_flat_values(n)`` on the paired buffer would see the
    two deques in different states if a writer thread squeezed an append
    (or a size-cap trim) between the two reads.  We therefore guard every
    mutating call AND every reading call with a per-buffer ``RLock`` so
    snapshots are atomic.  :class:`BufferPair` layers an outer ``RLock`` on
    top to make (LIB, LOB) pair reads atomic.  ``RLock`` (recursive) is
    used so a thread already inside ``BufferPair._lock`` may also acquire
    the inner LIB/LOB locks without self-deadlocking.

    Parameters
    ----------
    maxlen : int
        Maximum number of samples to keep. Oldest samples are evicted
        automatically once the buffer is full (FIFO).
    name : str
        Human-readable name used in __repr__ and logging.
    """

    def __init__(self, maxlen: int, name: str = "Buffer") -> None:
        if maxlen < 2:
            raise ValueError(f"maxlen must be >= 2, got {maxlen}")
        self._buf: deque[Sample] = deque(maxlen=maxlen)
        self.maxlen = maxlen
        self.name = name
        # Per-buffer reentrant lock — guards every mutation AND every
        # read so snapshots cannot be torn by a concurrent writer.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Core write / read API
    # ------------------------------------------------------------------

    def push(self, value: np.ndarray, metadata: Optional[dict] = None) -> Sample:
        """
        Append one sample to the buffer.

        Parameters
        ----------
        value : np.ndarray
            The raw array to store (feature vector or prediction).
        metadata : dict, optional
            Any extra context (e.g. {"kpi_latency": 12.3}).

        Returns
        -------
        Sample
            The Sample object that was stored.
        """
        value = np.asarray(value, dtype=float)
        sample = Sample(value=value, metadata=metadata or {})
        with self._lock:
            self._buf.append(sample)
        return sample

    def get_values(self, n: Optional[int] = None) -> np.ndarray:
        """
        Return the last *n* sample values as a 2-D numpy array
        of shape (n, n_features).  If n is None, returns everything.

        The most-recent sample is at index [-1].  The returned array is a
        fresh copy — never a view on the live deque — so callers can read
        it safely after the lock is released.
        """
        with self._lock:
            samples = list(self._buf) if n is None else list(self._buf)[-n:]
            if not samples:
                return np.empty((0,))
            # np.stack already produces a fresh contiguous array.
            return np.stack([s.value for s in samples])

    def get_samples(self, n: Optional[int] = None) -> list[Sample]:
        """Return the last *n* Sample objects (with timestamps + metadata)."""
        with self._lock:
            if n is None:
                return list(self._buf)
            return list(self._buf)[-n:]

    def get_window(self, seconds: float) -> np.ndarray:
        """
        Return all samples recorded within the last *seconds* seconds.
        Useful for time-based drift detection windows.
        """
        cutoff = time.time() - seconds
        with self._lock:
            recent = [s for s in self._buf if s.timestamp >= cutoff]
        if not recent:
            return np.empty((0,))
        return np.stack([s.value for s in recent])

    def clear(self) -> None:
        """Empty the buffer (e.g. after poisoning is confirmed)."""
        with self._lock:
            self._buf.clear()

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def is_full(self) -> bool:
        with self._lock:
            return len(self._buf) == self.maxlen

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._buf) == 0

    def __repr__(self) -> str:
        return (
            f"{self.name}(len={len(self)}/{self.maxlen}, "
            f"full={self.is_full()})"
        )

    # ------------------------------------------------------------------
    # Statistical helpers — used directly by detectors
    # ------------------------------------------------------------------

    def mean(self, n: Optional[int] = None) -> np.ndarray:
        """Column-wise mean of the last *n* samples."""
        # get_values() takes the lock; mean is computed on the fresh copy.
        return self.get_values(n).mean(axis=0)

    def std(self, n: Optional[int] = None) -> np.ndarray:
        """Column-wise standard deviation of the last *n* samples."""
        return self.get_values(n).std(axis=0)

    def split_reference_recent(
        self, reference_size: int, recent_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Split the buffer into an older 'reference' window and a newer
        'recent' window. Used by drift detectors that compare the two.

        Returns (reference_array, recent_array).
        Raises ValueError if the buffer does not have enough samples.
        """
        needed = reference_size + recent_size
        # Hold the lock across the size-check + slice so the buffer
        # cannot be trimmed between the two and produce a mismatched
        # (ref, rec) pair.  RLock: get_values() can re-acquire safely.
        with self._lock:
            if len(self._buf) < needed:
                raise ValueError(
                    f"{self.name} has only {len(self._buf)} samples but "
                    f"reference_size + recent_size = {needed} were requested."
                )
            all_vals = self.get_values()
            ref = all_vals[-(reference_size + recent_size): -recent_size]
            rec = all_vals[-recent_size:]
            return ref, rec


# ---------------------------------------------------------------------------
# LIB — Long Input Buffer
# ---------------------------------------------------------------------------

class LIB(_RollingBuffer):
    """
    Long Input Buffer.

    Stores a sliding history of AIF *input* feature vectors.
    Also accepts optional network KPI metadata alongside each sample.

    Usage
    -----
    >>> lib = LIB(maxlen=1000)
    >>> lib.push(np.array([0.1, 0.4, 0.9]), metadata={"kpi_latency": 11.2})
    >>> X_ref, X_rec = lib.split_reference_recent(500, 100)
    """

    def __init__(self, maxlen: int = 1000) -> None:
        super().__init__(maxlen=maxlen, name="LIB")

    def push_batch(
        self, X: np.ndarray, metadata_list: Optional[list[dict]] = None
    ) -> list[Sample]:
        """
        Push multiple input samples at once (e.g. a mini-batch from the AIF).

        The whole batch is appended under a single lock acquire so a
        concurrent reader sees either the full pre-batch buffer or the
        full post-batch buffer — never a half-applied batch.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        metadata_list : list of dicts, optional
            One metadata dict per sample.
        """
        X = np.atleast_2d(X)
        metadata_list = metadata_list or [{} for _ in range(len(X))]
        # RLock: self.push() re-acquires the same lock — the outer
        # ``with`` keeps the batch atomic; inner re-acquires are cheap.
        with self._lock:
            return [self.push(row, meta) for row, meta in zip(X, metadata_list)]


# ---------------------------------------------------------------------------
# LOB — Long Output Buffer
# ---------------------------------------------------------------------------

class LOB(_RollingBuffer):
    """
    Long Output Buffer.

    Stores a sliding history of AIF *output* predictions.
    Must stay synchronised with LIB — push one LOB sample for every
    LIB sample to keep indices aligned for concept drift detection.

    Usage
    -----
    >>> lob = LOB(maxlen=1000)
    >>> lob.push(np.array([1]))        # binary class prediction
    >>> lob.push(np.array([3.14]))     # regression output
    """

    def __init__(self, maxlen: int = 1000) -> None:
        super().__init__(maxlen=maxlen, name="LOB")

    def push_batch(self, y: np.ndarray) -> list[Sample]:
        """
        Push multiple output predictions at once.

        Held under a single lock acquire so readers see the full batch
        atomically (see :class:`_RollingBuffer` concurrency docs).

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,) or (n_samples, n_outputs)
        """
        y = np.atleast_2d(y) if y.ndim > 1 else y.reshape(-1, 1)
        with self._lock:
            return [self.push(row) for row in y]

    def get_flat_values(self, n: Optional[int] = None) -> np.ndarray:
        """
        Return predictions as a 1-D array (convenient for single-output
        classifiers / regressors).
        """
        # get_values() acquires the lock internally and returns a fresh
        # copy — the ravel below operates on that copy, no extra lock
        # needed.
        vals = self.get_values(n)
        if vals.ndim == 2 and vals.shape[1] == 1:
            return vals.ravel()
        return vals


# ---------------------------------------------------------------------------
# Synchronised pair — keeps LIB and LOB in lock-step
# ---------------------------------------------------------------------------

class BufferPair:
    """
    Convenience wrapper that pushes to LIB, LOB, and an optional YGT
    (ground-truth label) buffer together and keeps them aligned.

    This is the object RTP should hold and pass to all detectors.
    Ground-truth labels are stored whenever the caller provides them
    via ``observe(x, y_true=...)`` — missing entries are recorded as
    ``NaN`` so the three buffers stay index-synchronised.  ATM reads
    YGT to retrain on real labels when they are available, falling
    back to LOB pseudo-labels otherwise.

    Usage
    -----
    >>> pair = BufferPair(maxlen=1000)
    >>> pair.push(x=np.array([0.1, 0.4]), y=np.array([1]))
    >>> pair.lib.get_values()       # inputs
    >>> pair.lob.get_flat_values()  # predictions
    >>> pair.ygt.get_flat_values()  # ground truth (NaN where unknown)
    """

    def __init__(self, maxlen: int = 1000) -> None:
        self.lib = LIB(maxlen=maxlen)
        self.lob = LOB(maxlen=maxlen)
        # YGT piggy-backs on LOB's shape (single-value rolling buffer)
        # but stores ground-truth labels.  Callers must push NaN where
        # no label is available so indices stay aligned with LIB/LOB.
        self.ygt = LOB(maxlen=maxlen)
        self.ygt.name = "YGT"

        # Pair-level reentrant lock — guards three-way-paired mutations
        # (``push``, ``push_batch``) AND paired reads (``snapshot``) so
        # writers cannot split a pair between LIB/LOB/YGT and readers
        # cannot tear an atomic (X, y) snapshot.  Reentrant so a method
        # holding this lock can safely call into LIB/LOB's own per-buffer
        # locks (different lock objects, but the same thread recursing
        # through its own pair lock is what RLock buys us).  Security-
        # auditor HIGH #4 finding.
        self._lock = threading.RLock()

        # Ingress validation hooks.  The RTP wires its own
        # :class:`ValidationStats` and an ``INVALID_BATCH`` event emitter
        # here so that rows with NaN/inf/extreme values are dropped at
        # ``push_batch`` and never reach detector state.  When no hooks
        # are wired (e.g. a standalone BufferPair used in unit tests)
        # we fall back to a local stats instance so ``push_batch``
        # remains fully self-contained — the validation still runs, just
        # without event emission.
        self._validation_stats: ValidationStats = ValidationStats()
        self._invalid_batch_event_cb: Optional[Callable[[dict], None]] = None

    def set_validation_hooks(
        self,
        stats: ValidationStats,
        event_callback: Optional[Callable[[dict], None]] = None,
    ) -> None:
        """
        Wire an external :class:`ValidationStats` and (optionally) an
        event-emitter callback.  Called once at RTP construction time so
        that per-reason drop counters aggregate across observe() and
        push_batch() paths.

        ``event_callback`` receives a ``details`` dict when a batch push
        rejects one or more rows.  Leaving it ``None`` disables event
        emission while still running the per-row validation.
        """
        self._validation_stats = stats
        self._invalid_batch_event_cb = event_callback

    def push(
        self,
        x: np.ndarray,
        y: Any,
        metadata: Optional[dict] = None,
        y_true: Any = None,
    ) -> tuple[Sample, Sample]:
        """
        Push one (input, prediction, [ground_truth]) triple simultaneously.

        Parameters
        ----------
        x : np.ndarray
            AIF input feature vector.
        y : scalar or np.ndarray
            AIF prediction (class label, probability, regression value…).
        metadata : dict, optional
            KPI context attached to the LIB sample.
        y_true : scalar or np.ndarray, optional
            Ground-truth label for this step.  Stored in YGT; when
            ``None`` a ``NaN`` sentinel is recorded instead so YGT's
            index stays aligned with LIB and LOB.
        """
        y = np.atleast_1d(np.asarray(y, dtype=float))
        if y_true is None:
            y_true_arr = np.array([np.nan], dtype=float)
        else:
            y_true_arr = np.atleast_1d(np.asarray(y_true, dtype=float))
        # Atomic across LIB / LOB / YGT — a concurrent ``snapshot`` call
        # cannot observe the buffers mid-triple.
        with self._lock:
            lib_sample = self.lib.push(x, metadata)
            lob_sample = self.lob.push(y)
            self.ygt.push(y_true_arr)
        return lib_sample, lob_sample

    def push_batch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metadata_list: Optional[list[dict]] = None,
        y_true: Optional[np.ndarray] = None,
    ) -> None:
        """
        Push a batch of (input, prediction, [ground_truth]) triples.

        Ingress validation (security-auditor HIGH finding): each row is
        individually checked with
        :func:`aif.input_validation.is_finite_obs` against ``X[i]`` and
        ``y[i]`` (or ``y_true[i]`` when supplied — the label is the
        distribution attackers tamper with most aggressively).  Rows
        that fail validation are dropped; the remaining good rows go
        through the regular LIB / LOB / YGT append path.  Per-reason
        drop counters are accumulated in the shared
        :class:`ValidationStats` so callers can inspect the breakdown
        after the batch completes.  When the whole batch is bad we
        return early — the detectors never see any rows from this call.
        One ``INVALID_BATCH`` event (carrying the per-reason breakdown)
        is emitted whenever at least one row was dropped.
        """
        X = np.atleast_2d(X)
        y_arr = np.asarray(y)
        n_in = X.shape[0]

        # -- Ingress validation ----------------------------------------
        if n_in == 0:
            # Nothing to do — and, importantly, nothing to validate.
            # Skipping the event emission mirrors observe()'s behaviour
            # on an accepted sample: no news is good news.
            return

        y_true_arr: Optional[np.ndarray]
        if y_true is None:
            y_true_arr = None
        else:
            y_true_arr = np.asarray(y_true).ravel()

        # NOTE: the validation / filtering below is pure (it only reads
        # the caller-supplied arrays) so it does not need the pair lock.
        # The lock is only acquired around the three ``push_batch`` calls
        # at the very end so LIB / LOB / YGT land as one atomic unit and
        # a concurrent ``snapshot`` cannot split them.

        # We validate per-row against whichever signal the caller gave
        # us.  ``y_true`` is the preferred anchor because it IS the
        # attacker's target (label flipping, shadow poisoning etc.); if
        # the caller only provided predictions we fall back to those.
        # Either way a non-finite value fails the row.
        y_for_val = y_true_arr if y_true_arr is not None else y_arr.ravel()

        valid_mask = np.ones(n_in, dtype=bool)
        reasons: list[Optional[str]] = [None] * n_in
        per_reason_counts: dict[str, int] = {}

        for i in range(n_in):
            # Align shapes cheaply — y_for_val may be shorter than X
            # when callers use the legacy "just push predictions"
            # signature; in that case we validate X alone for the
            # tail rows, which is the safe default.
            yi = y_for_val[i] if i < len(y_for_val) else None
            ok, reason = is_finite_obs(X[i], yi)
            self._validation_stats.record(reason)
            if not ok:
                valid_mask[i] = False
                reasons[i] = reason
                per_reason_counts[reason] = per_reason_counts.get(reason, 0) + 1

        n_dropped = int((~valid_mask).sum())
        n_kept = n_in - n_dropped

        # -- Whole-batch rejection path --------------------------------
        if n_kept == 0:
            if self._invalid_batch_event_cb is not None:
                self._invalid_batch_event_cb({
                    "n_in": n_in,
                    "n_dropped": n_dropped,
                    "n_kept": 0,
                    "by_reason": per_reason_counts,
                    "all_bad": True,
                })
            return

        # -- Emit event if any rows were dropped -----------------------
        if n_dropped > 0 and self._invalid_batch_event_cb is not None:
            self._invalid_batch_event_cb({
                "n_in": n_in,
                "n_dropped": n_dropped,
                "n_kept": n_kept,
                "by_reason": per_reason_counts,
                "all_bad": False,
            })

        # -- Apply the mask to every aligned container -----------------
        X_kept = X[valid_mask]
        if y_arr.ndim > 1:
            y_kept = y_arr[valid_mask]
        else:
            y_kept = y_arr.ravel()[valid_mask]
        metadata_kept: Optional[list[dict]] = None
        if metadata_list is not None:
            metadata_kept = [md for md, ok in zip(metadata_list, valid_mask) if ok]
        y_true_kept: Optional[np.ndarray]
        if y_true_arr is None:
            y_true_kept = None
        else:
            y_true_kept = y_true_arr[valid_mask]

        # Atomic three-way append — LIB/LOB/YGT land together so the pair
        # invariants (len(LIB) == len(LOB) == len(YGT), same-index row
        # corresponds to the same observation) are preserved even under
        # concurrent readers.
        with self._lock:
            self.lib.push_batch(X_kept, metadata_kept)
            self.lob.push_batch(y_kept)
            if y_true_kept is None:
                self.ygt.push_batch(np.full(len(X_kept), np.nan, dtype=float))
            else:
                self.ygt.push_batch(y_true_kept)

    def __len__(self) -> int:
        return len(self.lib)

    def is_ready(self, min_samples: int) -> bool:
        """True once both buffers have at least *min_samples* observations."""
        # Take the pair lock so a concurrent push cannot leave LIB one
        # row ahead of LOB at the moment we check readiness.
        with self._lock:
            return len(self.lib) >= min_samples and len(self.lob) >= min_samples

    def snapshot(
        self, n: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Atomically read the last *n* (X, y) rows from LIB and LOB.

        The pair lock is acquired once and held across both reads, so the
        two arrays are guaranteed to come from the same buffer state —
        same length, and ``X[i]`` is paired with the ``y[i]`` that was
        pushed alongside it.  Without this (e.g. the legacy
        ``lib.get_values(n)`` + ``lob.get_flat_values(n)`` two-read
        pattern) a concurrent writer could squeeze an append + size-cap
        trim between the two reads and hand detectors a mismatched row.

        Security-auditor HIGH #5 finding: the CPD refit path in
        ``rtp.notify_model_updated`` uses this method in place of the
        two independent reads so ``sklearn.clone(...).fit(X, y)`` cannot
        raise ``ValueError`` (or, worse, silently train on misaligned
        pairs) when the buffer rolls during a burst ingestion.

        Parameters
        ----------
        n : int, optional
            Number of most-recent rows to return.  ``None`` (default)
            returns the full buffer contents.

        Returns
        -------
        (X, y) : tuple of np.ndarray
            ``X`` is 2-D with shape ``(k, n_features)``; ``y`` is 1-D
            with shape ``(k,)`` where ``k <= n`` (fewer if the buffer
            has not yet filled).  Both are fresh copies, never views.
        """
        with self._lock:
            # Both reads take their own per-buffer lock too (RLock ->
            # cheap re-entry on the same thread), but the outer pair
            # lock guarantees neither buffer mutates between them.
            X = self.lib.get_values(n)
            y = self.lob.get_flat_values(n)
        # Final safety net: if an upstream invariant violation has left
        # LIB / LOB with different lengths despite the lock (e.g. a
        # caller reached in and pushed to one side directly), clip to
        # the shorter so the returned pair is still consistent rather
        # than crashing the detectors with a shape mismatch.
        if X.shape[0] != y.shape[0]:
            k = min(X.shape[0], y.shape[0])
            X = X[-k:] if k > 0 else np.empty((0,))
            y = y[-k:] if k > 0 else np.empty((0,))
        return X, y

    def snapshot_triple(
        self, n: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Atomically read the last *n* (X, y, y_gt) rows from LIB, LOB and YGT.

        Identical contract to :meth:`snapshot` but extends it to the third
        buffer (ground-truth labels) under the SAME ``self._lock`` hold.
        The previous CPD shadow-refit path in
        ``rtp.notify_model_updated`` did this in two steps:

        1. ``self.buffers.snapshot(cpd_take)`` -> ``(X, lob_tail)``  (locked)
        2. ``ygt.get_flat_values(cpd_take)``                          (UNLOCKED)

        Between step 1 and step 2 a concurrent ``push_batch`` could trim
        YGT one row, leaving a misaligned triple where ``y_gt[i]`` no
        longer corresponds to ``X[i]``.  The downstream
        ``np.where(~np.isnan(y_gt_tail), y_gt_tail, lob_tail)`` then mixes
        ground-truth row *i* with predictions for row *i+1* — feeding the
        CPD shadow a quietly-shifted concept reference.

        QA review post-ship Risk 1: this method is the single-lock fix.

        Parameters
        ----------
        n : int, optional
            Number of most-recent rows to return.  ``None`` (default)
            returns the full buffer contents.

        Returns
        -------
        (X, y, y_gt) : tuple of np.ndarray
            All three arrays are guaranteed to have the same first-axis
            length and to come from the same buffer state.  ``y_gt``
            entries are ``np.nan`` for rows where no ground-truth label
            was supplied (matching the YGT push-time NaN sentinel).
        """
        with self._lock:
            X = self.lib.get_values(n)
            y = self.lob.get_flat_values(n)
            y_gt = self.ygt.get_flat_values(n)
        # Defensive clip: mirror snapshot()'s safety net so a stale
        # direct-write to one buffer cannot crash callers with a shape
        # mismatch.  Take the shortest of the three so X[i]/y[i]/y_gt[i]
        # are always the same row.
        lengths = (X.shape[0], y.shape[0], y_gt.shape[0])
        if len(set(lengths)) != 1:
            k = min(lengths)
            X = X[-k:] if k > 0 else np.empty((0,))
            y = y[-k:] if k > 0 else np.empty((0,))
            y_gt = y_gt[-k:] if k > 0 else np.empty((0,))
        return X, y, y_gt

    def __repr__(self) -> str:
        return f"BufferPair({self.lib}, {self.lob}, {self.ygt})"
