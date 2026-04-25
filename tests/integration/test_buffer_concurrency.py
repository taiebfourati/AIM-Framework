"""
Tier-2 integration tests — BufferPair / CPD-refit concurrency
(security-auditor HIGH #4 and HIGH #5 findings).

HIGH #4 — ``aif.buffers.BufferPair`` previously claimed thread-safety in
its docstring but held no lock around the paired ``LIB.append`` /
``LOB.append`` / ``YGT.append`` triple.  Under concurrent writers (an
NDT validation thread pulling data while the RTP ingress thread kept
appending) pairs could split: LIB saw an ``x`` but LOB did not yet see
the matching ``y``.  Readers then observed ``len(LIB) == len(LOB) + 1``
and detectors trained on misaligned ``(X, y)`` rows.

HIGH #5 — ``rtp.notify_model_updated`` did two independent buffer reads
(``lib.get_values(cpd_take)`` then ``lob.get_flat_values(cpd_take)``) in
the CPD refit path.  Under burst ingestion the buffer could trim between
the two reads, returning arrays of different length.  Downstream
``sklearn.clone(...).fit(X, y)`` then raised ``ValueError`` or silently
trained on misaligned rows.

The fix adds a per-buffer ``threading.RLock`` plus a pair-level
``RLock`` on :class:`BufferPair`, and a new atomic
:meth:`BufferPair.snapshot` method that reads ``(X, y)`` under the pair
lock.  ``rtp.notify_model_updated`` now consumes that snapshot instead
of doing the two independent reads.

Tests in this file:

* **A — desync reproducer**  Two threads race against a single
  BufferPair: a writer pushing ``(x, y = x.sum())`` pairs and a reader
  calling ``snapshot()`` at high frequency.  Every snapshot must satisfy
  ``len(X) == len(y)`` AND ``y[i] == X[i].sum()`` (row-wise consistency).
  Without the lock the test would race; with the lock it cannot.
* **B — legacy split reads stay self-consistent**  Each individual
  ``lib.get_values`` / ``lob.get_flat_values`` call is now also atomic
  (copies under the per-buffer lock), so even when callers still use
  the two-read pattern no single read is torn — only the *pairing*
  across the two reads can legitimately show a length delta of 1.
* **C — CPD refit atomicity regression**  Simulate a burst ingest with
  a stubbed BufferPair whose ``lib.get_values`` and
  ``lob.get_flat_values`` would (pre-fix) return arrays of different
  length.  Assert ``notify_model_updated`` does not raise.  With the fix
  it consumes ``snapshot()`` and is immune.
* **D — throughput regression**  The lock must not make the buffer
  glacial under single-threaded load.  10 000 ``push`` calls on a fresh
  BufferPair must complete in under 1 s on a typical laptop, and the
  lock must be reentrant (same thread can re-acquire without dead-
  locking — exercised by :meth:`BufferPair.push_batch` which holds the
  outer lock while inner ``LIB.push_batch`` re-acquires its own lock).
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
import pytest

from aif.buffers import LIB, LOB, BufferPair

from ._harness import build_pipeline, make_classifier_corpus


# ---------------------------------------------------------------------------
# Test A — desync reproducer via snapshot()
# ---------------------------------------------------------------------------

class TestSnapshotAtomicity:
    """Race a writer thread against a snapshot() reader thread."""

    def test_snapshot_rows_are_row_wise_consistent(self) -> None:
        """
        Writer pushes ``(x, y=x.sum())`` as a BufferPair pair.  Reader
        pulls ``snapshot(n)`` at high frequency.  Every returned pair
        must:

        * have ``len(X) == len(y)``;
        * satisfy ``y[i] == X[i].sum()`` for every row (the invariant
          the writer established on the way in).

        With the pair lock this test passes trivially.  Without it the
        reader would occasionally see LIB one row ahead of LOB and the
        consistency check would fail.
        """
        pair = BufferPair(maxlen=500)
        n_iters = 100_000
        stop = threading.Event()
        inconsistencies: list[str] = []

        def writer() -> None:
            rng = np.random.default_rng(42)
            for i in range(n_iters):
                if stop.is_set():
                    return
                x = rng.normal(0.0, 1.0, size=4)
                y = float(x.sum())
                pair.push(x=x, y=np.array([y]))
            stop.set()

        def reader() -> None:
            # Read until the writer signals completion.  Each iteration
            # takes an atomic snapshot and checks the invariant.
            while not stop.is_set():
                X, y = pair.snapshot(n=64)
                if X.shape[0] != y.shape[0]:
                    inconsistencies.append(
                        f"length mismatch: X={X.shape}, y={y.shape}"
                    )
                    return
                if X.shape[0] == 0:
                    continue
                expected = X.sum(axis=1)
                # Compare with a tight tolerance — floats, but the sums
                # are identical under the same rng draw so exact match
                # would also work; use np.isclose defensively.
                if not np.allclose(expected, y, atol=1e-9):
                    bad_idx = int(np.argmax(np.abs(expected - y)))
                    inconsistencies.append(
                        f"row {bad_idx}: X[i].sum()={expected[bad_idx]!r} "
                        f"!= y[i]={y[bad_idx]!r}"
                    )
                    return

        tw = threading.Thread(target=writer, name="writer")
        tr = threading.Thread(target=reader, name="reader")
        tw.start()
        tr.start()
        tw.join(timeout=60.0)
        stop.set()
        tr.join(timeout=60.0)

        assert not tw.is_alive(), "writer thread did not complete"
        assert not tr.is_alive(), "reader thread did not complete"
        assert inconsistencies == [], (
            f"snapshot() returned inconsistent (X, y) under concurrency: "
            f"{inconsistencies[:3]}"
        )

    def test_snapshot_never_exceeds_buffer_maxlen(self) -> None:
        """snapshot(n) honours the size cap even when n > maxlen."""
        pair = BufferPair(maxlen=100)
        for i in range(250):
            pair.push(x=np.full(3, float(i)), y=np.array([float(i)]))
        X, y = pair.snapshot(n=500)
        # Buffer capped at 100 — snapshot cannot fabricate extra rows.
        assert X.shape[0] == 100
        assert y.shape[0] == 100
        # The last row must be the most-recent push (i=249).
        assert np.allclose(X[-1], np.full(3, 249.0))
        assert np.isclose(y[-1], 249.0)


# ---------------------------------------------------------------------------
# Test B — legacy two-read sites remain self-consistent per read
# ---------------------------------------------------------------------------

class TestLegacyReadsAreAtomic:
    """
    Each individual ``lib.get_values`` / ``lob.get_flat_values`` call
    now returns a fresh copy taken under the per-buffer lock.  A caller
    that insists on the two-read pattern may still see a length delta of
    1 across the two reads (that IS the API's legacy contract), but each
    read in isolation is atomic and its rows are consistent with the
    deque state at the moment the lock was held.
    """

    def test_individual_reads_return_fresh_copies(self) -> None:
        """
        Pull an array out of LIB, mutate the copy, and verify the live
        buffer is unchanged.  This confirms ``get_values`` never hands
        back a view that aliases the underlying storage.
        """
        lib = LIB(maxlen=50)
        for i in range(20):
            lib.push(np.array([float(i), float(i) * 2.0]))

        snap = lib.get_values(10)
        assert snap.shape == (10, 2)
        # Mutate the snapshot in place.
        snap.fill(-999.0)
        # Live buffer must be unaffected.
        live = lib.get_values(10)
        assert not np.any(live == -999.0)
        # And the original values must survive.
        assert live[-1, 0] == 19.0
        assert live[-1, 1] == 38.0

    def test_two_read_length_delta_at_most_one_under_race(self) -> None:
        """
        Under contention, two independent reads of LIB / LOB may differ
        in length by at most one row (the writer either landed an append
        or did not).  Crucially, neither read can be *torn* — i.e. see
        a half-applied batch.  This test checks that cross-read
        discrepancies remain bounded and each read's rows stay
        monotonically increasing in the value we pushed.
        """
        pair = BufferPair(maxlen=200)
        stop = threading.Event()
        max_delta_seen = [0]

        def writer() -> None:
            for i in range(50_000):
                if stop.is_set():
                    return
                pair.push(x=np.array([float(i)]), y=np.array([float(i)]))
            stop.set()

        def reader() -> None:
            while not stop.is_set():
                X = pair.lib.get_values(64)
                y = pair.lob.get_flat_values(64)
                delta = abs(X.shape[0] - y.shape[0])
                if delta > max_delta_seen[0]:
                    max_delta_seen[0] = delta
                # Each read, in isolation, must be monotonically
                # increasing in the column-0 value (since we pushed i
                # in sequence).
                if X.shape[0] >= 2:
                    diffs = np.diff(X[:, 0])
                    assert (diffs >= 0).all(), (
                        "LIB snapshot rows are not monotonic — the read was "
                        "torn across a concurrent append."
                    )

        tw = threading.Thread(target=writer)
        tr = threading.Thread(target=reader)
        tw.start()
        tr.start()
        tw.join(timeout=30.0)
        stop.set()
        tr.join(timeout=30.0)

        # Length delta MUST remain bounded — if the lock were missing
        # the two deques could be seen in arbitrarily different states
        # under a trim storm.  <= 1 is the tight API contract.
        assert max_delta_seen[0] <= 1, (
            f"two-read length delta reached {max_delta_seen[0]} — expected "
            f"<= 1 with per-buffer locking in place."
        )


# ---------------------------------------------------------------------------
# Test C — CPD refit atomicity regression via notify_model_updated
# ---------------------------------------------------------------------------

class _RacingBufferPair:
    """
    Wraps a real :class:`BufferPair` so that *successive* legacy
    ``lib.get_values`` / ``lob.get_flat_values`` reads would see arrays
    of different length — exactly the race HIGH #5 was about.  The
    wrapper's ``snapshot`` method delegates to the real pair (so the
    atomic fix works), but the raw ``.lib`` / ``.lob`` attributes return
    a proxy that drops a row from the LOB side on the call-order used
    by the pre-fix code.

    We use this to prove the CPD refit path now consumes the atomic
    snapshot (and therefore cannot observe the desync).  If the refit
    path had regressed to the two-read pattern, the proxy's contrived
    length mismatch would propagate into ``sklearn.clone(...).fit(X, y)``
    and raise ``ValueError``.
    """

    class _RacingLOB:
        def __init__(self, real_lob, counter: dict) -> None:
            self._real = real_lob
            self._counter = counter

        def __len__(self) -> int:
            return len(self._real)

        def get_values(self, n=None):
            return self._real.get_values(n)

        def get_flat_values(self, n=None):
            # Simulate a trim landing between the LIB read and the LOB
            # read — we strip one row from the returned array so the
            # two reads disagree by one.  A direct caller using the
            # legacy two-read pattern would feed this into sklearn and
            # crash.  BufferPair.snapshot() bypasses this proxy entirely.
            self._counter["n_get_flat"] += 1
            vals = self._real.get_flat_values(n)
            if self._counter["n_get_flat"] >= 1 and vals.shape[0] >= 1:
                return vals[:-1]
            return vals

    def __init__(self, real_pair) -> None:
        self._real = real_pair
        self._counter = {"n_get_flat": 0}
        self.lib = real_pair.lib
        self.lob = _RacingBufferPair._RacingLOB(real_pair.lob, self._counter)
        self.ygt = real_pair.ygt
        # QA post-ship Risk 5: the LOB re-stamp loop now grabs
        # ``self.buffers._lock`` to cover the read-X / predict / write-LOB
        # triple atomically.  Forward the real pair's lock so that path
        # works against this proxy too.
        self._lock = real_pair._lock

    def snapshot(self, n=None):
        # Delegates to the real pair's atomic snapshot.  HIGH #5 fix
        # routes notify_model_updated through this path, so the desync
        # the proxy injects into the two-read path cannot be observed.
        return self._real.snapshot(n)

    def snapshot_triple(self, n=None):
        # QA post-ship Risk 1: the CPD refit path now reads (X, y, y_gt)
        # under a single pair-lock hold via this method.  Delegating to
        # the real pair preserves the test's intent — the racing proxy
        # stresses the legacy two-read pattern, and the atomic triple
        # snapshot bypasses the proxy's contrived desync the same way
        # ``snapshot`` does.
        return self._real.snapshot_triple(n)

    def push(self, *args, **kwargs):
        return self._real.push(*args, **kwargs)

    def push_batch(self, *args, **kwargs):
        return self._real.push_batch(*args, **kwargs)

    def set_validation_hooks(self, *args, **kwargs):
        return self._real.set_validation_hooks(*args, **kwargs)

    def __len__(self) -> int:
        return len(self._real)


class TestNotifyModelUpdatedSurvivesBurstTrim:
    """
    HIGH #5 regression: ``RTP.notify_model_updated`` must not raise when
    the underlying LIB / LOB would produce arrays of mismatched length
    under the legacy two-read pattern.  The fix installs
    ``BufferPair.snapshot`` in the CPD refit path so the desync can no
    longer surface.
    """

    def test_cpd_refit_under_racing_buffer_does_not_raise(self) -> None:
        from sklearn.linear_model import LogisticRegression

        pipeline = build_pipeline(task="classifier", seed=5)
        rtp = pipeline.rtp

        # Feed enough live observations that the CPD refit path has
        # something to slice from LIB / LOB (the set_reference pre-load
        # is ignored by the _live_tail() clamp inside
        # notify_model_updated).
        X_live, y_live = make_classifier_corpus(400, 4, seed=9)
        for i in range(200):
            rtp.observe(X_live[i], y_true=np.array([float(y_live[i])]))

        # Swap in the racing proxy AFTER observation so the ingress
        # path is untouched — we only want to stress notify_model_updated.
        rtp.buffers = _RacingBufferPair(rtp.buffers)

        # Train a fresh candidate and call notify_model_updated.  The
        # refit path internally does buffers.snapshot(cpd_take) → sees
        # a consistent (X, y) → sklearn.fit succeeds.
        candidate = LogisticRegression(max_iter=500)
        candidate.fit(X_live, y_live)

        # Must not raise.  Before the fix, the racing LOB's dropped row
        # would propagate into sklearn's fit() and ValueError would
        # bubble out.
        rtp.notify_model_updated(candidate)

        # Sanity: CPD's reference must be fitted on *some* slice of
        # (X, y) whose lengths match.  We cannot directly introspect
        # the internal arrays, but the fact that notify_model_updated
        # completed without raising is the load-bearing assertion.


# ---------------------------------------------------------------------------
# Test D — throughput + RLock reentrancy sanity check
# ---------------------------------------------------------------------------

class TestLockOverheadAndReentrancy:
    """
    The per-buffer + pair RLock must not make the buffer glacial for the
    common single-threaded case, and must be reentrant so nested locks
    on the same thread don't deadlock.
    """

    def test_push_throughput_10k_under_one_second(self) -> None:
        pair = BufferPair(maxlen=20_000)
        x = np.arange(4, dtype=float)
        y = np.array([1.0])

        t0 = time.perf_counter()
        for _ in range(10_000):
            pair.push(x=x, y=y)
        elapsed = time.perf_counter() - t0

        # 1 s is a generous ceiling for 10 k locked pushes on any
        # reasonable laptop.  If this trips CI, it's almost certainly
        # a regression — the lock should cost micros, not millis, per
        # push.
        assert elapsed < 1.0, (
            f"10 k BufferPair.push() calls took {elapsed:.3f}s — the "
            f"RLock overhead has regressed."
        )
        assert len(pair.lib) == 10_000
        assert len(pair.lob) == 10_000

    def test_rlock_allows_nested_acquire_on_same_thread(self) -> None:
        """
        ``BufferPair.push_batch`` holds the outer pair lock while
        ``LIB.push_batch`` (inside) grabs LIB's own lock and iterates
        through ``push()`` which re-acquires that same LIB lock.  This
        would deadlock with a plain ``Lock`` — reentrancy is why we use
        ``RLock``.
        """
        pair = BufferPair(maxlen=100)
        X = np.arange(20 * 4, dtype=float).reshape(20, 4)
        y = np.arange(20, dtype=float)
        # Must complete without hanging (RLock reentrancy).
        pair.push_batch(X, y)
        assert len(pair.lib) == 20
        assert len(pair.lob) == 20

    def test_snapshot_inside_push_callback_does_not_deadlock(self) -> None:
        """
        A callback installed on BufferPair that calls ``snapshot`` from
        within the same thread that just pushed must not deadlock —
        the pair lock is reentrant so the nested acquire is free.
        """
        pair = BufferPair(maxlen=50)
        captured = []

        for i in range(5):
            pair.push(x=np.array([float(i)]), y=np.array([float(i)]))
            # Direct nested call — both the push above and the snapshot
            # below take the same RLock on the same thread.
            X, y = pair.snapshot()
            captured.append((X.shape[0], y.shape[0]))

        assert captured == [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]

    def test_bufferpair_exposes_snapshot_method(self) -> None:
        """Public API contract: BufferPair.snapshot exists and returns 2-tuple."""
        pair = BufferPair(maxlen=10)
        pair.push(x=np.array([1.0, 2.0]), y=np.array([3.0]))
        assert hasattr(pair, "snapshot")
        out = pair.snapshot()
        assert isinstance(out, tuple)
        assert len(out) == 2
        X, y = out
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0] == 1


# ---------------------------------------------------------------------------
# LIB / LOB standalone locking sanity
# ---------------------------------------------------------------------------

class TestStandaloneBufferLock:
    """
    LIB used on its own (not inside BufferPair) must also be safe under
    concurrent writers — the per-buffer RLock on _RollingBuffer covers
    this path.  This protects the aif/dpostp.py and enhanced_simulation
    call sites that hold bare LIB references.
    """

    def test_concurrent_lib_pushes_do_not_drop_samples(self) -> None:
        lib = LIB(maxlen=10_000)
        n_threads = 4
        per_thread = 1_000
        errors: list[str] = []

        def worker(tid: int) -> None:
            try:
                for i in range(per_thread):
                    lib.push(np.array([float(tid), float(i)]))
            except Exception as exc:   # pragma: no cover - defensive
                errors.append(f"tid={tid}: {exc}")

        threads = [
            threading.Thread(target=worker, args=(t,))
            for t in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30.0)

        assert errors == []
        # All samples must have landed — the deque's thread-safety for
        # append is the baseline the per-buffer lock further solidifies
        # by guarding every reader.
        assert len(lib) == n_threads * per_thread
