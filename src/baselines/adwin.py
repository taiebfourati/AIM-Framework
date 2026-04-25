"""
baselines/adwin.py — ADWIN drift detector for the CRIT-7 comparison.

Implements the *Adaptive Window* (ADWIN) sequential change detector of
Bifet & Gavaldà, "Learning from Time-Changing Data with Adaptive
Windowing", SDM 2007.  The algorithm maintains a sliding window of
real-valued observations whose length adapts: whenever the window can
be split into two sub-windows whose means differ by more than the
ADWIN threshold (a function of the confidence parameter ``delta`` and
the sub-window sizes), the older sub-window is dropped and a change is
reported.  We use the classical "Hoeffding-bound + harmonic test"
formulation from Section~3 of the paper:

    eps_cut = sqrt( (1 / (2 m)) * ln(4 |W| / delta) )

with ``m`` the harmonic mean of the two sub-window sizes
( m = 1 / (1/n0 + 1/n1) ).  A change is signalled if
``|mean_left - mean_right| > eps_cut`` for ANY split of the current
window.

Implementation notes
--------------------
This is a faithful reference implementation, not the
linked-list / log-bucket variant used by MOA.  We keep the window as a
plain Python ``list``: this is O(n) per update but the windows in our
experiments stay below 1000 samples, so the constant is small.  The
algorithm is feature-stream agnostic: feed it any scalar observation
(an error indicator, a single feature, or the L2 norm of a feature
vector) and it tracks change in that scalar.

Public surface
--------------
``ADWIN(delta=0.002).update(x) -> bool``  — feed one observation,
returns True if a change point was just declared (and the older half
of the window has been dropped).

``ADWIN.window_size`` — current window length, useful for diagnostics.

``ADWIN.reset()``     — re-initialise the detector after a downstream
                        intervention (e.g. a model retrain).
"""
from __future__ import annotations

import math
from typing import List


class ADWIN:
    """Reference ADWIN implementation (Bifet & Gavaldà, 2007)."""

    def __init__(self, delta: float = 0.002, max_window: int = 1024) -> None:
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        self.delta = float(delta)
        self.max_window = int(max_window)
        self._window: List[float] = []

    # ------------------------------------------------------------------ helpers
    @property
    def window_size(self) -> int:
        return len(self._window)

    def reset(self) -> None:
        self._window.clear()

    def _eps_cut(self, n0: int, n1: int) -> float:
        """The ADWIN cut threshold (formula (1) of the paper)."""
        # Harmonic mean of the two sub-window sizes.
        m_inv = (1.0 / n0) + (1.0 / n1)
        m = 1.0 / m_inv
        n_total = n0 + n1
        delta_prime = self.delta / max(1, n_total)  # union bound
        # eps_cut = sqrt( (1/(2 m)) * ln(2 / delta_prime) )
        # The original paper uses ln(2/delta'); MOA uses ln(2/delta) and
        # absorbs the union bound elsewhere.  We follow the paper.
        return math.sqrt((1.0 / (2.0 * m)) * math.log(2.0 / delta_prime))

    # ------------------------------------------------------------------ update
    def update(self, x: float) -> bool:
        """Append ``x`` to the window and test for a change point.

        Returns True iff a cut was just made (i.e. drift detected and
        the older sub-window has been dropped).  When True is returned
        the window has already been compacted to the post-change tail.
        """
        self._window.append(float(x))
        # Cap the window so adversarial-streaming pathologies cannot
        # blow up memory.  The cap is generous (1024) — well above any
        # practical detection latency in our scenario.
        if len(self._window) > self.max_window:
            self._window = self._window[-self.max_window:]

        n = len(self._window)
        if n < 16:  # need a minimum number of samples to test cuts
            return False

        # Search for a cut.  We scan all O(n) split points; for each
        # split, compute the two sub-window means and check the
        # ADWIN cut bound.  As soon as one split fires, drop the older
        # half and return True.
        prefix_sum = 0.0
        total_sum = sum(self._window)
        for i in range(1, n):
            prefix_sum += self._window[i - 1]
            n0 = i
            n1 = n - i
            if n0 < 5 or n1 < 5:
                continue
            mu0 = prefix_sum / n0
            mu1 = (total_sum - prefix_sum) / n1
            eps = self._eps_cut(n0, n1)
            if abs(mu0 - mu1) > eps:
                # Drift detected at split point ``i`` — keep only the
                # post-change tail.
                self._window = self._window[i:]
                return True
        return False

    def __repr__(self) -> str:
        return f"ADWIN(delta={self.delta}, |W|={len(self._window)})"
