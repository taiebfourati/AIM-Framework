"""
input_validation.py — Ingress validation for the RTP observer pipeline.

Flagged as a HIGH-severity finding by the security auditor: ``rtp.observe``
and batch push helpers did not check that incoming feature vectors and
ground-truth labels were finite.  An upstream bug, a sensor fault, or an
adversarial input can feed ``float('inf')``, ``nan``, or values on the
order of ``1e300`` into the pipeline with catastrophic consequences:

* Page-Hinkley's cumulative sum overflows → the CDD detector gets stuck.
* The KS reference window becomes degenerate (NaN sort ordering is
  undefined in numpy, producing silent false negatives).
* IsolationForest refit raises or yields useless ``decision_function``
  values.
* Downstream detector state becomes unrecoverable without a manual
  reset.

This module provides a tiny, dependency-free utility that lets the RTP
drop invalid observations at the ingress layer, count them per-reason,
and expose the statistics for tests / dashboards.  The goal is
**fail closed, do not crash**: poisoned steps are dropped from the
stream entirely so detectors never see them.

Public API
----------
* ``is_finite_obs(x, y) -> (bool, reason_or_None)`` — the decision
  function.  Reason strings are stable identifiers (``"nan_x"``,
  ``"inf_x"``, ``"extreme_x"``, ``"nan_y"``, ``"inf_y"``,
  ``"extreme_y"``, ``"shape_x"``) so the RTP counter and the unit
  tests can key off them.
* ``ValidationStats`` — a small dataclass tracking the number of
  rejected observations per reason.  Exposed on the RTP so tests and
  dashboards can read it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Reason identifiers — stable strings used by callers (tests, dashboards,
# event_log entries).  Keeping them as module-level constants avoids typos
# spreading through the codebase and makes it easy to enumerate every
# reason from tests.
# ---------------------------------------------------------------------------

REASON_NAN_X     = "nan_x"
REASON_INF_X     = "inf_x"
REASON_EXTREME_X = "extreme_x"
REASON_NAN_Y     = "nan_y"
REASON_INF_Y     = "inf_y"
REASON_EXTREME_Y = "extreme_y"
REASON_SHAPE_X   = "shape_x"

ALL_REASONS: tuple[str, ...] = (
    REASON_NAN_X,
    REASON_INF_X,
    REASON_EXTREME_X,
    REASON_NAN_Y,
    REASON_INF_Y,
    REASON_EXTREME_Y,
    REASON_SHAPE_X,
)

# Magnitude threshold above which a value is considered "extreme".
# Anything past this is indistinguishable from garbage as far as the
# downstream detectors are concerned — Page-Hinkley, KS, and
# IsolationForest all lose numerical sanity long before double-precision
# actually overflows.  1e12 is conservative: real-world telemetry rarely
# exceeds 1e9 in SI units and detector-level statistics never do.
EXTREME_MAGNITUDE: float = 1e12


# ---------------------------------------------------------------------------
# Core decision function
# ---------------------------------------------------------------------------

def is_finite_obs(
    x: Union[np.ndarray, float, int],
    y: Optional[Union[np.ndarray, float, int]],
    *,
    expected_n_features: Optional[int] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Decide whether an observation ``(x, y)`` is safe to feed into the
    pipeline.

    Returns
    -------
    (True, None)
        The observation is clean — all finite, no shape surprises.
    (False, reason)
        The observation must be rejected.  ``reason`` is one of the
        stable strings exported at module scope (``"nan_x"``, ``"inf_x"``,
        ``"extreme_x"``, ``"nan_y"``, ``"inf_y"``, ``"extreme_y"``,
        ``"shape_x"``).

    Parameters
    ----------
    x : np.ndarray | float | int
        Input feature vector.  A scalar is treated as a single-feature
        sample.  Higher-dimensional arrays are flattened for the finite
        check but trigger ``"shape_x"`` if they cannot be coerced to a
        1-D feature vector.
    y : np.ndarray | float | int | None
        Ground-truth label.  ``None`` is accepted without triggering a
        rejection — the RTP's CDD proxy mode legitimately skips labels.
    expected_n_features : int, optional
        If provided, samples whose feature count differs from this
        value are rejected with ``"shape_x"``.  Leaving this ``None``
        (the default) skips the shape check — useful for the very
        first observations before the RTP has learnt its feature
        dimensionality.

    Notes
    -----
    * ``None`` entries in ``x`` raise at the ``np.asarray`` conversion
      and are caught as ``"shape_x"``.
    * The order of checks matters: NaN is checked before inf so that a
      mixed NaN/inf vector is reported as ``"nan_x"`` (more informative
      for the operator).  "Extreme" is checked last because the other
      two reasons are more specific.
    * Because the ``y`` parameter may be a 1-element array we always
      flatten it before inspection.  This mirrors what ``rtp.observe``
      does with its ``y_true`` argument.
    """
    # -- X conversion and shape guard ---------------------------------
    try:
        x_arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        return False, REASON_SHAPE_X

    # Scalars become 0-D arrays — promote to 1-D so downstream checks
    # behave identically regardless of caller input shape.
    if x_arr.ndim == 0:
        x_arr = x_arr.reshape(1)

    # Dimensionality guard: refuse anything that cannot be flattened
    # to a single-sample feature vector.  This protects against an
    # attacker feeding e.g. a 3-D tensor that would otherwise sneak
    # through the finite checks but crash the detector's 2-D contract.
    if x_arr.ndim > 1:
        # Still allow (1, n) shape — that's just an un-raveled scalar.
        if x_arr.shape[0] != 1:
            return False, REASON_SHAPE_X
        x_arr = x_arr.ravel()

    if expected_n_features is not None and x_arr.size != expected_n_features:
        return False, REASON_SHAPE_X

    # -- X finite / extreme checks ------------------------------------
    if np.isnan(x_arr).any():
        return False, REASON_NAN_X
    if np.isinf(x_arr).any():
        return False, REASON_INF_X
    if np.any(np.abs(x_arr) > EXTREME_MAGNITUDE):
        return False, REASON_EXTREME_X

    # -- Y conversion + checks ----------------------------------------
    # ``y`` being ``None`` is legitimate (CDD proxy mode); skip.
    if y is None:
        return True, None
    try:
        y_arr = np.asarray(y, dtype=float).ravel()
    except (TypeError, ValueError):
        return False, REASON_NAN_Y

    if y_arr.size == 0:
        # Empty label array behaves like "no label supplied" — accept.
        return True, None
    if np.isnan(y_arr).any():
        return False, REASON_NAN_Y
    if np.isinf(y_arr).any():
        return False, REASON_INF_Y
    if np.any(np.abs(y_arr) > EXTREME_MAGNITUDE):
        return False, REASON_EXTREME_Y

    return True, None


# ---------------------------------------------------------------------------
# Stats container
# ---------------------------------------------------------------------------

@dataclass
class ValidationStats:
    """
    Running counters of ingress rejections, partitioned by reason.

    One instance is held by each :class:`rtp.rtp.RTP` and is returned
    by :meth:`rtp.rtp.RTP.validation_stats`.  Tests can read it after a
    stream to assert both the total rejection count and the per-reason
    breakdown.  Dashboards can poll it to surface an "ingress hygiene"
    KPI.

    The container also tracks the total number of observations that
    have been validated (``total_seen``) so callers can compute a
    rejection rate without having to keep a separate counter.

    A small ring buffer (``_drops_window``) stores 1 for drops and 0 for
    accepts over the last ``window_size`` observations — used to compute
    ``dropped_fraction_last_1000`` for the ``INVALID_OBSERVATION`` event.
    Capping it at 1000 keeps the memory footprint constant regardless
    of stream length.
    """
    total_seen: int = 0
    total_rejected: int = 0
    by_reason: dict[str, int] = field(default_factory=lambda: {r: 0 for r in ALL_REASONS})

    # Rolling window used to compute ``dropped_fraction_last_1000``.
    # Bounded at 1000 entries: each entry is 1 (dropped) or 0 (accepted).
    window_size: int = 1000
    _drops_window: list[int] = field(default_factory=list)

    def record(self, reason: Optional[str]) -> None:
        """
        Log one validation outcome.

        Parameters
        ----------
        reason : str | None
            ``None`` means the observation was accepted.  A stable
            reason string from ``ALL_REASONS`` means it was rejected.
        """
        self.total_seen += 1
        if reason is None:
            self._drops_window.append(0)
        else:
            self.total_rejected += 1
            self.by_reason[reason] = self.by_reason.get(reason, 0) + 1
            self._drops_window.append(1)

        # Trim the window — cheap because we only ever trim by 1.
        if len(self._drops_window) > self.window_size:
            # Pop from the front.  ``list.pop(0)`` is O(n) but the
            # window is tiny (≤1000) so this is fine in practice.
            self._drops_window.pop(0)

    def dropped_fraction_last_1000(self) -> float:
        """
        Return the fraction of the last up-to-1000 observations that
        were dropped by the validator.

        ``0.0`` on an empty window so the caller never needs to guard
        against division-by-zero.
        """
        if not self._drops_window:
            return 0.0
        return sum(self._drops_window) / len(self._drops_window)

    def as_dict(self) -> dict:
        """Serialise for event payloads / dashboard consumption."""
        return {
            "total_seen": self.total_seen,
            "total_rejected": self.total_rejected,
            "by_reason": dict(self.by_reason),
            "dropped_fraction_last_1000": self.dropped_fraction_last_1000(),
        }
