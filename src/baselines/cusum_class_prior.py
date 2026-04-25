"""
baselines/cusum_class_prior.py — CUSUM monitor on the class prior.

A small, well-known sequential change detector for label-flip /
class-imbalance attacks.  We run two-sided cumulative sums (CUSUMs) on
the indicator variable ``y_t == 1`` (mean-equivalent to the empirical
positive-class rate).  When either CUSUM exceeds a threshold ``h`` the
detector signals a change in the class prior — exactly the symptom of
asymmetric label-flip poisoning.

The decision rule (Page, 1954) is:

    s_t^+ = max(0, s_{t-1}^+ + (y_t - mu_0 - k))
    s_t^- = max(0, s_{t-1}^- + (mu_0 - y_t - k))
    fire  iff  s_t^+ > h  OR  s_t^- > h

where ``mu_0`` is the in-control class-prior estimate, ``k`` is a
slack term (typically half of the smallest detectable shift), and
``h`` is the alarm threshold (typically 5 sigma of ``y``).

Public surface
--------------
``CUSUMClassPrior(mu_0=0.5, k=0.05, h=5.0).update(y) -> bool``
"""
from __future__ import annotations


class CUSUMClassPrior:
    """Two-sided CUSUM on the binary class label."""

    def __init__(
        self,
        mu_0: float = 0.5,
        k: float = 0.05,
        h: float = 5.0,
    ) -> None:
        self.mu_0 = float(mu_0)
        self.k = float(k)
        self.h = float(h)
        self._s_pos: float = 0.0
        self._s_neg: float = 0.0
        self._n: int = 0

    def reset(self) -> None:
        self._s_pos = 0.0
        self._s_neg = 0.0
        self._n = 0

    def update(self, y: int) -> bool:
        """Feed one binary label. Returns True if either CUSUM crossed
        ``h`` on this step. The detector does NOT auto-reset on
        firing — call :meth:`reset` after a downstream intervention.
        """
        y_f = float(y)
        self._n += 1
        self._s_pos = max(0.0, self._s_pos + (y_f - self.mu_0 - self.k))
        self._s_neg = max(0.0, self._s_neg + (self.mu_0 - y_f - self.k))
        return (self._s_pos > self.h) or (self._s_neg > self.h)

    def state(self) -> dict:
        return {
            "n": self._n,
            "s_pos": self._s_pos,
            "s_neg": self._s_neg,
            "fired": (self._s_pos > self.h) or (self._s_neg > self.h),
        }

    def __repr__(self) -> str:
        return (
            f"CUSUMClassPrior(mu0={self.mu_0}, k={self.k}, h={self.h}, "
            f"s+={self._s_pos:.2f}, s-={self._s_neg:.2f}, n={self._n})"
        )
