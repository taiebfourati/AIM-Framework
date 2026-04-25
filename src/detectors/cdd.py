"""
detectors/cdd.py — Concept Drift Detector (CDD)

Detects when the relationship between AIF inputs and outputs has changed —
i.e. the model's learned mapping is no longer valid for the current data,
even if input distributions look the same.

Paper reference: Section IV-C
  "Concept Drift Detector (CDD) analyses the LIB and LOB data for concept
   drift detection. Such detection may trigger the MToU process."

What is concept drift?
-----------------------
Data drift  (DDD) = P(X) has changed.
Concept drift (CDD) = P(Y|X) has changed — the true label/output for a
given input is no longer what the model learned.

For a classifier: accuracy degrades even though inputs look normal.
For a regressor:  residuals grow or become systematically biased.

Detection strategy
------------------
Two online monitors run on every new (input, prediction) pair pushed to
the LOB:

1. Page-Hinkley (PH) test  [implemented from scratch]
   A sequential change-point detector on a scalar performance metric
   (accuracy for classifiers, absolute error for regressors).
   Raises an alarm when the cumulative sum of deviations from the
   long-run mean exceeds a threshold λ.

   Paper refs: Sections III, IV-C — "concept drift detection can be
   performed by analysing ML model performance metrics such as accuracy,
   precision, and recall."

2. Sliding-window performance comparison
   Compares mean performance in a recent short window vs. a reference
   window. A large enough drop triggers an alert independently of PH.
   This catches slow, gradual drift that PH might absorb gradually.

Both monitors require ground-truth labels to compute real accuracy/error.
When ground truth is unavailable (common in production), the CDD falls
back to monitoring the *output distribution* of the model (LOB only),
treating a significant shift in prediction distribution as a proxy for
concept drift. This is less precise but requires no labels.

Usage
-----
# With ground truth (preferred):
cdd.update(y_pred=pred, y_true=label)

# Without ground truth (proxy mode):
cdd.update(y_pred=pred, y_true=None)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
from aif.buffers import LIB, LOB

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskType(Enum):
    CLASSIFIER = auto()
    REGRESSOR  = auto()


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CDDResult:
    """Outcome of one CDD.check() call."""
    drift_detected: bool
    # Page-Hinkley
    ph_statistic: float
    ph_threshold: float
    ph_triggered: bool
    # Sliding window
    perf_reference: float    # mean performance in reference window
    perf_recent: float       # mean performance in recent window
    perf_drop: float         # reference − recent  (positive = worse)
    window_triggered: bool
    # Meta
    n_updates: int           # total samples seen so far
    ground_truth_mode: bool  # True if real labels are available
    message: str = ""

    def __bool__(self) -> bool:
        return self.drift_detected


# ---------------------------------------------------------------------------
# Page-Hinkley sequential change-point detector
# ---------------------------------------------------------------------------

class PageHinkley:
    """
    Page-Hinkley test for detecting an *increase* in the mean of a stream.

    We feed it a *loss* metric (error rate or MAE) so that an increase
    signals performance degradation.

    Parameters
    ----------
    delta : float
        Minimum magnitude of change to detect (sensitivity).
        Smaller = more sensitive but more false alarms.
    lambda_ : float
        Detection threshold. Larger = fewer false alarms but slower detection.
    alpha : float
        Forgetting factor for the running mean (1.0 = no forgetting).
    """

    def __init__(self, delta: float = 0.005, lambda_: float = 50.0, alpha: float = 1.0) -> None:
        self.delta = delta
        self.lambda_ = lambda_
        self.alpha = alpha
        self._sum: float = 0.0
        self._x_mean: float = 0.0
        self._n: int = 0
        self._min_sum: float = 0.0
        # ``_mean_frozen`` toggles whether subsequent ``update()`` calls
        # are allowed to revise ``_x_mean``.  Set by ``freeze_mean()``.
        # Classic Page-Hinkley compares live samples against a FIXED
        # pre-change baseline — a mean that keeps drifting absorbs the
        # post-change mass into the baseline and dulls the detector.
        self._mean_frozen: bool = False

    def update(self, x: float) -> bool:
        """
        Feed one loss value. Returns True if drift is detected.
        """
        self._n += 1
        # Update running mean.
        #   - First sample: just set x_mean = x.
        #   - alpha >= 1.0: use the true cumulative mean (Welford update).
        #     This matches Page's original PH formulation — "long-run mean".
        #     The previous EWMA form with alpha=1.0 collapsed to
        #     x_mean ← x_mean (no update at all), freezing the baseline at
        #     the first sample and silently disabling drift detection when
        #     the stream starts already drifted.
        #   - alpha < 1.0: classic EWMA with forgetting — let recent
        #     samples weigh more so the baseline tracks slow distribution
        #     shifts without a full reset.
        #   - If the mean has been explicitly frozen (e.g. after warmup
        #     established the pre-change baseline), the running mean
        #     stops tracking — Page-Hinkley then measures deviations
        #     against that fixed baseline, which is what the classical
        #     change-point test assumes.
        if self._n == 1:
            self._x_mean = x
        elif self._mean_frozen:
            pass
        elif self.alpha >= 1.0:
            # Cumulative running mean: x_mean ← x_mean + (x − x_mean)/n
            self._x_mean += (x - self._x_mean) / self._n
        else:
            self._x_mean = self.alpha * self._x_mean + (1 - self.alpha) * x

        # PH cumulative sum
        self._sum += x - self._x_mean - self.delta
        self._min_sum = min(self._min_sum, self._sum)

        return (self._sum - self._min_sum) > self.lambda_

    @property
    def statistic(self) -> float:
        return self._sum - self._min_sum

    def freeze_mean(self) -> None:
        """
        Lock the running mean at its current value.

        Classical Page-Hinkley tests an increase in the mean of a stream
        against a KNOWN pre-change baseline.  The running-mean update
        during live observations implicitly treats every new sample as
        part of that baseline, which lets post-change samples bleed into
        the mean and reduces the per-step change signal.  After a
        reference / warmup phase has produced a reliable pre-change
        mean, calling this method freezes that estimate so subsequent
        ``update`` calls cleanly measure deviations FROM the baseline
        rather than deviations from a moving target.

        PH cumulative-sum state (``_sum``, ``_min_sum``) is NOT reset
        here — the caller is free to continue feeding live samples
        and the detector will accumulate deviations against the fixed
        mean.  Use ``reset()`` for a full wipe.
        """
        self._mean_frozen = True

    def reset(self) -> None:
        self._sum = 0.0
        self._x_mean = 0.0
        self._n = 0
        self._min_sum = 0.0
        self._mean_frozen = False

    def snapshot_state(self) -> dict:
        """
        Capture the Page-Hinkley internal state for later restoration.

        Returned dict mirrors every field mutated by ``update()`` so that
        a subsequent ``restore_state(snap)`` produces a PH instance that
        is behaviourally identical to the one that produced ``snap``.
        """
        return {
            "sum": float(self._sum),
            "x_mean": float(self._x_mean),
            "n": int(self._n),
            "min_sum": float(self._min_sum),
            "mean_frozen": bool(self._mean_frozen),
        }

    def restore_state(self, state: dict) -> None:
        """Restore PH internals from a ``snapshot_state()`` payload."""
        self._sum = float(state.get("sum", 0.0))
        self._x_mean = float(state.get("x_mean", 0.0))
        self._n = int(state.get("n", 0))
        self._min_sum = float(state.get("min_sum", 0.0))
        self._mean_frozen = bool(state.get("mean_frozen", False))


# ---------------------------------------------------------------------------
# CDD
# ---------------------------------------------------------------------------

class CDD:
    """
    Concept Drift Detector.

    Parameters
    ----------
    task : TaskType or str
        "classifier" or "regressor".
    reference_window : int
        Number of recent samples used as performance reference.
    recent_window : int
        Number of most-recent samples whose performance is compared against
        the reference window.
    perf_drop_threshold : float
        Minimum absolute drop in performance metric to trigger window alert.
        For classifiers: drop in accuracy (e.g. 0.10 = 10 pp drop).
        For regressors:  increase in MAE relative to reference MAE.
    ph_delta : float
        Page-Hinkley sensitivity parameter.
    ph_lambda : float
        Page-Hinkley detection threshold.
    proxy_ks_alpha : float
        KS significance level for proxy mode (no ground truth available).
    """

    def __init__(
        self,
        task: str = "classifier",
        reference_window: int = 200,
        recent_window: int = 50,
        perf_drop_threshold: float = 0.10,
        ph_delta: float = 0.005,
        ph_lambda: float = 50.0,
        proxy_ks_alpha: float = 0.01,
    ) -> None:
        if task.lower() in ("classifier", "classification"):
            self.task = TaskType.CLASSIFIER
        elif task.lower() in ("regressor", "regression"):
            self.task = TaskType.REGRESSOR
        else:
            raise ValueError(f"Unknown task type: {task!r}")

        self.reference_window = reference_window
        self.recent_window = recent_window
        self.perf_drop_threshold = perf_drop_threshold
        self.proxy_ks_alpha = proxy_ks_alpha

        self._ph = PageHinkley(delta=ph_delta, lambda_=ph_lambda)
        self._ph_triggered = False

        # Rolling buffers for performance metric
        self._perf_buf: list[float] = []   # full history of per-sample metrics
        # Rolling buffer of raw predictions (for proxy mode)
        self._pred_buf: list[float] = []
        self._n_updates: int = 0
        self._ground_truth_seen: bool = False

    # ------------------------------------------------------------------
    # Online update — called after each AIF inference step
    # ------------------------------------------------------------------

    def update(self, y_pred: np.ndarray, y_true: Optional[np.ndarray] = None) -> bool:
        """
        Feed one prediction (and optionally a ground-truth label) to the CDD.

        Parameters
        ----------
        y_pred : np.ndarray
            Model prediction (scalar or array).
        y_true : np.ndarray or None
            Ground-truth label/value. Pass None to use proxy mode.

        Returns
        -------
        bool
            True if this single update immediately triggers the PH alarm.
            Use check() for a full assessment including the window test.
        """
        self._n_updates += 1
        y_pred_val = float(np.atleast_1d(y_pred).ravel()[0])
        self._pred_buf.append(y_pred_val)

        if y_true is not None:
            self._ground_truth_seen = True
            y_true_val = float(np.atleast_1d(y_true).ravel()[0])
            loss = self._loss(y_pred_val, y_true_val)
        else:
            # Proxy mode: track prediction entropy (classifiers) or
            # absolute deviation from rolling mean (regressors)
            loss = self._proxy_loss(y_pred_val)

        self._perf_buf.append(loss)

        ph_alarm = self._ph.update(loss)
        if ph_alarm and not self._ph_triggered:
            self._ph_triggered = True
            logger.warning(
                "CDD Page-Hinkley alarm at sample %d (statistic=%.2f).",
                self._n_updates, self._ph.statistic,
            )
        return ph_alarm

    def check(self) -> CDDResult:
        """
        Return the current drift assessment based on all buffered data.
        """
        n = len(self._perf_buf)
        ph_stat = self._ph.statistic
        ph_triggered = self._ph_triggered

        # Not enough data to compute the sliding-window perf_drop check,
        # but the online Page-Hinkley test is valid from the first update
        # and must not be silenced by the window-buffer guard — PH is the
        # *early-warning* arm of CDD by design.  We still clear the
        # window fields so downstream code knows they're unreliable.
        needed = self.reference_window + self.recent_window
        if n < needed:
            msg = (
                f"CDD: need {needed} samples, have {n}; "
                f"PH {'ALARM' if ph_triggered else 'nominal'} "
                f"(stat={ph_stat:.2f}, thresh={self._ph.lambda_})."
            )
            if ph_triggered:
                logger.warning(
                    "CONCEPT DRIFT detected via PH alarm — %s", msg
                )
            else:
                logger.debug(msg)
            return CDDResult(
                drift_detected=ph_triggered,
                ph_statistic=ph_stat, ph_threshold=self._ph.lambda_,
                ph_triggered=ph_triggered,
                perf_reference=0.0, perf_recent=0.0, perf_drop=0.0,
                window_triggered=False, n_updates=n,
                ground_truth_mode=self._ground_truth_seen, message=msg,
            )

        # Sliding window comparison on the performance metric buffer
        buf = np.array(self._perf_buf)
        ref_perf  = buf[-(self.reference_window + self.recent_window): -self.recent_window].mean()
        rec_perf  = buf[-self.recent_window:].mean()
        perf_drop = float(rec_perf - ref_perf)   # positive = getting worse

        window_triggered = perf_drop > self.perf_drop_threshold
        detected = ph_triggered or window_triggered

        if detected:
            msg = (
                f"CONCEPT DRIFT detected — "
                f"PH={ph_stat:.2f} (thresh={self._ph.lambda_}), "
                f"perf_drop={perf_drop:.4f} (thresh={self.perf_drop_threshold}), "
                f"ground_truth={'yes' if self._ground_truth_seen else 'proxy'}"
            )
            logger.warning(msg)
        else:
            msg = (
                f"No concept drift — PH={ph_stat:.2f}, "
                f"perf_drop={perf_drop:.4f}"
            )
            logger.debug(msg)

        return CDDResult(
            drift_detected=detected,
            ph_statistic=ph_stat, ph_threshold=self._ph.lambda_,
            ph_triggered=ph_triggered,
            perf_reference=float(ref_perf), perf_recent=float(rec_perf),
            perf_drop=perf_drop, window_triggered=window_triggered,
            n_updates=n, ground_truth_mode=self._ground_truth_seen,
            message=msg,
        )

    def reset_ph(self) -> None:
        """Reset the Page-Hinkley state (e.g. after a model update)."""
        self._ph.reset()
        self._ph_triggered = False

    # ------------------------------------------------------------------
    # Snapshot / restore — used by DetectorResetCoordinator on rollback
    # ------------------------------------------------------------------

    def snapshot_state(self) -> dict:
        """
        Capture every piece of CDD state that ``update()`` / ``check()``
        mutate, so that :meth:`restore_state` can reinstate the detector
        to this exact moment. Used by the rollback-reset path to undo
        any contamination that a poisoned MLIN injected into the PH
        accumulator and the performance window.
        """
        return {
            "ph": self._ph.snapshot_state(),
            "ph_triggered": bool(self._ph_triggered),
            "perf_buf": list(self._perf_buf),
            "pred_buf": list(self._pred_buf),
            "n_updates": int(self._n_updates),
            "ground_truth_seen": bool(self._ground_truth_seen),
        }

    def restore_state(self, state: dict) -> None:
        """Restore CDD internals from a ``snapshot_state()`` payload."""
        self._ph.restore_state(state.get("ph", {}))
        self._ph_triggered = bool(state.get("ph_triggered", False))
        # Defensive copies so a later mutation of the stored snapshot
        # cannot leak into a live detector (and vice-versa).
        self._perf_buf = list(state.get("perf_buf", []))
        self._pred_buf = list(state.get("pred_buf", []))
        self._n_updates = int(state.get("n_updates", 0))
        self._ground_truth_seen = bool(state.get("ground_truth_seen", False))

    def reset(self) -> None:
        """
        Full reset: clears PH state AND the rolling performance buffer.

        After a model update, not only is the PH baseline stale — the
        window-comparison check is also contaminated by pre-update loss
        entries still sitting in ``_perf_buf``.  ``reset_ph()`` alone
        leaves those entries in place, causing the window test to flip
        between "reference = old bad losses" and "recent = new good
        losses" (triggering a spurious "perf_drop" alarm in the wrong
        direction, or reporting a false recovery spike when the tail
        finally rolls out).  This method wipes *all* CDD state so the
        new model starts with a clean slate.
        """
        self._ph.reset()
        self._ph_triggered = False
        self._perf_buf.clear()
        self._pred_buf.clear()
        self._n_updates = 0
        self._ground_truth_seen = False

    def warmup(
        self,
        X: np.ndarray,
        y: np.ndarray,
        predict_fn,
    ) -> None:
        """
        Seed the detector with a reference (X, y) batch.

        This establishes the "long-run mean" loss rate BEFORE live
        streaming begins.  Without this, the very first live sample
        anchors PH's running mean at its own loss value — if the stream
        starts already drifted (loss=1 from step 1), PH never sees a
        deviation and the concept-drift signal never fires.  Warming
        from a clean reference batch lets PH establish "normal = e.g.
        2% error" and subsequently recognise "live = 100% error" as
        the catastrophic deviation it really is.

        Design note — what is and isn't seeded
        --------------------------------------
        * Page-Hinkley state IS seeded with the reference losses so the
          cumulative-sum detector has a calibrated baseline before the
          first live sample.
        * The proxy-mode prediction window (``_pred_buf``) IS seeded so
          the rolling-mean comparison used by ``_proxy_loss`` does not
          anchor on the very first live prediction (which would make
          proxy losses collapse to zero even when the prediction
          distribution has genuinely shifted).
        * The performance buffer (``_perf_buf``) used by the sliding-
          window comparison check is DELIBERATELY left untouched.
          Mixing real reference losses (0/1 hits from the clean corpus)
          with live proxy losses (abs-deviations from a rolling mean)
          is not arithmetically meaningful — the two metrics live on
          different scales.  The window test therefore waits for a full
          ``reference_window + recent_window`` of pure live observations,
          which keeps it honest at the cost of a later activation.  PH
          remains the early-warning arm of CDD during that ramp-up.

        Parameters
        ----------
        X : np.ndarray of shape (n, n_features)
            Clean reference inputs.
        y : np.ndarray of shape (n,)
            Ground-truth labels for those inputs.
        predict_fn : Callable[[np.ndarray], np.ndarray]
            A predict function (typically ``aif.predict`` or
            ``estimator.predict``) used to compute per-sample losses
            against the ground truth.  Accepts a 2-D array of shape
            (k, n_features) and returns predictions of shape (k,).
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] == 0 or y.shape[0] == 0:
            return

        # Try batch predict first — produces the whole prediction vector
        # in one sklearn call.  If the predict_fn is single-sample only
        # (as is the case for ``AIF.predict``, which wraps a per-step
        # inference path), fall back to a per-row loop.  ``AIF.predict``
        # with a 2-D input silently returns a single-row prediction
        # rather than raising, so we can't just try/except — we detect
        # the short-output case by comparing lengths and fall through
        # to the per-row path.
        preds: Optional[np.ndarray] = None
        try:
            p = np.asarray(predict_fn(X), dtype=float).ravel()
            if len(p) == X.shape[0]:
                preds = p
        except Exception:   # pragma: no cover - defensive
            preds = None

        if preds is None:
            preds = np.empty(X.shape[0], dtype=float)
            for i in range(X.shape[0]):
                try:
                    pi = predict_fn(X[i])
                except Exception:   # pragma: no cover - defensive
                    pi = predict_fn(X[i].reshape(1, -1))
                preds[i] = float(np.asarray(pi, dtype=float).ravel()[0])

        n = min(len(preds), len(y))
        for i in range(n):
            y_pred_val = float(np.asarray(preds[i]).ravel()[0])
            y_true_val = float(np.asarray(y[i]).ravel()[0])
            loss = self._loss(y_pred_val, y_true_val)
            # Seed PH only — feed the loss directly into the detector
            # without touching the window-test performance buffer.
            self._ph.update(loss)
            # Seed the proxy rolling-mean window so post-warmup proxy
            # losses don't spuriously spike on the first live sample.
            self._pred_buf.append(y_pred_val)

        # After warmup has established the pre-change baseline mean,
        # reset the PH cumulative-sum state.  Two reasons:
        #   1. ``_sum`` and ``_min_sum`` built up during warmup reflect
        #      noise around the known-good regime — they are not part of
        #      the live detection accumulator.  Carrying them into the
        #      live phase either biases PH in or out of the alarm
        #      region for reasons unrelated to live data.
        #   2. We freeze ``_x_mean`` at the warmup-derived value so PH
        #      measures live deviations against a FIXED pre-change
        #      baseline — the classical Page-Hinkley assumption.
        #      Without the freeze, the cumulative running mean absorbs
        #      live post-change samples and erodes the per-step signal,
        #      making detection require ~2× more observations than the
        #      paper's expected ``λ / (Δ − δ)`` heuristic predicts.
        self._ph._sum = 0.0
        self._ph._min_sum = 0.0
        self._ph.freeze_mean()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _loss(self, y_pred: float, y_true: float) -> float:
        """Per-sample loss: 0/1 error for classifiers, |e| for regressors."""
        if self.task == TaskType.CLASSIFIER:
            return float(y_pred != y_true)       # 1 = wrong, 0 = correct
        else:
            return abs(y_pred - y_true)           # absolute error

    def _proxy_loss(self, y_pred: float) -> float:
        """
        Proxy loss when no ground truth is available.
        Uses deviation of the prediction from a short rolling mean —
        a sudden shift in the output distribution signals concept change.
        """
        window = self._pred_buf[-50:] if len(self._pred_buf) >= 50 else self._pred_buf
        mean_pred = float(np.mean(window))
        return abs(y_pred - mean_pred)

    def __repr__(self) -> str:
        return (
            f"CDD(task={self.task.name}, "
            f"ref_window={self.reference_window}, "
            f"recent_window={self.recent_window}, "
            f"drop_thresh={self.perf_drop_threshold}, "
            f"n_updates={self._n_updates})"
        )
