"""
Tier 1 property tests for detectors.cdd.CDD.

The CDD should:

* POSITIVE — fire when per-sample loss rises after a regime change
  (classifier: accuracy collapses; regressor: MAE explodes).
* NEGATIVE — stay silent while performance is stable.
* EDGE — return a no-op CDDResult when there are too few updates,
  and behave correctly in both classifier and regressor task modes.

CDD differs from DDD/DPD/CPD in that it consumes (y_pred, y_true)
streams via update() rather than buffers — so the tests drive it
sample-by-sample.
"""
from __future__ import annotations

import numpy as np
import pytest

from detectors.cdd import CDD, CDDResult


# ---------------------------------------------------------------------------
# Positive — classifier
# ---------------------------------------------------------------------------

def test_cdd_classifier_detects_accuracy_collapse() -> None:
    """Steady 95% accuracy, then 30% accuracy — must trigger."""
    rng = np.random.default_rng(0)
    cdd = CDD(task="classifier", reference_window=200, recent_window=50,
              perf_drop_threshold=0.10, ph_lambda=50.0)

    # Phase 1: 250 samples at ~95% accuracy
    for _ in range(250):
        y_true = int(rng.integers(0, 2))
        y_pred = y_true if rng.random() < 0.95 else 1 - y_true
        cdd.update(np.array([y_pred]), np.array([y_true]))

    # Phase 2: 50 more samples at ~30% accuracy (post-drift)
    for _ in range(50):
        y_true = int(rng.integers(0, 2))
        y_pred = y_true if rng.random() < 0.30 else 1 - y_true
        cdd.update(np.array([y_pred]), np.array([y_true]))

    result = cdd.check()
    assert result.drift_detected, f"expected drift, got {result.message}"
    # At least one of the two mechanisms must have fired
    assert result.window_triggered or result.ph_triggered


# ---------------------------------------------------------------------------
# Positive — regressor
# ---------------------------------------------------------------------------

def test_cdd_regressor_detects_mae_explosion() -> None:
    """MAE ≈ 0.1 for 200 samples, then MAE ≈ 5.0 — must trigger."""
    rng = np.random.default_rng(1)
    cdd = CDD(task="regressor", reference_window=200, recent_window=50,
              perf_drop_threshold=0.50, ph_lambda=50.0)

    # Phase 1: 250 tight residuals
    for _ in range(250):
        y_true = rng.normal(0.0, 1.0)
        y_pred = y_true + rng.normal(0.0, 0.1)
        cdd.update(np.array([y_pred]), np.array([y_true]))

    # Phase 2: 50 wide residuals
    for _ in range(50):
        y_true = rng.normal(0.0, 1.0)
        y_pred = y_true + rng.normal(0.0, 5.0)
        cdd.update(np.array([y_pred]), np.array([y_true]))

    result = cdd.check()
    assert result.drift_detected
    assert result.perf_drop > 0.0


# ---------------------------------------------------------------------------
# Negative — classifier
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [10, 11, 12, 13, 14])
def test_cdd_classifier_quiet_on_stable_accuracy(seed: int) -> None:
    """Consistent ~95% accuracy for 500 samples — no drift."""
    rng = np.random.default_rng(seed)
    cdd = CDD(task="classifier", reference_window=200, recent_window=50,
              perf_drop_threshold=0.10, ph_lambda=50.0)

    for _ in range(500):
        y_true = int(rng.integers(0, 2))
        y_pred = y_true if rng.random() < 0.95 else 1 - y_true
        cdd.update(np.array([y_pred]), np.array([y_true]))

    result = cdd.check()
    assert not result.drift_detected, (
        f"false-positive: seed={seed} msg={result.message} "
        f"ph_stat={result.ph_statistic:.2f} perf_drop={result.perf_drop:.4f}"
    )


# ---------------------------------------------------------------------------
# Negative — regressor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [20, 21, 22, 23, 24])
def test_cdd_regressor_quiet_on_stable_mae(seed: int) -> None:
    """MAE ≈ 0.1 throughout — no drift."""
    rng = np.random.default_rng(seed)
    cdd = CDD(task="regressor", reference_window=200, recent_window=50,
              perf_drop_threshold=0.50, ph_lambda=50.0)

    for _ in range(500):
        y_true = rng.normal(0.0, 1.0)
        y_pred = y_true + rng.normal(0.0, 0.1)
        cdd.update(np.array([y_pred]), np.array([y_true]))

    result = cdd.check()
    assert not result.drift_detected, (
        f"false-positive: seed={seed} perf_drop={result.perf_drop:.4f}"
    )


# ---------------------------------------------------------------------------
# Proxy mode — no ground truth
# ---------------------------------------------------------------------------

def test_cdd_proxy_mode_window_silent_on_stable_predictions() -> None:
    """
    Without ground truth + stable predictions: the sliding-WINDOW check
    must stay silent.

    Known limitation (documented here so the dashboard team can make an
    informed choice): Page-Hinkley in proxy mode is inherently drift-prone
    because the proxy loss is always non-negative and PH's running mean
    (with default alpha=1.0) locks to the first observation — so the PH
    cumulative sum grows monotonically on any non-degenerate stream and
    will eventually exceed lambda. The *window* check does not share this
    flaw, which is what we assert here. The dashboard uses ground-truth
    mode, where PH behaves correctly.
    """
    rng = np.random.default_rng(30)
    cdd = CDD(task="regressor", reference_window=200, recent_window=50,
              perf_drop_threshold=0.50, ph_lambda=50.0)

    for _ in range(500):
        cdd.update(np.array([rng.normal(0.0, 0.5)]), y_true=None)

    result = cdd.check()
    assert not result.window_triggered, (
        f"proxy window false-positive: perf_drop={result.perf_drop:.4f}"
    )
    assert result.ground_truth_mode is False


def test_cdd_proxy_mode_catches_prediction_shift() -> None:
    """Without ground truth: predictions jumping regime should trigger PH or window."""
    rng = np.random.default_rng(31)
    cdd = CDD(task="regressor", reference_window=200, recent_window=50,
              perf_drop_threshold=0.30, ph_lambda=20.0)

    # Predictions around 0 for 250 samples
    for _ in range(250):
        cdd.update(np.array([rng.normal(0.0, 0.1)]), y_true=None)
    # Then jump to predictions around 10 for 50 samples
    for _ in range(50):
        cdd.update(np.array([rng.normal(10.0, 0.1)]), y_true=None)

    result = cdd.check()
    assert result.drift_detected


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_cdd_not_ready_returns_false() -> None:
    """Too few updates must yield a non-triggering result, no error."""
    cdd = CDD(task="classifier", reference_window=200, recent_window=50)
    for _ in range(20):
        cdd.update(np.array([0]), np.array([0]))

    result = cdd.check()
    assert isinstance(result, CDDResult)
    assert not result.drift_detected
    assert "need" in result.message.lower()


def test_cdd_reset_ph_clears_alarm() -> None:
    """reset_ph() must clear an active PH alarm so the detector can re-arm."""
    cdd = CDD(task="regressor", reference_window=200, recent_window=50,
              ph_lambda=1.0)  # very sensitive — will fire easily
    rng = np.random.default_rng(40)
    for _ in range(300):
        y_true = rng.normal()
        y_pred = y_true + rng.normal(0.0, 3.0)
        cdd.update(np.array([y_pred]), np.array([y_true]))

    assert cdd.check().ph_triggered      # sanity: we fired
    cdd.reset_ph()
    # Immediately after reset, the PH statistic should be 0 and ph_triggered False
    result = cdd.check()
    assert not result.ph_triggered
    assert result.ph_statistic == 0.0


def test_cdd_rejects_unknown_task() -> None:
    with pytest.raises(ValueError):
        CDD(task="not_a_task")


# ---------------------------------------------------------------------------
# Arm isolation — PH-alone vs window-alone
# ---------------------------------------------------------------------------

def test_cdd_ph_alone_triggers_when_window_reverts_to_stable() -> None:
    """
    PH is a latch: once it fires it stays triggered until ``reset_ph()``.

    If drift occurs early in the stream and the regime later reverts to
    the pre-drift level, the sliding-window check sees matching reference
    and recent means (no ``perf_drop``) while PH still reports the
    alarm it latched during the drift burst.  The resulting
    ``drift_detected`` must come from the PH arm alone — this pins the
    OR semantics of ``check()``: ``detected = ph_triggered OR
    window_triggered``.
    """
    rng = np.random.default_rng(500)
    cdd = CDD(task="classifier", reference_window=100, recent_window=30,
              perf_drop_threshold=0.10, ph_lambda=5.0)  # sensitive PH

    # Phase 1: 100 samples at ~95% accuracy (establish PH baseline)
    for _ in range(100):
        y_true = int(rng.integers(0, 2))
        y_pred = y_true if rng.random() < 0.95 else 1 - y_true
        cdd.update(np.array([y_pred]), np.array([y_true]))

    # Phase 2: 40 samples at ~20% accuracy (drift burst — fires PH)
    for _ in range(40):
        y_true = int(rng.integers(0, 2))
        y_pred = y_true if rng.random() < 0.20 else 1 - y_true
        cdd.update(np.array([y_pred]), np.array([y_true]))

    # Verify PH fired during the burst
    assert cdd._ph_triggered, "drift burst should have fired PH"

    # Phase 3: 200 more samples at ~95% accuracy (revert to clean)
    # Reference window [-130, -30] and recent window [-30, 0] both sit
    # inside this clean regime, so ``perf_drop`` collapses to ~0.
    for _ in range(200):
        y_true = int(rng.integers(0, 2))
        y_pred = y_true if rng.random() < 0.95 else 1 - y_true
        cdd.update(np.array([y_pred]), np.array([y_true]))

    result = cdd.check()
    assert result.ph_triggered, "PH latch must survive the clean tail"
    assert not result.window_triggered, (
        f"window should be quiet in stable tail; got perf_drop={result.perf_drop:.4f}"
    )
    assert result.drift_detected, "PH-alone must still mark drift_detected=True"


def test_cdd_window_alone_triggers_when_ph_cannot_fire() -> None:
    """
    The sliding-window check is a peer detector, not a confirmation arm.

    Even with PH pinned to an effectively infinite ``lambda`` (so PH can
    never raise an alarm regardless of cumulative signal), a sufficiently
    large and sustained drop in performance in the recent window must
    still trigger ``drift_detected`` via the window path alone.  This
    pins the other side of the OR semantics in ``check()``.
    """
    rng = np.random.default_rng(510)
    cdd = CDD(task="classifier", reference_window=100, recent_window=30,
              perf_drop_threshold=0.20, ph_lambda=1.0e6)  # PH cannot fire

    # Phase 1: 200 samples at ~95% accuracy (fills reference window)
    for _ in range(200):
        y_true = int(rng.integers(0, 2))
        y_pred = y_true if rng.random() < 0.95 else 1 - y_true
        cdd.update(np.array([y_pred]), np.array([y_true]))

    # Phase 2: 30 samples at ~30% accuracy (fills recent window)
    for _ in range(30):
        y_true = int(rng.integers(0, 2))
        y_pred = y_true if rng.random() < 0.30 else 1 - y_true
        cdd.update(np.array([y_pred]), np.array([y_true]))

    result = cdd.check()
    assert not result.ph_triggered, "PH was pinned; it must not have fired"
    assert result.window_triggered, (
        f"window should trigger on a 65pp accuracy drop; "
        f"perf_drop={result.perf_drop:.4f}"
    )
    assert result.drift_detected, "window-alone must mark drift_detected=True"
    assert result.perf_drop > cdd.perf_drop_threshold


# ---------------------------------------------------------------------------
# warmup() — freezes the PH mean and clears the cumsum state
# ---------------------------------------------------------------------------

def test_cdd_warmup_freezes_mean_and_clears_ph_sum() -> None:
    """
    After ``warmup()`` seeds PH with a clean reference batch, the running
    mean is frozen and the cumulative sum is reset to zero.

    This is the classical Page-Hinkley precondition: measure deviations
    against a FIXED pre-change baseline.  Without the freeze, the
    running-mean update during live observations absorbs post-change
    samples and dulls the detector (the docstring on
    ``PageHinkley.freeze_mean`` explains why).  Without the sum reset,
    cumulative-sum noise built up while observing the clean reference
    biases the live detector in or out of the alarm region for reasons
    unrelated to live data.
    """
    cdd = CDD(task="classifier", reference_window=200, recent_window=50,
              ph_lambda=5.0)

    # Build a deterministic clean-reference corpus: 100 binary labels +
    # a predict_fn that matches 95 / 100 of them (5% error baseline).
    rng = np.random.default_rng(520)
    X_ref = rng.normal(size=(100, 4))
    y_ref = rng.integers(0, 2, size=100).astype(float)
    # Deterministically flip every 20th prediction so the error rate is
    # exactly 5% regardless of the rng state the test is run under.
    flip_idx = np.arange(0, 100, 20)

    def predict_fn(X: np.ndarray) -> np.ndarray:
        n = np.atleast_2d(X).shape[0]
        out = y_ref[:n].copy()
        out[flip_idx[flip_idx < n]] = 1.0 - out[flip_idx[flip_idx < n]]
        return out

    cdd.warmup(X_ref, y_ref, predict_fn)

    # After warmup: mean frozen, cumsum cleared
    assert cdd._ph._mean_frozen, "warmup must call freeze_mean()"
    assert cdd._ph._sum == 0.0, "warmup must zero _sum"
    assert cdd._ph._min_sum == 0.0, "warmup must zero _min_sum"

    # The frozen mean reflects the reference-batch loss rate (~5%).
    # PH's _n counts the reference updates but _x_mean holds the
    # cumulative average loss over those updates.
    assert cdd._ph._n == 100, "PH must have consumed all reference losses"
    assert 0.01 < cdd._ph._x_mean < 0.15, (
        f"frozen mean should be near 5% error; got {cdd._ph._x_mean}"
    )

    # Perf buffer is deliberately NOT seeded — the window arm waits for
    # pure live observations before it is valid.  Pred buffer IS seeded
    # so proxy-mode losses have a rolling-mean anchor from step 1.
    assert cdd._perf_buf == [], "warmup must NOT seed _perf_buf"
    assert len(cdd._pred_buf) == 100, "warmup must seed _pred_buf"


def test_cdd_warmup_then_drift_fires_ph_faster_than_cold_start() -> None:
    """
    Warmed detector catches drift earlier than a cold-start detector.

    The whole point of ``warmup`` is to establish a calibrated baseline
    before the live phase begins.  If live samples start already drifted,
    a cold-start detector anchors its PH mean at the first drifted
    observation and never sees a deviation — warmup prevents that failure
    mode.  This test asserts the operational consequence: given identical
    drifted streams, the warmed detector fires PH after fewer samples.
    """
    # Both detectors see the same adversarial stream: 100% error from
    # step 1.  Cold-start anchors on that 1.0 loss and never triggers.
    n_drift = 200

    # Cold-start
    cold = CDD(task="classifier", reference_window=200, recent_window=50,
               ph_lambda=5.0)
    for _ in range(n_drift):
        cold.update(np.array([0]), np.array([1]))  # always wrong
    assert not cold._ph_triggered, (
        "cold-start PH should NOT fire — baseline locks at 1.0 loss"
    )

    # Warmed
    rng = np.random.default_rng(525)
    X_ref = rng.normal(size=(100, 4))
    y_ref = rng.integers(0, 2, size=100).astype(float)

    def predict_fn(X: np.ndarray) -> np.ndarray:
        n = np.atleast_2d(X).shape[0]
        # 0% error — perfect predictions during warmup
        return y_ref[:n].copy()

    warm = CDD(task="classifier", reference_window=200, recent_window=50,
               ph_lambda=5.0)
    warm.warmup(X_ref, y_ref, predict_fn)
    assert warm._ph._mean_frozen
    for _ in range(n_drift):
        warm.update(np.array([0]), np.array([1]))  # always wrong
    assert warm._ph_triggered, (
        "warmed detector must fire when live stream deviates from the "
        "frozen baseline"
    )


# ---------------------------------------------------------------------------
# reset() vs reset_ph() — full wipe vs PH-only
# ---------------------------------------------------------------------------

def test_cdd_reset_ph_preserves_perf_buf_but_clears_ph() -> None:
    """
    ``reset_ph()`` clears the Page-Hinkley state only.  The performance
    buffer, prediction buffer, and update counter survive.

    Callers use this when PH has fired but the caller has opted to treat
    it as a controlled event (e.g. a validation flag) without discarding
    the history the sliding-window check depends on.
    """
    cdd = CDD(task="regressor", reference_window=10, recent_window=5,
              ph_lambda=1.0)  # very sensitive
    rng = np.random.default_rng(530)
    for _ in range(50):
        y_true = rng.normal()
        y_pred = y_true + rng.normal(0.0, 3.0)  # large residuals
        cdd.update(np.array([y_pred]), np.array([y_true]))

    assert cdd._ph_triggered
    assert cdd._ph.statistic > 0.0
    perf_snapshot = list(cdd._perf_buf)
    pred_snapshot = list(cdd._pred_buf)
    assert len(perf_snapshot) == 50

    cdd.reset_ph()

    # PH cleared
    assert not cdd._ph_triggered
    assert cdd._ph.statistic == 0.0
    assert cdd._ph._sum == 0.0
    assert cdd._ph._min_sum == 0.0
    assert cdd._ph._n == 0
    # Buffers intact
    assert cdd._perf_buf == perf_snapshot
    assert cdd._pred_buf == pred_snapshot
    assert cdd._n_updates == 50
    assert cdd._ground_truth_seen is True


def test_cdd_reset_wipes_all_state_including_perf_buf() -> None:
    """
    ``reset()`` is a full wipe for post-retrain recovery.

    After a retrain the model's loss distribution changes — the window
    comparison would otherwise swing between pre-retrain reference losses
    and post-retrain recent losses, producing a spurious perf_drop in
    either direction.  Wiping the perf buffer forces the window arm to
    wait for a full ``reference_window + recent_window`` of pure
    post-retrain observations before it is valid again.
    """
    cdd = CDD(task="regressor", reference_window=10, recent_window=5,
              ph_lambda=1.0)
    rng = np.random.default_rng(540)
    for _ in range(50):
        y_true = rng.normal()
        y_pred = y_true + rng.normal(0.0, 3.0)
        cdd.update(np.array([y_pred]), np.array([y_true]))

    assert cdd._ph_triggered
    assert len(cdd._perf_buf) == 50

    cdd.reset()

    # Everything cleared
    assert not cdd._ph_triggered
    assert cdd._ph.statistic == 0.0
    assert cdd._ph._n == 0
    assert cdd._perf_buf == []
    assert cdd._pred_buf == []
    assert cdd._n_updates == 0
    assert cdd._ground_truth_seen is False

    # A post-reset ``check()`` must fall through the "need more data" branch.
    result = cdd.check()
    assert not result.drift_detected
    assert "need" in result.message.lower()
