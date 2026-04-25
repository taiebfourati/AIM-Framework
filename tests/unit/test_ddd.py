"""
Tier 1 property tests for detectors.ddd.DDD.

The DDD should:

* POSITIVE — trigger when the recent window's marginal distribution
  differs from the reference (mean shift, variance shift).
* NEGATIVE — stay silent on clean IID data drawn from the same
  distribution as the reference.  This is the critical case: the live
  dashboard has been false-positive-firing MToUT signals on clean
  baselines, which would be caught here if DDD is the culprit.
* EDGE — behave sanely with tiny buffers, single-feature inputs, and
  when `use_mmd=False` so only the KS test runs.
"""
from __future__ import annotations

import numpy as np
import pytest

from detectors.ddd import DDD, DDDResult
from ._helpers import fill_lib, iid_gaussian


# ---------------------------------------------------------------------------
# Positive cases — drift SHOULD be detected
# ---------------------------------------------------------------------------

def test_ddd_detects_mean_shift() -> None:
    """Mean shift of +3σ in every feature must trigger KS and MMD."""
    ref    = iid_gaussian(300, 5, mean=0.0, std=1.0, seed=0)
    recent = iid_gaussian(100, 5, mean=3.0, std=1.0, seed=1)
    lib = fill_lib(np.vstack([ref, recent]))

    ddd = DDD(reference_size=300, recent_size=100)
    result = ddd.check(lib)

    assert result.drift_detected, f"expected drift; result={result.message}"
    assert len(result.ks_drifted_features) >= 1
    assert result.mmd_statistic > 0.0


def test_ddd_detects_variance_shift() -> None:
    """5× variance blow-up without mean change still flips KS."""
    ref    = iid_gaussian(300, 5, mean=0.0, std=1.0, seed=2)
    recent = iid_gaussian(100, 5, mean=0.0, std=5.0, seed=3)
    lib = fill_lib(np.vstack([ref, recent]))

    ddd = DDD(reference_size=300, recent_size=100)
    result = ddd.check(lib)

    assert result.drift_detected
    assert len(result.ks_drifted_features) >= 1


def test_ddd_detects_single_feature_drift() -> None:
    """Even a drift on ONE of D features should trigger (min_drifted=1)."""
    ref = iid_gaussian(300, 5, seed=4)
    # shift ONLY feature 2 in recent window
    recent = iid_gaussian(100, 5, seed=5)
    recent[:, 2] += 4.0
    lib = fill_lib(np.vstack([ref, recent]))

    ddd = DDD(reference_size=300, recent_size=100)
    result = ddd.check(lib)

    assert result.drift_detected
    assert 2 in result.ks_drifted_features


# ---------------------------------------------------------------------------
# Negative cases — drift must NOT be flagged
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [10, 11, 12, 13, 14])
def test_ddd_quiet_on_clean_iid(seed: int) -> None:
    """
    Clean IID data drawn from the same distribution must stay silent.

    This is the test that mirrors the dashboard's complaint of
    MToUT firings on clean baselines.  Repeat across 5 seeds so
    a single unlucky seed does not hide a systematic false-positive.
    """
    rng = np.random.default_rng(seed)
    # Draw 400 samples from the SAME distribution, split ref / recent
    all_data = rng.normal(0.0, 1.0, size=(400, 5))
    lib = fill_lib(all_data)

    ddd = DDD(reference_size=300, recent_size=100)
    result = ddd.check(lib)

    assert not result.drift_detected, (
        f"false-positive: seed={seed} msg={result.message} "
        f"drifted={result.ks_drifted_features} mmd={result.mmd_statistic:.4f}"
    )


def test_ddd_quiet_with_mmd_disabled_on_clean_data() -> None:
    """Turning MMD off should not make KS suddenly trigger on clean data."""
    data = iid_gaussian(400, 3, seed=20)
    lib = fill_lib(data)

    ddd = DDD(reference_size=300, recent_size=100, use_mmd=False)
    result = ddd.check(lib)

    assert not result.drift_detected
    assert not result.mmd_triggered


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_ddd_not_ready_returns_false() -> None:
    """Not enough samples must return a no-op result, no exception."""
    lib = fill_lib(iid_gaussian(50, 3, seed=30))
    ddd = DDD(reference_size=300, recent_size=100)
    result = ddd.check(lib)

    assert isinstance(result, DDDResult)
    assert not result.drift_detected
    assert "need" in result.message.lower()


def test_ddd_refit_reference_replaces_baseline() -> None:
    """
    After refit_reference, the old reference must be gone.

    Note on semantics: ``DDD.refit_reference`` takes the OLDEST
    ``reference_size`` samples currently in LIB (``lib.get_values()[:N]``).
    So to exercise refit properly we need the bootstrap regime to have
    been fully evicted from the ring buffer before calling refit.
    """
    # Capacity small enough that bootstrap gets evicted by the new regime
    lib = fill_lib(iid_gaussian(400, 3, mean=0.0, seed=40), capacity=400)

    ddd = DDD(reference_size=300, recent_size=100)
    ddd.check(lib)                       # auto-fits reference on bootstrap

    # Push 400 new-regime samples — fully evicts the bootstrap
    for row in iid_gaussian(400, 3, mean=5.0, seed=41):
        lib.push(row)
    assert len(lib) == 400
    ddd.refit_reference(lib)             # now fits on new-regime data

    # Push another 100 clean samples from the SAME new regime
    for row in iid_gaussian(100, 3, mean=5.0, seed=42):
        lib.push(row)
    result = ddd.check(lib)

    # Reference was refit to mean=5.0 data, recent is also mean=5.0 → no drift
    assert not result.drift_detected, f"expected quiet after refit, got {result.message}"


def test_ddd_single_feature_dimension() -> None:
    """d=1 input vectors must still work end-to-end."""
    ref    = iid_gaussian(300, 1, mean=0.0, seed=50)
    recent = iid_gaussian(100, 1, mean=2.5, seed=51)
    lib = fill_lib(np.vstack([ref, recent]))

    ddd = DDD(reference_size=300, recent_size=100)
    result = ddd.check(lib)

    assert result.drift_detected
    assert result.ks_pvalues.shape == (1,)


# ---------------------------------------------------------------------------
# Arm isolation — KS-only vs MMD-only
# ---------------------------------------------------------------------------

def test_ddd_mmd_alone_triggers_on_joint_shift_with_ks_pinned_silent() -> None:
    """
    MMD is a peer test, not a confirmation of KS.

    KS inspects marginal distributions independently, so a
    joint-distribution shift that preserves each feature's marginal can
    walk right past it (e.g. a correlation-structure change — marginals
    stay N(0,1) while the pair's covariance flips from identity to high
    correlation).  With ``min_drifted_features`` pinned above the number
    of features available, the KS arm is off by construction, forcing
    MMD to carry the detection load alone.  This test pins that branch
    of the OR semantics.
    """
    rng = np.random.default_rng(600)
    # Reference: independent bivariate Gaussian
    ref = rng.multivariate_normal([0.0, 0.0], np.eye(2), size=300)
    # Recent: same marginal variances but heavy off-diagonal correlation
    rec_cov = np.array([[1.0, 0.95], [0.95, 1.0]])
    recent = rng.multivariate_normal([0.0, 0.0], rec_cov, size=100)
    lib = fill_lib(np.vstack([ref, recent]))

    # With only 2 features, min_drifted_features=5 makes KS unable to fire.
    ddd = DDD(
        reference_size=300, recent_size=100,
        min_drifted_features=5, use_mmd=True, mmd_threshold=0.01,
    )
    result = ddd.check(lib)

    ks_triggered = len(result.ks_drifted_features) >= ddd.min_drifted_features
    assert not ks_triggered, (
        f"KS must be pinned off with min_drifted_features=5 on 2D data; "
        f"drifted={result.ks_drifted_features}"
    )
    assert result.mmd_triggered, (
        f"MMD must fire on correlation-structure drift; "
        f"mmd_statistic={result.mmd_statistic:.4f}"
    )
    assert result.drift_detected, "MMD alone must mark drift_detected=True"


def test_ddd_ks_alone_triggers_when_mmd_disabled() -> None:
    """
    With ``use_mmd=False``, a clear marginal shift still fires the detector
    purely through the KS arm.  Pins the KS-only branch.
    """
    ref = iid_gaussian(300, 5, seed=602)
    recent = iid_gaussian(100, 5, mean=3.0, seed=603)
    lib = fill_lib(np.vstack([ref, recent]))

    ddd = DDD(reference_size=300, recent_size=100, use_mmd=False)
    result = ddd.check(lib)

    assert not result.mmd_triggered
    assert result.mmd_statistic == 0.0, "MMD stat must be 0.0 when MMD is disabled"
    assert len(result.ks_drifted_features) >= 1
    assert result.drift_detected


# ---------------------------------------------------------------------------
# min_drifted_features boundary
# ---------------------------------------------------------------------------

def test_ddd_min_drifted_features_2_suppresses_single_feature_drift() -> None:
    """
    The KS arm triggers only when at least ``min_drifted_features``
    features fail the per-feature test.  A single-feature drift with
    ``min_drifted_features=2`` and MMD disabled must stay silent — the
    signal is there, but the configured gate requires more evidence.

    To guarantee only ONE feature drifts (and not have spurious KS hits
    from independent-seed noise on features 1 and 2 with Bonferroni α),
    we draw ref and recent from the SAME rng stream — features 1 and 2
    are therefore identically distributed by construction.  Only feature
    0 is shifted.
    """
    rng = np.random.default_rng(610)
    full = rng.normal(0.0, 1.0, size=(400, 3))   # shared stream
    ref = full[:300]
    recent = full[300:].copy()                   # same dist as ref
    recent[:, 0] += 4.0                          # drift ONLY feature 0
    data = np.vstack([ref, recent])

    strict = DDD(
        reference_size=300, recent_size=100,
        min_drifted_features=2, use_mmd=False,
    )
    strict_result = strict.check(fill_lib(data))
    assert 0 in strict_result.ks_drifted_features, (
        "feature 0 should have failed KS on a 4σ shift"
    )
    assert len(strict_result.ks_drifted_features) < 2, (
        f"only feature 0 should drift; got {strict_result.ks_drifted_features}"
    )
    assert not strict_result.drift_detected, (
        "min_drifted_features=2 must suppress a single-feature drift"
    )

    # Flip the gate; same data now triggers.
    loose = DDD(
        reference_size=300, recent_size=100,
        min_drifted_features=1, use_mmd=False,
    )
    loose_result = loose.check(fill_lib(data))
    assert loose_result.drift_detected, (
        "min_drifted_features=1 must fire on the same single-feature drift"
    )


def test_ddd_min_drifted_features_boundary_is_inclusive() -> None:
    """
    ``>=`` in the gate: exactly ``min_drifted_features`` drifted features
    is sufficient to trigger.  Drift two of three features with
    ``min_drifted_features=2`` and MMD off → triggers.
    """
    ref = iid_gaussian(300, 3, seed=615)
    recent = iid_gaussian(100, 3, seed=616)
    recent[:, 0] += 4.0
    recent[:, 2] += 4.0  # drift two features out of three
    lib = fill_lib(np.vstack([ref, recent]))

    ddd = DDD(
        reference_size=300, recent_size=100,
        min_drifted_features=2, use_mmd=False,
    )
    result = ddd.check(lib)

    assert len(result.ks_drifted_features) >= 2
    assert result.drift_detected, "exactly-threshold hit must trigger"


# ---------------------------------------------------------------------------
# refit_reference — guardrails for undersized LIB
# ---------------------------------------------------------------------------

def test_ddd_refit_reference_below_recent_size_is_noop() -> None:
    """
    When LIB holds fewer than ``recent_size`` samples, ``refit_reference``
    must leave the existing reference untouched and log a warning.

    Refitting on a tiny, statistically-meaningless sample would poison
    every subsequent drift check; the guard prevents that failure mode.
    """
    ddd = DDD(reference_size=300, recent_size=100)
    baseline = iid_gaussian(300, 4, mean=0.0, seed=620)
    ddd.fit_reference(baseline)
    saved_ref = ddd._reference.copy()

    # 50 samples < recent_size (100) → refit must no-op
    tiny_lib = fill_lib(iid_gaussian(50, 4, mean=10.0, seed=621), capacity=50)
    assert len(tiny_lib) == 50

    returned = ddd.refit_reference(tiny_lib)

    assert returned is ddd, "refit_reference must return self for chaining"
    np.testing.assert_array_equal(
        ddd._reference, saved_ref,
        err_msg="reference must be untouched when LIB < recent_size",
    )


def test_ddd_refit_reference_between_recent_and_reference_size_uses_available() -> None:
    """
    LIB sized between ``recent_size`` and ``reference_size`` → refit uses
    ``min(available, reference_size)`` samples rather than failing.

    This keeps the detector functional in partially-warmed pipelines
    (e.g. right after a model update when the buffer has refilled past
    ``recent_size`` but not yet reached the full reference target).
    """
    ddd = DDD(reference_size=300, recent_size=100)
    ddd.fit_reference(iid_gaussian(300, 4, mean=0.0, seed=625))

    # LIB holds 150 samples — between recent_size (100) and reference_size (300)
    mid_lib = fill_lib(iid_gaussian(150, 4, mean=5.0, seed=626), capacity=150)
    assert len(mid_lib) == 150

    ddd.refit_reference(mid_lib)

    assert ddd._reference.shape == (150, 4), (
        f"expected 150-row partial reference, got {ddd._reference.shape}"
    )
    assert abs(float(ddd._reference.mean()) - 5.0) < 0.5, (
        "new reference should reflect the mean=5.0 regime"
    )


def test_ddd_refit_reference_at_recent_size_boundary_succeeds() -> None:
    """
    LIB holding exactly ``recent_size`` samples → refit succeeds using
    them all (inclusive boundary of the ``>= recent_size`` gate).
    """
    ddd = DDD(reference_size=300, recent_size=100)
    ddd.fit_reference(iid_gaussian(300, 4, mean=0.0, seed=630))

    exact_lib = fill_lib(iid_gaussian(100, 4, mean=5.0, seed=631), capacity=100)
    assert len(exact_lib) == 100

    ddd.refit_reference(exact_lib)

    assert ddd._reference.shape == (100, 4)
    assert abs(float(ddd._reference.mean()) - 5.0) < 0.5
