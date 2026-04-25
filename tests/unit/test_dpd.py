"""
Tier 1 property tests for detectors.dpd.DPD.

The DPD should:

* POSITIVE — flag when the Isolation Forest arm fires (anomaly rate
  past ``contamination_threshold`` while remaining sparse), OR when
  the two-tier Mahalanobis rule fires:
    - **soft**: ≥ ``min_mahal_outliers`` samples past ``mahal_threshold``
      (default 4σ × 3 hits) while the hit fraction stays below 50 % of
      the window — one or two stray 4σ samples are in the normal tail
      and must not fire;
    - **hard**: ≥ 1 sample past ``mahal_hard_threshold`` (default 8σ)
      while the hit fraction stays below ``mahal_hard_max_fraction``
      (default 30 %);
    - **extreme escape hatch**: any single sample past
      ``2 × mahal_hard_threshold`` (default 16σ) fires unconditionally,
      regardless of sparsity.
  Each arm's sparsity gate hands uniform distribution shifts to
  DDD/CDD instead of mis-labelling them DATA_POISONING.
* NEGATIVE — stay silent on clean IID data that matches the reference.
  The live dashboard was reported firing DATA_POISONING on a clean
  baseline at steps s1950 / s2850 / s3450, so the negative cases below
  directly target that behaviour.
* EDGE — correctly report "not ready" when the LIB is too small, and
  work on single-feature inputs.

Notes on stochasticity
----------------------
DPD uses IsolationForest with random_state=42 pinned in DPD.fit_reference,
so positive/negative outcomes are reproducible.  The negative cases
still parametrise over several data seeds so that an unlucky IID sample
cannot mask a systematic false-positive.
"""
from __future__ import annotations

import numpy as np
import pytest

from detectors.dpd import DPD, DPDResult
from ._helpers import fill_lib, iid_gaussian


# ---------------------------------------------------------------------------
# Positive cases — poisoning SHOULD be detected
# ---------------------------------------------------------------------------

def test_dpd_detects_outlier_injection() -> None:
    """20% of recent samples at +15σ should trigger both IF and Mahalanobis."""
    rng    = np.random.default_rng(0)
    ref    = rng.normal(0.0, 1.0, size=(300, 4))
    recent = rng.normal(0.0, 1.0, size=(50, 4))
    # Replace 10/50 = 20% with extreme outliers
    recent[:10] = 15.0

    lib = fill_lib(np.vstack([ref, recent]))
    dpd = DPD(reference_size=300, recent_size=50,
              if_contamination=0.02, contamination_threshold=0.10,
              mahal_threshold=4.0)
    result = dpd.check(lib)

    assert result.poisoning_detected
    assert result.mahal_triggered
    assert result.if_triggered or result.mahal_triggered


def test_dpd_mahalanobis_catches_single_extreme_outlier() -> None:
    """A single 20σ sample must trip the Mahalanobis check even if IF misses."""
    rng = np.random.default_rng(1)
    ref    = rng.normal(0.0, 1.0, size=(300, 4))
    recent = rng.normal(0.0, 1.0, size=(50, 4))
    recent[25] = 20.0        # one absurd sample

    lib = fill_lib(np.vstack([ref, recent]))
    dpd = DPD(reference_size=300, recent_size=50, mahal_threshold=4.0)
    result = dpd.check(lib)

    assert result.poisoning_detected
    assert result.mahal_triggered
    assert 25 in result.mahal_anomalous_indices


# ---------------------------------------------------------------------------
# Negative cases — clean data must NOT fire
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [100, 101, 102, 103, 104])
def test_dpd_quiet_on_clean_iid(seed: int) -> None:
    """
    Clean IID gaussians drawn from the same distribution must not trigger.

    Targets the user-reported false positive at s1950 / s2850 / s3450.
    Parametrised over 5 seeds — a single unlucky draw should not mask
    a systematic bug.
    """
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 1.0, size=(500, 4))
    lib = fill_lib(data)

    dpd = DPD(reference_size=300, recent_size=50,
              if_contamination=0.02, contamination_threshold=0.10,
              mahal_threshold=4.0)
    result = dpd.check(lib)

    assert not result.poisoning_detected, (
        f"false-positive: seed={seed}  if_rate={result.if_anomaly_rate:.3f} "
        f"mahal_max={result.mahal_max:.2f}  msg={result.message}"
    )


def test_dpd_quiet_under_repeated_checks() -> None:
    """
    Simulate the dashboard loop: push one clean sample at a time and
    call check() repeatedly.  No call should ever trigger a false alert.

    This mimics the RTP tick pattern where DPD.check() runs at every
    check_interval and is the closest unit test to the user's scenario.
    """
    rng = np.random.default_rng(200)
    # Bootstrap reference with 400 clean samples
    bootstrap = rng.normal(0.0, 1.0, size=(400, 4))
    lib = fill_lib(bootstrap, capacity=2000)

    dpd = DPD(reference_size=300, recent_size=50,
              if_contamination=0.02, contamination_threshold=0.10,
              mahal_threshold=4.0)
    # Warm up so fit_reference happens
    dpd.check(lib)

    # Push 500 more clean samples one at a time, checking every 50 steps
    false_positives: list[int] = []
    for step in range(500):
        lib.push(rng.normal(0.0, 1.0, size=(4,)))
        if step % 50 == 0:
            result = dpd.check(lib)
            if result.poisoning_detected:
                false_positives.append(step)

    assert not false_positives, (
        f"DPD false-positive firings at steps {false_positives} "
        f"on clean IID data — matches the user-reported bug pattern."
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_dpd_not_ready_returns_false() -> None:
    """Tiny LIB must yield a no-op, non-triggering result."""
    lib = fill_lib(iid_gaussian(30, 4, seed=300))
    dpd = DPD(reference_size=300, recent_size=50)
    result = dpd.check(lib)

    assert isinstance(result, DPDResult)
    assert not result.poisoning_detected
    assert "need" in result.message.lower()


def test_dpd_single_feature() -> None:
    """d=1 inputs must still work."""
    rng = np.random.default_rng(400)
    ref    = rng.normal(0.0, 1.0, size=(300, 1))
    recent = rng.normal(0.0, 1.0, size=(50, 1))
    recent[25] = 25.0

    lib = fill_lib(np.vstack([ref, recent]))
    dpd = DPD(reference_size=300, recent_size=50, mahal_threshold=4.0)
    result = dpd.check(lib)

    assert result.poisoning_detected
    assert result.mahal_triggered


# ---------------------------------------------------------------------------
# Two-tier Mahalanobis rule — soft / hard / extreme in isolation
# ---------------------------------------------------------------------------
#
# The three arms (soft threshold, hard threshold, extreme-hit escape)
# overlap functionally but are gated by different rules.  Isolating
# each on its own lets a refactor rename a knob or swap a check without
# silently collapsing one arm into another.  The injection helper puts
# a controlled σ-valued spike into a single feature so that the
# Mahalanobis distance from the origin is ≈ σ (reference ≈ N(0, I))
# — we can then tune which tier fires purely via the σ choice.
# ---------------------------------------------------------------------------

def _inject_sigma(
    recent: np.ndarray, indices: list[int], sigma: float, *, feature: int = 0
) -> np.ndarray:
    """
    Overwrite ``recent[i]`` with ``(σ, 0, 0, …)`` for each i in ``indices``.

    Because the reference is clean N(0, I), the fitted covariance is
    ≈ I and Mahalanobis((σ, 0, …, 0)) ≈ σ.  Sampling noise on a 300-row
    reference puts the numerical distance inside σ × (1 ± ~0.08), which
    the tier-margin tests below comfortably absorb.
    """
    out = recent.copy()
    for i in indices:
        out[i] = 0.0
        out[i, feature] = sigma
    return out


def _clean_ref(seed: int, n: int = 300, d: int = 4) -> np.ndarray:
    """Deterministic clean IID reference for the arm-isolation tests."""
    return np.random.default_rng(seed).normal(0.0, 1.0, size=(n, d))


def test_dpd_soft_rule_fires_at_min_hits() -> None:
    """
    Exactly ``min_mahal_outliers`` hits at 5σ (past 4σ soft, below 8σ hard)
    must trip the soft arm while leaving the hard arm quiet.

    This is the load-bearing guard in the paper's two-tier rule: one or
    two stray 4σ samples sit inside the normal tail (~0.5 % of gaussian
    mass) and must not fire; three coordinated hits are the earliest
    sign of a structured poisoning run.
    """
    ref = _clean_ref(seed=500)
    recent = np.random.default_rng(501).normal(0.0, 1.0, size=(50, 4))
    recent = _inject_sigma(recent, [0, 10, 20], sigma=5.0)

    lib = fill_lib(np.vstack([ref, recent]))
    dpd = DPD(
        reference_size=300, recent_size=50,
        mahal_threshold=4.0, min_mahal_outliers=3,
        mahal_hard_threshold=8.0,
    )
    result = dpd.check(lib)

    assert result.poisoning_detected
    assert result.mahal_soft_triggered
    assert not result.mahal_hard_triggered, (
        "soft arm should fire alone — no hit above 8σ "
        f"(mahal_max={result.mahal_max:.2f})"
    )
    assert result.mahal_max < 8.0


def test_dpd_soft_rule_silent_below_min_hits() -> None:
    """
    ``min_mahal_outliers - 1`` hits at 5σ must stay silent on the
    Mahalanobis arm.  A 4σ sample on clean gaussian data occurs
    roughly once per 15 000 samples; two injected hits in a 50-sample
    window is well inside the paper's "normal tail" envelope and must
    not trigger an alert.

    The clean background uses std=0.5 so no clean row accidentally
    produces a natural 4σ Mahalanobis hit (under N(0, 1) in 4-D a
    50-row window hits ≥1 natural >4σ sample ~14 % of the time, which
    is enough flakiness to mask a real regression in the soft-arm
    count logic).  Small-std clean rows still look like inliers to
    the Isolation Forest — a tight cluster near the origin is
    surrounded by reference samples, so IF cannot hallucinate an
    "anomaly" from them either.
    """
    ref = _clean_ref(seed=502)
    recent = np.random.default_rng(503).normal(0.0, 0.5, size=(50, 4))
    # 2 hits — below the default min_mahal_outliers=3.
    recent = _inject_sigma(recent, [0, 10], sigma=5.0)

    lib = fill_lib(np.vstack([ref, recent]))
    dpd = DPD(
        reference_size=300, recent_size=50,
        mahal_threshold=4.0, min_mahal_outliers=3,
        mahal_hard_threshold=8.0,
    )
    result = dpd.check(lib)

    assert not result.mahal_soft_triggered, (
        f"only 2 injected hits at 5σ — below min_mahal_outliers=3; "
        f"soft_hits={len(result.mahal_anomalous_indices)}, "
        f"mahal_max={result.mahal_max:.2f}"
    )
    assert not result.mahal_hard_triggered


def test_dpd_hard_rule_fires_on_single_10sigma() -> None:
    """
    A lone 10σ sample — past the 8σ hard threshold but below the 16σ
    extreme escape — must trip the hard arm on its own.

    Under N(0, I) the probability of a single 8σ hit is ~2·10⁻¹⁵ per
    sample, so the hard rule's single-hit trigger is virtually never
    wrong on clean data.  The soft arm stays quiet because only one
    sample was injected (below ``min_mahal_outliers=3``).
    """
    ref = _clean_ref(seed=504)
    recent = np.random.default_rng(505).normal(0.0, 1.0, size=(50, 4))
    recent = _inject_sigma(recent, [7], sigma=10.0)

    lib = fill_lib(np.vstack([ref, recent]))
    dpd = DPD(
        reference_size=300, recent_size=50,
        mahal_threshold=4.0, min_mahal_outliers=3,
        mahal_hard_threshold=8.0,
    )
    result = dpd.check(lib)

    assert result.poisoning_detected
    assert result.mahal_hard_triggered
    assert 7 in result.mahal_hard_indices
    # Only one hit (< 3) — soft arm should not fire.
    assert not result.mahal_soft_triggered
    # Below 16σ — extreme escape must not have been the trigger.
    assert result.mahal_max < 16.0


def test_dpd_hard_rule_suppressed_when_widespread() -> None:
    """
    Hard-arm sparsity gate: if ≥ ``mahal_hard_max_fraction`` of the
    window is past 8σ the arm must be suppressed.  That pattern is
    the signature of a uniform distribution shift (drift), not sparse
    poisoning — DDD/CDD should handle it.  All injected hits stay
    under 16σ so the extreme escape hatch can not fire either.
    """
    ref = _clean_ref(seed=506)
    recent = np.random.default_rng(507).normal(0.0, 1.0, size=(50, 4))
    # 20/50 = 40 % hits at 10σ (past hard 8σ, below extreme 16σ).
    recent = _inject_sigma(recent, list(range(20)), sigma=10.0)

    lib = fill_lib(np.vstack([ref, recent]))
    dpd = DPD(
        reference_size=300, recent_size=50,
        mahal_threshold=4.0, min_mahal_outliers=3,
        mahal_hard_threshold=8.0,
        mahal_hard_max_fraction=0.30,
    )
    result = dpd.check(lib)

    # Diagnostic evidence still shows up in the result record …
    assert len(result.mahal_hard_indices) >= 15
    # … but the triggered flag is off because fraction ≥ 30 %.
    assert not result.mahal_hard_triggered, (
        "hard arm should have been suppressed: "
        f"hard_hits={len(result.mahal_hard_indices)}/50 "
        "≥ mahal_hard_max_fraction=0.30 → drift, not poisoning"
    )
    # And the extreme escape must stay off (10σ < 16σ).
    assert result.mahal_max < 16.0


def test_dpd_extreme_escape_fires_despite_hard_sparsity_gate() -> None:
    """
    Extreme escape hatch: when any sample exceeds ``2 × hard`` (≈16σ)
    the hard arm fires unconditionally, even when the hit fraction
    would normally trip the sparsity gate.

    Rationale (from the paper's Section IV-C note): a 4σ uniform shift
    on 4-dim N(0, I) produces Mahalanobis values topping out around
    10-12σ; anything past 16σ is physically incompatible with a
    realistic drift and can only come from injected data.
    """
    ref = _clean_ref(seed=508)
    recent = np.random.default_rng(509).normal(0.0, 1.0, size=(50, 4))
    # 30/50 = 60 % hits at 20σ → sparsity gate would suppress, escape wins.
    recent = _inject_sigma(recent, list(range(30)), sigma=20.0)

    lib = fill_lib(np.vstack([ref, recent]))
    dpd = DPD(
        reference_size=300, recent_size=50,
        mahal_threshold=4.0, min_mahal_outliers=3,
        mahal_hard_threshold=8.0,
        mahal_hard_max_fraction=0.30,
    )
    result = dpd.check(lib)

    assert result.poisoning_detected
    assert result.mahal_hard_triggered, (
        f"extreme escape should fire unconditionally past 2 × hard "
        f"(mahal_max={result.mahal_max:.1f}σ, 2×hard=16σ)"
    )
    assert result.mahal_max > 16.0


def test_dpd_soft_rule_suppressed_when_widespread() -> None:
    """
    Soft-arm sparsity gate: if ≥ 50 % of the recent window is past 4σ
    the soft arm must be suppressed — a uniform shift tips every
    sample past 4σ and that pattern is drift, not poisoning.
    """
    ref = _clean_ref(seed=510)
    recent = np.random.default_rng(511).normal(0.0, 1.0, size=(50, 4))
    # 30/50 = 60 % past 5σ (past soft, below hard).
    recent = _inject_sigma(recent, list(range(30)), sigma=5.0)

    lib = fill_lib(np.vstack([ref, recent]))
    dpd = DPD(
        reference_size=300, recent_size=50,
        mahal_threshold=4.0, min_mahal_outliers=3,
        mahal_hard_threshold=8.0,
    )
    result = dpd.check(lib)

    assert not result.mahal_soft_triggered, (
        "soft arm should be suppressed once ≥ 50 % of the window is "
        f"past 4σ: soft_hits={len(result.mahal_anomalous_indices)}/50"
    )


def test_dpd_if_sparsity_gate_suppresses_above_50_percent() -> None:
    """
    Isolation-Forest sparsity gate: once IF flags ≥ 50 % of the recent
    window it has detected a population-wide shift, not sparse
    injections.  The IF arm must then be suppressed so DDD can handle
    the signal; otherwise every drift fires DATA_POISONING at the
    security subsystem.
    """
    rng = np.random.default_rng(512)
    ref = rng.normal(0.0, 1.0, size=(300, 4))
    # 5σ shift on every sample — IF will flag the entire window.
    recent = rng.normal(5.0, 1.0, size=(50, 4))

    lib = fill_lib(np.vstack([ref, recent]))
    dpd = DPD(
        reference_size=300, recent_size=50,
        if_contamination=0.02, contamination_threshold=0.10,
        mahal_threshold=4.0, min_mahal_outliers=3,
        mahal_hard_threshold=8.0,
    )
    result = dpd.check(lib)

    # Precondition: we actually achieved the ≥ 50 % regime.
    assert result.if_anomaly_rate >= 0.5, (
        "test setup precondition: IF rate should be ≥ 50 % on a 5σ shift "
        f"(got {result.if_anomaly_rate:.2%})"
    )
    # Core assertion: sparsity gate engaged.
    assert not result.if_triggered, (
        f"IF sparsity gate failed: rate={result.if_anomaly_rate:.2%} "
        "should suppress the IF arm once ≥ 50 % of the window is flagged"
    )


# ---------------------------------------------------------------------------
# refit_reference — below-minimum guard
# ---------------------------------------------------------------------------

def test_dpd_refit_reference_below_minimum_noops() -> None:
    """
    ``refit_reference`` must no-op (not crash, not over-fit on scraps)
    when LIB holds fewer than ``recent_size`` rows.  The invariant: the
    previously fitted reference survives the call so a subsequent
    check() on a full LIB still behaves correctly.
    """
    ref = _clean_ref(seed=513, n=400)
    lib_full = fill_lib(ref, capacity=400)

    dpd = DPD(reference_size=300, recent_size=50)
    # Auto-fit the reference on the full LIB.
    dpd.check(lib_full)
    fitted_mean = dpd._ref_mean.copy()

    # Simulate "fresh LIB with only a handful of rows" and refit — must
    # leave the reference untouched (below recent_size guard).
    tiny = fill_lib(
        np.random.default_rng(514).normal(size=(10, 4)), capacity=400,
    )
    dpd.refit_reference(tiny)

    assert np.array_equal(dpd._ref_mean, fitted_mean), (
        "refit_reference must no-op below recent_size, preserving the "
        "previously fitted mean."
    )
