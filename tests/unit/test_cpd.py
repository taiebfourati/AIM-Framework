"""
Tier 1 property tests for detectors.cpd.CPD.

The CPD should:

* POSITIVE — flag concept poisoning when the live model's LOB outputs
  diverge from a clean shadow model's predictions, the LOB output
  distribution shifts relative to the reference, OR feature-output
  correlations reverse.
* NEGATIVE — stay silent when LOB outputs stay close to shadow-model
  predictions on the same feature distribution.  This case directly
  mirrors the live-dashboard false-positive CONCEPT_POISONING firings
  at s350 / s1450 on clean baselines.
* EDGE — handle constant LOB outputs (the NaN-in-pearsonr fix),
  report "not ready" for tiny buffers, and work for both classifier
  and regressor tasks.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Ridge

from detectors.cpd import CPD, CPDResult
from ._helpers import fill_lib, fill_lob, iid_gaussian


# ---------------------------------------------------------------------------
# Pinned estimators — used to isolate individual channels in the shadow-
# anchored corroboration rule.  ``clone()`` calls ``fit`` on these
# when the CPD registers them, so fit must be a no-op and the
# predictions must be stable regardless of the input shape.
# ---------------------------------------------------------------------------

class _ConstantClassifier(ClassifierMixin, BaseEstimator):
    """Classifier shadow that always predicts ``constant``."""

    def __init__(self, constant: int = 1) -> None:
        self.constant = constant
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):                             # noqa: D401 — no-op fit
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        n = np.atleast_2d(X).shape[0]
        return np.full(n, int(self.constant), dtype=int)


class _PinnedRegressor(RegressorMixin, BaseEstimator):
    """
    Regressor shadow that returns a pre-committed prediction vector,
    independent of X.  Used to isolate ``shadow_triggered`` so the
    KS / corr arms can be exercised on their own.
    """

    def __init__(self, preds=None) -> None:
        self.preds = preds

    def fit(self, X, y):                             # noqa: D401 — no-op fit
        return self

    def predict(self, X):
        n = np.atleast_2d(X).shape[0]
        preds = np.asarray(
            self.preds if self.preds is not None else np.zeros(n),
            dtype=float,
        ).ravel()
        if preds.shape[0] < n:
            # Pad with last value if caller asked for more rows.
            preds = np.concatenate([preds, np.full(n - preds.shape[0], preds[-1])])
        return preds[:n]


# ---------------------------------------------------------------------------
# Shared fixtures — deterministic clean training corpora
# ---------------------------------------------------------------------------

def _classifier_corpus(n: int, d: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Linearly separable 2-class corpus: label = sign(x·w + b).
    Both the live MLIN and the shadow can fit it well, so divergence
    on clean data should stay well below the 0.25 default threshold.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, d))
    w = rng.normal(0.0, 1.0, size=(d,))
    y = (X @ w > 0.0).astype(int)
    return X, y


def _regressor_corpus(n: int, d: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Simple linear regression corpus with small gaussian noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, d))
    w = rng.normal(0.0, 1.0, size=(d,))
    y = X @ w + rng.normal(0.0, 0.1, size=n)
    return X, y


# ---------------------------------------------------------------------------
# Positive cases — poisoning SHOULD be detected
# ---------------------------------------------------------------------------

def test_cpd_classifier_detects_label_flip() -> None:
    """
    Live model returns FLIPPED labels on recent samples — shadow should
    catch it (divergence ≈ 1.0) and the output-KS test should also fire
    because the output distribution shifts.
    """
    X_ref, y_ref = _classifier_corpus(600, 4, seed=0)

    # Live model == true labels on reference window
    lob_ref = y_ref[:300].astype(float)
    # Recent window: live model flips every label
    lob_rec = 1 - y_ref[300:].astype(float)

    lib = fill_lib(X_ref)
    lob = fill_lob(np.concatenate([lob_ref, lob_rec]))

    cpd = CPD(task="classifier", reference_size=300, recent_size=100,
              shadow_threshold=0.25)
    cpd.fit_reference(X_ref[:300], y_ref[:300], lob_ref)

    result = cpd.check(lib, lob)
    assert result.poisoning_detected
    assert result.shadow_triggered
    # Output distribution also changes dramatically
    assert result.shadow_divergence > 0.5


def test_cpd_regressor_detects_output_scale_attack() -> None:
    """
    Live regressor outputs scaled by 3× on recent window. Shadow
    (trained on clean data) will see large MAE vs LOB.
    """
    X_ref, y_ref = _regressor_corpus(600, 4, seed=1)

    # Clean LOB on ref, scaled LOB on recent
    lob_ref = y_ref[:300]
    lob_rec = y_ref[300:] * 3.0 + 5.0

    lib = fill_lib(X_ref)
    lob = fill_lob(np.concatenate([lob_ref, lob_rec]))

    cpd = CPD(task="regressor", reference_size=300, recent_size=100,
              shadow_threshold=0.30, output_ks_alpha=0.01,
              corr_threshold=0.40)
    cpd.fit_reference(X_ref[:300], y_ref[:300], lob_ref)

    result = cpd.check(lib, lob)
    assert result.poisoning_detected
    # At least one of the three checks should trigger
    assert (result.shadow_triggered
            or result.output_ks_triggered
            or result.corr_triggered)


def test_cpd_regressor_detects_correlation_reversal() -> None:
    """
    Reversing the sign of a key feature correlation while keeping the
    marginal output distribution roughly similar targets the
    correlation-consistency check specifically.
    """
    rng = np.random.default_rng(2)
    # Build two distinct regimes with opposite correlations in feature 0
    X_ref = rng.normal(0.0, 1.0, size=(300, 3))
    y_ref =  2.0 * X_ref[:, 0] + rng.normal(0.0, 0.1, size=300)
    X_rec = rng.normal(0.0, 1.0, size=(100, 3))
    # Live model outputs use -2x correlation instead of +2x
    lob_rec = -2.0 * X_rec[:, 0] + rng.normal(0.0, 0.1, size=100)

    lib = fill_lib(np.vstack([X_ref, X_rec]))
    lob = fill_lob(np.concatenate([y_ref, lob_rec]))

    cpd = CPD(task="regressor", reference_size=300, recent_size=100,
              shadow_threshold=0.30, output_ks_alpha=0.01,
              corr_threshold=0.40)
    cpd.fit_reference(X_ref, y_ref, y_ref)

    result = cpd.check(lib, lob)
    assert result.poisoning_detected
    assert result.corr_triggered
    assert result.corr_delta_max > 0.40


# ---------------------------------------------------------------------------
# Negative cases — clean data must NOT fire
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [100, 101, 102, 103, 104])
def test_cpd_classifier_quiet_on_clean_data(seed: int) -> None:
    """
    Live-model LOB == training labels across both ref and recent windows.
    Targets the user-reported s350 / s1450 false-positive CONCEPT_POISONING
    firings.
    """
    X, y = _classifier_corpus(600, 4, seed=seed)
    lob_all = y.astype(float)                # live == truth on clean data

    lib = fill_lib(X)
    lob = fill_lob(lob_all)

    cpd = CPD(task="classifier", reference_size=300, recent_size=100,
              shadow_threshold=0.25, output_ks_alpha=0.01,
              corr_threshold=0.40)
    cpd.fit_reference(X[:300], y[:300], lob_all[:300])

    result = cpd.check(lib, lob)
    assert not result.poisoning_detected, (
        f"false-positive: seed={seed} msg={result.message} "
        f"shadow_div={result.shadow_divergence:.3f} "
        f"ks_p={result.output_ks_pvalue:.4f} "
        f"corr_delta={result.corr_delta_max:.3f}"
    )


@pytest.mark.parametrize("seed", [200, 201, 202, 203, 204])
def test_cpd_regressor_quiet_on_clean_data(seed: int) -> None:
    """Same as above for a regressor corpus."""
    X, y = _regressor_corpus(600, 4, seed=seed)
    # Live model outputs = truth + small noise
    rng = np.random.default_rng(seed + 5000)
    lob_all = y + rng.normal(0.0, 0.05, size=len(y))

    lib = fill_lib(X)
    lob = fill_lob(lob_all)

    cpd = CPD(task="regressor", reference_size=300, recent_size=100,
              shadow_threshold=0.30, output_ks_alpha=0.01,
              corr_threshold=0.40)
    cpd.fit_reference(X[:300], y[:300], lob_all[:300])

    result = cpd.check(lib, lob)
    assert not result.poisoning_detected, (
        f"false-positive: seed={seed} msg={result.message}"
    )


def test_cpd_quiet_under_repeated_checks() -> None:
    """
    Mirror the dashboard tick pattern: push (x, y) one pair at a time
    and call check() often. No tick should ever false-positive.

    Important: the streaming data must come from the SAME label-generating
    process as the reference window — otherwise we are injecting concept
    drift and the detector is correct to fire. We generate one big corpus
    up front, fit on the head, and stream the tail.
    """
    # One big, homogeneous corpus — same w throughout, i.e. P(Y|X) fixed.
    X_all, y_all = _classifier_corpus(1000, 4, seed=500)
    lob_all = y_all.astype(float)

    lib = fill_lib(X_all[:300], capacity=2000)
    lob = fill_lob(lob_all[:300], capacity=2000)

    cpd = CPD(task="classifier", reference_size=300, recent_size=100)
    cpd.fit_reference(X_all[:300], y_all[:300], lob_all[:300])

    # Stream the next 600 (x, y) pairs from the same corpus, checking
    # every 50 steps. Nothing has changed — check() must stay silent.
    false_positives: list[int] = []
    for step in range(600):
        x  = X_all[300 + step]
        yi = lob_all[300 + step]
        lib.push(x)
        lob.push(np.array([yi]))
        if step % 50 == 0:
            result = cpd.check(lib, lob)
            if result.poisoning_detected:
                false_positives.append((step, result.message))

    assert not false_positives, (
        f"CPD false-positive firings on clean streaming data: "
        f"{false_positives}"
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_cpd_constant_lob_does_not_raise_nan() -> None:
    """
    If every recent LOB output is identical, pearsonr returns NaN.
    The NaN-safe _feature_correlations must coerce to 0 and prevent
    the detector from blowing up or silently misbehaving.
    """
    X, y = _classifier_corpus(400, 4, seed=600)
    # Ref: real labels; Recent: every output is 1 (constant)
    lob_ref = y[:300].astype(float)
    lob_rec = np.ones(100, dtype=float)

    lib = fill_lib(X)
    lob = fill_lob(np.concatenate([lob_ref, lob_rec]))

    cpd = CPD(task="classifier", reference_size=300, recent_size=100)
    cpd.fit_reference(X[:300], y[:300], lob_ref)

    result = cpd.check(lib, lob)       # must not raise
    assert isinstance(result, CPDResult)
    # corr_delta_max should be a finite number, NOT NaN
    assert np.isfinite(result.corr_delta_max)


def test_cpd_not_ready_returns_false() -> None:
    """Too few samples → no-op result, no exception."""
    X = iid_gaussian(50, 4, seed=700)
    lib = fill_lib(X)
    lob = fill_lob(np.zeros(50))

    cpd = CPD(task="classifier", reference_size=300, recent_size=100)
    result = cpd.check(lib, lob)

    assert isinstance(result, CPDResult)
    assert not result.poisoning_detected
    assert "need" in result.message.lower()


def test_cpd_custom_shadow_estimator() -> None:
    """Custom shadow estimator must be cloned and fitted correctly."""
    X, y = _regressor_corpus(400, 3, seed=800)
    lib = fill_lib(X)
    lob = fill_lob(y)

    cpd = CPD(task="regressor", reference_size=300, recent_size=100,
              shadow_estimator=Ridge(alpha=0.5))
    cpd.fit_reference(X[:300], y[:300], y[:300])
    assert isinstance(cpd._shadow, Ridge)

    result = cpd.check(lib, lob)
    assert isinstance(result, CPDResult)


# ---------------------------------------------------------------------------
# Shadow-anchored corroboration rule (paper Section IV-C):
#     detected = shadow_triggered AND (output_ks_triggered OR corr_triggered)
#
# The shadow is the required anchor because it cleanly distinguishes
# poisoning (where the attacker's labels contradict honest training)
# from organic drift (where shadow stays calibrated to the post-retrain
# regime).  KS and corr shifts also occur under drift, so neither can
# fire CPD on its own — each of the three "one-arm-alone" configurations
# below must keep ``poisoning_detected = False``.
# ---------------------------------------------------------------------------

def test_cpd_shadow_alone_does_not_fire() -> None:
    """
    Shadow diverges sharply from LOB while output distribution and
    feature-output correlations stay unchanged → must NOT fire.
    """
    X, y = _classifier_corpus(600, 4, seed=900)
    # Ref + recent LOB both all-zero so:
    #   - output distributions are identical → KS p = 1.0
    #   - std(LOB) = 0 on both sides → feature correlations pegged to
    #     0 by _feature_correlations → Δr = 0
    lob_all = np.zeros(len(X), dtype=float)

    lib = fill_lib(X)
    lob = fill_lob(lob_all)

    cpd = CPD(
        task="classifier", reference_size=300, recent_size=100,
        shadow_threshold=0.25,
        shadow_estimator=_ConstantClassifier(constant=1),
    )
    cpd.fit_reference(X[:300], y[:300], lob_all[:300])

    result = cpd.check(lib, lob)
    assert result.shadow_triggered, (
        f"setup precondition: shadow must fire (div={result.shadow_divergence:.3f})"
    )
    assert not result.output_ks_triggered
    assert not result.corr_triggered
    assert not result.poisoning_detected, (
        "shadow alone (without KS or corr corroboration) must NOT fire "
        "under the paper's AND-rule; otherwise an unstable shadow fit "
        "would false-fire on every retrain."
    )


def test_cpd_ks_alone_does_not_fire() -> None:
    """
    Output-distribution KS fires on a step change in LOB values while
    the shadow agrees with recent LOB and correlations stay flat →
    must NOT fire.
    """
    X, y = _classifier_corpus(600, 4, seed=901)
    # Ref LOB all zeros, recent LOB all ones → big KS jump.  Both
    # windows have zero LOB variance so correlations stay at 0.
    lob_ref = np.zeros(300, dtype=float)
    lob_rec = np.ones(300, dtype=float)
    lob_all = np.concatenate([lob_ref, lob_rec])

    lib = fill_lib(X)
    lob = fill_lob(lob_all)

    # Shadow pinned to constant=1 → agrees with every recent LOB sample.
    # NB: ``y`` must be multi-class so CPD installs our pinned shadow
    # instead of the DummyClassifier fallback used for single-class
    # references.
    cpd = CPD(
        task="classifier", reference_size=300, recent_size=100,
        shadow_threshold=0.25,
        shadow_estimator=_ConstantClassifier(constant=1),
    )
    cpd.fit_reference(X[:300], y[:300], lob_ref)

    result = cpd.check(lib, lob)
    assert result.output_ks_triggered, (
        f"setup precondition: KS must fire (p={result.output_ks_pvalue:.4f})"
    )
    assert not result.shadow_triggered
    assert not result.corr_triggered
    assert not result.poisoning_detected, (
        "KS alone (without shadow corroboration) must NOT fire — a "
        "post-retrain LOB distribution shift is DDD/CDD territory, "
        "not CPD territory."
    )


def test_cpd_corr_alone_does_not_fire() -> None:
    """
    Feature-output correlations flip sign (ref=+0.9, recent=-0.9, z>>4)
    but shadow predicts the recent LOB exactly and the overall LOB
    distribution stays similar → must NOT fire.
    """
    rng = np.random.default_rng(902)
    X_ref = rng.normal(0.0, 1.0, size=(300, 3))
    lob_ref = 2.0 * X_ref[:, 0] + rng.normal(0.0, 0.1, size=300)
    X_rec = rng.normal(0.0, 1.0, size=(100, 3))
    lob_rec = -2.0 * X_rec[:, 0] + rng.normal(0.0, 0.1, size=100)

    lib = fill_lib(np.vstack([X_ref, X_rec]))
    lob = fill_lob(np.concatenate([lob_ref, lob_rec]))

    # Pinned shadow returns the exact recent LOB → divergence = 0 even
    # though the correlation structure has flipped.
    cpd = CPD(
        task="regressor", reference_size=300, recent_size=100,
        shadow_threshold=0.30,
        output_ks_alpha=0.01,
        corr_threshold=0.40,
        corr_z_threshold=4.0,
        shadow_estimator=_PinnedRegressor(preds=lob_rec),
    )
    cpd.fit_reference(X_ref, lob_ref, lob_ref)

    result = cpd.check(lib, lob)
    assert result.corr_triggered, (
        f"setup precondition: corr must fire "
        f"(|Δr|={result.corr_delta_max:.3f}, z={result.corr_z_max:.2f})"
    )
    assert not result.shadow_triggered, (
        f"pinned shadow divergence should be 0 "
        f"(got {result.shadow_divergence:.3f})"
    )
    assert not result.poisoning_detected, (
        "corr alone (without shadow corroboration) must NOT fire — a "
        "one-off correlation swing on a noisy window is sampling wobble, "
        "not an attack."
    )


# ---------------------------------------------------------------------------
# Fisher-z gate — the second of the correlation rule's two conditions
# ---------------------------------------------------------------------------

def test_cpd_corr_fisher_z_gate_rejects_small_sample_noise() -> None:
    """
    Small reference + recent windows make Fisher-z SE large enough
    that a raw |Δr| above ``corr_threshold`` can still sit below the
    z_threshold.  In that regime the detector must stay silent — this
    is the load-bearing guard against the dashboard's s350/s1450
    CONCEPT_POISONING false positives (50-row GT slices produce
    |Δr| ≈ 0.5 under H0 through sampling noise alone).
    """
    # At n_ref = n_rec = 30 the Fisher-z SE is ≈ √(2/27) ≈ 0.27, so a
    # raw |Δr| ≈ 0.45 produces z ≈ 0.45/0.27 ≈ 1.7 — well below z=4.
    n_ref, n_rec = 30, 30
    rng = np.random.default_rng(903)

    X_ref = rng.normal(0.0, 1.0, size=(n_ref, 2))
    lob_ref = -0.4 * X_ref[:, 0] + rng.normal(0.0, 1.0, size=n_ref)
    X_rec = rng.normal(0.0, 1.0, size=(n_rec, 2))
    lob_rec = +0.5 * X_rec[:, 0] + rng.normal(0.0, 1.0, size=n_rec)

    lib = fill_lib(np.vstack([X_ref, X_rec]))
    lob = fill_lob(np.concatenate([lob_ref, lob_rec]))

    cpd = CPD(
        task="regressor", reference_size=n_ref, recent_size=n_rec,
        shadow_threshold=0.30,
        output_ks_alpha=0.01,
        corr_threshold=0.40,
        corr_z_threshold=4.0,
        shadow_estimator=_PinnedRegressor(preds=lob_rec),
    )
    cpd.fit_reference(X_ref, lob_ref, lob_ref)

    result = cpd.check(lib, lob)
    # Precondition: sample size is small enough that z stays below the
    # gate even if raw |Δr| has crossed (or is close to) its threshold.
    assert result.corr_z_max < 4.0, (
        "test setup precondition: Fisher-z should stay below z=4 at "
        f"n_ref=n_rec=30 (got z_max={result.corr_z_max:.2f})"
    )
    # Core assertion: the corr arm must stay silent while z is below
    # threshold, regardless of raw Δr magnitude.
    assert not result.corr_triggered, (
        f"Fisher-z gate failed: |Δr|_max={result.corr_delta_max:.3f}, "
        f"z_max={result.corr_z_max:.2f} — corr arm should require BOTH "
        "|Δr| > corr_threshold AND |z| > corr_z_threshold."
    )


# ---------------------------------------------------------------------------
# Single-class reference fallback — DummyClassifier
# ---------------------------------------------------------------------------

def test_cpd_single_class_reference_installs_dummy_shadow() -> None:
    """
    A post-retrain reference slice occasionally ends up single-class
    (e.g. rare-event detectors whose 50-row GT slice is all class 0).
    LogisticRegression would crash on such input; CPD must fall back
    to a DummyClassifier pinned to the present class and keep running.
    """
    X, _y = _classifier_corpus(400, 4, seed=904)
    y_single = np.zeros(300, dtype=int)             # one-class reference
    lob_single = y_single.astype(float)

    cpd = CPD(task="classifier", reference_size=300, recent_size=100)
    cpd.fit_reference(X[:300], y_single, lob_single)   # must not raise

    assert isinstance(cpd._shadow, DummyClassifier), (
        "single-class reference should install DummyClassifier fallback, "
        f"got {type(cpd._shadow).__name__}"
    )
    # DummyClassifier must be usable downstream.
    out = cpd._shadow.predict(X[:100])
    assert out.shape == (100,)
    assert (out == 0).all(), (
        "DummyClassifier must be pinned to the single reference class "
        "so the shadow stays calibrated to the post-deploy regime."
    )


def test_cpd_check_runs_on_single_class_reference() -> None:
    """
    Follow-on to the fallback test: after installing the DummyClassifier
    shadow, a full ``check()`` must complete without raising and the
    shadow disagreement rate should equal the fraction of recent LOB
    samples that are NOT the pinned class.
    """
    X, _y = _classifier_corpus(500, 4, seed=905)
    y_ref = np.zeros(300, dtype=int)
    lob_ref = np.zeros(300, dtype=float)
    # Recent LOB: half zeros, half ones — shadow agrees on the zeros
    # (50 %) and disagrees on the ones (50 %).
    lob_rec = np.concatenate([np.zeros(50), np.ones(50)])

    lib = fill_lib(X[:400])
    lob = fill_lob(np.concatenate([lob_ref, lob_rec]))

    cpd = CPD(task="classifier", reference_size=300, recent_size=100,
              shadow_threshold=0.25)
    cpd.fit_reference(X[:300], y_ref, lob_ref)
    result = cpd.check(lib, lob)

    assert result.shadow_divergence == pytest.approx(0.5, abs=1e-9), (
        "DummyClassifier(constant=0) should disagree on exactly the "
        f"non-zero half of recent LOB (got {result.shadow_divergence:.3f})"
    )
