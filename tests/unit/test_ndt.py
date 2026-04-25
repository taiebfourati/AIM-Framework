"""
Tier-3 unit tests for ndt.ndt.NDT — the validation gate.

The NDT decides whether a freshly trained candidate is allowed to
replace the active MLIN.  Its decision surface has three knobs:

  * ``min_score``       — absolute AI-QoS floor the candidate must clear.
  * ``min_improvement`` — the candidate must beat the current model by at
    least this much.  Zero means *tie goes to challenger*; negative
    values tolerate small regressions (useful during concept drift
    where both models may underperform until more data arrives).
  * ``y_val`` vs ``y_val_gt`` — pseudo-label targets drive the pass/fail
    decision (backward-compat), while optional ground-truth targets are
    scored alongside and recorded for the audit trail.  There is a
    separate entry point ``validate_with_ground_truth`` that flips the
    policy: GT drives the decision and pseudo-labels are not used.

These tests pin that contract with scripted estimators so the decision
math can be checked by inspection — no fitting, no stochasticity.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

import ndt.ndt as ndt_mod
from ndt.ndt import NDT


# ---------------------------------------------------------------------------
# Scripted classifier — returns a fixed prediction sequence
# ---------------------------------------------------------------------------

class ScriptedClf(ClassifierMixin, BaseEstimator):
    """Classifier whose predictions are pinned at construction time.

    Inherits from the modern sklearn mixin so ``is_classifier`` returns
    ``True`` via the tags machinery, which NDT uses to pick the
    ``accuracy`` branch of ``_score``.
    """

    def __init__(self, preds=None):
        self.preds = preds
        # NDT never calls ``fit`` on the models it evaluates — they arrive
        # fitted from MTP-L / MTP-E.  Populate ``classes_`` up front so
        # the estimator is considered fitted by sklearn's is-fitted check.
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        n = len(np.atleast_2d(X))
        return np.asarray(self.preds)[:n]


# ---------------------------------------------------------------------------
# Scripted regressor — pinned .score() so R² is exact
# ---------------------------------------------------------------------------

class ScriptedReg(RegressorMixin, BaseEstimator):
    """
    Regressor whose ``.score()`` returns a fixed R² value.

    NDT's ``_score`` helper falls through to ``model.score(X, y)`` when
    ``is_classifier(model)`` is False.  Inheriting from ``RegressorMixin``
    gets ``is_classifier`` to return False for the right reason (the
    estimator-tags path), and overriding ``score`` directly gives
    pin-exact control over the R² value NDT sees without having to
    fabricate a realistic regression dataset.
    """

    def __init__(self, r2=0.9):
        self.r2 = r2

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(np.atleast_2d(X)))

    def score(self, X, y, sample_weight=None):
        return float(self.r2)


# ---------------------------------------------------------------------------
# Helpers — 100-sample holdout so accuracy values are exact percents
# ---------------------------------------------------------------------------

N_VAL = 100


def _X_val() -> np.ndarray:
    """Filler inputs — their content never matters for a scripted model."""
    return np.zeros((N_VAL, 2), dtype=float)


def _preds(n_zero: int) -> np.ndarray:
    """``n_zero`` zeros followed by ``N_VAL - n_zero`` ones."""
    return np.concatenate([np.zeros(n_zero, dtype=int),
                           np.ones(N_VAL - n_zero, dtype=int)])


def _y_all_zero() -> np.ndarray:
    return np.zeros(N_VAL, dtype=float)


def _y_all_one() -> np.ndarray:
    return np.ones(N_VAL, dtype=float)


# ---------------------------------------------------------------------------
# TestNDTThresholdSemantics — absolute floor + improvement check
# ---------------------------------------------------------------------------

class TestNDTThresholdSemantics:
    """Verify both the absolute floor and the improvement-over-baseline check."""

    def test_candidate_below_floor_fails(self) -> None:
        """Candidate scores 0.60 against a 0.70 floor — hard reject."""
        baseline = ScriptedClf(preds=_preds(50))     # 0.50 vs all-zero
        candidate = ScriptedClf(preds=_preds(60))    # 0.60 vs all-zero
        ndt = NDT(
            current_model_getter=lambda: baseline,
            min_score=0.70,
            min_improvement=0.0,
        )

        passed = ndt.validate(candidate, _X_val(), _y_all_zero())

        assert passed is False
        rec = ndt.last_result()
        assert rec["candidate_score"] == pytest.approx(0.60)
        assert rec["baseline_score"] == pytest.approx(0.50)
        # Improvement is positive (+0.10) but the floor rejects it.
        assert rec["improvement"] == pytest.approx(0.10)
        assert rec["passed"] is False

    def test_candidate_above_floor_and_better_passes(self) -> None:
        """0.80 candidate beats 0.70 baseline, clears 0.70 floor — PASS."""
        baseline = ScriptedClf(preds=_preds(70))     # 0.70
        candidate = ScriptedClf(preds=_preds(80))    # 0.80
        ndt = NDT(
            current_model_getter=lambda: baseline,
            min_score=0.70,
            min_improvement=0.0,
        )

        passed = ndt.validate(candidate, _X_val(), _y_all_zero())

        assert passed is True
        rec = ndt.last_result()
        assert rec["candidate_score"] == pytest.approx(0.80)
        assert rec["baseline_score"] == pytest.approx(0.70)
        assert rec["improvement"] == pytest.approx(0.10)

    def test_candidate_above_floor_but_worse_fails_improvement(self) -> None:
        """0.80 candidate, 0.85 baseline, min_improvement=0.0 → regression rejected."""
        baseline = ScriptedClf(preds=_preds(85))     # 0.85
        candidate = ScriptedClf(preds=_preds(80))    # 0.80
        ndt = NDT(
            current_model_getter=lambda: baseline,
            min_score=0.70,                          # candidate clears floor
            min_improvement=0.0,                     # tie-or-better required
        )

        passed = ndt.validate(candidate, _X_val(), _y_all_zero())

        assert passed is False
        rec = ndt.last_result()
        assert rec["improvement"] == pytest.approx(-0.05)

    def test_per_call_min_score_override_shadows_instance_floor(self) -> None:
        """``validate(min_score=...)`` overrides the instance-level floor."""
        baseline = ScriptedClf(preds=_preds(50))     # 0.50
        candidate = ScriptedClf(preds=_preds(80))    # 0.80
        ndt = NDT(
            current_model_getter=lambda: baseline,
            min_score=0.70,
            min_improvement=0.0,
        )

        # Pump the floor per-call to 0.90 → candidate's 0.80 no longer qualifies.
        passed = ndt.validate(
            candidate, _X_val(), _y_all_zero(), min_score=0.90,
        )

        assert passed is False
        rec = ndt.last_result()
        assert rec["min_score"] == pytest.approx(0.90)
        assert rec["candidate_score"] == pytest.approx(0.80)

    def test_no_current_model_baseline_is_zero(self) -> None:
        """When the getter returns None, baseline=0 and only the floor gates."""
        candidate = ScriptedClf(preds=_preds(80))
        ndt = NDT(
            current_model_getter=lambda: None,
            min_score=0.70,
            min_improvement=0.0,
        )

        passed = ndt.validate(candidate, _X_val(), _y_all_zero())

        assert passed is True
        rec = ndt.last_result()
        assert rec["baseline_score"] == pytest.approx(0.0)
        assert rec["improvement"] == pytest.approx(0.80)


# ---------------------------------------------------------------------------
# TestNDTTieGoesToChallenger — the boundary of the improvement check
# ---------------------------------------------------------------------------

class TestNDTTieGoesToChallenger:
    """
    ``passes_improvement = improvement >= self.min_improvement`` is
    inclusive — the boundary belongs to the challenger.

    This matters in practice: during concept drift both models may score
    similarly on the holdout until more post-drift data arrives.  A
    strict greater-than check would starve retraining; the inclusive
    rule lets the fresher candidate win ties.
    """

    def test_exact_tie_passes_with_zero_min_improvement(self) -> None:
        """candidate == baseline, min_improvement=0.0 → candidate wins."""
        baseline = ScriptedClf(preds=_preds(80))     # 0.80
        candidate = ScriptedClf(preds=_preds(80))    # 0.80
        ndt = NDT(
            current_model_getter=lambda: baseline,
            min_score=0.70,
            min_improvement=0.0,
        )

        passed = ndt.validate(candidate, _X_val(), _y_all_zero())

        assert passed is True
        rec = ndt.last_result()
        assert rec["improvement"] == pytest.approx(0.0)

    def test_candidate_at_negative_tolerance_boundary_passes(self) -> None:
        """
        ``candidate - baseline == min_improvement`` → PASS.

        Picks binary-exact accuracy values (1.0, 0.75, 0.25 — all ratios
        whose denominator is a power of 2) so the subtraction is
        bit-precise and the boundary comparison is unambiguous.  With
        the naïve choice 0.80/0.75 the FP error on ``0.75 - 0.80``
        dropped the improvement just below ``-0.05`` and the boundary
        test started probing IEEE-754 rather than NDT's decision rule.
        """
        baseline = ScriptedClf(preds=_preds(100))    # 1.00 exactly
        candidate = ScriptedClf(preds=_preds(75))    # 0.75 exactly
        ndt = NDT(
            current_model_getter=lambda: baseline,
            min_score=0.70,
            min_improvement=-0.25,
        )

        passed = ndt.validate(candidate, _X_val(), _y_all_zero())

        assert passed is True
        rec = ndt.last_result()
        # 0.75 - 1.00 == -0.25 exactly in double precision
        assert rec["improvement"] == -0.25

    def test_candidate_just_below_tolerance_fails(self) -> None:
        """One accuracy point past the boundary flips PASS → FAIL."""
        baseline = ScriptedClf(preds=_preds(100))    # 1.00
        candidate = ScriptedClf(preds=_preds(74))    # 0.74 (≈, but strictly < 0.75)
        ndt = NDT(
            current_model_getter=lambda: baseline,
            min_score=0.70,
            min_improvement=-0.25,
        )

        passed = ndt.validate(candidate, _X_val(), _y_all_zero())

        assert passed is False
        rec = ndt.last_result()
        assert rec["improvement"] == pytest.approx(-0.26)

    def test_floor_still_gates_even_when_tie_would_pass(self) -> None:
        """
        An exact tie at the floor passes; an exact tie *below* the
        floor fails on the absolute AI-QoS check regardless of how
        permissive ``min_improvement`` is.
        """
        # Both at 0.65 — below the 0.70 floor but equal to each other.
        baseline = ScriptedClf(preds=_preds(65))
        candidate = ScriptedClf(preds=_preds(65))
        ndt = NDT(
            current_model_getter=lambda: baseline,
            min_score=0.70,
            min_improvement=-1.0,               # essentially disabled
        )

        passed = ndt.validate(candidate, _X_val(), _y_all_zero())

        assert passed is False
        rec = ndt.last_result()
        assert rec["improvement"] == pytest.approx(0.0)
        assert rec["candidate_score"] == pytest.approx(0.65)


# ---------------------------------------------------------------------------
# TestNDTDualScoreDivergence — pseudo vs ground-truth semantics
# ---------------------------------------------------------------------------

class TestNDTDualScoreDivergence:
    """
    NDT has two entry points:

      * ``validate`` — decision driven by ``y_val`` (LOB pseudo-labels);
        ``y_val_gt`` is optional and, when given, is recorded but
        deliberately does not alter the outcome.
      * ``validate_with_ground_truth`` — decision driven by the real
        labels and the history record is tagged
        ``"validation_mode": "ground_truth"``.

    The split exists because pseudo-labels reflect the *old* MLIN's
    belief: a candidate that correctly learns a new regime looks bad
    against pseudo-labels.  Tests that exercise this split pin the
    divergence at its most extreme (pseudo says pass, GT says fail and
    vice versa) so we can't silently regress either branch.
    """

    def test_pseudo_drives_decision_gt_only_recorded(self) -> None:
        """
        Pseudo-labels favour the candidate (0.90 vs 0.20).  Ground
        truth does the opposite (0.10 vs 0.80).  The decision must
        follow pseudo, but the history must contain the GT scores.
        """
        # Candidate: 90 zeros then 10 ones
        # Baseline: 20 zeros then 80 ones
        candidate = ScriptedClf(preds=_preds(90))
        baseline = ScriptedClf(preds=_preds(20))
        ndt = NDT(
            current_model_getter=lambda: baseline,
            min_score=0.70,
            min_improvement=0.0,
        )

        passed = ndt.validate(
            candidate,
            X_val=_X_val(),
            y_val=_y_all_zero(),          # pseudo: candidate=0.90, baseline=0.20
            y_val_gt=_y_all_one(),        # gt:     candidate=0.10, baseline=0.80
        )

        assert passed is True, "pseudo-label decision must pass"
        rec = ndt.last_result()
        # Pseudo-label scores (drive the decision)
        assert rec["candidate_score"] == pytest.approx(0.90)
        assert rec["baseline_score"] == pytest.approx(0.20)
        assert rec["improvement"] == pytest.approx(0.70)
        # Ground-truth scores recorded but not used for the decision
        assert rec["candidate_gt_score"] == pytest.approx(0.10)
        assert rec["baseline_gt_score"] == pytest.approx(0.80)
        # Mode key is absent for validate() — only set by validate_with_ground_truth
        assert "validation_mode" not in rec

    def test_validate_without_gt_leaves_gt_fields_none(self) -> None:
        """Skipping ``y_val_gt`` must leave the GT record fields as None."""
        candidate = ScriptedClf(preds=_preds(80))
        baseline = ScriptedClf(preds=_preds(70))
        ndt = NDT(
            current_model_getter=lambda: baseline,
            min_score=0.70,
            min_improvement=0.0,
        )

        ndt.validate(candidate, _X_val(), _y_all_zero())

        rec = ndt.last_result()
        assert rec["candidate_gt_score"] is None
        assert rec["baseline_gt_score"] is None

    def test_validate_with_ground_truth_uses_gt_for_decision(self) -> None:
        """
        ``validate_with_ground_truth`` ignores pseudo-labels entirely —
        candidate vs GT is 0.10, baseline vs GT is 0.80 → FAIL.
        """
        candidate = ScriptedClf(preds=_preds(90))   # vs all-one: 0.10
        baseline = ScriptedClf(preds=_preds(20))    # vs all-one: 0.80
        ndt = NDT(
            current_model_getter=lambda: baseline,
            min_score=0.70,
            min_improvement=0.0,
        )

        passed = ndt.validate_with_ground_truth(
            candidate,
            X_val=_X_val(),
            y_true_gt=_y_all_one(),
        )

        assert passed is False
        rec = ndt.last_result()
        assert rec["candidate_score"] == pytest.approx(0.10)
        assert rec["baseline_score"] == pytest.approx(0.80)
        assert rec["improvement"] == pytest.approx(-0.70)
        assert rec["validation_mode"] == "ground_truth"
        # The dedicated GT fields echo the primary scores when GT drives.
        assert rec["candidate_gt_score"] == pytest.approx(0.10)
        assert rec["baseline_gt_score"] == pytest.approx(0.80)

    def test_validate_with_ground_truth_passes_when_gt_better(self) -> None:
        """Same entry point, flipped stakes: candidate tracks GT → PASS."""
        candidate = ScriptedClf(preds=_preds(10))   # vs all-one: 0.90
        baseline = ScriptedClf(preds=_preds(20))    # vs all-one: 0.80
        ndt = NDT(
            current_model_getter=lambda: baseline,
            min_score=0.70,
            min_improvement=0.0,
        )

        passed = ndt.validate_with_ground_truth(
            candidate,
            X_val=_X_val(),
            y_true_gt=_y_all_one(),
        )

        assert passed is True
        rec = ndt.last_result()
        assert rec["candidate_score"] == pytest.approx(0.90)
        assert rec["baseline_score"] == pytest.approx(0.80)
        assert rec["improvement"] == pytest.approx(0.10)
        assert rec["validation_mode"] == "ground_truth"

    def test_validate_divergence_does_not_pollute_history_of_other_call(self) -> None:
        """
        A dual-score ``validate`` followed by a ``validate_with_ground_truth``
        should append two distinct history entries, each with its own
        score semantics — no field bleed between calls.
        """
        candidate = ScriptedClf(preds=_preds(90))
        baseline = ScriptedClf(preds=_preds(20))
        ndt = NDT(
            current_model_getter=lambda: baseline,
            min_score=0.70,
            min_improvement=0.0,
        )

        ndt.validate(
            candidate, _X_val(),
            y_val=_y_all_zero(),
            y_val_gt=_y_all_one(),
        )
        ndt.validate_with_ground_truth(
            candidate, _X_val(), y_true_gt=_y_all_one(),
        )

        assert len(ndt.history) == 2
        pseudo_rec, gt_rec = ndt.history
        assert "validation_mode" not in pseudo_rec
        assert gt_rec["validation_mode"] == "ground_truth"
        # Primary scores diverge between the two records as designed.
        assert pseudo_rec["candidate_score"] == pytest.approx(0.90)
        assert gt_rec["candidate_score"] == pytest.approx(0.10)

    def test_pseudo_fail_is_not_overridden_by_gt_pass(self) -> None:
        """
        Symmetric to ``test_pseudo_drives_decision_gt_only_recorded``:
        when pseudo-labels reject the candidate but ground truth would
        accept it, the decision must still be FAIL.

        This pins the invariant that ``validate()`` NEVER uses GT for
        the decision — callers who want GT-driven outcomes must use
        ``validate_with_ground_truth``.  Without this assertion a
        refactor could accidentally let GT scores "rescue" a candidate
        that failed pseudo-label evaluation, silently changing the
        deployment policy.
        """
        # Candidate: 10 zeros then 90 ones
        #   vs all-zero (pseudo): 0.10; vs all-one (gt): 0.90
        # Baseline: 80 zeros then 20 ones
        #   vs all-zero (pseudo): 0.80; vs all-one (gt): 0.20
        candidate = ScriptedClf(preds=_preds(10))
        baseline = ScriptedClf(preds=_preds(80))
        ndt = NDT(
            current_model_getter=lambda: baseline,
            min_score=0.70,
            min_improvement=0.0,
        )

        passed = ndt.validate(
            candidate,
            X_val=_X_val(),
            y_val=_y_all_zero(),      # pseudo: cand=0.10, base=0.80 → FAIL
            y_val_gt=_y_all_one(),    # gt:     cand=0.90, base=0.20 → would PASS
        )

        assert passed is False, (
            "pseudo-label FAIL must NOT be overridden by a better GT score"
        )
        rec = ndt.last_result()
        # Pseudo-label scores drove the decision
        assert rec["candidate_score"] == pytest.approx(0.10)
        assert rec["baseline_score"] == pytest.approx(0.80)
        assert rec["improvement"] == pytest.approx(-0.70)
        # GT scores recorded for audit but NOT used
        assert rec["candidate_gt_score"] == pytest.approx(0.90)
        assert rec["baseline_gt_score"] == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# TestNDTRegressor — R² path through _score
# ---------------------------------------------------------------------------

class TestNDTRegressor:
    """
    NDT must work for regressors.  ``_score`` branches on
    ``is_classifier(model)`` and falls through to ``model.score(X, y)``
    for regressors — which returns R² for sklearn regressors by default.
    These tests pin the regressor branch end-to-end.

    Scripted scores (via ``ScriptedReg.score``) let us check the decision
    math exactly without fabricating a regression dataset.
    """

    def test_regressor_passes_when_r2_beats_floor_and_baseline(self) -> None:
        baseline = ScriptedReg(r2=0.60)
        candidate = ScriptedReg(r2=0.90)
        ndt = NDT(
            current_model_getter=lambda: baseline,
            min_score=0.70,
            min_improvement=0.0,
        )

        passed = ndt.validate(candidate, _X_val(), _y_all_zero())

        assert passed is True
        rec = ndt.last_result()
        assert rec["candidate_score"] == pytest.approx(0.90)
        assert rec["baseline_score"] == pytest.approx(0.60)
        assert rec["improvement"] == pytest.approx(0.30)

    def test_regressor_below_floor_fails(self) -> None:
        """R² below the AI-QoS floor is rejected even if it beats baseline."""
        baseline = ScriptedReg(r2=0.30)
        candidate = ScriptedReg(r2=0.60)    # 0.60 < 0.70 floor
        ndt = NDT(
            current_model_getter=lambda: baseline,
            min_score=0.70,
            min_improvement=0.0,
        )

        passed = ndt.validate(candidate, _X_val(), _y_all_zero())

        assert passed is False
        rec = ndt.last_result()
        assert rec["candidate_score"] == pytest.approx(0.60)
        assert rec["improvement"] == pytest.approx(0.30)

    def test_regressor_negative_r2_clipped_to_its_sign(self) -> None:
        """
        sklearn's R² can go below zero when the model is worse than a
        constant predictor.  NDT must propagate that negative value
        through ``_score`` unmodified — the floor comparison will still
        reject it.
        """
        baseline = ScriptedReg(r2=0.50)
        candidate = ScriptedReg(r2=-0.30)   # worse than a mean predictor
        ndt = NDT(
            current_model_getter=lambda: baseline,
            min_score=0.70,
            min_improvement=0.0,
        )

        passed = ndt.validate(candidate, _X_val(), _y_all_zero())

        assert passed is False
        rec = ndt.last_result()
        assert rec["candidate_score"] == pytest.approx(-0.30)
        assert rec["improvement"] == pytest.approx(-0.80)

    def test_regressor_uses_score_method_not_predict_equals(self) -> None:
        """
        Route proof: NDT calls ``.score()`` on regressors, not
        ``predict == y``.  We use a ``ScriptedReg`` whose ``predict``
        would always return zero (zero accuracy against any non-zero y)
        but whose ``.score`` is pinned at 0.90.  The 0.90 value must be
        what NDT sees — confirming the regressor branch of ``_score``.
        """
        baseline = ScriptedReg(r2=0.0)
        candidate = ScriptedReg(r2=0.90)
        ndt = NDT(
            current_model_getter=lambda: baseline,
            min_score=0.70, min_improvement=0.0,
        )
        # Non-zero y — ``predict`` returns zeros, so ``predict == y`` would
        # give 0 accuracy.  Verifying candidate_score == 0.90 proves NDT
        # took the R² path.
        y_nonzero = np.ones(N_VAL, dtype=float)
        ndt.validate(candidate, _X_val(), y_nonzero)
        rec = ndt.last_result()
        assert rec["candidate_score"] == pytest.approx(0.90)


# ---------------------------------------------------------------------------
# TestNDTMLflowLogging — MLflow round-trip with a mock client
# ---------------------------------------------------------------------------

class _MockMlflowClient:
    """In-memory recorder — records every log_metric / set_tag call."""
    def __init__(self, tracking_uri=None):
        self.tracking_uri = tracking_uri
        self.metrics: list[tuple[str, str, float]] = []
        self.tags: list[tuple[str, str, str]] = []

    def log_metric(self, run_id, key, value):
        self.metrics.append((run_id, key, float(value)))

    def set_tag(self, run_id, key, value):
        self.tags.append((run_id, key, str(value)))


class _FailingMlflowClient:
    """Always raises — simulates an unreachable tracking server."""
    def __init__(self, tracking_uri=None):
        pass

    def log_metric(self, *args, **kwargs):
        raise ConnectionError("tracking server unreachable")

    def set_tag(self, *args, **kwargs):
        raise ConnectionError("tracking server unreachable")


class TestNDTMLflowLogging:
    """
    Pin the MLflow integration contract without requiring a real tracking
    server.  We patch ``ndt.ndt._mlflow_client`` so ``_log_to_mlflow``
    instantiates our mock instead of the real ``MlflowClient``.
    """

    def test_validate_with_run_id_logs_core_metrics_and_passed_tag(
        self, monkeypatch
    ) -> None:
        """
        On a successful validate(...run_id=X), NDT must log three core
        metrics (candidate_score, baseline_score, improvement) and set
        the ndt_passed tag.
        """
        captured: dict[str, _MockMlflowClient] = {}

        def client_factory():
            def _build(tracking_uri=None):
                c = _MockMlflowClient(tracking_uri=tracking_uri)
                captured["client"] = c
                return c
            return _build

        monkeypatch.setattr(ndt_mod, "_mlflow_client", client_factory)

        baseline = ScriptedClf(preds=_preds(70))
        candidate = ScriptedClf(preds=_preds(80))
        ndt = NDT(
            current_model_getter=lambda: baseline,
            min_score=0.70, min_improvement=0.0,
        )

        passed = ndt.validate(
            candidate, _X_val(), _y_all_zero(), run_id="run-abc",
        )
        assert passed is True

        client = captured["client"]
        metric_keys = {k for _, k, _ in client.metrics}
        assert "ndt_candidate_score" in metric_keys
        assert "ndt_baseline_score" in metric_keys
        assert "ndt_improvement" in metric_keys

        # Single ndt_passed tag with lowercase "true" (matches implementation)
        passed_tags = [t for t in client.tags if t[1] == "ndt_passed"]
        assert len(passed_tags) == 1
        assert passed_tags[0] == ("run-abc", "ndt_passed", "true")

        # Every call used the same run_id we passed in
        assert all(run_id == "run-abc" for run_id, _, _ in client.metrics)

        # Latch flipped to True after successful call
        assert ndt._mlflow_available is True

    def test_validate_without_run_id_skips_mlflow_entirely(
        self, monkeypatch
    ) -> None:
        """
        Omitting ``run_id`` must short-circuit BEFORE the Mlflow client
        is even instantiated — no connection attempts at all.
        """
        called = {"count": 0}

        def client_factory():
            called["count"] += 1
            return _MockMlflowClient

        monkeypatch.setattr(ndt_mod, "_mlflow_client", client_factory)

        baseline = ScriptedClf(preds=_preds(70))
        candidate = ScriptedClf(preds=_preds(80))
        ndt = NDT(current_model_getter=lambda: baseline)
        ndt.validate(candidate, _X_val(), _y_all_zero())  # no run_id

        assert called["count"] == 0, "MLflow must not be touched without run_id"
        # Latch stays None (never checked) — neither True nor False
        assert ndt._mlflow_available is None

    def test_mlflow_failure_latches_off_and_skips_future_calls(
        self, monkeypatch
    ) -> None:
        """
        The first log exception flips ``_mlflow_available`` to False and
        subsequent validate() calls with ``run_id`` must not even
        instantiate a client — fail-fast to keep the retraining loop
        moving while the tracking server is down.
        """
        instantiations = {"count": 0}

        def failing_factory():
            def _build(tracking_uri=None):
                instantiations["count"] += 1
                return _FailingMlflowClient(tracking_uri=tracking_uri)
            return _build

        monkeypatch.setattr(ndt_mod, "_mlflow_client", failing_factory)

        baseline = ScriptedClf(preds=_preds(70))
        candidate = ScriptedClf(preds=_preds(80))
        ndt = NDT(current_model_getter=lambda: baseline)

        # First call: the failing client raises, NDT catches, latch flips
        ndt.validate(candidate, _X_val(), _y_all_zero(), run_id="run-1")
        assert ndt._mlflow_available is False
        assert instantiations["count"] == 1

        # Second call: latch must short-circuit → factory not invoked again
        ndt.validate(candidate, _X_val(), _y_all_zero(), run_id="run-2")
        assert instantiations["count"] == 1, (
            "latch=False must short-circuit before creating a new client"
        )

    def test_mlflow_unavailable_module_latches_off_immediately(
        self, monkeypatch
    ) -> None:
        """
        If the module-level ``_mlflow_client()`` returns None (mlflow not
        installed), NDT must flip the latch to False without raising.
        """
        monkeypatch.setattr(ndt_mod, "_mlflow_client", lambda: None)

        baseline = ScriptedClf(preds=_preds(70))
        candidate = ScriptedClf(preds=_preds(80))
        ndt = NDT(current_model_getter=lambda: baseline)
        ndt.validate(candidate, _X_val(), _y_all_zero(), run_id="run-x")

        assert ndt._mlflow_available is False

    def test_validate_with_gt_logs_gt_scores_and_validation_mode_tag(
        self, monkeypatch
    ) -> None:
        """
        ``validate_with_ground_truth`` adds two extras to the MLflow log
        stream:
          * ``ndt_candidate_gt_score`` / ``ndt_baseline_gt_score`` metrics
          * ``ndt_validation_mode = "ground_truth"`` tag
        """
        captured: dict[str, _MockMlflowClient] = {}

        def client_factory():
            def _build(tracking_uri=None):
                c = _MockMlflowClient(tracking_uri=tracking_uri)
                captured["client"] = c
                return c
            return _build

        monkeypatch.setattr(ndt_mod, "_mlflow_client", client_factory)

        baseline = ScriptedClf(preds=_preds(70))
        candidate = ScriptedClf(preds=_preds(80))
        ndt = NDT(current_model_getter=lambda: baseline)

        ndt.validate_with_ground_truth(
            candidate, _X_val(), y_true_gt=_y_all_zero(), run_id="gt-run",
        )

        client = captured["client"]
        metric_keys = {k for _, k, _ in client.metrics}
        assert "ndt_candidate_gt_score" in metric_keys
        assert "ndt_baseline_gt_score" in metric_keys

        mode_tags = [t for t in client.tags if t[1] == "ndt_validation_mode"]
        assert mode_tags == [("gt-run", "ndt_validation_mode", "ground_truth")]

    def test_mlflow_logs_gt_scores_from_dual_score_validate(
        self, monkeypatch
    ) -> None:
        """
        ``validate()`` with ``y_val_gt`` provided → GT scores are logged
        as MLflow metrics even though they don't drive the decision.
        """
        captured: dict[str, _MockMlflowClient] = {}

        def client_factory():
            def _build(tracking_uri=None):
                c = _MockMlflowClient(tracking_uri=tracking_uri)
                captured["client"] = c
                return c
            return _build

        monkeypatch.setattr(ndt_mod, "_mlflow_client", client_factory)

        candidate = ScriptedClf(preds=_preds(90))   # pseudo=0.90, gt=0.10
        baseline = ScriptedClf(preds=_preds(20))    # pseudo=0.20, gt=0.80
        ndt = NDT(current_model_getter=lambda: baseline,
                  min_score=0.70, min_improvement=0.0)

        ndt.validate(
            candidate, _X_val(),
            y_val=_y_all_zero(),
            y_val_gt=_y_all_one(),
            run_id="dual-run",
        )

        client = captured["client"]
        gt_cand = [v for _, k, v in client.metrics if k == "ndt_candidate_gt_score"]
        gt_base = [v for _, k, v in client.metrics if k == "ndt_baseline_gt_score"]
        assert gt_cand == [pytest.approx(0.10)]
        assert gt_base == [pytest.approx(0.80)]
