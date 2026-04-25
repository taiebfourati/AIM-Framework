"""
Tier-2 integration tests — gap coverage called out by the qa-expert audit.

``test_retraining.py`` pins the happy paths and the cooldown / NDT-rejection
guardrails.  This file fills the gaps that audit identified:

* **CRITICAL escalation** — poisoning + a batch that overflows
  ``local_max_samples`` must route to MTP-E, not MTP-L.  The existing
  poisoning test stays under the local budget, so this branch was
  previously untested end-to-end.

* **MLflow registry side effects** — the ATM must call
  ``mtp_e.promote_to_production`` on deploy and ``mtp_e.mark_failed``
  after NDT rejection.  The fake already records these calls; this file
  asserts on them.

* **Post-rejection detector stability** — when NDT rejects a candidate,
  ``notify_model_updated`` MUST NOT be called, which means detector
  references are unchanged.  Streaming the same drifted batch a second
  time should therefore re-fire the detectors once the cooldown lifts.

* **Stationary baseline** — a long run of pure on-distribution traffic
  must not fire any MToUT.  This is the negative companion to every
  positive test in ``TestRetrainingHappyPaths`` and catches false-
  positive regressions that would silently trigger retraining loops on
  production-quiet systems.

* **DPostP cleaning in the ATM pipeline** — the ATM must invoke
  ``dpostp.process_training_batch`` before MTP training, so NaN rows
  are dropped and the resulting training batch seen by MTP-L/E is
  smaller than the raw LIB snapshot.  Previously only the DPostP unit
  tests exercised this behaviour; here we prove it integrates into the
  live retraining pipeline.
"""
from __future__ import annotations

import numpy as np
import pytest

from atm.atm import ATM, ATMPolicy, MTPVariant, TrainStatus
from rtp.rtp import RTPConfig, TriggerReason

from ._harness import (
    FakeNDT,
    build_pipeline,
    make_classifier_corpus,
    stream,
)


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def _drive(pipeline, X, y_true=None) -> None:
    stream(pipeline, X, y_true=y_true)


# ---------------------------------------------------------------------------
# CRITICAL escalation — poisoning + large batch → MTP-E
# ---------------------------------------------------------------------------

class TestCriticalEscalation:
    """
    The existing ``test_data_poisoning_routes_critical_to_local_first``
    keeps the batch small (n=150 << local_max_samples=500) so it lands
    on MTP-L.  The other branch of the CRITICAL rule — batch above the
    local budget → MTP-E — was previously uncovered at integration
    level.  This class exercises it end-to-end.
    """

    def test_poisoning_with_large_batch_escalates_to_external(self) -> None:
        """
        CRITICAL signal + batch > ``local_max_samples`` → MTP-E, not MTP-L.

        We set a tight local budget (30) so the realistic integration
        batch (~150 rows) has to escalate.  The fake MTP-E returns a
        fitted model so the rest of the pipeline completes honestly.
        """
        policy = ATMPolicy(
            critical_always_local_first=True,
            local_max_samples=30,              # tight budget forces escalation
            use_ndt=True, auto_deploy=True,
            max_retrain_attempts=1,
        )
        pipeline = build_pipeline(task="classifier", seed=100, policy=policy)
        pre_model_id = id(pipeline.aif.active_estimator)

        # 120 clean samples + 30 poisoning outliers (> 20σ each).
        X_clean, _ = make_classifier_corpus(120, 4, seed=43)
        rng = np.random.default_rng(44)
        X_bad = rng.normal(50.0, 1.0, size=(30, 4))
        _drive(pipeline, X_clean)
        _drive(pipeline, X_bad)

        assert pipeline.atm_results, "ATM was never invoked"
        result = pipeline.atm_results[-1]

        # Escalation path: MTP-E was used, MTP-L was not
        assert result.variant_used == MTPVariant.EXTERNAL, (
            f"expected MTP-E escalation; got {result.variant_used}"
        )
        assert pipeline.mtp_e.call_count >= 1
        assert pipeline.mtp_l.call_count == 0

        # If the training pipeline succeeded, the AIF must reflect the swap.
        if result.status == TrainStatus.SUCCESS:
            assert id(pipeline.rtp.aif.active_estimator) != pre_model_id
            assert result.deployed is True


# ---------------------------------------------------------------------------
# MLflow registry plumbing — promote_to_production / mark_failed
# ---------------------------------------------------------------------------

class TestMLflowRegistryPlumbing:
    """
    MTP-E exposes two side-effectful lifecycle hooks that ATM MUST call
    at the right time:

    * ``promote_to_production(run_id)`` — after a successful deploy, so
      the MLflow registry transitions the model from Staging → Production.
    * ``mark_failed(run_id)`` — after NDT rejection, so the registry
      records that this run was evaluated and discarded.

    The fake MTP-E records every call; the real contract would call
    ``MlflowClient.transition_model_version_stage``.
    """

    def test_promote_to_production_called_after_successful_mtpe_deploy(
        self,
    ) -> None:
        """Successful MTP-E run → ``promote_to_production`` called with its run_id."""
        pipeline = build_pipeline(
            task="classifier", seed=110,
            policy=ATMPolicy(
                prefer_variant=MTPVariant.EXTERNAL,
                use_ndt=True, auto_deploy=True,
                max_retrain_attempts=1,
            ),
        )
        X_drift, _ = make_classifier_corpus(200, 4, seed=42, shift=4.0)
        _drive(pipeline, X_drift)

        assert pipeline.atm_results
        result = pipeline.atm_results[-1]
        assert result.status == TrainStatus.SUCCESS
        assert result.deployed is True

        # The run_id reported by ATM must appear in the fake's promotions
        assert result.run_id in pipeline.mtp_e.promotions, (
            f"promote_to_production was not called with run_id={result.run_id}; "
            f"recorded promotions={pipeline.mtp_e.promotions}"
        )
        # mark_failed must NOT have been called on a successful deploy
        assert result.run_id not in pipeline.mtp_e.marked_failed

    def test_mark_failed_called_after_ndt_rejection_on_mtpe_run(self) -> None:
        """NDT rejects candidate → ``mark_failed`` called with its run_id."""
        failing_ndt = FakeNDT(verdict=False)
        pipeline = build_pipeline(
            task="classifier", seed=111,
            policy=ATMPolicy(
                prefer_variant=MTPVariant.EXTERNAL,
                use_ndt=True, auto_deploy=True,
                max_retrain_attempts=1,
            ),
            ndt=failing_ndt,
        )
        X_drift, _ = make_classifier_corpus(200, 4, seed=42, shift=4.0)
        _drive(pipeline, X_drift)

        assert pipeline.atm_results
        result = pipeline.atm_results[-1]
        assert result.status == TrainStatus.FAILED
        assert result.deployed is False

        # NDT was called
        assert failing_ndt.call_count >= 1
        # mark_failed invoked with the MTP-E run_id
        assert result.run_id in pipeline.mtp_e.marked_failed, (
            f"mark_failed was not called with run_id={result.run_id}; "
            f"marked_failed={pipeline.mtp_e.marked_failed}"
        )
        # And the run was NOT promoted
        assert result.run_id not in pipeline.mtp_e.promotions

    def test_mtpl_success_does_not_touch_mtpe_registry(self) -> None:
        """
        MTP-L deployments do not have an MLflow run_id — so
        ``promote_to_production`` / ``mark_failed`` on MTP-E are NEVER
        called for local training cycles.  Pins the isolation between
        the two variants.
        """
        pipeline = build_pipeline(task="classifier", seed=112)  # default policy
        X_drift, _ = make_classifier_corpus(200, 4, seed=42, shift=4.0)
        _drive(pipeline, X_drift)

        assert pipeline.atm_results
        assert pipeline.atm_results[-1].variant_used == MTPVariant.LOCAL
        assert pipeline.mtp_e.call_count == 0
        assert pipeline.mtp_e.promotions == []
        assert pipeline.mtp_e.marked_failed == []


# ---------------------------------------------------------------------------
# Post-rejection detector stability
# ---------------------------------------------------------------------------

class TestPostRejectionDetectorStability:
    """
    When NDT rejects a candidate, the active model is NOT replaced and
    ``rtp.notify_model_updated`` must NOT be called.  Consequently the
    detector references (DDD/DPD/CPD) remain unchanged from before the
    rejection — they still carry the pre-drift baseline.

    This is the complement of the self-healing test in
    ``TestPostDeploymentStability``: after a *successful* retrain the
    references SHOULD be refit; after a rejection they SHOULD NOT.
    """

    def test_ndt_rejection_preserves_detector_references(self) -> None:
        """
        After NDT rejection, DDD's reference is byte-identical to its
        pre-rejection value, and the model instance pointed to by AIF
        is unchanged.
        """
        failing_ndt = FakeNDT(verdict=False)
        pipeline = build_pipeline(task="classifier", seed=120, ndt=failing_ndt)

        # Snapshot the pre-drift detector references.
        ddd = pipeline.rtp.ddd
        dpd = pipeline.rtp.dpd

        # Let the pipeline warm up so the auto-fit references are set.
        X_warm, _ = make_classifier_corpus(100, 4, seed=121)
        _drive(pipeline, X_warm)

        ddd_ref_before = ddd._reference.copy() if ddd._reference is not None else None
        # DPD uses a triple of refs: isolation forest, covariance, mean vector.
        # The mean is the cheapest array-valued witness to refit.
        dpd_ref_mean_before = (
            dpd._ref_mean.copy() if getattr(dpd, "_ref_mean", None) is not None else None
        )
        dpd_iforest_before = getattr(dpd, "_iforest", None)
        pre_model = pipeline.aif.active_estimator

        # Now inject drift that WILL trigger ATM; NDT will reject.
        X_drift, _ = make_classifier_corpus(200, 4, seed=42, shift=4.0)
        _drive(pipeline, X_drift)

        assert pipeline.atm_results
        result = pipeline.atm_results[-1]
        assert result.status == TrainStatus.FAILED
        assert result.deployed is False

        # Model unchanged
        assert pipeline.aif.active_estimator is pre_model

        # Detector references unchanged — NOT refit on rejected candidate.
        if ddd_ref_before is not None:
            np.testing.assert_array_equal(
                ddd._reference, ddd_ref_before,
                err_msg="DDD reference was refit even though NDT rejected",
            )
        if dpd_ref_mean_before is not None:
            np.testing.assert_array_equal(
                dpd._ref_mean, dpd_ref_mean_before,
                err_msg="DPD _ref_mean was refit even though NDT rejected",
            )
            # Identity check on the IsolationForest estimator — a refit
            # would replace the object, so `is` equality is the sharpest
            # assertion that no refit occurred.
            assert dpd._iforest is dpd_iforest_before, (
                "DPD IsolationForest was replaced even though NDT rejected"
            )


# ---------------------------------------------------------------------------
# Stationary baseline — no drift → no MToUT
# ---------------------------------------------------------------------------

class TestStationaryBaseline:
    """
    The negative companion to every positive test in the file:
    on-distribution traffic must not fire any MToUT.  Mirrors the
    dashboard complaint of false-positive firings on clean baselines.
    """

    def test_stationary_stream_fires_no_severe_mtout(self) -> None:
        """
        Stream truly on-distribution traffic that the trained model
        classifies accurately, and verify NONE of the severe paper-
        mandated detectors false-fires.

        Two subtleties the earlier draft of this test missed:

        1. ``make_classifier_corpus(n, d, seed)`` draws ``X`` of shape
           ``(n, d)`` FIRST and then derives the weight vector ``w`` —
           changing ``n`` rewinds the rng past the X draw and lands at
           a different state for ``w``.  So
           ``make_classifier_corpus(1000, 4, 0)`` and
           ``make_classifier_corpus(600, 4, 0)`` produce the SAME first
           600 rows of ``X`` but DIFFERENT labels; feeding the 1000-row
           stream to a model trained on the 600-row corpus looks to
           the model like random noise (error ≈ 0.5) and CDD rightly
           fires.  Match the training call exactly.

        2. DDD's KS arm has a built-in per-check false-alarm rate of
           about α = 0.05 after Bonferroni.  Over many check intervals
           of a stationary stream, seeing a single DDD alarm is
           statistically expected.  The paper-relevant claim is that
           CONCEPT drift and DATA/CONCEPT poisoning do NOT fire on
           clean data — those are the assertions we keep strict.
        """
        pipeline = build_pipeline(task="classifier", seed=0)

        # Regenerate the training corpus with the EXACT same call
        # signature ``build_pipeline`` uses internally.  Because the rng
        # seed, shape, and operation order are identical, the returned
        # (X, y) is bit-for-bit the one the model was trained on.
        X_train, y_train = make_classifier_corpus(600, 4, seed=0)
        _drive(pipeline, X_train, y_true=y_train)

        # Paper-mandated strict invariants on a clean baseline:
        #   * NO concept-drift alarm (CDD should not fire — model
        #     predicts with training-set accuracy).
        #   * NO poisoning alarm of either flavour (those are CRITICAL
        #     and would trigger the heavy MTP-E/MTP-C path).
        #   * NO CRITICAL-severity MToUT at all.
        strict_reasons = {
            TriggerReason.CONCEPT_DRIFT,
            TriggerReason.DATA_POISONING,
            TriggerReason.CONCEPT_POISONING,
        }
        mtouts = [s for k, s in pipeline.events if k == "mtout"]
        for sig in mtouts:
            forbidden = strict_reasons.intersection(sig.reasons)
            assert not forbidden, (
                f"Stationary traffic raised paper-critical reason(s) "
                f"{forbidden} at step {sig.step}; full reasons: "
                f"{sig.reasons}"
            )
            # ``severity`` is a method returning one of the strings
            # {"CRITICAL","HIGH","MEDIUM","LOW"}.
            assert sig.severity() != "CRITICAL", (
                f"Stationary traffic raised a CRITICAL MToUT at step "
                f"{sig.step}: {sig}"
            )

        # Statistical-noise ceiling for DDD: over 600 samples / 50-step
        # check interval = 12 checks, with per-check false-alarm
        # probability ≈ α = 0.05, the expected number of spurious
        # alarms is ~0.6 and the 99th percentile is ≤ 3.  Anything
        # higher indicates a systematic bias, not statistical noise.
        assert len(mtouts) <= 3, (
            f"Stationary traffic fired {len(mtouts)} MToUTs — above "
            f"the expected statistical-noise ceiling of 3.  Reasons: "
            f"{[(s.step, s.reasons) for s in mtouts]}"
        )


# ---------------------------------------------------------------------------
# DPostP cleaning integrated into ATM
# ---------------------------------------------------------------------------

class TestDPostPCleaningInPipeline:
    """
    Prove that ATM actually invokes ``dpostp.process_training_batch``
    before training — ie NaN rows visible in LIB are dropped before the
    MTP ever sees them.  The DPostP unit tests already exercise the
    cleaning function in isolation; this test proves the wire-up holds.
    """

    def test_atm_drops_nan_rows_before_training(self) -> None:
        """
        Inject NaN rows into LIB (via direct buffer push), then fire a
        manual retrain.  The MTP-L spy must see FEWER rows than LIB
        contained — proving DPostP cleaned out the NaN rows on the way.
        """
        pipeline = build_pipeline(task="classifier", seed=130)

        # Populate LIB with 200 clean rows via normal streaming (so LOB
        # gets pseudo-labels assigned during observe()).
        X_clean, y_clean = make_classifier_corpus(200, 4, seed=131)
        _drive(pipeline, X_clean, y_true=y_clean)

        # Corrupt 40 rows of LIB in place — NaN on one column each.
        # The buffer stores ``Sample`` dataclasses whose ``.value`` is a
        # numpy array; writing into ``.value[0]`` mutates the array in
        # place, so no re-assignment into the deque is required.  This
        # bypasses the push() contract intentionally; we want a
        # controlled corruption to prove DPostP drops them.
        lib_buf = pipeline.rtp.buffers.lib._buf
        for i in range(40):
            lib_buf[i].value[0] = np.nan

        # Count LIB rows with NaN before we fire the retrain.
        lib_array = pipeline.rtp.buffers.lib.get_values()
        nan_rows_in_lib = int(np.any(np.isnan(lib_array), axis=1).sum())
        assert nan_rows_in_lib == 40, (
            f"test setup corrupt: expected 40 NaN rows in LIB, got {nan_rows_in_lib}"
        )

        # Fire a manual retrain so we don't depend on detector timing.
        pre_spy_count = pipeline.mtp_l.call_count
        result = pipeline.atm.operator_retrain(
            variant=MTPVariant.LOCAL, reason="integration-test-nan-cleaning",
        )

        assert result.status == TrainStatus.SUCCESS
        assert pipeline.mtp_l.call_count == pre_spy_count + 1

        # The spy recorded the shape MTP-L was given — it must be smaller
        # than LIB.  DPostP dropped at least the 40 NaN rows (possibly
        # more if the GT-only path trimmed further; we assert only the
        # NaN drop as a lower bound).
        mtp_rows = pipeline.mtp_l.last_X_shape[0]
        assert mtp_rows <= len(lib_array) - 40, (
            f"DPostP did not drop NaN rows: MTP-L saw {mtp_rows} rows, "
            f"LIB had {len(lib_array)} rows with 40 NaN rows."
        )
        # Positive: MTP-L still got enough to train on
        assert mtp_rows >= 20, (
            f"MTP-L under-fed after cleaning: {mtp_rows} rows"
        )


# ---------------------------------------------------------------------------
# Retry semantics — MTP-L failure + max_retrain_attempts > 1
# ---------------------------------------------------------------------------

class TestRetrySemantics:
    """
    ATM has a retry loop around the MTP training call.  When MTP-L
    raises on the first attempt, the loop must re-invoke it up to
    ``max_retrain_attempts`` times before giving up.  Pins the counter
    increment and the eventual fall-through to TrainStatus.FAILED.
    """

    def test_max_retrain_attempts_respected_on_repeated_failure(
        self, monkeypatch
    ) -> None:
        """MTP-L raises on every attempt → ATM retries exactly N times.

        Stream ON-distribution traffic so neither CDD nor DDD fires
        spuriously; otherwise the detector-driven MToUT would launch
        its own retrain cycle and the call counter would tally
        ``n_cycles * max_retrain_attempts`` rather than just the
        single operator-driven cycle's attempts.
        """
        pipeline = build_pipeline(
            task="classifier", seed=140,
            policy=ATMPolicy(
                use_ndt=True, auto_deploy=True,
                max_retrain_attempts=3,
            ),
        )

        # Patch the spy's train() to always raise, counting invocations.
        call_count = {"n": 0}

        def failing_train(X, y, base_model=None):
            call_count["n"] += 1
            raise RuntimeError("simulated MTP-L failure")

        monkeypatch.setattr(pipeline.mtp_l, "train", failing_train)

        # Populate LIB with ON-distribution traffic that the already-
        # trained model predicts accurately — no MToUT cycle will fire
        # from the stream itself, so our operator_retrain is the ONLY
        # retrain cycle in this test.  See harness note: seed=0 with
        # the matching size is the trained regime.
        X_stable, y_stable = make_classifier_corpus(600, 4, seed=140)
        _drive(pipeline, X_stable, y_true=y_stable)

        # Sanity: no retrain cycles yet beyond any that may have slipped
        # through during the warm-up stream.  Snapshot and assert the
        # OPERATOR cycle adds exactly ``max_retrain_attempts`` calls.
        pre_cycle_count = call_count["n"]

        result = pipeline.atm.operator_retrain(
            variant=MTPVariant.LOCAL, reason="retry-test",
        )

        assert result.status == TrainStatus.FAILED
        delta = call_count["n"] - pre_cycle_count
        assert delta == 3, (
            f"expected 3 MTP-L attempts per operator cycle "
            f"(max_retrain_attempts=3), operator cycle made {delta}; "
            f"total calls across stream+operator: {call_count['n']}"
        )
        assert result.attempts == 3
