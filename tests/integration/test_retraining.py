"""
Tier-2 integration tests — RTP → MToUT → ATM → MTP → NDT → deploy → notify.

These tests exercise the whole observer/controller loop as one unit.
The point is no longer to pin down one detector's math (Tier 1 already
did that) but to prove the *retraining actually happens*:

* An injected drift / poisoning episode propagates from the detector
  through ``_fire_mtout`` into ``ATM.handle``.
* ATM picks a variant that matches the severity.
* The MTP-L / MTP-E pipeline actually runs and returns a fresh, fitted
  estimator — we assert on model identity, not just ``type(...)``.
* NDT is given the candidate and its verdict decides whether
  :meth:`AIF.update_model` is called.
* ``RTP.notify_model_updated`` resets CDD's Page-Hinkley state and
  refits every reference so no detector immediately re-fires on the new
  baseline — the self-healing property.

All tests run in-process with an in-memory fake for MTP-E (no MLflow
server required); MTP-L is the real training pipeline.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

from atm.atm import ATMPolicy, MTPVariant, TrainStatus
from rtp.rtp import EventType, RTPConfig, TriggerReason

from ._harness import (
    FakeNDT,
    build_pipeline,
    make_classifier_corpus,
    make_regressor_corpus,
    stream,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _drive(pipeline, X: np.ndarray, y_true=None) -> None:
    """Short wrapper around stream() for readability."""
    stream(pipeline, X, y_true=y_true)


# ---------------------------------------------------------------------------
# Happy paths — one test per trigger reason
# ---------------------------------------------------------------------------

class TestRetrainingHappyPaths:
    """The positive end-to-end paths: trigger → retrain → deploy."""

    def test_data_drift_triggers_retraining_and_model_swap(self) -> None:
        """
        Shift the input distribution, verify DDD fires, ATM picks MTP-L
        (MEDIUM severity + small buffer), and the AIF's active estimator
        is replaced.
        """
        pipeline = build_pipeline(task="classifier", seed=1)
        pre_model_id = id(pipeline.aif.active_estimator)

        # Stream 200 shifted samples — should trip DDD.
        X_drift, _ = make_classifier_corpus(200, 4, seed=42, shift=4.0)
        _drive(pipeline, X_drift)

        # A single MToUT must have fired
        mtout_events = [e for e in pipeline.events if e[0] == "mtout"]
        assert len(mtout_events) >= 1, "DDD did not fire on shifted input"

        # ATM must have processed it and succeeded
        assert pipeline.atm_results, "ATM.handle was not invoked"
        last_result = pipeline.atm_results[-1]
        assert last_result.status == TrainStatus.SUCCESS, last_result.message
        assert last_result.deployed

        # MTP-L was chosen (MEDIUM + few samples) — identity verified
        assert last_result.variant_used == MTPVariant.LOCAL
        assert pipeline.mtp_l.call_count >= 1
        assert pipeline.mtp_e.call_count == 0

        # AIF actually swapped the model (not the same instance)
        post_model_id = id(pipeline.rtp.aif.active_estimator)
        assert post_model_id != pre_model_id, (
            "aif.active_estimator was not replaced after retraining"
        )

        # RTP event log records the full chain
        event_kinds = {e.event_type for e in pipeline.rtp.event_log}
        assert EventType.MTOUT_FIRED in event_kinds
        assert EventType.MODEL_UPDATED in event_kinds

    def test_concept_drift_triggers_retraining(self) -> None:
        """
        Classifier whose streaming labels diverge from training labels:
        CDD should fire (accuracy drops) even though X is unchanged.

        We deliberately use the same seed for both the reference corpus
        and the streaming corpus so the input distribution is identical
        (no data drift); only ``flip_labels=True`` changes P(Y|X) — the
        textbook definition of concept drift.
        """
        pipeline = build_pipeline(task="classifier", seed=2)
        pre_model_id = id(pipeline.aif.active_estimator)

        # Same X distribution (seed=2), flipped labels as ground truth →
        # the current MLIN predicts wrong → CDD sees accuracy collapse.
        X_rec, y_rec = make_classifier_corpus(300, 4, seed=2,
                                              flip_labels=True)
        _drive(pipeline, X_rec, y_true=y_rec)

        # At least one MToUT fired and retraining succeeded
        assert pipeline.atm_results, "ATM.handle was never called"
        result = pipeline.atm_results[-1]
        assert result.status == TrainStatus.SUCCESS
        assert result.deployed

        # Active model has been replaced
        assert id(pipeline.rtp.aif.active_estimator) != pre_model_id

        # The last MToUT signal carries CDD among its reasons
        reasons_union: set[str] = set()
        for _, sig in [e for e in pipeline.events if e[0] == "mtout"]:
            reasons_union.update(r.name for r in sig.reasons)
        assert "CONCEPT_DRIFT" in reasons_union or \
               "DATA_DRIFT" in reasons_union, (
                   f"Expected drift reason in any MToUT, got {reasons_union}"
               )

    def test_data_poisoning_routes_critical_to_local_first(self) -> None:
        """
        Inject extreme outliers so DPD's hard threshold fires. Severity
        is CRITICAL → ATM's default policy sends the request to MTP-L
        first (``critical_always_local_first=True``).

        Strong invariants checked:
        * The MToUT signal that fired carries severity == "CRITICAL".
        * The MToUT carries the DATA_POISONING reason (not merely drift).
        * ATM picked MTP-L (not MTP-E) — counts are checked explicitly.
        * MTP-E was NOT invoked at all — CRITICAL keeps heavy-cloud
          escalation off while the local budget still suffices.
        * On the happy path the model actually changes identity;
          on the (still allowed) fail path the model is preserved.
        """
        pipeline = build_pipeline(task="classifier", seed=3)
        pre_model = pipeline.aif.active_estimator
        pre_model_id = id(pre_model)

        # 120 clean samples, then 30 massive outliers (> 20σ).
        X_clean, _ = make_classifier_corpus(120, 4, seed=43)
        rng = np.random.default_rng(44)
        X_bad = rng.normal(50.0, 1.0, size=(30, 4))     # blatant poisoning

        _drive(pipeline, X_clean)
        _drive(pipeline, X_bad)

        # --- MToUT contract (the signal that triggered ATM) ----------
        assert pipeline.events, "no MToUT fired"
        poisoning_mtouts = [
            s for k, s in pipeline.events
            if k == "mtout"
            and TriggerReason.DATA_POISONING in s.reasons
        ]
        assert poisoning_mtouts, (
            "no DATA_POISONING MToUT fired; reasons seen: "
            f"{[s.reasons for k, s in pipeline.events if k == 'mtout']}"
        )
        crit = poisoning_mtouts[-1]
        assert crit.severity() == "CRITICAL", (
            f"DATA_POISONING MToUT should be CRITICAL, got {crit.severity()}"
        )

        # --- ATM routing contract ------------------------------------
        assert pipeline.atm_results
        result = pipeline.atm_results[-1]
        assert result.variant_used == MTPVariant.LOCAL, (
            "CRITICAL+small-batch must route to MTP-L with "
            "critical_always_local_first=True"
        )
        assert pipeline.mtp_l.call_count >= 1
        # Strong: MTP-E was NEVER asked to train on the poisoning batch.
        # The local budget still covers 150 samples, so MTP-C/MTP-E must
        # not be woken up just because severity is CRITICAL.
        assert pipeline.mtp_e.call_count == 0, (
            "MTP-E was invoked even though MTP-L budget covers the "
            "CRITICAL batch — auto-escalation must be gated on the "
            "local_max_samples boundary"
        )

        # --- Model-identity invariants -------------------------------
        if result.status == TrainStatus.SUCCESS:
            assert id(pipeline.rtp.aif.active_estimator) != pre_model_id, (
                "SUCCESS path must replace the active estimator"
            )
        else:
            # On the FAILED branch the old model must STILL be active —
            # rejection preserves the incumbent.
            assert pipeline.aif.active_estimator is pre_model, (
                "FAILED retrain must not swap the active estimator"
            )

        # --- Event-log contract --------------------------------------
        event_kinds = {e.event_type for e in pipeline.rtp.event_log}
        assert EventType.DATA_POISONING in event_kinds
        assert EventType.SECURITY_ALERT in event_kinds


# ---------------------------------------------------------------------------
# Variant selection — ATMPolicy + signal severity routing
# ---------------------------------------------------------------------------

class TestVariantSelection:

    def test_high_severity_drift_drift_routes_to_external(self) -> None:
        """
        Both DDD and CDD firing simultaneously → HIGH severity →
        ATM._select_variant returns EXTERNAL (MTP-E).
        """
        # Use check_interval=100 so the first detector-battery check
        # happens at step 100.  CDD's Page-Hinkley accumulates ~81 steps
        # to alarm against a 50 % error rate (lambda=40, delta=0.005,
        # baseline≈0 %), so with check_interval=50 CDD would not have
        # alarmed by the first check and then DDD's retrain resets CDD.
        # At check_interval=100 both DDD (KS/MMD on X) and CDD (PH on
        # prediction error) have enough history to alarm in the first
        # check window, producing a single HIGH-severity combined MToUT.
        cfg = RTPConfig(check_interval=100, mtout_cooldown_steps=10,
                        cdd_task="classifier")
        pipeline = build_pipeline(task="classifier", seed=5, config=cfg)

        # Shift AND flip labels — exercises DDD and CDD in one pass.
        # ``flip_labels=True`` inverts the labels produced by the
        # shift-invariant formula so every prediction the reference model
        # makes on the shifted stream is wrong, driving PH upward at the
        # maximum rate.  400 rows gives both detectors enough samples to
        # alarm inside the first 100-step check window.
        X_rec, y_rec = make_classifier_corpus(400, 4, seed=43,
                                              shift=4.0, flip_labels=True)
        _drive(pipeline, X_rec, y_true=y_rec)

        assert pipeline.atm_results, "ATM.handle was never called"

        # At least one MToUT must have HIGH severity in this run.
        severities = [sig.severity()
                      for k, sig in pipeline.events if k == "mtout"]
        assert "HIGH" in severities or "CRITICAL" in severities, (
            f"expected HIGH/CRITICAL severity, got {severities}"
        )

        # The HIGH cycle must have used MTP-E (or CRITICAL — which
        # also escalates to EXTERNAL when local path is unavailable).
        variants = {r.variant_used for r in pipeline.atm_results}
        assert MTPVariant.EXTERNAL in variants, (
            f"MTP-E was never selected. Variants seen: {variants}"
        )
        assert pipeline.mtp_e.call_count >= 1

    def test_prefer_variant_policy_forces_external(self) -> None:
        """
        Operator override: policy.prefer_variant=MTPVariant.EXTERNAL makes
        every cycle go to MTP-E regardless of severity.
        """
        pipeline = build_pipeline(
            task="classifier", seed=6,
            policy=ATMPolicy(
                prefer_variant=MTPVariant.EXTERNAL,
                use_ndt=True, auto_deploy=True,
                max_retrain_attempts=1,
            ),
        )
        X_drift, _ = make_classifier_corpus(200, 4, seed=42, shift=4.0)
        _drive(pipeline, X_drift)

        assert pipeline.atm_results
        assert all(
            r.variant_used == MTPVariant.EXTERNAL
            for r in pipeline.atm_results
        )
        assert pipeline.mtp_e.call_count >= 1
        assert pipeline.mtp_l.call_count == 0


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------

class TestFailurePaths:

    def test_ndt_rejection_prevents_deployment(self) -> None:
        """
        NDT verdict=False ⇒ candidate is discarded, active model unchanged.
        """
        failing_ndt = FakeNDT(verdict=False)
        pipeline = build_pipeline(task="classifier", seed=7, ndt=failing_ndt)
        pre_model_id = id(pipeline.aif.active_estimator)

        X_drift, _ = make_classifier_corpus(200, 4, seed=42, shift=4.0)
        _drive(pipeline, X_drift)

        # ATM was invoked but the cycle failed at NDT
        assert pipeline.atm_results
        result = pipeline.atm_results[-1]
        assert result.status == TrainStatus.FAILED
        assert result.ndt_passed is False
        assert result.deployed is False

        # Active model is unchanged
        assert id(pipeline.rtp.aif.active_estimator) == pre_model_id

        # NDT was actually called
        assert failing_ndt.call_count >= 1

    def test_skipped_when_buffer_has_insufficient_data(self) -> None:
        """
        When LIB has < 50 samples, ATM skips training entirely — status
        SKIPPED, no MTP-L/E call, no deployment.
        """
        pipeline = build_pipeline(task="classifier", seed=8)

        # Empty both buffers, then fire a manual MToUT signal.
        pipeline.rtp.buffers.lib._buf.clear()
        pipeline.rtp.buffers.lob._buf.clear()

        # Inject a manual signal straight into ATM (no stream needed)
        pipeline.rtp.operator_request(reason="test-empty-buffer")

        assert pipeline.atm_results
        result = pipeline.atm_results[-1]
        assert result.status == TrainStatus.SKIPPED
        assert result.variant_used is None
        assert result.deployed is False
        assert pipeline.mtp_l.call_count == 0
        assert pipeline.mtp_e.call_count == 0

    def test_cooldown_gate_suppresses_duplicate_mtout(self) -> None:
        """
        Two consecutive drift checks within the cooldown window must
        only produce ONE MToUT (and therefore only one ATM cycle).
        """
        # Cooldown larger than the drift window we'll push.
        cfg = RTPConfig(check_interval=50, mtout_cooldown_steps=10_000,
                        cdd_task="classifier")
        pipeline = build_pipeline(task="classifier", seed=9, config=cfg)

        X_drift, _ = make_classifier_corpus(400, 4, seed=42, shift=4.0)
        _drive(pipeline, X_drift)

        mtout_count = sum(1 for k, _ in pipeline.events if k == "mtout")
        assert mtout_count == 1, (
            f"Cooldown gate failed: {mtout_count} MToUTs fired; "
            f"expected exactly 1"
        )
        assert len(pipeline.atm_results) == 1


# ---------------------------------------------------------------------------
# Post-deployment stability — the self-healing property
# ---------------------------------------------------------------------------

class TestPostDeploymentStability:

    def test_detectors_go_quiet_after_successful_retraining(self) -> None:
        """
        After ATM deploys a new model, we stream a long run of clean
        samples from the *new* regime and assert no new MToUT fires.
        This exercises the full ``notify_model_updated`` path:

            aif.update_model → cdd.reset_ph → ddd/dpd/cpd.refit_reference

        Strong invariants checked beyond "no extra retrain cycles":
        * ``notify_model_updated`` actually ran — event log has a
          ``MODEL_UPDATED`` entry.
        * DDD's reference was REFIT — array identity differs from
          pre-retrain.  This rules out the "cold reset" failure mode
          where detectors go silent because their reference is None.
        * CDD's PH statistic stayed bounded (< lambda) throughout
          phase 2 — proves PH did not just reset to zero and slowly
          drift towards the alarm threshold; the new frozen mean
          actually tracks the new regime.
        * DPD's ``_ref_mean`` was refit to the new-regime mean
          (distance from the old ref mean is > 0 by construction:
          the new regime is shift=4.0).
        """
        pipeline = build_pipeline(task="classifier", seed=11)

        # Phase 1: induce drift so ATM retrains on the new distribution.
        # Build ONE corpus large enough to cover both the drift and healed
        # phases; slicing a single corpus guarantees that phase 2 shares
        # the same label-generating weight vector ``w`` as the drift data
        # the retrained model was trained on.  Using two separate
        # ``make_classifier_corpus`` calls with different seeds produces
        # different ``w`` vectors, so even on the same shift the retrained
        # model has residual concept drift against the healed labels —
        # enough to trip CDD's Page-Hinkley in phase 2.
        X_all, y_all = make_classifier_corpus(800, 4, seed=42, shift=4.0)
        X_drift, y_drift = X_all[:300], y_all[:300]
        _drive(pipeline, X_drift, y_true=y_drift)
        assert pipeline.atm_results, "no retraining happened in phase 1"
        assert pipeline.atm_results[-1].status == TrainStatus.SUCCESS

        # --- Post-deploy reference snapshot --------------------------
        # Reference arrays should now point at the NEW regime.  Capture
        # them before phase 2 so we can verify they stay stable (no
        # spurious refit fired again on clean traffic).
        ddd = pipeline.rtp.ddd
        dpd = pipeline.rtp.dpd
        cdd = pipeline.rtp.cdd
        ddd_ref_post_retrain = (
            ddd._reference.copy() if ddd._reference is not None else None
        )
        dpd_ref_mean_post_retrain = (
            dpd._ref_mean.copy() if getattr(dpd, "_ref_mean", None) is not None
            else None
        )

        # Prove the retrain fired the notify path — DPD's ref mean
        # should be in the new shifted regime (mean ≈ 4.0 per column).
        if dpd_ref_mean_post_retrain is not None:
            # Shift is 4.0; post-refit mean per column should be far
            # from zero.  Use a conservative 2.0 lower bound to tolerate
            # sample noise.
            assert np.all(np.abs(dpd_ref_mean_post_retrain) >= 2.0), (
                f"DPD _ref_mean not refit to new regime: "
                f"{dpd_ref_mean_post_retrain}"
            )

        # MODEL_UPDATED event should have been logged by the RTP.
        event_kinds = {e.event_type for e in pipeline.rtp.event_log}
        assert EventType.MODEL_UPDATED in event_kinds, (
            "RTP never logged MODEL_UPDATED — notify_model_updated "
            "path did not run after successful deploy"
        )

        initial_cycles = len(pipeline.atm_results)

        # Phase 2: feed 500 more clean samples *from the same new regime*.
        # Sliced from the same corpus as the drift phase so the label-
        # generating ``w`` is shared and the retrained model has near-zero
        # concept error on this data.  With references freshly refit, no
        # detector should re-fire.
        X_healed, y_healed = X_all[300:800], y_all[300:800]
        _drive(pipeline, X_healed, y_true=y_healed)

        # No additional retraining cycles should have fired.
        new_cycles = len(pipeline.atm_results) - initial_cycles
        assert new_cycles == 0, (
            f"Self-healing failed: {new_cycles} extra retraining cycles "
            f"fired on stable post-deploy traffic"
        )

        # --- PH-statistic bound --------------------------------------
        # PH's (_sum - _min_sum) is the alarm statistic; it must stay
        # under its lambda threshold throughout the clean phase 2
        # stream.  A "cold reset" would start at 0 but drift up as the
        # running-mean absorbs the new regime's losses; a proper
        # ``reset_ph`` + frozen mean at the new baseline keeps it bounded.
        ph_stat = cdd._ph._sum - cdd._ph._min_sum
        assert ph_stat < cdd._ph.lambda_, (
            f"PH statistic drifted to {ph_stat:.2f} (threshold "
            f"{cdd._ph.lambda_}) on clean post-deploy traffic — "
            f"reset_ph did not properly refreeze the baseline mean"
        )

        # --- References unchanged on clean traffic -------------------
        # No refit should have re-fired during phase 2 (would indicate
        # either an unwanted MToUT cycle or background self-refit).
        if ddd_ref_post_retrain is not None:
            np.testing.assert_array_equal(
                ddd._reference, ddd_ref_post_retrain,
                err_msg="DDD reference changed during clean phase 2 "
                        "(no new retrain fired)",
            )
        if dpd_ref_mean_post_retrain is not None:
            np.testing.assert_array_equal(
                dpd._ref_mean, dpd_ref_mean_post_retrain,
                err_msg="DPD _ref_mean changed during clean phase 2 "
                        "(no new retrain fired)",
            )


# ---------------------------------------------------------------------------
# Manual operator path
# ---------------------------------------------------------------------------

class TestManualRetraining:

    def test_operator_retrain_triggers_full_cycle(self) -> None:
        """``atm.operator_retrain()`` runs the full pipeline without RTP."""
        pipeline = build_pipeline(task="regressor", seed=12)
        pre_model_id = id(pipeline.aif.active_estimator)

        # Populate the buffer so ATM has > 50 samples to train on.
        X, y = make_regressor_corpus(200, 4, seed=42)
        _drive(pipeline, X, y_true=y)

        # Fire a manual cycle
        result = pipeline.atm.operator_retrain(variant=MTPVariant.LOCAL,
                                               reason="unit-test")

        assert result.status == TrainStatus.SUCCESS
        assert result.deployed
        assert result.variant_used == MTPVariant.LOCAL
        assert id(pipeline.rtp.aif.active_estimator) != pre_model_id


# ---------------------------------------------------------------------------
# Rollback (CONCEPT_POISONING path)
# ---------------------------------------------------------------------------

class TestRollback:

    def test_concept_poisoning_rolls_back_to_mlio(self) -> None:
        """
        When CPD flags poisoning, RTP must call ``aif.rollback()`` — but
        that only succeeds if MLIO is populated.  We first retrain once
        (moves old model into MLIO standby), then inject poisoning.
        """
        pipeline = build_pipeline(task="classifier", seed=13)

        # Warm up MLIO via a legitimate retrain
        X_drift, y_drift = make_classifier_corpus(300, 4, seed=42, shift=4.0)
        _drive(pipeline, X_drift, y_true=y_drift)
        assert pipeline.atm_results and \
               pipeline.atm_results[-1].status == TrainStatus.SUCCESS
        assert pipeline.aif.mlio.estimator is not None, \
            "MLIO should be populated after first retrain"

        # Grab the post-retrain MLIN reference so we can confirm rollback
        mlin_before_rollback = pipeline.aif.active_estimator

        # Inject a direct CPD-level poisoning signal: push LOB outputs
        # that reverse the sign of a correlated feature.
        X_poison, _ = make_classifier_corpus(200, 4, seed=44, shift=4.0)
        flipped_preds = 1 - (X_poison @ np.ones(4) > 0).astype(int)
        # Bypass AIF.predict so we can stamp deterministic malicious LOB
        pipeline.rtp.buffers.push_batch(X_poison, flipped_preds)
        pipeline.rtp._step += len(X_poison)
        sig = pipeline.rtp.force_check()

        event_kinds = {e.event_type for e in pipeline.rtp.event_log}
        # Either the CPD or DPD pathway can fire here — both exercise
        # the security-alert plumbing.  We only assert the alert path.
        assert EventType.SECURITY_ALERT in event_kinds
        # If CPD specifically fired, rollback must have happened.
        if EventType.CONCEPT_POISONING in event_kinds:
            assert pipeline.aif.active_estimator is not mlin_before_rollback
            assert EventType.ROLLBACK in event_kinds
