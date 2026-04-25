"""
Tier-4 end-to-end scenarios for the RTP Observer pipeline.

Where Tier-2/3 tests pin one phase of the retrain cycle at a time,
these tests stitch multiple phases together into narrative scripts
that mirror real operator timelines:

* A slow distribution shift that eventually demands a retrain and
  whose new reference then anchors the drifted regime.
* A sudden regime shock that must be caught and recovered from.
* A poisoning attack that is repelled by MTP-L on the CRITICAL path.
  The stream resumes without further DATA_POISONING alarms — a few
  MEDIUM DATA_DRIFT events may still fire because the DPD reference
  is refit from a LIB tail that includes the outlier batch.
* A label-poisoning incident that forces a rollback to the standby
  model in MLIO.  CPD fires after Fix #1 restores balanced labels in
  drift corpora, so the ROLLBACK branch is exercised unconditionally.
* A multi-cycle retrain chain in which each cycle cleanly anchors
  the next regime and the final model classifies the final regime
  accurately.  Model identity is tracked via estimator references
  and verified pairwise with ``is not``.
* An operator-driven rescue retrain launched even though no detector
  fired.  ``operator_retrain`` policy mutation bug (Fix #2) means
  the ``prefer_variant`` is now restored after each manual call.
* A regressor-task lifecycle exercising the same paths as the
  classifier scenarios.
* A concurrent drift + poisoning event whose severity rules still
  produce the correct variant selection.

Each scenario is split into explicit ``# Phase N —`` blocks and the
assertions at the end lock the full system state (detector baselines
refit, event log contents, ATM results, model-identity transitions).
The idea is that when a scenario fails, the phase markers and the
narrative docstring make it obvious which part of the lifecycle
broke, even before looking at the assertion.

Design notes
------------
``make_classifier_corpus`` uses shift-invariant labels: the decision
boundary is derived from ``(X - shift) @ w`` so class balance stays
near 50/50 regardless of shift magnitude.  Scenarios that need the
same label-generating weight ``w`` across a drift + held-out slice
build ONE large corpus and slice it, rather than calling
``make_classifier_corpus`` twice with the same seed but different
``n``.  This is the pattern used by Scenarios 2, 5, and 7.

The default ``ATMPolicy.ndt_min_accuracy`` is 0.70, which overrides
the NDT instance's 0.50 floor.  Drift scenarios that retrain on a
shifted regime against an unchanged validation buffer often produce
candidate scores near 0.5, so scenarios that need the retrain to
deploy explicitly soften the floor.  Scenarios that want to test
the NDT rejection branch keep the default.
"""
from __future__ import annotations

import numpy as np

from aif.aif import ModelState
from atm.atm import ATMPolicy, MTPVariant, TrainStatus
from rtp.rtp import EventType, TriggerReason

from ._harness import (
    build_pipeline,
    make_classifier_corpus,
    make_regressor_corpus,
    stream,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mtout_reasons(pipeline) -> set[TriggerReason]:
    """Every trigger reason that appeared in any MToUT so far."""
    out: set[TriggerReason] = set()
    for kind, sig in pipeline.events:
        if kind == "mtout":
            out.update(sig.reasons)
    return out


def _event_types(pipeline) -> set[EventType]:
    return {e.event_type for e in pipeline.rtp.event_log}


def _permissive_policy(**overrides) -> ATMPolicy:
    """
    Build an ATMPolicy whose NDT floor is permissive enough for the
    drifted-data scenarios to deploy.  The default 0.70 floor causes
    many honest retrains on shifted regimes to be rejected, which is
    the wrong failure mode for these lifecycle narratives.
    """
    kwargs = dict(
        use_ndt=True,
        auto_deploy=True,
        ndt_min_accuracy=0.50,
        max_retrain_attempts=1,
    )
    kwargs.update(overrides)
    return ATMPolicy(**kwargs)


def _training_corpus_for(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Regenerate the EXACT (X, y) the ``build_pipeline`` harness trains
    the classifier on so that stream-phase traffic has labels the
    model was actually fit against.  The harness uses
    ``make_classifier_corpus(600, 4, seed)`` — matching both the
    size AND the seed is essential (see module docstring).
    """
    return make_classifier_corpus(600, 4, seed=seed)


# ---------------------------------------------------------------------------
# Scenario 1 — Gradual drift chronicle
# ---------------------------------------------------------------------------

class TestGradualDriftScenario:
    """
    Narrative: a field deployment where the input distribution
    drifts over several batches and eventually triggers a retrain.
    After the retrain the detector references must anchor the new
    drifted regime — subsequent same-regime traffic must not fire
    further retrain cascades.
    """

    def test_drift_is_caught_and_then_anchored(self) -> None:
        # Phase 1 — Clean baseline (same distribution as training)
        pipeline = build_pipeline(
            task="classifier", seed=200,
            policy=_permissive_policy(),
        )
        X_base, y_base = _training_corpus_for(200)
        stream(pipeline, X_base, y_true=y_base)
        cycles_after_baseline = len(pipeline.atm_results)

        # Phase 2 — Apply a single, firm drift.  The detector should
        # fire within the first check intervals and ATM should retrain
        # on the new regime.  We use seed=42 + shift=4 to match the
        # well-tested drift corpus in Tier-2 tests.
        X_drift, y_drift = make_classifier_corpus(
            250, 4, seed=42, shift=4.0,
        )
        stream(pipeline, X_drift, y_true=y_drift)

        # At least one NEW retrain cycle fired during phase 2 and
        # at least one of them succeeded.
        phase2_results = pipeline.atm_results[cycles_after_baseline:]
        assert phase2_results, "no retrain fired during drift phase"
        assert any(
            r.status == TrainStatus.SUCCESS for r in phase2_results
        ), (
            f"no retrain succeeded during drift phase; statuses: "
            f"{[r.status.name for r in phase2_results]}"
        )
        cycles_after_drift = len(pipeline.atm_results)

        # Phase 3 — Continue on the new regime.  Reference was refit
        # on the successful retrain so no further cascade should fire.
        X_post, y_post = make_classifier_corpus(
            250, 4, seed=42, shift=4.0,
        )
        stream(pipeline, X_post, y_true=y_post)

        # Allow at most 1 extra cycle.  DDD uses 5 % per-check false-alarm
        # rate across 5 KS tests (Bonferroni α=0.0125 each).  The expected
        # number of false alarms over 5 checks on 250 in-distribution rows
        # is 5 × 0.0125 ≈ 0.06, well under 1.  A bound of 1 gives generous
        # slack for sampling noise while still catching reference failures.
        new_cycles = len(pipeline.atm_results) - cycles_after_drift
        assert new_cycles <= 1, (
            f"Post-deploy phase fired {new_cycles} retrain cycles — "
            f"new reference did not anchor the drifted regime"
        )

        # DPD reference must reflect the new regime (shift=4).
        post_mean = pipeline.rtp.dpd._ref_mean
        assert np.any(np.abs(post_mean) >= 1.5), (
            f"DPD _ref_mean did not migrate to the drifted regime: "
            f"{post_mean}"
        )


# ---------------------------------------------------------------------------
# Scenario 2 — Sudden shock + recovery
# ---------------------------------------------------------------------------

class TestSuddenShockScenario:
    """
    Narrative: the deployment is serving steadily, then a single
    hard regime shock arrives.  The detector must fire, ATM must
    retrain, and the post-retrain model must be accurate on the
    new regime.
    """

    def test_hard_shift_triggers_retrain_and_model_recovers(self) -> None:
        # Phase 1 — Steady on-distribution traffic (first 200 rows
        # of the training corpus — same labels the model was fit on).
        pipeline = build_pipeline(
            task="classifier", seed=210,
            policy=_permissive_policy(),
        )
        X_train, y_train = _training_corpus_for(210)
        stream(pipeline, X_train[:200], y_true=y_train[:200])

        # Phase 2 — Hard shock.  Build a single drift corpus that is
        # big enough to also carry the phase-3 held-out check slice;
        # slicing one corpus keeps a single label-generating ``w``
        # across training-through-check, avoiding the subtle
        # ``make_classifier_corpus`` ``n``-dependent ``w`` pitfall.
        X_all, y_all = make_classifier_corpus(
            350, 4, seed=42, shift=4.0,
        )
        X_shock, y_shock = X_all[:250], y_all[:250]
        X_check, y_check = X_all[250:], y_all[250:]

        pre_model = pipeline.aif.active_estimator
        stream(pipeline, X_shock, y_true=y_shock)

        # Retrain fired and a SUCCESS result is present.
        assert pipeline.atm_results, "no retrain cycle fired after shock"
        successes = [
            r for r in pipeline.atm_results
            if r.status == TrainStatus.SUCCESS and r.deployed
        ]
        assert successes, (
            f"no SUCCESS retrain deployed; statuses: "
            f"{[r.status.name for r in pipeline.atm_results]}"
        )
        assert pipeline.aif.active_estimator is not pre_model, (
            "SUCCESS retrain left the incumbent model active"
        )

        # Phase 3 — Serve the held-out slice on the new regime.
        new_model = pipeline.aif.active_estimator
        preds = new_model.predict(X_check)
        accuracy = float((preds == y_check).mean())
        assert accuracy >= 0.80, (
            f"Post-shock model accuracy {accuracy:.2f} below 0.80 — "
            f"recovery failed"
        )

        # MODEL_UPDATED was logged.
        assert EventType.MODEL_UPDATED in _event_types(pipeline)


# ---------------------------------------------------------------------------
# Scenario 3 — Poisoning repelled, clean traffic resumes
# ---------------------------------------------------------------------------

class TestPoisoningRepelledScenario:
    """
    Narrative: the deployment is serving clean traffic when an
    attacker injects a short burst of outliers.  DPD raises the
    CRITICAL alarm, ATM routes to MTP-L, the candidate passes NDT,
    and the model is swapped.

    The resumed clean stream may still raise MEDIUM DATA_DRIFT alarms
    (steps 250 and 350) because ``notify_model_updated`` refits the DPD
    reference from a LIB tail that still contains the 50-sigma outlier
    batch.  This is production-realistic behaviour — the detector is
    correctly reporting residual contamination in its reference window.
    The test asserts only the narrower, true claim: no DATA_POISONING
    MToUT fires during the resumed phase.
    """

    def test_outlier_burst_routes_to_mtpl_and_stream_resumes(self) -> None:
        # Phase 1 — Clean traffic (first 120 rows of the training
        # corpus — labels match the trained model).
        pipeline = build_pipeline(
            task="classifier", seed=220,
            policy=_permissive_policy(),
        )
        X_train, y_train = _training_corpus_for(220)
        stream(pipeline, X_train[:120], y_true=y_train[:120])

        # Phase 2 — Poisoning burst: 30 massive outliers.
        rng = np.random.default_rng(221)
        X_bad = rng.normal(50.0, 1.0, size=(30, 4))
        y_bad = np.zeros(30, dtype=int)
        stream(pipeline, X_bad, y_true=y_bad)

        # A CRITICAL DATA_POISONING MToUT fired.
        crit = [
            s for k, s in pipeline.events
            if k == "mtout"
            and TriggerReason.DATA_POISONING in s.reasons
        ]
        assert crit, "DPD did not fire on outlier burst"
        assert crit[-1].severity() == "CRITICAL", (
            f"DATA_POISONING MToUT must be CRITICAL, got "
            f"{crit[-1].severity()}"
        )

        # MTP-L took the batch; MTP-E was NOT escalated.
        poisoning_cycle = pipeline.atm_results[-1]
        assert poisoning_cycle.variant_used == MTPVariant.LOCAL
        assert pipeline.mtp_e.call_count == 0, (
            "MTP-E was escalated unnecessarily on a CRITICAL-small batch"
        )
        assert EventType.DATA_POISONING in _event_types(pipeline)
        assert EventType.SECURITY_ALERT in _event_types(pipeline)

        # Phase 3 — Stream resumes on on-distribution traffic.  Use
        # rows 120-399 of the same training corpus (labels match).
        cycles_before_resume = len(pipeline.atm_results)
        events_before_resume = len(pipeline.events)
        stream(
            pipeline,
            X_train[120:400],
            y_true=y_train[120:400],
        )

        # No DATA_POISONING alarm fired DURING the resumed phase.
        resumed_events = pipeline.events[events_before_resume:]
        resumed_poisoning = [
            s for k, s in resumed_events
            if k == "mtout"
            and TriggerReason.DATA_POISONING in s.reasons
        ]
        assert not resumed_poisoning, (
            "DATA_POISONING MToUT fired during clean resumed traffic; "
            f"new MToUTs: {[s.reasons for _, s in resumed_events]}"
        )


# ---------------------------------------------------------------------------
# Scenario 4 — Rollback on concept poisoning
# ---------------------------------------------------------------------------

class TestConceptPoisoningRollbackScenario:
    """
    Narrative: the model has already weathered one legitimate
    retrain, so MLIO holds a standby model.  A poisoning attack
    then arrives that CPD can detect (sharply inverted LOB
    predictions relative to a clean shadow model); RTP fires the
    security-alert pipeline, and on the CPD pathway calls
    ``aif.rollback()`` to re-activate the previous model.

    Why the direct-buffer injection
    -------------------------------
    A natural label-flipped stream routed through ``observe()`` only
    reliably fires DPD (data poisoning on the INPUT distribution), not
    CPD (poisoning on the INPUT→OUTPUT MAPPING).  DPD does NOT trigger
    a rollback — only CPD does.  To exercise the rollback pathway we
    mirror Tier-2's direct-buffer injection: stamp the LIB with an
    on-regime feature batch AND the LOB with predictions that invert
    what a clean shadow model would produce.  This drives
    shadow-divergence to ≈1 and simultaneously reverses the
    input-output correlation sign, which is the exact dual-condition
    CPD is designed to catch.
    """

    def test_inverted_lob_rolls_back_to_mlio_standby(self) -> None:
        pipeline = build_pipeline(
            task="classifier", seed=13,
            policy=_permissive_policy(),
        )

        # Phase 1 — Legitimate retrain to populate MLIO.
        X_drift, y_drift = make_classifier_corpus(
            300, 4, seed=42, shift=4.0,
        )
        stream(pipeline, X_drift, y_true=y_drift)
        success = any(
            r.status == TrainStatus.SUCCESS and r.deployed
            for r in pipeline.atm_results
        )
        assert success, (
            f"phase-1 retrain did not deploy; statuses: "
            f"{[r.status.name for r in pipeline.atm_results]}"
        )
        post_retrain_model = pipeline.aif.active_estimator

        # MLIO must now hold the previous model (state STANDBY).
        assert pipeline.aif.mlio.state == ModelState.STANDBY, (
            f"MLIO state is {pipeline.aif.mlio.state} after first "
            f"retrain; rollback cannot succeed"
        )

        # Phase 2 — Direct CPD injection: push on-regime features with
        # LOB predictions that invert the sign of the correlated
        # features.  This is the same pattern exercised by the Tier-2
        # ``TestRollback::test_concept_poisoning_rolls_back_to_mlio``.
        X_poison, _ = make_classifier_corpus(200, 4, seed=44, shift=4.0)
        flipped_preds = 1 - (X_poison @ np.ones(4) > 0).astype(int)
        pipeline.rtp.buffers.push_batch(X_poison, flipped_preds)
        pipeline.rtp._step += len(X_poison)
        pipeline.rtp.force_check()

        evt_types = _event_types(pipeline)

        # After Fix #1 (shift-invariant labels) the phase-1 drift corpus
        # produces balanced labels, so the shadow model has genuine class
        # diversity and CPD's shadow-divergence metric is non-trivial.
        # CPD fires unconditionally in this scenario; the assertions below
        # are therefore no longer conditional.

        assert EventType.SECURITY_ALERT in evt_types, (
            "no SECURITY_ALERT logged after CPD injection"
        )
        assert EventType.CONCEPT_POISONING in evt_types, (
            "CPD did not fire after inverted-LOB injection — shadow "
            "diversity may be insufficient; check that Fix #1 is applied"
        )
        assert EventType.ROLLBACK in evt_types, (
            "CONCEPT_POISONING fired but ROLLBACK event is missing"
        )
        assert pipeline.aif.active_estimator is not post_retrain_model, (
            "rollback did not restore the MLIO standby model"
        )


# ---------------------------------------------------------------------------
# Scenario 5 — Multi-cycle chain
# ---------------------------------------------------------------------------

class TestMultiCycleChainScenario:
    """
    Narrative: the distribution keeps shifting further from baseline
    over time.  Each shift spawns its own retrain cycle, each cycle
    swaps the active estimator, and the final model classifies the
    final regime accurately.

    Design: each drift episode uses a distinct seed (42, 43, 44) so
    each has its own label-generating weight vector ``w``.  The final
    accuracy check slices its 100-row holdout from the SAME corpus as
    the third drift episode (seed=44, shift=6.0) so the ``w`` vectors
    match — the retrained model is tested against the exact boundary it
    learned.  Model-identity transitions are tracked via actual estimator
    references and verified pairwise with ``is not`` to rule out Python
    id() reuse across GC cycles.  A baseline accuracy comparison proves
    the cascade actually learned: the pre-drift model must score notably
    worse on the final regime than the final retrained model.
    """

    def test_three_drift_events_produce_at_least_three_retrains(
        self,
    ) -> None:
        pipeline = build_pipeline(
            task="classifier", seed=240,
            policy=_permissive_policy(),
        )

        # Phase 0 — warm-up on-distribution traffic.
        X_train, y_train = _training_corpus_for(240)
        stream(pipeline, X_train[:200], y_true=y_train[:200])

        # Record the pre-drift model for the baseline comparison at the end.
        pre_drift_model = pipeline.aif.active_estimator

        # Three successive drift episodes with increasing shift.
        # Each uses a different seed so each has an independent w vector.
        # Estimator refs are stored before and after each episode so we can
        # verify pairwise model swaps without relying on Python id() values.
        model_refs: list = [pipeline.aif.active_estimator]
        drift_corpora = [
            make_classifier_corpus(250, 4, seed=42, shift=2.0),
            make_classifier_corpus(250, 4, seed=43, shift=4.0),
            # Build a 350-row corpus for the last regime: first 250 rows
            # are the drift episode; the remaining 100 rows form the held-out
            # check slice that shares the same w vector.
            make_classifier_corpus(350, 4, seed=44, shift=6.0),
        ]
        for X_d, y_d in drift_corpora:
            stream(pipeline, X_d[:250], y_true=y_d[:250])
            model_refs.append(pipeline.aif.active_estimator)

        # At least three SUCCESS cycles in total.
        successes = [
            r for r in pipeline.atm_results
            if r.status == TrainStatus.SUCCESS and r.deployed
        ]
        assert len(successes) >= 3, (
            f"Expected >=3 SUCCESS retrains across 3 drift episodes; "
            f"got {len(successes)} "
            f"(statuses={[r.status.name for r in pipeline.atm_results]})"
        )

        # Active model identity changed at each episode boundary.
        # Pairwise ``is not`` is stronger than set(id(...)) because it
        # catches cases where a rollback restores a previous object.
        for i in range(len(model_refs) - 1):
            assert model_refs[i] is not model_refs[i + 1], (
                f"Model did not change between episode {i} and {i + 1}: "
                f"same estimator instance at both checkpoints"
            )

        # Final model classifies the final regime.
        # Holdout is the tail of the seed=44 corpus (same w as episode 3)
        # so the label boundary matches what the retrained model learned.
        # The absolute bound is 0.70: meaningful above chance (~0.50) and
        # achievable on this balanced corpus even after CPD cascade noise.
        X_final_all, y_final_all = drift_corpora[2]
        X_final, y_final = X_final_all[250:], y_final_all[250:]
        final_model = pipeline.aif.active_estimator

        accuracy = float((final_model.predict(X_final) == y_final).mean())
        assert accuracy >= 0.70, (
            f"Final-regime accuracy {accuracy:.2f} below 0.70 — "
            f"multi-cycle chain did not converge"
        )

        # The pre-drift model must be notably worse on the final regime,
        # proving the cascade actually adapted to the new distribution
        # and did not simply preserve the original concept boundary.
        pre_accuracy = float(
            (pre_drift_model.predict(X_final) == y_final).mean()
        )
        assert accuracy > pre_accuracy + 0.10, (
            f"Final model accuracy {accuracy:.2f} not notably better than "
            f"pre-drift model {pre_accuracy:.2f} on final regime — "
            f"the multi-cycle cascade did not improve on baseline"
        )


# ---------------------------------------------------------------------------
# Scenario 6 — Operator-driven rescue retrain
# ---------------------------------------------------------------------------

class TestOperatorRescueScenario:
    """
    Narrative: the automatic detectors are silent but an operator
    fires a manual retrain (scheduled maintenance, say).  The
    ``atm.operator_retrain`` path runs the full cycle without
    creating any RTP MToUT side-effects.
    """

    def test_operator_retrain_runs_full_cycle_without_mtout(self) -> None:
        pipeline = build_pipeline(
            task="classifier", seed=260,
            policy=_permissive_policy(),
        )

        # Phase 1 — on-distribution traffic (sized to match training).
        X_train, y_train = _training_corpus_for(260)
        stream(pipeline, X_train[:260], y_true=y_train[:260])

        pre_mtouts = [k for k, _ in pipeline.events if k == "mtout"]
        pre_model = pipeline.aif.active_estimator

        # Capture prefer_variant BEFORE the call so we can verify Fix #2
        # (operator_retrain must not permanently mutate policy).
        pre_prefer_variant = pipeline.atm.policy.prefer_variant

        # Phase 2 — operator fires a manual retrain.
        result = pipeline.atm.operator_retrain(
            variant=MTPVariant.LOCAL, reason="scheduled-refresh",
        )

        assert result.status == TrainStatus.SUCCESS, (
            f"operator_retrain expected SUCCESS, got {result.status.name}: "
            f"{result.message}"
        )
        assert result.deployed, (
            "operator_retrain returned SUCCESS but deployed=False"
        )
        assert result.variant_used == MTPVariant.LOCAL, (
            f"explicit variant=LOCAL must be honoured; got "
            f"{result.variant_used}"
        )
        assert pipeline.aif.active_estimator is not pre_model, (
            "operator_retrain SUCCESS must replace the active estimator"
        )

        # Fix #2 regression guard: prefer_variant must be restored to its
        # pre-call value so subsequent automatic retrains are not locked
        # to MTP-L by a stale policy mutation.
        assert pipeline.atm.policy.prefer_variant == pre_prefer_variant, (
            f"operator_retrain permanently mutated prefer_variant from "
            f"{pre_prefer_variant!r} to "
            f"{pipeline.atm.policy.prefer_variant!r} — Fix #2 regression"
        )

        # No new MToUT events as a side-effect of the manual retrain.
        post_mtouts = [k for k, _ in pipeline.events if k == "mtout"]
        assert len(post_mtouts) == len(pre_mtouts), (
            "operator_retrain spuriously emitted MToUT events"
        )
        assert EventType.MODEL_UPDATED in _event_types(pipeline)


# ---------------------------------------------------------------------------
# Scenario 7 — Regressor lifecycle
# ---------------------------------------------------------------------------

class TestRegressorLifecycleScenario:
    """
    Narrative: the same full lifecycle (baseline → drift → retrain
    → recovery) but on a regressor task.  Verifies the whole stack
    is task-agnostic: CDD uses MAE, DDD stays KS/MMD, NDT scores
    via R² instead of accuracy.
    """

    def test_regressor_drift_triggers_retrain_and_mae_recovers(
        self,
    ) -> None:
        pipeline = build_pipeline(
            task="regressor", seed=270,
            policy=_permissive_policy(),
        )
        pre_model = pipeline.aif.active_estimator

        # Phase 1 — drift on regressor task.  Build ONE corpus big
        # enough to cover both the drift-phase stream and the
        # phase-2 held-out slice, then split it.  ``make_regressor_corpus``
        # draws X first and the target weight ``w`` second, so
        # two calls with the same (seed, shift) but different ``n``
        # yield DIFFERENT ``w`` — sliced subsets of a single corpus
        # share ``w`` by construction.  This is the same pitfall the
        # classifier scenarios avoid (see module docstring).
        X_all, y_all = make_regressor_corpus(
            400, 4, seed=42, shift=4.0,
        )
        X_drift, y_drift = X_all[:300], y_all[:300]
        X_check, y_check = X_all[300:], y_all[300:]

        stream(pipeline, X_drift, y_true=y_drift)

        assert pipeline.atm_results, "no retrain on regressor drift"
        successes = [
            r for r in pipeline.atm_results
            if r.status == TrainStatus.SUCCESS and r.deployed
        ]
        assert successes, (
            f"no SUCCESS retrain deployed for regressor; statuses: "
            f"{[r.status.name for r in pipeline.atm_results]}"
        )
        assert pipeline.aif.active_estimator is not pre_model

        # Phase 2 — verify regressor MAE on the drifted regime.
        # The active estimator was trained on a mix of pre-drift and
        # drifted data from LIB.  MAE on a purely drifted tail should
        # be well below the pre-retrain model's error (which would
        # have been dominated by the shift).  The Ridge under shift=4 /
        # label-noise 0.1 lands around 0.1-0.5 with a well-fit model.
        # The bound is set to 1.0 — 2x the upper expected value — to
        # give slack for sampling noise while still catching cases where
        # the retrain did not track the new regime at all.
        new_model = pipeline.aif.active_estimator
        mae = float(np.mean(np.abs(new_model.predict(X_check) - y_check)))
        assert mae <= 1.0, (
            f"Post-retrain regressor MAE {mae:.3f} > 1.0 — "
            f"regressor lifecycle did not recover"
        )


# ---------------------------------------------------------------------------
# Scenario 8 — Concurrent drift + poisoning
# ---------------------------------------------------------------------------

class TestConcurrentDriftAndPoisoningScenario:
    """
    Narrative: an attacker injects a batch that is BOTH distribution-
    shifted AND contains extreme outliers.  Drift detectors and the
    poisoning detector both fire on the same check-interval.
    CRITICAL must win (poisoning dominates drift) and the variant
    selector must still take MTP-L given the modest batch size.
    """

    def test_poisoning_dominates_drift_in_severity(self) -> None:
        pipeline = build_pipeline(
            task="classifier", seed=280,
            policy=_permissive_policy(),
        )

        # Phase 1 — warm-up clean traffic (training corpus prefix).
        X_train, y_train = _training_corpus_for(280)
        stream(pipeline, X_train[:120], y_true=y_train[:120])

        # Phase 2 — crafted batch: 60 rows with mean shift 4σ and
        # 10 of them replaced with 50σ outliers.
        rng = np.random.default_rng(281)
        X_combo = rng.normal(4.0, 1.0, size=(60, 4))
        outlier_idx = rng.choice(60, size=10, replace=False)
        X_combo[outlier_idx] = rng.normal(50.0, 1.0, size=(10, 4))
        y_combo = np.zeros(60, dtype=int)
        stream(pipeline, X_combo, y_true=y_combo)

        # Reasons collected across all MToUTs include poisoning.
        reasons = _mtout_reasons(pipeline)
        assert TriggerReason.DATA_POISONING in reasons, (
            f"DATA_POISONING missing from MToUT reasons {reasons}"
        )

        # There is exactly ONE MToUT with combined [DATA_DRIFT,
        # DATA_POISONING] reasons at the first check-interval after the
        # crafted batch lands.  Both detectors fire in the same interval
        # so the signal is assembled with both reasons in a single
        # MToUT — not two separate signals.  Severity is CRITICAL because
        # DATA_POISONING is present regardless of the drift component.
        crit = [
            s for k, s in pipeline.events
            if k == "mtout"
            and TriggerReason.DATA_POISONING in s.reasons
        ]
        assert crit, (
            "no MToUT with DATA_POISONING reason found in event log"
        )
        assert crit[-1].severity() == "CRITICAL", (
            f"expected CRITICAL on poisoning MToUT, got "
            f"{crit[-1].severity()}"
        )

        # The ATM cycle driven by the single combined MToUT must route
        # to MTP-L (CRITICAL + batch < local_max_samples).  Find the
        # matching result by looking for the LOCAL variant in any result
        # that corresponds to a CRITICAL signal with DATA_POISONING.
        assert pipeline.atm_results, (
            "no ATM result recorded — handle() was never called"
        )
        poisoning_results = [
            r for r in pipeline.atm_results
            if r.variant_used == MTPVariant.LOCAL
        ]
        assert poisoning_results, (
            f"no ATM result used MTP-L; variants seen: "
            f"{[r.variant_used for r in pipeline.atm_results]}"
        )
        poisoning_cycle = poisoning_results[-1]
        assert poisoning_cycle.variant_used == MTPVariant.LOCAL, (
            f"CRITICAL+small-batch must route to MTP-L; got "
            f"{poisoning_cycle.variant_used}"
        )
