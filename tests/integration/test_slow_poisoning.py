"""
Tier-2 integration tests — DPD cumulative (slow-poisoning) arm.

Threat model under test
-----------------------
The per-batch DPD arm inspects each recent window in isolation: an
attacker who keeps every batch's drift *just below* the configured
per-batch thresholds (mahal_threshold=4σ soft, mahal_hard_threshold=8σ
hard, min_mahal_outliers=3 sparsity gate) evades detection indefinitely
while the cumulative distribution still shifts.  This file exercises
the fix: an EWMA of the per-batch ``mahal_max`` with its own threshold,
α=0.05 → ~20-batch memory (faster α=0.10 is used inside the RTP
pipeline test).  A sustained ``_TARGET_MAHAL=6.5`` Mahal_max per window
sits above the 4σ soft threshold (but only 2 < 3 min-hit injections so
the soft count doesn't fire) and below the 8σ hard threshold (with
>1σ safety margin for empirical covariance sampling noise), so the
per-batch arm stays quiet — yet the EWMA climbs to ~6.5, past the
test's 5.5 slow-threshold, and the slow arm fires.

Calibrated injection magnitude
------------------------------
The injection magnitude is computed dynamically from the DPD's fitted
covariance (see ``_compute_injection_scale``) so the Mahalanobis
distance of each injected row is EXACTLY ``_TARGET_MAHAL`` regardless
of the ~6 % noise on the empirical-covariance diagonal a 300-sample
fit produces.  This is essential: a raw "inject at 7.5σ" pattern
would occasionally trip the 8σ hard arm on unlucky cov fits and
confuse the test.

Scenarios
---------
* **Test A** (fix verification) — inject a sustained small drift
  below the per-batch threshold for 200 batches; assert per-batch
  stays False throughout while slow_poisoning_detected becomes True,
  and that the SLOW_POISONING_SUSPECTED event was emitted via RTP.
* **Test B** (no false positives) — 200 batches of clean data;
  assert both per-batch and slow_poisoning stay False end to end.
* **Test C** (no double-fire) — one big spike fires the per-batch
  arm on its own; the slow arm must NOT additionally fire on the
  same window (the spec forbids double-firing on a single batch).
* **Test D** (EWMA memory) — 100 batches of sustained drift, then
  100 clean batches; assert the EWMA decays back below threshold
  and slow_poisoning_detected clears.
* **Test E** (MToUT integration) — sustained slow-poisoning path
  produces a DATA_POISONING trigger reason with severity ≥ MEDIUM
  (not CRITICAL — slow-poisoning is cumulative, not an immediate
  rollback trigger) and does NOT itself initiate a rollback.

Implementation notes
--------------------
* Tests A-D drive the DPD detector directly against a crafted LIB,
  bypassing RTP.  That isolates the EWMA math from RTP's
  check_interval / buffer-pair plumbing which would otherwise couple
  three or four concurrent detector contracts into the same test.
* Test E uses the shared ``_harness`` to exercise the full
  RTP → MToUT → ATM path.  The slow-poisoning injection matches
  Test A's pattern via ``_compute_injection_scale`` against the
  actual pipeline's fitted DPD covariance.
"""
from __future__ import annotations

import numpy as np
import pytest

from aif.buffers import LIB
from detectors.dpd import DPD
from rtp.rtp import EventType, MToUTSignal, RTPConfig, TriggerReason

from ._harness import (
    Pipeline,
    build_pipeline,
    make_classifier_corpus,
)
from tests.unit._helpers import fill_lib


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Target Mahalanobis distance for the injection spike — sits strictly
# between the soft threshold (4σ) and the hard threshold (8σ), with a
# comfortable 1.4σ margin below 8 to absorb the ~6% empirical-covariance
# noise that 300-sample fits produce (sqrt of diagonal can swing ±8%).
# Only 2 samples per 50-sample window are injected so the soft-hit count
# stays at 2 < min_mahal_outliers=3 → soft arm does not fire.  The
# hard arm does not fire because no sample crosses 8σ.  But Mahal_max
# converges to ~_TARGET_MAHAL every batch → EWMA climbs past the test's
# configured slow-poisoning threshold.
_TARGET_MAHAL: float = 6.5
_INJECTIONS_PER_BATCH: int = 2
# Clean baseline std tighter than reference so natural outliers among
# the 48 non-injected samples don't reach 4σ (under ref fitted on N(0,1),
# a clean sample at N(0, 0.6·I) has Mahal distance ~0.6 * chi_4, with a
# max-of-48 near 2.2σ — comfortably below the 4σ soft threshold).
_CLEAN_STD: float = 0.6


def _make_dpd(slow_poisoning_threshold: float = 5.5) -> DPD:
    """
    Build a DPD with production-default per-batch thresholds (which are
    the ones the injection is calibrated against) but a test-specific
    ``slow_poisoning_threshold`` lower than the production default 7.0.

    Why lower the threshold: the injection magnitude is chosen with a
    safety margin below the per-batch hard threshold (8σ), so Mahal_max
    converges to ~6.5, not 7.5+.  A 5.5 threshold is high enough that
    clean data never trips it (clean EWMA stabilises near 2-3σ) yet low
    enough that the 6.5-target injection rises past it in well under
    100 batches.  The DPD fix under test does NOT depend on the exact
    threshold — it depends on the EWMA-crossing-threshold behaviour,
    which the test exercises faithfully with these tuned numbers.
    """
    return DPD(
        reference_size=300,
        recent_size=50,
        if_contamination=0.02,
        contamination_threshold=0.10,
        mahal_threshold=4.0,
        min_mahal_outliers=3,
        mahal_hard_threshold=8.0,
        mahal_hard_max_fraction=0.30,
        slow_poisoning_alpha=0.05,
        slow_poisoning_threshold=slow_poisoning_threshold,
    )


def _build_lib_with_reference(
    ref: np.ndarray, capacity: int = 2000,
) -> LIB:
    """Fresh LIB seeded with the reference rows (capacity large enough
    to accumulate 200 batches of live data without eviction)."""
    lib = LIB(maxlen=capacity)
    for row in ref:
        lib.push(row)
    return lib


def _compute_injection_scale(
    dpd: DPD, target_mahal: float, n_features: int = 4,
) -> float:
    """
    Compute the coordinate scale ``s`` so that an injection row
    ``(s, 0, ..., 0)`` under the currently-fitted reference has
    Mahalanobis distance exactly ``target_mahal``.

    The Mahalanobis distance of a point x relative to mean μ and
    covariance Σ is ``sqrt((x-μ)ᵀ Σ⁻¹ (x-μ))``.  For x = (s, 0, ...)
    and μ ≈ 0 (reference is centred), this reduces to
    ``|s| * sqrt(precision[0, 0])`` where ``precision = Σ⁻¹``.  Solving
    for s yields ``s = target_mahal / sqrt(precision[0, 0])``.  Using
    this scale makes the injection's Mahal distance INVARIANT to the
    empirical covariance fluctuations that a raw ``s = target_mahal``
    would expose the test to, so the per-batch hard threshold (8σ) is
    never accidentally tripped.
    """
    assert dpd._cov is not None, "DPD must be fitted before scaling injection"
    precision = dpd._cov.get_precision()
    # sqrt(precision[0, 0]) maps a unit increment along axis 0 to its
    # Mahalanobis contribution; inverting scales the target distance to
    # the raw-coordinate magnitude we need to inject.
    diag_precision_0 = float(precision[0, 0])
    if diag_precision_0 <= 0.0:   # pragma: no cover - numerical guard
        return target_mahal       # fall back gracefully; test will still run
    return target_mahal / np.sqrt(diag_precision_0)


def _push_batch(
    lib: LIB,
    rng: np.random.Generator,
    recent_size: int,
    *,
    inject_sigma: float = 0.0,
    n_injections: int = 0,
    clean_std: float = _CLEAN_STD,
    n_features: int = 4,
) -> None:
    """
    Push one ``recent_size`` batch into LIB.

    Parameters
    ----------
    inject_sigma : float
        If > 0, overwrite ``n_injections`` samples with a spike of
        magnitude ``(inject_sigma, 0, ..., 0)``.  Under a reference
        fitted on N(0, I), the Mahalanobis distance of such a spike
        is ≈ inject_sigma; for tighter control callers should pass
        ``inject_sigma`` already scaled by ``_compute_injection_scale``.
    n_injections : int
        Number of samples to inject.  2 by default so the soft-hit
        count stays below ``min_mahal_outliers=3`` → per-batch arm
        does not fire even when ``inject_sigma`` is past 4σ.
    clean_std : float
        Standard deviation of the N(0, clean_std·I) clean samples
        filling the rest of the window.  Default ``_CLEAN_STD=0.6``
        is tighter than the reference's 1.0 so natural outliers among
        the 48 clean samples stay well below the 4σ soft threshold.
    """
    batch = rng.normal(0.0, clean_std, size=(recent_size, n_features))
    if inject_sigma > 0.0 and n_injections > 0:
        for i in range(min(n_injections, recent_size)):
            batch[i] = 0.0
            batch[i, 0] = inject_sigma
    for row in batch:
        lib.push(row)


# ---------------------------------------------------------------------------
# Test A — fix verification
# ---------------------------------------------------------------------------

def test_a_slow_poisoning_fires_on_sustained_sub_threshold_drift() -> None:
    """
    Inject a sustained below-per-batch-threshold drift for 200 batches.

    Each batch has 2 samples at Mahal distance _TARGET_MAHAL=6.5 (above
    4σ soft but below the hard 8σ, and only 2 < min_mahal_outliers=3
    soft hits).  Per-batch DPD must therefore stay quiet for the entire
    run.  But the EWMA of ``mahal_max`` — constant near 6.5 — converges
    upward past the 5.5 slow-poisoning threshold after ~20-30 batches,
    so slow_poisoning fires well before the run ends.

    Also verifies the RTP integration: feeding the same injection
    pattern through a real RTP produces a SLOW_POISONING_SUSPECTED
    event in the event log, with the spec payload ``{step, ewma,
    threshold, n_batches_above}``.
    """
    rng = np.random.default_rng(9001)
    ref = rng.normal(0.0, 1.0, size=(300, 4))

    dpd = _make_dpd()
    lib = _build_lib_with_reference(ref)

    # Warm up: first check() fits the reference on the baseline window.
    # Push one clean recent window so the detector has 50 rows to
    # fit against in addition to the 300 reference rows.
    _push_batch(lib, rng, dpd.recent_size, n_features=4)
    dpd.check(lib)  # auto-fits reference — EWMA stays 0.0 at this pass

    # Now that the DPD has fitted the reference covariance, compute the
    # injection-coordinate scale so Mahal distance is deterministically
    # _TARGET_MAHAL regardless of empirical sampling noise in the fit.
    inject_scale = _compute_injection_scale(dpd, _TARGET_MAHAL)

    per_batch_fired: list[int] = []
    slow_fired_at: int | None = None
    ewma_trace: list[float] = []

    for i in range(200):
        _push_batch(
            lib, rng, dpd.recent_size,
            inject_sigma=inject_scale,
            n_injections=_INJECTIONS_PER_BATCH,
        )
        result = dpd.check(lib)
        ewma_trace.append(result.mahal_ewma)

        if result.poisoning_detected:
            per_batch_fired.append(i)
        if result.slow_poisoning_detected and slow_fired_at is None:
            slow_fired_at = i

    # ── Primary contract: per-batch arm stays quiet, slow arm fires ──
    assert not per_batch_fired, (
        f"per-batch DPD fired at batch(es) {per_batch_fired} on the "
        f"sub-threshold slow-poisoning run — the attacker is supposed "
        f"to be able to keep per-batch quiet. Final mahal_max trace tail "
        f"{ewma_trace[-5:]!r}. Lower _TARGET_MAHAL or raise the clean_std."
    )
    assert slow_fired_at is not None, (
        f"slow-poisoning never fired in 200 batches. "
        f"Final EWMA={ewma_trace[-1]:.3f} (threshold=5.5). "
        f"Either α is too small or the threshold is too high."
    )
    # Must fire within a reasonable window — closed-form estimate with
    # α=0.05, Mahal=6.5, threshold=5.5 gives ~36 batches.  100 is generous.
    assert slow_fired_at <= 100, (
        f"slow-poisoning took too long to fire: first detection at "
        f"batch {slow_fired_at}, EWMA={ewma_trace[slow_fired_at]:.3f}"
    )

    # ── RTP integration: the SLOW_POISONING_SUSPECTED event fires ──
    # Drive the same injection pattern through a real RTP and verify
    # the event payload shape matches the spec.  The pipeline uses
    # ``_compute_injection_scale`` against the actual fitted DPD so
    # Mahal distance is deterministically ``_TARGET_MAHAL``, keeping
    # the per-batch hard arm silent on every injection window.
    pipeline, slow_threshold = _run_slow_poison_pipeline()
    slow_events = [
        e for e in pipeline.rtp.event_log
        if e.event_type == EventType.SLOW_POISONING_SUSPECTED
    ]
    assert slow_events, (
        "RTP pipeline never emitted SLOW_POISONING_SUSPECTED on the "
        "sustained sub-threshold drift stream"
    )
    payload = slow_events[-1].details
    # Spec payload keys: {step, ewma, threshold, n_batches_above}.
    for key in ("step", "ewma", "threshold", "n_batches_above"):
        assert key in payload, f"event payload missing '{key}': {payload}"
    assert payload["threshold"] == pytest.approx(slow_threshold)
    assert payload["ewma"] > payload["threshold"], (
        "event fired but ewma <= threshold — should be impossible"
    )
    assert payload["n_batches_above"] >= 1


def _run_slow_poison_pipeline() -> tuple[Pipeline, float]:
    """
    Build a real RTP pipeline and drive it with sustained below-
    per-batch-threshold drift so a SLOW_POISONING_SUSPECTED event
    reliably lands in the event log.

    The pipeline uses a smaller ``check_interval`` so the DPD runs
    often enough for the EWMA to climb past the slow threshold within
    the test's step budget.  We bump the slow-poisoning alpha to 0.1
    and use a lower slow-poisoning threshold so the EWMA converges
    past it with a modest per-batch Mahal distance — keeping the
    per-batch sparsity gate intact (2 hits at Mahal ≈ 6.5 is still
    below both the 8σ hard threshold and the 3-hit soft count gate).

    Test isolation: ATM-driven retrain bypassed
    -------------------------------------------
    The slow-poisoning arm under test relies on the EWMA of per-batch
    ``mahal_max`` accumulating across consecutive checks.  In a fully-
    wired pipeline the small (~0.26σ) population mean-shift the
    injection induces in feature 0 is enough for the DDD's KS test to
    fire — which dispatches MToUT → ATM.handle → MTP retrain →
    ``rtp.notify_model_updated`` → ``dpd.fit_reference`` → **EWMA = 0**.
    That refit destroys the accumulating cumulative-drift signal the
    test is trying to verify, even though the per-batch DPD itself
    behaves correctly throughout.  In production this is the right
    behaviour (the DDD already caught the anomaly through a different
    arm; further DPD-EWMA accumulation is redundant), but it makes
    direct verification of the slow-poisoning contract impossible at
    pipeline level without isolating the EWMA from the retrain chain.

    We swap the on_mtout handler to a recording-only stub *before*
    streaming the injection.  MToUT signals still fire (Test E's
    contract depends on them), the event log still accumulates the
    SLOW_POISONING_SUSPECTED entries (Test A's contract depends on
    them), but ATM never retrains and DPD's EWMA accumulates cleanly.

    Returns the pipeline and the slow-poisoning threshold the RTP was
    configured with, so callers can assert payload correctness against
    the actual (not hard-coded) threshold value.
    """
    slow_threshold = 5.5
    config = RTPConfig(
        cdd_task="classifier",
        check_interval=50,
        mtout_cooldown_steps=50,
        buffer_maxlen=5000,
        dpd_slow_poisoning_alpha=0.10,                 # faster EWMA convergence
        dpd_slow_poisoning_threshold=slow_threshold,
    )
    pipeline = build_pipeline(task="classifier", n_features=4, config=config)

    # Replace the ATM-dispatching MToUT handler with a recording-only
    # stub.  See the docstring above for why this is necessary.  We
    # still record into ``pipeline.events`` so downstream assertions
    # over MToUT signals (Test E) work unchanged.
    pipeline.rtp._on_mtout = lambda sig: pipeline.events.append(("mtout", sig))

    # Compute the injection scale AGAINST THE ACTUAL FITTED DPD the
    # pipeline just built, so the injection is deterministically at the
    # target Mahal distance regardless of empirical-covariance sampling
    # noise in the pipeline's reference fit.  The build_pipeline helper
    # fits DPD on the first 300 rows of its classifier corpus (which
    # come from ``rng.normal(shift=0, 1.0, ...)``), so DPD is ready.
    inject_scale = _compute_injection_scale(pipeline.rtp.dpd, _TARGET_MAHAL)

    # The pipeline builder pre-loads LIB with 300 clean reference rows.
    # The test streams 100 injection batches of 50 observations each —
    # each batch has 2 samples at Mahal=_TARGET_MAHAL and 48 clean
    # samples at N(0, _CLEAN_STD·I), so the DPD's recent_size=50 window
    # sees exactly the pattern Test A exercises.
    rng = np.random.default_rng(9002)
    n_features = 4
    recent_size = 50
    batches = 100
    for b in range(batches):
        batch = rng.normal(0.0, _CLEAN_STD, size=(recent_size, n_features))
        for i in range(_INJECTIONS_PER_BATCH):
            batch[i] = 0.0
            batch[i, 0] = inject_scale
        # Use labels consistent with the underlying corpus shape so
        # downstream CDD / CPD stays healthy — label=0 is a fine
        # placeholder and irrelevant to the DPD contract under test.
        for row in batch:
            pipeline.rtp.observe(row, y_true=np.array([0]))
    return pipeline, slow_threshold


# ---------------------------------------------------------------------------
# Test B — no false positives on clean data
# ---------------------------------------------------------------------------

def test_b_clean_data_does_not_trigger_slow_poisoning() -> None:
    """
    200 batches of clean IID data must leave both arms quiet.

    The EWMA on clean data converges to the mean of ``mahal_max``,
    which for a 50-sample window under N(0, I) (the same distribution
    as the reference) sits around 3-4σ — the max of 50 χ_4-distributed
    samples.  That is well below the 5.5 slow threshold, so
    slow_poisoning never fires, and obviously well below the 4σ soft /
    8σ hard per-batch thresholds.
    """
    rng = np.random.default_rng(9010)
    ref = rng.normal(0.0, 1.0, size=(300, 4))

    dpd = _make_dpd()
    lib = _build_lib_with_reference(ref)

    # Warm up.
    _push_batch(lib, rng, dpd.recent_size, clean_std=1.0, n_features=4)
    dpd.check(lib)

    per_batch_fires: list[int] = []
    slow_fires: list[int] = []

    # Use clean_std=1.0 (same as reference) so this explicitly tests
    # "same-distribution clean traffic doesn't false-fire" — the
    # strongest negative-control condition we can apply.
    for i in range(200):
        _push_batch(lib, rng, dpd.recent_size, clean_std=1.0, n_features=4)
        result = dpd.check(lib)
        if result.poisoning_detected:
            per_batch_fires.append(i)
        if result.slow_poisoning_detected:
            slow_fires.append(i)

    assert not per_batch_fires, (
        f"per-batch DPD false-fired on clean data at batches "
        f"{per_batch_fires}"
    )
    assert not slow_fires, (
        f"slow-poisoning false-fired on clean data at batches "
        f"{slow_fires}. Final EWMA={result.mahal_ewma:.3f}. "
        f"The EWMA should converge well below the 5.5 threshold on "
        f"clean N(0, I) data."
    )


# ---------------------------------------------------------------------------
# Test C — no double-fire on a single big spike
# ---------------------------------------------------------------------------

def test_c_big_spike_does_not_double_fire_slow_arm() -> None:
    """
    A single window with a massive injection fires the per-batch arm.
    The slow arm must NOT additionally fire on that same window —
    the design rule is "slow fires only when per-batch stays silent",
    preventing duplicate DATA_POISONING events for a single incident.
    """
    rng = np.random.default_rng(9020)
    ref = rng.normal(0.0, 1.0, size=(300, 4))

    dpd = _make_dpd()
    lib = _build_lib_with_reference(ref)

    # Warm up.
    _push_batch(lib, rng, dpd.recent_size, n_features=4)
    dpd.check(lib)

    # One large spike: 10 samples at 20σ in the recent window.
    # This trips the extreme escape hatch (2 × hard = 16σ) and
    # also the hard arm — per-batch definitely fires.
    _push_batch(
        lib, rng, dpd.recent_size,
        inject_sigma=20.0, n_injections=10,
    )
    result = dpd.check(lib)

    assert result.poisoning_detected, (
        "pre-condition failed — the 20σ spike should fire per-batch"
    )
    assert result.mahal_triggered
    # Core assertion: slow arm does NOT co-fire on this window.
    assert not result.slow_poisoning_detected, (
        f"slow arm double-fired on the same window the per-batch arm "
        f"already fired on (ewma={result.mahal_ewma:.3f}). The "
        f"slow-fires-only-when-per-batch-quiet rule is broken."
    )


# ---------------------------------------------------------------------------
# Test D — EWMA memory decay
# ---------------------------------------------------------------------------

def test_d_ewma_decays_after_sustained_drift_stops() -> None:
    """
    100 batches of sustained drift push the EWMA above the test's
    slow threshold (5.5), then 100 clean batches should decay the
    EWMA back below 5.5 — slow_poisoning stops firing once the
    cumulative evidence falls below threshold.

    Closed-form decay: EWMA_{t+k} = (1-α)^k · EWMA_t + (1 - (1-α)^k) · μ_clean.
    With α=0.05, EWMA_t ≈ _TARGET_MAHAL=6.5 after 100 drift batches.
    After k clean batches ``(0.95)^k · 6.5 + (1 - 0.95^k) · μ_clean``
    where μ_clean is the EWMA's stationary value on clean data (~1.5
    with our narrower clean_std); for k=50 this gives ≈ 2.0, deep
    below the 5.5 threshold.  100 clean batches is therefore generous.
    """
    rng = np.random.default_rng(9030)
    ref = rng.normal(0.0, 1.0, size=(300, 4))

    dpd = _make_dpd()
    lib = _build_lib_with_reference(ref)

    # Warm up.
    _push_batch(lib, rng, dpd.recent_size, n_features=4)
    dpd.check(lib)

    inject_scale = _compute_injection_scale(dpd, _TARGET_MAHAL)

    # Drive the EWMA above threshold.
    drift_fires_seen = 0
    for _ in range(100):
        _push_batch(
            lib, rng, dpd.recent_size,
            inject_sigma=inject_scale,
            n_injections=_INJECTIONS_PER_BATCH,
        )
        result = dpd.check(lib)
        if result.slow_poisoning_detected:
            drift_fires_seen += 1

    assert drift_fires_seen > 0, (
        "pre-condition failed — slow_poisoning never fired during "
        "the 100-batch sustained-drift phase"
    )
    assert result.mahal_ewma > 5.5, (
        f"pre-condition failed — EWMA={result.mahal_ewma:.3f} did not "
        f"rise above the slow threshold"
    )
    ewma_at_stop = result.mahal_ewma

    # Now drop the drift — 100 clean batches to let the EWMA decay.
    clean_slow_fires: list[int] = []
    for i in range(100):
        _push_batch(lib, rng, dpd.recent_size, n_features=4)
        result = dpd.check(lib)
        if result.slow_poisoning_detected:
            clean_slow_fires.append(i)

    # The EWMA should have decayed well below the threshold …
    assert result.mahal_ewma < 5.5, (
        f"EWMA did not decay below the slow threshold after 100 clean "
        f"batches: final EWMA={result.mahal_ewma:.3f} (started at "
        f"{ewma_at_stop:.3f}). Either α is too small or clean data is "
        f"leaving an above-threshold residual."
    )
    # … and slow_poisoning must stop firing well before the 100-batch
    # clean phase ends.  We tolerate a short tail of above-threshold
    # fires while the EWMA is still decaying through the threshold —
    # the important invariant is that the final stretch is clean.
    tail_fires = [i for i in clean_slow_fires if i >= 80]
    assert not tail_fires, (
        f"slow_poisoning still firing in the clean-tail window "
        f"(batches 80-99): fires at {tail_fires}. EWMA decay is "
        f"not reaching below-threshold fast enough."
    )


# ---------------------------------------------------------------------------
# Test E — MToUT integration
# ---------------------------------------------------------------------------

def test_e_slow_poisoning_only_path_produces_medium_severity_mtout() -> None:
    """
    When only the slow-poisoning arm fires (no per-batch DPD, no CPD,
    no DDD, no CDD), the MToUT signal must carry
    ``TriggerReason.DATA_POISONING`` AND severity ≥ MEDIUM — but NOT
    CRITICAL (the "auto-rollback immediately" level reserved for the
    per-batch / CPD fast path).

    Also asserts a couple of negative contract items:

    * No SECURITY_ALERT event was emitted — the slow path is
      informational-for-investigation, not an immediate evict-the-model
      signal.
    * No ROLLBACK event was emitted — same reason.

    Mechanism: drive the same sustained sub-threshold injection
    pattern as Test A through the full ``build_pipeline`` harness.
    The pipeline wires an MToUT handler that records every signal
    it receives; after the run we inspect the recorded signals.
    """
    config = RTPConfig(
        cdd_task="classifier",
        check_interval=50,
        mtout_cooldown_steps=50,
        buffer_maxlen=5000,
        # Faster EWMA convergence + test-calibrated slow threshold
        # matching the target Mahal injection magnitude so the test
        # finishes inside a reasonable step budget.  5.5 is safely
        # below the _TARGET_MAHAL=6.5 asymptote and safely above the
        # clean-data EWMA steady-state (~1.5 under _CLEAN_STD).
        dpd_slow_poisoning_alpha=0.10,
        dpd_slow_poisoning_threshold=5.5,
    )
    pipeline = build_pipeline(task="classifier", n_features=4, config=config)

    # Bypass ATM retrain chain so the DPD reference is not refit
    # mid-stream — every refit resets the EWMA the slow arm tracks.
    # See ``_run_slow_poison_pipeline`` for the full rationale; the
    # short version: an ATM-driven retrain (which DDD's KS would
    # eventually trigger on the small mean-shift the injection causes)
    # would call ``dpd.fit_reference`` and zero out the EWMA before it
    # could cross the slow-poisoning threshold, masking the slow-only
    # MToUT path this test is specifically here to verify.  Replacing
    # the ATM-dispatching ``_on_mtout`` with a recording-only stub
    # preserves every signal MToUT fires (the test asserts on those
    # below) while breaking the retrain chain.
    pipeline.rtp._on_mtout = lambda sig: pipeline.events.append(("mtout", sig))

    # Compute the injection-coordinate scale so the Mahal distance is
    # deterministically _TARGET_MAHAL regardless of empirical covariance
    # sampling noise.  This is what keeps the per-batch hard arm (8σ)
    # from firing accidentally on the same window the slow arm fires on.
    inject_scale = _compute_injection_scale(pipeline.rtp.dpd, _TARGET_MAHAL)

    # Drive 100 batches of the sustained sub-threshold injection —
    # enough for the EWMA to climb past 5.5 (closed-form: ~10 batches
    # with α=0.10, Mahal ≈ 6.5 per batch).
    rng = np.random.default_rng(9040)
    n_features = 4
    recent_size = 50
    for _ in range(100):
        batch = rng.normal(0.0, _CLEAN_STD, size=(recent_size, n_features))
        for i in range(_INJECTIONS_PER_BATCH):
            batch[i] = 0.0
            batch[i, 0] = inject_scale
        for row in batch:
            pipeline.rtp.observe(row, y_true=np.array([0]))

    # ── Locate the slow-poisoning-only MToUT signals. ────────────────
    # The harness records every MToUT the RTP emits.  A slow-only
    # signal carries (a) slow_poisoning_only=True, and (b) a DPD
    # result with slow_poisoning_detected=True but poisoning_detected
    # False.  If several signals are recorded we accept any that
    # satisfy both — CDD / DDD firing on the same batch would add
    # extra reasons, which is still a valid severity-bump scenario
    # as long as the DPD contribution was purely the slow arm.
    mtout_events = [
        payload for (kind, payload) in pipeline.events if kind == "mtout"
    ]
    assert mtout_events, (
        "no MToUT signals fired during the slow-poisoning run"
    )
    slow_signals: list[MToUTSignal] = [
        s for s in mtout_events
        if (
            s.dpd_result is not None
            and getattr(s.dpd_result, "slow_poisoning_detected", False)
            and not s.dpd_result.poisoning_detected
        )
    ]
    assert slow_signals, (
        f"no MToUT signal carried a slow-only DPD result. Signals "
        f"seen: {[str(s) for s in mtout_events]}"
    )

    # ── Contract A: TriggerReason.DATA_POISONING is present. ─────────
    for sig in slow_signals:
        assert TriggerReason.DATA_POISONING in sig.reasons, (
            f"slow-only MToUT signal missing DATA_POISONING reason: "
            f"{sig}"
        )

    # ── Contract B: severity is bumped to ≥ MEDIUM but NOT CRITICAL
    # (when slow-only — CPD or DDD firing concurrently may still push
    # higher, but this is the pure-slow-only path). ─────────────────
    pure_slow_only = [s for s in slow_signals if s.slow_poisoning_only]
    assert pure_slow_only, (
        "expected at least one MToUT signal with slow_poisoning_only="
        "True — the RTP did not tag the signal for severity-clamping"
    )
    severities = [s.severity() for s in pure_slow_only]
    # Must NOT auto-escalate to CRITICAL on the slow-only path.
    assert "CRITICAL" not in severities, (
        f"slow-only MToUT escalated to CRITICAL: {severities}. "
        f"Slow-poisoning is cumulative and must cap at MEDIUM."
    )
    # At least one signal must be MEDIUM or HIGH.
    assert any(s in ("MEDIUM", "HIGH") for s in severities), (
        f"slow-only MToUT severity below MEDIUM: {severities}"
    )

    # ── Contract C: no SECURITY_ALERT or ROLLBACK on slow-only path. ─
    # Both would indicate the RTP mistook the cumulative signal for an
    # immediate threat and took the per-batch / CPD response path.
    event_types = {e.event_type for e in pipeline.rtp.event_log}
    # SLOW_POISONING_SUSPECTED should definitely be there.
    assert EventType.SLOW_POISONING_SUSPECTED in event_types
    # Filter security / rollback events that occurred AFTER any
    # slow-only signal fired.
    if pure_slow_only:
        first_slow_step = min(s.step for s in pure_slow_only)
        later_security = [
            e for e in pipeline.rtp.event_log
            if e.event_type == EventType.SECURITY_ALERT
            and (e.step or 0) >= first_slow_step
        ]
        later_rollback = [
            e for e in pipeline.rtp.event_log
            if e.event_type == EventType.ROLLBACK
            and (e.step or 0) >= first_slow_step
        ]
        assert not later_security, (
            f"SECURITY_ALERT emitted after slow-only MToUT signal — "
            f"slow path should not trip the fast-alarm response. "
            f"events={later_security}"
        )
        assert not later_rollback, (
            f"ROLLBACK emitted after slow-only MToUT signal — slow "
            f"path should not auto-rollback. events={later_rollback}"
        )
