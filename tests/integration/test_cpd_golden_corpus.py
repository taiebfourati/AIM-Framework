"""
Tier-2 integration tests for the CPD-shadow hardening (GoldenCorpus).

Threat model under test
-----------------------
A patient attacker who can influence inputs that reach ``rtp.observe``
feeds a slow stream of label-flipped ``(x, y_true)`` pairs. Over many
windows the ``YGT`` buffer accumulates poisoned labels. When a retrain
cycle fires ``RTP.notify_model_updated`` the CPD shadow is (historically)
refit from the YGT-derived tail — so the shadow learns the poisoned
boundary and then ``shadow_divergence`` stays low even though the
production model has been taught the same poisoned boundary. The one
detector designed for concept-space attacks fails silently.

The fix under test
------------------
Introduce an operator-curated ``GoldenCorpus`` that is the *only*
source the CPD shadow may be refit from in production. The corpus is
hash-chained and (optionally) operator-signed; ``snapshot(n)`` returns
a content-hashed view that the CPD refit binds to.

Scenarios
---------
* ``test_case_a_raw_ygt_refit_fails_to_detect_poisoning`` — documents
  the vulnerability: without the golden corpus, the shadow is refit
  from the poisoned YGT tail and CPD does NOT fire.
* ``test_case_b_golden_corpus_refit_detects_poisoning`` — documents
  the fix: with the golden corpus wired in, the shadow is refit from
  the trusted snapshot and CPD DOES fire.
* ``test_case_c_clean_stream_with_golden_corpus_stays_quiet`` — the
  regression: on a clean stream the golden-corpus shadow must NOT
  false-fire across N check intervals.
* ``test_shadow_refit_event_captures_provenance`` — verifies the
  ``SHADOW_REFIT`` event carries ``source``, ``corpus_hash`` and
  ``n_rows`` so auditors can trace which commitment fed any given
  shadow decision.

Implementation notes
--------------------
* All (X, y) corpora used in a single scenario share the SAME
  label-generating weight vector ``w``. ``make_classifier_corpus``
  draws ``w`` from the seeded RNG each call, so two calls with the
  same seed but different ``n`` return DIFFERENT ``w`` vectors. The
  tests therefore build ONE large corpus per scenario and slice it
  into (golden, attack, probe) subsets — the same pattern the Tier-4
  scenarios use.
* The poisoning is modelled as a *full* label flip on the attacker-
  controlled traffic — this is the hardest case for the adversary:
  if the shadow learns this boundary it PERFECTLY matches the
  poisoned MLIN, driving shadow divergence to zero. Anything less
  extreme would leave residual divergence on the unsafe path, and
  the test would fail to reproduce the silent-failure mode.
* The test does NOT go through ATM end-to-end. It calls
  ``notify_model_updated`` directly with a pre-baked poisoned
  estimator so the CPD refit provenance is the only variable.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from aif.aif import AIF
from aif.golden_corpus import GoldenCorpus
from rtp.rtp import RTP, RTPConfig, EventType

from ._harness import make_classifier_corpus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slice_corpus(
    X: np.ndarray, y: np.ndarray, ranges: dict[str, tuple[int, int]]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return named slices of a corpus keyed by role (golden/attack/probe…)."""
    return {name: (X[a:b], y[a:b]) for name, (a, b) in ranges.items()}


def _fit_honest_baseline(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """The pre-attack MLIN: fit on honest labels."""
    est = LogisticRegression(max_iter=500, random_state=0)
    est.fit(X, y)
    return est


def _fit_poisoned_baseline(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """The post-attack MLIN: fit on flipped labels — simulates a patient
    YGT-poisoning attack that ended up in the ATM's training batch."""
    est = LogisticRegression(max_iter=500, random_state=0)
    est.fit(X, 1 - y)
    return est


def _build_bare_rtp(
    X_baseline: np.ndarray,
    y_baseline: np.ndarray,
    *,
    golden_corpus: GoldenCorpus | None = None,
) -> RTP:
    """
    Build a minimal RTP — fitted AIF, no ATM, no detectors wired to
    fire on their own. Tests poke the RTP directly with
    ``notify_model_updated`` and ``force_check`` so we can isolate the
    CPD refit provenance.
    """
    honest_mlin = _fit_honest_baseline(X_baseline, y_baseline)
    aif = AIF(estimator=honest_mlin, sib_capacity=1)
    cfg = RTPConfig(
        cdd_task="classifier",
        check_interval=10**6,       # never auto-runs the battery
        mtout_cooldown_steps=0,
        buffer_maxlen=5000,
        cpd_reference_size=300,
        cpd_recent_size=100,
    )
    rtp = RTP(aif=aif, config=cfg, golden_corpus=golden_corpus)
    rtp.set_reference(X_baseline[:300], y_baseline[:300])
    return rtp


def _feed_live_traffic(
    rtp: RTP, X: np.ndarray, y_true: np.ndarray
) -> None:
    """
    Stream (x, y_true) pairs through ``observe`` so LIB/LOB/YGT fill
    up naturally. The attacker's YGT-poisoning is exactly what this
    loop models when ``y_true`` is label-flipped.
    """
    for i, x in enumerate(X):
        rtp.observe(x, y_true=np.array([y_true[i]]))


# ---------------------------------------------------------------------------
# Case A — without the fix, the CPD is silent under the attack
# ---------------------------------------------------------------------------

def test_case_a_raw_ygt_refit_fails_to_detect_poisoning() -> None:
    """
    The vulnerable configuration: no ``golden_corpus`` is supplied, so
    ``notify_model_updated`` refits the CPD shadow from the poisoned
    YGT tail. The shadow learns the flipped boundary, matches the
    poisoned production model's outputs, and ``poisoning_detected``
    stays False — documenting the silent-failure mode.
    """
    # ── ONE big corpus, shared label-generating weight ──────────────
    X_big, y_big = make_classifier_corpus(2000, 4, seed=4200)
    slices = _slice_corpus(X_big, y_big, {
        "baseline": (0, 600),
        "attack":   (600, 1300),    # 700 rows of poisoned traffic
        "probe":    (1300, 1600),   # 300 rows to trigger CPD.check
    })
    X_baseline, y_baseline = slices["baseline"]
    X_attack, y_attack_true = slices["attack"]
    X_probe, y_probe_true = slices["probe"]

    # ── Bare RTP — honest baseline MLIN, no golden corpus ───────────
    rtp = _build_bare_rtp(X_baseline, y_baseline, golden_corpus=None)

    # ── Attack: slow-drip of label-flipped (x, y_true) pairs ────────
    # Every y_true the attacker supplies is the OPPOSITE of the honest
    # label — that is the poisoning channel that ends up in YGT.
    y_attack_poisoned = 1 - y_attack_true
    _feed_live_traffic(rtp, X_attack, y_attack_poisoned)

    # ── Simulate the retrained MLIN (also poisoned, since ATM would
    # have trained it on the YGT-poisoned rows) ─────────────────────
    poisoned_mlin = _fit_poisoned_baseline(X_baseline, y_baseline)
    rtp.notify_model_updated(poisoned_mlin)

    # Sanity: the SHADOW_REFIT event recorded the unsafe path.
    refit_events = [
        e for e in rtp.event_log
        if e.event_type == EventType.SHADOW_REFIT
    ]
    assert refit_events, "expected at least one SHADOW_REFIT event"
    last_refit = refit_events[-1]
    assert last_refit.details["source"] in (
        "unsafe_ygt", "unsafe_lob", "unsafe_caller_batch",
    ), (
        f"expected unsafe refit source, got {last_refit.details['source']!r}"
    )
    assert last_refit.details["corpus_hash"] == "unsafe_raw"

    # ── Probe: feed on-regime traffic with attacker-flipped y_true
    # so the RTP sees (clean X, poisoned MLIN predictions, flipped y_true). ─
    _feed_live_traffic(rtp, X_probe, 1 - y_probe_true)

    # Run the detector battery on the current buffers.
    rtp.force_check()
    cpd = rtp.last_cpd
    assert cpd is not None, "force_check did not populate last_cpd"

    # Core vulnerability assertion: CPD does NOT fire even though an
    # attacker has tutored both MLIN and the shadow onto the same
    # poisoned boundary.
    assert not cpd.poisoning_detected, (
        "VULNERABILITY NOT REPRODUCED: CPD fired on the unsafe refit "
        "path. Expected the shadow to have been trained on poisoned "
        "YGT data and therefore to agree with the poisoned MLIN. "
        f"result: {cpd.message}"
    )
    # The shadow was trained on the SAME poisoned boundary the new
    # MLIN learned, so divergence is near zero.
    assert cpd.shadow_divergence <= cpd.shadow_threshold, (
        f"shadow_divergence={cpd.shadow_divergence:.3f} exceeded "
        f"threshold {cpd.shadow_threshold:.3f} — the shadow was not "
        f"fully calibrated to the poisoned boundary"
    )
    # Provenance is stamped on the result so an auditor can see the
    # fire-or-not decision was made off an unsafe shadow.
    assert cpd.shadow_source_hash == "unsafe_raw", (
        f"shadow_source_hash must be 'unsafe_raw' on the vulnerable "
        f"path; got {cpd.shadow_source_hash!r}"
    )


# ---------------------------------------------------------------------------
# Case B — with the fix, the CPD correctly fires under the attack
# ---------------------------------------------------------------------------

def test_case_b_golden_corpus_refit_detects_poisoning() -> None:
    """
    The fixed configuration: a ``GoldenCorpus`` holds operator-signed
    clean rows. ``notify_model_updated`` refits the shadow from the
    snapshot, NOT from the poisoned YGT. The shadow therefore predicts
    the HONEST label on the attack's inputs, diverges sharply from the
    poisoned MLIN's outputs, and the corroboration rule fires.
    """
    # ── ONE big corpus, shared label-generating weight ──────────────
    X_big, y_big = make_classifier_corpus(2000, 4, seed=4200)
    slices = _slice_corpus(X_big, y_big, {
        "baseline": (0, 600),
        "golden":   (600, 1000),    # operator-curated CLEAN rows
        "attack":   (1000, 1600),
        "probe":    (1600, 1900),
    })
    X_baseline, y_baseline = slices["baseline"]
    X_golden, y_golden = slices["golden"]
    X_attack, y_attack_true = slices["attack"]
    X_probe, y_probe_true = slices["probe"]

    # ── Operator-signed golden corpus ───────────────────────────────
    golden = GoldenCorpus(n_features=4, allow_unauthorised=True)
    sig = golden.sign_append_payload("ops_2026Q1", X_golden, y_golden)
    golden.append(
        X_golden, y_golden, source="ops_2026Q1", operator_sig=sig,
    )
    assert golden.is_ready(300), "golden corpus not populated correctly"

    # ── Bare RTP — honest baseline MLIN, WITH golden corpus ─────────
    rtp = _build_bare_rtp(X_baseline, y_baseline, golden_corpus=golden)

    # ── Same attack as Case A ───────────────────────────────────────
    y_attack_poisoned = 1 - y_attack_true
    _feed_live_traffic(rtp, X_attack, y_attack_poisoned)

    # ── Swap in the poisoned MLIN ───────────────────────────────────
    poisoned_mlin = _fit_poisoned_baseline(X_baseline, y_baseline)
    rtp.notify_model_updated(poisoned_mlin)

    # The SHADOW_REFIT event must record the ``golden`` source AND
    # carry the snapshot's corpus_hash — the refit was bound to the
    # trusted commitment.
    refit_events = [
        e for e in rtp.event_log
        if e.event_type == EventType.SHADOW_REFIT
    ]
    assert refit_events
    golden_refits = [
        e for e in refit_events if e.details.get("source") == "golden"
    ]
    assert golden_refits, (
        f"expected a SHADOW_REFIT with source='golden'; got sources="
        f"{[e.details.get('source') for e in refit_events]}"
    )
    latest_golden = golden_refits[-1]
    assert latest_golden.details["n_rows"] > 0
    assert latest_golden.details["corpus_hash"] not in (None, "unsafe_raw")
    # Hash on the event must bind to the exact snapshot that would be
    # drawn for that refit size.
    latest_snapshot = golden.snapshot(rtp.cpd.reference_size)
    assert (
        latest_golden.details["corpus_hash"] == latest_snapshot.corpus_hash_hex
    ), "SHADOW_REFIT corpus_hash must bind to the exact snapshot rows"

    # ── Probe: feed poisoned traffic into the post-update pipeline ─
    _feed_live_traffic(rtp, X_probe, 1 - y_probe_true)

    rtp.force_check()
    cpd = rtp.last_cpd
    assert cpd is not None

    # Core fix assertion: with a trusted shadow, the poisoned MLIN's
    # outputs diverge from honest predictions — CPD fires.
    assert cpd.shadow_triggered, (
        f"shadow divergence stayed below threshold "
        f"({cpd.shadow_divergence:.3f} <= {cpd.shadow_threshold}) — "
        f"the golden-corpus shadow was supposed to catch the poisoned "
        f"MLIN"
    )
    assert cpd.poisoning_detected, (
        f"CPD did not fire even though shadow triggered; result: "
        f"{cpd.message}"
    )
    # Provenance must be the real corpus hash, not "unsafe_raw".
    assert cpd.shadow_source_hash == latest_snapshot.corpus_hash_hex, (
        f"shadow_source_hash should carry the snapshot hash; got "
        f"{cpd.shadow_source_hash!r}"
    )


# ---------------------------------------------------------------------------
# Regression — clean stream, golden corpus, no false positives
# ---------------------------------------------------------------------------

def test_case_c_clean_stream_with_golden_corpus_stays_quiet() -> None:
    """
    With the fix wired in and the stream entirely clean, CPD must stay
    silent across many check calls — the golden-corpus refit must not
    introduce new false positives.
    """
    X_big, y_big = make_classifier_corpus(2000, 4, seed=4201)
    slices = _slice_corpus(X_big, y_big, {
        "baseline": (0, 600),
        "golden":   (600, 1000),
        "stream":   (1000, 1800),   # 8 × check_interval=100-sized probes
    })
    X_baseline, y_baseline = slices["baseline"]
    X_golden, y_golden = slices["golden"]
    X_stream, y_stream = slices["stream"]

    golden = GoldenCorpus(n_features=4, allow_unauthorised=True)
    sig = golden.sign_append_payload("baseline", X_golden, y_golden)
    golden.append(X_golden, y_golden, source="baseline", operator_sig=sig)

    rtp = _build_bare_rtp(X_baseline, y_baseline, golden_corpus=golden)

    # Stream clean traffic. Every ``cpd.recent_size`` samples we run
    # force_check() and require CPD to stay quiet. This is the
    # N-window regression the audit asked for.
    recent = rtp.cpd.recent_size
    quiet_checks = 0
    fired: list[str] = []
    for start in range(0, len(X_stream) - recent, recent):
        batch_X = X_stream[start:start + recent]
        batch_y = y_stream[start:start + recent]
        _feed_live_traffic(rtp, batch_X, batch_y)
        rtp.force_check()
        cpd = rtp.last_cpd
        if cpd is None:
            continue
        if cpd.poisoning_detected:
            fired.append(
                f"step={rtp._step} shadow={cpd.shadow_divergence:.2f} "
                f"ks={cpd.output_ks_pvalue:.3f} "
                f"dr={cpd.corr_delta_max:.2f} z={cpd.corr_z_max:.2f}"
            )
        else:
            quiet_checks += 1
    assert not fired, (
        f"CPD false-fired on clean stream with golden corpus: {fired}"
    )
    assert quiet_checks >= 5, (
        f"not enough CPD.check() calls ran to form a meaningful "
        f"regression test (quiet_checks={quiet_checks})"
    )


# ---------------------------------------------------------------------------
# Audit trail — SHADOW_REFIT event provenance
# ---------------------------------------------------------------------------

def test_shadow_refit_event_captures_provenance() -> None:
    """
    Every CPD shadow refit must emit a ``SHADOW_REFIT`` event whose
    ``details`` carry {step, source, corpus_hash, n_rows}. Auditors
    use these fields to correlate a later firing (or silence) of CPD
    back to a specific corpus commitment.
    """
    X_big, y_big = make_classifier_corpus(1500, 4, seed=4202)
    X_baseline, y_baseline = X_big[:600], y_big[:600]
    X_golden, y_golden = X_big[600:1000], y_big[600:1000]

    # ── Case 1: bootstrap without a golden corpus ───────────────────
    rtp1 = _build_bare_rtp(X_baseline, y_baseline)
    refits1 = [
        e for e in rtp1.event_log
        if e.event_type == EventType.SHADOW_REFIT
    ]
    assert refits1, "set_reference must emit SHADOW_REFIT"
    boot = refits1[-1]
    assert boot.details["source"] == "bootstrap"
    assert boot.details["corpus_hash"] == "unsafe_raw"
    assert boot.details["n_rows"] > 0
    assert "step" in boot.details

    # ── Case 2: bootstrap WITH a golden corpus ──────────────────────
    golden = GoldenCorpus(n_features=4, allow_unauthorised=True)
    sig = golden.sign_append_payload("baseline", X_golden, y_golden)
    golden.append(X_golden, y_golden, source="baseline", operator_sig=sig)

    rtp2 = _build_bare_rtp(X_baseline, y_baseline, golden_corpus=golden)
    refits2 = [
        e for e in rtp2.event_log
        if e.event_type == EventType.SHADOW_REFIT
    ]
    sources = [e.details["source"] for e in refits2]
    assert "bootstrap" in sources and "golden" in sources, (
        f"set_reference with golden corpus must emit both bootstrap "
        f"and golden SHADOW_REFIT events; got sources={sources}"
    )
    golden_event = next(
        e for e in refits2 if e.details["source"] == "golden"
    )
    snap = golden.snapshot(rtp2.cpd.reference_size)
    assert golden_event.details["corpus_hash"] == snap.corpus_hash_hex
    assert golden_event.details["n_rows"] == snap.n_rows

    # The CPD's own shadow_source_hash property matches what the event
    # recorded — completing the audit-trail round-trip.
    assert rtp2.cpd.shadow_source_hash == snap.corpus_hash_hex
