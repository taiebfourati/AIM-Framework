"""
Tier-3 unit tests for atm.atm.ATM._select_variant — the composition gate.

This is the routing logic the paper labels *MTPC composition*: given an
MToUTSignal and a batch size, pick one of MTP-L / MTP-E / MTP-C (or fall
through to the operator override).  The rule table (paper Section V,
implemented in ``ATM._select_variant``) is:

  policy.prefer_variant  → hard override, wins against everything
  CRITICAL severity      → MTP-L if ``critical_always_local_first`` *and*
                           batch ≤ ``local_max_samples``; else MTP-E
  HIGH severity          → MTP-E always (two drifts simultaneously =
                           full retrain, route to the quality path)
  MEDIUM / LOW severity  → MTP-L if batch ≤ ``local_max_samples``;
                           else MTP-E (data volume forces the escalation)

These tests pin every branch of that table and both sides of the
``local_max_samples`` boundary so a future refactor cannot silently
change the routing.

The tests do not construct a full ATM (MTP-L/E/C, NDT, DPostP, RTP);
``_select_variant`` only reads ``self.policy``, so we build a bare
instance via ``ATM.__new__`` and attach just the policy.  That keeps
the tests focused on the composition rule without pulling in MLflow,
sklearn fit cycles, or the DPostP default-construction path.
"""
from __future__ import annotations

import pytest

from atm.atm import ATM, ATMPolicy, MTPVariant
from rtp.rtp import MToUTSignal, TriggerReason


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bare_atm(policy: ATMPolicy) -> ATM:
    """
    Build an ATM skeleton whose only populated attribute is ``policy``.

    ``_select_variant`` reads nothing else, so instantiating via
    ``__new__`` side-steps the real constructor's dependency graph
    (RTP, MTPLocal, MTPExternal, NDT, DPostP).  Every other ATM
    method would of course blow up — this helper is *only* for
    exercising the composition rule.
    """
    atm = ATM.__new__(ATM)
    atm.policy = policy
    return atm


def _signal(reasons: list[TriggerReason]) -> MToUTSignal:
    """Build a minimal MToUTSignal; only ``reasons`` shape the severity."""
    return MToUTSignal(reasons=list(reasons), step=0)


# Convenience aliases — one of each severity level
SIG_LOW      = _signal([])
SIG_MEDIUM   = _signal([TriggerReason.DATA_DRIFT])
SIG_HIGH     = _signal([TriggerReason.DATA_DRIFT, TriggerReason.CONCEPT_DRIFT])
SIG_CRITICAL = _signal([TriggerReason.DATA_POISONING])


# ---------------------------------------------------------------------------
# Severity plumbing — sanity checks so the helpers don't silently drift
# from the severity() contract that ``_select_variant`` relies on.
# ---------------------------------------------------------------------------

class TestSignalSeverityFixtures:
    def test_fixture_severities(self) -> None:
        assert SIG_LOW.severity() == "LOW"
        assert SIG_MEDIUM.severity() == "MEDIUM"
        assert SIG_HIGH.severity() == "HIGH"
        assert SIG_CRITICAL.severity() == "CRITICAL"


# ---------------------------------------------------------------------------
# TestATMSelectVariant_Override — policy.prefer_variant beats everything
# ---------------------------------------------------------------------------

class TestATMSelectVariantOverride:
    """``prefer_variant`` short-circuits the severity cascade."""

    @pytest.mark.parametrize("variant", list(MTPVariant))
    def test_prefer_variant_beats_critical(self, variant: MTPVariant) -> None:
        """Operator pin wins even when the signal screams CRITICAL."""
        atm = _bare_atm(ATMPolicy(prefer_variant=variant))
        # Huge batch + CRITICAL + local-first would normally escalate to
        # EXTERNAL (too big for local speed).  Override must still win.
        assert atm._select_variant(SIG_CRITICAL, n_samples=10_000) is variant

    @pytest.mark.parametrize("variant", list(MTPVariant))
    def test_prefer_variant_beats_high(self, variant: MTPVariant) -> None:
        atm = _bare_atm(ATMPolicy(prefer_variant=variant))
        assert atm._select_variant(SIG_HIGH, n_samples=100) is variant

    def test_prefer_variant_cloud_routes_to_cloud(self) -> None:
        """Sanity: the override can pick MTP-C (never selected by the auto rule)."""
        atm = _bare_atm(ATMPolicy(prefer_variant=MTPVariant.CLOUD))
        assert atm._select_variant(SIG_MEDIUM, n_samples=100) is MTPVariant.CLOUD


# ---------------------------------------------------------------------------
# TestATMSelectVariant_Critical — poisoning path
# ---------------------------------------------------------------------------

class TestATMSelectVariantCritical:
    """
    CRITICAL = poisoning.  Speed matters most; route to MTP-L when
    the batch fits, otherwise fall through to MTP-E.  Disabling
    ``critical_always_local_first`` forces MTP-E unconditionally.
    """

    def test_critical_with_local_first_and_small_batch_picks_local(self) -> None:
        policy = ATMPolicy(
            critical_always_local_first=True,
            local_max_samples=500,
        )
        atm = _bare_atm(policy)
        assert atm._select_variant(SIG_CRITICAL, n_samples=100) is MTPVariant.LOCAL

    def test_critical_with_local_first_and_large_batch_escalates(self) -> None:
        """Batch > local_max_samples defeats the speed preference."""
        policy = ATMPolicy(
            critical_always_local_first=True,
            local_max_samples=500,
        )
        atm = _bare_atm(policy)
        assert atm._select_variant(SIG_CRITICAL, n_samples=501) is MTPVariant.EXTERNAL

    def test_critical_without_local_first_always_external(self) -> None:
        """Turning off the speed preference pins CRITICAL to MTP-E."""
        policy = ATMPolicy(
            critical_always_local_first=False,
            local_max_samples=500,
        )
        atm = _bare_atm(policy)
        # Small batch *would* fit local, but the policy disables that path.
        assert atm._select_variant(SIG_CRITICAL, n_samples=10) is MTPVariant.EXTERNAL

    def test_critical_local_boundary_is_inclusive(self) -> None:
        """n_samples == local_max_samples → LOCAL (``<=`` in the code)."""
        policy = ATMPolicy(
            critical_always_local_first=True,
            local_max_samples=500,
        )
        atm = _bare_atm(policy)
        assert atm._select_variant(SIG_CRITICAL, n_samples=500) is MTPVariant.LOCAL

    def test_critical_one_past_boundary_picks_external(self) -> None:
        policy = ATMPolicy(
            critical_always_local_first=True,
            local_max_samples=500,
        )
        atm = _bare_atm(policy)
        assert atm._select_variant(SIG_CRITICAL, n_samples=501) is MTPVariant.EXTERNAL


# ---------------------------------------------------------------------------
# TestATMSelectVariant_High — two drifts at once
# ---------------------------------------------------------------------------

class TestATMSelectVariantHigh:
    """HIGH severity (both drifts) → MTP-E regardless of batch size."""

    @pytest.mark.parametrize("n_samples", [1, 50, 500, 501, 10_000])
    def test_high_always_picks_external(self, n_samples: int) -> None:
        atm = _bare_atm(ATMPolicy(local_max_samples=500))
        assert atm._select_variant(SIG_HIGH, n_samples=n_samples) is MTPVariant.EXTERNAL

    def test_high_ignores_critical_local_first_flag(self) -> None:
        """``critical_always_local_first`` is a CRITICAL-only knob."""
        atm = _bare_atm(ATMPolicy(critical_always_local_first=True))
        assert atm._select_variant(SIG_HIGH, n_samples=10) is MTPVariant.EXTERNAL


# ---------------------------------------------------------------------------
# TestATMSelectVariant_MediumLow — single-drift / operator-request branch
# ---------------------------------------------------------------------------

class TestATMSelectVariantMediumLow:
    """
    MEDIUM and LOW are routed by batch size only: local below the
    threshold, external above it.  The same rule applies to both
    severities — the branch has no severity gate beyond the LOW/MEDIUM
    fallthrough.
    """

    def test_medium_small_batch_picks_local(self) -> None:
        atm = _bare_atm(ATMPolicy(local_max_samples=500))
        assert atm._select_variant(SIG_MEDIUM, n_samples=100) is MTPVariant.LOCAL

    def test_medium_large_batch_picks_external(self) -> None:
        atm = _bare_atm(ATMPolicy(local_max_samples=500))
        assert atm._select_variant(SIG_MEDIUM, n_samples=2_000) is MTPVariant.EXTERNAL

    def test_medium_boundary_inclusive(self) -> None:
        atm = _bare_atm(ATMPolicy(local_max_samples=500))
        assert atm._select_variant(SIG_MEDIUM, n_samples=500) is MTPVariant.LOCAL

    def test_medium_one_past_boundary_picks_external(self) -> None:
        atm = _bare_atm(ATMPolicy(local_max_samples=500))
        assert atm._select_variant(SIG_MEDIUM, n_samples=501) is MTPVariant.EXTERNAL

    def test_low_small_batch_picks_local(self) -> None:
        """Empty-reasons signal (LOW) routes by batch size too."""
        atm = _bare_atm(ATMPolicy(local_max_samples=500))
        assert atm._select_variant(SIG_LOW, n_samples=10) is MTPVariant.LOCAL

    def test_low_large_batch_picks_external(self) -> None:
        atm = _bare_atm(ATMPolicy(local_max_samples=500))
        assert atm._select_variant(SIG_LOW, n_samples=10_000) is MTPVariant.EXTERNAL

    def test_operator_request_alone_is_medium_not_low(self) -> None:
        """
        ``[OPERATOR_REQUEST]`` has a non-empty reasons list so severity
        resolves to MEDIUM — routing follows the batch-size rule, not
        the empty-reasons LOW path.  We still land in MTP-L for a
        small batch, but the selection came through the MEDIUM branch,
        which matters for callers who key on ``signal.severity()`` in
        parallel logs.
        """
        sig = _signal([TriggerReason.OPERATOR_REQUEST])
        assert sig.severity() == "MEDIUM"
        atm = _bare_atm(ATMPolicy(local_max_samples=500))
        assert atm._select_variant(sig, n_samples=100) is MTPVariant.LOCAL


# ---------------------------------------------------------------------------
# TestATMSelectVariant_PolicyInteractions — cross-cutting edge cases
# ---------------------------------------------------------------------------

class TestATMSelectVariantPolicyInteractions:
    """Cross-cutting cases where multiple policy knobs interact."""

    def test_tiny_local_max_forces_escalation_on_medium(self) -> None:
        """A conservative ``local_max_samples`` pushes even small batches up."""
        atm = _bare_atm(ATMPolicy(local_max_samples=10))
        assert atm._select_variant(SIG_MEDIUM, n_samples=11) is MTPVariant.EXTERNAL

    def test_very_large_local_max_keeps_critical_local(self) -> None:
        """A generous ``local_max_samples`` keeps big CRITICAL batches on MTP-L."""
        policy = ATMPolicy(
            critical_always_local_first=True,
            local_max_samples=100_000,
        )
        atm = _bare_atm(policy)
        assert atm._select_variant(SIG_CRITICAL, n_samples=5_000) is MTPVariant.LOCAL

    def test_prefer_external_overrides_medium_small_batch(self) -> None:
        """
        Normal routing for MEDIUM + tiny batch is MTP-L, but the
        operator can pin MTP-E for quality reasons (e.g. a compliance
        run that must produce an MLflow audit trail).
        """
        atm = _bare_atm(ATMPolicy(prefer_variant=MTPVariant.EXTERNAL))
        assert atm._select_variant(SIG_MEDIUM, n_samples=10) is MTPVariant.EXTERNAL


# ---------------------------------------------------------------------------
# TestATMSelectVariant_ConceptPoisoning — second CRITICAL reason code
# ---------------------------------------------------------------------------

class TestATMSelectVariantConceptPoisoning:
    """
    ``severity()`` maps TWO TriggerReasons to CRITICAL:
    ``DATA_POISONING`` (caught by DPD) and ``CONCEPT_POISONING`` (caught
    by CPD).  The rest of the CRITICAL-branch tests above use
    ``DATA_POISONING``; this class pins the ``CONCEPT_POISONING`` side
    of the enum so a refactor that silently drops CPD from the CRITICAL
    set (e.g. by accidentally comparing against only ``DATA_POISONING``)
    is caught immediately.
    """

    SIG_CONCEPT_POISON = _signal([TriggerReason.CONCEPT_POISONING])

    def test_concept_poisoning_has_critical_severity(self) -> None:
        """Sanity: the MToUTSignal severity contract still maps this to CRITICAL."""
        assert self.SIG_CONCEPT_POISON.severity() == "CRITICAL"

    def test_concept_poisoning_small_batch_routes_local_first(self) -> None:
        """Same fast-path as DATA_POISONING when the batch fits local."""
        policy = ATMPolicy(
            critical_always_local_first=True,
            local_max_samples=500,
        )
        atm = _bare_atm(policy)
        assert atm._select_variant(
            self.SIG_CONCEPT_POISON, n_samples=100
        ) is MTPVariant.LOCAL

    def test_concept_poisoning_large_batch_escalates_external(self) -> None:
        """Above the local budget, CONCEPT_POISONING escalates to MTP-E."""
        policy = ATMPolicy(
            critical_always_local_first=True,
            local_max_samples=500,
        )
        atm = _bare_atm(policy)
        assert atm._select_variant(
            self.SIG_CONCEPT_POISON, n_samples=501
        ) is MTPVariant.EXTERNAL

    def test_concept_poisoning_without_local_first_always_external(self) -> None:
        """Same override semantics as DATA_POISONING — local-first disabled → MTP-E."""
        policy = ATMPolicy(
            critical_always_local_first=False,
            local_max_samples=500,
        )
        atm = _bare_atm(policy)
        assert atm._select_variant(
            self.SIG_CONCEPT_POISON, n_samples=10
        ) is MTPVariant.EXTERNAL

    def test_both_poisoning_reasons_still_route_to_local_first(self) -> None:
        """
        The paper's severity contract is OR-based: presence of EITHER
        ``DATA_POISONING`` or ``CONCEPT_POISONING`` elevates to CRITICAL.
        Signals can legitimately carry both (co-fired DPD+CPD on the same
        step).  Pin that combined-signal path — it must not be routed as
        HIGH (the ``DATA_DRIFT + CONCEPT_DRIFT`` path) by mistake.
        """
        combined = _signal([
            TriggerReason.DATA_POISONING,
            TriggerReason.CONCEPT_POISONING,
        ])
        assert combined.severity() == "CRITICAL"
        policy = ATMPolicy(
            critical_always_local_first=True,
            local_max_samples=500,
        )
        atm = _bare_atm(policy)
        assert atm._select_variant(combined, n_samples=100) is MTPVariant.LOCAL


# ---------------------------------------------------------------------------
# TestATMSelectVariant_NeverAutoCloud — MTP-C is operator-only
# ---------------------------------------------------------------------------

class TestATMSelectVariantNeverAutoCloud:
    """
    ``MTP-C`` (cloud) is reserved for operators who explicitly pin
    ``prefer_variant=MTPVariant.CLOUD`` — the automatic rule NEVER
    selects it.

    Rationale: the paper's decision table treats MTP-C as a deployment-
    specific choice (operator centralised training platform), not a
    property the ATM can derive from signal severity and batch size
    alone.  Choosing MTP-C has cost / data-sovereignty / latency
    implications the ATM cannot evaluate; only the operator can.

    These tests exhaustively exercise the automatic rule across severity
    × batch-size to prove CLOUD never leaks in.  A refactor that adds a
    CLOUD case to ``_select_variant`` without also touching ATMPolicy
    docs will be caught here.
    """

    @pytest.mark.parametrize(
        "signal",
        [SIG_LOW, SIG_MEDIUM, SIG_HIGH, SIG_CRITICAL],
        ids=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
    )
    @pytest.mark.parametrize("n_samples", [1, 50, 499, 500, 501, 10_000])
    def test_auto_select_never_picks_cloud(
        self, signal: MToUTSignal, n_samples: int
    ) -> None:
        """No combination of (severity, batch) in the auto rule yields CLOUD."""
        atm = _bare_atm(ATMPolicy(
            prefer_variant=None,          # auto rule active
            critical_always_local_first=True,
            local_max_samples=500,
        ))
        variant = atm._select_variant(signal, n_samples=n_samples)
        assert variant is not MTPVariant.CLOUD, (
            f"auto rule leaked CLOUD for severity={signal.severity()}, "
            f"n_samples={n_samples}"
        )
        assert variant in {MTPVariant.LOCAL, MTPVariant.EXTERNAL}

    def test_auto_select_never_picks_cloud_even_with_local_first_off(
        self,
    ) -> None:
        """
        Disabling ``critical_always_local_first`` removes the LOCAL option
        on CRITICAL — but the fallback is MTP-E, never MTP-C.
        """
        atm = _bare_atm(ATMPolicy(
            prefer_variant=None,
            critical_always_local_first=False,
        ))
        for sig in (SIG_LOW, SIG_MEDIUM, SIG_HIGH, SIG_CRITICAL):
            variant = atm._select_variant(sig, n_samples=100)
            assert variant is not MTPVariant.CLOUD
            assert variant in {MTPVariant.LOCAL, MTPVariant.EXTERNAL}

    def test_prefer_cloud_is_the_only_path_to_cloud(self) -> None:
        """
        Positive confirmation: ``prefer_variant=CLOUD`` is the single
        path that yields MTP-C.  Combined with the auto-rule sweeps
        above, this exhaustively pins the CLOUD entry point.
        """
        atm = _bare_atm(ATMPolicy(prefer_variant=MTPVariant.CLOUD))
        # Across every severity × batch size the override must still win
        for sig in (SIG_LOW, SIG_MEDIUM, SIG_HIGH, SIG_CRITICAL):
            for n in (1, 100, 500, 10_000):
                assert atm._select_variant(sig, n_samples=n) is MTPVariant.CLOUD
