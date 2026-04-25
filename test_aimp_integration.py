"""
test_aimp_integration.py — Integration test: AIMP against a real 5G NR simulation.

This exercises the full Paper 3 AIMP architecture end-to-end:
  1. AIMP (facade) registers an AIF with RTPC-composed RTP + ATM + NDT
  2. A multi-phase 5G NR stream (stable → drift → poisoning → recovery) is fed
     to RTP.observe() sample by sample
  3. RTP detectors fire MToUT signals → AIMP intercepts → MTPC composes MTPSpec
     → ATM handles → NDT validates → deploy / rollback
  4. AIMP stores every deployed model version in the ModelRepository and every
     retraining snapshot in the TrainingDataRepository
  5. After model updates, AIMP uses RTPC.reconfigure() to refit detector
     references (the key feature from Paper 3 Section IV-A)

This is intentionally close to `enhanced_simulation.py` but wires through AIMP
instead of constructing RTP/ATM directly, demonstrating that the new layer is
a drop-in replacement for the manual wiring used in the thesis experiments.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Make sure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from aimp import AIMP, AIMPPolicy, RTPComposer, MTPComposer  # noqa: E402
from atm.atm import ATMPolicy  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("aimp_integration_test")

# ---------------------------------------------------------------------------
# 5G NR stream generator — identical to enhanced_simulation.py
# ---------------------------------------------------------------------------

RSRP_RANGE = (-156.0, -31.0)
SINR_RANGE = (-23.0, 40.0)
THROUGHPUT_RANGE = (0.0, 1000.0)
LATENCY_RANGE = (1.0, 100.0)


def gen_stable_sample(rng: np.random.Generator) -> np.ndarray:
    """Good-channel 5G NR sample (UE near gNB)."""
    rsrp = rng.normal(-85, 5)
    sinr = rng.normal(18, 4)
    tput = rng.normal(450, 80)
    lat = rng.normal(12, 3)
    return np.clip(
        [rsrp, sinr, tput, lat],
        [RSRP_RANGE[0], SINR_RANGE[0], THROUGHPUT_RANGE[0], LATENCY_RANGE[0]],
        [RSRP_RANGE[1], SINR_RANGE[1], THROUGHPUT_RANGE[1], LATENCY_RANGE[1]],
    )


def gen_drifted_sample(rng: np.random.Generator) -> np.ndarray:
    """Concept-drift sample — UE at cell edge, handover conditions."""
    rsrp = rng.normal(-105, 7)
    sinr = rng.normal(3, 3)
    tput = rng.normal(80, 30)
    lat = rng.normal(55, 15)
    return np.clip(
        [rsrp, sinr, tput, lat],
        [RSRP_RANGE[0], SINR_RANGE[0], THROUGHPUT_RANGE[0], LATENCY_RANGE[0]],
        [RSRP_RANGE[1], SINR_RANGE[1], THROUGHPUT_RANGE[1], LATENCY_RANGE[1]],
    )


def gen_poisoned_sample(rng: np.random.Generator) -> np.ndarray:
    """3σ-displaced feature vector mimicking a label-flipping attack."""
    rsrp = rng.normal(-60, 5)   # Impossibly good RSRP
    sinr = rng.normal(35, 2)    # Near-max SINR
    tput = rng.normal(950, 30)  # Near gigabit
    lat = rng.normal(3, 1)      # Sub-5ms
    return np.clip(
        [rsrp, sinr, tput, lat],
        [RSRP_RANGE[0], SINR_RANGE[0], THROUGHPUT_RANGE[0], LATENCY_RANGE[0]],
        [RSRP_RANGE[1], SINR_RANGE[1], THROUGHPUT_RANGE[1], LATENCY_RANGE[1]],
    )


def ho_label(x: np.ndarray) -> int:
    """A3-event + latency-budget decision boundary."""
    rsrp, sinr, tput, lat = x
    return int((rsrp < -100 and sinr < 5) or lat > 50)


# ---------------------------------------------------------------------------
# Main integration test
# ---------------------------------------------------------------------------

def run_integration_test(seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    log.info("=" * 78)
    log.info(" AIMP Integration Test — seed=%d", seed)
    log.info("=" * 78)

    # ── 1. Generate reference data & train initial model ────────────────
    log.info("Phase 0: generating clean reference data (500 samples)...")
    X_ref = np.array([gen_stable_sample(rng) for _ in range(500)])
    y_ref = np.array([ho_label(x) for x in X_ref])
    # Ensure both classes present (A3 on clean stable data usually labels all 0)
    y_ref[:50] = 1  # inject synthetic handover positives for training

    clf = RandomForestClassifier(n_estimators=50, random_state=seed)
    clf.fit(X_ref, y_ref)
    log.info("Initial model: RF(50 trees), train acc=%.3f", clf.score(X_ref, y_ref))

    # ── 2. Instantiate AIMP ─────────────────────────────────────────────
    log.info("\n--- Instantiating AIMP ---")
    # Force MTP-L so we do not need an MLflow server running for this test.
    from atm.atm import MTPVariant as _MTPV  # local import to avoid top-level clash
    policy = AIMPPolicy(
        atm_policy=ATMPolicy(
            prefer_variant=_MTPV.LOCAL,
            use_ndt=True,
            ndt_min_accuracy=0.70,
            auto_deploy=True,
            critical_always_local_first=True,
        ),
        rtp_profile_name="classifier_default",
        cost_limit=1.5,
        reconfigure_rtp_on_model_change=True,
    )
    rtpc = RTPComposer()
    mtpc = MTPComposer()  # default MTP-L + MTP-E
    aimp = AIMP(policy=policy, rtpc=rtpc, mtpc=mtpc)
    log.info("AIMP instantiated with default RTPC, MTPC, NDT, repositories.")

    # ── 3. Register AIF ─────────────────────────────────────────────────
    log.info("\n--- Registering AIF with AIMP ---")
    aif, rtp, atm = aimp.register_aif(
        estimator=clf,
        X_ref=X_ref,
        y_ref=y_ref,
    )
    log.info("Registered: AIF=%s, RTP=%s, ATM=%s",
             type(aif).__name__, type(rtp).__name__, type(atm).__name__)

    # ── 4. Feed multi-phase stream ──────────────────────────────────────
    log.info("\n--- Feeding 4-phase 5G stream through AIMP-managed RTP ---")
    phase_counts = {"stable": 300, "drift": 300, "poison": 200, "recovery": 200}

    event_log: list[dict] = []

    def stream_samples():
        yield from (("stable", gen_stable_sample(rng)) for _ in range(phase_counts["stable"]))
        yield from (("drift", gen_drifted_sample(rng)) for _ in range(phase_counts["drift"]))
        yield from (("poison", gen_poisoned_sample(rng)) for _ in range(phase_counts["poison"]))
        yield from (("recovery", gen_stable_sample(rng)) for _ in range(phase_counts["recovery"]))

    # Wrap AIMP's MToUT handler so we can also log signals in this test
    aimp_handler = rtp._on_mtout
    def _spy_mtout(signal):
        event_log.append({
            "step": signal.step,
            "reasons": [r.name for r in signal.reasons],
            "severity": signal.severity(),
        })
        log.info("  step=%d MToUT: reasons=%s severity=%s",
                 signal.step,
                 [r.name for r in signal.reasons],
                 signal.severity())
        if aimp_handler is not None:
            aimp_handler(signal)
    rtp._on_mtout = _spy_mtout

    total = sum(phase_counts.values())
    for step, (phase, x) in enumerate(stream_samples()):
        y_true = ho_label(x)
        rtp.observe(x, y_true=y_true)
        if step % 200 == 0 and step > 0:
            log.info("  progress: %d/%d samples processed", step, total)

    mtout_count = len(event_log)

    # ── 5. Collect results ──────────────────────────────────────────────
    log.info("\n" + "=" * 78)
    log.info(" RESULTS")
    log.info("=" * 78)

    status = aimp.status()
    log.info("AIMP status: %s", status)

    hist = aimp.get_model_history(aif_id=1)
    log.info("ModelRepository: %d model versions stored", len(hist))
    for i, entry in enumerate(hist):
        src = entry.source_variant
        src_str = src.value if hasattr(src, "value") else str(src)
        log.info("  v%d: source=%s, meta=%s", entry.version, src_str,
                 entry.metadata)

    atm_hist = atm.history
    log.info("ATM.history: %d retraining events", len(atm_hist))
    for i, r in enumerate(atm_hist):
        log.info("  #%d: status=%s variant=%s deployed=%s ndt_passed=%s "
                 "duration=%.2fs reason='%s'",
                 i, r.status, r.variant_used.value if r.variant_used else None,
                 r.deployed, r.ndt_passed, r.duration_s, r.message)

    log.info("Total MToUT triggers: %d", mtout_count)

    summary = {
        "samples_processed": total,
        "mtout_triggers": mtout_count,
        "retrainings": len(atm_hist),
        "deployed_versions": len([r for r in atm_hist if r.deployed]),
        "ndt_passes": len([r for r in atm_hist if r.ndt_passed]),
        "model_versions": len(hist),
    }

    log.info("\nSummary: %s", summary)
    return summary


if __name__ == "__main__":
    summary = run_integration_test(seed=42)

    # Minimal assertions
    assert summary["samples_processed"] == 1000, "stream length wrong"
    assert summary["mtout_triggers"] >= 1, (
        "RTP should trigger at least one MToUT when phase switches to drift/poison"
    )
    assert summary["retrainings"] >= 1, (
        "ATM should handle at least one retraining when MToUTs fire"
    )

    print("\n" + "=" * 60)
    print("  AIMP INTEGRATION TEST PASSED")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:25s} = {v}")
