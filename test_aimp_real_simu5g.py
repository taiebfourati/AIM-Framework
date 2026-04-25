"""
test_aimp_real_simu5g.py
========================

End-to-end integration test that feeds **real Simu5G output** (not synthetic)
through the full AIMP pipeline.

Data pipeline
-------------
1. `simu5g_real_simulation_results.csv` is produced by
   `sim_parser/build_real_kpi_csvs.py` from opp_scavetool CSV exports of the
   `rtp_observer.ini` run (configs: RTP_Stable / RTP_Drift / RTP_Poisoning).

2. Columns used as the RTP feature vector:
        [rsrp_dbm, sinr_db, throughput_mbps, delay_ms]

3. Ground-truth handover label comes from Simu5G's `servingCell` vector
   change (`handover_flag`), augmented with the A3-event threshold for rows
   that have no handover signal logged (UEs that never handed over).

Flow
----
* Phase STABLE  -> reference / training data for the initial model.
* Phase DRIFT   -> streamed one sample at a time through `rtp.observe()`.
* Phase POISON  -> streamed after drift; 3σ channel anomalies + high-load
                   interference from bgCells should trip DPD and DDD.
* Phase RECOVERY-> emulated by replaying STABLE after POISON (since the
                   INI does not include a separate recovery config).

AIMP then composes MTPSpec, routes retraining via ATM, validates with NDT,
stores model versions in the ModelRepository, and reconfigures RTP.

Run
---
    py -3.13 test_aimp_real_simu5g.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aimp import AIMP, AIMPPolicy, RTPComposer, MTPComposer  # noqa: E402
from atm.atm import ATMPolicy, MTPVariant  # noqa: E402

log = logging.getLogger("aimp_real_simu5g")

# ---------------------------------------------------------------------------
# Constants matching rtp_observer.ini / enhanced_simulation.py
# ---------------------------------------------------------------------------

FEATURE_COLS = ["rsrp_dbm", "sinr_db", "throughput_mbps", "delay_ms"]

# 3GPP ranges used for clipping (Simu5G's clean-channel SINR can exceed 40 dB)
RSRP_RANGE = (-156.0, -31.0)
SINR_RANGE = (-23.0, 40.0)
TPUT_RANGE = (0.0, 1000.0)
LAT_RANGE  = (1.0, 100.0)


def a3_label(rsrp: float, sinr: float, lat_ms: float) -> int:
    """Fallback A3-event handover label (matches the synthetic test)."""
    return int((rsrp < -100.0 and sinr < 5.0) or lat_ms > 50.0)


def load_real_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"phase", "run", "ue", "t",
            "rsrp_dbm", "sinr_db", "throughput_mbps", "delay_ms",
            "handover_flag"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"missing columns in {path}: {missing}")

    # Clip to 3GPP ranges
    df["rsrp_dbm"]        = df["rsrp_dbm"].clip(*RSRP_RANGE)
    df["sinr_db"]         = df["sinr_db"].clip(*SINR_RANGE)
    df["throughput_mbps"] = df["throughput_mbps"].clip(*TPUT_RANGE)
    # delay_ms can be NaN for UEs that received no packets — drop those rows
    df = df.dropna(subset=["delay_ms"])
    df["delay_ms"] = df["delay_ms"].clip(*LAT_RANGE)

    # Final label: real handover OR A3 fallback
    fallback = df.apply(
        lambda r: a3_label(r["rsrp_dbm"], r["sinr_db"], r["delay_ms"]),
        axis=1,
    )
    df["label"] = (df["handover_flag"].fillna(0).astype(int) | fallback).astype(int)
    return df


def phase_slice(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    """Return rows for one phase, stable ordered by (run, ue, t)."""
    return (
        df[df["phase"] == phase]
        .sort_values(["run", "ue", "t"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    csv_path: Path,
    seed: int = 42,
    max_per_phase: int | None = None,
) -> dict:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("=" * 78)
    log.info(" AIMP × REAL Simu5G integration test — seed=%d", seed)
    log.info("=" * 78)

    # ── 1. Load ────────────────────────────────────────────────────────
    log.info("Loading real Simu5G CSV: %s", csv_path)
    df = load_real_csv(csv_path)
    log.info("  rows=%d, phases=%s", len(df), sorted(df["phase"].unique()))

    stable = phase_slice(df, "stable")
    drift  = phase_slice(df, "drift")
    poison = phase_slice(df, "poison")

    if max_per_phase is not None:
        stable = stable.iloc[:max_per_phase]
        drift  = drift.iloc[:max_per_phase]
        poison = poison.iloc[:max_per_phase]

    log.info("  stable=%d | drift=%d | poison=%d",
             len(stable), len(drift), len(poison))

    # ── 2. Train initial model on stable-phase reference ───────────────
    # Use first 60% of stable as training ref, last 40% as stream warm-up.
    split = int(0.6 * len(stable))
    ref = stable.iloc[:split]
    X_ref = ref[FEATURE_COLS].to_numpy(dtype=np.float64)
    y_ref = ref["label"].to_numpy(dtype=np.int64)
    # Force both classes (real stable data is mostly class 0)
    if y_ref.sum() == 0:
        y_ref[:max(50, len(y_ref) // 20)] = 1

    clf = RandomForestClassifier(n_estimators=50, random_state=seed)
    clf.fit(X_ref, y_ref)
    acc = clf.score(X_ref, y_ref)
    log.info("Initial RF(50) trained on %d real stable samples — acc=%.3f",
             len(X_ref), acc)

    # ── 3. Instantiate AIMP (MTP-L only — no MLflow, no cloud) ─────────
    policy = AIMPPolicy(
        atm_policy=ATMPolicy(
            prefer_variant=MTPVariant.LOCAL,     # honoured by MTPC now
            use_ndt=True,
            ndt_min_accuracy=0.70,
            auto_deploy=True,
            critical_always_local_first=True,
        ),
        rtp_profile_name="classifier_default",
        cost_limit=1.5,
        reconfigure_rtp_on_model_change=True,
    )
    aimp = AIMP(policy=policy, rtpc=RTPComposer(), mtpc=MTPComposer())
    log.info("AIMP instantiated (MTPC.policy.prefer_variant=%s)",
             aimp.mtpc.policy.prefer_variant)

    # ── 4. Register AIF ─────────────────────────────────────────────────
    aif, rtp, atm = aimp.register_aif(
        estimator=clf,
        X_ref=X_ref,
        y_ref=y_ref,
    )
    log.info("Registered AIF on real data. AIF=%s, RTP=%s, ATM=%s",
             type(aif).__name__, type(rtp).__name__, type(atm).__name__)

    # ── 5. Install MToUT spy ────────────────────────────────────────────
    event_log: list[dict] = []
    original_handler = rtp._on_mtout

    def _spy(signal):
        event_log.append({
            "step":     signal.step,
            "severity": signal.severity(),
            "reasons":  [r.name for r in signal.reasons],
        })
        log.info("  step=%d  MToUT[%s]  reasons=%s",
                 signal.step, signal.severity(),
                 [r.name for r in signal.reasons])
        if original_handler is not None:
            original_handler(signal)

    rtp._on_mtout = _spy

    # ── 6. Stream phases through rtp.observe() ──────────────────────────
    warmup = stable.iloc[split:]   # remaining stable rows
    stream_order = [
        ("stable (warmup)", warmup),
        ("drift",           drift),
        ("poison",          poison),
        ("recovery",        warmup.sample(frac=1.0, random_state=seed)),
    ]

    total = sum(len(p) for _, p in stream_order)
    step_global = 0
    for phase_name, phase_df in stream_order:
        log.info("--- streaming phase=%s (%d samples) ---",
                 phase_name, len(phase_df))
        for _, row in phase_df.iterrows():
            x = row[FEATURE_COLS].to_numpy(dtype=np.float64)
            y_true = int(row["label"])
            rtp.observe(x, y_true=y_true)
            step_global += 1
            if step_global % 2000 == 0:
                log.info("  progress: %d/%d", step_global, total)

    # ── 7. Summarise ────────────────────────────────────────────────────
    log.info("\n" + "=" * 78)
    log.info(" RESULTS (real Simu5G)")
    log.info("=" * 78)

    hist = aimp.get_model_history(aif_id=1)
    log.info("ModelRepository: %d versions", len(hist))
    for e in hist:
        log.info("  v%d  source=%s  meta=%s",
                 e["version"], e["source"], e["metadata"])

    for i, r in enumerate(atm.history):
        log.info("  retrain #%d  status=%s  variant=%s  deployed=%s  "
                 "ndt_passed=%s  dur=%.2fs  msg=%s",
                 i, r.status,
                 r.variant_used.value if r.variant_used else None,
                 r.deployed, r.ndt_passed, r.duration_s, r.message)

    summary = {
        "samples_processed":  step_global,
        "mtout_triggers":     len(event_log),
        "retrainings":        len(atm.history),
        "deployed_versions":  sum(1 for r in atm.history if r.deployed),
        "ndt_passes":         sum(1 for r in atm.history if r.ndt_passed),
        "model_versions":     len(hist),
    }
    log.info("\nSummary: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path,
                   default=Path(__file__).resolve().parent /
                           "simu5g_real_simulation_results.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-per-phase", type=int, default=None,
                   help="Cap each phase to N samples (fast smoke test).")
    args = p.parse_args(argv)

    if not args.csv.exists():
        print(f"ERROR: real-sim CSV not found: {args.csv}", file=sys.stderr)
        print("Run sim_parser/build_real_kpi_csvs.py first.", file=sys.stderr)
        return 2

    summary = run(args.csv, seed=args.seed, max_per_phase=args.max_per_phase)

    # Minimal assertions
    assert summary["samples_processed"] > 0
    assert summary["mtout_triggers"] >= 1, (
        "RTP should fire at least one MToUT on the drift/poison transition"
    )
    assert summary["retrainings"] >= 1, (
        "ATM should handle at least one retraining"
    )

    print("\n" + "=" * 60)
    print("  AIMP × REAL Simu5G INTEGRATION TEST PASSED")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:25s} = {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
