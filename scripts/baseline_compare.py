"""
scripts/baseline_compare.py — CRIT-7 baseline comparison.

Compares the four production detectors (DDD, DPD, CDD, CPD) against
three external baselines on the SAME Campaign~A data stream:

    * ADWIN [Bifet & Gavaldà, SDM 2007]                — drift family
    * Spectral Signatures [Tran et al., NeurIPS 2018]  — poisoning family
    * CUSUM on class prior [Page, 1954]                — poisoning family

The harness is deliberately apples-to-apples:

  1. We re-create the EXACT same data flow ``generate_figures.run_simulation``
     drives (same RNG seed, same ``make_data`` distribution, same phase
     boundaries, same 50-step check interval).  The production detectors
     are exercised by re-running ``run_simulation`` and harvesting its
     telemetry; the baselines are exercised by streaming the same X/y
     samples one step at a time through their own ``update`` methods
     and recording their decisions at each check boundary.
  2. Ground truth is phase-resolved: a check is labelled "drift" iff
     its 50-step window overlaps Phase 2 (steps 401--600); "poisoning"
     iff its window overlaps Phase 3 (601--700); "clean" otherwise.
  3. For each (detector, family) we compute precision / recall / F1
     against the family's ground-truth label at the check level.

The output is dumped to ``thesis/.repro_audit_data/baseline_comparison.json``
so a separate plotter can render the comparison figure without having
to re-run the simulation.

Usage:
    python scripts/baseline_compare.py
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from dataclasses import asdict, is_dataclass
from pathlib import Path

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from generate_figures import run_simulation, PHASE_RANGES  # noqa: E402
from baselines.adwin import ADWIN  # noqa: E402
from baselines.spectral_signatures import SpectralSignatures  # noqa: E402
from baselines.cusum_class_prior import CUSUMClassPrior  # noqa: E402

OUT_DIR = ROOT / "thesis" / ".repro_audit_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Phase boundaries (must match generate_figures.run_simulation())
PHASE_RANGES_LOCAL = {
    1: (1, 400),
    2: (401, 600),    # drift
    3: (601, 700),    # poisoning
    4: (701, 900),
}

CHECK_INTERVAL = 50           # same as RTPConfig.check_interval
N_STEPS = 900                 # 400 + 200 + 100 + 200
WINDOW_HALF = CHECK_INTERVAL  # we declare a check fires "for" the
                              # 50 steps preceding it


# ============================================================================
# 1. Re-create the exact same data stream the production sim uses
# ============================================================================

def build_stream():
    """Replay the exact same ``make_data`` flow that
    ``generate_figures.run_simulation`` constructs, with the same seed,
    same noise levels, same Phase 3 injection rule.

    Returns a length-N_STEPS list of (X_step, y_step, phase) tuples.
    """
    rng = np.random.default_rng(0)

    def make_data(n, noise=0.05, shift=0.0):
        X = rng.normal(0, 1, size=(n, 4))
        y = ((X[:, 0] + X[:, 1]) > shift).astype(int)
        flip = rng.random(n) < noise
        y[flip] = 1 - y[flip]
        return X, y

    # ── Match generate_figures.run_simulation() exactly ─────────────────────
    # We must consume the RNG in the same order so that the per-phase
    # samples line up bit-for-bit with the production telemetry.
    # The sim itself burns rng on:
    #   1. initial classifier training (500 samples)
    _Xtr, _ytr = make_data(500, noise=0.05)
    #   2. reference window (300 samples)
    _Xref, _yref = make_data(300, noise=0.05)
    # Now phases.
    X1, y1 = make_data(400, noise=0.05)
    X2, y2 = make_data(200, noise=0.55)
    # Phase 3: 90 clean + 10 OOD injections, then permute.
    X_clean, y_clean = make_data(90, noise=0.05)
    X_inject = rng.uniform(30, 50, size=(10, 4))
    y_inject = rng.integers(0, 2, size=10)
    X3 = np.vstack([X_clean, X_inject])
    y3 = np.concatenate([y_clean, y_inject])
    idx = rng.permutation(len(X3))
    X3, y3 = X3[idx], y3[idx]
    X4, y4 = make_data(200, noise=0.05)

    stream = []
    for i in range(400):
        stream.append((X1[i], int(y1[i]), 1))
    for i in range(200):
        stream.append((X2[i], int(y2[i]), 2))
    for i in range(len(X3)):
        stream.append((X3[i], int(y3[i]), 3))
    for i in range(200):
        stream.append((X4[i], int(y4[i]), 4))
    assert len(stream) == N_STEPS
    return stream


# ============================================================================
# 2. Replay each baseline through the stream
# ============================================================================

def run_adwin(stream) -> dict:
    """ADWIN on feature 0 — the channel that carries the covariate
    shift in Phase 2 (and is the same channel the production DDD's KS
    arm checks)."""
    det = ADWIN(delta=0.002)
    decisions_per_step: list[bool] = []
    for x, _y, _ph in stream:
        flag = det.update(float(x[0]))
        decisions_per_step.append(flag)
    return {"name": "ADWIN", "family": "drift",
            "decisions": decisions_per_step}


def run_spectral(stream) -> dict:
    """Spectral signatures on the 4-D feature window."""
    det = SpectralSignatures(epsilon=0.10, min_window=80, max_window=300)
    decisions_per_step: list[bool] = []
    for x, _y, _ph in stream:
        flag = det.update(x)
        decisions_per_step.append(flag)
    return {"name": "SpectralSig", "family": "poisoning",
            "decisions": decisions_per_step}


def run_cusum(stream) -> dict:
    """CUSUM on the class prior."""
    # mu_0 = 0.5 because make_data's threshold is 0 → balanced classes.
    det = CUSUMClassPrior(mu_0=0.5, k=0.05, h=5.0)
    decisions_per_step: list[bool] = []
    for _x, y, _ph in stream:
        flag = det.update(int(y))
        decisions_per_step.append(flag)
    return {"name": "CUSUM-prior", "family": "poisoning",
            "decisions": decisions_per_step}


# ============================================================================
# 3. Roll per-step decisions into per-check decisions
# ============================================================================

def per_check_from_per_step(decisions_per_step):
    """Aggregate to the same 18 check intervals the production sim
    reports.  A check 'fires' if ANY step inside its preceding 50-step
    window flagged."""
    flags = []
    steps = []
    for c in range(1, N_STEPS // CHECK_INTERVAL + 1):
        end = c * CHECK_INTERVAL  # 50, 100, …, 900
        start = end - CHECK_INTERVAL  # 0, 50, …, 850 (inclusive of 0)
        win = decisions_per_step[start:end]
        flags.append(any(win))
        steps.append(end)
    return steps, flags


# ============================================================================
# 4. Phase-resolved ground truth + scoring
# ============================================================================

def gt_labels(family: str):
    """Per-check ground truth for a detector family.

    A check at step ``end`` is labelled positive iff its 50-step window
    overlaps the family's target phase:
      drift     -> Phase 2 (steps 401--600)
      poisoning -> Phase 3 (steps 601--700)

    All other checks are labelled negative (clean).
    """
    if family == "drift":
        target = PHASE_RANGES_LOCAL[2]
    elif family == "poisoning":
        target = PHASE_RANGES_LOCAL[3]
    else:
        raise ValueError(family)
    labels = []
    for c in range(1, N_STEPS // CHECK_INTERVAL + 1):
        end = c * CHECK_INTERVAL
        start = end - CHECK_INTERVAL + 1
        # Overlap test: window [start, end] vs target [tlo, thi]
        tlo, thi = target
        overlap = (start <= thi) and (end >= tlo)
        labels.append(bool(overlap))
    return labels


def score(decisions: list[bool], labels: list[bool]) -> dict:
    tp = sum(1 for d, y in zip(decisions, labels) if d and y)
    fp = sum(1 for d, y in zip(decisions, labels) if d and not y)
    fn = sum(1 for d, y in zip(decisions, labels) if (not d) and y)
    tn = sum(1 for d, y in zip(decisions, labels) if (not d) and (not y))
    p = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    r = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if (not (p != p)) and (not (r != r)) and (p + r) > 0:
        f1 = 2 * p * r / (p + r)
    else:
        f1 = float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": float(p), "recall": float(r), "f1": float(f1)}


# ============================================================================
# 5. Production-detector telemetry
# ============================================================================

def production_check_decisions():
    """Re-run the production simulation and return per-check decision
    arrays for DDD, DPD, CDD, CPD (length 18 each)."""
    print("  Re-running production simulation for DDD/DPD/CDD/CPD ...")
    data = run_simulation()
    cr = data["check_records"]
    return {
        "DDD": [bool(r["ddd_drift"])     for r in cr],
        "DPD": [bool(r["dpd_poison"])    for r in cr],
        "CDD": [bool(r["cdd_drift"])     for r in cr],
        "CPD": [bool(r["cpd_poison"])    for r in cr],
        "steps": [r["step"] for r in cr],
    }


# ============================================================================
# 6. Main
# ============================================================================

def main() -> int:
    print("=" * 60)
    print("CRIT-7 baseline comparison")
    print("=" * 60)
    print("\n[1/3] Building shared step stream (same RNG as Campaign~A) ...")
    stream = build_stream()
    print(f"      stream length: {len(stream)} steps")

    print("\n[2/3] Production detectors ...")
    prod = production_check_decisions()

    print("\n[3/3] Baselines ...")
    baselines = []
    for fn in (run_adwin, run_spectral, run_cusum):
        out = fn(stream)
        steps, per_check = per_check_from_per_step(out["decisions"])
        out["steps"] = steps
        out["per_check"] = per_check
        baselines.append(out)
        n_fires_step = sum(out["decisions"])
        n_fires_check = sum(per_check)
        print(
            f"      {out['name']:>14s}  "
            f"per-step fires: {n_fires_step:>3d}  "
            f"per-check fires: {n_fires_check}/{len(per_check)}"
        )

    # ── Score ────────────────────────────────────────────────────────────────
    print("\n--- Per-detector precision / recall / F1 (per check, n=18) ---")
    results = {}
    drift_gt = gt_labels("drift")
    poison_gt = gt_labels("poisoning")
    print(f"  ground truth:   drift+ checks: {sum(drift_gt)}   "
          f"poison+ checks: {sum(poison_gt)}")

    # Production
    for name, decisions, family in (
        ("DDD", prod["DDD"], "drift"),
        ("DPD", prod["DPD"], "poisoning"),
        ("CDD", prod["CDD"], "drift"),
        ("CPD", prod["CPD"], "poisoning"),
    ):
        gt = drift_gt if family == "drift" else poison_gt
        s = score(decisions, gt)
        results[name] = {**s, "family": family, "kind": "production"}
        print(
            f"    {name:>12s}  fam={family:<9s}  "
            f"P={s['precision']:.3f}  R={s['recall']:.3f}  F1={s['f1']:.3f}  "
            f"TP={s['tp']:>2d} FP={s['fp']:>2d} FN={s['fn']:>2d} TN={s['tn']:>2d}"
        )

    # Baselines
    for b in baselines:
        family = b["family"]
        gt = drift_gt if family == "drift" else poison_gt
        s = score(b["per_check"], gt)
        results[b["name"]] = {**s, "family": family, "kind": "baseline"}
        print(
            f"    {b['name']:>12s}  fam={family:<9s}  "
            f"P={s['precision']:.3f}  R={s['recall']:.3f}  F1={s['f1']:.3f}  "
            f"TP={s['tp']:>2d} FP={s['fp']:>2d} FN={s['fn']:>2d} TN={s['tn']:>2d}"
        )

    snapshot = {
        "n_checks": len(drift_gt),
        "ground_truth_drift": drift_gt,
        "ground_truth_poisoning": poison_gt,
        "production": {k: v for k, v in prod.items() if k != "steps"},
        "baselines": [
            {
                "name": b["name"],
                "family": b["family"],
                "per_check": b["per_check"],
                "n_step_fires": int(sum(b["decisions"])),
            }
            for b in baselines
        ],
        "results": results,
    }
    out_path = OUT_DIR / "baseline_comparison.json"
    out_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(
        f"\nWrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
