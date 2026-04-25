"""
scripts/repro_audit_dump_telemetry.py — capture full simulation telemetry.

Re-runs the same 4-phase simulation that ``generate_figures.py`` drives, but
instead of plotting it dumps the raw per-check detector telemetry, the event
log, the ATM cycle outcomes and the NDT history to a JSON snapshot under
``thesis/.repro_audit_data/``.

This lets us:
  1. Verify what the code actually computes (is CPD ever ready? do its
     three sub-signals ever become non-zero? at which steps?)
  2. Hand the snapshot to a separate plotter so the figures are demonstrably
     a function of the captured data — not an interpretation of it.
  3. Diff future runs against the snapshot to confirm bit-level reproducibility
     of the simulation, not just the rendered PDF.

Usage:
    python scripts/repro_audit_dump_telemetry.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

# Re-use the simulation driver so we capture exactly what generate_figures.py
# captures — same seeds, same config, same patch.
from generate_figures import run_simulation  # noqa: E402

OUT_DIR = ROOT / "thesis" / ".repro_audit_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _safe(v):
    """Make a value JSON-serialisable."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, dict):
        return {str(k): _safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_safe(x) for x in v]
    if is_dataclass(v):
        return _safe(asdict(v))
    # Enums, custom objects — fall back to repr
    if hasattr(v, "name"):
        return v.name
    return repr(v)


def main() -> int:
    print("=" * 60)
    print("Telemetry capture")
    print("=" * 60)

    print("\n[1/3] Running 4-phase simulation ...")
    data = run_simulation()
    n_steps = len(data["step_records"])
    n_checks = len(data["check_records"])
    n_events = len(data["event_log"])
    n_atm = len(data["atm_results"])
    n_ndt = len(data["ndt_history"])
    print(f"      Steps: {n_steps} | Checks: {n_checks} | "
          f"Events: {n_events} | ATM cycles: {n_atm} | NDT: {n_ndt}")

    # ── CPD-specific diagnostics ─────────────────────────────────────────────
    print("\n[2/3] CPD readiness diagnostics:")
    cpd_zero = 0
    cpd_nonzero = 0
    cpd_first_nonzero_step = None
    for r in data["check_records"]:
        # "ready" is heuristic: shadow_div, ks_pvalue, or corr_delta_max
        # all at the not-ready defaults (0.0 / 1.0 / 0.0) means CPD likely
        # returned _not_ready.
        is_default = (
            r["cpd_shadow_div"] == 0.0
            and r["cpd_ks_pvalue"] == 1.0
            and r["cpd_corr_delta"] == 0.0
        )
        if is_default:
            cpd_zero += 1
        else:
            cpd_nonzero += 1
            if cpd_first_nonzero_step is None:
                cpd_first_nonzero_step = r["step"]
    print(f"      CPD checks at default (not-ready or all-quiet): {cpd_zero}/{n_checks}")
    print(f"      CPD checks with non-default values:             {cpd_nonzero}/{n_checks}")
    if cpd_first_nonzero_step is not None:
        print(f"      First step with non-default CPD output:        {cpd_first_nonzero_step}")
    else:
        print(f"      CPD never produced non-default output across the run.")

    # Print a per-check sample so we can SEE the values
    print("\n      CPD per-check sample (first 6 + last 4):")
    sample_idx = list(range(min(6, n_checks))) + list(range(max(0, n_checks - 4), n_checks))
    sample_idx = sorted(set(sample_idx))
    for i in sample_idx:
        r = data["check_records"][i]
        print(
            f"      step={r['step']:>3d}  "
            f"shadow={r['cpd_shadow_div']:.3f}  "
            f"ks_p={r['cpd_ks_pvalue']:.3e}  "
            f"corr={r['cpd_corr_delta']:.3f}  "
            f"poison={r['cpd_poison']}"
        )

    # ── Detector trigger summary ─────────────────────────────────────────────
    ddd_fires = sum(1 for r in data["check_records"] if r["ddd_drift"])
    dpd_fires = sum(1 for r in data["check_records"] if r["dpd_poison"])
    cdd_fires = sum(1 for r in data["check_records"] if r["cdd_drift"])
    cpd_fires = sum(1 for r in data["check_records"] if r["cpd_poison"])
    print(f"\n      Detector trigger counts (out of {n_checks} checks):")
    print(f"        DDD: {ddd_fires}   DPD: {dpd_fires}   CDD: {cdd_fires}   CPD: {cpd_fires}")

    # ── Persist snapshot ─────────────────────────────────────────────────────
    print("\n[3/3] Writing JSON snapshot ...")

    snapshot = {
        "summary": {
            "n_steps": n_steps,
            "n_checks": n_checks,
            "n_events": n_events,
            "n_atm_cycles": n_atm,
            "n_ndt_records": n_ndt,
            "ddd_fires": ddd_fires,
            "dpd_fires": dpd_fires,
            "cdd_fires": cdd_fires,
            "cpd_fires": cpd_fires,
        },
        "step_records": [_safe(r) for r in data["step_records"]],
        "check_records": [_safe(r) for r in data["check_records"]],
        "event_log": [
            {
                "step": _safe(getattr(ev, "step", None)),
                "event_type": _safe(getattr(ev, "event_type", None)),
                "details": _safe(getattr(ev, "details", None)),
            }
            for ev in data["event_log"]
        ],
        "atm_results": [
            {
                "variant_used": _safe(getattr(r, "variant_used", None)),
                "ndt_passed": _safe(getattr(r, "ndt_passed", None)),
                "deployed": _safe(getattr(r, "deployed", None)),
                "n_train": _safe(getattr(r, "n_train", None)),
                "duration_s": _safe(getattr(r, "duration_s", None)),
                "metadata": _safe(getattr(r, "metadata", None)),
            }
            for r in data["atm_results"]
        ],
        "ndt_history": [_safe(r) for r in data["ndt_history"]],
    }

    out_path = OUT_DIR / "simulation_telemetry.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    print(f"      Wrote {out_path} ({out_path.stat().st_size/1024:.1f} KB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
