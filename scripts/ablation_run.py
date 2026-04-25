"""
scripts/ablation_run.py — per-detector leave-one-out ablation study.

Re-runs the same 4-phase Campaign~A simulation that ``generate_figures.py``
drives FIVE times:

    Variant 0:  baseline (all four detectors enabled)
    Variant 1:  no DDD   (data-drift detector disabled)
    Variant 2:  no DPD   (data-poisoning detector disabled)
    Variant 3:  no CDD   (concept-drift detector disabled)
    Variant 4:  no CPD   (concept-poisoning detector disabled)

For each variant we capture, in a single JSON snapshot, the per-detector
trigger counts, ATM cycle outcomes (variant_used / ndt_passed / deployed),
NDT history, the rollback / security-alert / MToUT event totals from the
event log, and a phase-by-phase rolling accuracy summary.  The full
event log is also persisted so that a downstream plotter can render any
view of "what changed when this detector was removed" without re-running
the simulation.

The point of the study (and the resulting figure) is *not* to claim
that each detector is individually optimal in isolation — it is to
show, on a single deterministic seed, what each detector contributes
to the framework as a whole: which incident class is no longer
detected, whether ATM cycles still run, whether the gate still
catches anything, and which event types disappear from the audit
trail.  That is the property a reviewer asked for in CRIT-7.

Usage:
    python scripts/ablation_run.py
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from generate_figures import run_simulation, PHASE_RANGES  # noqa: E402

OUT_DIR = ROOT / "thesis" / ".repro_audit_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ABLATIONS = [
    ("baseline", None),
    ("no_DDD", "DDD"),
    ("no_DPD", "DPD"),
    ("no_CDD", "CDD"),
    ("no_CPD", "CPD"),
]


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
    if hasattr(v, "name"):
        return v.name
    return repr(v)


def summarise_variant(name: str, data: dict) -> dict:
    """Reduce one simulation run to the small set of numbers the figure
    will visualise.

    The reduction is deliberate: the figure itself stays readable when
    the whole study is one panel per outcome category and one bar per
    variant.  All raw arrays are still persisted in the snapshot so a
    different plotter can re-derive any other view it likes."""

    check_records = data["check_records"]
    event_log = list(data["event_log"])
    atm_results = data["atm_results"]
    ndt_history = data["ndt_history"]
    step_records = data["step_records"]

    # ── Per-detector trigger counts ──────────────────────────────────────
    triggers = {
        "DDD": sum(1 for r in check_records if r["ddd_drift"]),
        "DPD": sum(1 for r in check_records if r["dpd_poison"]),
        "CDD": sum(1 for r in check_records if r["cdd_drift"]),
        "CPD": sum(1 for r in check_records if r["cpd_poison"]),
    }

    # ── Event-log breakdown by event type ────────────────────────────────
    event_counts = Counter(ev.event_type.name for ev in event_log)

    # Selected event totals for the figure
    selected = {
        "MTOUT_FIRED": int(event_counts.get("MTOUT_FIRED", 0)),
        "MTOUT_SUPPRESSED": int(event_counts.get("MTOUT_SUPPRESSED", 0)),
        "DATA_DRIFT": int(event_counts.get("DATA_DRIFT", 0)),
        "DATA_POISONING": int(event_counts.get("DATA_POISONING", 0)),
        "CONCEPT_DRIFT": int(event_counts.get("CONCEPT_DRIFT", 0)),
        "CONCEPT_POISONING": int(event_counts.get("CONCEPT_POISONING", 0)),
        "ROLLBACK": int(event_counts.get("ROLLBACK", 0)),
        "SECURITY_ALERT": int(event_counts.get("SECURITY_ALERT", 0)),
        "SLOW_POISONING_SUSPECTED": int(
            event_counts.get("SLOW_POISONING_SUSPECTED", 0)
        ),
    }

    # ── ATM / NDT outcome counts ─────────────────────────────────────────
    n_atm = len(atm_results)
    n_atm_deployed = sum(
        1 for r in atm_results if bool(getattr(r, "deployed", False))
    )
    n_ndt_passed = sum(
        1 for r in atm_results if bool(getattr(r, "ndt_passed", False))
    )

    # ── Per-phase mean rolling accuracy (window=50) ──────────────────────
    # Quantifies the realised cost of disabling a detector.
    # We compute a phase-resolved mean of the per-step ``correct`` flag.
    by_phase = {ph: [] for ph in (1, 2, 3, 4)}
    for r in step_records:
        by_phase[r["phase"]].append(r["correct"])
    phase_acc = {
        f"P{ph}_mean_acc": (
            float(np.mean(vs)) if vs else float("nan")
        )
        for ph, vs in by_phase.items()
    }

    # ── Phase-resolved trigger map (which checks fired in which phase) ───
    def _phase_of(step: int) -> int:
        for ph, (s, e) in PHASE_RANGES.items():
            if s <= step <= e:
                return ph
        return 0

    phase_triggers = {
        f"P{ph}": {"DDD": 0, "DPD": 0, "CDD": 0, "CPD": 0}
        for ph in (1, 2, 3, 4)
    }
    for r in check_records:
        ph = _phase_of(r["step"])
        if ph == 0:
            continue
        if r["ddd_drift"]:
            phase_triggers[f"P{ph}"]["DDD"] += 1
        if r["dpd_poison"]:
            phase_triggers[f"P{ph}"]["DPD"] += 1
        if r["cdd_drift"]:
            phase_triggers[f"P{ph}"]["CDD"] += 1
        if r["cpd_poison"]:
            phase_triggers[f"P{ph}"]["CPD"] += 1

    return {
        "variant": name,
        "n_checks": len(check_records),
        "triggers_total": triggers,
        "triggers_by_phase": phase_triggers,
        "events": selected,
        "atm": {
            "cycles_initiated": n_atm,
            "ndt_passed": n_ndt_passed,
            "deployed": n_atm_deployed,
        },
        "ndt_history_count": len(ndt_history),
        "phase_acc": phase_acc,
    }


def main() -> int:
    print("=" * 60)
    print("Per-detector ablation study")
    print("=" * 60)

    snapshot = {
        "ablations_run": [name for name, _ in ABLATIONS],
        "variants": {},
    }

    for label, target in ABLATIONS:
        print(f"\n[ {label} ]  ablate={target!r}")
        data = run_simulation(ablate=target)
        summary = summarise_variant(label, data)
        snapshot["variants"][label] = {
            "summary": summary,
            "check_records": [_safe(r) for r in data["check_records"]],
            "events": [
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
        }
        # Console one-liner
        ev = summary["events"]
        atm = summary["atm"]
        tt = summary["triggers_total"]
        print(
            f"      triggers: DDD={tt['DDD']:>2d}  DPD={tt['DPD']:>2d}  "
            f"CDD={tt['CDD']:>2d}  CPD={tt['CPD']:>2d}  | "
            f"MToUT={ev['MTOUT_FIRED']:>2d} suppressed={ev['MTOUT_SUPPRESSED']:>2d} "
            f"rollback={ev['ROLLBACK']:>2d} alerts={ev['SECURITY_ALERT']:>2d} | "
            f"ATM cycles={atm['cycles_initiated']} deployed={atm['deployed']}"
        )
        for ph in (1, 2, 3, 4):
            acc = summary["phase_acc"][f"P{ph}_mean_acc"]
            print(f"      P{ph} mean rolling-acc: {acc:.3f}")

    out_path = OUT_DIR / "ablation.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    print(
        f"\nWrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
