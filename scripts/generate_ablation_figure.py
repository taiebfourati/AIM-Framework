"""
scripts/generate_ablation_figure.py — render fig_06 from ablation.json.

Reads the ablation snapshot produced by ``scripts/ablation_run.py`` and
emits ``thesis/figures/fig_06_ablation.pdf``.  The figure has two
panels arranged vertically:

    Top   : per-detector trigger counts per variant
            (5 variants  x  4 detectors  =  20 bars, grouped by detector)
    Bottom: pipeline outcomes per variant
            (5 variants  x  5 outcomes
              {MToUT fired, MToUT suppressed, rollback, security alerts,
               ATM cycles deployed}
             =  25 bars, grouped by outcome)

Reading the figure: each panel has one cluster per metric on the x-axis;
within a cluster, the five coloured bars are the five ablation variants
in fixed order (baseline, no-DDD, no-DPD, no-CDD, no-CPD).  The
baseline bar within each cluster is dark grey so the four ablations are
easy to compare against the all-detectors-on reference.

Usage:
    python scripts/generate_ablation_figure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "thesis" / ".repro_audit_data" / "ablation.json"
OUT_DIR = ROOT / "thesis" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "fig_06_ablation.pdf"


VARIANT_ORDER = ["baseline", "no_DDD", "no_DPD", "no_CDD", "no_CPD"]
VARIANT_LABEL = {
    "baseline": "Baseline\n(all detectors)",
    "no_DDD":   "$-$DDD",
    "no_DPD":   "$-$DPD",
    "no_CDD":   "$-$CDD",
    "no_CPD":   "$-$CPD",
}
# Baseline is dark grey so the four ablations stand out
VARIANT_COLOR = {
    "baseline": "#3a3a3a",
    "no_DDD":   "#1f77b4",
    "no_DPD":   "#d62728",
    "no_CDD":   "#2ca02c",
    "no_CPD":   "#9467bd",
}

DETECTORS = ["DDD", "DPD", "CDD", "CPD"]
OUTCOMES = [
    ("MTOUT_FIRED",     "MToUT\nfired"),
    ("MTOUT_SUPPRESSED","MToUT\nsuppressed"),
    ("ROLLBACK",        "Rollbacks"),
    ("SECURITY_ALERT",  "Security\nalerts"),
    ("__ATM_DEPLOYED",  "ATM cycles\ndeployed"),
]


def _grouped_bars(ax, values_by_group, title, ylabel,
                  group_keys, group_display):
    """Render a grouped bar chart.

    Parameters
    ----------
    ax : matplotlib axes
    values_by_group : dict[str, dict[str, float]]
        First key is the group (lookup key, matches ``group_keys``);
        second key is the variant; value is the bar height.
    title, ylabel : str
    group_keys : list[str]
        Lookup keys into ``values_by_group``.
    group_display : list[str]
        x-axis tick labels (human-readable, parallel to ``group_keys``).
    """
    n_variants = len(VARIANT_ORDER)
    n_groups = len(group_keys)
    bar_w = 0.8 / n_variants
    x = np.arange(n_groups)

    for vi, variant in enumerate(VARIANT_ORDER):
        offsets = x - 0.4 + bar_w * (vi + 0.5)
        heights = [
            values_by_group[g][variant] for g in group_keys
        ]
        ax.bar(
            offsets, heights, width=bar_w * 0.95,
            color=VARIANT_COLOR[variant],
            edgecolor="black", linewidth=0.4,
            label=VARIANT_LABEL[variant],
        )
        for xi, h in zip(offsets, heights):
            if h > 0:
                ax.text(
                    xi, h + 0.05, f"{int(h)}",
                    ha="center", va="bottom", fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(group_display, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, loc="left", pad=6)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    ymax = max(
        max(d.values()) for d in values_by_group.values()
    )
    ax.set_ylim(0, max(1, ymax) * 1.25)


def main() -> int:
    if not JSON_PATH.exists():
        print(
            f"ablation.json not found at {JSON_PATH}.\n"
            "Run scripts/ablation_run.py first."
        )
        return 1
    snap = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    variants = snap["variants"]

    # ── Panel 1: detector triggers ───────────────────────────────────────
    triggers = {
        det: {
            v: int(variants[v]["summary"]["triggers_total"][det])
            for v in VARIANT_ORDER
        }
        for det in DETECTORS
    }

    # ── Panel 2: pipeline outcomes ───────────────────────────────────────
    outcomes = {}
    for key, _label in OUTCOMES:
        outcomes[key] = {}
        for v in VARIANT_ORDER:
            sm = variants[v]["summary"]
            if key == "__ATM_DEPLOYED":
                outcomes[key][v] = int(sm["atm"]["deployed"])
            else:
                outcomes[key][v] = int(sm["events"][key])

    # ── Render ───────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 7))
    gs = GridSpec(2, 1, figure=fig, hspace=0.42, height_ratios=[1.0, 1.0])
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])

    _grouped_bars(
        ax_top, triggers,
        "(a) Detector trigger counts (out of 18 check intervals)",
        "Trigger count",
        DETECTORS, DETECTORS,
    )
    _grouped_bars(
        ax_bot, outcomes,
        "(b) Pipeline outcomes (event log + ATM dispatch)",
        "Count",
        [k for k, _ in OUTCOMES],
        [lbl for _, lbl in OUTCOMES],
    )

    # Single shared legend on the top axes
    ax_top.legend(
        loc="upper right",
        ncol=5, frameon=True, fontsize=8,
        columnspacing=0.9, handletextpad=0.5,
    )

    fig.suptitle(
        "Per-detector ablation study (Campaign~A, single seed)",
        fontsize=12, fontweight="bold", y=0.995,
    )

    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
