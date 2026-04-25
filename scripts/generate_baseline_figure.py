"""
scripts/generate_baseline_figure.py — render fig_07 from
``thesis/.repro_audit_data/baseline_comparison.json``.

Two-panel figure, one per detector family:

    Left  : Drift family       (DDD, CDD, ADWIN)
    Right : Poisoning family   (DPD, CPD, SpectralSig, CUSUM-prior)

Within each panel, every detector contributes one cluster of three
bars: precision, recall, F1.  Production detectors are coloured in
the same dark-grey baseline scheme used elsewhere in the thesis;
external baselines are coloured by family for visual contrast.

NaN scores (e.g. precision when the detector never fires) are
rendered as a hatched empty bar with an "n/a" annotation so the
reader does not mistake an absent bar for zero.
"""
from __future__ import annotations

import json
import sys
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "thesis" / ".repro_audit_data" / "baseline_comparison.json"
OUT_DIR = ROOT / "thesis" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "fig_07_baseline_comparison.pdf"

# Display order per family (production first, then baselines).
DRIFT_ORDER = ["DDD", "CDD", "ADWIN"]
POISON_ORDER = ["DPD", "CPD", "SpectralSig", "CUSUM-prior"]

PROD_DETECTORS = {"DDD", "DPD", "CDD", "CPD"}

COLOR = {
    # Production detectors — neutral dark greys
    "DDD": "#2c2c2c",
    "DPD": "#2c2c2c",
    "CDD": "#5a5a5a",
    "CPD": "#5a5a5a",
    # External baselines — distinct hues
    "ADWIN":       "#1f77b4",  # drift family
    "SpectralSig": "#d62728",  # poisoning family
    "CUSUM-prior": "#9467bd",  # poisoning family
}

METRICS = ["precision", "recall", "f1"]
METRIC_LABELS = {"precision": "Precision", "recall": "Recall", "f1": "F1"}


def _is_nan(v) -> bool:
    return isinstance(v, float) and math.isnan(v)


def _panel(ax, results, order, family_label, panel_letter):
    n_dets = len(order)
    n_metrics = len(METRICS)
    bar_w = 0.8 / n_dets
    x = np.arange(n_metrics)

    for di, det in enumerate(order):
        r = results[det]
        offsets = x - 0.4 + bar_w * (di + 0.5)
        heights = []
        for m in METRICS:
            v = r[m]
            heights.append(0.0 if _is_nan(v) else float(v))
        # Render bars
        bars = ax.bar(
            offsets, heights, width=bar_w * 0.95,
            color=COLOR[det],
            edgecolor="black", linewidth=0.4,
            label=det + (" (production)" if det in PROD_DETECTORS else " (baseline)"),
        )
        # Annotate / hatch NaN bars
        for bi, m in enumerate(METRICS):
            v = r[m]
            xc = offsets[bi]
            if _is_nan(v):
                # Replace solid fill with hatched empty bar
                bars[bi].set_facecolor("white")
                bars[bi].set_edgecolor(COLOR[det])
                bars[bi].set_hatch("///")
                bars[bi].set_height(0.04)
                ax.text(
                    xc, 0.06, "n/a",
                    ha="center", va="bottom", fontsize=7, color=COLOR[det],
                )
            else:
                ax.text(
                    xc, v + 0.015, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7,
                )

    # Mention TP/FP/FN counts under each detector (small annotation)
    for di, det in enumerate(order):
        r = results[det]
        # We re-purpose the area under the cluster as a small tag
        ax.text(
            x[0] - 0.4 + bar_w * (di + 0.5),
            -0.10,
            f"TP={r['tp']} FP={r['fp']}\nFN={r['fn']} TN={r['tn']}",
            ha="center", va="top", fontsize=6.5, color="black",
            transform=ax.transData,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=10)
    ax.set_ylim(-0.20, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title(f"({panel_letter}) {family_label} family", fontsize=11,
                 loc="left", pad=6)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=8, frameon=True,
              ncol=1, handlelength=1.2, handletextpad=0.4)


def main() -> int:
    if not JSON_PATH.exists():
        print(
            f"baseline_comparison.json not found at {JSON_PATH}.\n"
            "Run scripts/baseline_compare.py first."
        )
        return 1
    snap = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    results = snap["results"]

    fig = plt.figure(figsize=(11, 5.5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.22)
    ax_l = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])

    _panel(ax_l, results, DRIFT_ORDER, "Drift", "a")
    _panel(ax_r, results, POISON_ORDER, "Poisoning", "b")

    fig.suptitle(
        "Production detectors vs. external baselines (Campaign~A, n=18 checks)",
        fontsize=12, fontweight="bold", y=0.995,
    )
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
