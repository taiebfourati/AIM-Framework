"""
scripts/generate_bias_floor_figure.py — render fig_08 from
``thesis/.repro_audit_data/bias_floor.json``.

Two-panel figure illustrating the dual-score NDT gate's blind spot
against the clean-label distribution-preserving adversary defined in
``baselines/clean_label_attack.py``.

Left panel : Score divergence as a function of attacker_fraction.
             Three accuracy curves --- (a) candidate's GOLDEN-corpus
             accuracy (the operator's true-but-unobservable metric),
             (b) candidate's LOB pseudo-label accuracy (what the NDT
             gate evaluates), and (c) baseline's LOB accuracy (the
             gate's reference) --- plotted against attacker_fraction.
             The horizontal line marks the production theta_ndt=0.65
             floor.  A shaded band highlights the gap between
             cand_lob (~1.0) and cand_gold, which is the operator's
             unseen accuracy debt at gate-pass time.

Right panel: Gate-pass rate matrix over (theta_ndt, attacker_fraction).
             Cells are coloured by the fraction of seeds at which the
             dual-score gate accepts the poisoned candidate; overlaid
             contour lines mark the candidate's golden-corpus accuracy
             at each (theta_ndt, attacker_fraction) cell.  The figure
             makes visible that the gate accepts at 100% of seeds
             across the ENTIRE grid (no theta_ndt setting rejects the
             attack) while the operator's unobserved accuracy degrades
             monotonically with attacker_fraction --- exactly the
             ``bias floor'' the discussion documents as the third NDT
             failure regime.
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
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "thesis" / ".repro_audit_data" / "bias_floor.json"
OUT_DIR = ROOT / "thesis" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "fig_08_bias_floor.pdf"


def main() -> int:
    if not JSON_PATH.exists():
        print(
            f"bias_floor.json not found at {JSON_PATH}.\n"
            "Run scripts/bias_floor_sweep.py first."
        )
        return 1

    snap = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    meta = snap["metadata"]
    fs = snap["fraction_summary"]
    af_grid = np.array([row["attacker_fraction"] for row in fs])
    theta_grid = np.array(meta["theta_ndt_grid"])
    raw = snap["raw"]

    cand_gold = np.array([row["cand_gold_mean"] for row in fs])
    cand_gold_std = np.array([row["cand_gold_std"] for row in fs])
    cand_lob = np.array([row["cand_lob_mean"] for row in fs])
    cand_lob_std = np.array([row["cand_lob_std"] for row in fs])
    base_lob = np.array([row["base_lob_mean"] for row in fs])
    base_lob_std = np.array([row["base_lob_std"] for row in fs])
    base_gold = np.array([row["base_gold_mean"] for row in fs])

    # Gate pass rate: rows = af, cols = theta_ndt
    gate = np.array(snap["gate_pass_rate"])

    # Per-cell candidate gold accuracy (constant across theta_ndt for a
    # given af, but we tile so the heatmap reads naturally).
    cand_gold_grid = np.tile(cand_gold[:, None], (1, len(theta_grid)))

    # ----------------------------------------------------------------------
    # Figure
    # ----------------------------------------------------------------------
    fig = plt.figure(figsize=(12.5, 5.4))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.05], wspace=0.32)
    ax_l = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])

    # ── Left panel: score divergence ──────────────────────────────────────
    af_pct = af_grid * 100.0  # show as percent for readability

    # Candidate LOB (what the gate sees)
    ax_l.plot(af_pct, cand_lob, marker="s", linestyle=":",
              color="#1f77b4", linewidth=1.6, markersize=5,
              label="candidate accuracy on LOB (gate sees)")
    ax_l.fill_between(af_pct,
                      cand_lob - cand_lob_std,
                      cand_lob + cand_lob_std,
                      color="#1f77b4", alpha=0.10)

    # Baseline LOB (gate's reference)
    ax_l.plot(af_pct, base_lob, marker="^", linestyle="--",
              color="#7f7f7f", linewidth=1.4, markersize=5,
              label="baseline accuracy on LOB (gate ref.)")
    ax_l.fill_between(af_pct,
                      base_lob - base_lob_std,
                      base_lob + base_lob_std,
                      color="#7f7f7f", alpha=0.10)

    # Candidate GOLDEN (truth, unseen by gate)
    ax_l.plot(af_pct, cand_gold, marker="o", linestyle="-",
              color="#d62728", linewidth=2.0, markersize=6,
              label="candidate accuracy on GOLDEN (truth)")
    ax_l.fill_between(af_pct,
                      cand_gold - cand_gold_std,
                      cand_gold + cand_gold_std,
                      color="#d62728", alpha=0.18)

    # Baseline GOLDEN reference (constant)
    ax_l.axhline(y=base_gold.mean(), color="#2ca02c", linestyle="-.",
                 linewidth=1.2,
                 label=f"baseline accuracy on GOLDEN = {base_gold.mean():.3f}")

    # Production theta_ndt = 0.65 floor
    ax_l.axhline(y=0.65, color="black", linestyle=":", linewidth=1.0,
                 alpha=0.7)
    ax_l.text(af_pct[-1], 0.65 + 0.005,
              r"$\theta_{\mathrm{ndt}}=0.65$ (production)",
              ha="right", va="bottom", fontsize=8, color="black")

    ax_l.set_xlabel("Attacker fraction of LOB buffer  (%)", fontsize=10)
    ax_l.set_ylabel("Accuracy", fontsize=10)
    ax_l.set_title("(a) Score divergence under clean-label poisoning",
                   fontsize=11, loc="left", pad=6)
    ax_l.set_xlim(-1, max(af_pct) + 1)
    ax_l.set_ylim(0.60, 1.02)
    ax_l.grid(True, alpha=0.25, linestyle=":")
    ax_l.set_axisbelow(True)
    ax_l.legend(loc="lower left", fontsize=8, frameon=True,
                handlelength=1.8, handletextpad=0.5)

    # Annotate the gap between cand_lob and cand_gold at af=0.50
    last = -1
    ax_l.annotate(
        f"true degradation\n"
        f"$\\Delta$ = {cand_lob[last] - cand_gold[last]:.3f}\n"
        f"(gate is blind to it)",
        xy=(af_pct[last], (cand_lob[last] + cand_gold[last]) / 2),
        xytext=(af_pct[last] - 14, 0.74),
        fontsize=8,
        color="#7a0000",
        arrowprops=dict(arrowstyle="->", color="#7a0000",
                        connectionstyle="arc3,rad=0.15", linewidth=0.8),
    )

    # ── Right panel: gate pass rate heatmap ──────────────────────────────
    # Use a green colormap so "always pass" reads as a uniform alarming
    # block (which is the finding).
    cmap_gate = LinearSegmentedColormap.from_list(
        "gate", ["#ffffff", "#fdd49e", "#fdbb84", "#fc8d59", "#d7301f"]
    )
    # Transpose so x-axis is af and y-axis is theta_ndt (more intuitive)
    pcm = ax_r.imshow(
        gate.T,
        origin="lower",
        aspect="auto",
        cmap=cmap_gate,
        vmin=0.0, vmax=1.0,
        extent=(af_pct[0] - (af_pct[1] - af_pct[0]) / 2,
                af_pct[-1] + (af_pct[1] - af_pct[0]) / 2,
                theta_grid[0] - (theta_grid[1] - theta_grid[0]) / 2,
                theta_grid[-1] + (theta_grid[-1] - theta_grid[-2]) / 2),
    )

    # Cell annotations: gate pass rate as a number
    for i, af in enumerate(af_pct):
        for j, theta in enumerate(theta_grid):
            v = gate[i, j]
            txt = "1.00" if v == 1.0 else f"{v:.2f}"
            ax_r.text(af, theta, txt,
                      ha="center", va="center", fontsize=7,
                      color="#3a3a3a")

    # Contour overlay: candidate's golden-corpus accuracy
    AF_M, THETA_M = np.meshgrid(af_pct, theta_grid)
    cs = ax_r.contour(
        AF_M, THETA_M, cand_gold_grid.T,
        levels=[0.880, 0.890, 0.900, 0.910],
        colors="#0a3d62", linewidths=1.2, linestyles="-",
    )
    ax_r.clabel(cs, fmt="cand_gold=%.3f", fontsize=7, inline=True)

    # Mark the production theta_ndt
    ax_r.axhline(y=0.65, color="black", linestyle=":", linewidth=1.0,
                 alpha=0.6)

    ax_r.set_xlabel("Attacker fraction of LOB buffer  (%)", fontsize=10)
    ax_r.set_ylabel(r"NDT floor  $\theta_{\mathrm{ndt}}$", fontsize=10)
    ax_r.set_title(
        "(b) Gate pass rate (color) and candidate truth (contours)",
        fontsize=11, loc="left", pad=6,
    )
    cb = fig.colorbar(pcm, ax=ax_r, fraction=0.040, pad=0.02)
    cb.set_label("Gate pass rate over seeds", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Bias-floor analysis: dual-score NDT gate vs. clean-label "
        "distribution-preserving adversary  "
        f"({meta['n_seeds']} seeds/cell, n_buffer={meta['n_clean_buffer']}, "
        f"bias_band={meta['bias_band']})",
        fontsize=11.5, fontweight="bold", y=1.005,
    )
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
