"""
scripts/wilcoxon_effects.py — MAJ-10 + MAJ-14 supporting analysis.

Reads the cached per-seed Campaign~C sweep results from
``thesis/results_v2/sweep.csv`` and produces:

  * A pairwise Wilcoxon signed-rank test on the per-seed
    cross-detector mean F1, paired across seeds, for the four
    check-interval settings  ``C in {10, 25, 50, 100}``.
  * Holm--Bonferroni step-down correction over the six pairwise
    p-values (m = binom(4, 2) = 6).
  * Matched-pair **rank-biserial r** effect size for every
    pair (Kerby 2014):
        r = 2 * (W_+ / N_p) - 1
    where ``W_+`` is the sum of positive signed ranks (ranks of
    |d_i| over the non-zero differences) and ``N_p`` is the sum
    of all positive- and negative-rank weights.
  * Per-detector recall MDES (minimum detectable effect size)
    over the 60-seed Campaign~B for the a-priori power statement.
  * A re-rendered ``fig_significance_matrix.pdf`` whose cells
    show **two** numbers: Holm-corrected p (top) and rank-biserial
    r (bottom).

Outputs:
  * thesis/.repro_audit_data/wilcoxon_effects.json
  * thesis/figures/fig_significance_matrix.pdf  (overwritten)

The script is intentionally cheap to re-run (no simulation, no
detector training): every artefact derives from the cached
``sweep.csv`` and Campaign~B headline CSV that already live in
``thesis/results_v2/``.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
SWEEP_CSV = ROOT / "thesis" / "results_v2" / "sweep.csv"
HEADLINE_CSV = ROOT / "thesis" / "results_v2" / "headline.csv"
JSON_OUT = ROOT / "thesis" / ".repro_audit_data" / "wilcoxon_effects.json"
FIG_OUT = ROOT / "thesis" / "figures" / "fig_significance_matrix.pdf"

ALPHA = 0.05
DETECTORS = ["ddd", "dpd", "cdd", "cpd"]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def holm_correct(pvals: list[float]) -> list[float]:
    """Holm--Bonferroni step-down correction; returns adjusted p-values
    in the original order of ``pvals``.

    Adjusted p_(k) = max_{j<=k} ((m - j + 1) * p_(j)), monotonic, clipped to 1.
    """
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running_max = 0.0
    for k, idx in enumerate(order, start=1):
        candidate = (m - k + 1) * pvals[idx]
        running_max = max(running_max, candidate)
        adj[idx] = min(running_max, 1.0)
    return adj.tolist()


def rank_biserial_r(a: np.ndarray, b: np.ndarray) -> float:
    """Matched-pair rank-biserial correlation r (Kerby, 2014).

    r in [-1, +1].  r > 0 indicates ``a`` stochastically dominates ``b``;
    r < 0 indicates the reverse.  Zero differences are dropped per
    standard Wilcoxon protocol.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[d != 0.0]
    if d.size == 0:
        return float("nan")
    ranks = stats.rankdata(np.abs(d))
    w_plus = float(np.sum(ranks[d > 0.0]))
    total = float(np.sum(ranks))
    return 2.0 * (w_plus / total) - 1.0


def mdes_from_std_paired(std_diff: float, n: int,
                         alpha: float = 0.05, power: float = 0.80) -> float:
    """A-priori MDES (mean shift) for a paired test with given std and
    sample size, using normal approximation:

        MDES = (z_{alpha/2} + z_{power}) * std_diff / sqrt(n)

    For Wilcoxon signed-rank the asymptotic relative efficiency vs the
    paired t-test is 3/pi ~= 0.955; the normal approximation here is a
    standard upper bound.
    """
    z_a = stats.norm.ppf(1.0 - alpha / 2.0)
    z_b = stats.norm.ppf(power)
    return (z_a + z_b) * std_diff / np.sqrt(n)


def mdes_from_recall(p_hat: float, n: int, alpha: float = 0.05) -> float:
    """Half-width of a Wilson 95% CI on a recall point estimate with
    n trials.  Used as a per-detector MDES proxy when the detector
    fires at most once per seed (binary outcome aggregated across n
    seeds produces a Bernoulli-like sampling distribution for the
    seed-level recall).
    """
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    if p_hat <= 0 or p_hat >= 1:
        # Edge cases — fall back to the maximum half-width 0.5 / sqrt(n)
        return float(z * 0.5 / np.sqrt(n))
    return float(z * np.sqrt(p_hat * (1.0 - p_hat) / n))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> int:
    if not SWEEP_CSV.exists():
        print(f"sweep.csv not found at {SWEEP_CSV}.")
        return 1
    if not HEADLINE_CSV.exists():
        print(f"headline.csv not found at {HEADLINE_CSV}.")
        return 1

    sweep = pd.read_csv(SWEEP_CSV)
    headline = pd.read_csv(HEADLINE_CSV)

    # ------------------------------------------------------------------
    # Pairwise Wilcoxon + rank-biserial r over Campaign C (per-seed
    # mean F1 across the four production detectors).
    # ------------------------------------------------------------------
    intervals = sorted(int(c) for c in sweep["check_interval"].unique())
    f1_by_c: dict[int, np.ndarray] = {}
    for c in intervals:
        sub = sweep[sweep["check_interval"] == c].sort_values("seed")
        f1 = sub[["ddd_f1", "dpd_f1", "cdd_f1", "cpd_f1"]].mean(axis=1).to_numpy()
        f1_by_c[c] = f1

    # Pairwise raw p, rank-biserial r, paired Wilcoxon (zero-method='wilcox')
    pairs = list(combinations(intervals, 2))
    raw_p: list[float] = []
    r_vals: list[float] = []
    for ci, cj in pairs:
        a, b = f1_by_c[ci], f1_by_c[cj]
        m = min(len(a), len(b))
        a, b = a[:m], b[:m]
        d = a - b
        if np.allclose(d, 0.0):
            raw_p.append(float("nan"))
            r_vals.append(float("nan"))
            continue
        try:
            res = stats.wilcoxon(a, b, alternative="two-sided",
                                 zero_method="wilcox", method="auto")
            raw_p.append(float(res.pvalue))
        except ValueError:
            raw_p.append(float("nan"))
        r_vals.append(rank_biserial_r(a, b))

    holm_p = holm_correct([0.0 if np.isnan(p) else p for p in raw_p])

    # Per-pair record + matrices
    pair_rows = []
    n = len(intervals)
    p_mat = np.full((n, n), np.nan)
    r_mat = np.full((n, n), np.nan)
    for k, (ci, cj) in enumerate(pairs):
        i, j = intervals.index(ci), intervals.index(cj)
        p_mat[i, j] = holm_p[k]
        p_mat[j, i] = holm_p[k]
        r_mat[i, j] = r_vals[k]
        r_mat[j, i] = -r_vals[k]  # sign reverses for the symmetric pair
        pair_rows.append({
            "ci": ci, "cj": cj,
            "n_seeds": int(min(len(f1_by_c[ci]), len(f1_by_c[cj]))),
            "p_raw": float(raw_p[k]),
            "p_holm": float(holm_p[k]),
            "rank_biserial_r": float(r_vals[k]),
            "median_diff": float(np.median(f1_by_c[ci][:min(len(f1_by_c[ci]), len(f1_by_c[cj]))]
                                            - f1_by_c[cj][:min(len(f1_by_c[ci]), len(f1_by_c[cj]))])),
        })

    # ------------------------------------------------------------------
    # MDES per detector (Campaign B headline.csv has 60 seeds @ C=50)
    # ------------------------------------------------------------------
    mdes = {}
    for det in DETECTORS:
        recall_col = f"{det}_recall"
        f1_col = f"{det}_f1"
        if recall_col not in headline.columns:
            continue
        n_seeds_b = int(headline[recall_col].notna().sum())
        rec = headline[recall_col].to_numpy(dtype=float)
        f1 = headline[f1_col].to_numpy(dtype=float)
        rec_finite = rec[np.isfinite(rec)]
        f1_finite = f1[np.isfinite(f1)]
        rec_mean = float(rec_finite.mean()) if rec_finite.size else float("nan")
        rec_std = float(rec_finite.std(ddof=1)) if rec_finite.size > 1 else float("nan")
        f1_mean = float(f1_finite.mean()) if f1_finite.size else float("nan")
        f1_std = float(f1_finite.std(ddof=1)) if f1_finite.size > 1 else float("nan")

        # A-priori MDES (paired Wilcoxon, n_b seeds, alpha=0.05, power=0.80)
        mdes_paired = (mdes_from_std_paired(f1_std, n_seeds_b)
                       if np.isfinite(f1_std) and n_seeds_b > 1 else float("nan"))
        # Wilson half-width on recall
        mdes_recall = mdes_from_recall(rec_mean, n_seeds_b)
        mdes[det.upper()] = {
            "n_seeds_campaignB": n_seeds_b,
            "recall_mean": rec_mean,
            "recall_std": rec_std,
            "f1_mean": f1_mean,
            "f1_std": f1_std,
            "mdes_recall_wilson_half_width": float(mdes_recall),
            "mdes_paired_f1_alpha05_power80": float(mdes_paired),
        }

    # Pairwise Wilcoxon MDES on the 30-seed paired Campaign~C design
    n_c = int(min(len(v) for v in f1_by_c.values()))
    paired_diffs = []
    for ci, cj in pairs:
        m = min(len(f1_by_c[ci]), len(f1_by_c[cj]))
        paired_diffs.append((f1_by_c[ci][:m] - f1_by_c[cj][:m]).std(ddof=1))
    sd_pooled = float(np.sqrt(np.mean(np.square(paired_diffs))))
    mdes_pair = float(mdes_from_std_paired(sd_pooled, n_c))

    payload = {
        "metadata": {
            "source_sweep_csv": str(SWEEP_CSV.relative_to(ROOT)),
            "source_headline_csv": str(HEADLINE_CSV.relative_to(ROOT)),
            "alpha": ALPHA,
            "intervals": intervals,
            "n_pairs": len(pairs),
            "holm_correction": "step-down (Holm 1979)",
            "effect_size": "matched-pair rank-biserial r (Kerby 2014)",
            "mdes_method": "paired-Wilcoxon normal approximation, "
                            "(z_{alpha/2}+z_{power})*sd/sqrt(n)",
        },
        "pairs": pair_rows,
        "mdes_per_detector_campaignB": mdes,
        "mdes_pairwise_check_interval_campaignC": {
            "n_seeds": n_c,
            "pooled_sd_diff": sd_pooled,
            "mdes_alpha05_power80": mdes_pair,
        },
    }
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {JSON_OUT.relative_to(ROOT)}")

    # ------------------------------------------------------------------
    # Re-render fig_significance_matrix.pdf with both p_holm and r_rb
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    cmap = plt.colormaps["RdYlGn_r"]
    im = ax.imshow(p_mat, cmap=cmap, vmin=0.0, vmax=0.10, origin="lower")
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"C={c}" for c in intervals])
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"C={c}" for c in intervals])
    ax.set_xlabel("check interval $C$ (samples)")
    ax.set_ylabel("check interval $C$ (samples)")

    # Cell annotation: Holm-corrected p (top), rank-biserial r (bottom)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p_v = p_mat[i, j]
            r_v = r_mat[i, j]
            if np.isnan(p_v):
                continue
            colour = "white" if p_v < 0.05 else "black"
            ax.text(j, i + 0.18, f"$p = {p_v:.3f}$",
                    ha="center", va="center", fontsize=8, color=colour)
            ax.text(j, i - 0.22, f"$r = {r_v:+.2f}$",
                    ha="center", va="center", fontsize=8, color=colour,
                    fontstyle="italic")

    cbar = plt.colorbar(im, ax=ax, shrink=0.85,
                        label="Holm-corrected $p$-value")
    cbar.set_ticks([0.0, 0.025, 0.05, 0.075, 0.10])
    ax.set_title(
        "Pairwise check-interval comparison\n"
        "(Wilcoxon signed-rank, Holm-corrected $p$ + rank-biserial $r$)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIG_OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_OUT.relative_to(ROOT)} "
          f"({FIG_OUT.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
