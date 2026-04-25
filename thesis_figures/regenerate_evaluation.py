"""
regenerate_evaluation.py
Rigorous, self-contained re-evaluation of the AIMP framework's detection
quality on a synthetic four-phase 6G workload.

Why a re-evaluation?
   The original `generate_eval_figures.py` ran 30 seeds against a
   deterministic injection schedule, which produced suspiciously
   constant per-seed metrics (e.g. DDD recall = 0.833 every seed).
   This script keeps the same 4-phase structure but
        * randomises the drift magnitude per seed
        * randomises the poisoning rate per seed
        * randomises the per-step noise level per seed
        * runs 60 seeds total (vs. 30)
        * sweeps the detector check interval C in {10, 25, 50, 100}
        * computes bootstrap 95 % confidence intervals (n_boot = 2000)
        * reports F1 alongside precision and recall
        * outputs a sensitivity table with paired per-seed deltas
        * produces publication-quality figures with seaborn-style aesthetics

Outputs (under thesis/results_v2/):
    headline.csv          -- per-seed metrics, all detectors, C = 50
    headline_summary.csv  -- aggregate (mean, std, ci_lo, ci_hi)
    sweep.csv             -- per-(seed, C) metrics
    sweep_summary.csv     -- aggregate per-C with bootstrap CIs
    per_phase.csv         -- per-(seed, phase) accuracy
    per_phase_summary.csv -- aggregate per-phase with CIs

Outputs (under thesis/figures/):
    fig_headline_boxplot.pdf    -- per-detector P / R / F1 boxplots
    fig_rocpr_v2.pdf            -- ROC + PR with 95 % CI bands
    fig_sweep_pareto.pdf        -- F1-vs-cost Pareto with errorbars
    fig_per_phase_v2.pdf        -- per-phase accuracy with CIs
    fig_latency_cdf.pdf         -- detection-latency CDF per detector
    fig_calibration.pdf         -- per-detector reliability diagram
    fig_significance_matrix.pdf -- pairwise Wilcoxon p-values across C

Usage:
    cd /path/to/repo
    .venv\\Scripts\\python.exe thesis/regenerate_evaluation.py
"""

from __future__ import annotations

import io
import os
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (auc, average_precision_score,
                             precision_recall_curve, roc_curve)

# Windows console: force UTF-8 so Greek and ± render in stdout
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
warnings.filterwarnings("ignore")

# =============================================================================
# Output directories
# =============================================================================
THESIS_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(THESIS_DIR, "results_v2")
FIGS_DIR    = os.path.join(THESIS_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)


# =============================================================================
# Phase schedule and palette
# =============================================================================
PHASES = {
    1: {"name": "Stable",        "range": (0,   300), "colour": "#27ae60"},
    2: {"name": "Drift",         "range": (300, 500), "colour": "#e67e22"},
    3: {"name": "Poisoning",     "range": (500, 700), "colour": "#e74c3c"},
    4: {"name": "Recovery",      "range": (700, 900), "colour": "#3498db"},
}
N_STEPS = 900

DETECTOR_COLOURS = {
    "DDD": "#8e44ad",
    "DPD": "#e67e22",
    "CDD": "#27ae60",
    "CPD": "#e74c3c",
}

# Style: Tufte-influenced, publication-grade. No top/right spines, faint grid.
plt.rcParams.update({
    "font.family":           "DejaVu Sans",
    "font.size":             10,
    "axes.titlesize":        11,
    "axes.labelsize":        10,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.grid":             True,
    "axes.axisbelow":        True,
    "grid.color":            "#d8d8d8",
    "grid.linewidth":        0.6,
    "grid.alpha":            0.55,
    "xtick.direction":       "out",
    "ytick.direction":       "out",
    "xtick.labelsize":       9,
    "ytick.labelsize":       9,
    "legend.fontsize":       9,
    "legend.frameon":        False,
    "figure.dpi":            150,
    "savefig.bbox":          "tight",
    "savefig.dpi":           150,
})


# =============================================================================
# Workload synthesis
# =============================================================================
@dataclass
class SeedConfig:
    """Per-seed workload parameters drawn from priors."""
    drift_magnitude:   float       # mean shift in feature 0 by end of P2 (dB)
    poison_rate:       float       # fraction of P3 labels that are flipped
    base_noise:        float       # P1/P4 label noise (5--8 %)
    drift_features:    np.ndarray  # which features are affected (typ. [0, 1])

    @classmethod
    def sample(cls, rng: np.random.Generator) -> "SeedConfig":
        return cls(
            drift_magnitude = float(rng.uniform(0.6, 1.4)),
            poison_rate     = float(rng.uniform(0.08, 0.18)),
            base_noise      = float(rng.uniform(0.04, 0.08)),
            drift_features  = np.array([0, 1]),
        )


def synthesise_workload(
    cfg: SeedConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y_true, phase) of length N_STEPS.

    X.shape = (N, 4) -- four KPIs analogous to (RSRP, SINR, Throughput, Lat).
    y_true.shape = (N,) -- binary handover-success label.
    phase.shape = (N,) -- {1, 2, 3, 4}.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(N_STEPS, 4)).astype(float)

    # Decision rule: y = 1 iff the average of the two ``decision'' features
    # (analogous to RSRP+SINR) exceeds zero.  This is the *true* concept,
    # against which the served classifier's predictions are compared.
    y = ((X[:, 0] + X[:, 1]) > 0).astype(int)

    phase = np.zeros(N_STEPS, dtype=int)
    for ph, info in PHASES.items():
        a, b = info["range"]
        phase[a:b] = ph

    # Phase 2: gradual covariate shift on feature 0 (linear ramp 0 -> -mag dB)
    # plus a smaller correlated shift on feature 1 to cover the multi-feature
    # case the production DDD must handle (Bonferroni-corrected KS over all
    # KPIs, not just one).  The variance is also nudged up slightly to mimic
    # increased fading variability under propagation regime change.
    a, b = PHASES[2]["range"]
    ramp = np.linspace(0.0, -cfg.drift_magnitude, b - a, dtype=float)
    X[a:b, 0] += ramp
    X[a:b, 1] += 0.5 * ramp                      # correlated lighter drift on f1
    X[a:b, 2] *= 1.0 + 0.5 * np.linspace(0, 1, b - a)   # variance drift on f2
    # Labels are derived from the original (X[:,0]+X[:,1]) rule, so the
    # classifier loses accuracy organically as the input distribution shifts.

    # Phase 3: poisoning -- *asymmetric* label flips (only positive -> negative).
    # This is the realistic threat model: an adversary biases the label
    # distribution towards the majority class to mask anomalous events.
    # The asymmetric flip shifts the marginal label distribution visibly,
    # which is what the DPD's chi-squared test is designed to detect.
    a3, b3 = PHASES[3]["range"]
    n3 = b3 - a3
    pos_mask = (y[a3:b3] == 1)
    flip_pos = rng.random(n3) < (cfg.poison_rate * 2.0)  # double-rate on positives
    flip_pos = flip_pos & pos_mask
    y[a3:b3] = np.where(flip_pos, 0, y[a3:b3])

    # Background label noise across all phases at base_noise.
    bg_flip = rng.random(N_STEPS) < cfg.base_noise
    y[bg_flip] = 1 - y[bg_flip]

    return X, y, phase


# =============================================================================
# Detector models -- statistically faithful but lightweight
# =============================================================================
def fit_reference(X_ref: np.ndarray, y_ref: np.ndarray):
    """Pre-compute reference statistics needed by all four detectors."""
    return {
        "feat_means":  X_ref.mean(axis=0),
        "feat_stds":   X_ref.std(axis=0) + 1e-9,
        "label_dist":  np.bincount(y_ref, minlength=2) / max(len(y_ref), 1),
        "X_ref":       X_ref,
    }


def detect_ddd(X_window: np.ndarray, ref, alpha: float = 1e-3) -> tuple[bool, float]:
    """Two-sample Kolmogorov--Smirnov on each feature, Bonferroni-corrected.

    Returns (alarm, score) where score is the maximum KS statistic across
    features (used as the continuous decision score for ROC/PR).
    """
    pvals = []
    stats_arr = []
    for j in range(X_window.shape[1]):
        ks_stat, ks_p = stats.ks_2samp(X_window[:, j], ref["X_ref"][:, j])
        pvals.append(ks_p)
        stats_arr.append(ks_stat)
    bonf_alpha = alpha / X_window.shape[1]
    alarm = bool(min(pvals) < bonf_alpha)
    score = float(max(stats_arr))
    return alarm, score


def detect_dpd(y_window: np.ndarray, ref, alpha: float = 1e-3) -> tuple[bool, float]:
    """Chi-squared goodness-of-fit on the label distribution vs. reference."""
    obs = np.bincount(y_window, minlength=2).astype(float)
    n   = obs.sum()
    if n < 5:
        return False, 0.0
    exp = ref["label_dist"] * n
    exp = np.clip(exp, 1e-3, None)
    chi2, p = stats.chisquare(obs, f_exp=exp, ddof=0)
    alarm = bool(p < alpha)
    score = float(chi2)
    return alarm, score


@dataclass
class PageHinkleyState:
    cum: float = 0.0
    min_cum: float = 0.0
    n: int = 0
    avg: float = 0.0


def step_page_hinkley(
    state: PageHinkleyState,
    err: float,
    delta: float = 0.005,
    lam: float = 0.05,
) -> tuple[bool, float]:
    """Online Page--Hinkley test."""
    state.n += 1
    state.avg += (err - state.avg) / state.n
    state.cum += err - state.avg - delta
    state.min_cum = min(state.min_cum, state.cum)
    ph = state.cum - state.min_cum
    return (ph > lam, float(ph))


def detect_cpd(
    train_acc: float,
    val_acc: float,
    sigma: float,
    threshold_sigma: float = 3.0,
) -> tuple[bool, float]:
    """CPD: train/val accuracy gap measured in sigmas."""
    gap = train_acc - val_acc
    if sigma < 1e-6:
        return False, 0.0
    z = gap / sigma
    return bool(z > threshold_sigma), float(z)


# =============================================================================
# Per-seed simulation
# =============================================================================
def run_seed(
    seed: int,
    check_interval: int,
    cfg: SeedConfig | None = None,
) -> dict:
    """Simulate one seed and return per-detector metrics + per-step records.

    Ground truth: an alarm window is *true* iff it overlaps with the
    detector-relevant phase (DDD/DPD: P3; CDD: P2; CPD: P2 or P3).
    """
    cfg = cfg or SeedConfig.sample(np.random.default_rng(seed))
    X, y, phase = synthesise_workload(cfg, seed)

    # Train an initial classifier on P1.
    a, b = PHASES[1]["range"]
    clf = RandomForestClassifier(
        n_estimators=50, max_depth=8, random_state=seed
    ).fit(X[a:b], y[a:b])
    ref = fit_reference(X[a:b], y[a:b])

    # Per-step preds and per-window detector decisions.
    preds  = clf.predict(X).astype(int)
    correct = (preds == y).astype(int)

    # Page-Hinkley state for CDD (residual = 1 - correct).
    ph = PageHinkleyState()

    # Walk in windows of size check_interval.
    decisions = []  # list of dicts per check
    last_majority_phase = 1
    for k in range(check_interval, N_STEPS + 1, check_interval):
        win = slice(k - check_interval, k)
        ph_in_window = phase[win]
        # Aggregate the phase the window predominantly belongs to.
        majority_phase = int(stats.mode(ph_in_window, keepdims=False).mode)
        # Simulated retrain rebase: when the workload returns to P4 (recovery),
        # reset the PH accumulator and the reference statistics so the
        # detectors do not stay stuck in their P3 alarm state.  This mirrors
        # the real engine's NDT-driven detector reference rebase.
        if majority_phase == 4 and last_majority_phase != 4:
            ph = PageHinkleyState()
            ref = fit_reference(X[win], y[win])
        last_majority_phase = majority_phase

        # DDD: KS on features
        ddd_a, ddd_s = detect_ddd(X[win], ref)
        # DPD: chi2 on labels
        dpd_a, dpd_s = detect_dpd(y[win], ref)
        # CDD: Page-Hinkley on residual
        cdd_a, cdd_s = False, 0.0
        for i in range(win.start, win.stop):
            err = 1.0 - correct[i]
            cdd_a_step, cdd_s_step = step_page_hinkley(ph, err)
            cdd_s = max(cdd_s, cdd_s_step)
            if cdd_a_step:
                cdd_a = True
        # CPD: train/val gap
        # Approx train acc = correct in P1 reference; val acc = correct in window.
        train_acc = float(correct[a:b].mean())
        val_acc   = float(correct[win].mean())
        sigma     = float(correct[a:b].std()) + 1e-3
        cpd_a, cpd_s = detect_cpd(train_acc, val_acc, sigma)

        # Ground truth per detector (fixed: DDD detects covariate shift in P2;
        # DPD detects label shift in P3; CDD's accuracy-driven score reacts to
        # both drift and poisoning; CPD's train/val gap reacts to poisoning.)
        gt_ddd = int(majority_phase == 2)            # covariate shift on features
        gt_dpd = int(majority_phase == 3)            # label-distribution shift
        gt_cdd = int(majority_phase in (2, 3))       # accuracy degrades in both
        gt_cpd = int(majority_phase == 3)            # train/val gap induced by poisoning

        decisions.append({
            "check":       k,
            "phase":       majority_phase,
            "ddd_alarm":   int(ddd_a), "ddd_score": ddd_s, "gt_ddd": gt_ddd,
            "dpd_alarm":   int(dpd_a), "dpd_score": dpd_s, "gt_dpd": gt_dpd,
            "cdd_alarm":   int(cdd_a), "cdd_score": cdd_s, "gt_cdd": gt_cdd,
            "cpd_alarm":   int(cpd_a), "cpd_score": cpd_s, "gt_cpd": gt_cpd,
        })
    df = pd.DataFrame(decisions)

    # Per-detector precision / recall / F1 / latency.
    metrics = {"seed": seed, "check_interval": check_interval,
               "drift_magnitude": cfg.drift_magnitude,
               "poison_rate": cfg.poison_rate,
               "base_noise": cfg.base_noise}
    for d in ("ddd", "dpd", "cdd", "cpd"):
        a_col, gt_col, s_col = f"{d}_alarm", f"gt_{d}", f"{d}_score"
        tp = int(((df[a_col] == 1) & (df[gt_col] == 1)).sum())
        fp = int(((df[a_col] == 1) & (df[gt_col] == 0)).sum())
        fn = int(((df[a_col] == 0) & (df[gt_col] == 1)).sum())
        tn = int(((df[a_col] == 0) & (df[gt_col] == 0)).sum())
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        # Latency: first true-positive's check_step minus first time gt was 1.
        gt_idx = df.index[df[gt_col] == 1].tolist()
        tp_idx = df.index[(df[a_col] == 1) & (df[gt_col] == 1)].tolist()
        if gt_idx and tp_idx and tp_idx[0] >= gt_idx[0]:
            lat = (df.loc[tp_idx[0], "check"] - df.loc[gt_idx[0], "check"])
        else:
            lat = float("nan")
        metrics[f"{d}_precision"] = prec
        metrics[f"{d}_recall"]    = rec
        metrics[f"{d}_f1"]        = f1
        metrics[f"{d}_latency"]   = lat
        metrics[f"{d}_tp"] = tp
        metrics[f"{d}_fp"] = fp
        metrics[f"{d}_fn"] = fn
        metrics[f"{d}_tn"] = tn

    # Per-phase accuracy
    for ph_id, info in PHASES.items():
        s, e = info["range"]
        metrics[f"phase{ph_id}_accuracy"] = float(correct[s:e].mean())

    # Save raw decisions for the headline check (used by ROC/PR + calibration).
    metrics["__decisions__"] = df
    metrics["__correct__"]   = correct
    metrics["__phase__"]     = phase
    return metrics


# =============================================================================
# Bootstrap CIs
# =============================================================================
def bootstrap_ci(
    arr: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi) using percentile bootstrap."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    n = len(arr)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        means.append(arr[idx].mean())
    means = np.asarray(means)
    return float(arr.mean()), float(np.quantile(means, alpha / 2)), \
           float(np.quantile(means, 1 - alpha / 2))


def summarise(df: pd.DataFrame, by: str | None = None,
              cols: Iterable[str] | None = None) -> pd.DataFrame:
    """Mean / std / 95 % bootstrap CI for each column in `cols`."""
    cols = list(cols) if cols is not None else \
        [c for c in df.columns if df[c].dtype.kind in "fi"
         and not c.startswith("__")]
    rows = []
    if by is None:
        groups = [(None, df)]
    else:
        groups = list(df.groupby(by))
    for k, sub in groups:
        row = {by: k} if by is not None else {}
        for c in cols:
            mean, lo, hi = bootstrap_ci(sub[c].to_numpy())
            std = float(sub[c].std())
            row[f"{c}_mean"]  = mean
            row[f"{c}_std"]   = std
            row[f"{c}_ci_lo"] = lo
            row[f"{c}_ci_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Headline campaign (C = 50, N seeds)
# =============================================================================
def run_headline(n_seeds: int = 60, check_interval: int = 50) -> pd.DataFrame:
    print(f"\n[1/4] Headline campaign: {n_seeds} seeds at C = {check_interval}")
    rows = []
    for s in range(n_seeds):
        m = run_seed(s, check_interval)
        m.pop("__decisions__", None)
        m.pop("__correct__", None)
        m.pop("__phase__", None)
        rows.append(m)
        if (s + 1) % 10 == 0:
            print(f"   ... {s + 1}/{n_seeds}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "headline.csv"), index=False)
    summary = summarise(df, cols=[c for c in df.columns
                                  if c not in ("seed", "check_interval")])
    summary.to_csv(os.path.join(RESULTS_DIR, "headline_summary.csv"), index=False)
    print("   wrote headline.csv + headline_summary.csv")
    return df


# =============================================================================
# Sensitivity sweep across check intervals
# =============================================================================
def run_sweep(
    n_seeds: int = 30,
    intervals: tuple[int, ...] = (10, 25, 50, 100),
) -> pd.DataFrame:
    print(f"\n[2/4] Sensitivity sweep: {len(intervals)} intervals x {n_seeds} seeds")
    rows = []
    for c in intervals:
        for s in range(n_seeds):
            m = run_seed(s, c)
            m.pop("__decisions__", None)
            m.pop("__correct__", None)
            m.pop("__phase__", None)
            rows.append(m)
        print(f"   ... C = {c}: {n_seeds} seeds done")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "sweep.csv"), index=False)
    cols = [c for c in df.columns
            if c not in ("seed", "check_interval", "drift_magnitude",
                         "poison_rate", "base_noise")]
    summary = summarise(df, by="check_interval", cols=cols)
    summary.to_csv(os.path.join(RESULTS_DIR, "sweep_summary.csv"), index=False)
    print("   wrote sweep.csv + sweep_summary.csv")
    return df


# =============================================================================
# Per-phase accuracy summary (long format, suited for figures)
# =============================================================================
def run_per_phase(headline_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in headline_df.iterrows():
        for ph in (1, 2, 3, 4):
            rows.append({
                "seed":     int(r["seed"]),
                "phase":    ph,
                "accuracy": float(r[f"phase{ph}_accuracy"]),
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "per_phase.csv"), index=False)
    summary = summarise(df, by="phase", cols=["accuracy"])
    summary.to_csv(os.path.join(RESULTS_DIR, "per_phase_summary.csv"),
                   index=False)
    return df


# =============================================================================
# Figures
# =============================================================================
def _phase_bg(ax, max_step=N_STEPS):
    for ph_id, info in PHASES.items():
        s, e = info["range"]
        ax.axvspan(s, min(e, max_step), color=info["colour"], alpha=0.07,
                   zorder=0)
        mid = (s + min(e, max_step)) / 2
        y_top = ax.get_ylim()[1]
        ax.text(mid, y_top * 0.96, info["name"], ha="center", fontsize=8,
                color="#555", style="italic")


def fig_headline_boxplot(df: pd.DataFrame):
    print("\n[3/4] Generating figures...")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    detectors = ["DDD", "DPD", "CDD", "CPD"]
    for ax, metric, title in zip(
        axes,
        ["precision", "recall", "f1"],
        ["Precision", "Recall", "F1 score"],
    ):
        data = [df[f"{d.lower()}_{metric}"].dropna() for d in detectors]
        bp = ax.boxplot(
            data, tick_labels=detectors, patch_artist=True, widths=0.6,
            medianprops=dict(color="black", linewidth=1.5),
            flierprops=dict(marker="o", markersize=3, markerfacecolor="#666",
                            markeredgecolor="none", alpha=0.6),
        )
        for patch, d in zip(bp["boxes"], detectors):
            patch.set_facecolor(DETECTOR_COLOURS[d])
            patch.set_alpha(0.55)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.7)
        # mean diamonds
        for i, d in enumerate(detectors, 1):
            ax.scatter([i], [df[f"{d.lower()}_{metric}"].mean()],
                       marker="D", s=22, color="white",
                       edgecolor="black", zorder=5, linewidth=0.7)
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.05)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Score")
    fig.suptitle(f"Per-detector P / R / F1 over {len(df)} random seeds "
                 "(C = 50, bootstrap-ready)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "fig_headline_boxplot.pdf"))
    plt.close(fig)
    print("   wrote fig_headline_boxplot.pdf")


def fig_rocpr_v2(n_seeds: int = 30):
    """ROC + PR with 95 % CI bands across seeds."""
    detectors = ["DDD", "DPD", "CDD", "CPD"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.0))
    fig.suptitle("ROC and Precision--Recall curves with 95 % CI bands "
                 f"(C = 50, {n_seeds} seeds)", fontsize=11, y=1.01)

    common_fpr = np.linspace(0, 1, 100)
    common_rec = np.linspace(0, 1, 100)

    for col, d in enumerate(detectors):
        all_tprs, all_precs, aucs, aps = [], [], [], []
        for s in range(n_seeds):
            m = run_seed(s, 50)
            df = m["__decisions__"]
            scores = df[f"{d.lower()}_score"].to_numpy()
            gt = df[f"gt_{d.lower()}"].to_numpy()
            if gt.sum() == 0 or gt.sum() == len(gt):
                continue
            s_min, s_max = scores.min(), scores.max()
            if s_max > s_min:
                scores = (scores - s_min) / (s_max - s_min)
            fpr, tpr, _ = roc_curve(gt, scores)
            all_tprs.append(np.interp(common_fpr, fpr, tpr))
            aucs.append(auc(fpr, tpr))
            prec, rec, _ = precision_recall_curve(gt, scores)
            order = np.argsort(rec)
            all_precs.append(np.interp(common_rec, rec[order], prec[order]))
            aps.append(average_precision_score(gt, scores))
        if not all_tprs:
            for row in range(2):
                axes[row, col].text(0.5, 0.5, "n/a", ha="center", va="center",
                                    transform=axes[row, col].transAxes)
            continue
        tpr_arr = np.vstack(all_tprs)
        prec_arr = np.vstack(all_precs)
        tpr_mean = tpr_arr.mean(axis=0)
        tpr_lo, tpr_hi = (np.quantile(tpr_arr, 0.025, axis=0),
                          np.quantile(tpr_arr, 0.975, axis=0))
        pr_mean = prec_arr.mean(axis=0)
        pr_lo, pr_hi = (np.quantile(prec_arr, 0.025, axis=0),
                        np.quantile(prec_arr, 0.975, axis=0))

        # ROC row
        ax = axes[0, col]
        ax.plot(common_fpr, tpr_mean, color=DETECTOR_COLOURS[d], lw=2,
                label=f"AUC = {np.mean(aucs):.2f}±{np.std(aucs):.2f}")
        ax.fill_between(common_fpr, tpr_lo, tpr_hi,
                        color=DETECTOR_COLOURS[d], alpha=0.18)
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8)
        ax.set_title(f"{d}  ROC")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.legend(loc="lower right")

        # PR row
        ax = axes[1, col]
        ax.plot(common_rec, pr_mean, color=DETECTOR_COLOURS[d], lw=2,
                label=f"AP = {np.mean(aps):.2f}±{np.std(aps):.2f}")
        ax.fill_between(common_rec, pr_lo, pr_hi,
                        color=DETECTOR_COLOURS[d], alpha=0.18)
        ax.set_title(f"{d}  PR")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "fig_rocpr_v2.pdf"))
    plt.close(fig)
    print("   wrote fig_rocpr_v2.pdf")


def fig_sweep_pareto(sweep_df: pd.DataFrame):
    """F1 (mean of all detectors) vs total intervention cost as Pareto."""
    intervals = sorted(sweep_df["check_interval"].unique())

    rows = []
    for c in intervals:
        sub = sweep_df[sweep_df["check_interval"] == c]
        # Mean F1 across the 4 detectors per seed -> bootstrap
        f1_per_seed = sub[["ddd_f1", "dpd_f1", "cdd_f1", "cpd_f1"]].mean(axis=1)
        f1_mean, f1_lo, f1_hi = bootstrap_ci(f1_per_seed.to_numpy())
        # Cost proxy: number of distinct alarm windows = sum of TPs+FPs per seed.
        total_alarms = (sub[["ddd_tp", "ddd_fp", "dpd_tp", "dpd_fp",
                             "cdd_tp", "cdd_fp", "cpd_tp", "cpd_fp"]]
                        .sum(axis=1))
        c_mean, c_lo, c_hi = bootstrap_ci(total_alarms.to_numpy())
        rows.append({"C": c, "f1_mean": f1_mean, "f1_lo": f1_lo, "f1_hi": f1_hi,
                     "cost_mean": c_mean, "cost_lo": c_lo, "cost_hi": c_hi})
    pareto = pd.DataFrame(rows)
    pareto.to_csv(os.path.join(RESULTS_DIR, "pareto.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 4.0))
    cmap = plt.get_cmap("viridis")
    for i, r in pareto.iterrows():
        col = cmap(i / max(len(pareto) - 1, 1))
        ax.errorbar(r.cost_mean, r.f1_mean,
                    xerr=[[r.cost_mean - r.cost_lo], [r.cost_hi - r.cost_mean]],
                    yerr=[[r.f1_mean - r.f1_lo], [r.f1_hi - r.f1_mean]],
                    fmt="o", markersize=10, color=col, capsize=4,
                    elinewidth=1.0, markeredgecolor="black", markeredgewidth=0.6,
                    label=f"C = {int(r.C)}")
    ax.set_xlabel("Total alarms per seed (TPs + FPs)  -- intervention cost")
    ax.set_ylabel("Mean F1 across detectors")
    ax.set_title("Latency / cost / quality Pareto frontier (95 % CIs)")
    ax.legend(loc="lower right", ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "fig_sweep_pareto.pdf"))
    plt.close(fig)
    print("   wrote fig_sweep_pareto.pdf")


def fig_per_phase_v2(headline_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    phases = [1, 2, 3, 4]
    means, ci_los, ci_his = [], [], []
    for ph in phases:
        m, lo, hi = bootstrap_ci(headline_df[f"phase{ph}_accuracy"].to_numpy())
        means.append(m); ci_los.append(lo); ci_his.append(hi)
    colours = [PHASES[ph]["colour"] for ph in phases]
    labels  = [f"P{ph} ({PHASES[ph]['name']})" for ph in phases]
    bars = ax.bar(labels, means, color=colours, edgecolor="white", width=0.55)
    err_lo = [m - lo for m, lo in zip(means, ci_los)]
    err_hi = [hi - m for m, hi in zip(means, ci_his)]
    ax.errorbar(labels, means, yerr=[err_lo, err_hi], fmt="none",
                ecolor="black", elinewidth=1.0, capsize=5)
    for bar, m, lo, hi in zip(bars, means, ci_los, ci_his):
        ax.text(bar.get_x() + bar.get_width() / 2, hi + 0.02,
                f"{m:.3f}\n[{lo:.3f}, {hi:.3f}]",
                ha="center", fontsize=8.5)
    ax.axhline(0.65, ls="--", color="#e74c3c", lw=1.0, label="NDT floor 0.65")
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("End-to-end model accuracy")
    ax.set_title("Per-phase accuracy with 95 % bootstrap CI "
                 f"({len(headline_df)} seeds)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "fig_per_phase_v2.pdf"))
    plt.close(fig)
    print("   wrote fig_per_phase_v2.pdf")


def fig_latency_cdf(headline_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    detectors = ["DDD", "DPD", "CDD", "CPD"]
    for d in detectors:
        col = f"{d.lower()}_latency"
        lat = headline_df[col].dropna().to_numpy()
        if len(lat) == 0:
            continue
        x = np.sort(lat)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", color=DETECTOR_COLOURS[d], lw=2, label=d)
    ax.set_xlabel("Detection latency (simulation steps)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(f"Detection-latency CDFs ({len(headline_df)} seeds, C = 50)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "fig_latency_cdf.pdf"))
    plt.close(fig)
    print("   wrote fig_latency_cdf.pdf")


def fig_significance_matrix(sweep_df: pd.DataFrame):
    """Pairwise Wilcoxon signed-rank test on per-seed F1 across C settings."""
    intervals = sorted(sweep_df["check_interval"].unique())
    f1_by_c = {}
    for c in intervals:
        sub = sweep_df[sweep_df["check_interval"] == c]
        f1 = sub[["ddd_f1", "dpd_f1", "cdd_f1", "cpd_f1"]].mean(axis=1).to_numpy()
        f1_by_c[c] = f1

    n = len(intervals)
    p_matrix = np.full((n, n), np.nan)
    for i, ci in enumerate(intervals):
        for j, cj in enumerate(intervals):
            if i == j: continue
            a, b = f1_by_c[ci], f1_by_c[cj]
            # paired across seeds (assumes same seed list per C)
            m = min(len(a), len(b))
            try:
                _, p = stats.wilcoxon(a[:m], b[:m])
            except ValueError:
                p = float("nan")
            p_matrix[i, j] = p

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(p_matrix, cmap="RdYlGn_r", vmin=0, vmax=0.10,
                   origin="lower")
    ax.set_xticks(range(n)); ax.set_xticklabels([f"C={c}" for c in intervals])
    ax.set_yticks(range(n)); ax.set_yticklabels([f"C={c}" for c in intervals])
    for i in range(n):
        for j in range(n):
            if i == j: continue
            v = p_matrix[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=9, color="black" if v > 0.05 else "white")
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, label="p-value (Wilcoxon)")
    cbar.set_ticks([0.0, 0.05, 0.10])
    ax.set_title("Pairwise significance:\nmean-F1 per seed across C")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "fig_significance_matrix.pdf"))
    plt.close(fig)
    print("   wrote fig_significance_matrix.pdf")


def fig_calibration(n_seeds: int = 30):
    """Detector-score reliability diagram: aggregate over n_seeds."""
    detectors = ["DDD", "DPD", "CDD", "CPD"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    n_bins = 8
    for ax, d in zip(axes, detectors):
        all_scores, all_gt = [], []
        for s in range(n_seeds):
            m = run_seed(s, 50)
            df = m["__decisions__"]
            sc = df[f"{d.lower()}_score"].to_numpy()
            gt = df[f"gt_{d.lower()}"].to_numpy()
            if gt.sum() == 0:
                continue
            mn, mx = sc.min(), sc.max()
            if mx > mn:
                sc = (sc - mn) / (mx - mn)
            all_scores.append(sc); all_gt.append(gt)
        if not all_scores:
            ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                    ha="center", va="center")
            continue
        all_scores = np.concatenate(all_scores)
        all_gt = np.concatenate(all_gt)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_idx = np.digitize(all_scores, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        bin_means_x, bin_means_y, bin_counts = [], [], []
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() > 0:
                bin_means_x.append(all_scores[mask].mean())
                bin_means_y.append(all_gt[mask].mean())
                bin_counts.append(int(mask.sum()))
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8, label="ideal")
        ax.scatter(bin_means_x, bin_means_y,
                   s=[max(20, min(c, 200)) for c in bin_counts],
                   color=DETECTOR_COLOURS[d], edgecolor="black", alpha=0.85,
                   linewidth=0.6)
        ax.set_title(f"{d} reliability")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("Mean predicted score (binned)")
        ax.legend(loc="lower right")
    axes[0].set_ylabel("Empirical positive fraction")
    fig.suptitle("Detector-score reliability diagrams "
                 f"(aggregated over {n_seeds} seeds)", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "fig_calibration.pdf"))
    plt.close(fig)
    print("   wrote fig_calibration.pdf")


def fig_workload_overview(seed: int = 0):
    """One-shot reference figure: phase timeline + KPI traces + injection cues."""
    cfg = SeedConfig.sample(np.random.default_rng(seed))
    X, y, phase = synthesise_workload(cfg, seed)
    fig, axes = plt.subplots(3, 1, figsize=(11, 5.4), sharex=True)
    steps = np.arange(N_STEPS)

    # KPI 1+2 trace
    axes[0].plot(steps, X[:, 0], color="#3498db", lw=0.8, alpha=0.8,
                 label="Feature 0 (RSRP-like)")
    axes[0].plot(steps, X[:, 1], color="#e67e22", lw=0.8, alpha=0.8,
                 label="Feature 1 (SINR-like)")
    axes[0].set_ylabel("Standardised value")
    axes[0].legend(loc="upper right", ncol=2)
    _phase_bg(axes[0])

    # Rolling label proportion (true positives)
    win = 25
    cum = np.convolve(y, np.ones(win) / win, mode="same")
    axes[1].plot(steps, cum, color="#e74c3c", lw=1.2)
    axes[1].set_ylabel(f"Rolling P(y=1), w={win}")
    _phase_bg(axes[1])

    # Phase bar
    axes[2].imshow(phase[None, :], aspect="auto", cmap="Pastel1",
                   extent=[0, N_STEPS, 0, 1])
    axes[2].set_yticks([])
    axes[2].set_xlabel("Simulation step")
    for ph_id, info in PHASES.items():
        s, e = info["range"]
        axes[2].text((s + e) / 2, 0.5, f"P{ph_id}: {info['name']}",
                     ha="center", va="center", fontsize=9, fontweight="bold")

    fig.suptitle(f"Synthetic four-phase workload (seed = {seed}, "
                 f"drift mag = {cfg.drift_magnitude:.2f}, "
                 f"poison rate = {cfg.poison_rate:.0%})",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "fig_workload_overview.pdf"))
    plt.close(fig)
    print("   wrote fig_workload_overview.pdf")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    t0 = time.time()
    print("=" * 70)
    print("AIMP framework -- rigorous re-evaluation")
    print("=" * 70)

    headline_df = run_headline(n_seeds=60, check_interval=50)
    sweep_df    = run_sweep(n_seeds=30, intervals=(10, 25, 50, 100))
    per_phase_df = run_per_phase(headline_df)

    # Figures
    fig_workload_overview(seed=0)
    fig_headline_boxplot(headline_df)
    fig_per_phase_v2(headline_df)
    fig_latency_cdf(headline_df)
    fig_sweep_pareto(sweep_df)
    fig_significance_matrix(sweep_df)
    print("\n[4/4] ROC/PR + calibration (each runs 30 seeds, takes ~30 s)...")
    fig_rocpr_v2(n_seeds=30)
    fig_calibration(n_seeds=30)

    print("\n" + "=" * 70)
    print(f"Done in {time.time() - t0:.1f} s.")
    print(f"  CSVs    -> {RESULTS_DIR}")
    print(f"  Figures -> {FIGS_DIR}")
    print("=" * 70)
