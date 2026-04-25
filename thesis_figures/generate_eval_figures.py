"""
generate_eval_figures.py
Generates all evaluation figures for the thesis expansions.

Run from the project root:
    .venv\\Scripts\\python.exe thesis/generate_eval_figures.py
"""
from __future__ import annotations

import sys
import os
import io
import time
import timeit
import warnings

# Fix Windows console encoding for special chars
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from aif.aif import AIF
from rtp.rtp import RTP, RTPConfig
from atm.atm import ATM, ATMPolicy, ATMResult, MTPVariant
from atm.mtp_l import MTPLocal
from ndt.ndt import NDT

# ── Output dir ────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

PHASE_RANGES = {1: (1, 400), 2: (401, 600), 3: (601, 700), 4: (701, 900)}
PHASE_COLOURS_BG = {
    1: "#27ae60", 2: "#e67e22", 3: "#e74c3c", 4: "#3498db"
}
PHASE_NAMES = {1: "Stable", 2: "Concept Drift", 3: "Data Poisoning", 4: "Recovery"}

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


def save(name: str):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close("all")
    print("  saved -> " + path)


def phase_bg(ax, max_step=900):
    """Add translucent phase bands to a matplotlib axis."""
    alpha_vals = [0.06, 0.09, 0.12, 0.06]
    for i, (ph, (s, e)) in enumerate(PHASE_RANGES.items()):
        ax.axvspan(s, min(e, max_step),
                   color=PHASE_COLOURS_BG[ph], alpha=alpha_vals[i], zorder=0)
        mid = (s + min(e, max_step)) / 2
        ax.text(mid, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 1,
                PHASE_NAMES[ph], ha="center", fontsize=7.5, color="#666", style="italic")


# =============================================================================
# 1.  Core simulation builder
# =============================================================================

def _make_data_fn(rng):
    """Return a factory function that generates data using the given rng."""
    def make_data(n: int, noise: float = 0.05):
        X = rng.normal(0, 1, size=(n, 4))
        y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
        flip = rng.random(n) < noise
        y[flip] = 1 - y[flip]
        return X.astype(float), y.astype(int)
    return make_data


def _build_sim(seed=42, local_only=True):
    """Build and return a fully wired (rtp, make_data) pair."""
    rng = np.random.default_rng(seed)
    make_data = _make_data_fn(rng)

    X_train, y_train = make_data(500, 0.05)
    clf = RandomForestClassifier(n_estimators=50, random_state=1).fit(X_train, y_train)
    aif = AIF(clf)

    cfg = RTPConfig(
        buffer_maxlen=2000, check_interval=50, cdd_task="classifier",
        ddd_reference_size=200, ddd_recent_size=100,
        dpd_reference_size=200, dpd_recent_size=50,
        dpd_contamination_threshold=0.08, dpd_mahal_threshold=5.0,
        cdd_reference_window=150, cdd_recent_window=50,
        cdd_perf_drop_threshold=0.12, cdd_ph_lambda=40.0,
        cpd_reference_size=200, cpd_recent_size=100,
        cpd_shadow_threshold=0.38, cpd_output_ks_alpha=0.0001,
        cpd_corr_threshold=0.60, mtout_cooldown_steps=150,
    )
    rtp = RTP(aif, config=cfg)

    mtp_l = MTPLocal(n_splits=3, fine_tune_first=True)
    ndt = NDT(current_model_getter=lambda: rtp.aif.active_estimator,
              min_score=0.65, min_improvement=-0.05)
    policy = ATMPolicy(
        prefer_variant=MTPVariant.LOCAL,   # always use MTP-L
        local_max_samples=600,
        use_ndt=True, ndt_min_accuracy=0.65,
        auto_deploy=True, max_retrain_attempts=2,
    )
    atm = ATM(rtp=rtp, mtp_l=mtp_l, mtp_e=None, ndt=ndt, policy=policy)
    rtp._on_mtout = atm.handle

    X_ref, y_ref = make_data(300, 0.05)
    lob_ref = clf.predict(X_ref).astype(float)
    rtp.set_reference(X_ref, y_ref, lob_ref)
    return rtp, make_data, rng, clf


def _run_phases(rtp, make_data, rng, collect_fn=None):
    """Run all 4 phases; call collect_fn(step, phase_num, pred, y_true) every step."""
    def run_phase(X, y, ph):
        for i in range(len(X)):
            pred = rtp.observe(X[i], y_true=float(y[i]))
            prediction = int(pred.ravel()[0])
            correct = prediction == int(y[i])
            if collect_fn:
                collect_fn(rtp._step, ph, prediction, int(y[i]), correct,
                           rtp.last_ddd, rtp.last_dpd, rtp.last_cdd, rtp.last_cpd)

    run_phase(*make_data(400, 0.05), 1)
    run_phase(*make_data(200, 0.55), 2)

    X_c, y_c = make_data(90, 0.05)
    X_inj = rng.uniform(30, 50, (10, 4)).astype(float)
    y_inj = rng.integers(0, 2, 10).astype(int)
    X3 = np.vstack([X_c, X_inj])
    y3 = np.concatenate([y_c, y_inj])
    idx = rng.permutation(len(X3))
    run_phase(X3[idx], y3[idx], 3)

    run_phase(*make_data(200, 0.05), 4)


# =============================================================================
# 2.  Full simulation — collect per-step and per-check records
# =============================================================================

def run_full_data_collection(seed=42):
    rtp, make_data, rng, clf = _build_sim(seed)

    steps = []        # list of dicts, one per step
    checks = []       # list of dicts, one per check (every 50 steps)
    last_check = [{}] # carry-forward

    def collect(step, phase, pred, y_true, correct, ddd, dpd, cdd, cpd):
        steps.append({"step": step, "phase": phase,
                       "pred": pred, "y_true": y_true, "correct": correct})

        if step % 50 == 0:
            # Ground-truth labels: is this a "true anomaly" check interval?
            gt_ddd = 1 if phase == 3 else 0          # input data poisoning
            gt_dpd = 1 if phase == 3 else 0          # data poisoning
            gt_cdd = 1 if phase == 2 else 0          # concept drift
            gt_cpd = 1 if phase in (2, 3) else 0     # concept poisoning

            def safe_float(v):
                try:
                    return float(v) if v is not None else 0.0
                except Exception:
                    return 0.0

            check = {
                "step": step, "phase": phase,
                "gt_ddd": gt_ddd, "gt_dpd": gt_dpd,
                "gt_cdd": gt_cdd, "gt_cpd": gt_cpd,
                "ddd_score": safe_float(ddd.mmd_statistic) if ddd else 0.0,
                "ddd_pred":  int(ddd.drift_detected) if ddd else 0,
                "dpd_score": safe_float(dpd.mahal_max) if dpd else 0.0,
                "dpd_pred":  int(dpd.poisoning_detected) if dpd else 0,
                "cdd_score": safe_float(cdd.ph_statistic) if cdd else 0.0,
                "cdd_pred":  int(cdd.drift_detected) if cdd else 0,
                "cpd_score": safe_float(cpd.shadow_divergence) if cpd else 0.0,
                "cpd_pred":  int(cpd.poisoning_detected) if cpd else 0,
            }
            checks.append(check)
            last_check[0] = check

    _run_phases(rtp, make_data, rng, collect_fn=collect)
    print("  collected {} steps, {} check records".format(len(steps), len(checks)))
    return steps, checks


# =============================================================================
# 3.  ROC / PR curves
# =============================================================================

def fig_roc_pr(checks):
    print("Generating ROC/PR curves...")
    if not checks:
        print("  WARNING: no check records, skipping")
        return

    detectors = [
        ("DDD", "ddd_score", "gt_ddd", "#8e44ad"),
        ("DPD", "dpd_score", "gt_dpd", "#e67e22"),
        ("CDD", "cdd_score", "gt_cdd", "#27ae60"),
        ("CPD", "cpd_score", "gt_cpd", "#e74c3c"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    fig.suptitle("ROC and Precision-Recall Curves (per detector)", fontsize=12, y=1.01)

    for col, (name, score_key, gt_key, colour) in enumerate(detectors):
        scores = np.array([c[score_key] for c in checks], dtype=float)
        gt     = np.array([c[gt_key]    for c in checks], dtype=int)

        # normalise scores to [0,1]
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            scores_n = (scores - s_min) / (s_max - s_min)
        else:
            scores_n = np.zeros_like(scores)

        if gt.sum() == 0 or gt.sum() == len(gt):
            for row in range(2):
                axes[row][col].text(0.5, 0.5, "Insufficient\nlabel variance",
                                    ha="center", va="center",
                                    transform=axes[row][col].transAxes, fontsize=9)
                axes[row][col].set_title(name, fontsize=10)
            continue

        # ROC
        fpr, tpr, _ = roc_curve(gt, scores_n)
        roc_auc = auc(fpr, tpr)
        axes[0][col].plot(fpr, tpr, color=colour, lw=2, label=f"AUC={roc_auc:.2f}")
        axes[0][col].plot([0, 1], [0, 1], "--", color="gray", lw=1)
        axes[0][col].set_title(f"{name} ROC", fontsize=10)
        axes[0][col].set_xlabel("FPR"); axes[0][col].set_ylabel("TPR")
        axes[0][col].legend(fontsize=9)
        axes[0][col].set_xlim([0, 1]); axes[0][col].set_ylim([0, 1.02])

        # PR
        prec, rec, _ = precision_recall_curve(gt, scores_n)
        ap = average_precision_score(gt, scores_n)
        axes[1][col].plot(rec, prec, color=colour, lw=2, label=f"AP={ap:.2f}")
        axes[1][col].set_title(f"{name} PR", fontsize=10)
        axes[1][col].set_xlabel("Recall"); axes[1][col].set_ylabel("Precision")
        axes[1][col].legend(fontsize=9)
        axes[1][col].set_xlim([0, 1]); axes[1][col].set_ylim([0, 1.02])

    plt.tight_layout()
    save("fig_roc_pr_curves.pdf")


# =============================================================================
# 4.  Ablation study
# =============================================================================

def _run_ablation_variant(seed, disable: str):
    """Run simulation with one detector disabled; return rolling accuracy array."""
    rtp, make_data, rng, clf = _build_sim(seed)

    # Monkey-patch disabled detector
    if disable == "DDD":
        from detectors.ddd import DDDResult
        _empty = DDDResult(drift_detected=False, ks_pvalues=[], ks_drifted_features=[],
                           mmd_statistic=0.0, mmd_threshold=0.05, mmd_triggered=False,
                           reference_size=200, recent_size=100, message="disabled")
        rtp.ddd.check = lambda lib: _empty
    elif disable == "DPD":
        from detectors.dpd import DPDResult
        _empty = DPDResult(poisoning_detected=False, if_anomaly_rate=0.0, if_threshold=0.08,
                           if_triggered=False, if_anomalous_indices=[],
                           mahal_max=0.0, mahal_threshold=5.0, mahal_triggered=False,
                           mahal_anomalous_indices=[], message="disabled")
        rtp.dpd.check = lambda lib: _empty
    elif disable == "CDD":
        from detectors.cdd import CDDResult
        _empty = CDDResult(drift_detected=False, ph_statistic=0.0, ph_threshold=40.0,
                           ph_triggered=False, perf_reference=0.0, perf_recent=0.0,
                           perf_drop=0.0, window_triggered=False, n_updates=0,
                           ground_truth_mode=True, message="disabled")
        rtp.cdd.check = lambda: _empty
    elif disable == "CPD":
        from detectors.cpd import CPDResult
        _empty = CPDResult(poisoning_detected=False, shadow_divergence=0.0,
                           shadow_threshold=0.38, shadow_triggered=False,
                           output_ks_pvalue=1.0, output_ks_threshold=0.0001,
                           output_ks_triggered=False, corr_delta_max=0.0,
                           corr_threshold=0.60, corr_triggered=False, message="disabled")
        rtp.cpd.check = lambda lib, lob: _empty

    corrects = []

    def collect(step, phase, pred, y_true, correct, *_):
        corrects.append(float(correct))

    _run_phases(rtp, make_data, rng, collect_fn=collect)

    arr = np.array(corrects, dtype=float)
    kernel = np.ones(50) / 50
    return np.convolve(arr, kernel, mode="same") * 100


def fig_ablation():
    print("Running ablation study (5 variants x 900 steps)...")
    SEED = 42
    steps = np.arange(1, 901)

    variants = [
        ("Full system", None,  "#2980b9", "-",  2.5),
        ("No DDD",      "DDD", "#8e44ad", "--", 1.6),
        ("No DPD",      "DPD", "#e67e22", "--", 1.6),
        ("No CDD",      "CDD", "#27ae60", "--", 1.6),
        ("No CPD",      "CPD", "#e74c3c", "--", 1.6),
    ]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(1, 900)
    ax.set_ylim(0, 105)

    for label, disable, colour, ls, lw in variants:
        roll = _run_ablation_variant(SEED, disable if disable else "_NONE_")
        ax.plot(steps[:len(roll)], roll, label=label,
                color=colour, ls=ls, lw=lw)

    # Phase bands
    for ph, (s, e) in PHASE_RANGES.items():
        ax.axvspan(s, e, color=PHASE_COLOURS_BG[ph], alpha=0.07, zorder=0)
        ax.text((s + e) / 2, 4, PHASE_NAMES[ph],
                ha="center", fontsize=8, color="#555", style="italic")

    ax.axhline(65, ls=":", color="#e74c3c", lw=1.3, label="NDT floor 65%")
    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Rolling accuracy (%)")
    ax.set_title("Ablation Study -- Effect of Disabling Each Detector")
    ax.legend(fontsize=9, loc="lower right", ncol=2)
    plt.tight_layout()
    save("fig_ablation_study.pdf")


# =============================================================================
# 5.  Computational overhead
# =============================================================================

def fig_overhead():
    print("Profiling computational overhead...")
    rng2 = np.random.default_rng(0)
    X_ref = rng2.normal(0, 1, (200, 4))
    X_rec = rng2.normal(0, 1, (100, 4))
    y_ref = rng2.integers(0, 2, 200).astype(int)
    p_ref = rng2.integers(0, 2, 200).astype(float)
    p_rec = rng2.integers(0, 2, 100).astype(float)

    from detectors.ddd import DDD
    from detectors.dpd import DPD
    from detectors.cdd import CDD
    from detectors.cpd import CPD
    from aif.buffers import BufferPair

    ddd = DDD(reference_size=200, recent_size=100, ks_alpha=0.05)
    ddd.fit_reference(X_ref)
    dpd = DPD(reference_size=200, recent_size=50)
    dpd.fit_reference(X_ref)
    cdd = CDD(task="classifier", reference_window=150, recent_window=50, ph_lambda=40.0)
    for xi, yi, pi in zip(X_ref, y_ref, p_ref):
        cdd.update(y_pred=np.array([pi]), y_true=float(yi))
    cpd = CPD(task="classifier", reference_size=200, recent_size=100, shadow_threshold=0.38)
    cpd.fit_reference(X_ref, y_ref, p_ref)

    buf = BufferPair(maxlen=2000)
    buf.push_batch(X_ref, p_ref)
    buf.push_batch(X_rec, p_rec)

    clf2 = RandomForestClassifier(n_estimators=50, random_state=1).fit(X_ref, y_ref)
    aif2 = AIF(clf2)

    N = 50
    timings = {
        "AIF.predict":  timeit.timeit(lambda: aif2.predict(X_rec[0]), number=N * 10) / (N * 10) * 1000,
        "DDD.check":    timeit.timeit(lambda: ddd.check(buf.lib), number=N) / N * 1000,
        "DPD.check":    timeit.timeit(lambda: dpd.check(buf.lib), number=N) / N * 1000,
        "CDD.check":    timeit.timeit(lambda: cdd.check(), number=N) / N * 1000,
        "CPD.check":    timeit.timeit(lambda: cpd.check(buf.lib, buf.lob), number=N) / N * 1000,
    }

    # Bar chart
    names = list(timings.keys())
    vals  = [timings[n] for n in names]
    colours = ["#3498db", "#8e44ad", "#e67e22", "#27ae60", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names, vals, color=colours, edgecolor="white", height=0.55)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.02 * max(vals), bar.get_y() + bar.get_height() / 2,
                f"{v:.2f} ms", va="center", fontsize=9)

    total = sum(vals)
    ax.axvline(total, color="gray", ls="--", lw=1)
    ax.text(total + 0.01 * max(vals), len(names) - 0.5,
            f"Total: {total:.2f} ms", fontsize=9, color="gray")

    ax.set_xlabel("Mean latency per call (ms)")
    ax.set_title(f"Computational Overhead (mean over {N} calls)")
    ax.set_xlim(0, max(vals) * 1.35)
    plt.tight_layout()
    save("fig_computational_overhead.pdf")
    return timings


# =============================================================================
# 6.  Per-phase accuracy summary
# =============================================================================

def fig_phase_accuracy(steps):
    print("Generating per-phase accuracy chart...")
    phases = [1, 2, 3, 4]
    accs = []
    for ph in phases:
        ph_steps = [s for s in steps if s["phase"] == ph]
        if ph_steps:
            acc = 100.0 * sum(s["correct"] for s in ph_steps) / len(ph_steps)
        else:
            acc = 0.0
        accs.append(acc)

    colours = [PHASE_COLOURS_BG[ph] for ph in phases]
    labels = [PHASE_NAMES[ph] for ph in phases]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(labels, accs, color=colours, edgecolor="white", width=0.55)
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8,
                f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")

    ax.axhline(65, ls="--", color="#e74c3c", lw=1.3, label="NDT floor 65%")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Mean accuracy (%)")
    ax.set_title("Per-Phase Model Accuracy")
    ax.legend(fontsize=9)
    plt.tight_layout()
    save("fig_per_phase_accuracy.pdf")


# =============================================================================
# 7.  Print LaTeX overhead table
# =============================================================================

def print_overhead_table(timings):
    print("")
    print("% ---- LaTeX overhead table ----")
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\caption{Computational overhead per component (mean over 50 calls, "
          "Python 3.13, i7 laptop).}")
    print("\\label{tab:overhead}")
    print("\\begin{tabular}{@{}lrr@{}}")
    print("\\toprule")
    print("\\textbf{Component} & \\textbf{Mean (ms)} "
          "& \\textbf{Budget per 50-step interval} \\\\")
    print("\\midrule")
    for name, ms in timings.items():
        budget = "negligible" if ms < 1 else f"{ms:.1f}\\,ms"
        print(f"{name} & {ms:.3f} & {budget} \\\\")
    total = sum(timings.values())
    print("\\midrule")
    print(f"\\textbf{{Full detector battery}} & \\textbf{{{total:.3f}}} "
          f"& \\textbf{{{total:.1f}\\,ms}} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RTP Observer -- Thesis Evaluation Figure Generator")
    print("=" * 60)

    print("\n[1/5] Collecting simulation data...")
    steps, checks = run_full_data_collection(seed=42)

    print("\n[2/5] ROC / PR curves...")
    fig_roc_pr(checks)

    print("\n[3/5] Ablation study...")
    fig_ablation()

    print("\n[4/5] Computational overhead...")
    timings = fig_overhead()
    print_overhead_table(timings)

    print("\n[5/5] Per-phase accuracy chart...")
    fig_phase_accuracy(steps)

    print("\n" + "=" * 60)
    print("Done. Figures in: " + OUT_DIR)
    print("=" * 60)
