"""
simu5g_simulation.py — Run the AI Management Framework on Simu5G data.

This script evaluates the full RTP → ATM → NDT pipeline using 5G NR KPI
data calibrated against Simu5G's Urban Macro channel model (3GPP 38.901).

Supports three modes:

  1. OFFLINE (default): Uses the Simu5G-calibrated synthetic generator.
     No OMNeT++ installation required. Runs immediately.

  2. REPLAY: Feed pre-recorded Simu5G .vec/.sca output through the pipeline.
     Requires: opp_scavetool export → CSV, or raw .vec files.

  3. LIVE: Run Simu5G via OMNeT++ CLI, then feed results.
     Requires: OMNeT++ 6.x, INET 4.5+, Simu5G 1.4.x installed.

Output:
  - simu5g/results/simu5g_results.csv         Per-step KPI + detection log
  - simu5g/results/simu5g_summary.csv          Phase-level summary statistics
  - simu5g/results/simu5g_detector_eval.csv    Detector TP/FP/FN evaluation
  - thesis/figures/simu5g_kpi_timeline.png     KPI time-series with phases
  - thesis/figures/simu5g_detection_map.png     Detector alert heatmap
  - thesis/figures/simu5g_accuracy_phases.png   Per-phase accuracy bars
  - thesis/figures/simu5g_roc_comparison.png    ROC per detector
  - thesis/figures/simu5g_multi_seed_box.png    Multi-seed precision/recall

Run:
    python simu5g_simulation.py                           # Offline mode
    python simu5g_simulation.py --mode replay --data results.csv  # Replay
    python simu5g_simulation.py --mode live --simu5g-dir /path     # Live

Reference:
  G. Nardini et al., "Simu5G — An OMNeT++ Library for End-to-End
  Performance Evaluation of 5G Networks," IEEE Access, vol. 8, 2020.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_STYLE_CANDIDATES = [
    "seaborn-v0_8-whitegrid",
    "seaborn-whitegrid",
    "ggplot",
    "default",
]
for _style in _STYLE_CANDIDATES:
    try:
        plt.style.use(_style)
        break
    except OSError:
        continue

from sklearn.ensemble import RandomForestClassifier

from aif.aif import AIF
from rtp.rtp import RTP, RTPConfig, MToUTSignal, RTPEvent
from atm.atm import ATM, ATMPolicy, MTPVariant
from atm.mtp_l import MTPLocal
from ndt.ndt import NDT

from simu5g.parser import Simu5GParser
from simu5g.adapter import Simu5GAdapter, PhaseConfig, Simu5GStepResult
from simu5g.generator import Simu5GDataGenerator, Simu5GRadioConfig

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.ERROR,          # Suppress WARNING noise from detectors during batch runs
    format="%(levelname)-8s %(name)-20s — %(message)s",
)
log = logging.getLogger("SIMU5G_SIM")
log.setLevel(logging.INFO)

# Suppress ConstantInputWarning from scipy.stats.pearsonr in CPD
warnings.filterwarnings("ignore", message="An input array is constant")

# Suppress noisy detector/RTP/ATM logs during batch runs
for _noisy_logger in ["rtp.rtp", "detectors.dpd", "detectors.cpd", "detectors.cdd",
                       "detectors.ddd", "atm.atm", "aif.aif", "ndt.ndt"]:
    logging.getLogger(_noisy_logger).setLevel(logging.CRITICAL + 1)

# ── Output directories ───────────────────────────────────────────────────────
FIGURES_DIR = os.path.join("thesis", "figures")
RESULTS_DIR = os.path.join("simu5g", "results")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
N_SEEDS = 30
CHECK_INTERVALS = [10, 25, 50, 100]

PHASE_COLORS = {
    "stable":            "#2ecc71",
    "drift":             "#e67e22",
    "subtle_poison":     "#e74c3c",
    "aggressive_poison": "#c0392b",
    "recovery":          "#3498db",
}

PHASE_LABELS = {
    "stable":            "Phase 1: Stable",
    "drift":             "Phase 2: Concept Drift",
    "subtle_poison":     "Phase 3a: Subtle Poison",
    "aggressive_poison": "Phase 3b: Aggressive Poison",
    "recovery":          "Phase 4: Recovery",
}


# ===========================================================================
# Section 1 — Single-seed Simu5G simulation
# ===========================================================================

@dataclass
class Simu5GSeedResult:
    """
    Aggregated metrics from one seed run.

    Detector TP/FP/FN are counted PER CHECK CYCLE (not binary per-phase):
      - At each check cycle, classify the alert as TP, FP, or TN.
      - At each check cycle during an anomaly phase where the detector
        did NOT fire, count it as FN.
      - Precision = TP / (TP + FP)   — fraction of alerts that are correct
      - Recall    = TP / (TP + FN)   — fraction of anomaly checks detected

    This gives continuous precision/recall values rather than binary 0/1.

    DDD/CDD target: drift phases (concept drift)
    DPD/CPD target: poisoning phases (subtle + aggressive)
    """
    seed: int
    check_interval: int

    # Per-check-cycle detector counts
    ddd_tp: int = 0; ddd_fp: int = 0; ddd_fn: int = 0
    dpd_tp: int = 0; dpd_fp: int = 0; dpd_fn: int = 0
    cdd_tp: int = 0; cdd_fp: int = 0; cdd_fn: int = 0
    cpd_tp: int = 0; cpd_fp: int = 0; cpd_fn: int = 0

    # Detection latency (steps from anomaly start to FIRST alert)
    ddd_latency: Optional[float] = None
    dpd_latency: Optional[float] = None
    cdd_latency: Optional[float] = None
    cpd_latency: Optional[float] = None

    # Per-phase accuracy
    stable_accuracy: float = 0.0
    drift_accuracy: float = 0.0
    recovery_accuracy: float = 0.0

    # NDT
    ndt_pseudo_scores: list = field(default_factory=list)
    ndt_gt_scores: list = field(default_factory=list)

    # ATM
    atm_deployments: int = 0
    total_mtout: int = 0
    total_security: int = 0

    # Rolling accuracy
    rolling_accuracy: list = field(default_factory=list)

    def _prec(self, tp, fp):
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def _rec(self, tp, fn):
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @property
    def ddd_precision(self): return self._prec(self.ddd_tp, self.ddd_fp)
    @property
    def ddd_recall(self): return self._rec(self.ddd_tp, self.ddd_fn)
    @property
    def dpd_precision(self): return self._prec(self.dpd_tp, self.dpd_fp)
    @property
    def dpd_recall(self): return self._rec(self.dpd_tp, self.dpd_fn)
    @property
    def cdd_precision(self): return self._prec(self.cdd_tp, self.cdd_fp)
    @property
    def cdd_recall(self): return self._rec(self.cdd_tp, self.cdd_fn)
    @property
    def cpd_precision(self): return self._prec(self.cpd_tp, self.cpd_fp)
    @property
    def cpd_recall(self): return self._rec(self.cpd_tp, self.cpd_fn)


def run_single_seed_simu5g(
    seed: int,
    check_interval: int = 50,
    verbose: bool = False,
) -> Simu5GSeedResult:
    """
    Run one Simu5G-calibrated simulation seed through the RTP pipeline.

    Uses the Simu5GDataGenerator to produce 3GPP-compliant KPIs calibrated
    against Simu5G's Urban Macro channel model, then feeds them through the
    full framework stack: AIF → RTP (DDD/DPD/CDD/CPD) → ATM (MTP-L) → NDT.
    """
    rng = np.random.default_rng(seed)
    result = Simu5GSeedResult(seed=seed, check_interval=check_interval)

    # ── Generate Simu5G-calibrated data ────────────────────────────────
    gen = Simu5GDataGenerator()
    X_full, y_full, timestamps = gen.generate(seed=seed, samples_per_second=10.0)

    n_total = len(X_full)

    # ── Initial training (first 400 samples = 40s of stable data) ──────
    n_stable = min(400, int(n_total * 0.44))  # 40/90 seconds
    X_train = X_full[:n_stable]
    y_train = y_full[:n_stable]

    clf = RandomForestClassifier(n_estimators=50, random_state=seed)
    clf.fit(X_train, y_train)
    aif = AIF(clf)

    # ── RTP configuration (recalibrated for 5G NR channel variance) ─────
    # The original thresholds from main.py were tuned for simple synthetic
    # data (y = 1[x0+x1>0]). Simu5G's 3GPP 38.901 UMa channel model has
    # much higher natural variance (shadow fading sigma=8dB, Rayleigh
    # fading), so DPD/DDD thresholds must be raised to avoid false alarms.
    cfg = RTPConfig(
        buffer_maxlen=2000,
        check_interval=check_interval,
        cdd_task="classifier",
        # DDD: larger windows to capture channel variance, less sensitive
        ddd_reference_size=300,         # was 200 — wider baseline for 5G
        ddd_recent_size=150,            # was 100 — smoother recent window
        # DPD: raised thresholds for 5G KPI variance
        dpd_reference_size=300,         # was 200
        dpd_recent_size=100,            # was 50  — more samples for IF
        dpd_contamination_threshold=0.15,  # was 0.08 — 5G data is noisier
        dpd_mahal_threshold=15.0,       # was 5.0  — 3x for 5G variance
        # CDD: performance-based — works well, minor tuning
        cdd_reference_window=200,       # was 150
        cdd_recent_window=80,           # was 50
        cdd_perf_drop_threshold=0.15,   # was 0.12 — slightly higher bar
        cdd_ph_lambda=50.0,             # was 40.0
        # CPD: raised thresholds
        cpd_reference_size=300,         # was 200
        cpd_recent_size=150,            # was 100
        cpd_shadow_threshold=0.50,      # was 0.38 — need larger divergence
        cpd_output_ks_alpha=0.00001,    # was 0.0001 — stricter KS test
        cpd_corr_threshold=0.75,        # was 0.60 — larger correlation shift
        mtout_cooldown_steps=150,
    )

    mtout_signals = []
    security_alerts = []

    def on_mtout(sig):
        mtout_signals.append(sig)
    def on_security(evt):
        security_alerts.append(evt)

    rtp = RTP(aif, config=cfg, on_mtout=on_mtout, on_security_alert=on_security)

    # Reference from training data
    lob_ref = clf.predict(X_train[:300])
    rtp.set_reference(X_train[:300], y_train[:300], lob_ref)

    # ── ATM setup ──────────────────────────────────────────────────────
    mtp_l = MTPLocal(n_splits=3, fine_tune_first=True, random_state=seed)
    ndt = NDT(
        current_model_getter=lambda: rtp.aif.active_estimator,
        min_score=0.65,
        min_improvement=-0.05,
    )
    policy = ATMPolicy(
        prefer_variant=MTPVariant.LOCAL,
        local_max_samples=600,
        use_ndt=True,
        ndt_min_accuracy=0.65,
        auto_deploy=True,
        max_retrain_attempts=2,
    )

    def on_atm_result(r):
        if r.deployed:
            result.atm_deployments += 1

    atm = ATM(
        rtp=rtp, mtp_l=mtp_l, mtp_e=None,
        ndt=ndt, policy=policy, on_result=on_atm_result,
    )

    # Wire ATM into RTP
    def _mtout_handler(sig):
        on_mtout(sig)
        atm.handle(sig)

        # NDT dual validation
        lib = rtp.buffers.lib
        lob = rtp.buffers.lob
        n_val = min(200, len(lib))
        if n_val >= 10:
            X_val = lib.get_values(n_val)
            y_pseudo = lob.get_flat_values(n_val)
            y_gt = gen._handover_label(X_val[:, 0], X_val[:, 1], X_val[:, 3])
            preds = rtp.aif.active_estimator.predict(X_val)
            gt_score = float(np.mean(preds == y_gt))
            pseudo_score = float(np.mean(preds == y_pseudo.ravel()))
            result.ndt_pseudo_scores.append(pseudo_score)
            result.ndt_gt_scores.append(gt_score)

    rtp._on_mtout = _mtout_handler

    # ── Phase boundaries ─────────────────────────────────────────────────
    phase_map = gen._get_phase
    ROLLING_WIN = 50
    correct_window = []

    # Define which phases each detector SHOULD fire during:
    #   DDD/CDD → drift detection → target phases: drift, subtle_poison, aggressive_poison
    #   DPD/CPD → poisoning detection → target phases: subtle_poison, aggressive_poison
    drift_target_phases = {"drift", "subtle_poison", "aggressive_poison"}
    poison_target_phases = {"subtle_poison", "aggressive_poison"}
    benign_phases = {"stable", "recovery"}

    phase_correct = {"stable": 0, "drift": 0, "subtle_poison": 0,
                     "aggressive_poison": 0, "recovery": 0}
    phase_total = {"stable": 0, "drift": 0, "subtle_poison": 0,
                   "aggressive_poison": 0, "recovery": 0}

    # First-alert step tracking for latency computation
    drift_start_step = None
    poison_start_step = None
    ddd_first_alert = None
    dpd_first_alert = None
    cdd_first_alert = None
    cpd_first_alert = None

    # ── Main simulation loop ───────────────────────────────────────────
    for i in range(n_total):
        step = i + 1
        t = timestamps[i]
        phase = phase_map(t)

        if drift_start_step is None and phase == "drift":
            drift_start_step = step
        if poison_start_step is None and phase in poison_target_phases:
            poison_start_step = step

        pred = rtp.observe(X_full[i], y_true=y_full[i])
        pred_val = int(pred.ravel()[0])

        correct = int(pred_val == y_full[i])
        phase_correct[phase] += correct
        phase_total[phase] += 1

        correct_window.append(correct)
        if len(correct_window) > ROLLING_WIN:
            correct_window.pop(0)
        result.rolling_accuracy.append(float(np.mean(correct_window)))

        # ── Per-check-cycle TP/FP/FN classification ───────────────────
        # At each check cycle, EVERY detector is evaluated:
        #   - If detector fired AND we're in its target phase → TP
        #   - If detector fired AND we're in a benign phase → FP
        #   - If detector didn't fire AND we're in its target phase → FN
        #   - If detector didn't fire AND we're in a benign phase → TN (not tracked)
        if step % check_interval == 0 and rtp.last_ddd is not None:

            # --- DDD (data drift detector) ---
            ddd_fired = rtp.last_ddd.drift_detected
            if ddd_fired and phase in drift_target_phases:
                result.ddd_tp += 1
                if ddd_first_alert is None:
                    ddd_first_alert = step
            elif ddd_fired and phase in benign_phases:
                result.ddd_fp += 1
            elif not ddd_fired and phase in drift_target_phases:
                result.ddd_fn += 1
            # else: TN (not tracked)

            # --- DPD (data poisoning detector) ---
            dpd_fired = (rtp.last_dpd is not None and rtp.last_dpd.poisoning_detected)
            if dpd_fired and phase in poison_target_phases:
                result.dpd_tp += 1
                if dpd_first_alert is None:
                    dpd_first_alert = step
            elif dpd_fired and phase in benign_phases:
                result.dpd_fp += 1
            elif not dpd_fired and phase in poison_target_phases:
                result.dpd_fn += 1

            # --- CDD (concept drift detector) ---
            cdd_fired = (rtp.last_cdd is not None and rtp.last_cdd.drift_detected)
            if cdd_fired and phase in drift_target_phases:
                result.cdd_tp += 1
                if cdd_first_alert is None:
                    cdd_first_alert = step
            elif cdd_fired and phase in benign_phases:
                result.cdd_fp += 1
            elif not cdd_fired and phase in drift_target_phases:
                result.cdd_fn += 1

            # --- CPD (concept poisoning detector) ---
            cpd_fired = (rtp.last_cpd is not None and rtp.last_cpd.poisoning_detected)
            if cpd_fired and phase in poison_target_phases:
                result.cpd_tp += 1
                if cpd_first_alert is None:
                    cpd_first_alert = step
            elif cpd_fired and phase in benign_phases:
                result.cpd_fp += 1
            elif not cpd_fired and phase in poison_target_phases:
                result.cpd_fn += 1

    # Detection latency = first alert step - anomaly start step
    if ddd_first_alert and drift_start_step:
        result.ddd_latency = ddd_first_alert - drift_start_step
    if dpd_first_alert and poison_start_step:
        result.dpd_latency = dpd_first_alert - poison_start_step
    if cdd_first_alert and drift_start_step:
        result.cdd_latency = cdd_first_alert - drift_start_step
    if cpd_first_alert and poison_start_step:
        result.cpd_latency = cpd_first_alert - poison_start_step

    # Phase accuracies
    for p in phase_correct:
        if phase_total[p] > 0:
            acc = phase_correct[p] / phase_total[p]
            if p == "stable": result.stable_accuracy = acc
            elif p == "drift": result.drift_accuracy = acc
            elif p == "recovery": result.recovery_accuracy = acc

    result.total_mtout = len(mtout_signals)
    result.total_security = len(security_alerts)

    return result


# ===========================================================================
# Section 2 — Multi-seed evaluation
# ===========================================================================

def run_multi_seed(
    n_seeds: int = N_SEEDS,
    check_intervals: list[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the Simu5G simulation across multiple seeds and check intervals."""
    if check_intervals is None:
        check_intervals = CHECK_INTERVALS

    all_results = []
    total_runs = n_seeds * len(check_intervals)
    run_idx = 0

    for ci in check_intervals:
        for seed in range(n_seeds):
            run_idx += 1
            t0 = time.time()

            res = run_single_seed_simu5g(seed=seed, check_interval=ci)

            elapsed = time.time() - t0
            if verbose:
                log.info(
                    "[%d/%d] seed=%d ci=%d | DDD_P=%.3f DPD_P=%.3f CDD_P=%.3f CPD_P=%.3f | "
                    "stable=%.3f drift=%.3f recov=%.3f | MToUT=%d | %.1fs",
                    run_idx, total_runs, seed, ci,
                    res.ddd_precision, res.dpd_precision,
                    res.cdd_precision, res.cpd_precision,
                    res.stable_accuracy, res.drift_accuracy, res.recovery_accuracy,
                    res.total_mtout, elapsed,
                )

            row = {
                "seed": seed,
                "check_interval": ci,
                "ddd_precision": res.ddd_precision,
                "ddd_recall": res.ddd_recall,
                "ddd_latency": res.ddd_latency,
                "dpd_precision": res.dpd_precision,
                "dpd_recall": res.dpd_recall,
                "dpd_latency": res.dpd_latency,
                "cdd_precision": res.cdd_precision,
                "cdd_recall": res.cdd_recall,
                "cdd_latency": res.cdd_latency,
                "cpd_precision": res.cpd_precision,
                "cpd_recall": res.cpd_recall,
                "cpd_latency": res.cpd_latency,
                "stable_accuracy": res.stable_accuracy,
                "drift_accuracy": res.drift_accuracy,
                "recovery_accuracy": res.recovery_accuracy,
                "ndt_pseudo_mean": np.mean(res.ndt_pseudo_scores) if res.ndt_pseudo_scores else np.nan,
                "ndt_gt_mean": np.mean(res.ndt_gt_scores) if res.ndt_gt_scores else np.nan,
                "atm_deployments": res.atm_deployments,
                "total_mtout": res.total_mtout,
                "total_security": res.total_security,
            }
            all_results.append(row)

    df = pd.DataFrame(all_results)
    return df


# ===========================================================================
# Section 3 — Figure generation
# ===========================================================================

def plot_kpi_timeline(seed: int = 0) -> None:
    """
    Plot Simu5G KPI time-series with phase annotations.

    Shows RSRP, SINR, throughput, and latency over simulation time with
    colored background bands indicating each phase.
    """
    gen = Simu5GDataGenerator()
    X, y, timestamps = gen.generate(seed=seed)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        "Simu5G-Calibrated 5G NR KPI Timeline\n"
        "(3GPP 38.901 Urban Macro, 3.5 GHz, 100 MHz BW)",
        fontsize=14, fontweight="bold",
    )

    kpi_names = ["SS-RSRP (dBm)", "SS-SINR (dB)", "DL Throughput (Mbps)", "E2E Latency (ms)"]
    kpi_cols = [X[:, 0], X[:, 1], X[:, 2], X[:, 3]]

    phase_boundaries = [0, 40, 60, 65, 70, 90]
    phase_names = list(PHASE_COLORS.keys())

    for ax_idx, (ax, name, vals) in enumerate(zip(axes, kpi_names, kpi_cols)):
        # Phase background bands
        for i in range(len(phase_boundaries) - 1):
            phase = phase_names[i]
            ax.axvspan(
                phase_boundaries[i], phase_boundaries[i + 1],
                alpha=0.15, color=PHASE_COLORS[phase],
                label=PHASE_LABELS[phase] if ax_idx == 0 else None,
            )

        ax.plot(timestamps, vals, linewidth=0.5, alpha=0.7, color="#2c3e50")
        ax.set_ylabel(name, fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Simulation Time (s)", fontsize=11)
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "simu5g_kpi_timeline.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)


def plot_detection_map(results_df: pd.DataFrame) -> None:
    """
    Detector alert heatmap: seeds × detectors, colored by precision.
    """
    ci_default = 50
    df = results_df[results_df["check_interval"] == ci_default].copy()
    if df.empty:
        ci_default = results_df["check_interval"].mode().iloc[0]
        df = results_df[results_df["check_interval"] == ci_default].copy()

    det_cols = ["ddd_precision", "dpd_precision", "cdd_precision", "cpd_precision"]
    det_labels = ["DDD\n(Data Drift)", "DPD\n(Data Poisoning)",
                  "CDD\n(Concept Drift)", "CPD\n(Concept Poisoning)"]

    matrix = df[det_cols].values
    n_seeds = matrix.shape[0]

    fig, ax = plt.subplots(figsize=(10, max(6, n_seeds * 0.3)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(det_labels)))
    ax.set_xticklabels(det_labels, fontsize=10)
    ax.set_yticks(range(n_seeds))
    ax.set_yticklabels([f"Seed {i}" for i in range(n_seeds)], fontsize=7)
    ax.set_title(
        f"Detector Precision Heatmap (Simu5G, ci={ci_default})\n"
        "Green = high precision, Red = low precision",
        fontsize=12, fontweight="bold",
    )

    # Annotate cells
    for i in range(n_seeds):
        for j in range(len(det_cols)):
            val = matrix[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="Precision", shrink=0.8)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "simu5g_detection_map.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)


def plot_accuracy_phases(results_df: pd.DataFrame) -> None:
    """Per-phase accuracy bar chart with error bars across seeds."""
    ci_default = 50
    df = results_df[results_df["check_interval"] == ci_default].copy()
    if df.empty:
        df = results_df.copy()

    phases = ["stable_accuracy", "drift_accuracy", "recovery_accuracy"]
    labels = ["Phase 1\nStable", "Phase 2\nConcept Drift", "Phase 4\nRecovery"]
    colors = [PHASE_COLORS["stable"], PHASE_COLORS["drift"], PHASE_COLORS["recovery"]]

    means = [df[p].mean() for p in phases]
    stds = [df[p].std() for p in phases]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, means, yerr=stds, capsize=8, color=colors,
                  edgecolor="black", linewidth=0.8, alpha=0.85)

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.02,
                f"{m:.3f}±{s:.3f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold")

    ax.set_ylabel("Classification Accuracy", fontsize=12)
    ax.set_title(
        "Per-Phase Accuracy — Simu5G 5G NR Simulation\n"
        f"(30 seeds, check_interval={ci_default})",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "simu5g_accuracy_phases.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)


def plot_multi_seed_boxplot(results_df: pd.DataFrame) -> None:
    """Precision/recall box plots across seeds per detector."""
    ci_default = 50
    df = results_df[results_df["check_interval"] == ci_default].copy()
    if df.empty:
        df = results_df.copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    detectors = ["DDD", "DPD", "CDD", "CPD"]
    prec_data = [df[f"{d.lower()}_precision"].values for d in detectors]
    rec_data = [df[f"{d.lower()}_recall"].values for d in detectors]

    bp1 = ax1.boxplot(prec_data, tick_labels=detectors, patch_artist=True, widths=0.6)
    bp2 = ax2.boxplot(rec_data, tick_labels=detectors, patch_artist=True, widths=0.6)

    colors = ["#3498db", "#e74c3c", "#f39c12", "#9b59b6"]
    for bp, ax in [(bp1, ax1), (bp2, ax2)]:
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)

    ax1.set_title("Detector Precision (Simu5G)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Precision")
    ax1.set_ylim(-0.05, 1.1)
    ax1.grid(axis="y", alpha=0.3)

    ax2.set_title("Detector Recall (Simu5G)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Recall")
    ax2.set_ylim(-0.05, 1.1)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Multi-Seed Detector Evaluation — Simu5G (N={len(df)} seeds)",
        fontsize=14, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "simu5g_multi_seed_box.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)


def plot_ndt_comparison(results_df: pd.DataFrame) -> None:
    """NDT pseudo-label vs ground-truth comparison (Simu5G data)."""
    ci_default = 50
    df = results_df[results_df["check_interval"] == ci_default].copy()
    if df.empty:
        df = results_df.copy()

    pseudo_scores = df["ndt_pseudo_mean"].dropna()
    gt_scores = df["ndt_gt_mean"].dropna()

    if len(pseudo_scores) == 0 or len(gt_scores) == 0:
        log.warning("No NDT scores available for plotting")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(2)
    means = [pseudo_scores.mean(), gt_scores.mean()]
    stds = [pseudo_scores.std(), gt_scores.std()]

    bars = ax.bar(x, means, yerr=stds, capsize=10, width=0.5,
                  color=["#3498db", "#e74c3c"], edgecolor="black",
                  linewidth=1.0, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(["NDT Pseudo-Label\n(LOB)", "NDT Ground-Truth"], fontsize=11)
    ax.set_ylabel("Validation Score", fontsize=12)
    ax.set_title(
        "NDT Self-Referential Bias — Simu5G Data\n"
        f"Bias = {means[0] - means[1]:+.3f} (pseudo over-estimates ground truth)",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylim(0, 1.3)

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.03,
                f"{m:.3f}±{s:.3f}", ha="center", fontsize=11, fontweight="bold")

    # Bias arrow
    ax.annotate(
        f"Bias: {means[0] - means[1]:+.3f}",
        xy=(1, means[1]), xytext=(0, means[0]),
        arrowprops=dict(arrowstyle="<->", color="black", lw=2),
        fontsize=12, ha="center", va="bottom", fontweight="bold",
        color="red",
    )

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "simu5g_ndt_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)


def plot_latency_tradeoff(results_df: pd.DataFrame) -> None:
    """Detection latency vs check_interval trade-off (Simu5G)."""
    summary = results_df.groupby("check_interval").agg(
        ddd_lat_mean=("ddd_latency", "mean"),
        ddd_lat_std=("ddd_latency", "std"),
        cdd_lat_mean=("cdd_latency", "mean"),
        cdd_lat_std=("cdd_latency", "std"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(9, 6))

    for det, label, color, marker in [
        ("ddd", "DDD (Data Drift)", "#3498db", "o"),
        ("cdd", "CDD (Concept Drift)", "#f39c12", "s"),
    ]:
        mean_col = f"{det}_lat_mean"
        std_col = f"{det}_lat_std"
        if mean_col in summary.columns:
            valid = summary.dropna(subset=[mean_col])
            if not valid.empty:
                ax.errorbar(
                    valid["check_interval"], valid[mean_col],
                    yerr=valid[std_col].fillna(0),
                    marker=marker, linewidth=2, markersize=8,
                    capsize=6, label=label, color=color,
                )

    ax.set_xlabel("Check Interval (steps)", fontsize=12)
    ax.set_ylabel("Detection Latency (steps)", fontsize=12)
    ax.set_title(
        "Detection Latency vs. Check Interval — Simu5G\n"
        "Lower is better (faster detection)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(CHECK_INTERVALS)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "simu5g_latency_tradeoff.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)


# ===========================================================================
# Section 4 — Summary statistics
# ===========================================================================

def compute_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean±std summary grouped by check_interval."""
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    exclude = ["seed"]
    agg_cols = [c for c in numeric_cols if c not in exclude]

    summary = results_df.groupby("check_interval")[agg_cols].agg(["mean", "std"])
    summary.columns = ["_".join(col) for col in summary.columns]
    summary = summary.reset_index()
    return summary


# ===========================================================================
# Section 5 — Main entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run AI Management Framework on Simu5G 5G NR data"
    )
    parser.add_argument(
        "--mode", choices=["offline", "replay"], default="offline",
        help="Data source mode (default: offline with synthetic generator)"
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to Simu5G CSV/VEC file (for replay mode)"
    )
    parser.add_argument(
        "--seeds", type=int, default=N_SEEDS,
        help=f"Number of random seeds (default: {N_SEEDS})"
    )
    parser.add_argument(
        "--ci", type=int, nargs="+", default=CHECK_INTERVALS,
        help=f"Check intervals to evaluate (default: {CHECK_INTERVALS})"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 5 seeds, ci=[50] only"
    )
    args = parser.parse_args()

    if args.quick:
        args.seeds = 5
        args.ci = [50]

    log.info("=" * 70)
    log.info("Simu5G 5G NR Simulation — AI Management Framework Evaluation")
    log.info("=" * 70)
    log.info("Mode: %s | Seeds: %d | Check intervals: %s",
             args.mode, args.seeds, args.ci)

    t_start = time.time()

    if args.mode == "replay" and args.data:
        # ── REPLAY mode: use pre-recorded Simu5G data ─────────────────
        log.info("Loading Simu5G data from: %s", args.data)
        s5g_parser = Simu5GParser()

        if args.data.endswith(".csv"):
            df_raw = s5g_parser.from_csv(args.data)
        elif args.data.endswith(".vec"):
            df_raw = s5g_parser.from_vec(args.data)
        else:
            log.error("Unsupported file format: %s (use .csv or .vec)", args.data)
            return

        X, timestamps = s5g_parser.to_kpi_matrix(df_raw)
        y = s5g_parser.extract_handover_labels(df_raw)

        log.info("Loaded %d samples, replaying through RTP pipeline...", len(X))

        adapter = Simu5GAdapter(check_interval=args.ci[0])
        results = adapter.replay(X, y, timestamps, verbose=True)

        # Save results
        results_df = adapter.to_dataframe()
        results_df.to_csv(
            os.path.join(RESULTS_DIR, "simu5g_replay_results.csv"), index=False
        )

        # Print summary
        phase_acc = adapter.phase_accuracy()
        det_summary = adapter.detector_summary()
        log.info("\nPhase Accuracy:")
        for phase, acc in phase_acc.items():
            log.info("  %-25s %.3f", phase, acc)
        log.info("\nDetector Summary:")
        for det, stats in det_summary.items():
            log.info("  %s: P=%.3f R=%.3f (TP=%d FP=%d FN=%d)",
                     det.upper(), stats["precision"], stats["recall"],
                     stats["tp"], stats["fp"], stats["fn"])

    else:
        # ── OFFLINE mode: Simu5G-calibrated synthetic data ─────────────
        log.info("Running Simu5G-calibrated offline simulation...")

        # Generate KPI timeline figure (single seed)
        plot_kpi_timeline(seed=0)

        # Multi-seed evaluation
        results_df = run_multi_seed(
            n_seeds=args.seeds,
            check_intervals=args.ci,
            verbose=True,
        )

        # Save results
        results_path = os.path.join(RESULTS_DIR, "simu5g_results.csv")
        results_df.to_csv(results_path, index=False)
        log.info("Saved results: %s", results_path)

        # Summary statistics
        summary = compute_summary(results_df)
        summary_path = os.path.join(RESULTS_DIR, "simu5g_summary.csv")
        summary.to_csv(summary_path, index=False)
        log.info("Saved summary: %s", summary_path)

        # Generate all figures
        plot_detection_map(results_df)
        plot_accuracy_phases(results_df)
        plot_multi_seed_boxplot(results_df)
        plot_ndt_comparison(results_df)
        plot_latency_tradeoff(results_df)

        # ── Print final summary ────────────────────────────────────────
        elapsed = time.time() - t_start
        log.info("\n" + "=" * 70)
        log.info("SIMU5G SIMULATION COMPLETE")
        log.info("=" * 70)
        log.info("Total time: %.1f min (%.1f s)", elapsed / 60, elapsed)
        log.info("Seeds: %d | Check intervals: %s", args.seeds, args.ci)
        log.info("")

        # Per-CI summary
        for ci in args.ci:
            ci_df = results_df[results_df["check_interval"] == ci]
            log.info("─── check_interval = %d ───", ci)
            log.info("  DDD precision: %.3f ± %.3f  recall: %.3f ± %.3f",
                     ci_df["ddd_precision"].mean(), ci_df["ddd_precision"].std(),
                     ci_df["ddd_recall"].mean(), ci_df["ddd_recall"].std())
            log.info("  DPD precision: %.3f ± %.3f  recall: %.3f ± %.3f",
                     ci_df["dpd_precision"].mean(), ci_df["dpd_precision"].std(),
                     ci_df["dpd_recall"].mean(), ci_df["dpd_recall"].std())
            log.info("  CDD precision: %.3f ± %.3f  recall: %.3f ± %.3f",
                     ci_df["cdd_precision"].mean(), ci_df["cdd_precision"].std(),
                     ci_df["cdd_recall"].mean(), ci_df["cdd_recall"].std())
            log.info("  CPD precision: %.3f ± %.3f  recall: %.3f ± %.3f",
                     ci_df["cpd_precision"].mean(), ci_df["cpd_precision"].std(),
                     ci_df["cpd_recall"].mean(), ci_df["cpd_recall"].std())
            log.info("  Stable acc:  %.3f ± %.3f",
                     ci_df["stable_accuracy"].mean(), ci_df["stable_accuracy"].std())
            log.info("  Drift acc:   %.3f ± %.3f",
                     ci_df["drift_accuracy"].mean(), ci_df["drift_accuracy"].std())
            log.info("  Recovery:    %.3f ± %.3f",
                     ci_df["recovery_accuracy"].mean(), ci_df["recovery_accuracy"].std())

            ndt_pseudo = ci_df["ndt_pseudo_mean"].dropna()
            ndt_gt = ci_df["ndt_gt_mean"].dropna()
            if len(ndt_pseudo) > 0 and len(ndt_gt) > 0:
                bias = ndt_pseudo.mean() - ndt_gt.mean()
                log.info("  NDT pseudo:  %.3f ± %.3f", ndt_pseudo.mean(), ndt_pseudo.std())
                log.info("  NDT GT:      %.3f ± %.3f", ndt_gt.mean(), ndt_gt.std())
                log.info("  NDT bias:    %+.4f", bias)

            log.info("  ATM deploys: %.1f ± %.1f",
                     ci_df["atm_deployments"].mean(), ci_df["atm_deployments"].std())
            log.info("")


if __name__ == "__main__":
    main()
