"""
generate_figures.py — Thesis figure generator for AI Management Framework

Re-runs the full 4-phase simulation and generates 5 publication-quality
PDF figures for the thesis chapter.

Usage:
    python generate_figures.py
"""

import os
import sys
import warnings
import logging

# ── Silence everything before importing project modules ───────────────────────
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")
os.environ["MLFLOW_DISABLE_ENV_MANAGER_CONDAENV_CREATION"] = "true"
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Standard imports ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from pathlib import Path
from collections import defaultdict

# ── Project imports ───────────────────────────────────────────────────────────
from aif.aif import AIF
from rtp.rtp import RTP, RTPConfig, MToUTSignal, RTPEvent, EventType
from atm.atm import ATM, ATMPolicy, MTPVariant
from atm.mtp_l import MTPLocal
from ndt.ndt import NDT

try:
    from atm.mtp_e import MTPExternal
    import mlflow
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier

# ── Matplotlib academic style ─────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

# ── Phase background colours ──────────────────────────────────────────────────
PHASE_COLORS = {
    1: "#d4e6f1",
    2: "#fdebd0",
    3: "#fadbd8",
    4: "#d5f5e3",
}
PHASE_LABELS = {
    1: "Phase 1: Stable",
    2: "Phase 2: Concept Drift",
    3: "Phase 3: Poisoning",
    4: "Phase 4: Recovery",
}
PHASE_RANGES = {
    1: (1, 400),
    2: (401, 600),
    3: (601, 700),
    4: (701, 900),
}

# ── Output directory ──────────────────────────────────────────────────────────
FIGURES_DIR = Path(__file__).parent / "thesis" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Simulation
# =============================================================================

def synthesise_thesis_workload(
    seed: int = 0,
    drift_magnitude: float = 1.0,
    poison_rate: float = 0.13,
    base_noise: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Methodology-faithful Campaign A workload generator.

    Produces a single 900-step ``(X, y, phase)`` trace at the
    thesis-canonical phase boundaries (P1=1-400, P2=401-600,
    P3=601-700, P4=701-900) using the SAME mechanics that drive the
    multi-seed Campaigns B/C/D in ``thesis/regenerate_evaluation.py``:

      * Phase 2 — gradual covariate shift on feature 0 (linear ramp
        ``0 -> -drift_magnitude``) plus a correlated half-amplitude
        ramp on feature 1 and a variance drift on feature 2.  The
        ``drift_magnitude`` units are dimensionless standardised-feature
        units; ``regenerate_evaluation.py`` samples them from
        Uniform(0.6, 1.4), and Campaign A pins the midpoint 1.0.  The
        decision rule ``y = 1{X[:,0]+X[:,1] > 0}`` is computed on the
        *original* features before the ramp is added, so the served
        classifier loses accuracy organically as the input distribution
        shifts away from the trained reference.
      * Phase 3 — asymmetric label flips (positive -> negative only)
        at double the per-step poisoning rate, applied only to the
        positive-class subset.  This is the realistic threat model the
        DPD's chi-squared label-distribution test is designed to catch.
      * Background label noise of ``base_noise`` is applied uniformly
        across all 900 steps to model nominal labelling error.

    Returns
    -------
    X : ``(900, 4)`` array
        Feature matrix; the four columns play the role of (RSRP, SINR,
        Throughput, Latency) KPIs in the methodology narrative.
    y : ``(900,)`` int array
        Binary handover-success labels with the asymmetric P3 corruption
        and background noise applied.
    phase : ``(900,)`` int array
        Phase index in {1, 2, 3, 4} per step (1-indexed phases on the
        0-indexed array).
    """
    rng = np.random.default_rng(seed)

    # Thesis-canonical phase boundaries (1-indexed steps; convert to
    # 0-indexed slice ranges).
    phase_ranges_0idx = {
        1: (0,   400),   # steps 1-400
        2: (400, 600),   # steps 401-600
        3: (600, 700),   # steps 601-700
        4: (700, 900),   # steps 701-900
    }

    X = rng.normal(0, 1, size=(900, 4)).astype(float)

    # True concept BEFORE any perturbation.
    y = ((X[:, 0] + X[:, 1]) > 0).astype(int)

    phase = np.zeros(900, dtype=int)
    for ph, (a, b) in phase_ranges_0idx.items():
        phase[a:b] = ph

    # Phase 2: gradual covariate shift.  Linear ramp on feature 0,
    # correlated half-ramp on feature 1, variance drift on feature 2.
    a, b = phase_ranges_0idx[2]
    ramp = np.linspace(0.0, -drift_magnitude, b - a, dtype=float)
    X[a:b, 0] += ramp
    X[a:b, 1] += 0.5 * ramp
    X[a:b, 2] *= 1.0 + 0.5 * np.linspace(0, 1, b - a)

    # Phase 3: asymmetric positive -> negative label flips at double
    # rate on the positive-class subset.
    a3, b3 = phase_ranges_0idx[3]
    n3 = b3 - a3
    pos_mask = (y[a3:b3] == 1)
    flip_pos = rng.random(n3) < (poison_rate * 2.0)
    flip_pos = flip_pos & pos_mask
    y[a3:b3] = np.where(flip_pos, 0, y[a3:b3])

    # Background label noise across all 900 steps.
    bg_flip = rng.random(900) < base_noise
    y[bg_flip] = 1 - y[bg_flip]

    return X, y, phase


def run_simulation(ablate: str | None = None):
    """Re-run the full 4-phase simulation with data collection.

    Uses the methodology-faithful Campaign A workload generator
    (``synthesise_thesis_workload``) — covariate shift in P2 and
    asymmetric label flips in P3 — at the thesis-canonical phase
    boundaries (P1=1-400, P2=401-600, P3=601-700, P4=701-900).  This is
    the same generator that drives the multi-seed Campaigns B/C/D in
    ``thesis/regenerate_evaluation.py``; pinning seed=0 here makes
    Campaign A the deterministic single-seed instance of that family.

    Parameters
    ----------
    ablate : {None, "DDD", "DPD", "CDD", "CPD"}, optional
        If not ``None``, the named detector's ``check()`` method is
        replaced with a stub that always returns a *neutral* result
        (``drift_detected=False`` / ``poisoning_detected=False`` and the
        statistic fields zeroed in the same shape as ``_not_ready``).
        This is the harness used by ``scripts/ablation_run.py`` to
        measure what the framework loses when one detector is removed:
        the rest of the pipeline (RTP buffers, ATM, NDT, AIF rollback,
        cooldown FSM) keeps running unchanged, so we observe whether
        the remaining three detectors are sufficient to catch each
        injection class on this seed.
    """

    # The training set for the initial AIF and the reference set for
    # the detectors are drawn from a SEPARATE rng so that the simulation
    # rng (seed=0) drives the 900-step trace deterministically.
    rng_train = np.random.default_rng(1)

    def make_train_data(n, noise=0.05):
        X = rng_train.normal(0, 1, size=(n, 4))
        y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
        flip = rng_train.random(n) < noise
        y[flip] = 1 - y[flip]
        return X, y

    # ── Build AIF ─────────────────────────────────────────────────────────────
    X_train, y_train = make_train_data(500, noise=0.05)
    clf = RandomForestClassifier(n_estimators=50, random_state=1).fit(X_train, y_train)
    aif = AIF(clf)

    # ── RTP config ────────────────────────────────────────────────────────────
    cfg = RTPConfig(
        buffer_maxlen=2000,
        check_interval=50,
        cdd_task="classifier",
        ddd_reference_size=200, ddd_recent_size=100,
        dpd_reference_size=200, dpd_recent_size=50,
        dpd_contamination_threshold=0.08, dpd_mahal_threshold=5.0,
        cdd_reference_window=150, cdd_recent_window=50,
        cdd_perf_drop_threshold=0.12, cdd_ph_lambda=40.0,
        cpd_reference_size=200, cpd_recent_size=100,
        cpd_shadow_threshold=0.38, cpd_output_ks_alpha=0.0001,
        cpd_corr_threshold=0.60,
        mtout_cooldown_steps=150,
    )

    received_signals = []
    security_alerts = []

    def on_mtout(signal):
        received_signals.append(signal)

    def on_security(event):
        security_alerts.append(event)

    rtp = RTP(aif, config=cfg, on_mtout=on_mtout, on_security_alert=on_security)

    # ── Optional detector ablation ───────────────────────────────────────────
    # The ablation harness disables ONE detector at a time by replacing its
    # ``check()`` with a stub that returns a neutral result.  We replicate
    # the exact dataclass shape each detector uses on its ``_not_ready``
    # path so downstream code (telemetry capture, MToUT assembly, ATM
    # dispatch) does not see ``None`` and crash.
    if ablate is not None:
        from detectors.ddd import DDDResult
        from detectors.dpd import DPDResult
        from detectors.cdd import CDDResult
        from detectors.cpd import CPDResult
        import numpy as _np

        ablate = ablate.upper()
        if ablate == "DDD":
            def _stub_ddd(_lib):
                return DDDResult(
                    drift_detected=False, ks_pvalues=_np.array([]),
                    ks_drifted_features=[], mmd_statistic=0.0,
                    mmd_threshold=rtp.ddd.mmd_threshold, mmd_triggered=False,
                    reference_size=0, recent_size=0,
                    message="DDD: ablated.",
                )
            rtp.ddd.check = _stub_ddd  # type: ignore[assignment]
        elif ablate == "DPD":
            def _stub_dpd(_lib):
                return DPDResult(
                    poisoning_detected=False,
                    if_anomaly_rate=0.0,
                    if_threshold=rtp.dpd.contamination_threshold,
                    if_triggered=False, if_anomalous_indices=[],
                    mahal_max=0.0,
                    mahal_threshold=rtp.dpd.mahal_threshold,
                    mahal_hard_threshold=rtp.dpd.mahal_hard_threshold,
                    mahal_triggered=False,
                    mahal_soft_triggered=False,
                    mahal_hard_triggered=False,
                    mahal_anomalous_indices=[],
                    mahal_hard_indices=[],
                    slow_poisoning_detected=False,
                    mahal_ewma=0.0,
                    slow_poisoning_threshold=0.0,
                    slow_poisoning_alpha=0.0,
                    message="DPD: ablated.",
                )
            rtp.dpd.check = _stub_dpd  # type: ignore[assignment]
        elif ablate == "CDD":
            def _stub_cdd():
                return CDDResult(
                    drift_detected=False,
                    ph_statistic=0.0,
                    ph_threshold=rtp.cdd._ph.lambda_,
                    ph_triggered=False,
                    perf_reference=0.0, perf_recent=0.0,
                    perf_drop=0.0, window_triggered=False,
                    n_updates=0, ground_truth_mode=True,
                    message="CDD: ablated.",
                )
            rtp.cdd.check = _stub_cdd  # type: ignore[assignment]
            # CDD also exposes ``update`` to feed running statistics; the
            # stub keeps it functional so other code paths that touch
            # ``cdd.update`` still work.
        elif ablate == "CPD":
            def _stub_cpd(_lib, _lob):
                return CPDResult(
                    poisoning_detected=False,
                    shadow_divergence=0.0,
                    shadow_threshold=rtp.cpd.shadow_threshold,
                    shadow_triggered=False,
                    output_ks_pvalue=1.0,
                    output_ks_threshold=rtp.cpd.output_ks_alpha,
                    output_ks_triggered=False,
                    corr_delta_max=0.0,
                    corr_threshold=rtp.cpd.corr_threshold,
                    corr_z_max=0.0,
                    corr_z_threshold=rtp.cpd.corr_z_threshold,
                    corr_triggered=False,
                    message="CPD: ablated.",
                    shadow_source_hash=None,
                )
            rtp.cpd.check = _stub_cpd  # type: ignore[assignment]
        else:
            raise ValueError(
                f"ablate must be one of None, DDD, DPD, CDD, CPD; got {ablate!r}"
            )

    X_ref, y_ref = make_train_data(300, noise=0.05)
    lob_ref = clf.predict(X_ref)
    rtp.set_reference(X_ref, y_ref, lob_ref)

    # ── ATM components ────────────────────────────────────────────────────────
    mtp_local = MTPLocal(n_splits=3, fine_tune_first=True)

    _MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

    mtp_ext = None
    if MLFLOW_AVAILABLE:
        try:
            mtp_ext = MTPExternal(
                experiment_name="rtp_aif_retraining",
                model_name="aif_classifier",
                mlflow_uri=_MLFLOW_URI,
                tune_hyperparams=False,
                tags={"thesis": "6G-AINN", "component": "MTP-E"},
            )
        except Exception:
            mtp_ext = None

    ndt = NDT(
        current_model_getter=lambda: rtp.aif.active_estimator,
        min_score=0.65,
        min_improvement=-0.05,
        mlflow_uri=_MLFLOW_URI,
    )

    policy = ATMPolicy(
        prefer_variant=MTPVariant.LOCAL if (not MLFLOW_AVAILABLE or mtp_ext is None) else None,
        local_max_samples=600,
        use_ndt=True,
        ndt_min_accuracy=0.65,
        auto_deploy=True,
        max_retrain_attempts=2,
    )

    atm_results = []

    def on_atm_result(result):
        atm_results.append(result)

    atm = ATM(
        rtp=rtp,
        mtp_l=mtp_local,
        mtp_e=mtp_ext,
        ndt=ndt,
        policy=policy,
        on_result=on_atm_result,
    )

    rtp._on_mtout = lambda sig: (on_mtout(sig), atm.handle(sig))

    # ── Data structures for collection ────────────────────────────────────────
    step_records = []   # per-step: step, phase, y_true, y_pred, correct
    check_records = []  # per-check: detector results at each check_interval

    # Patch _run_detectors to capture per-check data
    _orig_run_detectors = rtp._run_detectors

    def _patched_run_detectors(kpi_context):
        result = _orig_run_detectors(kpi_context)
        # Capture results after run.
        # NOTE: each detector's *Result dataclass implements __bool__ that
        # returns the detection flag (e.g. ``CPDResult.__bool__`` returns
        # ``poisoning_detected``). A naive ``if cpd_r else default`` would
        # therefore zero-out every check where the detector did not fire,
        # which is exactly the silent-instrument failure mode we want the
        # figures to expose. We must use ``is not None`` so the captured
        # signal series reflects the real detector outputs at every check
        # — both the quiet ones and the triggers.
        ddd_r = rtp.last_ddd
        dpd_r = rtp.last_dpd
        cdd_r = rtp.last_cdd
        cpd_r = rtp.last_cpd
        check_records.append({
            "step": rtp._step,
            "ddd_mmd": ddd_r.mmd_statistic if ddd_r is not None else 0.0,
            "ddd_drift": ddd_r.drift_detected if ddd_r is not None else False,
            "dpd_mahal_max": dpd_r.mahal_max if (dpd_r is not None and hasattr(dpd_r, "mahal_max")) else 0.0,
            "dpd_poison": dpd_r.poisoning_detected if dpd_r is not None else False,
            "cdd_ph_statistic": cdd_r.ph_statistic if (cdd_r is not None and hasattr(cdd_r, "ph_statistic")) else 0.0,
            "cdd_perf_drop": cdd_r.perf_drop if (cdd_r is not None and hasattr(cdd_r, "perf_drop")) else 0.0,
            "cdd_drift": cdd_r.drift_detected if cdd_r is not None else False,
            "cpd_shadow_div": cpd_r.shadow_divergence if cpd_r is not None else 0.0,
            "cpd_ks_pvalue": cpd_r.output_ks_pvalue if cpd_r is not None else 1.0,
            "cpd_corr_delta": cpd_r.corr_delta_max if cpd_r is not None else 0.0,
            "cpd_poison": cpd_r.poisoning_detected if cpd_r is not None else False,
        })
        return result

    rtp._run_detectors = _patched_run_detectors

    # ── Methodology-faithful 900-step workload ────────────────────────────────
    # Pinned seed=0 (deterministic) + thesis-canonical phase boundaries
    # (P1=1-400 stable, P2=401-600 covariate-shift drift, P3=601-700
    # asymmetric label-flip poisoning, P4=701-900 recovery).  See the
    # ``synthesise_thesis_workload`` docstring for the exact mechanics.
    X_full, y_full, phase_full = synthesise_thesis_workload(
        seed=0,
        drift_magnitude=1.0,
        poison_rate=0.13,
        base_noise=0.05,
    )

    for i in range(900):
        pred = rtp.observe(X_full[i], y_true=y_full[i])
        step = i + 1
        y_pred_val = int(pred.ravel()[0]) if hasattr(pred, "__len__") else int(pred)
        step_records.append({
            "step": step,
            "phase": int(phase_full[i]),
            "y_true": int(y_full[i]),
            "y_pred": y_pred_val,
            "correct": int(y_full[i]) == y_pred_val,
        })

    return {
        "step_records": step_records,
        "check_records": check_records,
        "event_log": rtp.event_log,
        "atm_results": atm_results,
        "ndt_history": ndt.history,
        "received_signals": received_signals,
    }


# =============================================================================
# Helper utilities
# =============================================================================

def add_phase_backgrounds(ax, ymin=None, ymax=None):
    """Add coloured phase background spans to an axes."""
    ylim = ax.get_ylim()
    y0 = ymin if ymin is not None else ylim[0]
    y1 = ymax if ymax is not None else ylim[1]
    for ph, (s, e) in PHASE_RANGES.items():
        ax.axvspan(s, e, alpha=0.25, color=PHASE_COLORS[ph], zorder=0)
    ax.set_ylim(ylim)


def add_event_markers(
    ax,
    event_log,
    event_types_colors,
    label_top=False,
    event_short_labels=None,
):
    """Draw vertical event markers as bold dashed lines.

    Lines use ``alpha=1.0`` and ``linewidth=1.6`` so they remain visible
    over coloured ``axvspan`` phase backgrounds at print resolution.  When
    several event types fire at the same simulation step (e.g. MTOUT_FIRED
    is followed by MODEL_UPDATED at the same step), each subsequent marker
    is given a small horizontal offset (in display points) so the second
    line does not visually overlap the first.

    Parameters
    ----------
    ax : matplotlib axis
    event_log : iterable of RTPEvent
    event_types_colors : dict[EventType, str]
        Mapping from event type to colour.
    label_top : bool
        If ``True``, attach a small short text label at the top of each
        marker (e.g. "MToUT", "Upd", "RB").  Used on the top panel only
        to keep the rest of the figure tidy.
    event_short_labels : dict[EventType, str] or None
        Per-event short label used when ``label_top`` is true.
    """
    # Group events by step so we can horizontally nudge co-located markers.
    by_step: dict[int, list] = defaultdict(list)
    for ev in event_log:
        if ev.event_type in event_types_colors:
            by_step[ev.step].append(ev)

    # Display-point offset between co-located markers.  Small enough that
    # the lines clearly belong to the same step but separated enough that
    # both colours are visible.
    nudge_pts = 4.0

    # Preserve xlim/ylim — drawing artists with custom transforms can
    # otherwise nudge the autoscale machinery.
    saved_xlim = ax.get_xlim()
    saved_ylim = ax.get_ylim()

    for step, events in by_step.items():
        n = len(events)
        for idx, ev in enumerate(events):
            color = event_types_colors[ev.event_type]
            # Centre the group of n markers around ``step``.  The offset is
            # applied in display-points via a ``ScaledTranslation`` composed
            # on top of ``get_xaxis_transform`` (data x, axes-fraction y),
            # so the line always spans the full axis height regardless of
            # the panel's data range.
            offset_pts = (idx - (n - 1) / 2.0) * nudge_pts
            line_trans = (
                ax.get_xaxis_transform()
                + mtransforms.ScaledTranslation(
                    offset_pts / 72.0, 0, ax.figure.dpi_scale_trans
                )
            )
            ax.plot(
                [step, step], [0.0, 1.0],
                color=color,
                linestyle="--",
                alpha=1.0,
                linewidth=1.6,
                zorder=4,
                transform=line_trans,
                clip_on=True,
            )
            if label_top and event_short_labels is not None:
                short = event_short_labels.get(ev.event_type)
                if short:
                    ax.text(
                        step,
                        0.86,
                        short,
                        rotation=90,
                        ha="center",
                        va="top",
                        fontsize=6.5,
                        color=color,
                        transform=line_trans,
                        zorder=5,
                        clip_on=False,
                    )

    ax.set_xlim(saved_xlim)
    ax.set_ylim(saved_ylim)


def rolling_accuracy(step_records, window=50):
    """Compute rolling accuracy over step_records."""
    steps = [r["step"] for r in step_records]
    correct = np.array([r["correct"] for r in step_records], dtype=float)
    roll_acc = []
    for i in range(len(correct)):
        start = max(0, i - window + 1)
        roll_acc.append(correct[start:i + 1].mean())
    return steps, roll_acc


def events_per_phase(event_log):
    """Return dict[phase][event_type_name] = count."""
    result = {ph: defaultdict(int) for ph in range(1, 5)}
    for ev in event_log:
        step = ev.step
        phase = None
        for ph, (s, e) in PHASE_RANGES.items():
            if s <= step <= e:
                phase = ph
                break
        if phase is not None:
            result[phase][ev.event_type.name] += 1
    return result


# =============================================================================
# Figure 1 — Simulation timeline
# =============================================================================

def fig_01_simulation_timeline(data):
    """Controlled four-phase scenario timeline (simplified).

    Four stacked panels share a single 1-900 step axis:
        Row 1 — Rolling classifier accuracy (50-step window).
        Row 2 — CDD Page-Hinkley statistic + alarm threshold.
        Row 3 — DPD maximum Mahalanobis distance + alarm threshold.
        Row 4 — CPD shadow divergence + alarm threshold.

    Phase backgrounds + the four phase names at the top of the figure
    place every event in its phase context.  Lifecycle events
    (MToUT / model_updated / rollback) are reported separately in the
    per-phase event-counts figure and table; they are intentionally
    omitted from this view to keep the detector traces uncluttered.
    """
    step_records = data["step_records"]
    check_records = data["check_records"]

    steps, roll_acc = rolling_accuracy(step_records, window=50)
    check_steps = [r["step"] for r in check_records]
    ph_stat = [r["cdd_ph_statistic"] for r in check_records]
    mahal = [r["dpd_mahal_max"] for r in check_records]
    shadow_div = [r["cpd_shadow_div"] for r in check_records]

    fig = plt.figure(figsize=(11, 8.5))
    gs = GridSpec(4, 1, figure=fig, hspace=0.10)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]

    phase_boundaries = [PHASE_RANGES[1][1], PHASE_RANGES[2][1],
                        PHASE_RANGES[3][1]]

    # ── Panel 1: Rolling accuracy ────────────────────────────────────────────
    ax = axes[0]
    ax.plot(steps, roll_acc, color="#2c3e50", linewidth=1.4,
            label="Rolling accuracy (w=50)")
    ax.axhline(0.5, color="#7f8c8d", linestyle="--", linewidth=0.8,
               alpha=0.7, label="Chance level (0.5)")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_xlim(1, 900)
    add_phase_backgrounds(ax)

    # Phase labels along the top of the first panel only — they apply to
    # every panel below thanks to the shared X-axis.
    for ph, (s, e) in PHASE_RANGES.items():
        mid = (s + e) / 2
        ax.text(mid, 1.06, PHASE_LABELS[ph], ha="center", va="bottom",
                fontsize=8.5, color="#2c3e50", fontweight="bold",
                transform=ax.get_xaxis_transform())
    ax.legend(loc="lower left", fontsize=8, framealpha=0.85)

    # ── Panel 2: CDD Page-Hinkley ────────────────────────────────────────────
    ax = axes[1]
    ax.plot(check_steps, ph_stat, color="#8e44ad", linewidth=1.3,
            marker="o", markersize=2.5, label="PH statistic")
    ax.axhline(40.0, color="#c0392b", linestyle="--", linewidth=0.9,
               label=r"$\lambda = 40$ alarm")
    ax.set_ylabel("CDD Page–Hinkley")
    ax.set_xlim(1, 900)
    add_phase_backgrounds(ax)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)

    # ── Panel 3: DPD Mahalanobis ─────────────────────────────────────────────
    ax = axes[2]
    ax.plot(check_steps, mahal, color="#e67e22", linewidth=1.3,
            marker="s", markersize=2.5, label="Max. Mahalanobis dist.")
    ax.axhline(5.0, color="#c0392b", linestyle="--", linewidth=0.9,
               label="Threshold = 5.0")
    ax.set_ylabel("DPD Mahalanobis")
    ax.set_xlim(1, 900)
    add_phase_backgrounds(ax)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)

    # ── Panel 4: CPD shadow divergence ───────────────────────────────────────
    ax = axes[3]
    ax.plot(check_steps, shadow_div, color="#16a085", linewidth=1.3,
            marker="^", markersize=2.5, label="Shadow divergence")
    ax.axhline(0.38, color="#c0392b", linestyle="--", linewidth=0.9,
               label="Threshold = 0.38")
    ax.set_ylabel("CPD Shadow div.")
    ax.set_xlabel("Simulation step")
    ax.set_xlim(1, 900)
    add_phase_backgrounds(ax)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)

    # ── Phase boundary verticals through every panel ─────────────────────────
    for ax in axes:
        for bx in phase_boundaries:
            ax.axvline(bx, color="#34495e", linewidth=0.9,
                       linestyle="-", alpha=0.55, zorder=2)

    # Hide x-tick labels on every panel except the bottom one
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    plt.suptitle("Controlled four-phase scenario "
                 "(Campaign A, seed 0, $C{=}50$): "
                 "rolling accuracy and detector statistics",
                 fontsize=11.5, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIGURES_DIR / "fig_01_simulation_timeline.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# Figure 2 — Events per phase
# =============================================================================

def fig_02_events_per_phase(data):
    event_log = data["event_log"]
    per_phase = events_per_phase(event_log)

    event_types = [
        "DATA_DRIFT", "CONCEPT_DRIFT", "DATA_POISONING",
        "CONCEPT_POISONING", "ROLLBACK", "MODEL_UPDATED",
        "MTOUT_FIRED", "SECURITY_ALERT",
    ]
    phase_colors = [PHASE_COLORS[ph] for ph in range(1, 5)]
    phase_labels_short = ["Phase 1: Stable", "Phase 2: Drift",
                          "Phase 3: Poisoning", "Phase 4: Recovery"]

    fig, ax = plt.subplots(figsize=(9, 5))

    n_events = len(event_types)
    n_phases = 4
    bar_height = 0.18
    group_gap = 0.85

    y_positions = np.arange(n_events) * group_gap

    for pi, ph in enumerate(range(1, 5)):
        offsets = (pi - (n_phases - 1) / 2) * bar_height
        counts = [per_phase[ph].get(et, 0) for et in event_types]
        bars = ax.barh(
            y_positions + offsets, counts,
            height=bar_height * 0.9,
            color=phase_colors[pi],
            edgecolor="#555555",
            linewidth=0.6,
            label=phase_labels_short[pi],
        )
        # Value annotations
        for bar, val in zip(bars, counts):
            if val > 0:
                ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                        str(val), va="center", ha="left", fontsize=7.5)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(event_types, fontsize=9)
    ax.set_xlabel("Count")
    ax.set_title("Event Distribution Across Simulation Phases")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    ax.invert_yaxis()

    plt.tight_layout()
    out = FIGURES_DIR / "fig_02_events_per_phase.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# Figure 3 — NDT results
# =============================================================================

def fig_03_ndt_results(data):
    ndt_history = data["ndt_history"]

    if not ndt_history:
        print("  WARNING: No NDT history — skipping fig_03")
        return

    n_cycles = len(ndt_history)
    cycles = list(range(1, n_cycles + 1))
    candidate_scores = [r["candidate_score"] for r in ndt_history]
    baseline_scores = [r["baseline_score"] for r in ndt_history]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.array(cycles, dtype=float)
    bar_w = 0.35

    bars_c = ax.bar(x - bar_w / 2, candidate_scores, width=bar_w,
                    color="#2980b9", label="Candidate score", zorder=3)
    bars_b = ax.bar(x + bar_w / 2, baseline_scores, width=bar_w,
                    color="#95a5a6", label="Baseline score", zorder=3)

    # Annotate bars
    for bar in bars_c:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8)
    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8)

    # Improvement arrows
    for i, (cand, base) in enumerate(zip(candidate_scores, baseline_scores)):
        xi = x[i]
        improvement = cand - base
        arrow_color = "#27ae60" if improvement >= 0 else "#e74c3c"
        ax.annotate(
            "", xy=(xi, cand), xytext=(xi, base),
            arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.5),
        )

    ax.axhline(0.65, color="red", linestyle="--", linewidth=1.0,
               label="NDT min accuracy = 0.65", zorder=4)

    ax.set_xticks(cycles)
    ax.set_xlabel("Training Cycle")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.12)
    ax.set_title("NDT Validation Results per Training Cycle")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)

    plt.tight_layout()
    out = FIGURES_DIR / "fig_03_ndt_results.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# Figure 4 — CPD breakdown
# =============================================================================

def fig_04_cpd_breakdown(data):
    check_records = data["check_records"]

    check_steps = [r["step"] for r in check_records]
    shadow_div = [r["cpd_shadow_div"] for r in check_records]
    ks_pvalue = [max(r["cpd_ks_pvalue"], 1e-10) for r in check_records]
    corr_delta = [r["cpd_corr_delta"] for r in check_records]
    cpd_triggered = [r["cpd_poison"] for r in check_records]

    trigger_steps = [check_steps[i] for i, t in enumerate(cpd_triggered) if t]
    trigger_shadow = [shadow_div[i] for i, t in enumerate(cpd_triggered) if t]
    trigger_ks = [ks_pvalue[i] for i, t in enumerate(cpd_triggered) if t]
    trigger_corr = [corr_delta[i] for i, t in enumerate(cpd_triggered) if t]

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)

    # Panel 1: Shadow divergence
    ax = axes[0]
    ax.plot(check_steps, shadow_div, color="#8e44ad", linewidth=1.2, label="Shadow divergence")
    ax.axhline(0.38, color="red", linestyle="--", linewidth=0.9, label="Threshold = 0.38")
    if trigger_steps:
        ax.plot(trigger_steps, trigger_shadow, "ro", markersize=5,
                zorder=5, label="CPD triggered")
    ax.set_ylabel("Shadow Divergence")
    ax.set_xlim(1, 900)
    add_phase_backgrounds(ax)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)

    # Panel 2: Output KS p-value (log scale)
    ax = axes[1]
    ax.semilogy(check_steps, ks_pvalue, color="#e67e22", linewidth=1.2,
                label="Output KS p-value")
    ax.axhline(0.0001, color="red", linestyle="--", linewidth=0.9,
               label="α = 0.0001")
    if trigger_steps:
        ax.semilogy(trigger_steps, trigger_ks, "ro", markersize=5,
                    zorder=5, label="CPD triggered")
    ax.set_ylabel("KS p-value (log)")
    ax.set_xlim(1, 900)
    add_phase_backgrounds(ax)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)

    # Panel 3: Correlation delta
    ax = axes[2]
    ax.plot(check_steps, corr_delta, color="#16a085", linewidth=1.2,
            label="Corr. delta max")
    ax.axhline(0.60, color="red", linestyle="--", linewidth=0.9,
               label="Threshold = 0.60")
    if trigger_steps:
        ax.plot(trigger_steps, trigger_corr, "ro", markersize=5,
                zorder=5, label="CPD triggered")
    ax.set_ylabel("Corr. Delta Max")
    ax.set_xlabel("Simulation Step")
    ax.set_xlim(1, 900)
    add_phase_backgrounds(ax)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)

    plt.suptitle("CPD Sub-Detector Analysis", fontsize=12, y=1.01)
    plt.tight_layout()
    out = FIGURES_DIR / "fig_04_cpd_breakdown.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# Figure 5 — Detector heatmap
# =============================================================================

def fig_05_detector_heatmap(data):
    check_records = data["check_records"]

    check_steps = np.array([r["step"] for r in check_records])
    ddd_vals = np.array([1 if r["ddd_drift"] else 0 for r in check_records])
    dpd_vals = np.array([1 if r["dpd_poison"] else 0 for r in check_records])
    cdd_vals = np.array([1 if r["cdd_drift"] else 0 for r in check_records])
    cpd_vals = np.array([1 if r["cpd_poison"] else 0 for r in check_records])

    matrix = np.vstack([ddd_vals, dpd_vals, cdd_vals, cpd_vals])
    detector_labels = ["DDD", "DPD", "CDD", "CPD"]

    fig, ax = plt.subplots(figsize=(9, 4))

    im = ax.imshow(
        matrix, aspect="auto", cmap="Reds", vmin=0, vmax=1,
        extent=[0, len(check_steps), -0.5, 3.5],
        origin="lower",
        interpolation="nearest",
    )

    # Phase boundary vertical lines
    phase_boundaries_check = []
    for ph_end in [400, 600, 700]:
        idx = np.searchsorted(check_steps, ph_end)
        if idx < len(check_steps):
            phase_boundaries_check.append(idx)

    for bi in phase_boundaries_check:
        ax.axvline(bi, color="#333333", linewidth=1.2, zorder=5)

    # Phase labels on x-axis top
    phase_boundaries_all = [0] + phase_boundaries_check + [len(check_steps)]
    phase_mids = [(phase_boundaries_all[i] + phase_boundaries_all[i + 1]) / 2
                  for i in range(4)]
    phase_short = ["P1", "P2", "P3", "P4"]
    for mid, label in zip(phase_mids, phase_short):
        ax.text(mid, 3.65, label, ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(detector_labels)
    ax.set_xlabel("Check Index (50-step intervals)")
    ax.set_title("Detector Trigger Map Across Simulation")

    cbar = plt.colorbar(im, ax=ax, orientation="vertical", fraction=0.025, pad=0.02)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Not triggered", "Triggered"])

    plt.tight_layout()
    out = FIGURES_DIR / "fig_05_detector_heatmap.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Thesis figure generator")
    print(f"Output: {FIGURES_DIR}")
    print("=" * 60)

    print("\n[1/6] Running 4-phase simulation ...")
    data = run_simulation()
    n_steps = len(data["step_records"])
    n_checks = len(data["check_records"])
    n_events = len(data["event_log"])
    n_atm = len(data["atm_results"])
    n_ndt = len(data["ndt_history"])
    print(f"      Steps: {n_steps} | Checks: {n_checks} | "
          f"Events: {n_events} | ATM cycles: {n_atm} | NDT: {n_ndt}")

    print("\n[2/6] Generating fig_01_simulation_timeline.pdf ...")
    try:
        fig_01_simulation_timeline(data)
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\n[3/6] Generating fig_02_events_per_phase.pdf ...")
    try:
        fig_02_events_per_phase(data)
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\n[4/6] Generating fig_03_ndt_results.pdf ...")
    try:
        fig_03_ndt_results(data)
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\n[5/6] Generating fig_04_cpd_breakdown.pdf ...")
    try:
        fig_04_cpd_breakdown(data)
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\n[6/6] Generating fig_05_detector_heatmap.pdf ...")
    try:
        fig_05_detector_heatmap(data)
    except Exception as e:
        print(f"  ERROR: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GENERATED FIGURES")
    print("=" * 60)
    generated = sorted(FIGURES_DIR.glob("*.pdf"))
    for f in generated:
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name}  ({size_kb:.1f} KB)")

    print(f"\nAll figures saved to: {FIGURES_DIR}")

    # ── Simulation summary ────────────────────────────────────────────────────
    print("\nSIMULATION SUMMARY")
    print("-" * 40)
    event_counts = {}
    for ev in data["event_log"]:
        event_counts[ev.event_type.name] = event_counts.get(ev.event_type.name, 0) + 1
    for k, v in sorted(event_counts.items()):
        print(f"  {k:<30} {v}")

    print(f"\n  ATM training cycles: {n_atm}")
    for i, r in enumerate(data["atm_results"], 1):
        variant = r.variant_used.value if r.variant_used else "none"
        ndt_str = "pass" if r.ndt_passed else ("fail" if r.ndt_passed is False else "skip")
        print(f"  Cycle {i}: variant={variant}, ndt={ndt_str}, deployed={r.deployed}")

    print("\n  NDT history:")
    for i, rec in enumerate(data["ndt_history"], 1):
        print(f"  [{i}] candidate={rec['candidate_score']:.4f}, "
              f"baseline={rec['baseline_score']:.4f}, "
              f"improvement={rec['improvement']:+.4f}, "
              f"passed={rec['passed']}")


if __name__ == "__main__":
    main()
