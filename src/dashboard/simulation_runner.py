"""
simulation_runner.py
Runs the 4-phase RTP simulation and collects all data for the dashboard.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from sklearn.ensemble import RandomForestClassifier

from aif.aif import AIF
from rtp.rtp import RTP, RTPConfig
from atm.atm import ATM, ATMPolicy, ATMResult, MTPVariant
from atm.mtp_l import MTPLocal
from ndt.ndt import NDT


# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class StepRecord:
    step: int
    phase: int
    phase_name: str
    prediction: int
    y_true: int
    correct: bool
    # Detector metrics — updated every check_interval, carried forward
    ddd_ks_max_pval: Optional[float] = None
    ddd_mmd: Optional[float] = None
    ddd_triggered: Optional[bool] = None
    dpd_if_rate: Optional[float] = None
    dpd_mahal_max: Optional[float] = None
    dpd_triggered: Optional[bool] = None
    cdd_ph_stat: Optional[float] = None
    cdd_perf_drop: Optional[float] = None
    cdd_triggered: Optional[bool] = None
    cpd_shadow_div: Optional[float] = None
    cpd_triggered: Optional[bool] = None


@dataclass
class SimulationData:
    steps: List[StepRecord] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    atm_cycles: List[Dict[str, Any]] = field(default_factory=list)
    ndt_history: List[Dict[str, Any]] = field(default_factory=list)
    check_steps: List[int] = field(default_factory=list)
    wall_time_s: float = 0.0


# ── Runner ───────────────────────────────────────────────────────────────────

def run_full_simulation(seed: int = 42, use_mtp_l_only: bool = True) -> SimulationData:
    """Run the complete 4-phase simulation and return collected data."""
    t0 = time.time()
    data = SimulationData()
    rng = np.random.default_rng(seed)

    # ── data factory ─────────────────────────────────────────────────────
    def make_data(n: int, noise: float = 0.05) -> tuple:
        X = rng.normal(0, 1, size=(n, 4))
        y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
        flip = rng.random(n) < noise
        y[flip] = 1 - y[flip]
        return X.astype(float), y.astype(int)

    # ── build AIF ────────────────────────────────────────────────────────
    X_train, y_train = make_data(500, noise=0.05)
    clf = RandomForestClassifier(n_estimators=50, random_state=1).fit(X_train, y_train)
    aif = AIF(clf)

    # ── RTP config ───────────────────────────────────────────────────────
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

    rtp = RTP(aif, config=cfg)

    # ── ATM wiring ───────────────────────────────────────────────────────
    mtp_local = MTPLocal(n_splits=3, fine_tune_first=True)

    def _current_model():
        return rtp.aif.active_estimator

    ndt = NDT(
        current_model_getter=_current_model,
        min_score=0.65,
        min_improvement=-0.05,
    )
    policy = ATMPolicy(
        prefer_variant=MTPVariant.LOCAL,   # MTP-L only (no MLflow needed)
        local_max_samples=600,
        use_ndt=True,
        ndt_min_accuracy=0.65,
        auto_deploy=True,
        max_retrain_attempts=2,
    )

    def on_atm_result(result: ATMResult):
        data.atm_cycles.append({
            "step": rtp._step,
            "status": result.status.name,
            "variant": result.variant_used.value if result.variant_used else "none",
            "ndt_passed": result.ndt_passed,
            "deployed": result.deployed,
            "train_score": getattr(result, "train_score", None),
        })

    atm = ATM(
        rtp=rtp,
        mtp_l=mtp_local,
        mtp_e=None,          # local-only for dashboard speed
        ndt=ndt,
        policy=policy,
        on_result=on_atm_result,
    )

    # Wire callbacks
    def on_mtout(signal):
        data.events.append({
            "step": signal.step,
            "type": "MTOUT_FIRED",
            "severity": signal.severity(),
            "reasons": [r.name for r in signal.reasons],
        })
        atm.handle(signal)

    def on_security(event):
        data.events.append({
            "step": event.step,
            "type": "SECURITY_ALERT",
            "severity": "CRITICAL",
            "reasons": [event.details.get("type", "")],
        })

    rtp._on_mtout = on_mtout
    rtp._on_security_alert = on_security

    # ── Reference ────────────────────────────────────────────────────────
    X_ref, y_ref = make_data(300, noise=0.05)
    lob_ref = clf.predict(X_ref).astype(float)
    rtp.set_reference(X_ref, y_ref, lob_ref)

    # ── Cached detector results (carry-forward between check steps) ──────
    last_ddd = [None]
    last_dpd = [None]
    last_cdd = [None]
    last_cpd = [None]

    def run_phase(phase_num: int, phase_name: str, X: np.ndarray, y: np.ndarray):
        for i in range(len(X)):
            pred = rtp.observe(X[i], y_true=float(y[i]))
            prediction = int(pred.ravel()[0])
            correct = prediction == int(y[i])
            step = rtp._step

            # Check if detectors just ran (carry-forward otherwise)
            if step % cfg.check_interval == 0:
                data.check_steps.append(step)
                if rtp.last_ddd: last_ddd[0] = rtp.last_ddd
                if rtp.last_dpd: last_dpd[0] = rtp.last_dpd
                if rtp.last_cdd: last_cdd[0] = rtp.last_cdd
                if rtp.last_cpd: last_cpd[0] = rtp.last_cpd

            ddd = last_ddd[0]
            dpd = last_dpd[0]
            cdd = last_cdd[0]
            cpd = last_cpd[0]

            data.steps.append(StepRecord(
                step=step,
                phase=phase_num,
                phase_name=phase_name,
                prediction=prediction,
                y_true=int(y[i]),
                correct=correct,
                ddd_ks_max_pval=float(max(ddd.ks_pvalues)) if ddd and len(ddd.ks_pvalues) > 0 else None,
                ddd_mmd=float(ddd.mmd_statistic) if ddd and ddd.mmd_statistic is not None else None,
                ddd_triggered=bool(ddd.drift_detected) if ddd else None,
                dpd_if_rate=float(dpd.if_anomaly_rate) if dpd else None,
                dpd_mahal_max=float(dpd.mahal_max) if dpd and dpd.mahal_max is not None else None,
                dpd_triggered=bool(dpd.poisoning_detected) if dpd else None,
                cdd_ph_stat=float(cdd.ph_statistic) if cdd else None,
                cdd_perf_drop=float(cdd.perf_drop) if cdd and cdd.perf_drop is not None else None,
                cdd_triggered=bool(cdd.drift_detected) if cdd else None,
                cpd_shadow_div=float(cpd.shadow_divergence) if cpd and cpd.shadow_divergence is not None else None,
                cpd_triggered=bool(cpd.poisoning_detected) if cpd else None,
            ))

    # ── Phase 1: Stable ──────────────────────────────────────────────────
    X1, y1 = make_data(400, noise=0.05)
    run_phase(1, "Stable", X1, y1)

    # ── Phase 2: Concept Drift ───────────────────────────────────────────
    X2, y2 = make_data(200, noise=0.55)
    run_phase(2, "Concept Drift", X2, y2)

    # ── Phase 3: Data Poisoning ──────────────────────────────────────────
    X_clean, y_clean = make_data(90, noise=0.05)
    X_inject = rng.uniform(30, 50, size=(10, 4)).astype(float)
    y_inject = rng.integers(0, 2, size=10).astype(int)
    X3 = np.vstack([X_clean, X_inject])
    y3 = np.concatenate([y_clean, y_inject])
    idx = rng.permutation(len(X3))
    X3, y3 = X3[idx], y3[idx]
    run_phase(3, "Data Poisoning", X3, y3)

    # ── Phase 4: Recovery ────────────────────────────────────────────────
    X4, y4 = make_data(200, noise=0.05)
    run_phase(4, "Recovery", X4, y4)

    # ── Collect remaining RTP events ──────────────────────────────────────
    rtp_event_types_already_captured = {"MTOUT_FIRED", "SECURITY_ALERT"}
    for event in rtp.event_log:
        if event.event_type.name not in rtp_event_types_already_captured:
            data.events.append({
                "step": event.step,
                "type": event.event_type.name,
                "severity": "INFO",
                "reasons": [event.event_type.name],
            })

    # ── NDT history ───────────────────────────────────────────────────────
    if hasattr(ndt, "history"):
        data.ndt_history = list(ndt.history)

    data.wall_time_s = time.time() - t0
    return data
