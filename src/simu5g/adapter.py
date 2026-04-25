"""
simu5g/adapter.py — Bridge between Simu5G data and the RTP observer framework.

This adapter translates parsed Simu5G KPI data into the exact format expected
by rtp.observe(x, y_true, kpi_context), enabling direct evaluation of the
AI Management Framework on Simu5G simulation outputs.

Architecture:
    Simu5G (.vec/.sca) → parser.py → Simu5GAdapter → RTP.observe()
                                                      ├─ DDD
                                                      ├─ DPD
                                                      ├─ CDD
                                                      └─ CPD → ATM → NDT

The adapter supports three operation modes:
  1. REPLAY mode:  Feed pre-recorded Simu5G data step-by-step through RTP
  2. STREAM mode:  Poll a live Simu5G result directory for new .vec files
  3. OFFLINE mode: Use Simu5G-calibrated synthetic data (no OMNeT++ needed)

Reference: 3GPP TS 38.215 (NR measurements), Simu5G IEEE Access paper
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd

from simu5g.parser import Simu5GParser

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Phase annotation for Simu5G time-series
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PhaseConfig:
    """
    Map Simu5G simulation time (seconds) to framework phases.

    Aligns with the 5-phase protocol from enhanced_simulation.py:
      Phase 1: Stable operation    (normal radio conditions)
      Phase 2: Concept drift       (UE mobility → handover pressure)
      Phase 3a: Subtle anomaly     (mild interference)
      Phase 3b: Aggressive anomaly (strong interference / jamming)
      Phase 4: Recovery            (interference cleared)
    """
    phase1_end:   float = 40.0   # seconds
    phase2_end:   float = 60.0
    phase3a_end:  float = 65.0
    phase3b_end:  float = 70.0
    phase4_end:   float = 90.0

    def get_phase(self, sim_time: float) -> str:
        """Return phase name for a given simulation time."""
        if sim_time <= self.phase1_end:
            return "stable"
        elif sim_time <= self.phase2_end:
            return "drift"
        elif sim_time <= self.phase3a_end:
            return "subtle_poison"
        elif sim_time <= self.phase3b_end:
            return "aggressive_poison"
        else:
            return "recovery"

    def is_anomaly_phase(self, sim_time: float) -> bool:
        """True if the simulation time falls in an anomaly phase."""
        phase = self.get_phase(sim_time)
        return phase in ("drift", "subtle_poison", "aggressive_poison")


# ──────────────────────────────────────────────────────────────────────────────
# Step result (one observation cycle)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Simu5GStepResult:
    """Record of one RTP.observe() call with Simu5G data."""
    step: int
    sim_time: float
    phase: str
    ue_id: int

    # Input features
    rsrp: float
    sinr: float
    throughput: float
    latency: float

    # Ground-truth and prediction
    y_true: int
    y_pred: int
    correct: bool

    # Detector alerts at this step (if check cycle)
    ddd_alert: bool = False
    dpd_alert: bool = False
    cdd_alert: bool = False
    cpd_alert: bool = False

    # RTP event counts
    mtout_fired: bool = False
    security_alert: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Main adapter class
# ──────────────────────────────────────────────────────────────────────────────

class Simu5GAdapter:
    """
    Bridge between Simu5G simulation output and the RTP observer framework.

    Usage (REPLAY mode):
        from simu5g.adapter import Simu5GAdapter
        from simu5g.parser import Simu5GParser

        parser = Simu5GParser()
        df = parser.from_vec("results/run0.vec")
        X, timestamps = parser.to_kpi_matrix(df)
        y = parser.extract_handover_labels(df)

        adapter = Simu5GAdapter()
        results = adapter.replay(X, y, timestamps)

    Usage (OFFLINE mode with synthetic data):
        from simu5g.adapter import Simu5GAdapter

        adapter = Simu5GAdapter()
        results = adapter.run_offline(n_samples=900, seed=42)
    """

    def __init__(
        self,
        phase_config: Optional[PhaseConfig] = None,
        check_interval: int = 50,
    ) -> None:
        self.phase_config = phase_config or PhaseConfig()
        self.check_interval = check_interval
        self.step_results: list[Simu5GStepResult] = []

    def replay(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: np.ndarray,
        *,
        ue_id: int = 0,
        seed: int = 42,
        verbose: bool = True,
    ) -> list[Simu5GStepResult]:
        """
        Replay Simu5G KPI data through the full RTP → ATM → NDT pipeline.

        Parameters
        ----------
        X : np.ndarray, shape (n, 4)
            Feature matrix [rsrp, sinr, throughput, latency]
        y : np.ndarray, shape (n,)
            Ground-truth handover labels
        timestamps : np.ndarray, shape (n,)
            Simulation timestamps (seconds)
        ue_id : int
            UE identifier (for logging)
        seed : int
            Random seed for classifier initialisation
        verbose : bool
            Print progress

        Returns
        -------
        list[Simu5GStepResult]
        """
        # Lazy imports to avoid circular dependency
        from sklearn.ensemble import RandomForestClassifier
        from aif.aif import AIF
        from rtp.rtp import RTP, RTPConfig, MToUTSignal, RTPEvent
        from atm.atm import ATM, ATMPolicy, MTPVariant
        from atm.mtp_l import MTPLocal
        from ndt.ndt import NDT

        n = len(X)
        assert len(y) == n and len(timestamps) == n, \
            f"Shape mismatch: X={X.shape}, y={y.shape}, timestamps={timestamps.shape}"

        if verbose:
            log.info("Replaying %d Simu5G samples through RTP pipeline", n)

        # ── Build framework components ────────────────────────────────────
        rng = np.random.default_rng(seed)

        # Initial training on first 20% of stable data
        n_train = max(50, int(n * 0.2))
        X_train, y_train = X[:n_train], y[:n_train]

        clf = RandomForestClassifier(
            n_estimators=50, random_state=seed
        ).fit(X_train, y_train)
        aif = AIF(clf)

        cfg = RTPConfig(
            buffer_maxlen=2000,
            check_interval=self.check_interval,
            cdd_task="classifier",
            ddd_reference_size=200,
            ddd_recent_size=100,
            dpd_reference_size=200,
            dpd_recent_size=50,
            dpd_contamination_threshold=0.08,
            dpd_mahal_threshold=5.0,
            cdd_reference_window=150,
            cdd_recent_window=50,
            cdd_perf_drop_threshold=0.12,
            cdd_ph_lambda=40.0,
            cpd_reference_size=200,
            cpd_recent_size=100,
            cpd_shadow_threshold=0.38,
            cpd_output_ks_alpha=0.0001,
            cpd_corr_threshold=0.60,
            mtout_cooldown_steps=150,
        )

        mtout_signals = []
        security_alerts = []

        def _on_mtout(sig: MToUTSignal):
            mtout_signals.append(sig)

        def _on_security(evt: RTPEvent):
            security_alerts.append(evt)

        rtp = RTP(aif, config=cfg, on_mtout=_on_mtout, on_security_alert=_on_security)

        # Set reference from training data
        lob_ref = clf.predict(X_train)
        rtp.set_reference(X_train, y_train, lob_ref)

        # ATM setup
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

        atm = ATM(
            rtp=rtp, mtp_l=mtp_l, mtp_e=None,
            ndt=ndt, policy=policy,
        )

        # Wire ATM into RTP
        prev_mtout_count = 0

        def _mtout_with_atm(sig: MToUTSignal):
            _on_mtout(sig)
            atm.handle(sig)

        rtp._on_mtout = _mtout_with_atm

        # ── Replay loop ───────────────────────────────────────────────────
        self.step_results = []
        prev_mtout_len = 0
        prev_sec_len = 0

        for i in range(n):
            step = i + 1
            x_i = X[i]
            y_i = int(y[i])
            t_i = float(timestamps[i])

            phase = self.phase_config.get_phase(t_i)

            # Observe
            pred = rtp.observe(x_i, y_true=y_i)
            pred_val = int(pred.ravel()[0])

            # Check detector states
            ddd_alert = (rtp.last_ddd is not None and rtp.last_ddd.drift_detected)
            dpd_alert = (rtp.last_dpd is not None and rtp.last_dpd.poisoning_detected)
            cdd_alert = (rtp.last_cdd is not None and rtp.last_cdd.drift_detected)
            cpd_alert = (rtp.last_cpd is not None and rtp.last_cpd.poisoning_detected)

            # Check if MToUT or security alert fired this step
            mtout_fired = len(mtout_signals) > prev_mtout_len
            sec_fired = len(security_alerts) > prev_sec_len
            prev_mtout_len = len(mtout_signals)
            prev_sec_len = len(security_alerts)

            result = Simu5GStepResult(
                step=step,
                sim_time=t_i,
                phase=phase,
                ue_id=ue_id,
                rsrp=float(x_i[0]),
                sinr=float(x_i[1]),
                throughput=float(x_i[2]),
                latency=float(x_i[3]),
                y_true=y_i,
                y_pred=pred_val,
                correct=(pred_val == y_i),
                ddd_alert=ddd_alert,
                dpd_alert=dpd_alert,
                cdd_alert=cdd_alert,
                cpd_alert=cpd_alert,
                mtout_fired=mtout_fired,
                security_alert=sec_fired,
            )
            self.step_results.append(result)

            if verbose and step % 100 == 0:
                acc = sum(r.correct for r in self.step_results[-100:]) / 100
                log.info(
                    "Step %4d | t=%.2fs | phase=%-18s | acc=%.3f | MToUT=%d | SecAlert=%d",
                    step, t_i, phase, acc,
                    len(mtout_signals), len(security_alerts),
                )

        return self.step_results

    # ── Results aggregation ───────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """Convert step results to a pandas DataFrame for analysis."""
        if not self.step_results:
            return pd.DataFrame()

        records = []
        for r in self.step_results:
            records.append({
                "step": r.step,
                "sim_time": r.sim_time,
                "phase": r.phase,
                "ue_id": r.ue_id,
                "rsrp": r.rsrp,
                "sinr": r.sinr,
                "throughput": r.throughput,
                "latency": r.latency,
                "y_true": r.y_true,
                "y_pred": r.y_pred,
                "correct": r.correct,
                "ddd_alert": r.ddd_alert,
                "dpd_alert": r.dpd_alert,
                "cdd_alert": r.cdd_alert,
                "cpd_alert": r.cpd_alert,
                "mtout_fired": r.mtout_fired,
                "security_alert": r.security_alert,
            })
        return pd.DataFrame(records)

    def phase_accuracy(self) -> dict[str, float]:
        """Compute per-phase prediction accuracy."""
        df = self.to_dataframe()
        if df.empty:
            return {}
        return df.groupby("phase")["correct"].mean().to_dict()

    def detector_summary(self) -> dict[str, dict]:
        """
        Compute TP/FP/FN per detector based on phase annotations.

        TP = alert during anomaly phase
        FP = alert during stable/recovery phase
        FN = anomaly phase ends without alert
        """
        df = self.to_dataframe()
        if df.empty:
            return {}

        anomaly_phases = {"drift", "subtle_poison", "aggressive_poison"}
        stable_phases = {"stable", "recovery"}

        summary = {}
        for det in ["ddd", "dpd", "cdd", "cpd"]:
            col = f"{det}_alert"
            alerts = df[df[col]]

            tp = len(alerts[alerts["phase"].isin(anomaly_phases)])
            fp = len(alerts[alerts["phase"].isin(stable_phases)])

            # FN: check if any anomaly phase had zero alerts
            fn = 0
            for phase in anomaly_phases:
                phase_alerts = alerts[alerts["phase"] == phase]
                if len(phase_alerts) == 0:
                    phase_data = df[df["phase"] == phase]
                    if len(phase_data) > 0:
                        fn += 1

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            summary[det] = {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": prec, "recall": rec,
            }

        return summary

    def event_timeline(self) -> pd.DataFrame:
        """Return a DataFrame of all MToUT and security alert events."""
        df = self.to_dataframe()
        events = df[(df["mtout_fired"]) | (df["security_alert"])]
        return events[["step", "sim_time", "phase", "mtout_fired", "security_alert"]]
