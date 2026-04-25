"""
app.py — RTP Observer Dashboard
================================
Run with:
    cd C:\\Users\\taieb\\PycharmProjects\\rtp_observer
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.simulation_runner import run_full_simulation, SimulationData

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RTP Observer — AI-Native Network Monitor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 1.6rem; }
  .section-title { color: #2c3e50; border-bottom: 2px solid #3498db;
                   padding-bottom: 4px; margin-bottom: 12px; }
  .badge-stable   { background:#27ae60; color:white; padding:2px 8px;
                    border-radius:10px; font-size:11px; font-weight:bold; }
  .badge-drift    { background:#e67e22; color:white; padding:2px 8px;
                    border-radius:10px; font-size:11px; font-weight:bold; }
  .badge-poison   { background:#e74c3c; color:white; padding:2px 8px;
                    border-radius:10px; font-size:11px; font-weight:bold; }
  .badge-recovery { background:#3498db; color:white; padding:2px 8px;
                    border-radius:10px; font-size:11px; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# ── Phase constants ───────────────────────────────────────────────────────────
PHASE_RANGES  = {1: (1, 400), 2: (401, 600), 3: (601, 700), 4: (701, 900)}
PHASE_NAMES   = {1: "Phase 1: Stable", 2: "Phase 2: Concept Drift",
                 3: "Phase 3: Data Poisoning", 4: "Phase 4: Recovery"}
PHASE_COLOURS = {
    1: "rgba(39,174,96,0.08)",
    2: "rgba(230,126,34,0.13)",
    3: "rgba(231,76,60,0.15)",
    4: "rgba(52,152,219,0.08)",
}
PHASE_EMOJIS  = {1: "🟢", 2: "🟡", 3: "🔴", 4: "🔵"}

def add_phase_bands(fig: go.Figure, max_step: int, row: int = 1, col: int = 1):
    """Add translucent background bands + labels for each phase."""
    for phase, (s, e) in PHASE_RANGES.items():
        fig.add_vrect(
            x0=s, x1=min(e, max_step),
            fillcolor=PHASE_COLOURS[phase],
            layer="below", line_width=0,
            row=row, col=col,
        )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 RTP Observer")
    st.markdown("*AI-Native 6G Network Runtime Monitor*")
    st.divider()

    run_btn = st.button("▶  Run / Refresh Simulation", type="primary",
                        use_container_width=True)
    if run_btn:
        st.session_state.pop("sim_data", None)

    st.divider()
    st.markdown("**Display settings**")
    rolling_win = st.slider("Rolling accuracy window (steps)", 10, 100, 50, 5)

    st.divider()
    st.markdown("**Phase reference**")
    for p, name in PHASE_NAMES.items():
        s, e = PHASE_RANGES[p]
        st.markdown(f"{PHASE_EMOJIS[p]} **{name}** — steps {s}–{e}")

    st.divider()
    st.markdown("**Thresholds**")
    st.markdown("| Detector | Threshold |\n|---|---|\n"
                "| DDD MMD² | 0.05 |\n"
                "| DPD IF rate | 8 % |\n"
                "| DPD Mahal | 5.0 σ |\n"
                "| CDD PH λ | 40.0 |\n"
                "| CDD drop | 12 pp |\n"
                "| CPD shadow | 38 % |\n"
                "| NDT floor | 65 % |")

# ── Load simulation ───────────────────────────────────────────────────────────
if "sim_data" not in st.session_state:
    with st.spinner("⚙️  Running 4-phase simulation (900 steps)…"):
        st.session_state["sim_data"] = run_full_simulation()

data: SimulationData = st.session_state["sim_data"]

# ── Build DataFrames ──────────────────────────────────────────────────────────
steps_df = pd.DataFrame([vars(r) for r in data.steps])
steps_df["rolling_acc"] = (
    steps_df["correct"].rolling(rolling_win, min_periods=1).mean() * 100
)
steps_df["rolling_err"] = 100 - steps_df["rolling_acc"]

events_df  = pd.DataFrame(data.events)  if data.events  else pd.DataFrame(
    columns=["step","type","severity","reasons"])
atm_df     = pd.DataFrame(data.atm_cycles) if data.atm_cycles else pd.DataFrame(
    columns=["step","status","variant","ndt_passed","deployed"])

check_df   = steps_df[steps_df["ddd_mmd"].notna()].copy()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📡 RTP Observer — AI-Native Network Runtime Monitor")
st.caption(
    "Real-time visualisation of the Runtime Pipeline (RTP) for a 6G AI-native network. "
    f"Simulation completed in **{data.wall_time_s:.1f} s**."
)

# ── Step slider ───────────────────────────────────────────────────────────────
total_steps = len(steps_df)
step_cursor = st.slider("🕐 Simulation step cursor", 1, total_steps, total_steps,
                        key="step_cursor")

cur_steps  = steps_df[steps_df["step"] <= step_cursor]
cur_check  = check_df[check_df["step"] <= step_cursor]
cur_events = events_df[events_df["step"] <= step_cursor] if not events_df.empty else events_df
cur_atm    = atm_df[atm_df["step"] <= step_cursor]       if not atm_df.empty    else atm_df

st.divider()

# ── KPI cards ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
last = cur_steps.iloc[-1] if len(cur_steps) else None

with c1:
    phase_n = int(last["phase"]) if last is not None else 1
    st.metric("Phase", f"{PHASE_EMOJIS[phase_n]} {last['phase_name']}" if last is not None else "—")

with c2:
    acc = float(last["rolling_acc"]) if last is not None else 0.0
    st.metric("Rolling Acc.", f"{acc:.1f}%",
              delta=f"{acc-65:.1f}pp vs NDT floor",
              delta_color="normal")

with c3:
    n_mtout = len(cur_events[cur_events["type"] == "MTOUT_FIRED"]) if not cur_events.empty else 0
    st.metric("MToUT Fires", n_mtout)

with c4:
    n_sec = len(cur_events[cur_events["type"] == "SECURITY_ALERT"]) if not cur_events.empty else 0
    st.metric("Security Alerts", n_sec,
              delta=str(n_sec) if n_sec else None,
              delta_color="inverse")

with c5:
    st.metric("ATM Cycles", len(cur_atm))

with c6:
    n_dep = int(cur_atm["deployed"].sum()) if not cur_atm.empty and "deployed" in cur_atm.columns else 0
    st.metric("Models Deployed", n_dep)

with c7:
    st.metric("Sim. time", f"{data.wall_time_s:.1f} s")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 1 — Rolling Accuracy + Phase Timeline
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">Model Performance Over Time</p>',
            unsafe_allow_html=True)

fig_acc = go.Figure()

# Phase bands
for phase, (s, e) in PHASE_RANGES.items():
    end = min(e, step_cursor)
    if end >= s:
        mid = (s + end) / 2
        fig_acc.add_vrect(x0=s, x1=end,
                          fillcolor=PHASE_COLOURS[phase],
                          layer="below", line_width=0)
        fig_acc.add_annotation(
            x=mid, y=103, text=PHASE_NAMES[phase],
            showarrow=False, font=dict(size=9, color="#555"),
            xanchor="center",
        )

# Rolling accuracy fill
fig_acc.add_trace(go.Scatter(
    x=cur_steps["step"], y=cur_steps["rolling_acc"],
    name=f"Rolling accuracy (w={rolling_win})",
    mode="lines",
    line=dict(color="#2980b9", width=2.5),
    fill="tozeroy", fillcolor="rgba(41,128,185,0.10)",
))

# NDT floor
fig_acc.add_hline(y=65, line_dash="dash", line_color="#e74c3c",
                  annotation_text="NDT floor 65%",
                  annotation_position="bottom right",
                  annotation_font_size=10)

# MToUT markers
if not cur_events.empty:
    mt = cur_events[cur_events["type"] == "MTOUT_FIRED"]
    if len(mt):
        mt_acc = []
        for s in mt["step"]:
            row = cur_steps[cur_steps["step"] == s]
            mt_acc.append(float(row["rolling_acc"].iloc[0]) if len(row) else 50.0)
        fig_acc.add_trace(go.Scatter(
            x=mt["step"], y=mt_acc, mode="markers",
            name="MToUT", marker=dict(symbol="triangle-up", size=13,
                                       color="#e74c3c",
                                       line=dict(color="white", width=1.5)),
        ))
    # Model update markers
    upd = cur_events[cur_events["type"] == "MODEL_UPDATED"]
    if len(upd):
        upd_acc = []
        for s in upd["step"]:
            row = cur_steps[cur_steps["step"] == s]
            upd_acc.append(float(row["rolling_acc"].iloc[0]) if len(row) else 50.0)
        fig_acc.add_trace(go.Scatter(
            x=upd["step"], y=upd_acc, mode="markers",
            name="Model updated", marker=dict(symbol="star", size=14,
                                               color="#27ae60",
                                               line=dict(color="white", width=1.5)),
        ))

fig_acc.update_layout(
    height=280,
    margin=dict(l=0, r=0, t=20, b=0),
    yaxis=dict(range=[0, 107], title="Accuracy (%)",
               gridcolor="#f0f0f0", zeroline=False),
    xaxis=dict(title="Step", gridcolor="#f0f0f0"),
    legend=dict(orientation="h", yanchor="bottom", y=1.01,
                xanchor="right", x=1),
    plot_bgcolor="white", paper_bgcolor="white",
    hovermode="x unified",
)
st.plotly_chart(fig_acc, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 2 — Detector Metrics (4 tabs)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">Detector Metrics</p>',
            unsafe_allow_html=True)

tab_ddd, tab_dpd, tab_cdd, tab_cpd = st.tabs([
    "🔵 DDD — Data Drift",
    "🟠 DPD — Data Poisoning",
    "🟢 CDD — Concept Drift",
    "🔴 CPD — Concept Poisoning",
])

def _marker_colours(series_bool):
    return series_bool.map({True: "#e74c3c", False: "#555", None: "#555"})

def _threshold_line(fig, y, label, row=1, col=1):
    fig.add_hline(y=y, line_dash="dash", line_color="#e74c3c",
                  annotation_text=label, annotation_position="bottom right",
                  annotation_font_size=9, row=row, col=col)

# ── DDD ───────────────────────────────────────────────────────────────────────
with tab_ddd:
    col_l, col_r = st.columns(2)

    with col_l:
        fig = go.Figure()
        add_phase_bands(fig, step_cursor)
        fig.add_trace(go.Scatter(
            x=cur_check["step"],
            y=cur_check["ddd_mmd"],
            mode="lines+markers", name="MMD²",
            line=dict(color="#8e44ad", width=2),
            marker=dict(size=6,
                        color=_marker_colours(cur_check["ddd_triggered"]).tolist()),
        ))
        _threshold_line(fig, 0.05, "threshold 0.05")
        fig.update_layout(title="MMD² (multivariate drift)", height=230,
                          margin=dict(l=0,r=0,t=30,b=0),
                          yaxis=dict(title="MMD²", gridcolor="#f0f0f0", rangemode="tozero"),
                          xaxis=dict(title="Step"),
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Red markers = DDD triggered. Values > 0.05 indicate multivariate input drift.")

    with col_r:
        fig2 = go.Figure()
        add_phase_bands(fig2, step_cursor)
        fig2.add_trace(go.Scatter(
            x=cur_check["step"],
            y=cur_check["ddd_ks_max_pval"],
            mode="lines+markers", name="Max KS p-value",
            line=dict(color="#3498db", width=2),
            marker=dict(size=6),
        ))
        _threshold_line(fig2, 0.05, "α=0.05")
        fig2.update_layout(title="KS max p-value (per feature)", height=230,
                           margin=dict(l=0,r=0,t=30,b=0),
                           yaxis=dict(title="p-value", gridcolor="#f0f0f0"),
                           xaxis=dict(title="Step"),
                           plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Bonferroni-corrected per-feature KS test. Low p-value = feature drift.")

    n_ddd = int(cur_check["ddd_triggered"].sum()) if "ddd_triggered" in cur_check.columns else 0
    st.info(f"**DDD summary:** {n_ddd} triggers in {len(cur_check)} checks "
            f"({100*n_ddd/max(len(cur_check),1):.1f}% trigger rate)")

# ── DPD ───────────────────────────────────────────────────────────────────────
with tab_dpd:
    col_l, col_r = st.columns(2)

    with col_l:
        fig = go.Figure()
        add_phase_bands(fig, step_cursor)
        fig.add_trace(go.Scatter(
            x=cur_check["step"],
            y=cur_check["dpd_if_rate"] * 100,
            mode="lines+markers", name="IF anomaly rate (%)",
            line=dict(color="#e67e22", width=2),
            marker=dict(size=6,
                        color=_marker_colours(cur_check["dpd_triggered"]).tolist()),
        ))
        _threshold_line(fig, 8, "threshold 8%")
        fig.update_layout(title="Isolation Forest anomaly rate", height=230,
                          margin=dict(l=0,r=0,t=30,b=0),
                          yaxis=dict(title="Anomaly rate (%)", gridcolor="#f0f0f0", rangemode="tozero"),
                          xaxis=dict(title="Step"),
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig2 = go.Figure()
        add_phase_bands(fig2, step_cursor)
        fig2.add_trace(go.Scatter(
            x=cur_check["step"],
            y=cur_check["dpd_mahal_max"],
            mode="lines+markers", name="Max Mahalanobis dist.",
            line=dict(color="#c0392b", width=2),
            marker=dict(size=6),
        ))
        _threshold_line(fig2, 5.0, "threshold 5.0σ")
        fig2.update_layout(title="Max Mahalanobis distance (σ)", height=230,
                           margin=dict(l=0,r=0,t=30,b=0),
                           yaxis=dict(title="Distance (σ)", gridcolor="#f0f0f0", rangemode="tozero"),
                           xaxis=dict(title="Step"),
                           plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

    n_dpd = int(cur_check["dpd_triggered"].sum()) if "dpd_triggered" in cur_check.columns else 0
    st.info(f"**DPD summary:** {n_dpd} triggers in {len(cur_check)} checks "
            f"({100*n_dpd/max(len(cur_check),1):.1f}% trigger rate). "
            "Phase 3 outliers x∈[30,50] should produce Mahalanobis distances ≫5.")

# ── CDD ───────────────────────────────────────────────────────────────────────
with tab_cdd:
    col_l, col_r = st.columns(2)

    with col_l:
        fig = go.Figure()
        add_phase_bands(fig, step_cursor)
        fig.add_trace(go.Scatter(
            x=cur_check["step"],
            y=cur_check["cdd_ph_stat"],
            mode="lines+markers", name="PH statistic",
            line=dict(color="#27ae60", width=2),
            marker=dict(size=6,
                        color=_marker_colours(cur_check["cdd_triggered"]).tolist()),
        ))
        _threshold_line(fig, 40.0, "λ=40")
        fig.update_layout(title="Page-Hinkley statistic", height=230,
                          margin=dict(l=0,r=0,t=30,b=0),
                          yaxis=dict(title="PH stat.", gridcolor="#f0f0f0", rangemode="tozero"),
                          xaxis=dict(title="Step"),
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig2 = go.Figure()
        add_phase_bands(fig2, step_cursor)
        if cur_check["cdd_perf_drop"].notna().any():
            fig2.add_trace(go.Scatter(
                x=cur_check["step"],
                y=cur_check["cdd_perf_drop"] * 100,
                mode="lines+markers", name="Perf. drop (pp)",
                line=dict(color="#16a085", width=2),
                marker=dict(size=6),
            ))
            _threshold_line(fig2, 12, "threshold 12pp")
        fig2.update_layout(title="Accuracy drop vs. reference window (pp)", height=230,
                           margin=dict(l=0,r=0,t=30,b=0),
                           yaxis=dict(title="Drop (pp)", gridcolor="#f0f0f0"),
                           xaxis=dict(title="Step"),
                           plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

    n_cdd = int(cur_check["cdd_triggered"].sum()) if "cdd_triggered" in cur_check.columns else 0
    st.info(f"**CDD summary:** {n_cdd} triggers in {len(cur_check)} checks. "
            "PH resets to 0 after each model update (vertical drops).")

# ── CPD ───────────────────────────────────────────────────────────────────────
with tab_cpd:
    col_l, col_r = st.columns(2)

    with col_l:
        fig = go.Figure()
        add_phase_bands(fig, step_cursor)
        if cur_check["cpd_shadow_div"].notna().any():
            fig.add_trace(go.Scatter(
                x=cur_check["step"],
                y=cur_check["cpd_shadow_div"] * 100,
                mode="lines+markers", name="Shadow divergence (%)",
                line=dict(color="#c0392b", width=2),
                marker=dict(size=6,
                            color=_marker_colours(cur_check["cpd_triggered"]).tolist()),
            ))
            _threshold_line(fig, 38, "threshold 38%")
        fig.update_layout(title="Shadow model divergence (%)", height=230,
                          margin=dict(l=0,r=0,t=30,b=0),
                          yaxis=dict(title="Divergence (%)", gridcolor="#f0f0f0", rangemode="tozero"),
                          xaxis=dict(title="Step"),
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Detector trigger heatmap
        detectors = ["DDD", "DPD", "CDD", "CPD"]
        det_cols  = ["ddd_triggered", "dpd_triggered", "cdd_triggered", "cpd_triggered"]
        heat_df   = cur_check[["step"] + det_cols].dropna()
        if len(heat_df):
            z = heat_df[det_cols].astype(float).T.values
            fig_h = go.Figure(data=go.Heatmap(
                z=z, x=heat_df["step"].tolist(), y=detectors,
                colorscale=[[0,"#eafaf1"],[1,"#e74c3c"]],
                showscale=False,
                hovertemplate="Step %{x} · %{y}: %{z}<extra></extra>",
            ))
            fig_h.update_layout(
                title="Detector trigger heatmap", height=230,
                margin=dict(l=0,r=0,t=30,b=0),
                xaxis=dict(title="Step"),
                yaxis=dict(title=""),
                plot_bgcolor="white", paper_bgcolor="white",
            )
            st.plotly_chart(fig_h, use_container_width=True)

    n_cpd = int(cur_check["cpd_triggered"].sum()) if "cpd_triggered" in cur_check.columns else 0
    st.info(f"**CPD summary:** {n_cpd} triggers in {len(cur_check)} checks. "
            "CPD is the most sensitive detector — all 3 sub-checks fire during Phase 2.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 3 — Event log & ATM table
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">Event Log & ATM Cycles</p>',
            unsafe_allow_html=True)

col_ev, col_atm = st.columns(2)

with col_ev:
    st.subheader("📋 Event Log")
    if not cur_events.empty:
        disp = cur_events.copy()
        disp["reasons"] = disp["reasons"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else str(x))
        sev_icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "INFO": "🟢"}
        disp["severity"] = disp["severity"].apply(
            lambda s: f"{sev_icon.get(s,'⚪')} {s}")
        st.dataframe(disp[["step","type","severity","reasons"]].tail(25),
                     use_container_width=True, height=320)
    else:
        st.info("No events yet.")

with col_atm:
    st.subheader("🤖 ATM Training Cycles")
    if not cur_atm.empty:
        disp2 = cur_atm.copy()
        disp2["deployed"]   = disp2["deployed"].map({True:"✅ Yes", False:"❌ No", None:"—"})
        disp2["ndt_passed"] = disp2["ndt_passed"].map({True:"✅ Pass", False:"❌ Fail", None:"—"})
        st_icon = {"SUCCESS":"✅","FAILED":"❌","NDT_REJECTED":"⛔","SKIPPED":"⏭"}
        disp2["status"] = disp2["status"].apply(
            lambda s: f"{st_icon.get(s,'❓')} {s}")
        st.dataframe(disp2[["step","status","variant","ndt_passed","deployed"]],
                     use_container_width=True, height=320)
    else:
        st.info("No ATM cycles yet.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 4 — NDT Validation
# ══════════════════════════════════════════════════════════════════════════════
if data.ndt_history:
    st.markdown('<p class="section-title">NDT Validation History</p>',
                unsafe_allow_html=True)
    ndt_df = pd.DataFrame(data.ndt_history)
    ndt_cycle = list(range(1, len(ndt_df) + 1))

    fig_ndt = go.Figure()
    colours = ["#27ae60" if p else "#e74c3c" for p in ndt_df.get("passed", [True]*len(ndt_df))]
    if "candidate_score" in ndt_df.columns:
        fig_ndt.add_trace(go.Bar(
            x=ndt_cycle, y=ndt_df["candidate_score"] * 100,
            name="Candidate accuracy", marker_color=colours,
        ))
    if "baseline_score" in ndt_df.columns:
        fig_ndt.add_trace(go.Bar(
            x=ndt_cycle, y=ndt_df["baseline_score"] * 100,
            name="Baseline accuracy", marker_color="rgba(41,128,185,0.50)",
        ))
    fig_ndt.add_hline(y=65, line_dash="dash", line_color="#e74c3c",
                      annotation_text="NDT floor 65%",
                      annotation_position="bottom right")
    fig_ndt.update_layout(
        title="NDT Validation — Candidate vs. Baseline", height=280,
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis=dict(title="Accuracy (%)", range=[0,105], gridcolor="#f0f0f0"),
        xaxis=dict(title="Validation cycle"),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_ndt, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# Section 5 — Event Distribution Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
if not cur_events.empty:
    st.markdown('<p class="section-title">Event Distribution by Phase</p>',
                unsafe_allow_html=True)

    # Assign phase to each event
    def _phase(step):
        for ph, (s, e) in PHASE_RANGES.items():
            if s <= step <= e:
                return PHASE_NAMES[ph]
        return "Unknown"

    ev_phase = cur_events.copy()
    ev_phase["phase_label"] = ev_phase["step"].apply(_phase)

    pivot = (ev_phase.groupby(["phase_label","type"])
             .size().reset_index(name="count"))

    fig_bar = go.Figure()
    for etype in pivot["type"].unique():
        sub = pivot[pivot["type"] == etype]
        fig_bar.add_trace(go.Bar(
            x=sub["phase_label"], y=sub["count"],
            name=etype,
        ))
    fig_bar.update_layout(
        title="Event counts by phase and type", height=280, barmode="group",
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis=dict(title="Count", gridcolor="#f0f0f0"),
        xaxis=dict(title="Phase"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.divider()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "🎓 **Thesis:** Implementation-Aware AI Management Framework for AI-Native Networks &nbsp;|&nbsp; "
    "🔬 **Stack:** RTP · DDD · DPD · CDD · CPD · ATM · MTP-L · NDT &nbsp;|&nbsp; "
    "📡 **Context:** 6G AI-Native Network Management",
    unsafe_allow_html=True,
)
