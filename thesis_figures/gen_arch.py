"""
High-level architecture — paper Fig.1 style.
Strictly vertical/horizontal arrows, no crossings.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.set_aspect("equal")
ax.axis("off")

C_MP  = "#D5D5D5"
C_AIF = "#C8D8E8"
C_RTP = "#B0C8E0"
C_ATM = "#C8E0C8"
C_NDT = "#D8E8D0"
C_ETP = "#F5E0C8"
C_CTP = "#E8D8E8"
BD    = "#555555"

def box(x, y, w, h, fc, ec=BD, lw=1.2, zo=2):
    ax.add_patch(FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.08",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zo))

def txt(x, y, s, fs=9, bold=False, ha="center", va="center", c="#222", zo=3):
    ax.text(x, y, s, fontsize=fs, fontweight="bold" if bold else "normal",
            ha=ha, va=va, color=c, linespacing=1.3, zorder=zo, family="sans-serif")

def arrow(x1, y1, x2, y2, lw=1.2, c="#444", style="-|>", ls="-", zo=4):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle=style, color=c, lw=lw,
                                linestyle=ls, shrinkA=2, shrinkB=2), zorder=zo)

def lbl(x, y, s, fs=7.5, c="#444", zo=5, ha="center"):
    ax.text(x, y, s, fontsize=fs, ha=ha, va="center", color=c,
            zorder=zo, family="sans-serif", style="italic")

# ═══════ ROW 1 — MP ═══════
box(2.0, 7.0, 6.0, 0.65, fc=C_MP)
txt(5.0, 7.32, "Network Management & Orchestration Plane (MP)", fs=10, bold=True)

# ═══════ ROW 2 — AIF + RTP ═══════
box(0.5, 5.1, 4.0, 1.5, fc=C_AIF)
txt(2.5, 6.05, "AI Function (AIF)", fs=11, bold=True)
txt(2.5, 5.65, "DPP  \u2192  SIB  \u2192  MLI", fs=9)
txt(2.5, 5.32, "MLIN (active) | MLIO (standby)", fs=7.5, c="#555")

box(5.5, 5.1, 4.0, 1.5, fc=C_RTP)
txt(7.5, 6.05, "Runtime Pipeline (RTP)", fs=11, bold=True)
txt(7.5, 5.65, "DDD | DPD | CDD | CPD", fs=8.5)
txt(7.5, 5.32, "MToUT  \u2022  LIB  \u2022  LOB", fs=8.5, c="#555")

# ═══════ ROW 3 — NDT (left) + ATM (right) ═══════
box(0.5, 3.1, 4.0, 1.3, fc=C_NDT)
txt(2.5, 3.95, "Network Digital Twin (NDT)", fs=10, bold=True)
txt(2.5, 3.55, "AI-QoS floor check", fs=8, c="#555")
txt(2.5, 3.28, "Improvement-over-baseline gate", fs=8, c="#555")

box(5.5, 3.1, 4.0, 1.3, fc=C_ATM)
txt(7.5, 3.95, "AIF Training Manager (ATM)", fs=10, bold=True)
txt(7.5, 3.55, "Variant selection \u2022 ATMPolicy", fs=8, c="#555")
txt(7.5, 3.28, "Retry logic \u2022 Rollback control", fs=8, c="#555")

# ═══════ ROW 4 — MTP-L, MTP-E, MTP-C, ETP ═══════
box(0.3, 0.7, 2.0, 1.4, fc=C_ATM)
txt(1.3, 1.6, "MTP-L", fs=10, bold=True)
txt(1.3, 1.25, "Local Training", fs=8)
txt(1.3, 0.95, "Fine-tune / Retrain", fs=7, c="#555")

box(2.7, 0.7, 2.3, 1.4, fc=C_ETP)
txt(3.85, 1.6, "MTP-E", fs=10, bold=True)
txt(3.85, 1.25, "External (MLflow)", fs=8)
txt(3.85, 0.95, "Experiment tracking", fs=7, c="#555")

box(5.4, 0.7, 2.0, 1.4, fc=C_CTP, ec="#999")
txt(6.4, 1.6, "MTP-C", fs=10, bold=True, c="#666")
txt(6.4, 1.25, "Centralised (CTP)", fs=8, c="#666")
txt(6.4, 0.95, "(stub)", fs=7, c="#888")

box(7.8, 0.7, 2.0, 1.4, fc=C_ETP)
txt(8.8, 1.6, "External Training", fs=9, bold=True)
txt(8.8, 1.25, "Platform (ETP)", fs=9, bold=True)
txt(8.8, 0.95, "MLflow backend", fs=7, c="#555")

# ═══════ ARROWS — vertical/horizontal only ═══════

# ① MP → AIF (down)
arrow(2.5, 7.0, 2.5, 6.6, c="#555")

# ② MP → RTP (down)
arrow(7.5, 7.0, 7.5, 6.6, c="#555")

# ③ AIF ↔ RTP (horizontal bidirectional)
arrow(4.5, 5.85, 5.5, 5.85, c="#333", style="<->")
lbl(5.0, 6.03, "observe / infer", c="#333")

# ④ RTP → ATM (straight down — MToUT)
arrow(7.5, 5.1, 7.5, 4.4, lw=1.4, c="#B8860B")
lbl(8.2, 4.75, "MToUT signal", c="#8B6914")

# ⑤ ATM → NDT (horizontal left — validate)
arrow(5.5, 3.75, 4.5, 3.75, lw=1.2, c="#2E7D32")
lbl(5.0, 3.95, "validate", c="#2E7D32")

# ⑥ NDT → AIF (straight up — deploy)
arrow(2.5, 4.4, 2.5, 5.1, lw=1.3, c="#2E7D32")
lbl(1.5, 4.75, "deploy\n(if pass)", c="#2E7D32")

# ⑦ ATM → MTP variants (down, fan-out via L-shapes)
# ATM bottom center → down to y=2.5 → then fan left to each MTP
# Main trunk: x=7.5 down to y=2.5
ax.plot([7.5, 7.5], [3.1, 2.55], color="#555", lw=1.0, zorder=3)
# Horizontal bar at y=2.55
ax.plot([1.3, 6.4], [2.55, 2.55], color="#555", lw=1.0, zorder=3)
# Down into each MTP
arrow(1.3, 2.55, 1.3, 2.1, c="#555", lw=1.0)
arrow(3.85, 2.55, 3.85, 2.1, c="#555", lw=1.0)
arrow(6.4, 2.55, 6.4, 2.1, c="#555", lw=0.8, ls="dashed")
lbl(4.6, 2.75, "select variant", c="#555")

# ⑧ MTP-E → ETP (routed below MTP blocks to avoid clipping)
# MTP-E bottom → down to y=0.4 → right below all blocks → up into ETP
ax.plot([3.85, 3.85], [0.7, 0.4], color="#B8630A", lw=1.0, zorder=4)
ax.plot([3.85, 8.8], [0.4, 0.4], color="#B8630A", lw=1.0, zorder=4)
arrow(8.8, 0.4, 8.8, 0.7, lw=1.0, c="#B8630A")
lbl(6.4, 0.2, "log + register", c="#B8630A")

# ⑨ LIB/LOB train data: RTP → MTP (routed along right margin)
# Down from RTP right edge, along right margin, then left into MTP row
ax.plot([9.7, 9.7], [5.5, 2.55], color="#666", lw=0.9, ls="--", zorder=3)
ax.plot([9.7, 7.5], [2.55, 2.55], color="#666", lw=0.9, ls="--", zorder=3)
# (merges into the ⑦ fan-out bar — data flows to MTP through ATM)
lbl(9.9, 4.0, "LIB/LOB\ntrain data", c="#666", ha="left", fs=7)

# ═══════ Section refs ═══════
txt(0.2, 5.85, "\u00a7IV-B", fs=7, c="#999", ha="center")
txt(0.2, 3.75, "\u00a7IV-D", fs=7, c="#999", ha="center")
txt(0.2, 1.4,  "\u00a7IV-D", fs=7, c="#999", ha="center")

# ── Save ──
fig.tight_layout(pad=0.3)
out = r"C:\Users\taieb\PycharmProjects\rtp_observer\thesis\figures\01_architecture.png"
fig.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
plt.close()
