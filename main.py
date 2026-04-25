"""
main.py — End-to-end simulation: RTP observer + ATM controller + MLflow

Phases
------
  1. Stable operation     (steps   1-400)  — no alerts expected
  2. Concept drift        (steps 401-600)  — CDD fires, ATM retrains via MTP-L
  3. Data poisoning       (steps 601-700)  — DPD fires, ATM retrains via MTP-E
                                             (MLflow stub used when offline)
  4. Recovery             (steps 701-900)  — system stabilises post-update

MLflow mode
-----------
Set MLFLOW_TRACKING_URI to your server for full tracking, e.g.:
    export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

If MLflow is not reachable, the simulation falls back to MTP-L automatically
and prints a clear notice — the rest of the pipeline still runs.

Run:
    python main.py
"""

import logging
import sys
import os

sys.path.insert(0, '.')

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from aif.aif    import AIF
from rtp.rtp    import RTP, RTPConfig, MToUTSignal, RTPEvent
from atm.atm    import ATM, ATMPolicy, MTPVariant
from atm.mtp_l  import MTPLocal
from atm.mtp_e  import MTPExternal
from ndt.ndt    import NDT

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)-8s %(name)-20s — %(message)s",
)
log = logging.getLogger("SIMULATION")
log.setLevel(logging.INFO)

rng = np.random.default_rng(0)

# ── Detect MLflow availability ────────────────────────────────────────────────
try:
    import mlflow
    MLFLOW_AVAILABLE = True
    log.info("MLflow %s detected.", mlflow.__version__)
except ImportError:
    MLFLOW_AVAILABLE = False
    log.warning(
        "MLflow not installed — MTP-E will be skipped, ATM will use MTP-L. "
        "Install with: pip install mlflow"
    )

# ── Data helpers ──────────────────────────────────────────────────────────────

def make_data(n, noise=0.05, shift=0.0):
    X = rng.normal(0, 1, size=(n, 4))
    y = ((X[:, 0] + X[:, 1]) > shift).astype(int)
    flip = rng.random(n) < noise
    y[flip] = 1 - y[flip]
    return X, y

# ── Build AIF ─────────────────────────────────────────────────────────────────
log.info("━" * 62)
log.info("Building AIF…")
X_train, y_train = make_data(500, noise=0.05)
clf = RandomForestClassifier(n_estimators=50, random_state=1).fit(X_train, y_train)
aif = AIF(clf)

# ── Build RTP ─────────────────────────────────────────────────────────────────
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

received_signals: list[MToUTSignal] = []
security_alerts:  list[RTPEvent]    = []

def on_mtout(signal):
    received_signals.append(signal)

def on_security(event):
    security_alerts.append(event)

rtp = RTP(aif, config=cfg, on_mtout=on_mtout, on_security_alert=on_security)

X_ref, y_ref = make_data(300, noise=0.05)
lob_ref = clf.predict(X_ref)
rtp.set_reference(X_ref, y_ref, lob_ref)

# ── Build controller components ───────────────────────────────────────────────
mtp_local = MTPLocal(n_splits=3, fine_tune_first=True)

_MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

mtp_ext = MTPExternal(
    experiment_name="rtp_aif_retraining",
    model_name="aif_classifier",
    mlflow_uri=_MLFLOW_URI,
    tune_hyperparams=False,
    tags={"thesis": "6G-AINN", "component": "MTP-E"},
) if MLFLOW_AVAILABLE else None

ndt = NDT(
    current_model_getter=lambda: rtp.aif.active_estimator,
    min_score=0.65,
    min_improvement=-0.05,   # tolerate small regressions after drift
    mlflow_uri=_MLFLOW_URI,
)

policy = ATMPolicy(
    # Force MTP-L when MLflow is unavailable, otherwise let ATM decide
    prefer_variant=MTPVariant.LOCAL if not MLFLOW_AVAILABLE else None,
    local_max_samples=600,
    use_ndt=True,
    ndt_min_accuracy=0.65,
    auto_deploy=True,
    max_retrain_attempts=2,
)

atm_results: list = []
def on_atm_result(result):
    atm_results.append(result)
    log.info(
        "  ► ATM cycle done  %-8s  variant=%-6s  ndt=%s  deployed=%s",
        result.status.name,
        result.variant_used.value if result.variant_used else "none",
        result.ndt_passed,
        result.deployed,
    )

atm = ATM(
    rtp=rtp,
    mtp_l=mtp_local,
    mtp_e=mtp_ext,
    ndt=ndt,
    policy=policy,
    on_result=on_atm_result,
)

# Wire ATM into RTP's MToUT callback
rtp._on_mtout = lambda sig: (on_mtout(sig), atm.handle(sig))

log.info("System ready.  RTP=%s  ATM=%s", rtp, atm)
log.info("━" * 62)

# =============================================================================
# Phase 1 — Stable operation
# =============================================================================
log.info("")
log.info("PHASE 1 — Stable operation (steps 1-400)")
X1, y1 = make_data(400, noise=0.05)
for i in range(400):
    rtp.observe(X1[i], y_true=y1[i])
log.info("  Events: %s", rtp.event_summary() or "none")

# =============================================================================
# Phase 2 — Concept drift
# =============================================================================
log.info("")
log.info("PHASE 2 — Concept drift (steps 401-600)")
X2, y2 = make_data(200, noise=0.55)   # 55% label flip
for i in range(200):
    rtp.observe(X2[i], y_true=y2[i])
log.info("  Events: %s", rtp.event_summary())
log.info("  MToUT signals: %d  |  ATM cycles: %d", len(received_signals), len(atm_results))

# =============================================================================
# Phase 3 — Data poisoning attack
# =============================================================================
log.info("")
log.info("PHASE 3 — Poisoning attack (steps 601-700)")
X_clean, y_clean = make_data(90, noise=0.05)
X_inject = rng.uniform(30, 50, size=(10, 4))   # extreme outliers
y_inject  = rng.integers(0, 2, size=10)
X3 = np.vstack([X_clean, X_inject])
y3 = np.concatenate([y_clean, y_inject])
idx = rng.permutation(len(X3))
X3, y3 = X3[idx], y3[idx]
for i in range(len(X3)):
    rtp.observe(X3[i], y_true=y3[i])
log.info("  Events: %s", rtp.event_summary())
log.info("  Security alerts: %d", len(security_alerts))

# =============================================================================
# Phase 4 — Recovery
# =============================================================================
log.info("")
log.info("PHASE 4 — Recovery (steps 701-900)")
X4, y4 = make_data(200, noise=0.05)
for i in range(200):
    rtp.observe(X4[i], y_true=y4[i])

# =============================================================================
# Final report
# =============================================================================
log.info("")
log.info("━" * 62)
log.info("FINAL STATUS")
log.info("━" * 62)
for k, v in rtp.status().items():
    log.info("  %-26s %s", k, v)

log.info("")
log.info("Event log:")
for etype, count in rtp.event_summary().items():
    log.info("  %-32s %d", etype, count)

log.info("")
log.info("ATM training cycles: %d", len(atm_results))
for r in atm_results:
    log.info(
        "  variant=%-6s  status=%-8s  ndt=%-5s  deployed=%s  mlflow_run=%s",
        r.variant_used.value if r.variant_used else "none",
        r.status.name,
        str(r.ndt_passed),
        r.deployed,
        r.run_id or "n/a",
    )

log.info("")
log.info("NDT validation history: %d checks", len(ndt.history))
for rec in ndt.history:
    log.info(
        "  candidate=%.4f  baseline=%.4f  improvement=%+.4f  passed=%s",
        rec["candidate_score"], rec["baseline_score"],
        rec["improvement"], rec["passed"],
    )

log.info("━" * 62)

# ── Assertions ────────────────────────────────────────────────────────────────
assert len(received_signals) > 0,  "Expected MToUT to fire at least once"
assert any(r.deployed for r in atm_results), "Expected at least one deployment"
assert len(ndt.history) > 0, "Expected NDT to run at least once"
print("\nAll assertions passed.")
print(f"MLflow mode: {'ENABLED' if MLFLOW_AVAILABLE else 'OFFLINE (MTP-L fallback)'}")
