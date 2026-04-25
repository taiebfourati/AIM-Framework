"""
scripts/bias_floor_sweep.py — CRIT-6 bias-floor analysis.

Quantifies the failure regime of the dual-score NDT gate documented in
Section~\\ref{sec:disc:rqs} (RQ3): a clean-label, distribution-
preserving adversary that biases the candidate's training stream
without inserting a single mislabelled sample and without shifting
the marginal feature distribution.

For every (attacker_fraction, theta_ndt) combination the harness:

  1. Builds a clean baseline (the operator's currently-deployed model)
     by training a RandomForestClassifier on n_clean clean samples
     drawn from ``baselines.clean_label_attack.make_clean_data``.
  2. Builds a poisoned candidate buffer with
     ``baselines.clean_label_attack.make_clean_label_poisoned_buffer``
     at the requested attacker_fraction.  The buffer is the LOB the
     candidate is trained against and the LOB the NDT gate sees as
     ``y_val`` at validation time --- exactly the operational setting
     the operator faces in deployment.
  3. Trains a candidate RandomForestClassifier on the poisoned buffer.
  4. Scores both models on:
        * the LOB pseudo-labels (this is what the NDT gate evaluates).
        * a held-out clean ``golden corpus`` (n=5000) drawn from the
          same generator with a fresh RNG.  This is the operator's
          honest-but-unobservable performance metric.
  5. Evaluates the production dual-score gate:
        gate_pass = (cand_lob_score >= theta_ndt)
                  AND (cand_lob_score - base_lob_score >= min_improvement)
     for every theta_ndt in the sweep.

The result is the joint distribution of (gate_pass, golden-corpus
accuracy) across attacker_fraction and theta_ndt, averaged over
``n_seeds`` independent draws.  Section~\\ref{sec:eval:bias_floor}
reads this JSON and renders it as fig_08.

Output:
    thesis/.repro_audit_data/bias_floor.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from baselines.clean_label_attack import (  # noqa: E402
    make_clean_data,
    make_clean_label_poisoned_buffer,
)

OUT_DIR = ROOT / "thesis" / ".repro_audit_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "bias_floor.json"


# =============================================================================
# Sweep parameters (chosen to match Campaign A's NDT configuration)
# =============================================================================

# Attacker strengths to probe.  0.0 is the clean control point; 0.50 is
# the largest fraction at which the attacker still controls a minority
# of the buffer (so individual labels remain plausible) but has tilted
# half the training mass toward the poisoned half-space.
ATTACKER_FRACTIONS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

# NDT gate thresholds to sweep.  0.65 is Campaign A's production value;
# we explore both relaxations (0.50) and tightenings (up to 0.95) so
# the reader can see how the bias floor moves as the operator hardens
# or relaxes the gate.
THETA_NDT_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

# Production tolerance for small candidate regressions on the LOB.
MIN_IMPROVEMENT = -0.05

# Number of independent seeds per (attacker_fraction) cell.
N_SEEDS = 5

# Buffer / corpus sizes.
N_CLEAN_BUFFER = 600           # candidate + baseline training-buffer size
N_GOLDEN = 5000                # held-out clean test corpus
NOISE = 0.05                   # same i.i.d. label noise the production sim uses
BIAS_BAND = 0.6                # bias band of the clean-label adversary
BIAS_TARGET = 0                # attacker prefers class 0


# =============================================================================
# Per-seed inner loop
# =============================================================================

def _accuracy(model, X, y) -> float:
    return float(np.mean(model.predict(X) == y))


def _one_seed_one_fraction(seed: int, attacker_fraction: float) -> dict:
    """Train baseline + poisoned candidate at the given attacker_fraction
    and return the four scores (cand_lob, cand_gold, base_lob, base_gold).

    The baseline is fit on a fresh clean buffer of the same size as the
    candidate's clean component, mirroring the operator's situation:
    the deployed model was trained on clean data of comparable volume,
    and the operator now sees a candidate trained on what looks like
    the same distribution.
    """
    rng = np.random.default_rng(seed)

    # Baseline: clean buffer of the same size as candidate's CLEAN
    # component.  The baseline knows nothing about the poison.
    X_base, y_base = make_clean_data(rng, N_CLEAN_BUFFER, noise=NOISE)
    base = RandomForestClassifier(n_estimators=50, random_state=seed).fit(
        X_base, y_base
    )

    # Candidate buffer (poisoned via clean-label adversary).
    X_cand_buf, y_cand_buf = make_clean_label_poisoned_buffer(
        rng,
        n_clean=N_CLEAN_BUFFER,
        attacker_fraction=attacker_fraction,
        bias_band=BIAS_BAND,
        bias_target=BIAS_TARGET,
        noise=NOISE,
    )
    cand = RandomForestClassifier(n_estimators=50, random_state=seed + 1).fit(
        X_cand_buf, y_cand_buf
    )

    # Golden corpus: large held-out clean sample, fresh RNG so the
    # baseline cannot have memorised it.
    gold_rng = np.random.default_rng(seed + 10_000)
    X_gold, y_gold = make_clean_data(gold_rng, N_GOLDEN, noise=NOISE)

    # Scores.  The NDT gate evaluates ``cand`` and ``base`` against the
    # candidate's LOB --- this is the self-referential evaluation
    # documented in Section~\\ref{sec:disc:rqs}.
    cand_lob = _accuracy(cand, X_cand_buf, y_cand_buf)
    base_lob = _accuracy(base, X_cand_buf, y_cand_buf)
    cand_gold = _accuracy(cand, X_gold, y_gold)
    base_gold = _accuracy(base, X_gold, y_gold)

    return {
        "seed": seed,
        "attacker_fraction": attacker_fraction,
        "n_cand_buffer": int(len(X_cand_buf)),
        "cand_lob": cand_lob,
        "base_lob": base_lob,
        "cand_gold": cand_gold,
        "base_gold": base_gold,
    }


# =============================================================================
# Sweep
# =============================================================================

def main() -> int:
    print("=" * 70)
    print("CRIT-6 bias-floor sweep")
    print("=" * 70)
    print(
        f"  attacker_fractions: {ATTACKER_FRACTIONS}\n"
        f"  theta_ndt grid    : {THETA_NDT_GRID}\n"
        f"  min_improvement   : {MIN_IMPROVEMENT}\n"
        f"  n_seeds per cell  : {N_SEEDS}\n"
        f"  n_clean buffer    : {N_CLEAN_BUFFER}\n"
        f"  n_golden corpus   : {N_GOLDEN}\n"
        f"  bias_band         : {BIAS_BAND}\n"
        f"  bias_target       : {BIAS_TARGET}\n"
        f"  candidate model   : RandomForestClassifier(n_estimators=50)\n"
    )

    # ── 1. Per-seed scores -------------------------------------------------
    print(f"[1/2] Training {len(ATTACKER_FRACTIONS) * N_SEEDS} "
          "(attacker_fraction, seed) cells ...")
    raw: list[dict] = []
    for af in ATTACKER_FRACTIONS:
        for s in range(N_SEEDS):
            row = _one_seed_one_fraction(s, af)
            raw.append(row)
        # Per-fraction summary
        cells = [r for r in raw if r["attacker_fraction"] == af]
        m_cl = float(np.mean([r["cand_lob"] for r in cells]))
        m_cg = float(np.mean([r["cand_gold"] for r in cells]))
        m_bl = float(np.mean([r["base_lob"] for r in cells]))
        m_bg = float(np.mean([r["base_gold"] for r in cells]))
        print(
            f"    af={af:.2f}  cand_lob={m_cl:.3f}  cand_gold={m_cg:.3f}"
            f"   base_lob={m_bl:.3f}  base_gold={m_bg:.3f}"
        )

    # ── 2. Aggregate scores per attacker_fraction --------------------------
    fraction_summary = []
    for af in ATTACKER_FRACTIONS:
        cells = [r for r in raw if r["attacker_fraction"] == af]
        cl = np.array([r["cand_lob"] for r in cells])
        cg = np.array([r["cand_gold"] for r in cells])
        bl = np.array([r["base_lob"] for r in cells])
        bg = np.array([r["base_gold"] for r in cells])
        fraction_summary.append({
            "attacker_fraction": af,
            "cand_lob_mean":   float(cl.mean()), "cand_lob_std":   float(cl.std(ddof=0)),
            "cand_gold_mean":  float(cg.mean()), "cand_gold_std":  float(cg.std(ddof=0)),
            "base_lob_mean":   float(bl.mean()), "base_lob_std":   float(bl.std(ddof=0)),
            "base_gold_mean":  float(bg.mean()), "base_gold_std":  float(bg.std(ddof=0)),
        })

    # ── 3. Gate-pass matrix:  rows = attacker_fraction, cols = theta_ndt --
    # gate_pass_rate[i][j] = fraction of seeds at which the gate accepts
    # the poisoned candidate for (af_i, theta_j).
    print(
        f"\n[2/2] Evaluating gate at {len(THETA_NDT_GRID)} "
        f"theta_ndt levels ..."
    )
    gate_pass_rate: list[list[float]] = []
    for af in ATTACKER_FRACTIONS:
        cells = [r for r in raw if r["attacker_fraction"] == af]
        row = []
        for theta in THETA_NDT_GRID:
            n_pass = 0
            for r in cells:
                passes_floor = r["cand_lob"] >= theta
                improvement = r["cand_lob"] - r["base_lob"]
                passes_improvement = improvement >= MIN_IMPROVEMENT
                if passes_floor and passes_improvement:
                    n_pass += 1
            row.append(n_pass / len(cells))
        gate_pass_rate.append(row)

    # ── 4. Bias-floor curve: for each theta_ndt, the largest af at
    # which the gate still admits the poisoned candidate >= 50% of the
    # time.  This is the operational ``bias floor'' the operator
    # cannot push below without rejecting clean candidates as well.
    bias_floor = []
    for j, theta in enumerate(THETA_NDT_GRID):
        af_passable = [
            ATTACKER_FRACTIONS[i]
            for i in range(len(ATTACKER_FRACTIONS))
            if gate_pass_rate[i][j] >= 0.5
        ]
        bias_floor.append({
            "theta_ndt": theta,
            "max_passable_attacker_fraction":
                float(max(af_passable)) if af_passable else 0.0,
        })

    # ── 5. Dump ------------------------------------------------------------
    snapshot = {
        "metadata": {
            "attacker_fractions": ATTACKER_FRACTIONS,
            "theta_ndt_grid": THETA_NDT_GRID,
            "min_improvement": MIN_IMPROVEMENT,
            "n_seeds": N_SEEDS,
            "n_clean_buffer": N_CLEAN_BUFFER,
            "n_golden": N_GOLDEN,
            "bias_band": BIAS_BAND,
            "bias_target": BIAS_TARGET,
            "noise": NOISE,
            "candidate_model": "RandomForestClassifier(n_estimators=50)",
        },
        "raw": raw,
        "fraction_summary": fraction_summary,
        "gate_pass_rate": gate_pass_rate,
        "bias_floor": bias_floor,
    }

    OUT_PATH.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(
        f"\nWrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.1f} KB)"
    )
    print("\nBias floor (max passable attacker_fraction):")
    for bf in bias_floor:
        print(
            f"    theta_ndt={bf['theta_ndt']:.2f}  ->  "
            f"max passable attacker_fraction = "
            f"{bf['max_passable_attacker_fraction']:.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
