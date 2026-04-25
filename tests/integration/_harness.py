"""
Shared harness for Tier-2 integration tests.

Each test wires a full RTP → ATM → MTP → NDT pipeline end-to-end using
real components with two deliberate substitutions:

* ``FakeMTPExternal`` replaces :class:`atm.mtp_e.MTPExternal` so the tests
  never need an MLflow tracking server.  It still returns a freshly
  trained sklearn estimator and a deterministic ``run_id``/``model_uri``
  pair, so ATM's MTP-E selection branch is exercised honestly.
* ``FakeNDT`` lets a test force a PASS or FAIL outcome without depending
  on MLflow and without exercising the real scorer (useful for rejection
  and retry tests).  The real :class:`ndt.ndt.NDT` is used wherever the
  test *does* want to exercise the validator.

Everything else — RTP detector battery, ATM variant selection, MTP-L
training, AIF model slot swap, RTP.notify_model_updated — runs unmodified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.linear_model import LogisticRegression, Ridge

from aif.aif import AIF
from aif.golden_corpus import GoldenCorpus
from atm.atm import ATM, ATMPolicy, ATMResult, MTPVariant
from atm.mtp_l import MTPLocal
from ndt.ndt import NDT
from rtp.rtp import RTP, RTPConfig, MToUTSignal, TriggerReason


# ---------------------------------------------------------------------------
# Fake MTP-E — honest in-process substitute for MLflow-backed MTPExternal
# ---------------------------------------------------------------------------

class FakeMTPExternal:
    """
    Drop-in for :class:`atm.mtp_e.MTPExternal` that trains a real sklearn
    model in-process and skips MLflow.  Returns the same dict shape ATM
    expects (``model``, ``run_id``, ``model_uri``) so ATM's MTP-E branch
    runs unchanged.
    """

    def __init__(self) -> None:
        self.call_count: int = 0
        self.last_signal: Optional[MToUTSignal] = None
        self.promotions: list[str] = []
        self.marked_failed: list[str] = []

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        base_model: Optional[BaseEstimator] = None,
        signal: Optional[MToUTSignal] = None,
    ) -> dict[str, Any]:
        self.call_count += 1
        self.last_signal = signal

        # Honest retrain — clone the base model (or fall back to a
        # sensible default) and fit on (X, y).
        if base_model is not None:
            candidate = clone(base_model)
        else:
            candidate = LogisticRegression(max_iter=500)
        candidate.fit(X, y)

        run_id = f"fake-run-{self.call_count:03d}"
        model_uri = f"models:/fake-model/{self.call_count}"
        return {"model": candidate, "run_id": run_id, "model_uri": model_uri}

    def promote_to_production(self, run_id: str) -> None:
        self.promotions.append(run_id)

    def mark_failed(self, run_id: str) -> None:
        self.marked_failed.append(run_id)


# ---------------------------------------------------------------------------
# Fake NDT — lets a test pin the validator outcome
# ---------------------------------------------------------------------------

class FakeNDT:
    """Mimics :class:`ndt.ndt.NDT` with a fixed pass/fail verdict."""

    def __init__(self, verdict: bool = True) -> None:
        self.verdict = verdict
        self.history: list[dict] = []
        self.call_count: int = 0

    def validate(
        self,
        candidate: BaseEstimator,
        X_val: np.ndarray,
        y_val: np.ndarray,
        min_score: Optional[float] = None,
        run_id: Optional[str] = None,
        y_val_gt: Optional[np.ndarray] = None,
    ) -> bool:
        self.call_count += 1
        self.history.append({"run_id": run_id, "passed": self.verdict})
        return self.verdict


# ---------------------------------------------------------------------------
# MTP-L spy — wraps the real MTPLocal and records every call
# ---------------------------------------------------------------------------

class SpyMTPLocal(MTPLocal):
    """Real MTP-L that also records each call for test assertions."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.call_count: int = 0
        self.last_X_shape: Optional[tuple[int, int]] = None

    def train(self, X, y, base_model=None):      # type: ignore[override]
        self.call_count += 1
        self.last_X_shape = (len(X), X.shape[1] if X.ndim == 2 else 1)
        return super().train(X, y, base_model=base_model)


# ---------------------------------------------------------------------------
# Corpora — deterministic classifier / regressor datasets
# ---------------------------------------------------------------------------

def make_classifier_corpus(
    n: int,
    d: int,
    seed: int,
    shift: float = 0.0,
    flip_labels: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Linearly separable 2-class corpus.

    * ``shift`` translates X by a constant along every dimension — useful
      for simulating data drift.  The X distribution IS shifted so drift
      detectors still fire, but labels are derived from the zero-centred
      residual ``(X - shift) @ w`` so the class balance stays near 50/50
      regardless of shift magnitude.  Without this correction, large shifts
      (≥ 2σ) would cause the corpus to be heavily skewed or single-class
      because ``X @ w ≈ shift · Σw`` dominates the per-row noise.
    * ``flip_labels`` inverts the labels — useful for simulating label
      poisoning (triggers CPD).
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(shift, 1.0, size=(n, d))
    w = rng.normal(0.0, 1.0, size=(d,))
    y = ((X - shift) @ w > 0.0).astype(int)
    if flip_labels:
        y = 1 - y
    return X, y


def make_regressor_corpus(
    n: int,
    d: int,
    seed: int,
    shift: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(shift, 1.0, size=(n, d))
    w = rng.normal(0.0, 1.0, size=(d,))
    y = X @ w + rng.normal(0.0, 0.1, size=n)
    return X, y


# ---------------------------------------------------------------------------
# Pipeline builder — the single entry point every test uses
# ---------------------------------------------------------------------------

@dataclass
class Pipeline:
    """Bundles every wired component so tests can poke each one."""
    aif: AIF
    rtp: RTP
    atm: ATM
    mtp_l: SpyMTPLocal
    mtp_e: FakeMTPExternal
    ndt: Any
    events: list = field(default_factory=list)
    atm_results: list[ATMResult] = field(default_factory=list)


def build_pipeline(
    *,
    task: str = "classifier",
    n_features: int = 4,
    seed: int = 0,
    config: Optional[RTPConfig] = None,
    policy: Optional[ATMPolicy] = None,
    ndt: Optional[Any] = None,
    golden_corpus: Optional[GoldenCorpus] = None,
) -> Pipeline:
    """
    Wire up a fully operational RTP + ATM pipeline with a fitted AIF.

    Returns
    -------
    Pipeline
        Convenience container so tests can reach each component directly.
    """
    # ── AIF with an already-fitted MLIN ──────────────────────────────
    X_ref, y_ref = (
        make_classifier_corpus(600, n_features, seed)
        if task == "classifier"
        else make_regressor_corpus(600, n_features, seed)
    )
    estimator: BaseEstimator = (
        LogisticRegression(max_iter=500) if task == "classifier"
        else Ridge(alpha=0.5)
    )
    estimator.fit(X_ref, y_ref)
    aif = AIF(estimator=estimator, sib_capacity=1)

    # ── RTP ──────────────────────────────────────────────────────────
    cfg = config or RTPConfig(
        cdd_task=task,
        check_interval=50,
        mtout_cooldown_steps=50,        # let tests fire twice quickly
        buffer_maxlen=2000,
    )
    atm_results: list[ATMResult] = []
    pipeline = Pipeline(
        aif=aif, rtp=None,              # type: ignore[arg-type]
        atm=None,                       # type: ignore[arg-type]
        mtp_l=None,                     # type: ignore[arg-type]
        mtp_e=None,                     # type: ignore[arg-type]
        ndt=None,
        atm_results=atm_results,
    )

    def _on_mtout(sig: MToUTSignal) -> None:
        pipeline.events.append(("mtout", sig))
        atm_result = pipeline.atm.handle(sig)
        atm_results.append(atm_result)

    rtp = RTP(
        aif=aif, config=cfg, on_mtout=_on_mtout,
        golden_corpus=golden_corpus,
    )
    rtp.set_reference(X_ref[:300], y_ref[:300])

    # ── ATM + its pipelines ──────────────────────────────────────────
    mtp_l = SpyMTPLocal(n_splits=0)                # skip CV to stay fast
    mtp_e = FakeMTPExternal()
    ndt_inst = ndt if ndt is not None else NDT(
        current_model_getter=lambda: rtp.aif.active_estimator,
        min_score=0.50,          # permissive floor so honest refits pass
        min_improvement=-1.0,    # never fail on regression in these tests
    )
    atm = ATM(
        rtp=rtp,
        mtp_l=mtp_l,
        mtp_e=mtp_e,
        ndt=ndt_inst,
        policy=policy or ATMPolicy(
            use_ndt=True, auto_deploy=True,
            max_retrain_attempts=1,    # one attempt is plenty for tests
        ),
    )

    # Backfill the pipeline bundle
    pipeline.rtp = rtp
    pipeline.atm = atm
    pipeline.mtp_l = mtp_l
    pipeline.mtp_e = mtp_e
    pipeline.ndt = ndt_inst
    return pipeline


# ---------------------------------------------------------------------------
# Stream helpers
# ---------------------------------------------------------------------------

def stream(
    pipeline: Pipeline,
    X: np.ndarray,
    y_true: Optional[np.ndarray] = None,
) -> list[np.ndarray]:
    """
    Feed every row of ``X`` through ``rtp.observe()`` and return the
    prediction stream.  When ``y_true`` is ``None`` the CDD operates in
    proxy mode.
    """
    preds: list[np.ndarray] = []
    for i, x in enumerate(X):
        yt = None if y_true is None else np.asarray([y_true[i]])
        preds.append(pipeline.rtp.observe(x, y_true=yt))
    return preds
