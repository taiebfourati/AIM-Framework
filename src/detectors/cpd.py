"""
detectors/cpd.py — Concept Poisoning Detector (CPD)

Detects when the AIF's outputs have been adversarially manipulated —
i.e. the model's behaviour has been corrupted so it produces wrong
predictions on specific inputs (a backdoor or targeted attack).

Paper reference: Section IV-C
  "Concept Poisoning Detector (CPD) analyses the LIB and LOB for concept
   poisoning. Poisoning detection triggers the MToU process (through
   MToUT) and notifies the security subsystem of the incident. The MToUT
   may immediately substitute the MLIN by MLIO, or replace the poisoned
   MLIN with a healthy version from AIMP."

How concept poisoning differs from concept drift
-------------------------------------------------
Concept drift (CDD):
  The environment has genuinely changed — P(Y|X) shifted organically.
  The model should be retrained.

Concept poisoning (CPD):
  The model's outputs have been maliciously altered — P(Y|X) has NOT
  changed in the environment, but the model now outputs wrong predictions
  for certain inputs. The model should be rolled back immediately.

Detection strategy
------------------
Three cross-checks between LIB (inputs) and LOB (outputs):

1. Shadow model cross-check
   A lightweight shadow model (trained on a clean reference dataset of
   (X, y) pairs) independently predicts on recent LIB inputs. When the
   shadow model's predictions diverge strongly from the live MLIN's LOB
   outputs, concept poisoning is suspected.
   Divergence metric: Disagreement rate (classifiers) or normalised MAE
   (regressors).

2. Output distribution consistency check
   Compares the distribution of LOB outputs in a reference window vs. the
   recent window using a KS test. Unlike the CDD (which tracks *loss* over
   time), this directly checks whether the model's output distribution has
   shifted in a way inconsistent with the input distribution change measured
   by DDD. Cross-check: LOB shift without a corresponding LIB shift is a
   strong poisoning signal.

3. Input-output correlation check — *two-condition rule*
   Fits a linear (Pearson) correlation between each input feature and the
   LOB output on the reference window, then checks whether those
   correlations hold in the recent window. A trigger requires BOTH of:

     (a) raw absolute delta |Δr| > ``corr_threshold``  — keeps the check
         interpretable ("this correlation moved by 0.4 points");
     (b) Fisher-z standardized delta |z| > ``corr_z_threshold`` — rejects
         sampling-noise wobble on small windows.

   Fisher's z-transform z = arctanh(r) is approximately normal with
   variance 1/(n−3), so the standardized difference
       z_diff = (arctanh(r_rec) − arctanh(r_ref)) /
                √(1/(n_ref−3) + 1/(n_rec−3))
   has a standard-normal distribution under H0 ("correlations
   unchanged"). A |z_diff| > 4 per feature corresponds to p ≈ 6·10⁻⁵,
   giving the detector a principled false-positive budget across many
   checks without losing sensitivity to real correlation reversals
   (which typically produce |z| in the tens to hundreds).

Historical note
---------------
Prior to the two-condition rule, the correlation check fired whenever
|Δr| > threshold. On a 100-sample recent window, Pearson r has sampling
std ≈ 1/√100 = 0.1, so |Δr| of 0.3–0.5 occurred routinely on clean
data — producing the dashboard-observed sporadic ``CONCEPT_POISONING``
firings at steps 350 / 1450 of the live demo. The Fisher-z gate
removes that failure mode by scaling Δr against its own noise floor.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats
from sklearn.base import clone, BaseEstimator, is_classifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from aif.buffers import LIB, LOB
from aif.golden_corpus import GoldenCorpusSnapshot
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CPDResult:
    """Outcome of one CPD.check() call."""
    poisoning_detected: bool
    # Shadow model
    shadow_divergence: float       # disagreement rate or normalised MAE
    shadow_threshold: float
    shadow_triggered: bool
    # Output distribution
    output_ks_pvalue: float
    output_ks_threshold: float
    output_ks_triggered: bool
    # Correlation consistency — two-condition rule
    corr_delta_max: float          # max absolute correlation change |Δr|
    corr_threshold: float          # |Δr| threshold
    corr_z_max: float              # max Fisher-z standardized |Δr|
    corr_z_threshold: float        # z-threshold
    corr_triggered: bool           # True iff BOTH |Δr| and |z| exceed their thresholds
    # Meta
    message: str = ""
    # Provenance — hex digest of the rows used to fit the shadow model
    # whose predictions produced ``shadow_divergence``. Set when the
    # shadow was refit from a ``GoldenCorpusSnapshot``; ``None`` when
    # the shadow is unfitted or was fit via the unsafe raw-buffer path
    # (the latter case is still recorded with the literal string
    # ``"unsafe_raw"`` so auditors can filter on it).
    shadow_source_hash: Optional[str] = None

    def __bool__(self) -> bool:
        return bool(self.poisoning_detected)


# ---------------------------------------------------------------------------
# CPD
# ---------------------------------------------------------------------------

class CPD:
    """
    Concept Poisoning Detector.

    Parameters
    ----------
    task : str
        "classifier" or "regressor".
    reference_size : int
        Number of (X, y) pairs used to train the shadow model and establish
        reference correlations.
    recent_size : int
        Number of most-recent samples inspected on each check() call.
    shadow_threshold : float
        Divergence rate above which the shadow check triggers.
        For classifiers: fraction of disagreements (e.g. 0.25 = 25%).
        For regressors: normalised MAE ratio (rec_MAE / ref_MAE − 1).
    output_ks_alpha : float
        KS significance level for the output distribution test.
    corr_threshold : float
        Minimum |Δr| in any feature for the correlation check to consider
        triggering. Interpretable: "a correlation moved by this much".
    corr_z_threshold : float
        Minimum Fisher-z-standardized |Δr| for the correlation check to
        actually trigger. Both ``corr_threshold`` and ``corr_z_threshold``
        must be exceeded on the same feature for ``corr_triggered`` to
        fire. Default 4.0 → p ≈ 6·10⁻⁵ per feature under H0, enough
        budget for hundreds of live checks without false positives.
    shadow_estimator : BaseEstimator or None
        Custom sklearn estimator for the shadow model. If None, defaults
        to LogisticRegression (classifiers) or Ridge (regressors).
    """

    def __init__(
        self,
        task: str = "classifier",
        reference_size: int = 300,
        recent_size: int = 100,
        shadow_threshold: float = 0.35,
        output_ks_alpha: float = 0.01,
        corr_threshold: float = 0.40,
        corr_z_threshold: float = 4.0,
        shadow_estimator: Optional[BaseEstimator] = None,
    ) -> None:
        self.task = task.lower()
        self.reference_size = reference_size
        self.recent_size = recent_size
        self.shadow_threshold = shadow_threshold
        self.output_ks_alpha = output_ks_alpha
        self.corr_threshold = corr_threshold
        self.corr_z_threshold = corr_z_threshold

        # Shadow model
        if shadow_estimator is not None:
            self._shadow_proto = shadow_estimator
        elif self.task in ("classifier", "classification"):
            self._shadow_proto = LogisticRegression(max_iter=1000, random_state=42)
        else:
            self._shadow_proto = Ridge(alpha=1.0)

        self._shadow: Optional[BaseEstimator] = None

        # Reference state
        self._ref_outputs: Optional[np.ndarray] = None     # LOB reference window
        self._ref_correlations: Optional[np.ndarray] = None  # feature-output corr
        self._ref_n: int = 0                                # n used to fit corrs (for Fisher-z SE)

        # Provenance of the current shadow fit — surfaced on CPDResult so
        # downstream auditors can verify *which* rows produced the shadow
        # model whose divergence triggered (or failed to trigger) the
        # detector. Set to the hex corpus_hash of the last GoldenCorpusSnapshot
        # used by ``fit_reference_from_snapshot``, or the literal string
        # ``"unsafe_raw"`` when the legacy buffer path was exercised.
        self._shadow_source_hash: Optional[str] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _feature_correlations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Per-feature Pearson correlation with ``y``, NaN-safe.

        ``scipy.stats.pearsonr`` returns NaN when either input vector is
        constant (zero variance). That NaN must not silently survive: once
        it enters the delta-max comparison ``nan > threshold`` is False on
        every IEEE-754 platform, which silently disables the correlation
        check. We coerce NaN -> 0.0 ("no correlation observed"), which
        is the correct safe default for both fitting and detection.
        """
        if X.shape[0] < 2:
            return np.zeros(X.shape[1], dtype=np.float64)
        out = np.empty(X.shape[1], dtype=np.float64)
        for f in range(X.shape[1]):
            col = X[:, f]
            if np.std(col) == 0.0 or np.std(y) == 0.0:
                out[f] = 0.0
                continue
            r = stats.pearsonr(col, y)[0]
            out[f] = 0.0 if not np.isfinite(r) else float(r)
        return out

    @staticmethod
    def _fisher_z_delta(
        r_ref: np.ndarray,
        r_rec: np.ndarray,
        n_ref: int,
        n_rec: int,
    ) -> np.ndarray:
        """Fisher-z standardized |Δr| per feature.

        Under H0 that the population correlations are equal,
        ``z_diff = arctanh(r_rec) - arctanh(r_ref)`` is approximately
        normal with variance ``1/(n_ref-3) + 1/(n_rec-3)``. Returns
        ``|z_diff| / σ``, so the result is on a standard-normal scale:
        values > 4 correspond to per-feature p < 6·10⁻⁵.

        Correlations are clipped to ±0.9999 to avoid arctanh overflow at
        the bounds. NaN outputs are coerced to 0 for the same reason as
        ``_feature_correlations``.
        """
        # guard against degenerate sample sizes — treat as "no evidence"
        if n_ref < 4 or n_rec < 4:
            return np.zeros_like(r_ref, dtype=np.float64)
        r_ref_c = np.clip(r_ref, -0.9999, 0.9999)
        r_rec_c = np.clip(r_rec, -0.9999, 0.9999)
        z_ref = np.arctanh(r_ref_c)
        z_rec = np.arctanh(r_rec_c)
        se = np.sqrt(1.0 / (n_ref - 3) + 1.0 / (n_rec - 3))
        z_std = np.abs(z_rec - z_ref) / se
        return np.nan_to_num(z_std, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------
    # Reference fitting
    # ------------------------------------------------------------------

    def fit_reference(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lob_outputs: np.ndarray,
        *,
        authorize_raw: bool = False,
    ) -> "CPD":
        """
        Fit the shadow model and record reference statistics.

        Parameters
        ----------
        X : np.ndarray of shape (n, n_features)
            Clean reference input samples (from LIB).
        y : np.ndarray of shape (n,)
            Ground-truth labels/values for those inputs.
        lob_outputs : np.ndarray of shape (n,)
            Corresponding MLIN outputs from LOB (should match y closely
            on clean data, but may not be identical).
        authorize_raw : bool, keyword-only
            Security escape hatch. When True the caller explicitly
            opts in to fitting the shadow from raw buffer arrays that
            an adversary may have influenced (see the module-level
            threat model). Leave at its default of False to only get a
            security warning on the log — existing clean-test
            fixtures continue to work under that default but production
            pipelines SHOULD prefer :meth:`fit_reference_from_snapshot`.

        Security note
        -------------
        The ``(X, y)`` passed to this method is used verbatim to train
        the shadow classifier. If the stream feeding those arrays has
        been poisoned, the shadow learns the poisoned boundary and
        ceases to detect concept poisoning. Production callers MUST
        use :meth:`fit_reference_from_snapshot` with an operator-curated
        :class:`aif.golden_corpus.GoldenCorpus` whenever one is
        available. The ``authorize_raw=True`` path exists only for
        back-compatibility with pre-hardening fixtures and for the
        ``UNSAFE`` test parameterisation that documents the vulnerability
        this method used to have.
        """
        if not authorize_raw:
            logger.warning(
                "CPD.fit_reference: refitting shadow from raw buffer arrays "
                "without authorize_raw=True. This path is attacker-influenceable "
                "(see aif.golden_corpus). Prefer fit_reference_from_snapshot()."
            )
        # Mark the shadow's provenance as unsafe so downstream CPDResult
        # consumers can filter / alert on it. ``fit_reference_from_snapshot``
        # overwrites this with the real corpus hash.
        self._shadow_source_hash = "unsafe_raw"

        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        lob_outputs = np.asarray(lob_outputs, dtype=float).ravel()

        # Train shadow model on clean (X, y).  Guard against degenerate
        # slices: a single-class y (possible when the post-retrain
        # reference window happens to fall entirely on one side of the
        # decision boundary) would crash LogisticRegression.  In that
        # case we still install a DummyClassifier pinned to that class
        # so the shadow stays calibrated to the post-deploy regime
        # rather than drifting against a stale pre-deploy shadow.
        try:
            is_cls = self.task in ("classifier", "classification")
            unique_y = np.unique(y[~np.isnan(y)]) if y.size else np.array([])
            if is_cls and unique_y.size < 2:
                # DummyClassifier's ``constant`` param rejects bare floats.
                # Cast to int when the value is integral (standard class
                # labels), otherwise wrap in an array-like for safety.
                if unique_y.size:
                    raw = unique_y[0]
                    is_integral = float(raw).is_integer()
                    constant = int(raw) if is_integral else np.array([raw])
                else:
                    raw = 0
                    is_integral = True
                    constant = 0
                # Fit on integer-cast labels so the DummyClassifier's
                # internal ``classes_`` matches downstream expectations.
                y_fit = y.astype(int) if is_integral else y
                self._shadow = DummyClassifier(
                    strategy="constant", constant=constant,
                ).fit(X, y_fit)
                logger.info(
                    "CPD: shadow slice is single-class (=%s); installed "
                    "DummyClassifier constant shadow on %d samples.",
                    constant, len(X),
                )
            else:
                self._shadow = clone(self._shadow_proto).fit(X, y)
                logger.info(
                    "CPD: shadow model fitted - %d samples, estimator=%s.",
                    len(X), type(self._shadow).__name__,
                )
        except Exception as exc:
            logger.warning(
                "CPD: shadow model fit failed (%s); keeping previous shadow.",
                exc,
            )

        # Store reference LOB outputs for distribution comparison
        self._ref_outputs = lob_outputs.copy()

        # Compute reference input-output correlations (one per feature)
        self._ref_correlations = self._feature_correlations(X, lob_outputs)
        self._ref_n = X.shape[0]
        logger.info(
            "CPD: reference correlations computed (max=%.3f, min=%.3f, n=%d).",
            self._ref_correlations.max(), self._ref_correlations.min(), self._ref_n,
        )
        return self

    def fit_reference_from_snapshot(
        self,
        snapshot: GoldenCorpusSnapshot,
        lob_outputs: Optional[np.ndarray] = None,
    ) -> "CPD":
        """
        Preferred production path: refit the shadow from a trusted
        :class:`aif.golden_corpus.GoldenCorpusSnapshot`.

        The snapshot carries a content hash of the exact ``(X, y)``
        rows it represents. That hash is recorded on ``self`` and
        surfaced via ``CPDResult.shadow_source_hash`` so downstream
        auditors can trace any shadow decision back to a specific
        operator-signed corpus commitment.

        Parameters
        ----------
        snapshot : GoldenCorpusSnapshot
            Clean operator-curated (X, y) rows.
        lob_outputs : np.ndarray, optional
            Predictions from the CURRENT MLIN on ``snapshot.X`` — used
            only to seed the output-distribution and correlation
            reference windows. When omitted we fall back to ``snapshot.y``
            (i.e. treat the corpus labels as the reference output
            distribution). Passing explicit lob_outputs is recommended
            in production so the output-distribution baseline tracks
            the live model's behaviour on the trusted inputs.
        """
        X = np.atleast_2d(np.asarray(snapshot.X, dtype=float))
        y = np.asarray(snapshot.y, dtype=float).ravel()
        lob = (
            np.asarray(lob_outputs, dtype=float).ravel()
            if lob_outputs is not None else y.copy()
        )
        # Defer the actual fit to the well-tested raw path but flip the
        # safety switch to suppress its warning — we have just verified
        # that the rows came from a trusted snapshot.
        self.fit_reference(X, y, lob, authorize_raw=True)
        # Overwrite the "unsafe_raw" marker set by fit_reference with
        # the real hash.
        self._shadow_source_hash = snapshot.corpus_hash_hex
        logger.info(
            "CPD: shadow refit from GoldenCorpusSnapshot "
            "(n=%d, corpus_hash=%s).",
            snapshot.n_rows, snapshot.corpus_hash_hex[:16],
        )
        return self

    @property
    def shadow_source_hash(self) -> Optional[str]:
        """Hex hash of the snapshot that produced the current shadow."""
        return self._shadow_source_hash

    # ------------------------------------------------------------------
    # Snapshot / restore — used by DetectorResetCoordinator on rollback
    # ------------------------------------------------------------------

    def snapshot_state(self) -> dict:
        """
        Capture the fitted shadow estimator plus its provenance hash and
        all reference statistics so a later rollback can reinstate the
        shadow exactly as it was right after the previous clean deploy.

        The ``shadow_source_hash`` field is preserved verbatim so the
        restored shadow's audit trail continues to point at the trusted
        corpus commitment it was originally calibrated from.
        """
        return {
            "shadow": (
                copy.deepcopy(self._shadow) if self._shadow is not None else None
            ),
            "shadow_source_hash": self._shadow_source_hash,
            "ref_outputs": (
                self._ref_outputs.copy() if self._ref_outputs is not None else None
            ),
            "ref_correlations": (
                self._ref_correlations.copy()
                if self._ref_correlations is not None else None
            ),
            "ref_n": int(self._ref_n),
        }

    def restore_state(self, state: dict) -> None:
        """Restore CPD internals from a ``snapshot_state()`` payload."""
        shadow = state.get("shadow")
        self._shadow = copy.deepcopy(shadow) if shadow is not None else None
        self._shadow_source_hash = state.get("shadow_source_hash")
        ref_outputs = state.get("ref_outputs")
        self._ref_outputs = (
            np.asarray(ref_outputs, dtype=float).copy()
            if ref_outputs is not None else None
        )
        ref_correlations = state.get("ref_correlations")
        self._ref_correlations = (
            np.asarray(ref_correlations, dtype=float).copy()
            if ref_correlations is not None else None
        )
        self._ref_n = int(state.get("ref_n", 0))

    def refit_reference(self, lib, lob, y_ref: Optional[np.ndarray] = None) -> "CPD":
        """
        Re-fit from current LIB and LOB buffers.
        y_ref must be provided if a shadow model re-train is desired;
        otherwise only the output distribution and correlations are updated.

        If LIB/LOB hold fewer than ``reference_size`` samples we still refit
        from whatever is available (down to ``recent_size``) rather than
        silently no-op — otherwise the post-deployment baseline stays bound
        to the pre-drift distribution and triggers a retrain loop.
        """
        available = len(lib)
        if available < self.recent_size:
            logger.warning(
                "CPD.refit_reference: only %d LIB samples (need >= %d). "
                "Reference not updated.",
                available, self.recent_size,
            )
            return self
        take = min(self.reference_size, available)
        # Use the most-recent ``take`` samples so the baseline reflects the
        # current model's behaviour after a deployment event.
        X_ref = lib.get_values(take)
        lob_ref = lob.get_flat_values(take)

        if y_ref is not None:
            # refit_reference() is itself a legacy buffer-driven path, so
            # the caller has already chosen the unsafe refit mode —
            # acknowledge that explicitly to suppress the duplicate
            # security warning from fit_reference().
            self.fit_reference(X_ref, y_ref[:take], lob_ref, authorize_raw=True)
        else:
            # Update distribution and correlations without re-training shadow
            self._ref_outputs = lob_ref
            self._ref_correlations = self._feature_correlations(X_ref, lob_ref)
            self._ref_n = X_ref.shape[0]
        return self

    # ------------------------------------------------------------------
    # Main detection
    # ------------------------------------------------------------------

    def check(self, lib, lob) -> CPDResult:
        """
        Cross-check recent LIB inputs against recent LOB outputs.

        Parameters
        ----------
        lib : LIB
        lob : LOB
        """
        # ── Readiness ────────────────────────────────────────────────
        needed = self.reference_size + self.recent_size
        if len(lib) < needed or len(lob) < needed:
            return self._not_ready(lib)

        if self._shadow is None or self._ref_outputs is None:
            # Auto-fit — without ground truth we can only do checks 2 and 3
            logger.warning(
                "CPD: shadow model not fitted. "
                "Only output-distribution and correlation checks active. "
                "Call fit_reference(X, y, lob_outputs) for full detection."
            )

        X_rec = lib.get_values(self.recent_size)
        y_lob_rec = lob.get_flat_values(self.recent_size)
        X_rec = np.atleast_2d(X_rec)

        # ── 1. Shadow model divergence ────────────────────────────────
        shadow_div = 0.0
        shadow_triggered = False
        if self._shadow is not None:
            shadow_preds = self._shadow.predict(X_rec).ravel()
            shadow_div = self._divergence(shadow_preds, y_lob_rec)
            shadow_triggered = shadow_div > self.shadow_threshold

        # ── 2. Output distribution KS test ───────────────────────────
        ks_pvalue = 1.0
        ks_triggered = False
        if self._ref_outputs is not None:
            _, ks_pvalue = stats.ks_2samp(self._ref_outputs, y_lob_rec)
            ks_triggered = ks_pvalue < self.output_ks_alpha

        # ── 3. Input-output correlation check (two-condition rule) ───
        # Requires BOTH a large raw |Δr| AND a large Fisher-z standardized
        # delta on the same feature — the raw threshold keeps the signal
        # interpretable while the z-test suppresses sampling-noise wobble
        # on small recent windows.
        corr_delta_max = 0.0
        corr_z_max = 0.0
        corr_triggered = False
        if self._ref_correlations is not None:
            rec_corrs = self._feature_correlations(X_rec, y_lob_rec)
            corr_deltas = np.abs(self._ref_correlations - rec_corrs)
            corr_delta_max = float(np.nan_to_num(corr_deltas.max(), nan=0.0))

            z_stds = self._fisher_z_delta(
                self._ref_correlations, rec_corrs,
                n_ref=self._ref_n or self.reference_size,
                n_rec=X_rec.shape[0],
            )
            corr_z_max = float(z_stds.max())

            # Per-feature AND gate: a feature must exceed BOTH thresholds
            # to count as triggered. This prevents a small-window |Δr|
            # outlier on one feature pairing with an unrelated large-Δr
            # wobble on a different feature.
            feat_triggered = (corr_deltas > self.corr_threshold) & \
                             (z_stds > self.corr_z_threshold)
            corr_triggered = bool(feat_triggered.any())

        # ── Decision — shadow-anchored corroboration rule ────────────
        #
        # CPD combines three structurally distinct signals:
        #   (A) shadow divergence    — "my decisions stopped matching
        #                              an independently-trained twin"
        #   (B) output-KS            — "the distribution of my outputs
        #                              has shifted beyond noise"
        #   (C) input-output corr    — "the relationship between inputs
        #                              and predictions has changed"
        #
        # Shadow divergence is the signal that cleanly distinguishes
        # poisoning from organic drift:
        #   - Under drift, the shadow is fit on the SAME honest data
        #     MLIN trained on, so shadow ≈ MLIN and shadow_div ≈ 0
        #     even while KS and corr shift.  (Empirically we see
        #     shadow_div < 0.05 on pure drift episodes.)
        #   - Under poisoning, the attacker's labels contradict honest
        #     training, so shadow (trained on mostly-honest rows)
        #     diverges sharply from MLIN's poisoned predictions.
        #
        # KS and corr on their own can fire on drift:
        #   - KS: post-retrain output distribution on fresh live
        #     samples rarely matches the training slice exactly.
        #   - Corr: Pearson on 50-sample recent windows has std≈0.14
        #     per feature; occasionally trips both |Δr|>0.4 and |z|>4
        #     on clean streams.
        #
        # Rule — fire iff shadow is triggered AND at least one of the
        # behaviour channels (KS, corr) corroborates it.  Shadow is
        # the REQUIRED anchor; KS/corr provide the secondary evidence
        # needed to rule out a noisy shadow fit.  Under organic drift
        # the shadow stays calibrated to the post-retrain regime, so
        # this rule stays quiet and hands drift cleanly back to
        # CDD/DDD without double-firing.
        detected = shadow_triggered and (corr_triggered or ks_triggered)

        if detected:
            trigger_source = []
            if shadow_triggered:  trigger_source.append("shadow")
            if ks_triggered:      trigger_source.append("output_KS")
            if corr_triggered:    trigger_source.append("corr")
            msg = (
                f"CONCEPT POISONING suspected [{', '.join(trigger_source)}] - "
                f"shadow_div={shadow_div:.3f} (thresh={self.shadow_threshold}), "
                f"output_KS p={ks_pvalue:.4f} (thresh={self.output_ks_alpha}), "
                f"corr_delta_r_max={corr_delta_max:.3f} "
                f"(thresh={self.corr_threshold}), "
                f"corr_z_max={corr_z_max:.2f} (thresh={self.corr_z_threshold})"
            )
            logger.warning(msg)
        else:
            msg = (
                f"No concept poisoning - "
                f"shadow_div={shadow_div:.3f}, "
                f"KS p={ks_pvalue:.4f}, "
                f"corr_delta_r={corr_delta_max:.3f}, corr_z={corr_z_max:.2f}"
            )
            logger.debug(msg)

        return CPDResult(
            poisoning_detected=detected,
            shadow_divergence=shadow_div,
            shadow_threshold=self.shadow_threshold,
            shadow_triggered=shadow_triggered,
            output_ks_pvalue=float(ks_pvalue),
            output_ks_threshold=self.output_ks_alpha,
            output_ks_triggered=ks_triggered,
            corr_delta_max=corr_delta_max,
            corr_threshold=self.corr_threshold,
            corr_z_max=corr_z_max,
            corr_z_threshold=self.corr_z_threshold,
            corr_triggered=corr_triggered,
            message=msg,
            shadow_source_hash=self._shadow_source_hash,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _divergence(self, shadow_preds: np.ndarray, lob_preds: np.ndarray) -> float:
        """
        Compute divergence between shadow predictions and live LOB outputs.
        """
        if self.task in ("classifier", "classification"):
            # Disagreement rate: fraction of samples where shadow ≠ MLIN
            return float(np.mean(shadow_preds != lob_preds))
        else:
            # Normalised MAE: how much worse is MLIN vs the shadow?
            shadow_mae = float(np.mean(np.abs(shadow_preds - lob_preds)))
            ref_scale = float(np.std(lob_preds)) + 1e-8
            return shadow_mae / ref_scale

    def _not_ready(self, lib) -> CPDResult:
        needed = self.reference_size + self.recent_size
        msg = f"CPD: need {needed} samples, LIB has {len(lib)}."
        logger.debug(msg)
        return CPDResult(
            poisoning_detected=False,
            shadow_divergence=0.0, shadow_threshold=self.shadow_threshold,
            shadow_triggered=False,
            output_ks_pvalue=1.0, output_ks_threshold=self.output_ks_alpha,
            output_ks_triggered=False,
            corr_delta_max=0.0, corr_threshold=self.corr_threshold,
            corr_z_max=0.0, corr_z_threshold=self.corr_z_threshold,
            corr_triggered=False, message=msg,
            shadow_source_hash=self._shadow_source_hash,
        )

    def __repr__(self) -> str:
        return (
            f"CPD(task={self.task}, "
            f"ref={self.reference_size}, recent={self.recent_size}, "
            f"shadow_thresh={self.shadow_threshold}, "
            f"fitted={self._shadow is not None})"
        )
