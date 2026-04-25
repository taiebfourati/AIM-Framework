"""
mtpc.py — MTP Composer (MTPC)

Paper reference: Section IV-A
  "MTP Composers (MTPC) are responsible for the creation of MTPs used to
   train AIF.  It may select one of the three MTP variants to optimise
   MToU performance, cost, or resource consumption according to the policy
   provided by AIMP Policy Engine."

The MTPC enriches the ATM's existing variant selection logic with:
  1. Resource estimation (compute cost, training duration, data transfer)
  2. Policy-driven variant preferences (from AIMP Policy Engine)
  3. MTP-C stub with CTP-like configuration (more estimators, grid search)
  4. Composable MTPSpec objects that bundle variant + pipeline + metadata

Backward compatibility: ATM continues to work with its internal
_select_variant() when no MTPC is provided.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, clone

if TYPE_CHECKING:
    from rtp.rtp import MToUTSignal, TriggerReason

from atm.atm import MTPVariant, ATMPolicy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MTP Specification — the output of the composer
# ---------------------------------------------------------------------------

@dataclass
class MTPSpec:
    """
    Specification for a concrete MTP pipeline instance.

    Produced by MTPComposer.compose() and consumed by ATM.handle().

    Attributes
    ----------
    variant : MTPVariant
        Selected training variant (LOCAL, CLOUD, EXTERNAL).
    pipeline_ref : object
        Reference to the actual pipeline object (MTPLocal, MTPExternal,
        or the CTP-enhanced MTPLocal).
    config : dict
        Variant-specific configuration passed to the pipeline.
    estimated_cost : float
        Normalised resource cost estimate (0.0 = free, 1.0 = expensive).
    estimated_duration_s : float
        Estimated training duration in seconds.
    selection_reason : str
        Human-readable explanation of why this variant was selected.
    """
    variant: MTPVariant
    pipeline_ref: object
    config: dict = field(default_factory=dict)
    estimated_cost: float = 0.0
    estimated_duration_s: float = 0.0
    selection_reason: str = ""

    def __str__(self) -> str:
        return (
            f"MTPSpec(variant={self.variant.value}, "
            f"cost={self.estimated_cost:.2f}, "
            f"est_time={self.estimated_duration_s:.1f}s, "
            f"reason='{self.selection_reason}')"
        )


# ---------------------------------------------------------------------------
# Resource estimation model
# ---------------------------------------------------------------------------

@dataclass
class ResourceEstimate:
    """
    Resource estimate for a single MTP variant.

    Based on MLPerf benchmarking analysis from Paper Section V:
      - LT (MTP-L): mean training ~1h, 62% >18min, limited resources
      - CTP (MTP-C): mean 12min, 75% <18min, GPU accelerators
      - ETP (MTP-E): mean 2min, 90% <6min, vast resources
    """
    variant: MTPVariant
    compute_cost: float       # normalised [0, 1]
    data_transfer_cost: float # normalised [0, 1]
    estimated_time_s: float   # seconds
    privacy_risk: float       # normalised [0, 1]

    @property
    def total_cost(self) -> float:
        return self.compute_cost + self.data_transfer_cost


def _estimate_resources(
    variant: MTPVariant,
    n_samples: int,
    n_features: int = 4,
) -> ResourceEstimate:
    """
    Estimate resource costs for each MTP variant.

    Derived from MLPerf benchmarking data in Paper Section V, Table V-A.
    Values are normalised estimates for the prototype's lightweight models
    (RandomForestClassifier, 50 trees, 4 features).
    """
    if variant == MTPVariant.LOCAL:
        return ResourceEstimate(
            variant=variant,
            compute_cost=0.1,
            data_transfer_cost=0.0,   # collocated, no transfer
            estimated_time_s=max(0.5, n_samples * 0.002),
            privacy_risk=0.0,         # data stays local
        )
    elif variant == MTPVariant.CLOUD:
        return ResourceEstimate(
            variant=variant,
            compute_cost=0.4,
            data_transfer_cost=0.2,   # internal transfer to CTP
            estimated_time_s=max(0.3, n_samples * 0.001),
            privacy_risk=0.1,         # operator-controlled
        )
    else:  # EXTERNAL
        return ResourceEstimate(
            variant=variant,
            compute_cost=0.7,
            data_transfer_cost=0.5,   # transfer to third-party
            estimated_time_s=max(0.1, n_samples * 0.0005),
            privacy_risk=0.5,         # third-party data exposure
        )


# ---------------------------------------------------------------------------
# MTP Composer
# ---------------------------------------------------------------------------

class MTPComposer:
    """
    MTP Composer (MTPC) — creates and selects MTP pipelines.

    Paper reference: Section IV-A
      "MTP Composers (MTPC) are responsible for the creation of MTPs used
       to train AIF.  It may select one of the three MTP variants to optimise
       MToU performance, cost, or resource consumption according to the
       policy provided by AIMP Policy Engine."

    Parameters
    ----------
    mtp_l : MTPLocal
        Local training pipeline instance.
    mtp_e : MTPExternal
        External (MLflow) training pipeline instance.
    policy : ATMPolicy
        Operator policy governing variant selection.
    variant_preferences : dict, optional
        Maps severity level to preferred MTP variant.
        E.g. {"CRITICAL": MTPVariant.LOCAL, "HIGH": MTPVariant.EXTERNAL}
    cost_limit : float, optional
        Maximum normalised cost (0-1).  Variants exceeding this are excluded.

    Example
    -------
    >>> mtpc = MTPComposer(mtp_l=mtp_l, mtp_e=mtp_e, policy=policy)
    >>> spec = mtpc.compose(signal, n_samples=500)
    >>> print(spec)  # MTPSpec(variant=MTP-E, cost=0.70, ...)
    """

    def __init__(
        self,
        mtp_l=None,
        mtp_e=None,
        mtp_c=None,
        policy: Optional[ATMPolicy] = None,
        variant_preferences: Optional[dict[str, MTPVariant]] = None,
        cost_limit: float = 1.0,
    ) -> None:
        # Lazy-construct defaults if not provided
        if mtp_l is None:
            try:
                from atm.mtp_l import MTPLocal
                mtp_l = MTPLocal()
            except Exception as e:
                logger.warning("MTPC: could not construct default MTPLocal: %s", e)
        if mtp_e is None:
            try:
                from atm.mtp_e import MTPExternal
                mtp_e = MTPExternal()
            except Exception as e:
                logger.warning("MTPC: could not construct default MTPExternal: %s", e)
        # mtp_c is opt-in: the caller supplies one with a historical corpus
        # path. Falling back silently would hide the feature, so we leave it
        # None when not given — ATM will then fall back to MTP-L for CLOUD.
        self.mtp_l = mtp_l
        self.mtp_e = mtp_e
        self.mtp_c = mtp_c
        self.policy = policy or ATMPolicy()
        self.variant_preferences = variant_preferences or {}
        self.cost_limit = cost_limit

        # CTP-enhanced MTP-L: acts like MTP-C by using MTP-L with
        # enhanced configuration (larger grids, more estimators).
        # Paper: CTP has "advanced mechanisms needed for training
        # optimisation (including meta-learning) and selects a new
        # ML model in case of retraining."
        self._mtp_c_config = {
            "use_grid_search": True,
            "n_estimators": 200,
            "max_depth_grid": [5, 10, 20, None],
            "description": (
                "MTP-C stub: enhanced MTP-L with larger parameter grid "
                "and more estimators, modelling the Centralised Training "
                "Platform described in Paper Section IV-D-2."
            ),
        }

        self.history: list[MTPSpec] = []
        logger.info(
            "MTPC initialised. cost_limit=%.2f, preferences=%s",
            self.cost_limit, self.variant_preferences,
        )

    # ------------------------------------------------------------------
    # Main compose method
    # ------------------------------------------------------------------

    def compose(
        self,
        signal: "MToUTSignal",
        n_samples: int,
        n_features: int = 4,
    ) -> MTPSpec:
        """
        Select and configure an MTP variant for the given trigger signal.

        Selection criteria (Paper Section V):
          1. Operator policy override (prefer_variant)
          2. Severity-based preferences from AIMP Policy Engine
          3. Resource estimation and cost budgeting
          4. Fallback to ATM's default selection logic

        Parameters
        ----------
        signal : MToUTSignal
            The trigger signal from RTP's MToUT.
        n_samples : int
            Number of training samples available in LIB.
        n_features : int
            Feature dimensionality.

        Returns
        -------
        MTPSpec
            Complete specification for the selected MTP variant.
        """
        severity = signal.severity()

        # 1. Operator override from ATMPolicy
        if self.policy.prefer_variant is not None:
            variant = self.policy.prefer_variant
            reason = f"Operator override: prefer_variant={variant.value}"
        # 2. Severity-based preference from AIMP Policy Engine
        elif severity in self.variant_preferences:
            variant = self.variant_preferences[severity]
            reason = f"AIPE preference for severity={severity}: {variant.value}"
        # 3. Resource-aware selection
        else:
            variant, reason = self._resource_aware_select(
                severity, n_samples, n_features
            )

        # Build the MTPSpec
        estimate = _estimate_resources(variant, n_samples, n_features)
        spec = MTPSpec(
            variant=variant,
            pipeline_ref=self._get_pipeline(variant),
            config=self._get_variant_config(variant),
            estimated_cost=estimate.total_cost,
            estimated_duration_s=estimate.estimated_time_s,
            selection_reason=reason,
        )

        self.history.append(spec)
        logger.info("MTPC: composed %s", spec)
        return spec

    # ------------------------------------------------------------------
    # MTP-C composition (CTP stub)
    # ------------------------------------------------------------------

    def compose_mtp_c(
        self,
        signal: "MToUTSignal",
        n_samples: int,
    ) -> MTPSpec:
        """
        Explicitly compose an MTP-C variant using the CTP stub.

        Paper Section IV-D-2:
          "The use of CTP for MToU brings the benefits of huge memory and
           compute resources, including hardware accelerators, a large
           library of models, and sophisticated learning optimisation tools."

        In this prototype, MTP-C is modelled as MTP-L with enhanced
        configuration: larger parameter grids, more estimators, and
        model selection capability.
        """
        estimate = _estimate_resources(MTPVariant.CLOUD, n_samples)
        spec = MTPSpec(
            variant=MTPVariant.CLOUD,
            pipeline_ref=self.mtp_l,   # uses MTP-L engine with CTP config
            config=self._mtp_c_config,
            estimated_cost=estimate.total_cost,
            estimated_duration_s=estimate.estimated_time_s,
            selection_reason="Explicit MTP-C request (CTP-enhanced MTP-L).",
        )
        self.history.append(spec)
        logger.info("MTPC: composed MTP-C (CTP stub) — %s", spec)
        return spec

    # ------------------------------------------------------------------
    # Resource estimation
    # ------------------------------------------------------------------

    def estimate_all_variants(
        self, n_samples: int, n_features: int = 4
    ) -> dict[MTPVariant, ResourceEstimate]:
        """
        Estimate resources for all three MTP variants.

        Useful for operator dashboards and policy decisions.
        """
        return {
            v: _estimate_resources(v, n_samples, n_features)
            for v in MTPVariant
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resource_aware_select(
        self,
        severity: str,
        n_samples: int,
        n_features: int,
    ) -> tuple[MTPVariant, str]:
        """
        Apply resource-aware selection logic (Paper Section V).

        Priority:
          CRITICAL → local first (speed), escalate if insufficient
          HIGH     → external (quality) or CTP if within budget
          MEDIUM   → local if data fits, else external
          LOW      → cheapest within cost limit
        """
        estimates = self.estimate_all_variants(n_samples, n_features)

        if severity == "CRITICAL":
            if (self.policy.critical_always_local_first
                    and n_samples <= self.policy.local_max_samples):
                return (
                    MTPVariant.LOCAL,
                    "CRITICAL severity: local-first for speed "
                    f"(n={n_samples} ≤ {self.policy.local_max_samples})"
                )
            return (
                MTPVariant.EXTERNAL,
                "CRITICAL severity: escalated to external "
                f"(n={n_samples} > {self.policy.local_max_samples})"
            )

        if severity == "HIGH":
            # Prefer CTP if within cost limit, else external
            ctp_est = estimates[MTPVariant.CLOUD]
            if ctp_est.total_cost <= self.cost_limit:
                return (
                    MTPVariant.CLOUD,
                    f"HIGH severity: CTP selected (cost={ctp_est.total_cost:.2f} "
                    f"≤ limit={self.cost_limit:.2f})"
                )
            return (
                MTPVariant.EXTERNAL,
                "HIGH severity: CTP over budget, using external"
            )

        # MEDIUM / LOW
        if n_samples <= self.policy.local_max_samples:
            return (
                MTPVariant.LOCAL,
                f"MEDIUM/LOW: local fine-tune (n={n_samples} ≤ "
                f"{self.policy.local_max_samples})"
            )

        # Data too large for local — pick cheapest within budget
        viable = [
            (v, e) for v, e in estimates.items()
            if e.total_cost <= self.cost_limit and v != MTPVariant.LOCAL
        ]
        if viable:
            best = min(viable, key=lambda x: x[1].total_cost)
            return (
                best[0],
                f"MEDIUM/LOW: {best[0].value} selected "
                f"(cheapest within budget, cost={best[1].total_cost:.2f})"
            )

        # Fallback: external always available
        return (
            MTPVariant.EXTERNAL,
            "Fallback: external (no variant within cost limit)"
        )

    def _get_pipeline(self, variant: MTPVariant):
        """Return the pipeline object for a given variant."""
        if variant == MTPVariant.LOCAL:
            return self.mtp_l
        elif variant == MTPVariant.CLOUD:
            return self.mtp_l  # CTP uses enhanced MTP-L engine
        else:
            return self.mtp_e

    def _get_variant_config(self, variant: MTPVariant) -> dict:
        """Return variant-specific configuration."""
        if variant == MTPVariant.CLOUD:
            return dict(self._mtp_c_config)
        elif variant == MTPVariant.EXTERNAL:
            return {"tune_hyperparams": True}
        else:
            return {"incremental": True}

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, int]:
        """Count of MTP variants selected historically."""
        counts: dict[str, int] = {}
        for spec in self.history:
            key = spec.variant.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def __repr__(self) -> str:
        return (
            f"MTPComposer(cost_limit={self.cost_limit:.2f}, "
            f"preferences={self.variant_preferences}, "
            f"history_len={len(self.history)})"
        )
