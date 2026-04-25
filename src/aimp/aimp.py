"""
aimp.py — AI Management Platform (AIMP)

Paper reference: Section IV-A
  "The AI Management Platform (AIMP) is a logically centralised platform
   responsible for managing all AI functions deployed in the network. It is
   installed in the network operator's cloud, equipped with hardware
   accelerators (GPUs, DPUs) that support the training process."

  "Its main components are:
   - Repositories (AI models, RTP and MTP components, training historical
     data, and other related information)
   - AIF Training Manager (ATM)
   - Centralised Training Platform (CTP) with Model Training Engines (MTEs)
     and Training Optimisers (TOPs)
   - RTP Composer (RTPC)
   - MTP Composers (MTPC)
   - AIMP Policy Engine (AIPE)
   - NDT, Management Plane, and External Training Platform Interfaces"

The AIMP is a facade that coordinates the existing components (ATM, NDT,
MTP-L, MTP-E) through the two composers (RTPC, MTPC).  Existing simulation
code that directly constructs ATM/RTP continues to work; AIMP is an
optional higher-level API for managed lifecycle orchestration.

Architecture
------------

    ┌────────────────────────────────────────────────────────────┐
    │                          AIMP                              │
    │                                                            │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
    │  │  RTPC    │  │  MTPC    │  │  AIPE    │                │
    │  │(Composer)│  │(Composer)│  │(Policy)  │                │
    │  └────┬─────┘  └────┬─────┘  └──────────┘                │
    │       │             │                                      │
    │       ▼             ▼                                      │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
    │  │  RTP(s)  │  │  MTP-L   │  │  MTP-E   │  │   NDT    │ │
    │  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │
    │                                                            │
    │  ┌──────────────────────────────────────────────────────┐ │
    │  │         Repositories (models, data, history)         │ │
    │  └──────────────────────────────────────────────────────┘ │
    └────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator

from aif.aif import AIF
from atm.atm import ATM, ATMPolicy, ATMResult, MTPVariant
from ndt.ndt import NDT
from rtp.rtp import RTP, RTPConfig, MToUTSignal, RTPEvent

from aimp.rtpc import RTPComposer, RTPProfile
from aimp.mtpc import MTPComposer, MTPSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AIMP Policy Engine (AIPE) — centralised policy
# ---------------------------------------------------------------------------

@dataclass
class AIMPPolicy:
    """
    Centralised policy engine for the AIMP.

    Paper reference: Section IV-A
      "AIMP Policy Engine (AIPE) is a repository of the policies regarding
       RTP thresholds that trigger the MToU process, selecting the MTP
       variant, and impacting all other operations of the AIMP."

    Parameters
    ----------
    atm_policy : ATMPolicy
        Operator policy for ATM variant selection and lifecycle decisions.
    default_rtp_config : RTPConfig
        Default RTP configuration for new AIF registrations.
    rtp_profile_name : str
        Default RTP profile name used by RTPC.
    mtp_variant_preferences : dict
        Maps severity level → preferred MTP variant.
    cost_limit : float
        Maximum normalised cost for MTP variant selection.
    reconfigure_rtp_on_model_change : bool
        Whether RTPC should reconfigure RTP detectors after model updates.
    """
    atm_policy: ATMPolicy = field(default_factory=ATMPolicy)
    default_rtp_config: RTPConfig = field(default_factory=RTPConfig)
    rtp_profile_name: Optional[str] = None
    mtp_variant_preferences: dict = field(default_factory=dict)
    cost_limit: float = 1.0
    reconfigure_rtp_on_model_change: bool = True


# ---------------------------------------------------------------------------
# Repositories — in-memory artifact stores
# ---------------------------------------------------------------------------

@dataclass
class _ModelEntry:
    """A single entry in the model repository."""
    model: BaseEstimator
    version: int
    source_variant: Optional[MTPVariant]
    timestamp: float
    metadata: dict = field(default_factory=dict)


class ModelRepository:
    """
    Repository of AI models managed by the AIMP.

    Paper reference: Section IV-A
      "Repositories.  The repositories consist of AI models, RTP and MTP
       components, training historical data, and other related information."

    In-memory dict-backed store.  Each AIF (by id) maintains a version
    history of deployed models.
    """

    def __init__(self) -> None:
        self._store: dict[int, list[_ModelEntry]] = {}

    def store(
        self,
        aif_id: int,
        model: BaseEstimator,
        source_variant: Optional[Union[MTPVariant, str]] = None,
        metadata: Optional[dict] = None,
    ) -> int:
        """Store a model and return its version number."""
        history = self._store.setdefault(aif_id, [])
        version = len(history) + 1
        entry = _ModelEntry(
            model=copy.deepcopy(model),
            version=version,
            source_variant=source_variant,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        history.append(entry)
        logger.debug(
            "ModelRepo: stored model v%d for AIF %d (source=%s).",
            version, aif_id,
            source_variant.value if hasattr(source_variant, 'value') else str(source_variant or "initial"),
        )
        return version

    def get_latest(self, aif_id: int) -> Optional[_ModelEntry]:
        """Return the latest model entry for an AIF."""
        history = self._store.get(aif_id, [])
        return history[-1] if history else None

    def get_version(self, aif_id: int, version: int) -> Optional[_ModelEntry]:
        """Return a specific model version."""
        history = self._store.get(aif_id, [])
        if 0 < version <= len(history):
            return history[version - 1]
        return None

    def version_count(self, aif_id: int) -> int:
        return len(self._store.get(aif_id, []))

    def all_aif_ids(self) -> list[int]:
        return list(self._store.keys())

    def __repr__(self) -> str:
        total = sum(len(v) for v in self._store.values())
        return f"ModelRepository(aifs={len(self._store)}, total_models={total})"


@dataclass
class _DataSnapshot:
    """A training data snapshot from LIB/LOB."""
    X: np.ndarray
    y: np.ndarray
    timestamp: float
    n_samples: int
    metadata: dict = field(default_factory=dict)


class TrainingDataRepository:
    """
    Repository of historical training data snapshots.

    Stores LIB/LOB snapshots taken at each MToU cycle, enabling
    audit trails and training data versioning.
    """

    def __init__(self, max_snapshots_per_aif: int = 10) -> None:
        self._store: dict[int, list[_DataSnapshot]] = {}
        self._max = max_snapshots_per_aif

    def store(
        self,
        aif_id: int,
        X: np.ndarray,
        y: np.ndarray,
        metadata: Optional[dict] = None,
    ) -> int:
        """Store a data snapshot; return snapshot index."""
        history = self._store.setdefault(aif_id, [])
        snap = _DataSnapshot(
            X=X.copy(),
            y=y.copy(),
            timestamp=time.time(),
            n_samples=len(X),
            metadata=metadata or {},
        )
        history.append(snap)
        # Evict oldest if over limit
        if len(history) > self._max:
            history.pop(0)
        return len(history)

    def get_latest(self, aif_id: int) -> Optional[_DataSnapshot]:
        history = self._store.get(aif_id, [])
        return history[-1] if history else None

    def snapshot_count(self, aif_id: int) -> int:
        return len(self._store.get(aif_id, []))

    def __repr__(self) -> str:
        total = sum(len(v) for v in self._store.values())
        return f"TrainingDataRepository(aifs={len(self._store)}, snapshots={total})"


# ---------------------------------------------------------------------------
# Managed AIF record
# ---------------------------------------------------------------------------

@dataclass
class _ManagedAIF:
    """Internal record for a managed AIF lifecycle."""
    aif: AIF
    rtp: RTP
    atm: ATM
    aif_id: int
    registered_at: float = field(default_factory=time.time)
    training_cycles: int = 0


# ---------------------------------------------------------------------------
# AIMP — the top-level management facade
# ---------------------------------------------------------------------------

class AIMP:
    """
    AI Management Platform — the logically centralised management entity.

    Coordinates ATM, NDT, RTPC, MTPC, and repositories to provide
    managed lifecycle orchestration for AI Functions in AI-native networks.

    Parameters
    ----------
    policy : AIMPPolicy
        Centralised policy engine configuration.
    rtpc : RTPComposer
        RTP Composer for creating and reconfiguring RTP instances.
    mtpc : MTPComposer
        MTP Composer for creating and selecting MTP pipelines.
    ndt : NDT
        Network Digital Twin for pre-deployment validation.
    model_repo : ModelRepository, optional
        Model artifact store.  Created automatically if not provided.
    data_repo : TrainingDataRepository, optional
        Training data snapshot store.  Created automatically if not provided.

    Example
    -------
    >>> from aimp import AIMP, AIMPPolicy, RTPComposer, MTPComposer
    >>> policy = AIMPPolicy()
    >>> rtpc = RTPComposer()
    >>> mtpc = MTPComposer(mtp_l=mtp_l, mtp_e=mtp_e)
    >>> ndt = NDT()
    >>> aimp = AIMP(policy=policy, rtpc=rtpc, mtpc=mtpc, ndt=ndt)
    >>>
    >>> # Register an AIF — AIMP handles all wiring
    >>> aif, rtp, atm = aimp.register_aif(estimator=clf,
    ...                                    X_ref=X_ref, y_ref=y_ref)
    >>>
    >>> # Run inference — same loop as before
    >>> for x, y_true in stream:
    ...     rtp.observe(x, y_true=y_true)
    """

    def __init__(
        self,
        policy: Optional[AIMPPolicy] = None,
        rtpc: Optional[RTPComposer] = None,
        mtpc: Optional[MTPComposer] = None,
        ndt: Optional[NDT] = None,
        model_repo: Optional[ModelRepository] = None,
        data_repo: Optional[TrainingDataRepository] = None,
    ) -> None:
        self.policy = policy or AIMPPolicy()
        self.rtpc = rtpc or RTPComposer()
        # Default MTPC is constructed with the AIMP's ATMPolicy so that
        # operator overrides (e.g. prefer_variant=LOCAL) propagate into
        # variant selection.  If the caller supplied a pre-built MTPC, we
        # retro-fit the policy when the caller left it at the default.
        if mtpc is None:
            mtpc = MTPComposer(
                policy=self.policy.atm_policy,
                cost_limit=self.policy.cost_limit,
            )
        else:
            if mtpc.policy is None or mtpc.policy.prefer_variant is None:
                mtpc.policy = self.policy.atm_policy
        self.mtpc = mtpc
        # NDT requires a model-getter callable that is AIF-specific; it is
        # constructed in register_aif().  An externally-provided NDT is kept
        # as-is (tests may supply a mock).
        self.ndt = ndt
        self.model_repo = model_repo or ModelRepository()
        self.data_repo = data_repo or TrainingDataRepository()

        self._managed: dict[int, _ManagedAIF] = {}
        self._next_id = 1

        logger.info(
            "AIMP initialised. Components: RTPC=%s, MTPC=%s, NDT=%s, "
            "ModelRepo=%s, DataRepo=%s",
            self.rtpc, self.mtpc, self.ndt,
            self.model_repo, self.data_repo,
        )

    # ------------------------------------------------------------------
    # AIF registration — the main entry point
    # ------------------------------------------------------------------

    def register_aif(
        self,
        estimator: BaseEstimator,
        X_ref: np.ndarray,
        y_ref: np.ndarray,
        rtp_profile: Optional[str] = None,
        rtp_config_overrides: Optional[dict] = None,
        on_security_alert: Optional[Callable[[RTPEvent], None]] = None,
    ) -> tuple[AIF, RTP, ATM]:
        """
        Register a new AI Function with the AIMP.

        This method performs the full wiring described in the paper:
          1. Creates an AIF with the provided estimator
          2. Uses RTPC to compose a configured RTP
          3. Creates an ATM with appropriate MTP pipelines
          4. Wires the MToUT callback from RTP to ATM
          5. Sets detector references from the provided reference data
          6. Stores the initial model in the repository

        Parameters
        ----------
        estimator : BaseEstimator
            The initial ML model for the AIF.
        X_ref : np.ndarray
            Clean reference input data for detector calibration.
        y_ref : np.ndarray
            Ground-truth labels/values for the reference data.
        rtp_profile : str, optional
            Named RTP profile to use.  If None, auto-selected by RTPC.
        rtp_config_overrides : dict, optional
            Override specific RTPConfig parameters.
        on_security_alert : callback, optional
            Security alert handler.

        Returns
        -------
        tuple of (AIF, RTP, ATM)
            The created and wired components.
        """
        aif_id = self._next_id
        self._next_id += 1

        # 1. Create AIF
        aif = AIF(estimator)

        # Build a per-AIF NDT with a model-getter bound to this AIF.
        # AIF holds the estimator in the MLIN slot (`mlin.estimator`); the
        # slot may be swapped with MLIO after a rollback, so we always
        # resolve the ACTIVE slot at call-time.
        def _current_model():
            slot = aif.mlin if getattr(aif.mlin, "state", None) is not None \
                   and aif.mlin.state.name == "ACTIVE" else aif.mlio
            return getattr(slot, "estimator", None)

        ndt_for_aif = self.ndt or NDT(
            current_model_getter=_current_model,
            min_score=self.policy.atm_policy.ndt_min_accuracy,
        )

        # 2. Compose RTP via RTPC (callbacks wired after ATM creation)
        profile = rtp_profile or self.policy.rtp_profile_name
        rtp = self.rtpc.compose(
            aif=aif,
            profile_name=profile,
            config_overrides=rtp_config_overrides,
            on_security_alert=on_security_alert,
        )

        # 3. Create ATM
        atm = ATM(
            rtp=rtp,
            mtp_l=self.mtpc.mtp_l if self.mtpc else None,
            mtp_e=self.mtpc.mtp_e if self.mtpc else None,
            mtp_c=getattr(self.mtpc, "mtp_c", None) if self.mtpc else None,
            ndt=ndt_for_aif,
            policy=self.policy.atm_policy,
            on_result=lambda result: self._on_training_complete(
                aif_id, result
            ),
        )

        # 4. Wire MToUT → ATM.handle (with AIMP interception)
        def _managed_mtout_handler(signal: MToUTSignal) -> None:
            self._on_mtout(aif_id, signal)

        rtp._on_mtout = _managed_mtout_handler

        # 5. Set detector references
        rtp.set_reference(X_ref, y_ref)

        # 6. Store initial model in repository
        self.model_repo.store(
            aif_id=aif_id,
            model=estimator,
            source_variant=None,
            metadata={"event": "initial_registration"},
        )

        # 7. Store initial training data
        self.data_repo.store(
            aif_id=aif_id,
            X=np.asarray(X_ref),
            y=np.asarray(y_ref),
            metadata={"event": "initial_reference"},
        )

        # Track
        self._managed[aif_id] = _ManagedAIF(
            aif=aif, rtp=rtp, atm=atm, aif_id=aif_id,
        )

        logger.info(
            "AIMP: registered AIF #%d (model=%s, ref=%d samples). "
            "Components: RTP, ATM, NDT wired.",
            aif_id, type(estimator).__name__, len(X_ref),
        )
        return aif, rtp, atm

    # ------------------------------------------------------------------
    # MToUT handler — intercepts before delegating to ATM
    # ------------------------------------------------------------------

    def _on_mtout(self, aif_id: int, signal: MToUTSignal) -> None:
        """
        AIMP-level MToUT handler.

        1. Uses MTPC to select the optimal MTP variant (if available)
        2. Snapshots training data to the repository
        3. Delegates to ATM.handle()
        """
        managed = self._managed.get(aif_id)
        if not managed:
            logger.error("AIMP: unknown AIF #%d in MToUT handler.", aif_id)
            return

        atm = managed.atm
        rtp = managed.rtp

        # Snapshot training data
        lib = rtp.buffers.lib
        lob = rtp.buffers.lob
        if len(lib) > 0:
            self.data_repo.store(
                aif_id=aif_id,
                X=lib.get_values(),
                y=lob.get_flat_values(),
                metadata={
                    "event": "mtout_snapshot",
                    "step": signal.step,
                    "severity": signal.severity(),
                },
            )

        # Use MTPC for variant selection if available
        if self.mtpc:
            spec = self.mtpc.compose(signal, n_samples=len(lib))
            # Temporarily set the preferred variant so ATM uses it
            original_pref = atm.policy.prefer_variant
            atm.policy.prefer_variant = spec.variant
            try:
                atm.handle(signal)
            finally:
                atm.policy.prefer_variant = original_pref
        else:
            atm.handle(signal)

    # ------------------------------------------------------------------
    # Post-training hook
    # ------------------------------------------------------------------

    def _on_training_complete(
        self, aif_id: int, result: ATMResult
    ) -> None:
        """Called after each ATM training cycle completes."""
        managed = self._managed.get(aif_id)
        if not managed:
            return

        managed.training_cycles += 1

        if result.deployed and result.variant_used:
            # Store the new model in the repository
            model = managed.aif.active_estimator
            if model is not None:
                self.model_repo.store(
                    aif_id=aif_id,
                    model=model,
                    source_variant=result.variant_used,
                    metadata={
                        "run_id": result.run_id,
                        "ndt_passed": result.ndt_passed,
                        "duration_s": result.duration_s,
                    },
                )

            # Reconfigure RTP if policy says so
            if self.policy.reconfigure_rtp_on_model_change:
                self.rtpc.reconfigure(
                    rtp=managed.rtp,
                    new_model=model,
                )

        logger.info(
            "AIMP: training cycle complete for AIF #%d — %s. "
            "Total cycles: %d, model versions: %d",
            aif_id, result.status.name,
            managed.training_cycles,
            self.model_repo.version_count(aif_id),
        )

    # ------------------------------------------------------------------
    # Management Plane interface
    # ------------------------------------------------------------------

    def request_training(
        self,
        aif_id: int,
        variant: Optional[MTPVariant] = None,
        reason: str = "operator request",
    ) -> Optional[ATMResult]:
        """
        Operator-initiated training request via the Management Plane.

        Parameters
        ----------
        aif_id : int
            ID of the registered AIF.
        variant : MTPVariant, optional
            Force a specific training variant.
        reason : str
            Human-readable reason for the request.

        Returns
        -------
        ATMResult or None
            Training outcome, or None if AIF not found.
        """
        managed = self._managed.get(aif_id)
        if not managed:
            logger.error("AIMP: request_training for unknown AIF #%d.", aif_id)
            return None

        return managed.atm.operator_retrain(variant=variant, reason=reason)

    def get_model_history(self, aif_id: int) -> list[dict]:
        """Return the model version history for an AIF."""
        history = self.model_repo._store.get(aif_id, [])
        return [
            {
                "version": e.version,
                "source": (
                    e.source_variant.value
                    if hasattr(e.source_variant, "value")
                    else (str(e.source_variant) if e.source_variant else "initial")
                ),
                "timestamp": e.timestamp,
                "metadata": e.metadata,
            }
            for e in history
        ]

    # ------------------------------------------------------------------
    # Status and introspection
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """
        Aggregate status of the AIMP and all managed AIFs.

        Returns a dict suitable for dashboard display or MP reporting.
        """
        aif_statuses = []
        for aif_id, m in self._managed.items():
            aif_statuses.append({
                "aif_id": aif_id,
                "model_type": type(m.aif.active_estimator).__name__,
                "model_versions": self.model_repo.version_count(aif_id),
                "training_cycles": m.training_cycles,
                "rtp_step": m.rtp._step,
                "data_snapshots": self.data_repo.snapshot_count(aif_id),
            })

        return {
            "managed_aifs": len(self._managed),
            "total_model_versions": sum(
                self.model_repo.version_count(i) for i in self._managed
            ),
            "rtpc": repr(self.rtpc),
            "mtpc": repr(self.mtpc) if self.mtpc else "None",
            "aifs": aif_statuses,
        }

    @property
    def managed_aif_ids(self) -> list[int]:
        """IDs of all registered AIFs."""
        return list(self._managed.keys())

    def get_aif(self, aif_id: int) -> Optional[AIF]:
        """Return the AIF instance for a given ID."""
        m = self._managed.get(aif_id)
        return m.aif if m else None

    def get_rtp(self, aif_id: int) -> Optional[RTP]:
        """Return the RTP instance for a given ID."""
        m = self._managed.get(aif_id)
        return m.rtp if m else None

    def get_atm(self, aif_id: int) -> Optional[ATM]:
        """Return the ATM instance for a given ID."""
        m = self._managed.get(aif_id)
        return m.atm if m else None

    def __repr__(self) -> str:
        return (
            f"AIMP(managed_aifs={len(self._managed)}, "
            f"model_repo={self.model_repo}, "
            f"data_repo={self.data_repo})"
        )
