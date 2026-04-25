"""
rtpc.py — RTP Composer (RTPC)

Paper reference: Section IV-A
  "RTP Composer (RTPC) is responsible for the creation of RTP supporting a
   specific MLI (Machine Learning Inference). In the event that the model
   is changed, the RTPC may modify components of the RTP, such as those
   responsible for concept poisoning or drift detection, to ensure
   compatibility with the new model."

The RTPC formalises what existing simulation scripts do manually:
  1. Inspect the model type (classifier vs regressor)
  2. Select a detector configuration profile
  3. Build a fully wired RTP instance
  4. After model changes, reconfigure detectors for the new model

Backward compatibility: existing code that constructs RTP directly continues
to work.  RTPC is an optional higher-level API.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from sklearn.base import BaseEstimator, is_classifier

from aif.aif import AIF
from rtp.rtp import RTP, RTPConfig, MToUTSignal, RTPEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RTP Profile — named detector configuration presets
# ---------------------------------------------------------------------------

@dataclass
class RTPProfile:
    """
    A named detector configuration preset.

    Different MLI types (e.g. a lightweight classifier vs a deep regressor)
    may warrant different detector sensitivities.  Profiles let the RTPC
    select appropriate settings automatically.

    Parameters
    ----------
    name : str
        Human-readable profile name (e.g. "classifier_default").
    task : str
        "classifier" or "regressor" — propagated to CDD and CPD.
    config : RTPConfig
        Full RTP configuration for this profile.
    description : str
        Why this profile exists / when to use it.
    """
    name: str
    task: str
    config: RTPConfig
    description: str = ""


# ---------------------------------------------------------------------------
# Default profiles
# ---------------------------------------------------------------------------

def _default_classifier_profile() -> RTPProfile:
    """Balanced profile for classification models."""
    return RTPProfile(
        name="classifier_default",
        task="classifier",
        config=RTPConfig(
            cdd_task="classifier",
            cdd_perf_drop_threshold=0.10,
            cpd_shadow_threshold=0.25,
            cpd_output_ks_alpha=0.01,
            cpd_corr_z_threshold=4.0,              # Fisher-z gate — p ≈ 6·10⁻⁵
            dpd_contamination_threshold=0.10,
            dpd_mahal_threshold=4.0,               # soft cutoff (σ)
            dpd_min_mahal_outliers=3,              # ≥3 coordinated soft hits to fire
            dpd_mahal_hard_threshold=8.0,          # single 8σ hit triggers
        ),
        description="Default profile for classification AIFs.",
    )


def _default_regressor_profile() -> RTPProfile:
    """Profile for regression models — uses MAE-based drift detection."""
    return RTPProfile(
        name="regressor_default",
        task="regressor",
        config=RTPConfig(
            cdd_task="regressor",
            cdd_perf_drop_threshold=0.15,          # MAE-based, slightly looser
            cpd_shadow_threshold=0.30,
            cpd_output_ks_alpha=0.005,
            cpd_corr_z_threshold=4.0,
            dpd_contamination_threshold=0.10,
            dpd_mahal_threshold=5.0,
            dpd_min_mahal_outliers=3,
            dpd_mahal_hard_threshold=10.0,
        ),
        description="Default profile for regression AIFs.",
    )


def _5g_nr_profile() -> RTPProfile:
    """
    Profile recalibrated for 5G NR channel variance (Simu5G-like).

    5G NR data exhibits higher intrinsic variance than a bench-grade
    classifier corpus, so we keep the soft Mahalanobis cutoff slightly
    tighter than the legacy single-hit 15σ value (which effectively
    disabled the check) but combine it with the new min-outliers rule
    so that coordinated poisoning still fires. A conservative hard cut
    of 20σ catches one-shot extreme injections without false-positiving
    on channel noise.
    """
    return RTPProfile(
        name="5g_nr",
        task="classifier",
        config=RTPConfig(
            cdd_task="classifier",
            cdd_perf_drop_threshold=0.10,
            cdd_ph_lambda=50.0,
            cpd_shadow_threshold=0.50,
            cpd_output_ks_alpha=0.00001,
            cpd_corr_threshold=0.75,
            cpd_corr_z_threshold=5.0,              # stricter — 5G wobble is large
            dpd_contamination_threshold=0.15,
            dpd_mahal_threshold=6.0,               # soft (σ) — tighter than legacy 15
            dpd_min_mahal_outliers=5,              # coordinated-attack requirement
            dpd_mahal_hard_threshold=20.0,         # hard one-shot cutoff
            ddd_reference_size=300,
            ddd_recent_size=150,
        ),
        description="Recalibrated for higher 5G NR channel variance.",
    )


# ---------------------------------------------------------------------------
# RTP Composer
# ---------------------------------------------------------------------------

class RTPComposer:
    """
    RTP Composer (RTPC) — creates and reconfigures RTP instances.

    Paper reference: Section IV-A
      "RTP Composer (RTPC) is responsible for the creation of RTP supporting
       a specific MLI.  In the event that the model is changed, the RTPC may
       modify components of the RTP, such as those responsible for concept
       poisoning or drift detection, to ensure compatibility with the new
       model."

    Parameters
    ----------
    default_config : RTPConfig, optional
        Fallback configuration when no profile matches.
    profiles : dict[str, RTPProfile], optional
        Named profiles.  If None, the three built-in profiles are registered.

    Example
    -------
    >>> rtpc = RTPComposer()
    >>> aif = AIF(classifier)
    >>> rtp = rtpc.compose(aif, on_mtout=atm.handle)
    >>> rtp.set_reference(X_ref, y_ref)
    """

    def __init__(
        self,
        default_config: Optional[RTPConfig] = None,
        profiles: Optional[dict[str, RTPProfile]] = None,
    ) -> None:
        self.default_config = default_config or RTPConfig()
        self.profiles: dict[str, RTPProfile] = profiles or {
            "classifier_default": _default_classifier_profile(),
            "regressor_default":  _default_regressor_profile(),
            "5g_nr":              _5g_nr_profile(),
        }
        self._managed_rtps: dict[int, RTP] = {}  # id(rtp) -> RTP
        logger.info(
            "RTPC initialised with %d profiles: %s",
            len(self.profiles), list(self.profiles.keys()),
        )

    # ------------------------------------------------------------------
    # Profile management
    # ------------------------------------------------------------------

    def register_profile(self, profile: RTPProfile) -> None:
        """Register a new named profile."""
        self.profiles[profile.name] = profile
        logger.info("RTPC: registered profile '%s'.", profile.name)

    # ------------------------------------------------------------------
    # Compose — create a new RTP
    # ------------------------------------------------------------------

    def compose(
        self,
        aif: AIF,
        profile_name: Optional[str] = None,
        config_overrides: Optional[dict] = None,
        on_mtout: Optional[Callable[[MToUTSignal], None]] = None,
        on_security_alert: Optional[Callable[[RTPEvent], None]] = None,
    ) -> RTP:
        """
        Create a fully wired RTP instance for the given AIF.

        Parameters
        ----------
        aif : AIF
            The AI Function to monitor.
        profile_name : str, optional
            Named profile to use.  If None, auto-selects based on model type.
        config_overrides : dict, optional
            Key-value pairs to override on the selected RTPConfig
            (e.g. {"check_interval": 25, "dpd_mahal_threshold": 10.0}).
        on_mtout : callback
            Registered as the MToUT handler (typically ATM.handle).
        on_security_alert : callback
            Registered for security event notifications.

        Returns
        -------
        RTP
            A configured and ready-to-use RTP instance.
        """
        # 1. Determine profile
        if profile_name and profile_name in self.profiles:
            profile = self.profiles[profile_name]
        else:
            profile = self._auto_select_profile(aif)

        # 2. Build config (profile base + overrides)
        config = self._build_config(profile, config_overrides)

        # 3. Construct RTP
        rtp = RTP(
            aif=aif,
            config=config,
            on_mtout=on_mtout,
            on_security_alert=on_security_alert,
        )

        # 4. Track for lifecycle management
        self._managed_rtps[id(rtp)] = rtp

        logger.info(
            "RTPC: composed RTP for AIF (model=%s) using profile '%s'. "
            "check_interval=%d, task=%s",
            type(aif.active_estimator).__name__,
            profile.name,
            config.check_interval,
            config.cdd_task,
        )
        return rtp

    # ------------------------------------------------------------------
    # Reconfigure — adapt existing RTP after model change
    # ------------------------------------------------------------------

    def reconfigure(
        self,
        rtp: RTP,
        new_model: BaseEstimator,
        X_ref: Optional[np.ndarray] = None,
        y_ref: Optional[np.ndarray] = None,
    ) -> None:
        """
        Reconfigure an existing RTP after a model change.

        Paper: "the RTPC may modify components of the RTP, such as those
        responsible for concept poisoning or drift detection, to ensure
        compatibility with the new model."

        This method:
          1. Detects if the model type changed (classifier ↔ regressor)
          2. Adjusts CDD/CPD task parameter if needed
          3. Resets detector references using current buffer data
          4. Optionally re-fits detectors on provided reference data

        Parameters
        ----------
        rtp : RTP
            The running RTP instance to reconfigure.
        new_model : BaseEstimator
            The newly trained model (not yet installed — ATM handles that).
        X_ref : np.ndarray, optional
            Fresh reference inputs for detector recalibration.
        y_ref : np.ndarray, optional
            Corresponding ground-truth labels/values.
        """
        old_task = rtp.cfg.cdd_task
        new_task = "classifier" if is_classifier(new_model) else "regressor"

        if old_task != new_task:
            logger.warning(
                "RTPC: model type changed from '%s' to '%s'. "
                "Reconfiguring CDD and CPD.",
                old_task, new_task,
            )
            rtp.cfg.cdd_task = new_task
            # Rebuild CDD with new task type
            from detectors.cdd import CDD
            rtp.cdd = CDD(
                task=new_task,
                reference_window=rtp.cfg.cdd_reference_window,
                recent_window=rtp.cfg.cdd_recent_window,
                perf_drop_threshold=rtp.cfg.cdd_perf_drop_threshold,
                ph_delta=rtp.cfg.cdd_ph_delta,
                ph_lambda=rtp.cfg.cdd_ph_lambda,
            )
            # Rebuild CPD with new task type
            from detectors.cpd import CPD
            rtp.cpd = CPD(
                task=new_task,
                reference_size=rtp.cfg.cpd_reference_size,
                recent_size=rtp.cfg.cpd_recent_size,
                shadow_threshold=rtp.cfg.cpd_shadow_threshold,
                output_ks_alpha=rtp.cfg.cpd_output_ks_alpha,
                corr_threshold=rtp.cfg.cpd_corr_threshold,
                corr_z_threshold=rtp.cfg.cpd_corr_z_threshold,
            )

        # Re-fit detector references from current buffers
        if X_ref is not None and y_ref is not None:
            rtp.ddd.fit_reference(X_ref)
            rtp.dpd.fit_reference(X_ref)
            lob_ref = new_model.predict(X_ref)
            rtp.cpd.fit_reference(X_ref, y_ref, lob_ref)
        else:
            rtp.ddd.refit_reference(rtp.buffers.lib)
            rtp.dpd.refit_reference(rtp.buffers.lib)
            rtp.cpd.refit_reference(rtp.buffers.lib, rtp.buffers.lob)

        rtp.cdd.reset_ph()

        logger.info(
            "RTPC: reconfigured RTP for new model (%s). task=%s",
            type(new_model).__name__, new_task,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _auto_select_profile(self, aif: AIF) -> RTPProfile:
        """Auto-select a profile based on the model type."""
        model = aif.active_estimator
        if model is None:
            return self.profiles.get(
                "classifier_default", _default_classifier_profile()
            )
        if is_classifier(model):
            return self.profiles.get(
                "classifier_default", _default_classifier_profile()
            )
        return self.profiles.get(
            "regressor_default", _default_regressor_profile()
        )

    def _build_config(
        self,
        profile: RTPProfile,
        overrides: Optional[dict] = None,
    ) -> RTPConfig:
        """Build an RTPConfig from a profile, applying optional overrides."""
        config = copy.deepcopy(profile.config)
        if overrides:
            for key, value in overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    logger.warning(
                        "RTPC: unknown config override '%s' — ignored.", key
                    )
        return config

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def managed_count(self) -> int:
        """Number of RTP instances created by this composer."""
        return len(self._managed_rtps)

    def __repr__(self) -> str:
        return (
            f"RTPComposer(profiles={list(self.profiles.keys())}, "
            f"managed={self.managed_count})"
        )
