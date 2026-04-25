"""
ran_simulator.py
================

Live RAN physics simulator — the **Level 2** replacement for the CSV-replay
sample source.  Wraps the project's existing 3GPP 38.901 Urban Macro physics
(``simu5g/generator.py``) into a *stateful*, *mutable*, *tick-by-tick*
simulator whose state can be modified by a closed-loop controller while the
engine is running.

Why this exists
---------------
The previous ``Simu5GCsvSource`` re-plays a pre-recorded CSV — the network
state is fixed at simulation time and the dashboard's "closed-loop" toggle
was just a linear slider decay (no real RAN response to drift).  The
``RANSimulator`` here exposes the **same Simu5G-calibrated radio model**
but as a live computation:

  * Mutable state — gNB TX power, UE distance, interference level, UE speed
  * Per-tick step — recompute path loss → SINR → throughput → latency
  * Public action API — ``apply_action(RANAction)`` mutates state with a
    physically-motivated effect (e.g. TX-power boost +3 dB)

A separate :class:`dashboard.live.ran_actuator.RANActuator` consumes detector
state from RTP and decides which action to apply, closing the AI-native
loop:

::

    detector fires ─▶ ATM (retrains MLIN) ────▶ MTP deploy
           │
           └─▶ RANActuator.select_action() ─▶ RANSimulator.apply_action()
                                                       │
                                                       ▼
                                         next tick's KPIs reflect the action

Upgrade path to OMNeT++-live
----------------------------
The same surface (``step``, ``apply_action``, ``snapshot_state``) can be
back-ended by an OMNeT++ subprocess + control socket later — only this file
changes; the engine and actuator stay put.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from simu5g.generator import (
    Simu5GRadioConfig,
    _compute_sinr,
    _path_loss_uma_nlos,
    _sinr_to_throughput,
    _compute_latency,
)

log = logging.getLogger("dashboard.live.ran_simulator")


# ---------------------------------------------------------------------------
# Action surface — what a controller can do to the RAN
# ---------------------------------------------------------------------------

class RANActionType(str, Enum):
    """Catalogue of supported RAN control actions (Level-1 fidelity)."""

    NOOP            = "noop"
    TX_POWER_UP     = "tx_power_up"      # +ΔdB on serving gNB
    TX_POWER_DOWN   = "tx_power_down"    # ΔdB rollback
    HANDOVER        = "handover"          # switch to neighbour cell
    SCHED_PRIORITY  = "sched_priority"    # URLLC slice prio → lower delay
    MCS_DOWN        = "mcs_down"          # robust MCS → less throughput, lower BLER
    INTERFERER_NULL = "interferer_null"   # null-steering kills an interferer


@dataclass
class RANAction:
    """One control message addressed to the RAN simulator."""
    type:     RANActionType
    delta:    float = 0.0       # action-specific magnitude (dB, ms, etc.)
    duration_s: float = 0.0     # how long the effect persists (0 = permanent until reverted)
    reason:   str   = ""        # human-readable cause ("DDD fired on SINR")
    issued_at_step: int = 0


# ---------------------------------------------------------------------------
# RAN state — the mutable physics knobs
# ---------------------------------------------------------------------------

@dataclass
class RANState:
    """
    Live, controller-mutable parameters of the simulated RAN.

    All values are independent of the simulator's *internal* fixed Simu5G
    radio config (``Simu5GRadioConfig``).  The fixed config provides the
    reference (e.g. baseline gNB TX power = 46 dBm); ``RANState`` holds the
    *deltas* and override values that the controller can change.
    """
    # gNB control surface
    tx_power_offset_db:   float = 0.0     # ΔdB applied to baseline gnb_tx_power_dbm
    serving_cell_id:      int   = 1       # for HANDOVER actions (1, 2, 3 in this scenario)
    interference_offset_db: float = 0.0   # ΔdB on background interference

    # Scheduler / link-adaptation overrides
    sched_priority_boost: float = 0.0     # ms removed from latency budget
    mcs_robustness:       float = 0.0     # 0..1, fraction of throughput traded for BLER

    # UE state — this drives baseline KPIs absent any control action
    ue_distance_m:        float = 250.0
    ue_speed_mps:         float = 0.0
    ue_direction_deg:     float = 0.0     # bearing for LinearMobility

    def snapshot(self) -> dict:
        return {
            "tx_power_offset_db":     self.tx_power_offset_db,
            "serving_cell_id":        self.serving_cell_id,
            "interference_offset_db": self.interference_offset_db,
            "sched_priority_boost":   self.sched_priority_boost,
            "mcs_robustness":         self.mcs_robustness,
            "ue_distance_m":          self.ue_distance_m,
            "ue_speed_mps":           self.ue_speed_mps,
            "ue_direction_deg":       self.ue_direction_deg,
        }


# ---------------------------------------------------------------------------
# RANSimulator — the live physics tick
# ---------------------------------------------------------------------------

@dataclass
class RANEffect:
    """Bookkeeping for an active action with a finite duration."""
    action:     RANAction
    expires_at: float          # wall-clock seconds (time.monotonic()) at which
                                # the effect reverts; ``inf`` = never


class RANSimulator:
    """
    Stateful RAN physics simulator.

    Each :meth:`step` advances UE mobility one slot, recomputes path loss /
    SINR / throughput / latency from the current :class:`RANState`, applies
    any active action effects, and returns one KPI dict in the same shape
    that :class:`dashboard.live.engine.Simu5GCsvSource` yields.

    Thread-safety: ``step`` and ``apply_action`` may be called concurrently
    from the engine thread and the actuator thread; an internal ``RLock``
    serialises them.
    """

    # 3GPP 38.331 A3-event handover thresholds (same as generator.py)
    HO_RSRP_DBM = -100.0
    HO_SINR_DB  = 5.0
    HO_LAT_MS   = 50.0

    def __init__(
        self,
        seed:       int = 42,
        radio_cfg:  Optional[Simu5GRadioConfig] = None,
        ue_speed_mps: float = 5.0,
        initial_distance_m: float = 200.0,
        ue_id:      int = 0,
    ) -> None:
        self.cfg = radio_cfg or Simu5GRadioConfig()
        self.rng = np.random.default_rng(seed)
        self.state = RANState(
            ue_distance_m  = initial_distance_m,
            ue_speed_mps   = ue_speed_mps,
            ue_direction_deg = 0.0,
        )
        self.ue_id = ue_id
        self._sim_time = 0.0
        self._step_count = 0
        self._lock = threading.RLock()
        self._active_effects: list[RANEffect] = []
        # Action history (recent tail) — consumed by the WS feed for
        # telemetry.  ``deque(maxlen)`` evicts in O(1); replaces a list +
        # slice-truncation that would copy up to 200 dicts on every fire.
        self._action_log_cap = 200
        self._action_log: deque[dict] = deque(maxlen=self._action_log_cap)

    # ── action API ────────────────────────────────────────────────────────

    def apply_action(self, action: RANAction) -> None:
        """
        Apply a control action to RAN state.  Effects with positive
        ``duration_s`` revert automatically after that wall-clock window;
        permanent actions persist until explicitly reversed.
        """
        with self._lock:
            now = time.monotonic()
            t = action.type
            if   t == RANActionType.NOOP:
                pass

            elif t == RANActionType.TX_POWER_UP:
                self.state.tx_power_offset_db += action.delta

            elif t == RANActionType.TX_POWER_DOWN:
                self.state.tx_power_offset_db -= action.delta

            elif t == RANActionType.HANDOVER:
                # Cycle through the 3 cells in the scenario
                old = self.state.serving_cell_id
                new = (old % 3) + 1
                self.state.serving_cell_id = new
                # Handover effect: temporary RSRP boost (closer to new BS)
                # modelled as a distance reset toward serving-cell centre.
                # This is the simplification — in OMNeT++ the new cell's
                # path loss replaces the old; here we approximate by
                # halving the effective distance for one HO_LATENCY window.
                self.state.ue_distance_m = max(
                    50.0, self.state.ue_distance_m * 0.5,
                )

            elif t == RANActionType.SCHED_PRIORITY:
                self.state.sched_priority_boost += action.delta

            elif t == RANActionType.MCS_DOWN:
                # Inline scalar clamp — avoids np.clip dispatch overhead.
                v = self.state.mcs_robustness + action.delta
                if   v < 0.0: v = 0.0
                elif v > 0.8: v = 0.8
                self.state.mcs_robustness = v

            elif t == RANActionType.INTERFERER_NULL:
                self.state.interference_offset_db -= action.delta  # less interference

            if action.duration_s > 0:
                self._active_effects.append(RANEffect(
                    action=action,
                    expires_at=now + action.duration_s,
                ))

            self._action_log.append({
                "step":   self._step_count,
                "type":   t.value,
                "delta":  float(action.delta),
                "duration_s": float(action.duration_s),
                "reason": action.reason,
            })
            # ``deque(maxlen=...)`` evicts on append — no manual cap needed.

            log.info(
                "RAN action: %s (Δ=%.2f, dur=%.1fs) — %s",
                t.value, action.delta, action.duration_s, action.reason,
            )

    # ── tick ──────────────────────────────────────────────────────────────

    def step(self, dt: float = 0.1) -> dict:
        """
        Advance the simulation by ``dt`` seconds and return one KPI sample.

        Output shape mirrors :class:`Simu5GCsvSource` so the engine loop
        doesn't need to special-case it:

        ::

            {
                "phase":            "live",
                "run":              0,
                "ue":               int,
                "t":                float (seconds),
                "rsrp_dbm":         float,
                "sinr_db":          float,
                "throughput_mbps":  float,
                "delay_ms":         float,
                "handover_flag":    int (0/1),
            }
        """
        with self._lock:
            # 1. Auto-revert expired effects
            now = time.monotonic()
            still_active = []
            for eff in self._active_effects:
                if now >= eff.expires_at:
                    self._revert_effect(eff)
                else:
                    still_active.append(eff)
            self._active_effects = still_active

            # 2. Advance UE position (LinearMobility)
            self._sim_time += dt
            self._step_count += 1
            if self.state.ue_speed_mps > 0:
                # Move along bearing; allow distance to drift in [10, 800]m.
                # Use math.* (not numpy) — these are scalars and math.cos/
                # math.radians is ~30x faster than the numpy equivalents
                # because there's no array dispatch.
                step_m = self.state.ue_speed_mps * dt
                d = self.state.ue_distance_m + step_m * math.cos(
                    math.radians(self.state.ue_direction_deg)
                )
                if   d < 10.0:  d = 10.0
                elif d > 800.0: d = 800.0
                self.state.ue_distance_m = d

            # 3. Compute physics
            cfg = self.cfg
            d = self.state.ue_distance_m

            pl = _path_loss_uma_nlos(
                d, cfg.carrier_freq_ghz, cfg.gnb_height_m, cfg.ue_height_m,
            )
            shadow = self.rng.normal(0, cfg.shadowing_std_db)
            # Scalar log10 — math.log10 is much faster than np.log10(scalar)
            r = self.rng.rayleigh(1.0)
            if r < 0.01:
                r = 0.01
            fast_fade = 20.0 * math.log10(r)

            tx_power = cfg.gnb_tx_power_dbm + self.state.tx_power_offset_db
            rsrp = (
                tx_power
                + cfg.gnb_antenna_gain_dbi
                + cfg.ue_antenna_gain_dbi
                - pl
                - shadow
                + fast_fade
            )

            # Background interference baseline + controller offset
            interf_base = -100.0  # baseline urban-macro inter-cell interference
            interf = interf_base + self.state.interference_offset_db + (
                self.rng.normal(0, 3.0)
            )

            sinr = _compute_sinr(rsrp, cfg.noise_power_dbm, interf)

            # Throughput — apply MCS-down trade-off if active
            tput = _sinr_to_throughput(
                sinr, cfg.bandwidth_mhz, cfg.num_prbs, cfg.scheduling_overhead,
            )
            tput *= (1.0 - self.state.mcs_robustness)

            # Latency — apply scheduler-priority boost
            lat = _compute_latency(
                sinr, d, self.rng, cfg.harq_max_retx, cfg.tti_ms,
            )
            lat = max(1.0, lat - self.state.sched_priority_boost)

            # 4. Clip to 3GPP ranges (inline scalar clamps — np.clip on a
            # single float carries the same dispatch overhead as np.log10).
            if   rsrp < -156.0: rsrp = -156.0
            elif rsrp >  -31.0: rsrp =  -31.0
            if   sinr <  -23.0: sinr =  -23.0
            elif sinr >   40.0: sinr =   40.0
            if   tput <    0.0: tput =    0.0
            elif tput > 1000.0: tput = 1000.0
            if   lat  <    1.0: lat  =    1.0
            elif lat  >  100.0: lat  =  100.0

            # 5. Handover-flag heuristic — A3-event style
            ho_flag = int(
                ((rsrp < self.HO_RSRP_DBM) and (sinr < self.HO_SINR_DB))
                or (lat > self.HO_LAT_MS)
            )

            return {
                "phase":           "live",
                "run":             0,
                "ue":              self.ue_id,
                "t":               self._sim_time,
                "rsrp_dbm":        rsrp,
                "sinr_db":         sinr,
                "throughput_mbps": tput,
                "delay_ms":        lat,
                "handover_flag":   ho_flag,
            }

    # ── housekeeping ──────────────────────────────────────────────────────

    def _revert_effect(self, eff: RANEffect) -> None:
        """Reverse a finite-duration effect when it expires."""
        a = eff.action
        t = a.type
        if   t == RANActionType.TX_POWER_UP:
            self.state.tx_power_offset_db -= a.delta
        elif t == RANActionType.TX_POWER_DOWN:
            self.state.tx_power_offset_db += a.delta
        elif t == RANActionType.SCHED_PRIORITY:
            self.state.sched_priority_boost -= a.delta
        elif t == RANActionType.MCS_DOWN:
            self.state.mcs_robustness = float(np.clip(
                self.state.mcs_robustness - a.delta, 0.0, 0.8,
            ))
        elif t == RANActionType.INTERFERER_NULL:
            self.state.interference_offset_db += a.delta
        # HANDOVER and NOOP have no auto-revert

    def reset(self) -> None:
        with self._lock:
            self.state = RANState(
                ue_distance_m   = 200.0,
                ue_speed_mps    = self.state.ue_speed_mps,
                ue_direction_deg = 0.0,
            )
            self._sim_time   = 0.0
            self._step_count = 0
            self._active_effects.clear()
            self._action_log.clear()
            self.rng = np.random.default_rng()

    def snapshot_state(self) -> dict:
        with self._lock:
            return {
                **self.state.snapshot(),
                "sim_time":  self._sim_time,
                "step":      self._step_count,
                "active_effects": [
                    {
                        "type":     e.action.type.value,
                        "delta":    e.action.delta,
                        "expires_in_s": max(
                            0.0, e.expires_at - time.monotonic(),
                        ),
                    }
                    for e in self._active_effects
                ],
                # deque doesn't support slicing; materialise then slice.
                "recent_actions": list(self._action_log)[-20:],
            }
