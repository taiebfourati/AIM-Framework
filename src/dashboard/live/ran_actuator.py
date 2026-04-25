"""
ran_actuator.py
===============

The **closed-loop policy** that turns RTP detector state into concrete RAN
control actions.  This is the AI-native counterpart to the dashboard's
existing MTP/ATM closed loop:

::

    detector signal  ──▶ RANActuator.select_action() ──▶ RANSimulator.apply_action()
                                                                  │
                                                                  ▼
                                                       next-tick KPIs reflect it

Where MTP/ATM react to drift by *retraining the MLIN*, this actuator reacts
to drift by *changing the network* — boosting TX power, triggering a
handover, prioritising the URLLC slice, or trading throughput for BLER.

Design principles
-----------------
1. **Rule-based, not learned** — the rule table below is a transparent
   mapping from detector signature + KPI symptom to RAN action.  This is
   intentional for the MVP: a learned policy (e.g. an RL agent) is the
   obvious follow-up but adds an opaque component, and the goal here is to
   demonstrate that *any* closed RAN loop measurably reduces drift in the
   live KPIs.
2. **Cooldown gate** — we never fire the same action twice within
   ``cooldown_s`` of each other to avoid oscillation, and we never have
   more than ``max_concurrent`` finite-duration effects active at once.
3. **Symptom-driven** — the rule table looks at *which* KPI dimension the
   detector flagged (RSRP-low -> handover, delay-high -> scheduler, SINR-low
   with stable RSRP -> interference null) instead of treating "drift" as a
   single binary trigger.
4. **Pluggable** — ``select_action`` accepts an optional ``policy``
   callable that, if set, overrides the rule table; this is the hook for
   the RL upgrade path.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from dashboard.live.ran_simulator import RANAction, RANActionType, RANSimulator

log = logging.getLogger("dashboard.live.ran_actuator")


# ---------------------------------------------------------------------------
# Detector signal — the actuator's input contract
# ---------------------------------------------------------------------------

@dataclass
class DetectorSignal:
    """
    Minimal projection of the engine's ``detector`` event used by the
    actuator.  Built from the same RTP snapshot that
    ``LiveEngine._emit_detector_snapshot()`` emits.
    """
    step:        int
    ddd_fired:   bool = False
    cdd_fired:   bool = False
    dpd_fired:   bool = False
    cpd_fired:   bool = False

    # Magnitudes — used for severity ranking
    ks_max_p:    Optional[float] = None
    mmd:         Optional[float] = None
    ph_stat:     float           = 0.0
    perf_drop:   Optional[float] = None
    if_rate:     Optional[float] = None

    # Recent KPI tail (median over the last few samples) — symptom diagnosis
    recent_rsrp: Optional[float] = None
    recent_sinr: Optional[float] = None
    recent_delay: Optional[float] = None
    recent_tput: Optional[float] = None

    @classmethod
    def from_event(cls, ev: dict, kpi_tail: Optional[dict] = None) -> "DetectorSignal":
        """Build a signal from a ``detector`` event dict + optional KPI tail."""
        ddd = ev.get("ddd") or {}
        dpd = ev.get("dpd") or {}
        cdd = ev.get("cdd") or {}
        cpd = ev.get("cpd") or {}
        kt  = kpi_tail or {}
        return cls(
            step       = int(ev.get("step", 0)),
            ddd_fired  = bool(ddd.get("triggered", False)),
            dpd_fired  = bool(dpd.get("triggered", False)),
            cdd_fired  = bool(cdd.get("triggered", False)),
            cpd_fired  = bool(cpd.get("triggered", False)),
            ks_max_p   = ddd.get("ks_max_p"),
            mmd        = ddd.get("mmd"),
            ph_stat    = float(cdd.get("ph_stat") or 0.0),
            perf_drop  = cdd.get("perf_drop"),
            if_rate    = dpd.get("if_rate"),
            recent_rsrp  = kt.get("rsrp_dbm"),
            recent_sinr  = kt.get("sinr_db"),
            recent_delay = kt.get("delay_ms"),
            recent_tput  = kt.get("throughput_mbps"),
        )


# ---------------------------------------------------------------------------
# Policy callable type — for the RL upgrade path
# ---------------------------------------------------------------------------

Policy = Callable[[DetectorSignal], Optional[RANAction]]


# ---------------------------------------------------------------------------
# RANActuator
# ---------------------------------------------------------------------------

class RANActuator:
    """
    Rule-based policy that translates :class:`DetectorSignal` to
    :class:`RANAction` and forwards it to a :class:`RANSimulator`.

    Cooldowns and concurrent-effect caps are enforced to keep the loop
    stable; without them the actuator would spam TX-power boosts while the
    detector still observes the drifted distribution it's about to fix.
    """

    # ── Symptom thresholds (same units as the KPI tail) ──────────────────
    LOW_RSRP_DBM   = -100.0   # -> consider handover
    LOW_SINR_DB    = 5.0      # -> consider TX-power up / interferer null
    HIGH_DELAY_MS  = 30.0     # -> consider scheduler boost
    LOW_TPUT_MBPS  = 20.0     # -> consider MCS-down trade-off

    # ── Action magnitudes (Level-1 fidelity) ─────────────────────────────
    TX_POWER_DELTA_DB     = 3.0
    TX_POWER_DURATION_S   = 25.0
    SCHED_PRIORITY_MS     = 5.0
    SCHED_PRIORITY_DUR_S  = 15.0
    MCS_DOWN_DELTA        = 0.2
    MCS_DOWN_DUR_S        = 30.0
    INTERF_NULL_DELTA_DB  = 6.0
    INTERF_NULL_DUR_S     = 20.0

    def __init__(
        self,
        ran:               RANSimulator,
        cooldown_s:        float          = 5.0,
        max_concurrent:    int            = 3,
        policy:            Optional[Policy] = None,
        enabled:           bool           = True,
    ) -> None:
        self.ran            = ran
        self.cooldown_s     = float(cooldown_s)
        self.max_concurrent = int(max_concurrent)
        self.policy         = policy
        self._enabled       = bool(enabled)
        self._lock          = threading.RLock()
        # action_type -> wall-clock seconds when next firing is allowed
        self._next_allowed: dict[RANActionType, float] = {}
        # Telemetry — surfaced via snapshot()
        self._fire_count = 0
        self._suppressed_count = 0
        self._last_signal: Optional[DetectorSignal] = None
        self._last_action: Optional[RANAction]      = None

    # ── enable / disable (mirrors closed_loop_enabled in the UI) ─────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, on: bool) -> None:
        with self._lock:
            self._enabled = bool(on)
            log.info("RANActuator: enabled=%s", self._enabled)

    def set_policy(self, policy: Optional[Policy]) -> None:
        """Inject a learned policy (or revert to the rule table by passing None)."""
        with self._lock:
            self.policy = policy

    # ── main entry point ──────────────────────────────────────────────────

    def on_detector_event(
        self,
        ev:       dict,
        kpi_tail: Optional[dict] = None,
    ) -> Optional[RANAction]:
        """
        Engine hook — called from the engine loop after every detector
        snapshot.  Returns the action that was *applied* (or ``None`` if
        the actuator was disabled, on cooldown, or had no rule for this
        signal).
        """
        if not self._enabled:
            return None
        sig = DetectorSignal.from_event(ev, kpi_tail)
        return self._fire(sig)

    def _fire(self, sig: DetectorSignal) -> Optional[RANAction]:
        with self._lock:
            self._last_signal = sig

            # 1. Pluggable policy first (RL upgrade path)
            action = None
            if self.policy is not None:
                try:
                    action = self.policy(sig)
                except Exception as exc:
                    log.warning("custom policy raised; falling back to rules: %s", exc)
                    action = None

            # 2. Rule-based fallback
            if action is None:
                action = self._select_rule_based(sig)

            if action is None or action.type == RANActionType.NOOP:
                return None

            # 3. Cooldown gate
            now = time.monotonic()
            next_ok = self._next_allowed.get(action.type, 0.0)
            if now < next_ok:
                self._suppressed_count += 1
                log.debug(
                    "actuator: %s suppressed (cooldown %.1fs left)",
                    action.type.value, next_ok - now,
                )
                return None

            # 4. Concurrent-effect cap
            if len(self.ran._active_effects) >= self.max_concurrent:
                self._suppressed_count += 1
                log.debug(
                    "actuator: %s suppressed (max_concurrent=%d reached)",
                    action.type.value, self.max_concurrent,
                )
                return None

            # 5. Apply
            action.issued_at_step = sig.step
            self.ran.apply_action(action)
            self._next_allowed[action.type] = now + self.cooldown_s
            self._fire_count += 1
            self._last_action = action
            return action

    # ── rule table (transparent symptom -> action mapping) ────────────────

    def _select_rule_based(self, s: DetectorSignal) -> Optional[RANAction]:
        """
        Translate a :class:`DetectorSignal` into a :class:`RANAction`.

        Rule priority (top wins):

        1. DDD + low RSRP             -> HANDOVER       (coverage loss)
        2. DDD + low SINR + ok RSRP   -> INTERFERER_NULL (interferer dominates)
        3. CDD + perf-drop strong     -> TX_POWER_UP    (link budget)
        4. High delay (any trigger)   -> SCHED_PRIORITY (URLLC slice)
        5. CDD + low throughput       -> MCS_DOWN       (BLER trade-off)
        6. otherwise                  -> NOOP

        DPD/CPD are *not* mapped to RAN actions: poisoning is an MLIN-level
        problem solved by ATM retraining MTP, not by changing the network.
        """
        rsrp  = s.recent_rsrp  if s.recent_rsrp  is not None else 0.0
        sinr  = s.recent_sinr  if s.recent_sinr  is not None else 0.0
        delay = s.recent_delay if s.recent_delay is not None else 0.0
        tput  = s.recent_tput  if s.recent_tput  is not None else 1000.0

        # 1. DDD + coverage-loss symptom
        if s.ddd_fired and rsrp < self.LOW_RSRP_DBM:
            return RANAction(
                type       = RANActionType.HANDOVER,
                delta      = 0.0,
                duration_s = 0.0,   # permanent until next HO
                reason     = (
                    f"DDD fired with RSRP={rsrp:.1f} dBm "
                    f"< {self.LOW_RSRP_DBM:.0f} -> handover"
                ),
            )

        # 2. DDD + interference symptom (SINR low, RSRP ok)
        if s.ddd_fired and sinr < self.LOW_SINR_DB and rsrp >= self.LOW_RSRP_DBM:
            return RANAction(
                type       = RANActionType.INTERFERER_NULL,
                delta      = self.INTERF_NULL_DELTA_DB,
                duration_s = self.INTERF_NULL_DUR_S,
                reason     = (
                    f"DDD fired, SINR={sinr:.1f} dB low while RSRP ok "
                    f"({rsrp:.1f} dBm) -> null interferer −"
                    f"{self.INTERF_NULL_DELTA_DB:.0f} dB"
                ),
            )

        # 3. CDD with strong perf-drop -> boost link budget
        perf_drop = s.perf_drop if s.perf_drop is not None else 0.0
        if s.cdd_fired and perf_drop > 0.05:
            return RANAction(
                type       = RANActionType.TX_POWER_UP,
                delta      = self.TX_POWER_DELTA_DB,
                duration_s = self.TX_POWER_DURATION_S,
                reason     = (
                    f"CDD fired (perf_drop={perf_drop:.3f}) -> +"
                    f"{self.TX_POWER_DELTA_DB:.0f} dB TX for "
                    f"{self.TX_POWER_DURATION_S:.0f}s"
                ),
            )

        # 4. Latency-symptom (regardless of which detector fired, if any)
        if (s.ddd_fired or s.cdd_fired) and delay > self.HIGH_DELAY_MS:
            return RANAction(
                type       = RANActionType.SCHED_PRIORITY,
                delta      = self.SCHED_PRIORITY_MS,
                duration_s = self.SCHED_PRIORITY_DUR_S,
                reason     = (
                    f"delay={delay:.1f} ms > {self.HIGH_DELAY_MS:.0f} -> "
                    f"URLLC scheduler boost −{self.SCHED_PRIORITY_MS:.0f} ms"
                ),
            )

        # 5. CDD + throughput symptom -> trade rate for BLER
        if s.cdd_fired and tput < self.LOW_TPUT_MBPS:
            return RANAction(
                type       = RANActionType.MCS_DOWN,
                delta      = self.MCS_DOWN_DELTA,
                duration_s = self.MCS_DOWN_DUR_S,
                reason     = (
                    f"CDD fired, tput={tput:.1f} Mbps low -> "
                    f"MCS robustness +{self.MCS_DOWN_DELTA:.2f}"
                ),
            )

        return None

    # ── snapshot / telemetry ──────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Returned in the engine ``status()`` payload for the UI."""
        with self._lock:
            return {
                "enabled":          self._enabled,
                "fire_count":       self._fire_count,
                "suppressed_count": self._suppressed_count,
                "cooldown_s":       self.cooldown_s,
                "max_concurrent":   self.max_concurrent,
                "last_signal_step": (
                    self._last_signal.step if self._last_signal else None
                ),
                "last_action": (
                    {
                        "type":   self._last_action.type.value,
                        "delta":  self._last_action.delta,
                        "duration_s": self._last_action.duration_s,
                        "reason": self._last_action.reason,
                        "issued_at_step": self._last_action.issued_at_step,
                    } if self._last_action else None
                ),
            }
