"""
aif/signals.py — Shared signal and enum types for the AIF subsystem.

Extracted from aif/aif.py to break the import cycle:

    aif/aif.py  ←  atm/atm.py  ←  rtp/rtp.py  ←  detectors/*.py  ←  aif/aif.py

By placing MToUTSignal, TriggerReasons, and Severity in a neutral leaf
module (no imports from rtp or detectors), every consumer can import
freely without creating a cycle.

Backward compatibility: aif/aif.py re-exports all three names so any
existing import of the form ``from aif.aif import MToUTSignal`` keeps
working without modification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Severity — IntEnum so ordering and repr work cleanly (architect MEDIUM fix)
# ---------------------------------------------------------------------------

class Severity(IntEnum):
    """
    Numeric severity level for an MToUTSignal.

    Using IntEnum (not plain int constants) means:
      * ``Severity.CRITICAL > Severity.MEDIUM`` evaluates correctly.
      * ``repr`` / ``str`` shows the name, not a bare integer.
      * Comparison with plain integers still works (IntEnum is a subclass
        of int), so existing code that stores severity as a number is
        not broken.

    Levels
    ------
    LOW      — single-detector, operator-initiated.
    MEDIUM   — one drift type, or slow-poisoning (cumulative, not urgent).
    HIGH     — both drift types (data + concept drift).
    CRITICAL — poisoning confirmed (immediate rollback needed).
    """
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4


# ---------------------------------------------------------------------------
# TriggerReasons — why MToUT fired
# ---------------------------------------------------------------------------

class TriggerReasons(IntEnum):
    """
    Coded reasons carried in an MToUTSignal.

    Defined as IntEnum so they can be stored compactly and compared
    ordinally when needed.  The canonical TriggerReason enum in
    rtp/rtp.py is kept for backward compatibility; this enum is the
    neutral copy that lives outside the import cycle.
    """
    DATA_DRIFT        = auto()
    CONCEPT_DRIFT     = auto()
    DATA_POISONING    = auto()
    CONCEPT_POISONING = auto()
    OPERATOR_REQUEST  = auto()


# ---------------------------------------------------------------------------
# MToUTSignal — the payload the trigger sends to the controller (ATM)
# ---------------------------------------------------------------------------

@dataclass
class MToUTSignal:
    """
    Signal fired by MToUT toward the ATM (controller).

    This is the *neutral* copy that lives in aif/signals.py and carries no
    dependency on rtp.rtp or detectors.  The canonical class in rtp/rtp.py
    is the live runtime version; this one is provided so downstream code
    that only needs the shape (e.g. ATM unit tests, atm/atm.py TYPE_CHECKING
    imports) can reference it from the leaf module.

    Fields
    ------
    reasons : list[TriggerReasons]
        Which detectors contributed to this trigger.
    step : int
        Observation step at which the signal was raised.
    severity_level : Severity
        Pre-computed severity; default LOW.
    kpi_context : dict
        Arbitrary KPI metadata from the management plane.
    extra : dict
        Catch-all for additional structured data (detector results, etc.)
        that callers want to attach without changing the dataclass fields.
    """
    reasons: list[TriggerReasons]
    step: int
    timestamp: float = field(default_factory=__import__("time").time)
    severity_level: Severity = Severity.LOW
    kpi_context: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def severity(self) -> str:
        """Return the string name of the severity level."""
        return self.severity_level.name

    def __str__(self) -> str:
        return (
            f"MToUTSignal(step={self.step}, severity={self.severity()}, "
            f"reasons={[r.name for r in self.reasons]})"
        )
