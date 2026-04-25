"""
simu5g — Simu5G integration package for the AI Management Framework.

Provides:
  - parser: Parse Simu5G OMNeT++ output (.vec/.sca/CSV) into KPI DataFrames
  - adapter: Bridge between Simu5G data and the RTP observer framework
  - generator: Simu5G-calibrated synthetic data generator for offline testing
"""

from simu5g.parser import Simu5GParser
from simu5g.adapter import Simu5GAdapter

__all__ = ["Simu5GParser", "Simu5GAdapter"]
