"""
aimp — AI Management Platform package.

Paper reference: Section IV-A
  "The AI Management Platform (AIMP) is a logically centralised platform
   responsible for managing all AI functions deployed in the network."

This package provides:
  - AIMP          : top-level management facade
  - RTPComposer   : creates and reconfigures RTP instances (RTPC)
  - MTPComposer   : creates and selects MTP pipelines (MTPC)
  - AIMPPolicy    : centralised policy engine (AIPE)
  - ModelRepository / TrainingDataRepository : artifact stores
"""

from aimp.aimp import AIMP, AIMPPolicy, ModelRepository, TrainingDataRepository
from aimp.rtpc import RTPComposer, RTPProfile
from aimp.mtpc import MTPComposer, MTPSpec

__all__ = [
    "AIMP",
    "AIMPPolicy",
    "ModelRepository",
    "TrainingDataRepository",
    "RTPComposer",
    "RTPProfile",
    "MTPComposer",
    "MTPSpec",
]
