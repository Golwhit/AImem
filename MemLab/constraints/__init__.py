"""
Layer 0: Axiomatic Constraint System

Defines law-like rules that determine what character sequences cannot exist.
These are structural constraints, not statistical observations.
"""

from .laws import LawSet, ForbiddenTrigram, AdjacencyRule
from .normalizer import NormalizationPipeline

__all__ = ["LawSet", "ForbiddenTrigram", "AdjacencyRule", "NormalizationPipeline"]
