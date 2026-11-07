"""
MemLab Language Model — Constraint-based embryonic language layer.

Provides the forbidden-pattern lexicon and normalization engine for the
initial speed-training layer.
"""

from .lexicon import (
    PatternRule,
    ForbiddenPattern,
    ConfigurablePattern,
    NormalizationResult,
    Lexicon,
)
from .main import LanguageModelMain
from .run import main as cli_main

__all__ = [
    "PatternRule",
    "ForbiddenPattern",
    "ConfigurablePattern",
    "NormalizationResult",
    "Lexicon",
    "LanguageModelMain",
    "cli_main",
]
