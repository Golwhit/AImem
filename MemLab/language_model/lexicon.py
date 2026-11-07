"""
Lexicon — Layer 0 constraint system for the embryonic language model.

Implements a multi-layered character-ngram filter that:
  - Defines "forbidden" patterns (hard constraints) — e.g., "nnn" never appears.
  - Normalizes violations back into valid space before training or generation.
  - Grows from a small seminal dataset (single-letter words in valid trigrams).

Layer 0 = absolute anti-patterns ("100% never" for this config)
Layer 1 = configurable patterns (context-dependent validity)

Design principles:
  - Only one-way growth: patterns build up, never degenerate.
  - Normalization is contextual — "tenn" → "ten" or "tennis" depending on context.
  - Embryonic scope: starts with valid single-letter word trigrams like "</>a<space>", "<space>a<space>".
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable


class Layer(Enum):
    """Constraint layers in the lexicon hierarchy."""
    FORBIDDEN = "forbidden"       # Layer 0: absolute anti-patterns
    CONFIGURABLE = "configurable" # Layer 1: context-dependent validity


@dataclass
class PatternRule:
    """A single constraint rule in the lexicon."""

    pattern: str
    layer: Layer
    description: str
    normalization_fn: Optional[Callable[[str, str], str]] = None
    notes: str = ""


@dataclass
class Lexicon:
    """
    The living lexicon of forbidden and configurable patterns.

    Usage:
        lex = Lexicon()
        lex.add_forbidden("nnn", "Triple consonant cluster — never valid in English")
        lex.add_configurable("a", "Single-letter word, position-dependent")

        result = lex.normalize("tennis", context="he played tennis")
        # → "tennis" (allowed)

        result = lex.normalize("tennnis", context="he played tennnis")
        # → "tennis" (normalized from forbidden "nnn")
    """

    _forbidden: list[PatternRule] = field(default_factory=list)
    _configurable: list[PatternRule] = field(default_factory=list)
    _compiled_forbidden: list[tuple[re.Pattern, str]] = field(default_factory=list)
