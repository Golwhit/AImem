"""
Law definitions for the constraint system.

A Law is an axiomatic rule that defines what cannot exist in valid text.
Laws are bidirectional: if pattern P is forbidden, then reverse(P) is also forbidden.
"""

from dataclasses import dataclass, field
from typing import Tuple, Set, List, Optional
import re


@dataclass(frozen=True)
class ForbiddenTrigram:
    """A single forbidden character trigram with bidirectional enforcement."""
    
    pattern: Tuple[str, str, str]
    description: str = ""
    
    @property
    def reversed_pattern(self) -> Tuple[str, str, str]:
        """Return the reverse (bidirectional) form."""
        return (self.pattern[2], self.pattern[1], self.pattern[0])
    
    def matches(self, text: str, pos: int) -> bool:
        """Check if this trigram occurs at position pos in text."""
        if pos < 0 or pos + 3 > len(text):
            return False
        return (text[pos], text[pos+1], text[pos+2]) == self.pattern
    
    def __str__(self):
        return f"❌ '{''.join(self.pattern)}' ({self.description})"


@dataclass
class AdjacencyRule:
    """
    A rule that forbids certain character adjacencies.
    
    Unlike trigrams, adjacency rules check pairs and can enforce
    directional constraints (A cannot follow B).
    """
    
    forbidden_pairs: Set[Tuple[str, str]] = field(default_factory=set)
    description: str = ""
    
    def add_pair(self, a: str, b: str):
        """Add a forbidden adjacency: a followed by b."""
        self.forbidden_pairs.add((a, b))
    
    def matches(self, text: str, pos: int) -> bool:
        """Check if forbidden adjacency occurs at position pos."""
        if pos < 0 or pos + 2 > len(text):
            return False
        return (text[pos], text[pos+1]) in self.forbidden_pairs
