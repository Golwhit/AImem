"""
MemLab: LLM from Scratch - Layer 0 Constraint System

This module implements a multi-layered constraint architecture for building an LLM
from first principles, starting with a "speed training layer" that uses hard 
character-level constraints to rapidly establish valid English patterns.

DESIGN PHILOSOPHY
================
- Layer 0: Hard anti-pattern filters - patterns that should NEVER exist in valid English
- Layer 1: Configurable soft constraints - context-dependent validity rules  
- Future layers: Domain-specific pattern sets (math, science, other languages)

The constraint system operates bidirectionally:
- Generation: Model cannot produce forbidden patterns
- Input: Invalid input is normalized to valid configurations before reaching the model

This creates a closed constraint system that reduces the hypothesis space, allowing
faster convergence to a "viable single layer product."

DNA ANALOGY
===========
Like biological DNA, this system builds complexity from simple constraints.
However, unlike degenerate DNA (many sequences → same outcome), our system is
non-degenerate: only valid configurations exist; invalid ones are filtered or normalized.
The system builds one way - from simple constraints to complex language capability.

USAGE
=====
from constraint_layer import ConstraintLayer0, Layer1Config

# Initialize the constraint layer
constraints = ConstraintLayer0()

# Check if a pattern is forbidden
if constraints.is_forbidden("nnn"):
    # Normalize or reject
    normalized = constraints.normalize("nnn")  # → "nn" or "n n"
    
# Validate an entire text
is_valid, violations = constraints.validate_text("tennis shoes")
"""
