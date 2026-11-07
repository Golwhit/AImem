# Session Notes: LLM From Scratch — Constraint Architecture

**Session Date:** 2026-11-06
**Working Directory:** /home/al/Desktop/Apps/MemLab
**Status:** Conceptual exploration, no code written yet

---

## Objective

Develop an LLM from scratch using efficient, unique approaches. Initial focus on establishing a constrained "speed training layer" based on forbidden character n-gram patterns in English.

---

## Core Concept: Hierarchical Constraint System

### Layer 0 — Anti-pattern DNA (Hard Filter)
- **Purpose:** Establish absolute forbidden character sequences that will never appear in valid English text
- **Mechanism:** Bidirectional gate — rejects invalid patterns from input AND prevents generation of those patterns
- **Normalization:** Invalid patterns are transformed to valid alternatives based on context
  - "nnn" → normalizes to "nn" or "n n" depending on context
  - "tennn" → could become "ten" or "tennis", never remains "tennnis"
  - "tnights" would not be generated; "ten nights" is the valid output
- **Philosophy:** These constraints are axiomatic — they "just exist" as laws of the language. No empirical proof needed; they are foundational rules.

### Layer 1 — Configurable Patterns (Soft Constraint)
- Allows context-dependent validity
- Example: Single-letter words like "a" and "I" have positional rules:
  - `</>a<space>` — valid (start of sentence + space)
  - `<space>a<space>` — valid (surrounded by spaces)
  - `</>i<space>` — valid
  - `<space>i<space>` — valid
- Multi-letter patterns follow combinatorial rules
- This is essentially a finite state automaton / regex grammar layered on top of Layer 0

### Future Layers (N+)
- Domain-specific pattern sets: mathematics, scientific notation, non-English scripts
- Each layer builds on the one before, expanding expressiveness

---

## Key Design Principles

1. **Builds One Way Only:** Unlike biological DNA (which is degenerate — many sequences code for same outcome), this system only permits valid configurations. Invalid patterns are excluded entirely.
2. **Embryonic Dataset:** Initial training data will be very small — essentially just the valid 3-character combinations of single-letter words. This creates a "seminal model" that can rudimentarily communicate.
3. **Constraints as Preprocessing/Understanding:** Not learned through trial-and-error (100,000 attempts), but understood axiomatically from the start.
4. **Punctuation Deferred:** Early stages may exclude punctuation to accelerate initial training; it can be added in later layers.
5. **Multi-layer Progression:** Several layers may be needed before reaching a "viable single layer product" — a model that can effectively communicate despite occasional errors, but is not glaringly incapable.

---

## Decisions Made

| Decision | Rationale |
|---|---|
| English-only for initial layer | Focused scope; other languages/math/science as future layers |
| Axiomatic constraints (no empirical proof needed) | Speeds development; avoids massive corpus search |
| Normalization is contextual, not fixed | "nnn" → depends on surrounding text, not a simple substitution table |
| Very small initial dataset intentional | Creates seminal model quickly; expands as layers add |
| Constraint system operates at raw text level (pre-tokenization) | Must catch patterns before BPE/tokenizer fragments them |

---

## Open Questions (To Be Addressed Next)

1. **Normalization ambiguity resolution:** How exactly does the system decide between "nn" vs "n n", or "ten" vs "tennis"? Lookup tables? Contextual rules? A separate small resolver model?
2. **Formal specification of Layer 1 rules:** Need to enumerate all valid/invalid patterns for single-letter words and their positional constraints
3. **Validation strategy:** How do we confirm the seminal model reaches "viable communication" threshold before adding next layer?
4. **Integration with training loop:** Are constraints hard-coded preprocessing, or learned parameters?

---

## Files Referenced

- None yet — this is a conceptual session

---

## Next Steps

1. Begin building out the Layer 0 forbidden n-gram lexicon (start with trigrams)
2. Define Layer 1 positional rules for single-letter words
3. Design the normalization engine architecture
4. Create initial embryonic dataset from valid 3-char combinations

