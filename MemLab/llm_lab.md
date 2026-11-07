## 2026-11-06 — Conceptual Framework Established

### The Core Idea: Embryonic LLM with Axiomatic Constraints

The user is developing an LLM from scratch using a **multi-layer constraint system** as the foundational approach. Key principles:

#### Layer 0 — "100% Never" Patterns (Hard DNA)
- Forbidden character n-grams that are structurally impossible in this language
- Examples: "nnn", "qz", other patterns that violate fundamental English orthography
- **Bidirectional enforcement**: Model both rejects these in input AND never generates them
- Normalization maps violations to valid configurations (e.g., "nnn" → "nn" or "n n" depending on context)

#### Layer 1 — Configurable/Contextual Patterns (Soft Constraints)
- Patterns that are conditionally valid based on position/context
- Example: "a" can appear as `<space>a<space>`, `</>a<space>`, but NOT as isolated invalid clusters
- Single-letter words have positional rules; multi-letter patterns have combinatorial constraints
- This layer uses a finite-state or regex-like grammar approach

#### Future Layers (N+)
- Domain-specific pattern sets: mathematics, scientific notation, alternative languages
- Each layer builds on the previous, like DNA compounding

### Key Strategic Insights from Discussion

1. **The model is embryonic** — starts with almost no data, defined by what it CANNOT do rather than what it can
2. **Constraints are axiomatic**, not empirical — "nnn is invalid because it violates the fundamental nature of this language"
3. **Speed training benefit**: Aggressive hypothesis space constraining avoids needing massive datasets; model grows from a pure seed
4. **Non-degenerate DNA metaphor** — only builds one way, other patterns aren't viable for living language
5. **Punctuation deferred** — would be detrimental to speed training at early stage

### Open Questions (Unresolved)
- Scope of initial corpus: purely embryonic/seed, not real corpora yet
- Normalization mapping logic: how ambiguity is resolved (lookup tables? contextual rules?)
- What "viable single layer product" looks like operationally
- How constraint system integrates with training loop vs. preprocessing gates

---