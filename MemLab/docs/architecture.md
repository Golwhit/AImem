# Architecture: Organic Growth LLM

## The Core Insight

Instead of training on massive corpora (downward), we define constraints (upward) and let valid patterns emerge.

```
Laws → Microblocks → Patterns → Predictions → Thoughts → Actions → Memories
                                                              ↓
                                                    Clusters → Galaxies → Dimensions → Universes
```

## Layer 0: Laws (Axiomatic Constraints)

Law-like rules that define what **cannot** exist. These are not statistical — they are structural.

### Bidirectional Adjacency Laws (Current)

```python
# Example law: 'n' cannot be followed by 'n' when preceded by 'n'
# i.e., "nnn" is forbidden in either direction
FORBIDDEN_TRIGRAMS = {
    ('n', 'n', 'n'),      # nnn — triple repetition of same consonant
    ('t', 'n', 'n'),      # tnn — invalid cluster
    # ... to be expanded empirically
}
```

**Key property**: Laws are **bidirectional**. If `abc` is forbidden, then `cba` is also forbidden (read right-to-left). This captures the symmetry of language constraints.

## Layer 1: Microblocks

Valid character sequences that survive Layer 0 filtering.

From the seed, valid microblocks include:
- `<end><space>a<space>` → "a " at sentence start
- `<space>a<space>` → "a" as standalone word
- `<end><space>i<space>` → "i " at sentence start  
- `<space>i<space>` → "i" as standalone word
- Uppercase variants: `I`, `A`

Microblocks are the **atoms** of this language — everything builds from them.

## Layer 2: Patterns

Recognized valid sequences formed by combining microblocks.

Pattern recognition applies probabilistic weighting:
- High-frequency pairs get stronger weights
- Low-frequency pairs get weaker weights
- Forbidden transitions get zero weight

## Layer 3: Predictions

Probabilistic transition engine. Given current state, predicts next valid token with probability distribution over allowed continuations.

## Layer 4+: Thoughts → Actions → Memories → Clusters → Galaxies → Dimensions → Universes

Higher-order emergent structures built from predictive chains. Each layer adds complexity through accumulation and refinement.

## Normalization Pipeline

When input contains a forbidden pattern:
1. Detect violation (which trigram fails?)
2. Identify context (what precedes/follows?)
3. Apply normalization rule (split, delete, or substitute)
4. Return normalized form — **never** the invalid pattern

Example: `"tennis"` → detect `nn` + `i` boundary → normalize to `"tenis"` or `"ten nis"` depending on context rules.

## Design Principles

1. **Laws are axioms, not statistics** — We don't need corpus evidence; we establish what *cannot* be
2. **Growth is organic** — Each layer emerges from the one below
3. **Normalization is bidirectional** — Input cleaning and output constraint enforcement use the same rules
4. **Memory drives learning** — Persisted states enable accumulation across sessions
