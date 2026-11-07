# Conversation Context: LLM Development from Scratch

## Core Philosophy (User's Stance)
- Multi-layer constraint system for LLM training
- Layer 0: Hard anti-pattern filters (e.g., "nnn" never appears; normalizes to "nn" or "n n")
- Layer 1: Configurable patterns (context-dependent validity, e.g., "I" allowed, "iii" forbidden)
- DNA analogy = ONE-WAY non-degenerate growth. Valid configs exist BECAUSE other things are impossible. Not a corpus — a generative constraint system.
- Purely embryonic/axiomatic: constraints are laws of nature for this language, not empirical claims needing 100k verification attempts
- Viable product may require multiple layers but this speeds the process

## Key Clarifications from User
- "nnn" → model trains on VALID patterns ("nn", "n n"), never on forbidden ones
- Embryonic dataset: starts tiny — only valid 3-char combos of single-letter words (</>a<space>, <space>a<space>, </>i<space>, <space>i<space> + uppercase)
- Punctuation deferred to later speed-training phase
- Normalization is the bridge between outside input and inside valid space
- Constraints = preprocess/understood training axioms, not empirical discoveries

## Current Status
- Awaiting next design decision or implementation step