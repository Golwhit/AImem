# Copilot Instructions

This project is a local chat-memory and visualization prototype.

## Current working model

- The conversation transcript in `chat_memory.json` is the source of truth.
- `memory_graph.html` is the rendered view of that transcript.
- `update_chat_memory.py` appends a new exchange, advances decay, regenerates the graph, and seeds the backend graph-state snapshot.
- The browser view uses backend-persisted learning state and semantic search when the API server is available, and it polls the backend snapshot so the graph refreshes when the saved state changes.
- Phone responses in the push flow should automatically append an exchange to `chat_memory.json` through the backend so memory updates are not manual.

## Architecture rules

- Preserve provenance for every memory.
- Keep domains and namespaces separate when topics diverge.
- Treat evaluation and meta discussion as their own space.
- Promote a topic into a stronger cluster only when it earns weight through repetition, importance, or shared use.
- Favor retrieval-friendly structure so future agents can query memory directly.
- Keep durable summaries stable and let transient turns decay naturally.

## Editing guidance

- Update `README.md` and the memory graph when behavior changes.
- Keep changes surgical and consistent with the existing graph and transcript workflow.
