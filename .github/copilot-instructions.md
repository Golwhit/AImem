# Copilot Instructions

This project is a local chat-memory and visualization prototype.

## Ownership and dependency boundary

- Keep the learning core's model, scoring, and training calculations from scratch; the user does not want third-party pretrained models or ML-library implementations in that core.
- Third-party viewer/UI dependencies are acceptable when their licenses permit the intended use. Check commercial use, redistribution, and notice obligations; public availability alone is not permission, and permissive licensing is not obligation-free.
- Future dependencies may be considered after reviewing their role and licenses. This is not blanket approval to replace the original learning core with a library or hosted model.
- Do not introduce copied implementation code or external training datasets without provenance and rights review.
- The learning calculations in original_learning.py now use only Python's standard library. The old NumPy/scikit-learn checkpoint is preserved but not loaded. React/Express and Python's runtime still require review; do not describe the whole application as obligation-free.
- Separate original product components from third-party development tools and runtimes; do not assume the latter are exempt from review.
- Preserve code and data provenance. Historical conversations, AI-generated code/text, employer material, and user-supplied documents are not automatically cleared for commercial use or training.
- Never guarantee exclusive ownership or absence of third-party rights claims. Commercial release requires qualified legal review of the implementation, data, dependencies, and applicable service terms.
- Preserve the existing prototype and memories until replacements and data disposition are explicitly agreed; do not delete dependencies, license notices, or historical data merely to claim compliance.

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

- For delivered code/documentation checkpoints, follow `docs/VERSIONING.md`: increment patch for fixes/maintenance, minor for features, or major for breaking changes/approved stable release. Keep both package manifests and the newest `docs/SESSION_LOG.md` entry synchronized, including the actual session ID, changes, validation, and limitations. Do this before commit/sync/push; do not bump for every memory capture or push-only operation.

- Coach the user in plain language at each milestone: explain what changed, why, what a test proves, and what it does not prove. Keep docs/FROM_SCRATCH.md current and give small, concrete ways the user can contribute.
- Historical memory is searchable, not automatically approved training data. Preserve the explicit content-hashed memory/query review gate; review records are user declarations, not legal clearance.
- Update `README.md` and the memory graph when behavior changes.
- Keep changes surgical and consistent with the existing graph and transcript workflow.
