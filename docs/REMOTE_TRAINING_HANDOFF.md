# Remote Training and Reintegration Handoff

This handoff is for an AI assistant working with the user on another machine.
Read this alongside the [project instructions](../.github/copilot-instructions.md),
[training details](FROM_SCRATCH.md), and [backlog](TASKS.md). This document
describes the state observed on **2026-11-13**, not a live pointer. Check
the actual files again before copying or training.

## What the user wants

- Train the existing model from its latest accepted weights, not a fresh
  initialization, and retain useful learning across training rounds.
- Use a more powerful system, potentially including a GPU, and be able to
  bring completed or partially completed training back to this project.
- Preserve the current model, rollback, provenance, and explicit review of the
  exact training content. Avoid promoting a model solely because training
  accuracy increased; report what it gained and what it forgot.
- Keep the learning calculations from scratch. Do not substitute pretrained
  models, hosted models, or ML-library implementations for the learning core.
  Any GPU toolchain or dependency needs a separate role, license, and rights
  review; its availability alone is not permission.

## Observed starting point

The code containing the active-weight continuation path was committed in
`9acbced`; this handoff is a later documentation checkpoint. Check out the
commit containing this document and record its full Git commit hash in the
transfer manifest. Git does **not** contain the private model files: `.data/`
is ignored.

At the observation time, both active chat and permanent progress selected:

| Item | Observed value |
|---|---|
| Conversation training round | 24 |
| Run ID | `fae295fa9f434f89aac874a2505fbfbe` |
| Active and progress checkpoint SHA-256 | `ea715f7f3c3f46399974133a872157a84eefff20438d2772165f2d8d2fb5fd1c` |
| Round report SHA-256 | `6838f7b839c699386f717216803b30fd85789988a2c5bef941a111feda3e47de` |
| Reviewed authored fixture | `fixtures/neural-conversation-meal-authored-256.json` |
| Fixture SHA-256 | `c60c4e3e6971cec59e5bdd9704e1c47584174dd34b4f8eeb058e56fbc2b8f99e` |
| Explicitly approved feedback records | 18 |
| Plasticity architecture / input slots / parameters | `plain` / 64 / 17,883 |
| Symbol entries | 385, plus the model's boundary and unknown tokens |
| Latest round kind | `immediate_user_correction`, completed and replay-verified |
| Rollback selection SHA-256 | `b2f6818f65bb5b12a869046cc3facdfe131596a9e291c6c2e6630661052b1778` |
| Rollback checkpoint SHA-256 | `31314866dbefef4ac797a36c7ef663b5903a10239a3a555aeb6355b148f8e8c6` |

These are a snapshot, not permanent constants. Feedback can advance the active
selection while a handoff is being prepared. The older round-22 scores in
[TASKS.md](TASKS.md) and its focused trials predate round 24; they are not a
fresh assessment of the current checkpoint. The current development cases
have been repeatedly inspected and used as a gate, so they are not an
independent generalization test.

## Current capabilities and gaps

- [neural_active_training.py](../neural_active_training.py) continues the
  active weights on the exact approved authored fixture and explicitly
  approved feedback. It runs bounded rounds, measures and replays them, and
  stops on rejection or error. Its current strict non-regression rule can
  reject a useful tradeoff; do not silently loosen it.
- The learning core uses standard-library Python and handwritten forward,
  gradient, and SGD calculations. **It does not use a GPU.** A GPU will not
  accelerate the existing runner merely by being present.
- Completed round endpoints can be saved and selected. **There is no general
  portable export/import tool or mid-round resumable checkpoint protocol.**
  During the update loop, interruption can lose that round's in-memory
  updates. `trained.json` written after fitting is candidate evidence, not
  automatically a validated or active checkpoint.
- Existing replay checks compare weights and milestones exactly. Different
  hardware, math libraries, and numeric precision can affect exact replay.
  Do not disable or weaken the current checks to force acceptance. Design,
  document, and test any cross-device validation policy separately.
- Current learning uses plain SGD without momentum. A future optimizer may
  require additional state, which must be checkpointed explicitly.

## Prepare a private transfer

1. Agree on a trusted destination and transfer method with the user. Review
   privacy, consent, provenance, and rights for each data item; an approval
   record is a user declaration, not legal clearance. The progress pointer,
   feedback records, reports, source archives, and transcript can contain
   private conversation content. Do not publish them to Git or send them to
   an unapproved service.
2. Stop or coordinate all training, feedback updates, and activation while
   creating a consistent snapshot. Preserve the existing live files. Confirm
   that `.data/neural-chat/active.json` and
   `.data/neural-training/progress.json` select the same checkpoint; if they
   differ, stop and investigate rather than guessing which to overwrite.
3. Use a private, checksum-verified bundle or equivalent secure transfer.
   At minimum, include the two pointers; each corresponding
   `<checkpoint_sha256>.json` from `.data/neural-chat/` and
   `.data/neural-training/`; and the referenced
   `.data/neural-training/rounds/<run_id>/` evidence directory. The progress
   pointer carries approved feedback and the reviewed fixture hash; the
   fixture itself is tracked in Git. Include the relevant previous selection
   `retired-selection-<rollback_selection_sha256>.json` and its referenced
   chat checkpoint if rollback is needed. Preserve deeper lineage and the
   original comparison selection/checkpoint where the intended work needs
   them; do not assume one rollback file is the whole history.
4. Record a manifest with the exact code commit, relative paths and SHA-256
   of every transferred file, active and parent checkpoint hashes, model
   architecture and vocabulary, approved input hashes and scope, and runtime
   and hardware details. Keep private manifests private if they disclose data.
   The transfer/export mechanism and manifest format still need implementation
   and testing; this document does not claim an existing exporter.
5. On the other system, restore into a **separate workspace** without
   overwriting any existing state. Verify every manifest hash and model
   selection before training. From the project root, this existing check
   validates the referenced checkpoint bytes and pointer evidence:

   ```sh
   python3 -c 'from pathlib import Path; from neural_chat import load_active; from neural_training import load_progress; r=Path("."); a,_,_=load_active(r); p,_=load_progress(r); assert a["checkpoint_sha256"] == p["checkpoint_sha256"]; print("matching checkpoint", p["checkpoint_sha256"], "round", p["round"])'
   ```

   Also verify the copied report against `report_sha256`, the fixture against
   `reviewed_file_sha256`, and the selected rollback files against their
   hashes. This check is not an exporter, full provenance audit, or proof of
   conversational quality.

The transcript `chat_memory.json` and generated graph are not required merely
to load the active model. The immediate feedback workflow also checks the
original saved response and preceding prompt in the transcript. Transfer
those only if needed and explicitly agreed; preserve their privacy and
provenance. Do not assume unmarked chat is approved training material.

## Remote training and partial progress

Before a long run, implement and test periodic atomic checkpoints. Each
checkpoint should bind the complete model state and vocabulary to the parent
hash, code/protocol version, approved input and split hashes, update count,
random-generator state, learning schedule, and any optimizer state. Include
the backend, precision, hardware/runtime, and validation information needed
to interpret and resume it. A checkpoint must not advertise itself as
complete until its bytes and manifest are durable and verified.

Test interruption and resume in an isolated workspace: compare an
uninterrupted run with an interrupted/resumed run under the same backend,
check provenance and input hashes, and prove that failure leaves the prior
checkpoint intact. Define any GPU numerical compatibility and replay
criteria before training or promotion. Do not claim CPU bit-for-bit parity
without testing it, and do not train on new text or external datasets without
the user's approval of exact content and a provenance/rights review.

## Bring results back safely

1. Transfer the candidate, all required checkpoints, manifests, exact approved
   input hashes, code and training protocol, validation evidence, and metrics
   through the agreed private channel. Verify hashes before loading; treat
   unfamiliar executable code and unsafe serialized formats as untrusted.
2. Import into an **isolated comparison workspace**, not the active `.data/`
   tree. Check architecture, symbol/token ordering, dimensions, numeric
   precision, output masks, checkpoint schema, and the recorded parent.
   Different architectures or independently trained models cannot simply
   be merged by copying weights.
3. Compare against the actual current local selection, not just the old
   exported parent. The user may have continued training locally. Preserve
   both lineages; do not overwrite either branch or claim that weights can be
   automatically merged. If branches diverged, decide how to reconcile them
   with the user.
4. Evaluate approved feedback and retention as well as a genuinely fresh,
   separately documented evaluation set. Record individual and paired
   metrics and any losses. Do not train on evaluation answers or describe the
   repeatedly used development set as an independent test.
5. Only after compatibility, provenance, validation, and the user's desired
   gain/loss policy are satisfied, design or use a guarded promotion that
   preserves the current selection for rollback. The current project has no
   general remote import/promote command; do not overwrite `progress.json`
   or `active.json` with remote files as a shortcut.

Useful existing checks include
`python3 -m unittest -q test_neural_active_training test_neural_training test_neural_correction`
and `python3 neural_chat.py status`. Passing them does not by itself prove a
GPU implementation, the transfer process, or general conversation quality.
