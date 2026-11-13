# Building Our Learning Core

## Goal And Boundary

**Current direction:** a persistent conversation learner uses the layer-free
word/punctuation design with character fallback and the tested 64-slot context.
The original four-slot New Base remains the preserved prose baseline.
L0-L2 are retired; layer
sections below preserve historical evidence, not the current development
plan. The original character MLP remains the comparison control and the
teaching lab model. See [working model commands](../README.md#working-model-new-base-no-layers).
Preserved prose checkpoint: run `10255643d34e41749c6bcc1f68ee8b5c`, `new`, seed 17,
four passes, 7,078 parameters; paired original control: 2,103 parameters.
This selected model trained on Acts 6-12 only, not the retired model's
whole-Acts/Corinthians history. Activation copies both states, validates
evidence/finite weight shapes and pins their SHA256; previous selection and
weights are retained. Working training compares only original and new, without
loading layer banks/dictionaries. Greedy generation is deterministic.
Unmarked chat is memory capture only; clicking Relevant or submitting a
correction explicitly approves immediate conversation-weight training.
See [immediate feedback training](#immediate-feedback-training).

**Memory stays connected:** development conversations, internal app-chat
exchanges and important findings continue updating the transcript and graph
through their existing capture paths. Isolating a model experiment means
controlling its training inputs and inference context, not disconnecting
memory or pausing its growth. Capture/retrieval is separate from permission
to train; preserve provenance and the explicit content-hashed review gate.

Application checkpoints use [patch/minor/major versioning](VERSIONING.md) and
the [versioned session log](SESSION_LOG.md). Model generations, rulesets, and
neural run identifiers remain separate from application versions.

We are building the learning calculations ourselves, rather than calling a
model service or using a machine-learning library. The first milestone is a
small relevance learner: given a question and a memory, estimate how useful
that memory is for answering the question.

That relevance component is not a language model, a reasoning engine, or a
system that knows whether a statement is true. A separate tiny neural
language-model baseline is now available below; neither establishes reasoning
or truthfulness.

The core uses Python's standard library. Third-party viewer/UI dependencies
such as React and Express are acceptable to the user when suitably licensed;
future dependencies can be considered separately. Python itself is also
third-party software. Public availability does not establish commercial rights.
Implementing our calculations does not certify that the whole product is
obligation-free or that AI-assisted code has exclusive ownership. Keep license
and provenance review separate from engineering progress.

## Working Base Stability Diagnostic

The next step toward conversation is to establish what the working base can
learn and what information it loses. [neural_base_diagnostic.py](../neural_base_diagnostic.py)
does not change architecture, active selection, memory capture or the original
comparison control. It uses the exact approved Acts 6-12 training vocabulary
and only the first two approximately 384-character **training** snippets.
No evaluation chapters or conversation history are training inputs.

```sh
python3 neural_base_diagnostic.py --acknowledge-review
python3 -m unittest test_neural_base_diagnostic -v
```

### Results: run 2d557e54d04543efa9b8dc0fe2220d56

Three fresh seeds (17/117/217), rates .1/.03/.01, same per-seed random draws
across rates. Phase A: 392 characters, 332 token targets, 1,500 replacement-
sampled updates with checks at 250/750/1,500. Phase B: 384 characters,
321 targets, another 500 updates without replay. Both phases use one fixed
full-training vocabulary. This is a brief fitting/interference diagnostic,
not equal-quality convergence, conversational training, or unseen evaluation.

| Rate | Initial A token loss | After fitting A | A loss change after B | A word accuracy after fitting |
|---|---:|---:|---:|---:|
| .1 | 4.324 | 2.589 | +.354 (worse) | 10.42% |
| .03 | 4.324 | 3.098 | -.028 (better) | 4.17% |
| .01 | 4.324 | 3.386 | -.086 (better) | 0.00% |

Values average three seeds. The word measurement generates 32 characters at
the first 16 source word boundaries and counts only a complete exact first
word; it is not teacher-forced token accuracy. Rate .1's token accuracy is
30.02% on A, not 10.42%. Each trial takes about 3.5 CPU seconds for updates,
excluding audits, metrics and writes. Small shared probes and three seeds
do not establish statistical confidence or intelligence.

Rate .1 learns A faster but increases its loss after B in **all three seeds**.
Both lower rates improve A loss slightly while learning B. This is evidence
of a rate/interference tradeoff, not proof that .03 is globally better:
lower rates have learned less after the same update budget. No active rate
or weights were changed. Near-perfect fitting has not been demonstrated.

### What extra training cannot solve

The audit groups examples by the actual four input IDs **and allowed-output
mask**. If the same observable input has conflicting targets, a deterministic
next-token predictor cannot get all those positions right. Full training
has 24,142 token examples but only 6,939 distinct observable inputs, of which
2,195 conflict. The empirical maximum exact teacher-forced token accuracy
is **60.67%**; minimum mean cross-entropy is at least **1.079**. Phase A's
corresponding accuracy bound is 87.35%, versus the observed 30.02% at .1.
The model is not merely hitting the bound; substantial optimization/capacity
limitations may remain.

These are **dataset-specific token bounds**, not ceilings on word accuracy,
understanding, or all possible architectures. Natural text can have several
legitimate continuations; ambiguity is not automatically a software bug.

A more decisive conversation example comes from the pending teaching draft:

- "Mira has a red box. What color is the box?" -> "Red."
- "Mira has a blue box. What color is the box?" -> "Blue."

With the current word/gap features, both prompts produce **identical context
IDs and output masks**. They also have identical suffixes, so the discarded
color cannot affect later features. Greedy continuations must therefore be
identical, irrespective of how much the weights train. No rate change can
make this representation answer both correctly. The draft was inspected
for this collision only, not used for training or vocabulary construction.

The next architecture experiment should first preserve the relevant earlier
fact and distinguish word spellings, with explicit speaker/turn boundaries.
Freeze a bounded single-context design before comparing it; do not jump to
parallel branches or reinstate the retired layers. Continuous memory capture
remains connected. Separately label model-only versus memory-assisted inference
so retrieval is not mistaken for learning.

### Teaching material awaiting review

[neural-conversation-draft.json](../fixtures/neural-conversation-draft.json)
contains eight proposed training exchanges and ten proposed tests: direct
facts, attributes, transfers, negation, corrections, missing information,
cross-turn references, paraphrases and an unfamiliar name. It is newly
AI-assisted illustrative material with explicit provenance, not copied from
an external training corpus; that is not legal clearance or exclusive-rights
assurance. It remains **draft_pending_user_review_not_approved_for_training**.
There is deliberately no training loader or runtime answer lookup.

The proposed test split is visible during design, not an untouched locked
benchmark. Review the content, acceptable alternative answers and generalization
expectations, then approve/hash final splits before use. Keep separate development
examples for tuning. A tiny teaching set alone will not make a broad assistant.
A concrete user contribution: rewrite two examples in natural phrasing and
specify how the answer should change when one fact changes.

### Verification

All nine final weight sets were independently replayed from initialization
with the saved seeds and draws. Weights, complete word continuations and all
non-timing checkpoint/retention metrics match exactly. Saved-state restoration
also checks both phases. Pinned working/control and historical model file
hashes are unchanged. Four focused tests verify input-mask-aware bounds,
exact miniature replay, observable name/context collisions and review-only
disjoint teaching/test drafts. Editor diagnostics report no errors; direct
Pylance syntax calls timed out, so execution/tests and editor diagnostics are
the available validation evidence.

Run evidence is in `.data/neural-base-diagnostic/<run_id>/`: source snippets,
vocabulary, protocol/code hashes, checkpoints, report and `validation.json`.
No experiment was activated and no draft conversation was trained during that
diagnostic. The subsequent explicitly approved trial is recorded below.

## First Approved Conversation Training

After reviewing the proposed next step, the user instructed: "let's train and
see what we get". Run `576f66ae430b4f0088fde6af5fc87e76` records that declaration
and exact curriculum hash in its `review.json`:
`e222bcf6b88394bbe801fa9fd146df14658d6d06289a5f2386d1660130f7f46b`.
The original draft file, including its historical pending-review status,
is unchanged; approval is specific to these bytes and the eight training
exchanges, not test-answer training or blanket reuse of history.
Changed file content fails the explicit review gate before model loading.
This is a user declaration, not legal clearance.

```sh
python3 neural_conversation_trial.py --acknowledge-review \
  --reviewed-sha256 e222bcf6b88394bbe801fa9fd146df14658d6d06289a5f2386d1660130f7f46b
```

### Actual continuation, not a replacement model

Each rate restores the same actual working seed-17/four-pass Acts checkpoint.
It is **not** a fresh model or a new architecture. The initial attempt failed
before training because a target contained a straight apostrophe absent from
the parent alphabet. The completed run uses the existing `extend_alphabet`
helper to append only `char:'`, derived from approved teaching content.
It preserves existing token IDs, embeddings and connection values and copies
the unknown embedding/output values for the new symbol. Output normalization
changes when adding a class: the prepared model has 7,103 parameters versus
7,078 in the active parent. Both original and prepared parent states are saved.
No test content expands vocabulary or updates parameters; word identities
remain the parent's limited Acts vocabulary.

Prompts use ordinary `user: ...` / `assistant: ...` text with newlines, ending
in `assistant: `. These labels are not new role embeddings. The only gradient
targets are assistant-answer tokens plus a newline terminator: 97 examples
across eight teaching exchanges. User/prior turns are causal context only.
This avoids training the test answers or silently training every user word.
Greedy generation allows 96 characters, scoring the first newline-terminated
line; absent newline means failure. Expected-answer length never controls
decoding. Raw continuations and termination flags are retained.

Three fixed rates .1/.03/.01, same seed-17 target draws, checks at
500/2,000/5,000 updates. These are three learning rates from **one parent**,
not independent training seeds. Teaching scores are checked at milestones;
the ten separate test cases are checked before adaptation and at fixed final
endpoints only, with no test-guided tuning or best-rate activation.
They remain developer-visible development tests, not fresh locked evaluation.

### Results

| Rate | Initial answer-token loss | Final loss | Exact teaching answers | Exact test answers | Update CPU |
|---|---:|---:|---:|---:|---:|
| .1 | 3.733 | .600 | 1/8 | 1/10 | 9.13s |
| .03 | 3.733 | .761 | 0/8 | 0/10 | 9.17s |
| .01 | 3.733 | 1.121 | 0/8 | 0/10 | 9.54s |

Before adaptation both splits score zero. Every final test response terminates,
but termination is not meaningful answering. Case/whitespace-normalized scores
equal exact scores here. Single-reference scoring can reject valid paraphrases;
the raw outputs below show a more fundamental failure than cosmetic mismatch.

At .1:

- Jon keeps a box; who has it? **"Jon."** (correct).
- Jon gives the box to Mira; who has it? **"Jon."** (wrong).
- Red-box and blue-box color questions: **"No."** for both.
- Correct the box from blue to red: **"What it is."**
- Unfamiliar holder Tavi: **"No."**

At .03 frequent outputs include **"Nor."** and **"Bo color."**; at .01,
**"Bor."** is returned across all test prompts. Lower teacher-forced loss
does not imply coherent free-running replies. The token-accuracy ceiling on
these 97 answer targets is 87.63% given current features/masks, so even the
teaching examples contain irreducible collisions under this representation.

The changed-holder pair has identical first context IDs and masks. Role text
uses part of the tiny word history, further pushing the actual question/facts
out. No amount of training can distinguish those identical observable inputs.
This trial confirms the need to preserve fact-bearing context and useful
word distinctions; it is not evidence that an untested larger design will
automatically understand. No architecture improvement was smuggled into
this baseline trial. No original-corpus replay or post-adaptation retention
evaluation was performed, so no preservation-of-knowledge claim is made.

All three full final weight sets were independently replayed from the saved
prepared parent. Weights, complete training/test predictions, losses and
all non-timing milestone metrics match exactly. Saved-state restoration also
reproduces test replies. Five focused tests cover answer-only causal targets,
exact miniature replay, termination semantics, content-hash rejection,
training-only alphabet expansion and role-format context collisions.
The ten related base/chat regression tests pass. Editor diagnostics report
no errors. Active/chat comparison checkpoint files remain hash-identical.

Memory capture and important findings remain connected throughout. The
experimental outputs were not activated. Evidence, approval, source snapshot,
parent states, checkpoints and `validation.json` remain under
`.data/neural-conversation-trial/576f66ae430b4f0088fde6af5fc87e76/`.

## Permanent Conversation Training

The user approved five update-CPU minutes and moving saved learning into
chat, rather than repeatedly discarding experiments. This is separate
from permission to train automatically on future conversation content.
The exact previously approved 64-exchange file is the only optimizer data.
No new generated answers, development chat, historical memories or test
answers enter these neural weights.

[neural_training.py](../neural_training.py) bootstraps from the first
protocol seed (17), full-64, fresh-scaled checkpoint of the meal run
`ee7f3c1be14f477a82e2db03ae72ff71`, not a test-selected winning seed.
It continues **actual learned weights**, not their initialization.
Subsequent invocations load the last checksum-pinned `progress.json`
checkpoint and continue it. Plain SGD has no optimizer accumulators;
rate .1 is fixed and each round records its new deterministic draw seed
`1000000 + round`. This resumes learned state, not the old random stream.
The default rate is .1; `--rate` overrides it per round (see the health guard below).
The original prose and character-control checkpoints are not overwritten.
Resetting this conversation model originally discarded prose learning;
continued training does not establish retained Acts knowledge.

Each round records exact approved inputs, source copies, initial state and
parent hash, then spends a bounded CPU allowance on updates with teaching
checks at one-fifth and all of the budget. The maximum final overshoot is
one 100-update batch. It saves trained weights before final development
evaluation, independently replays all updates/milestones and raw test
predictions, and only then advances the permanent progress pointer.
The immutable prior completed checkpoint remains available if a round
fails or is interrupted. Failed/unverified rounds do not advance progress.
The completed report hash is bound into checkpoint evidence. A file lock
prevents concurrent training rounds from losing each other's updates.
There is no training daemon; each invocation finishes. Evaluation, writes
and full replay require additional time beyond the update allowance.

Promotion is an explicit command, or `--activate-after` on a training
invocation, acting only on a completed, replay-verified checkpoint.
It stores a checksum-pinned copy in chat, archives the previous active
selection and binds its hash for rollback. Repeated promotion of the same
checkpoint is idempotent. Rollback validates the saved selection and its
checkpoint **before** changing the active pointer; it does not erase
permanent training progress. The original character comparison remains
available through its separate selection. Missing/corrupt states and empty
first-line generation fail explicitly, never falling back to canned answers.

Runtime prompt formatting matches teaching: `user: CONTENT\nassistant: `.
Only the current message is supplied to neural generation, not prior app
turns or search results. The reply is the first newline-delimited line of
the unchanged greedy decoder, with a 96-character ceiling. A missing newline
is reported in generation metadata, not silently counted as a terminated
answer in evaluation. Backend/UI/graph identify this as
`local-conversation-neural`, with training-round provenance, rather than
mislabeling it as a historical layered model or claiming scripture grounding.
Normal retrieval and transcript/graph capture continue; memory storage is
not neural training permission. Explicit Relevant clicks and submitted user
corrections now approve their exact pair and its input-symbol expansion.
Other new teaching bytes still need separate exact approval; no hidden
self-training loop.

```sh
python3 neural_training.py --cpu-seconds 300 --rate 0.1 --activate-after
python3 neural_training.py --activate
python3 neural_training.py --rollback
python3 neural_chat.py status
```

Rounds no longer require an approval flag; the fixture bytes are hashed
on every run. `--activate-after` promotes only a round whose status is
`completed`; a rejected round is never activated.

Tests exercise two successive saved rounds, exact parent-weight continuation,
independent replay, content-hash rejection, corrupt-state rejection,
promotion/idempotence/rollback, training-format inference, first-line
truncation, empty-answer errors, and unchanged original comparison generation.
The HTTP integration test checks the conversation-mode label, saved model
provenance and no neural weight updates even when legacy prompt-training
flags are supplied. Production viewer build succeeds.

### Health guard and lower-rate rounds

Round 10 (`2cbf7c04…`) ran at rate .1 with the 300 CPU-second budget and
scored 0.000 on teaching and test. Its hidden units saturated (saturation
1.0), so gradients died.

`neural_training.py` now rejects any round where hidden saturation exceeds
`MAX_SATURATION = .6` at any checkpoint, or where final teaching exact falls
below `MIN_TEACHING_EXACT = .9`. Rejected rounds keep their evidence with
`status: rejected` and a `rejection` reason, and they do not advance
`progress.json`. Round 11 (saturation 1.000) and the retried round 10
(saturation 1.000) were rejected this way.

Round 10 at rate .02 with a 60 CPU-second budget passed: saturation peaked
at .443, teaching exact 1.000, development test exact 0.688 (`17bc4a09…`).
It was activated. Round 14 (`b60ce460…`), trained from the correction-chain
round 13, also passed: saturation .49–.50, teaching 1.000, development test
0.594. Round 14 is the active model. Development test is not used to choose
checkpoints, so the 0.688 and 0.594 scores do not rank the rounds.

Click-triggered corrections (`neural_correction.py`) previously promoted
without the health check. They now reject a candidate whose hidden
saturation over the rehearsal set exceeds `MAX_SATURATION`, and the chain
pointer stays unchanged. Only one run at rate .02 has been checked; the
lower rate is not yet proven stable over repeated rounds.

### First permanent round and chat promotion

Round 1: `57f452944e2c4767b5586c8320bfbad7`, 57,100 additional updates,
300.28 update-CPU seconds, fixed seed 1000001. At the 60.22-second teaching
checkpoint, all 64 teaching answers are already exact; final loss falls
to .000714 with all 64 still correct. This is fitting, not proof of
general understanding. Development evaluation is **22/32 exact**, including
16/16 familiar-phrasing combinations and 6/16 changed-wording examples.
Both members are correct for 10/16 pairs; two pairs receive identical answers.
Every evaluation answer terminates, and normalized scores equal exact scores.
The actual seed-17 parent scored 21/32, so the extra training adds one
correct development answer, not a 30% compute-qualified gain.
All ten remaining errors are in changed wording; training has not fixed
the previously documented missing word-identity distinctions.

The entire trained state, both non-timing teaching milestones and all
development continuations replay exactly. The completed checkpoint SHA256 is
`c7dec17d83b210de79293e78d814056ef03dcc9e31c7579932c38a0413e878d9`.
It is **now active in chat**, with the previous New Base selection archived
for rollback and the original character control unchanged.
Live Python/API/browser checks produce "Mira." for keeps, "Jon." for gives,
and "Red."/"Blue." for familiar color prompts. App responses retain this
checkpoint's provenance in memory; inference does not change its weights.
Those are narrow smoke checks, not independent evaluation.

Thirty-six targeted training/mathematical/chat/HTTP tests pass, Python
compilation and Node syntax checks pass, and the production viewer builds.
The API was restarted with the new conversation-mode routing; the shared
browser shows the persistent learner, correct training-round metadata and
its experimental limits. Evidence, exact approved inputs, source archive,
initial/trained states and completed report persist under
`.data/neural-training/rounds/57f452944e2c4767b5586c8320bfbad7/`.
No indefinite training job remains running; the production API stays
available. Future approved rounds can resume using `--activate-after`
without another review of the unchanged teaching file. New training content
still requires the existing exact-content review gate.

### Immediate Feedback Training

The user clarified: submitting their own correction must train immediately,
and clicking **Relevant** must train immediately. **Unmarked replies never
train.** Merely clicking Needs correction does not train the incorrect
answer; submitting **Train correction now** approves the edited answer.
No extra scoring checkbox or second review is required. The exact saved
prompt and approved answer are hashed and recorded as the scope of consent,
not the entire transcript. A Relevant click uses the saved model answer,
never a replacement answer supplied by the request. This approval is a
user declaration, not independent factual or legal clearance.

[neural_correction.py](../neural_correction.py) loads the active permanent
checkpoint, validates saved prompt/response provenance, and trains actual
weights at rate .03: 75% target draws from the approved pair, 25% from
previously approved teaching/feedback for rehearsal. Only explicit approved
pairs join that rehearsal. For the same exact prompt, the latest submitted
answer supersedes conflicting older targets, including the base curriculum
when necessary. Future ordinary permanent rounds include these approved
pairs too, rather than silently reverting to the old target.

Missing characters and normalized identities are appended only from that
approved prompt/answer. Existing IDs and weights remain intact; new rows
reuse the existing unknown-row expansion helper. Architecture and fixed
normalization identity survive expansion. This can change output
normalization and model behavior; no universal retention claim is made.
The tiny decoder accepts single-line feedback answers of 1-95 characters
to leave space for the newline terminator. A longer answer or a multiline
correction is rejected explicitly instead of truncated or silently stored
as trained. The UI displays this limit.

Training starts within the feedback request. After each 50 updates, it
checks the exact prompt's free-running, newline-terminated answer.
The budget is 10 update-CPU seconds or 2,000 updates; generation,
independent replay and disk writes add latency. This is immediate initiation,
not an instantaneous or guaranteed successful fit. If it cannot learn within
that budget, it reports failure and does not change active/progress weights.
The failed candidate evidence is retained as **not learned**, not as success.
On success, every candidate weight and the raw approved answer must replay
exactly; the permanent and active selections advance, preserving rollback.
A busy training lock or an independently changed active selection causes an
explicit error, not lost concurrent updates.

The feedback record now contains `training_approved`, the exact approved-pair
hash and checkpoint evidence. The relevance learner validates the checkpoint
hash and approved pair before recording neural training success; this is
not training the retrieval model or the legacy ngram scorer instead.
Repeated clicks for an already-approved, still-correct pair are idempotent.
Saved weights contain the update and the pair remains available for future
rehearsal; this does not guarantee every paraphrase or later task will retain
the behavior. There is no runtime answer lookup or scripted response.

Tests verify actual changed weights and next-generation output for an edited
answer, training on the saved answer after Relevant, no training for unmarked
chat or a bare Needs correction click, exact replay, review/evidence hashes,
new-character expansion, idempotency, failure isolation and HTTP activation.
The production viewer reports training/in-progress/success/error rather than
"saved for review" for these approved conversation-model actions.
Memory capture remains connected; feedback does not erase original exchanges
or mutate their historical model provenance.

## Larger Balanced Conversation Meal

The user selected 64 original conversational teaching exchanges rather than
more Bible text, then approved the exact generated file
[neural-conversation-meal.json](../fixtures/neural-conversation-meal.json):
SHA256 `ce196cbef192c0f497afca080c25881136f21d748229061addb3152e7ce73461`.
The preserved file remains labeled pending approval as its historical
generation state; each experiment's review record contains the subsequent
user declaration. This is training permission, not legal clearance.
The old eight-example fixture and previous experiment evidence are unchanged.

[neural_conversation_meal.py](../neural_conversation_meal.py) generates
original synthetic teaching examples, **not** runtime scripted replies or
an answer lookup. It also runs the isolated size comparison and exact replay.
The generated teaching file is preserved; `--prepare` refuses to overwrite it.

A larger 256-exchange teaching set is prepared separately with
`python3 neural_conversation_meal.py --prepare --size 256`, written to
[neural-conversation-meal-256.json](../fixtures/neural-conversation-meal-256.json)
(SHA256 `3c8720173eda1b53b2f3cb7cd6916f6e12e2f65ebcb51bf19c2230965932c725`).
It keeps the original 64 rows unchanged, adds 12 new owner/object cases
with fresh names, and keeps the same 32 evaluation cases. It was approved
by exact hash and run from the same pinned layer-free base as the 64-exchange
comparison (`8887c490...`), not from the active trained model. Run
`c025c2ff0efa4808b07778cc32df43f1` replayed exactly. Result: at equal CPU,
the 256 set taught less (about 71-73% of training pairs) and scored lower on
the 32 held-out cases (12-15/32 for fresh weights, versus 19-20/32 with the
16-exchange control). Adding templated exchanges alone did not help. It is not
activated.

- **64 teaching exchanges:** eight skills, eight examples per skill, organized
  into 32 pairs with different correct answers. Skills: explicit owner, color,
  keeps/gives transfer, negation, cross-turn correction, missing information,
  cross-turn reference, and known/unknown location.
- **16-example control:** one complete pair from each skill, drawn from the
  64 teaching examples. This is a new balanced subset, **not** the historical
  eight-example diagnostic; comparisons with its 1/10 score are not equivalent.
- **32 evaluation exchanges:** 16 new entity/object combinations with
  familiar phrasing, plus 16 changed-wording examples that also change
  combinations. Both groups use familiar names/objects, not new-name spelling.
  Complete prompts are disjoint from teaching. These new developer-visible
  cases are not a locked independent benchmark; wording-group effects also
  include changed combinations and cannot isolate wording alone.

Each pair changes a fact or question so the answer must change. For example:
"Mira keeps the cup" -> "Mira."; "Mira gives the cup to Jon" -> "Jon.".
Skill counts and paired alternatives are balanced, but the complete answer
distribution is not uniform across different skills. Training samples
**answer tokens uniformly**, not skills, exchanges or pairs uniformly;
longer answers get more target draws. No contrastive loss or auxiliary
state-supervision head is introduced in this experiment.

### Frozen size-comparison protocol

Both previously tested designs, fresh scale-aware and inherited fixed
pre-tanh normalization, restart from new isolated copies. They do not
continue a previously successful seed's final weights. Their architecture,
64 feature slots, plain SGD and rate .1 remain fixed; that rate came from
the preceding teaching-only selections, not these new test results.

For each arm/seed, the 16- and 64-example runs have identical initial
weights and vocabulary/masks. The expanded character/identity vocabulary
uses only the 64 **teaching** exchanges; thus the small control knows their
input symbols but receives optimizer targets only from its 16 selected rows.
The comparison tests added teaching cases, not a simultaneous vocabulary or
capacity increase between sizes. There are 15,508 parameters in both,
versus 15,308 in the earlier eight-example experiment because this approved
teaching text adds different input identities. No new whole-word output
tokens are added; the original Acts lexicon remains frozen.

Seeds 17/117/217, two designs, two sizes: **12 trained states**.
Each gets 24 measured update-CPU seconds with teaching checks at 8 and 24.
Budget checks occur every 100 updates, with actual overshoot reported.
These are shared target-draw sequences within each same-size comparison,
but CPU cutoffs change update counts; data-size changes also change sampled
target identities. Seeds vary fresh initialization/target draws, not the
actual inherited parent's pretraining.

The full teaching set has 420 answer-token targets. Every teaching pair
has distinct first-context IDs, and the longest formatted teaching prompt
has 21 words, fitting within this bounded representation. The exact
token/mask audits find **no observed conflicting inputs**: 105 distinct
observable inputs for 105 small-control targets, 420 for 420 full-meal
targets. Their empirical deterministic token ceilings are 100%, so these
particular teaching targets are not blocked by identical-input collisions.
This does not prove sufficient capacity, successful optimization, semantic
understanding or any bound on unseen answers.

All 12 final endpoints are saved before any test evaluation. No rate,
checkpoint or seed is selected using tests. Generation/scoring reuses the
existing greedy 96-character decoder and newline termination requirement,
including complete raw continuations. Report both exact answers and **pairs
where both members are correct**, plus repeated identical pair answers.
A constant common answer cannot pass a distinct-answer pair.

Update CPU excludes measurements, evaluation, setup, persistence and
independent replay. Teaching checkpoints report measurement CPU separately;
final test CPU is also reported. Replay uses recorded update counts and
checks every full weight set and non-timing milestone, not CPU cutoff
reproducibility. Active chat and original comparison files remain
checksum-protected. No claim of original-corpus retention after either
reset or normalization is made. Memory capture remains connected.

```sh
# Prepare once, then review the exact file before authorizing training.
python3 neural_conversation_meal.py --prepare
python3 neural_conversation_meal.py --acknowledge-review \
  --reviewed-sha256 ce196cbef192c0f497afca080c25881136f21d748229061addb3152e7ce73461
python3 neural_conversation_meal.py --validate-run RUN_ID
python3 -m unittest test_neural_conversation_meal test_neural_plasticity \
  test_neural_experiment test_neural_conversation_context \
  test_neural_conversation_trial test_neural_chat test_neural_base_comparison
```

Thirty-four targeted tests pass, including curriculum counts, pair validity,
prompt disjointness, teaching-only vocabulary, identical initial models
across data sizes, constant-answer pair rejection, exact review gating,
test access only after all endpoints are saved, miniature full-state replay
and existing mathematical/chat regression tests. Python compilation passes.
No new dependency, pretrained model, external dataset or copied code is used.

### Results: run ee7f3c1be14f477a82e2db03ae72ff71

| Design | Teaching size | Exact teaching, seeds 17/117/217 | Exact test, same seeds | Both-correct test pairs | Mean update CPU |
|---|---:|---|---|---|---:|
| Fresh scaled | 16 | 16/16, 16/16, 16/16 | 20/32, 20/32, 19/32 | 9/16, 10/16, 9/16 | 24.23s |
| Fresh scaled | 64 | 59/64, 58/64, 59/64 | 21/32, 18/32, 15/32 | 9/16, 8/16, 7/16 | 24.35s |
| Inherited normalized | 16 | 10/16, 11/16, 10/16 | 14/32, 15/32, 14/32 | 3/16, 3/16, 3/16 | 24.38s |
| Inherited normalized | 64 | 24/64, 28/64, 31/64 | 7/32, 9/32, 12/32 | 0/16, 1/16, 1/16 | 24.27s |

Case/whitespace-normalized test scores are identical to exact scores.
Each seed uses the **same 32 evaluation cases**, not a new independent
32-case sample. No seed is selected for deployment.

The fresh design can now fit a complete balanced 16-example set exactly
and roughly 92% of the 64-example set. This is a useful tiny-task fitting
milestone, but the new curriculum, answers, vocabulary and evaluation
differ from the preceding eight/ten experiment. **Do not compare 21/32
with its old 1/10 as an isolated training-method improvement.**

At matched CPU, the bigger meal does **not** improve overall evaluation:
fresh mean exact accuracy is 61.46% for 16 examples versus 56.25% for 64
(-8.47% relative); normalized 44.79% versus 29.17% (-34.88% relative).
Both-correct pair counts also decline. No overall 30% gain or
30%-less-time-to-quality result is supported.

The evaluation groups reveal a limitation:

- Fresh/16 gets all **16/16 familiar-phrasing combination cases** correct
  in every seed; fresh/64 gets 15/16, 13/16 and 11/16.
- Fresh/16 changed-wording scores are 4/16, 4/16, 3/16;
  fresh/64 scores 6/16, 5/16, 4/16. This modest subgroup improvement does
  not offset the overall regression and is not a separate confirmed win.
- Direct inspection finds no unknown input word identities in the
  combination group, but **current, holder, its, name, person, tell**
  all map to OTHER in the wording group. Test words were deliberately
  not added to training vocabulary. This documents missing distinctions,
  not proof that unknown words explain every error.

For fresh/64 seed 17, the model correctly changes "Mira." to "Jon." when
the book scenario switches keeps/gives, answers both
red/blue variants correctly, and replies "I do not know." when color is
missing. On changed wording, "Name the person with the cup" instead
elicits "Red.", and a transfer can produce "Lena." or "Ba.".
It has learned useful narrow patterns, not robust conversational meaning.
Simple synthetic pairs do not establish multi-hop reasoning or broad
intelligence; there is no runtime template lookup.

The larger set receives about four times fewer sampled updates per target
at the same compute: fresh/16 averages 5,000 updates / 105 targets
(47.62 draws per target), fresh/64 4,967 / 420 (11.83).
Normalized equivalents are 39.68 versus 10.08. This is expected in a
compute-matched data-size comparison, not evidence that more data is
inherently harmful. Whether a longer budget helps remains untested;
these test results were not used to tune another run.

All 12 fits use 291.69 update-CPU seconds, teaching measurements 188.87,
and final evaluation 54.05. Setup, persistence and exact replay add cost.
The next targeted experiment should review more varied question wording
and sufficient exposure, rather than only adding more entity substitutions
or declaring conversation solved. Keep architecture and held-out wording
separate; new training bytes require another exact content-hash approval.
Concrete user contribution: write alternate ways of asking the same
possession/color question and verify which facts should change its answer.

All **12 full trained states**, complete evaluation continuations, paired
scores and every non-timing teaching milestone independently replay exactly.
All seven active working/original-control JSON files remain hash-identical.
Evidence, approved inputs, pre-test endpoints, source archives, analysis
and `validation.json` persist under
`.data/neural-conversation-meal/ee7f3c1be14f477a82e2db03ae72ff71/`.
No model was activated; results and development decisions are captured in
connected transcript/graph memory, separately from neural training approval.

## Initialization And Plasticity Diagnostic

The user approved the research recommendation ("let's do it"). The first
stage isolates initialization/normalization, not new teaching data or a
combined architecture/data change. [neural_plasticity.py](../neural_plasticity.py)
uses the existing eight approved exchanges, 97 answer targets, training-only
vocabulary and 64-slot fact-preserving context. The same ten already-visible
tests remain exploratory evaluation, not a fresh generalization benchmark.
No historical chat or research-paper text becomes training data.

Three arms have the same 15,308 parameters, output masks and plain SGD:

- **inherited:** copy the actual pinned working weights, with the established
  vocabulary/context extension.
- **fresh_scaled:** reset all learned weights. Embedding initialization has
  variance 1/8; hidden/output uniform bounds are
  `sqrt(6 / (fan_in + fan_out))`; biases are zero. Resetting and choosing these
  scales are one bundled intervention, not proof that scaling alone helps.
  Independent new identity embeddings also break the inherited unknown-row
  equality. No claim that this preserves previously learned knowledge.
- **inherited_normalized:** retain parameter values but normalize hidden
  preactivations before tanh using per-example mean/variance and epsilon
  `1e-5`. This is fixed unit-gain, zero-offset normalization, **not** full
  learned-affine LayerNorm. It changes the model function immediately and
  is not a behavior-preserving conversion. Backprop differentiates through
  both mean and variance; the shared core keeps ordinary tanh as its default.
  The experimental backward pass recomputes normalization statistics, and
  its CPU cost is included.

The methods are motivated by [initialization research](https://proceedings.mlr.press/v9/glorot10a.html)
and [Layer Normalization](https://arxiv.org/abs/1607.06450). Implementations
are original standard-library calculations, not copied external code.
No new dependency, pretrained weights or external dataset is introduced.
This is not a license or commercial-rights clearance.

```sh
python3 neural_plasticity.py --acknowledge-review \
  --reviewed-sha256 e222bcf6b88394bbe801fa9fd146df14658d6d06289a5f2386d1660130f7f46b
python3 neural_plasticity.py --validate-run RUN_ID
python3 -m unittest test_neural_plasticity test_neural_experiment \
  test_neural_conversation_context test_neural_conversation_trial \
  test_neural_chat test_neural_base_comparison
```

### Frozen measurement protocol

Three seeds (17/117/217), two rates (.01/.1), 18 trained copies. Each rate
gets 12 seconds of measured update CPU, with teaching-only checkpoints at
3/6/12 seconds. Budget checks follow each 100 updates, so actual overshoot
is reported, not hidden. Initial inherited weights are identical across
seeds; seeds vary target draws and fresh initialization, not parent
pretraining. Faster arms see more updates; this is a CPU comparison, not
equal target exposures.

Each arm/seed gets the same two-rate search allocation. The final teaching
token loss selects its rate; all nine selections are saved **before any
test evaluation**. All three teaching checkpoints and both rates remain
visible. There is no test-guided selection or automatic deployment.
Metrics, gradient diagnostics, evaluation and persistence are outside the
update budget; measurement/evaluation CPU is reported separately. Total
experiment cost also includes an independent replay of all 18 models.
Replay uses recorded update counts, since CPU cutoffs are not deterministic.

Hidden saturation, mean tanh derivative and hidden-weight gradient norms
are measured on actual teaching examples. They diagnose learning behavior,
not intelligence. Even successful tiny-target fitting would not prove
conversation or generalization. The existing representation still has
some conflicting token inputs. No original-corpus retention benchmark is
included, so neither reset nor normalization can claim to retain knowledge.
Active chat, original comparison checkpoints and connected memory remain
separate from these experimental weights.

### Results: run 29395ea2dc794a499c608b489959221f

| Arm | Selected rate, all seeds | Teaching correct per seed | Test correct per seed | Mean teaching loss | Mean selected update CPU |
|---|---:|---|---|---:|---:|
| Inherited | .01 | 0/8, 0/8, 0/8 | 0/10, 0/10, 0/10 | 1.807 | 12.29s |
| Fresh scaled | .1 | 4/8, 7/8, 5/8 | 1/10, 1/10, 1/10 | .216 | 12.26s |
| Inherited normalized | .1 | 3/8, 4/8, 4/8 | 1/10, 1/10, 1/10 | .492 | 12.36s |

Counts are exact, newline-terminated answers. Case/whitespace normalization
does not improve these test scores. Each seed reuses the same eight/ten
examples; these are not 24/30 independent examples. All nine successful
test answers are the **same blue-box case**. The unchanged historical short
conversation trial also scored 1/10, so this is not an advance over its
best observed exact test score.

Both interventions improve teaching-example fitting at similar update CPU.
They do **not** establish understanding or useful conversation. The inherited
normalized seed-17 model answers "Blue." to both red and blue questions,
"Jed." for a transfer, and "No." when Jon keeps the box. Fresh models also
produce wrong names, malformed text and unterminated replies.
The best fresh teaching run gets 7/8, not perfect training or generalization.

Hidden activations above absolute .99:

| Arm | Initial | Final, selected models (mean) |
|---|---:|---:|
| Inherited | 91.95% | 97.66% |
| Fresh scaled | 0% | 30.15% |
| Inherited normalized | 0% | .17% |

Normalization demonstrably reduces saturation and improves fitting here.
It does not prove saturation was the only obstacle, or that retained
parameters still preserve earlier knowledge. Fresh initialization resets
all prior learning; its better fit cannot be credited to scaling alone.
Normalization's backward pass and different update counts are included in
the update CPU, not hidden behind equal-step comparisons.

Both rates consume their own budget: the selected-model column is **not**
the total search cost. All 18 training copies consume 222.75 update-CPU
seconds, teaching measurements another 108.64 seconds, and the nine final
test assessments 12.55 seconds. Setup, initial diagnostics, persistence and
independent replay add further cost outside those totals.
Final budget overshoot is at most one 100-update batch; raw times, updates,
measurement CPU and final test CPU are saved for all runs.
The inherited test baseline is zero, so a relative test gain is
undefined, not a 30% win. No 30%-less-time-to-quality claim is supported.

The next step is a reviewed, balanced counterfactual teaching curriculum
with held-out combinations and wording, using fresh/normalized fitting
controls rather than blindly extending the inherited run. Training-only
owner/color/unknown supervision is a separate later ablation, not something
already implemented or validated. A concrete user contribution is to write
paired "keeps/gives" or "red/blue" exchanges and review their exact answers;
new teaching bytes need a new content-hashed approval. Do not reuse the
already-inspected ten cases as a fresh locked benchmark.

### Mathematical and regression checks

Thirty targeted tests pass. Finite differences check normalization gradients
through every parameter group, including repeated embeddings, output masks
and reference conditioning. Zero-variance normalization remains finite.
A conflict-free single-target smoke test fits below .02 cross-entropy;
that is a backprop sanity check, not conversational learning.
Tests also verify ordinary tanh equations remain exact, experimental
architecture survives restoration, initialization does not mutate its
parent, CPU-budgeted updates replay, the content-hash gate rejects invalid
approval, and rate selections precede all test access.
Python compilation succeeds. Pylance syntax requests timed out; no claim
of a completed editor-analysis check is made for this experiment.
All **18 full trained weight sets**, all non-timing teaching milestones and
all **nine selected test assessments** independently replay exactly.
All seven pinned working/control JSON files remain hash-identical.
Evidence, approved inputs, pre-test selections, initial/final weights,
hash-verified source copies, derived analysis and `validation.json` remain
under `.data/neural-plasticity/29395ea2dc794a499c608b489959221f/`.
No model was activated; findings are captured through the connected
transcript/graph workflow, not fed back into neural weights.

## Longer-Context Conversation Ablation

The user approved the proposed fact-preserving context experiment ("go for it").
[neural_conversation_context.py](../neural_conversation_context.py) implements
an isolated extension, not a replacement for active chat or the original control.
Run `1fe67c6c59064ddab1e088843cbe1c60` continues the same pinned actual parent
on the same eight approved teaching exchanges. The ten developer-visible test
answers remain evaluation only. Approval records retain the unchanged file hash
`e222bcf6b88394bbe801fa9fd146df14658d6d06289a5f2386d1660130f7f46b`.

```sh
python3 neural_conversation_context.py --acknowledge-review \
  --reviewed-sha256 e222bcf6b88394bbe801fa9fd146df14658d6d06289a5f2386d1660130f7f46b
```

### Three controlled arms

1. **short_original:** reproduces the previous conversation trial, including
   the one training-only apostrophe addition.
2. **short_identities:** appends 21 normalized identities found in teaching
   prompts, answers and ordinary role labels. New embedding/output rows copy
   the old unknown rows; existing IDs and parameters remain unchanged.
   The new identities are input features, never permitted output actions.
   No direct word payload is added: names still need character spelling.
3. **long_identities:** exactly the same vocabulary/output masks as arm 2,
   with 60 older word/gap feature slots prepended to the original four local
   features. Completed-word contexts cover up to 32 words; inside a word,
   the last four retain the original last-word/gap/two-character features.
   Older features are right-aligned and boundary-padded, with no future text.
   Only 16 hidden units remain. Older hidden connections initialize to zero;
   the existing hidden connections are placed at the tail unchanged.

The two identity-expanded arms have **exactly equal initial output distributions**
for tested full/partial prefixes. New older connections receive real gradients;
they are not frozen or manually injected facts. Structural tests confirm red-
and blue-box prompts now have distinct long contexts. Unknown names absent
from teaching (e.g. Tavi) still share OTHER, and context remains bounded.
This is neither full-memory access nor general identity resolution.

Word output vocabulary, causal masks, 97 answer targets, .1 learning rate,
seed-17 draw schedule, 500/2,000/5,000 checkpoints, newline termination and
greedy96 scoring are identical across arms. No architecture-specific gradients
are introduced: the shared standard-library MLP handles the longer input.
Same updates are **not** equal compute or capacity. All experiments use one
actual parent, not independent-seed replicates. No corpus replay/retention
measurement, test-guided checkpoint selection or activation.

### Fixed-rate results

| Arm | Parameters | Final teaching loss | Exact teaching | Exact test | Update CPU |
|---|---:|---:|---:|---:|---:|
| Short original | 7,103 | .600 | 1/8 | 1/10 | 8.67s |
| Short + identities | 7,628 | .506 | 1/8 | 1/10 | 9.25s |
| Long + identities | 15,308 | 3.016 | 0/8 | 0/10 | 25.19s |

The short original reproduces previous full weights and test rows exactly.
Identity distinctions improve the empirical teaching token-accuracy ceiling
from 87.63% to 95.88%; long context raises it to 98.97%. These ceilings count
conflicting observed token inputs/masks only, not answer quality or intelligence.
The longer representation removes most observed collisions but **does not**
make the optimizer learn a useful function.

The long .1 model produces repeated "roror..." and fails to terminate all ten
test replies. Inspection of actual teaching-target hidden values finds:

- Short-original initialization: 95.10% have absolute value above .99.
- Identity-expanded initialization (both short and long): 91.95%.
- Final short+identities: 95.49%.
- Final long .1: **100%**.

This inherited `tanh` saturation makes hidden derivatives small and is a
plausible optimization obstacle, not proof it is the only cause. Increased
input width also changes gradient/update behavior; a successful initialization/
scale strategy has not been demonstrated by this trial.

### Explicit lower-rate follow-up

After observing the .1 failure, a separately recorded follow-up trained long
models at .03 and .01 from the identical saved long initial state, same 5,000
updates. It inspected only teaching metrics during selection and chose the
lowest final teaching loss **before** evaluating that model's test replies.
The original fixed-rate report/checkpoints remain unchanged.

```sh
python3 neural_conversation_context.py --acknowledge-review \
  --reviewed-sha256 e222bcf6b88394bbe801fa9fd146df14658d6d06289a5f2386d1660130f7f46b \
  --lower-rate-screen 1fe67c6c59064ddab1e088843cbe1c60
```

This command intentionally refuses to overwrite an existing screen.
The screen protocol, pre-test selection and final assessment are separate files.
The test set was already developer-visible and inspected in earlier trials:
this is development diagnosis, not an independent locked test.

| Long rate | Final teaching loss | Exact teaching | Update CPU |
|---|---:|---:|---:|
| .03 | 2.512 | 0/8 | 23.36s |
| .01 (selected by teaching loss) | 1.719 | 0/8 | 24.64s |

Selected .01 test: **0/10 exact, 1/10 case/whitespace-normalized**, only
4/10 newline-terminated. "blue." answers the blue-box reference correctly
ignoring case, but also incorrectly answers transfer/correction questions.
Other replies are blank spaces or "box not not not .". These are failures
of useful generation, not merely strict capitalization scoring.

### Outcome and verification

Longer context successfully preserves earlier facts **as distinct inputs**;
useful learned conversation did not improve. Do not deploy the larger model
or infer that more compute alone is the answer. Next investigate initialization,
activation scale/gradient flow and tiny-example fitting while retaining the
fact-preserving inputs; compare against the inherited checkpoint rather than
discard it. More approved teaching data and independent evaluation would
still be necessary after that, not a replacement for fitting checks.

All five full trained states (three main arms and two lower-rate follow-ups)
were independently replayed. Every weight, full continuation and non-timing
milestone metric matches; the selected follow-up test also matches. Saved
checkpoints restore correctly. Nineteen targeted context/training/base/chat
tests pass, including equality of initial short/long distributions, actual
learning of older connections, causal features, training-only vocabulary,
nonfinite-state rejection, and selection before the single follow-up test.
Editor diagnostics report no errors. Active/chat original-control files remain
hash-identical. Memory capture and findings remain connected.

Evidence: `.data/neural-conversation-context/1fe67c6c59064ddab1e088843cbe1c60/`
contains approval, exact inputs, parent/initial/final states, raw report,
lower-rate protocol/selection/results and independent `validation.json`.
No model was activated and no extra training corpus was introduced.

## Visual Base Model Walkthrough

Start at the **Base Model Lab** tab (`/?view=base-lab`) before the historical
layer experiments below. This is a separate, base-only teaching space, not a
new active model and not the older extraction lab.

Each stage presents three connected views:

1. A diagram plus actual numbers from the calculation.
2. A small conceptual code sketch alongside the visualization.
3. The actual shared Python method, with source line numbers, highlighted
   relevant statements and junior-engineer explanations beside its lines.
   The construction view includes the module prelude from source line 1.

The code is `NeuralModel` in `neural_experiment.py`; `neural_base_lab.py`
creates a disposable instance and traces it. The source is inspected at
runtime rather than copied into the viewer, and its SHA256 accompanies the
trace. Teaching annotations are separate from executable source. Some real
methods contain optional reference/prior branches: annotations explicitly
identify those as unused here. There are no L0/L1 recognizers, placement
counts, reference vectors or pretrained weights in this walkthrough.

### Follow one example through all eleven stages

- **Alphabet:** turn the entered characters into distinct IDs. Space counts.
  Boundary ID 0 pads the start; unknown ID 1 represents missing alphabet
  entries. Real characters start at ID 2.
- **Construction:** a local seed-17 generator initializes five ordinary
  Python lists: embeddings, hidden connections, hidden biases, output
  connections and output biases. Inspect every initial value through paging.
  Random initial numbers are not knowledge.
- **Example:** show the selected source character and four previous IDs.
  Early contexts contain boundary padding. The target is known from the
  input text, not the model's previous guess (teacher forcing).
- **Embeddings:** look up eight trainable coordinates per context token;
  concatenate all four rows into 32 inputs. A vector is just a list of numbers.
- **Hidden layer:** each of 16 neurons adds a bias to 32 weighted inputs and
  applies `tanh`. The grid shows all activations; expand neuron 0's exact
  products, sum and activation. Activations are temporary; weights persist.
- **Output scores:** multiply hidden activations by output weights and add
  output biases. Inspect all 16 products for the target output. These scores,
  called logits, are not probabilities.
- **Probabilities:** stable softmax exponentiates scores and divides by their
  sum. Show every output, including reserved tokens, and mark the target and
  most likely slot. Training includes the reserved slots; ordinary character
  sampling excludes IDs 0 and 1.
- **Loss:** `-log(probability[target])` measures error continuously. It can
  improve even when the most likely character remains wrong.
- **Gradients:** backpropagation calculates derivatives for all five groups.
  Output bias gradients are probability minus the one-hot correct target.
  Hidden errors include tanh's derivative; repeated context tokens accumulate
  into their shared embedding row.
- **Update:** every parameter follows `after = before - rate * gradient`.
  The inspector shows exact before, derivative, signed change and after
  values, with counts including unchanged entries. Gradient zero means no
  change; the learning rate controls update size.
- **Recheck:** recompute the same example, then average loss over the whole
  entered string. Improving one example need not improve every other one.
  The loss curve has a labeled scaled axis and exact initial/current values;
  it measures this tiny training string, not generalization.

**Back/Next** or stage buttons reveal an already calculated update, not
successive hidden calls to train the live model. **Train next character
example** replays the seeded prefix with one more actual SGD step, cycling
through source positions. It is one character target per update, not one
sentence per update. **Build / reset** applies entered text/rate and starts
again at update 1. Initialization always shows the original untrained
parameters; forward/backward views show the selected update's values.

Requests are deliberately bounded to 32 updates, 128 printable ASCII
characters, 24 distinct characters and a positive learning rate no greater
than 1. The teaching state is disposable and not persisted. No transcript or
active chat file is loaded or saved by the trace. Reopening the tab resets
the teaching exercise; the real chat checkpoint is unaffected.

### What the checks prove

Tests replay the real core updates and compare every parameter, gradient,
probability, loss-curve point and displayed source line. Numerical finite
differences check the largest gradient coordinate in each parameter group.
HTTP tests verify explicit input errors and unchanged transcript bytes.
Live checks verify unchanged active selection, checkpoint and transcript
hashes. Browser checks cover all eleven stages, weighted products,
Back/Next, another update, custom text/reset, errors and narrow layout.

These establish that the walkthrough shows real calculations. They do not
establish that a small training-loss drop means understanding or that the
active language model improves.

Try this first: keep `ab ab ab `, visit the eleven stages, train the next
example, and compare the target loss with the whole-string loss. Then reset
with a smaller learning rate and inspect how the exact weight changes differ.

## Quick Multi-Context Comparison

The user requested a quick test of short and longer contexts together.
`neural_context_mix.py` compares fresh base-only models without changing
active chat or the visual teaching lab:

- A standard context-4 MLP.
- A standard context-16 MLP.
- Independent context-4 and context-16 MLPs whose probabilities are combined
  by one learned global sigmoid gate, initially a 50/50 blend.
- A single context-31 MLP, approximately matching the mixture's parameters.

The gate is a global preference, not a per-input learned selector or attention.
For long-branch weight `w`, target probability is
`p = (1-w)*p_short + w*p_long`. Exact mixture-loss gradients weight branch
updates by their target responsibilities `(1-w)*p_short/p` and
`w*p_long/p`. The gate's derivative is
`w*(1-w)*(p_short-p_long)/p`. Both branch gradients and the gate are
computed from the pre-update predictions. Finite-difference tests verify
representative coordinates in every weight group of both branches and the
gate. The original shared base implementation is unchanged.

Run `d9c11456613f4c0aaa3e3685fe1026ec` uses the existing rights/hash-validated
Acts 1 training text and Acts 2 reused validation text. Acts 3 is not loaded.
Vocabulary derives only from training. Seeds 17/117/217 each use 1,500
identical sampled target positions per arm at rate .03, with measurements
at 750/1,500 updates. Each branch has its own seeded initialization; these
are not identical initial weights across architectures. Evaluation uses
512 fixed evenly spaced character targets per split, not word accuracy.
Earlier experiments already used Acts 2, so this is not untouched evidence.

| Three-seed mean after 1,500 targets | Parameters | Validation loss (lower better) | Character accuracy | Training CPU |
|---|---:|---:|---:|---:|
| Context 4 | 1,978 | 3.06480 | 17.51% | .746s |
| Context 16 | 3,514 | 3.08575 | 16.21% | 1.491s |
| Mixed 4+16 | 5,493 | 3.13436 | 19.53% | 2.958s |
| Context 31 capacity control | 5,434 | 3.11671 | 14.00% | 2.508s |

Mixture loss minus context-4 loss is +.14983/+.03430/+.02455 nats by
seed; minus context-16 is +.07445/+.03735/+.03405. The mixture has worse
loss in every seed against either smaller model. Against the capacity
control, differences are +.04705/-.00199/+.00791 (worse in two of three).
Learned long-context weights are .662/.460/.575. The gate changed, but that
does not prove useful routing or justify choosing a favorable seed.

Higher top-character accuracy and worse correct-target loss can coexist:
ranking the target first slightly more often does not guarantee assigning
good probability to all targets. The primary loss result does not support a
benefit in this short run. Training CPU excludes setup/evaluation and includes
both branches plus their extra forward calculations in the mixture. Equal
target exposures are not equal operations, and approximate parameter matching
does not match topology, initialization or optimization. Smaller responsibility
updates are inherent in this mixture's gradient; longer training or a different
combination may behave differently, but neither was established here.

Reports retain protocol, inputs, draw hashes, core/code hashes, per-seed
histories, paired loss differences and all twelve final checkpoints under
`.data/neural-context-mix/`. All saved final training/validation metrics
were reproduced from restored weights; source/code hashes verified.
Thirteen targeted regression tests passed, including finite-difference and
protocol/approval checks. No active model or latest-report pointer changed.
This tests one simple implementation, not the general usefulness of
multi-scale models, convergence, language understanding or generalization.

### Requested 5+12 versus 4-only variant

Run `66afdebcd2824735b2a44bd9aed5c233` uses `--contexts 5 12 --pair-only`,
preserving all default 4+16 behavior and earlier results. The same three
seeds, 1,500 target draws at rate .03, and 512 reused Acts 2 character checks
apply. The context-4 baseline reproduces the previous run's metrics exactly.

| Three-seed mean | Parameters | Validation loss | Character accuracy | Training CPU |
|---|---:|---:|---:|---:|
| Context 4 | 1,978 | 3.06480 | 17.51% | .716s |
| Mixed 5+12 | 5,109 | 3.13958 | 19.27% | 2.601s |

Mixture loss minus baseline is +.16320/+.04565/+.01547 by seed: worse in all
three. As with 4+16, better mean top-character accuracy coexists with worse
correct-target probability loss. CPU is ~3.6x baseline and capacity differs;
this pair-only run does not include a parameter-matched control. No clear
overall advantage demonstrated; no convergence or generalization claim.
All six saved checkpoint training/validation evaluations reproduced exactly.
Three focused tests passed, including requested window sizes, branch update,
input validation, protocol and the original numerical-gradient checks.
Active chat and the teaching lab remain unchanged.

### Triple 3+5+7 versus 4

Requested run `3bec0fc1df4f46268b80d7279ead98df` retains the same three seeds,
1,500 target exposures per arm at .03 and 512 reused Acts 2 character checks.
Three independent MLPs combine probabilities with three learned global
softmax logits, initialized to equal weights (one redundant gate degree of
freedom). Each branch receives its mixture-loss responsibility-scaled update;
gate gradient is blend weight minus target responsibility. This is not
input-dependent attention. Existing two-branch behavior remains unchanged.

| Three-seed mean | Parameters | Validation loss | Character accuracy | Training CPU |
|---|---:|---:|---:|---:|
| Context 4 | 1,978 | 3.06480 | 17.51% | .726s |
| Mixed 3+5+7 | 6,321 | 3.17557 | 20.64% | 3.118s |

Loss differences mixture minus baseline are +.20680/+.05159/+.07390: worse
in all seeds despite better mean character accuracy. Training CPU is ~4.3x
baseline; target exposure is equal, parameters and operations are not.
Final global weights for contexts 3/5/7 are .759/.107/.133, .085/.815/.100,
and .117/.303/.581 by seed. Different seeds favor different branches;
this is not evidence that a specific window is best.

All six final checkpoints were independently retrained from their seeds and
draws to reproduce every weight exactly, plus training/validation metrics.
Four focused tests passed, including finite differences for all three gates
and selected coordinates in every parameter group of every branch.
The short reused-development result does not show convergence or a general
multi-context advantage. Active chat and the teaching lab remain unchanged.

## Word Identity And Punctuation Screening

The user requested a brief test of recognizing word spelling independently
of surrounding punctuation while preserving punctuation as a structural signal.
`neural_word_context.py` uses the shared base MLP and a different prediction
task: choose the normalized next word at a known word boundary. This is
not the earlier next-character experiment or a new chat generator.

All three arms have the same 4,153 parameters, vocabulary, seed initialization,
output mask and word-target draw order. Embedding size 8, hidden size 16,
four input slots, rate .03, three seeds and 1,500 target exposures per arm.
The arms are raw four-character context; two prior word IDs with ignored-gap
slots; and two prior word IDs with their following punctuation-gap IDs.
Only past text is used. A gap ID records its exact non-whitespace punctuation
sequence plus newline/inline status; vocabulary is training-derived and unseen
gaps use unknown. Case and exact original whitespace remain in lossless source
records, not fully in the neural features. Internal apostrophes/hyphens remain
inside word identity; curly apostrophes normalize for lookup, but `can't`
and `cant` remain distinct. The English-oriented parser is not a universal
tokenizer. No output punctuation generation or grammar understanding is tested.

The training-only 64 most frequent words form the candidate list. All other
words map to OTHER in context. Initial run
`f5a091dcf6b043838ae6f3b6a2c0544b` also allowed OTHER as a target; all models
predicted that bucket throughout the 512 validation checks. Its 51.56%
all-target accuracy was just unknown-bucket prediction: known-word accuracy
was zero. Preserve that unsuccessful screening rather than present it as
useful word prediction.

The explicitly narrowed follow-up
`7b65fe2ad4e4493c92c19ca38d8dad78` excludes OTHER from training targets and
the output mask (`--known-targets-only`). There are 403 eligible training
word occurrences. All arms draw the same 1,500 eligible targets, with
replacement, per seed. Of 512 fixed evenly spaced Acts 2 validation word
positions, only 248 have a known target and are scored; 264 are excluded.
Filtering target eligibility is an oracle restriction, so this is a closed-set
mechanism diagnostic, not end-to-end next-word performance. Follow-up scope
was changed after observing the degenerate result; it is not preregistered
untouched evidence.

| Three-seed mean on 248 known-word positions | Word accuracy | Target loss (lower better) | Training CPU |
|---|---:|---:|---:|
| Raw four characters | 12.63% | 3.76434 | 1.476s |
| Two word identities, gaps ignored | 13.98% | 3.75880 | 1.471s |
| Two words plus punctuation-gap signals | 15.46% | 3.74443 | 1.482s |

Punctuation model minus raw loss is -.03250/-.03972/+.01248 by seed:
better in two of three, not consistent against raw. Against words-only,
loss is better in all three (-.02543/-.00441/-.01329). There are just 39
scored positions immediately following punctuation; mean accuracy there is
9.40% raw, 7.69% words-only, 11.97% words+punctuation. Supports are small
and reused across seeds; do not treat them as independent new data.

The mean accuracy gain versus raw is 2.82 percentage points (~22% relative),
not 2x quality. Mean loss reduction is only ~.53%. Training CPU is essentially
unchanged in this matched single-network setup. Combined feature-construction
setup took .0125 CPU seconds; source loading, vocabulary construction,
evaluation and historical development are not charged to training CPU.
Word IDs supply longer aligned lexical context, so gains over raw cannot
be attributed solely to punctuation. The words-only ablation provides
narrower evidence that punctuation signals can help this word predictor.

This is a small promising direction for a cheaper representation, not a
demonstrated compute-efficiency threshold, grammar gain or generalization win.
Acts 1/2 is reused authorized development material; no new source or active
checkpoint is loaded into the learner. Raw/extracted source records,
input/output features, masks, protocol/code/source hashes, before/after
metrics and all checkpoints persist in `.data/neural-word-context/`.
Four focused tests cover lossless reconstruction, apostrophes/hyphens,
no target leakage, training-only vocabulary, oracle exclusions and exact
small-fixture replay. All nine final follow-up models were independently
replayed weight-for-weight and evaluations reproduced. Chat/lab unchanged.

### Larger Saved-Corpus Word Screening

The user clarified: reuse the slightly larger earlier text and its baseline,
not whole Acts/Corinthians or a newly selected split. `--larger` loads the
rights/hash-validated staged Acts 6-12 (30,905 characters) and Acts 13-14
(9,730 characters), then checks exact equality with saved inputs and source
hashes from benchmark `6045eb140d8e47b1a5c6890199251039`. Mismatch fails
explicitly. That benchmark's full-character generation task cannot serve as
a numerical baseline for this closed-set word task, so all matched controls
are retrained on the same passages.

Run `0baa5fd9d2384a3784538b85703bc371` retains the 64-word candidate limit,
three seeds and rate .03, increasing to 5,000 identical eligible word-target
draws per arm/seed (not the old benchmark's five seeds, epochs or rate).
There are 3,321 eligible training occurrences. All arms have 4,753 parameters
and identical initial weights per seed. The 512 fixed evaluation positions
contain 263 eligible known-word targets; 249 others are excluded. This remains
an oracle-filtered, reused-development diagnostic, not full-vocabulary
generation, grammar evaluation or untouched generalization evidence.

| Three-seed mean on 263 known targets | Word accuracy | Target loss | Training CPU |
|---|---:|---:|---:|
| Raw4 word-task control | 18.38% | 3.39667 | 6.549s |
| Two words, gaps ignored | 19.01% | 3.44055 | 5.798s |
| Two words plus punctuation gaps | 20.91% | 3.30765 | 5.837s |

Punctuation improves both accuracy and loss against both controls in every
seed. Against raw4: +2.53 percentage points accuracy (~13.8% relative), ~2.62%
lower loss. It does not double quality or halve error. On the 54 eligible
positions after punctuation, accuracy is 15.43/3.70/17.28%; supports are
small and repeated across seeds, not independent samples.

Feature setup for all arms combined is .140 CPU seconds. Training-only CPU
excludes source/vocabulary setup and evaluation. The raw4 first seed's CPU
was higher than its other seeds; equal network shapes/update counts are
stronger evidence of comparable work than claiming an architecture speedup
from these short timing differences. Relative to words-only, punctuation
adds ~.7% measured training CPU, not another neural branch.

All nine final models were independently replayed from seeds and draws to
match every saved weight and evaluation exactly. Source/code hashes verified;
five focused tests include archived-corpus equality and explicit mismatch
rejection. This larger check strengthens the small representation signal,
but does not meet a 2x benefit threshold or justify changing active chat.

## Full-Generation Five-Arm Comparison

The closed-word screen above is not a full language generator. The separate
offline runner [neural_base_comparison.py](../neural_base_comparison.py) now
compares **Original Base, New Base, New+L0, New+L0+L1, New+L0+L1+L2**:

```bash
python3 neural_base_comparison.py --acknowledge-review --include-retired-layers
```

It verifies the exact saved Acts 6-12 / 13-14 texts against archived run
`6045eb140d8e47b1a5c6890199251039`. Every arm starts fresh, trains complete
source passes 1/2/4 at rate .1 with seeds 17/117/217, and generates on the same
256 archived word-start prefixes. Greedy continuations contain 64 characters;
one common apostrophe/hyphen-aware parser scores the first complete word,
including exact case. Every probe counts, including words outside the 64
common-word identities. Punctuation is generated and retained, but this
word-accuracy measurement does **not** establish punctuation/grammar accuracy.
These extensively reused development passages are not a fresh test.

The original is our actual context-4 character MLP (2,103 parameters on this
61-character training alphabet), not the raw4 word-classification control.
The experimental new base reuses the same mathematical MLP with different
inputs/outputs: two preceding word identities and punctuation gaps; inside
an unfinished fallback word, last-word/gap/two-character features. Its top64
identities come only from training. Exact case-preserving surface forms are
whole-word outputs; every training character is also an output. Unlisted
words can be spelled character by character, with no OTHER output and no
oracle target filter. Unknown input symbols use the existing unknown ID;
neither base can emit a character absent from its training alphabet.
The new base has **7,078 parameters**: this changes capacity and tokenization,
not just punctuation preprocessing. It is not a parameter-matched causal
proof that punctuation alone helps.

All four new arms share vocabulary, network size and initial weights.
Layered arms retain the existing recognition/extraction and separately
learned placement counts: L0's seven literals; corrected unique-fragment L1
checked against the expanded SCOWL-derived inventory plus training words;
existing repeated-block/hidden-backed L2. They receive no neural spelling
updates for extracted spans. The base learns the concatenated remainder,
retokenized losslessly, including fragments created by extraction. A neural
word payload is queued character by character so the existing placement
selector still runs at **every source event**, including inside a word
payload. Indexed emission does not consume queued remainder or advance base
context. This avoids silently skipping placement decisions to make word
tokens cheaper. No reference-vector injection, pretraining or occurrence
replay is added in this comparison.

Compare exact word accuracy and normalized edit distance, not token losses
with different denominators. Source passes are equal; neural update counts,
parameters and compute are not. Reports retain preparation, training and
generation CPU separately, index update/state costs, all predictions and
checkpoints. Setup-inclusive training totals charge the required inventory,
bank loading and index/feature construction. Historical bank/dictionary
construction and checkpoint/evaluation overhead are excluded, not free.
The historical bank has seen some evaluation material, so L2 is not evidence
of unseen-text performance. A first measured threshold crossing is only a
checkpoint observation, not an interpolated convergence estimate.

The target is at least 30% relative accuracy improvement at comparable
compute, or at least 30% less time to the same accuracy. If the original
declines after an earlier checkpoint, matching its weak final score must not
be described as faster than its earlier successful crossing. Show the whole
curve rather than only a favorable endpoint.

Small tests verify lossless case/punctuation/apostrophe/hyphen handling,
fallback generation, causal features, placement inside queued tokens,
identical new-arm initialization, exact miniature training replay, and saved
checkpoint/evaluation restoration. They prove those mechanics, not good
language learning. The offline runner checks active-chat file hashes and
does not activate any experimental model or alter historical report pointers.
A useful next contribution is a few **new, explicitly approved** prompt/next-word
pairs, especially novel spellings and punctuation boundaries, kept separate
from training until a protocol is frozen.

### Measured full-generation results

Run `10255643d34e41749c6bcc1f68ee8b5c`; each accuracy is the mean of three
seeds on all 256 probes. CPU seconds below include source/index/feature
preparation and training, but exclude evaluation and checkpoint writing.

| Arm | Accuracy, pass 1 | Pass 2 | Pass 4 | CPU, pass 1 | Pass 2 | Pass 4 |
|---|---:|---:|---:|---:|---:|---:|
| Original character base | 2.60% | 0.52% | 1.95% | 16.26s | 32.20s | 64.07s |
| New word/punctuation base | 6.12% | 8.20% | 7.16% | 41.96s | 83.19s | 165.93s |
| New + L0 | 6.64% | 7.81% | 6.51% | 42.08s | 83.48s | 165.85s |
| New + L0+L1 | 7.68% | 7.29% | 3.52% | 45.74s | 85.93s | 166.68s |
| New + L0+L1+L2 | 5.99% | 5.99% | 5.86% | 30.96s | 56.44s | 107.15s |

At four passes, the plain new base gives **266.7% higher relative accuracy**
than original, but uses **2.59x total CPU**. L0 adds no consistent benefit.
L01 starts best on average but declines badly; seed 117 falls from 9.77%
at two passes to .39% at four. L012 uses **35.4% less CPU than the new base**
at four passes but has **18.2% lower accuracy** and worse normalized edit
distance (.90451 versus .81521 new / .88131 original). Fewer updates are not
automatically better learning.

There is a promising **approximately matched-cost** comparison: L012's first
pass gives 5.99% at 30.96s, against original's second pass at .52% / 32.20s.
Do not exploit the latter decline: taking each original seed's **best of its
first two checkpoints** instead gives 2.86%. L012 still improves this mean
by about **109%** under that approximately 32-second budget ceiling.
Some selected original peaks were obtained earlier, well below that ceiling;
this is not a claim of faster time to those peaks. This is a descriptive
development comparison, not exact equal-time stopping or a new preregistered
test. L012 loses to the original's earlier peak in seed 17 (6.64% vs 7.03%);
it improves the other two. Thus the 30% accuracy goal looks promising on
average at similar measured cost, not established consistently across seeds.

Time-to-quality must also respect earlier success. Against each original
seed's **best observed** accuracy, only seed 117 shows a >=30% measured
time reduction for new, new+L0 and new+L012; none of those is an all-seed
speed win. New+L01 has no >=30% crossing. Original seed 17 reaches its
7.03% peak after one pass in 16.63s; comparing only against its four-pass
.39% score would badly exaggerate speed.

Every new arm retains 7,078 neural parameters. Targets/pass:
30,905 original / 24,142 new / 23,983 L0 / 23,441 L01 / 14,696 L012.
L012 removes 3,560 L2 occurrences, 67 L1 and 113 L0 per pass, leaving
15,982 remainder characters; placement/index cost remains included.

Fallback exists mechanically, but is not yet a strong spelling learner.
Of 256 probes, 140 have a direct word-surface output available in the
training-only new vocabulary; 116 require character fallback. At four
passes, new gets **55/420 direct-word judgments and 0/348 fallback judgments**
correct across seeds. L012 gets 39/420 and 6/348; original gets 14/420 and
1/348 under the same grouping. These repeated-seed judgments are not 768
independent examples. Most new-base improvement comes from familiar word
units, not demonstrated new-spelling ability or grammar understanding.

Restore and verify the saved full continuations with:

```bash
python3 neural_base_comparison.py --analyze-run 10255643d34e41749c6bcc1f68ee8b5c
```

The derived `analysis.json` corrects earliest-crossing analysis and remeasures
inference CPU on restored states with bounded original-character context
lookup; exact output equality is required. Raw `report.json` and checkpoints
remain unchanged, including their original code hashes/timings.
`validation.json` records checkpoint hashes and restoration evidence.
Preparation/training schedules, CPU and weights are not revised by analysis.
All **60 states** (15 initial plus 45 trained checkpoints) were restored;
all **15,360 complete continuations**, word predictions, accuracy and edit
metrics matched exactly. The 27 focused tests also pass, with full-size
restoration kept distinct from exact training replay on a miniature fixture.

Restored four-pass inference CPU for all 256 probes: 2.77s original /
8.18s new / 9.44s L0 / 10.14s L01 / 14.03s L012. L2's reduced training
work does **not** make this sampler faster; prompt extraction, per-character
placement and the larger base still cost time. These are local CPU
measurements, not end-to-end app latency or portable throughput guarantees.
No model was activated; the Acts/Corinthians chat checkpoint is preserved.

## Layer 2 Evidence Milestone

The earlier milestones below distinguish lab generation from the standalone
neural baseline. The new isolated [full-layer learning experiment](#full-layer-learning-and-efficiency)
connects all three layer roles to gradients without changing those historical
runs or the permanent lab-only admission registry.

### Developmental foundation and admission

The seven seed forms are the deliberately small starting foundation. They
remain unchanged. Other observations are captured for evaluation, not declared
universally impossible. A separate developmental registry now permits growth
without rewriting Layer 0 or confusing lab-generation permission with training.

`seed_life.development_assessment` takes explicit source-start and completion
flags. It reports excluded, developing, unreviewed, or confirmed, separately
from foundation eligibility. A definite exclusion wins even for unfinished
text; surviving incomplete text is developing unless it is already an exact
confirmed foundation form. A completed unknown remains unreviewed. These are
operational labels, not biological life or semantic understanding.

Only the English seed experiment domain is supported. A rule catalog records
scope, boundary conditions, rationale and examples. No new exclusions are
introduced. Source start is not a sentence boundary; literal spaces are not
silently substituted for newlines; a source ending or budget cutoff does not
prove that a word or response is grammatically finished. The existing
`</>` notation remains a compatibility representation of source start.

Extraction ruleset `seed-life-v5` adds per-boundary developmental metadata
alongside the old alive/dead/maybe groups. Extracted 1–3-character windows are
incomplete observations, not completed words. Their foundation eligibility
remains separate. Lab v7 additionally assesses the submitted input as a
completed observation; this still does not declare it linguistically complete.
Older extraction rulesets and lab reports remain intact.

`seed_development.py` implements explicit defer/admit/revoke decisions for
1–4-character selections in exact saved lab prompts. It verifies source hashes,
records rationale, tests the selected boundary and observed neighboring joins,
and stores every decision with its original text/offset, scope/version, tests,
and timestamp. The latest decision per exact pattern/placement determines
eligibility for developmental lab generation. Earlier events remain available;
re-admission after revocation requires another explicit review and passing
checks. These are user declarations, not legal clearance or training approval.

Mechanical tests cannot guarantee all future combinations. Consequently the
developmental generator checks the whole proposed output on every join. It
starts with foundation captures and active admitted blocks, preserves their
start/interior placement, rejects excluded compositions, then scores each
survivor by the geometric mean of its character probabilities under Layer 2.
This avoids multiplying an automatic length penalty into every longer block.
Candidates unseen in the prompt receive the explicit probability floor 1e-12
for ranking; admission, not prompt frequency, authorizes their availability.
No fitting/no surviving block stops generation visibly; no unrelated fallback.

This replaces the original rule-only sampler in **new** reports, without adding
another card. It is now one genuine foundation/admission → exclusion →
attraction path; the other comparison paths retain their clearly labeled roles.
Accepted pieces do not prove the entire composition is a confirmed word or
sentence. Registry hashes/snapshots and choice traces identify the exact
admissions used. Decisions affect future comparisons, not old report downloads.

Training still has its own explicit approvals. Ordinary chat selection, memory
capture, existing generator training, and neural weights are not changed by
admitting or revoking a block.

**A small contribution:** review one original short fragment from a saved lab,
write why it should be available, admit it for lab use, and compare. Then revoke
it and verify that the next comparison no longer uses it. No initial automatic
admissions are supplied.

`layer_two.py` now reports directional character attraction in new chat lab
records, independently of neural training. It counts every overlapping
2–4-character window; the preceding 1–3 characters form a context and the last
character is its observed continuation. Occurrences and opportunities are
separate, so common patterns do not imply certainty without support.

For `" am amount "`, `am` occurs twice: once followed by space, once by `o`.
The conditional likelihood of each is 1/2. For `"ab ab a"`, `a` is followed by
`b` only 2/3 of the time (its final occurrence has no character successor),
but both `b` occurrences are preceded by `a`. Hence only the backward
relationship is observed-always. For `"ab ab"`, that pair is mutual in this
source. None of these observations becomes a prohibition.

Rows retain exact source offsets and identify source-start windows; spaces,
newlines, and case are never normalized. Source-end outcomes are counted
explicitly. A Wilson 95% lower bound shows the difference between support one
and repeated observations under an independent-trials assumption; because
overlapping windows and repeated text are correlated, it is not a calibrated
guarantee of real-world confidence.

The report is saved as evidence in the existing lab record with the exact
prompt hash. There is no separate learned Layer 2 checkpoint or automatic
reuse of approved generator text. The initial evidence-only milestone was
subsequently extended with explicitly approved experimental generation, not
persistent training.

Lab v6 keeps four baselines and adds prompt-local Layer 2 generation and an
old-transition/Layer 2 product sampler. These modes do not require a living
seed; they are explicitly outside Layer 0 admission. Existing Layer 1
exclusions are applied to the entire output plus each proposed character,
including source-start context, before sampling. Source-end choices stop;
exhausted candidates stop with a reason. No neural gradients change.

The distribution starts with prompt unigrams and interpolates successively
longer contexts (one through three characters), giving each context weight
`support / (support + 2)`. This is a fixed, inspectable experimental smoothing
choice, not a learned hyperparameter. For the hybrid, each existing
four-character transition count is multiplied by
`0.05 + attraction_probability`; the floor keeps unseen old characters
available. Missing old contexts never trigger a restart or shorter-context
fallback. Candidate weights and exclusions are recorded at each step.

Both new samplers use the same seed and maximum length. They are new algorithms,
not a claim that the old model's four-candidate response selection has been
reversed. The original aligned overlay only makes supported A/I case edits
and initial consonant repairs; identical output is an honest, frequent outcome.

Per-mode metrics show prompt-attraction negative log likelihood (natural
logarithms; unseen probability floored to 1e-12 for scoring), supported next
characters, Layer 1 violations, and equality to old text. Source-end probability
is used in generation but not included as an extra scored character. Empty
output gets no loss, not an artificial perfect score. These are descriptive
prompt-fit measures; unequal output lengths, prompt reuse, and different
sampling algorithms prevent treating them as held-out improvement evidence.
No independent generalization or semantic-quality test is implemented here.

**A small contribution:** try `" am amount "`, `"ab ab"`, then `"ab ab a"` in
Chat with lab comparisons enabled. Inspect offsets, denominators, and both
directions—not just an “always” label.

## Neural Weight Milestone

### Full-layer learning and efficiency

The user reaffirmed Layer 0 as fundamental and explicitly approved
evidence-driven candidate growth inside an isolated training experiment,
with candidates separate from confirmed forms. `neural_layers.py` and
`neural_efficiency.py --acknowledge-review` now provide the first full
developmental neural-learning experiment. This is one inspectable numerical
operationalization, not proof that it is the definitive interpretation of
the user's theory or that the theory succeeds.

Layer 0 retains all seven initial possibilities and the existing five
confirmed forms. It scans approved training windows of sizes 1, 2, 3, 4,
in that order. Observed primitive characters can become candidates; longer
blocks require available prefix/suffix subblocks. Exact source-start versus
interior placement, first offsets, support and component identities are
preserved. Existing Layer 1 exclusions veto conflicting candidates in the
full path. The ledger is saved, never written to the permanent admission
registry; candidates are not automatic confirmations.

For predicting a next character, the longest suffix (up to 4 characters,
with its actual boundary) gets availability weight 1 if it is foundational
or an observed candidate, otherwise 0.05. Unseen is not impossible; this
exploration value is a fixed experimental choice, not a user-defined law.
The source boundary and unknown neural output tokens also retain 0.05
availability. Unknown validation characters are not added to the vocabulary.

Layer 2 uses the existing training-only attraction interpolation, built once.
Source-end mass is removed and remaining character mass renormalized because
this experiment predicts original characters, not end tokens. It reserves
0.05 uniform mass over neural output tokens, including unknown. Layer 0
availability multiplies that prior; Layer 1 zeros forbidden ordinary next
characters using the existing boundary-aware exclusions. The probabilities
are `softmax(neural_logits + log(prior))`. Backpropagation uses the resulting
distribution minus the one-hot target; priors are fixed, not differentiated.
Finite-difference tests verify each weight group against this exact loss.
Default no-prior neural calculations remain unchanged.

Arms are baseline, full_012, without_0, without_1, without_2 and layers_only.
Neural models share architecture, starting weights, 5,000 target draws,
rate 0.03 and context 4. Without Layer 1 also disables its candidate veto.
Layers-only normalizes the full prior without a neural forward pass or
gradient updates. Layers affect both training loss and deployed experimental
predictions; saved neural weights must be used with their matching ledger,
priors and policy to reproduce a layered predictor.

Both splits are evaluated against every original target, not cleaned text.
Any forbidden target makes raw full-text loss infinite (JSON null).
Coverage/blocked offsets and sampled training conflicts are explicit; an
infinite score cannot match a finite target. Full source/validation texts
remain in each evidence file, including conflicts. Existing exclusions are
scoped experimental rules, not guarantees that every excluded text is invalid.

The target is the matched baseline's final validation loss, measured before
the full arm. First match is recorded only at saved steps 0, 500, ..., 5,000,
without interpolation or early stopping; all final neural checkpoints remain.
The experiment is one seed and reused Acts 2 validation, not a final test.
Acts 3 text is not read. No output is trained back in.

All arms read Acts 1 in full. Layer construction and training prior preparation
are charged even when an arm matches at zero updates. Reports retain unique
source characters, Layer 0 window observations, Layer 1 source checks, Layer 2
context and evidence-analysis windows, serialized layer state bytes, CPU/wall
preparation/update time, evaluation, checkpoint and end-to-end time.
Timers on a shared machine are descriptive, not a general speed benchmark.
Observation counts are algorithmic counts, not a complete CPU operation model.
Serialized evidence bytes are not peak resident memory.
Repeated evaluations reuse fixed prior caches; validation-prior preparation
is timed separately and samples recompute priors per character. Cached
validation prediction timings alone do not measure uncached serving latency.

Generation polls `latest-layered.json` independently of ordinary/context/rate
reports. Every run keeps its report, exact-text/provenance/ledger evidence
per arm and neural checkpoints; ordinary models and admissions remain
unchanged. Numeric/I/O errors attempt a failed report; abrupt termination
can leave running and has no automatic recovery.

Tests cover gradient correctness and changed updates, each ablation's effect
on probabilities, exact foundation preservation, compositional candidate
growth, placement/exclusion boundaries, never-confirmed candidates, nonzero
novelty/unknown support, full raw-target conflicts, first-match rules, effort
fields, held-out isolation for every weight set and ledger, existing-report
preservation, no final-test files, and explicit failure reporting.

**A small contribution:** inspect one candidate's saved source offset and its
component blocks, then compare the full path with without_0. That review
can help judge whether this availability mapping reflects your intended
development—not merely whether its validation number is lower.

#### First integrated result

Run `48790f42b24b4bc2b865f5c437edb024` completed. The matched baseline
reproduced the earlier 0.03 weights and established target loss 2.6582.

| Path | Final validation loss | Updates at first measured match | Preparation + updates at match |
|---|---:|---:|---:|
| Neural baseline | 2.6582 | 5,000 | 2.586 s |
| Full 0/1/2 + neural | 2.6389 | 0 | 0.452 s |
| Without Layer 0 | 2.1276 | 0 | 0.376 s |
| Without Layer 1 | 2.6389 | 0 | 0.336 s |
| Without Layer 2 | 2.3301 | 500 | 0.522 s |
| Layers 0/1/2 only | 2.6360 | 0 | 0.405 s |

Final neural losses above are after 5,000 updates, whereas the effort columns
refer to the first saved threshold match. Full-path validation before neural
updates was 2.6370. Thus the layers supplied baseline-level prediction without
gradient training, but further full-path neural updates did not improve
validation. All arms read the same 3,411 source characters. This demonstrates
an example of less measured preparation/update work for similar prediction,
not zero learning, less source data, or universal computational superiority.

End-to-end at first match, including shared source verification, prior-cache
preparation, scheduled evaluation and earlier checkpoint writes, was about
18.19 s for baseline, 2.80 s for full, and 1.13 s for layers-only.
These numbers depend heavily on the measurement schedule: baseline had eleven
evaluations before matching, while layers matched at their first measurement.
They must not be interpreted as a general training or serving speedup.
All final neural arms ran their full budgets despite earlier matches.

Coverage was 100% of all 5,274 original validation characters for every arm,
including 18 unknown-character occurrences. No exclusions caused skipped
training targets on this source. Separate tests with rule-conflicting text
verify that null/infinite full loss and skipped-update counts are exposed,
never a favorable filtered metric.

The full path produced 3,696 unconfirmed candidate blocks, retaining the
five confirmed foundation forms. Its roughly 1.68 MB of serialized
candidate/attraction audit evidence is additional storage, not peak memory
or evidence that total model storage decreased.

**Important negative result:** this first Layer 0 availability mapping
penalizes novel suffix combinations too strongly on validation. Removing it
improved loss substantially; removing Layer 1 made only a tiny difference
on this source. Lower loss in the full path than baseline does not prove
every layer helps. It motivates revisiting how Layer 0 expresses compositional
possibility rather than discarding the user's foundational role. Layer 2
provides a strong count-based prior, and standalone neural updates added
little to the combined path at this budget. Samples have recognizable
fragments but remain garbled, not a viable agent or demonstrated understanding.

Every final loss and sample was reproduced from stored weights and evidence.
All 56 checkpoints were present; baseline weights and prior report/generator
hashes matched, the permanent admission registry was untouched, and no Acts 3
text was read. The browser displayed the full quality/effort comparison.
The layers-only checkpoint completion flag was corrected after fixing save
ordering; no loss, timing or weight measurement was changed.

#### Proof-shaping run: size curve, seeds, effort, rules and a fresh test

The user asked how to prove or shape the faster result and said "run the
whole set". `neural_proof.py` adds, in one controlled run:

1. **A standard comparison**: interpolated Kneser-Ney character counting
   (`neural_counting.py`, written from scratch, textbook discount 0.75, not tuned).
2. **A training-size curve**: the first 10/25/50/100% of Acts 1, with
   vocabulary taken from that subset only and Acts 2 validation unchanged.
3. **Five seeds** per neural path (seed 1 reproduces the earlier run exactly:
   baseline 2.65818, full_012 2.63890).
4. **Operation counts**: formula-based estimates where each multiply-add,
   table update/read or rule lookup counts as one. Not hardware counters.
5. **Layer 0 v2**: same foundation and unconfirmed candidates, but an unseen
   3–4 character block counts as possible when its overlapping smaller
   pieces are possible. Single characters and pairs must still be observed.
6. **Rule breaks in generated text**: 5 prompts × 200 characters per run,
   counting Layer 1 exclusions in generated characters only.
7. **A fresh test**: success criteria were frozen in
   `.data/neural-proof/preregistration.json` *before* Acts 4–5 were
   downloaded, then evaluated exactly once (a second evaluation is refused).

Development run `ac6c58288c0f447ebf5cdad9844c3775` (200 runs), mean
validation loss on Acts 2 (lower is better):

| Path | 10% | 25% | 50% | 100% | Training ops at 100% | Rule breaks / 1,000 |
|---|---:|---:|---:|---:|---:|---:|
| Neural baseline | 3.008 | 2.689 | 2.665 | 2.639 | 32,970,000 | 0.20 |
| Neural + Layers 1/2 | 2.800 | 2.488 | 2.365 | 2.127 | 34,326,239 | 0 |
| Neural + Layers 0v1/1/2 | 3.196 | 3.006 | 2.910 | 2.637 | 34,486,832 | 0 |
| Neural + Layers 0v2/1/2 | 3.078 | 2.678 | 2.490 | 2.188 | 34,486,832 | 0 |
| Standard counting (Kneser-Ney) | **2.628** | **2.303** | **2.177** | **1.974** | 27,276 | 0.20 |
| Layer 2 only | 2.785 | 2.496 | 2.370 | 2.133 | 23,865 | 0 |
| Layers 1/2 | 2.784 | 2.495 | 2.370 | 2.132 | 27,276 | 0 |
| Layers 0v2/2 | 3.071 | 2.681 | 2.490 | 2.188 | 37,503 | 0 |
| Layers 0v1/1/2 | 3.191 | 3.008 | 2.907 | 2.636 | 40,914 | 0 |
| Layers 0v2/1/2 | 3.070 | 2.681 | 2.490 | 2.188 | 40,914 | 0 |

Neural seed spread was about ±0.02 for the baseline and under ±0.006 with
layers; counts-only paths are deterministic. The selection rule was revised
once, after these results but **before** fresh data existed: the first draft
("lowest validation loss") picked neural + Layers 1/2, only ~0.005 better than
counts-only at ~1,000× the operations, so the effort test would have failed by
construction. The frozen rule picks the cheapest layered path within 0.02 nats
of the best: **Layer 2 only**. The first-draft pick was scored as a secondary.

Fresh Acts 4–5 (9,650 characters, never used for training or selection), at
100% training data: baseline 2.547, neural + Layers 1/2 2.087, Layer 2 only
2.095, Layers 1/2 2.094, Layers 0v2/1/2 2.124, Layers 0v1/1/2 2.605, standard
counting **1.940**. No fresh character was blocked by Layer 1.

| Pre-registered criterion | Result |
|---|---|
| C1 similar result to the neural baseline | **Met** (2.095 vs 2.547) |
| C2 ≤10% of baseline operations and ≤25% CPU | **Met** (~0.07% ops; 0.06 s vs 5.3 s) |
| C3 better than standard counting | **Not met** (2.095 vs 1.940) |
| C4 Layer 0 adds value over Layers 1/2 | **Not met** for v1 or v2 |
| C5 Layer 1 holds (zero generated rule breaks, full coverage) | **Met** |
| C6 data efficiency | **Met**: 25% of Acts 1 beats the baseline trained on 100% |

**What this proves:** on this source, building count-based likelihood
(Layer 2) gives a better next-character predictor than our small neural net
trained 5,000 steps, for about 1/1,000 of the estimated work and a quarter of
the data, and it held on text no part of the system had seen. Layer 1 removes
the rare rule breaks (0.2 per 1,000 characters) without hurting loss.

**What it does not prove:** that the *developmental* layers are the reason.
A standard counting method beats every layered path at every data size, so
the advantage is explained by counting, which Layer 2 is a weaker form of.
Layer 0 still costs prediction quality: v2 recovered most of v1's loss
(2.636 → 2.188 on validation) but remains worse than no Layer 0 (2.132).
Nothing here shows understanding; samples remain garbled fragments.

**What could shape it next:** give Layer 2 the standard counting method's
two ideas (subtract a small discount from seen counts; base the fallback on
how many *different* contexts a character follows); let Layer 0 mark
possibility without penalizing unseen observed-character pairs; and test
Layer 1 where rules actually bite (rule-heavy or noisy text). Any new claim
should again be frozen before a new untouched source is fetched; Acts 4–5
are now spent as a fresh test.

#### Layer 0 only states: one-way configurations

The user refined Layer 0: besides impossible configurations, some
configurations of 1–9 characters exist **only one way**, like DNA building
blocks. Letters alone are free; as they combine, some combinations lock into
a single path (in English, `eq` is always followed by `u`; `jec` by `t`).

`neural_onlies.py` earns only states from Acts 1 in reading order. A block
seen 3 times with one outcome becomes an only state. If later text shows a
different outcome, the original path is **kept** and the new one recorded as a
compatible variant (a mutation). Forward states say what must come next;
backward states say what must come before. They are unconfirmed candidates
and never forbid a character; off-path characters keep a reduced weight.

Run `7a445dd9bd814f20a6be10b6bf8fbd88`: 572 forward (85 mutated) and 592
backward only states, mostly 4–6 characters. On Acts 2 they held (path or
known variant) 85–96% of the time. Mean validation loss, Acts 1 → Acts 2:

| Path | 10% | 25% | 50% | 100% |
|---|---:|---:|---:|---:|
| Standard counting (Kneser-Ney) | 2.628 | 2.303 | 2.177 | 1.974 |
| Layers 1/2 | 2.784 | 2.495 | 2.370 | 2.132 |
| Only states + 1/2, off-path weight 0.5 | 2.799 | 2.494 | 2.371 | 2.128 |
| Only states + 1/2, off-path weight 0.05 | 2.902 | 2.583 | 2.517 | 2.270 |
| Only states + Layer 0 v2 + 1/2, weight 0.5 | 3.087 | 2.683 | 2.495 | 2.190 |

**What this shows:** only states are real and mostly stable on new text, and
they are a readable inventory of building blocks. **What it does not show:**
better prediction. The best setting gained 0.004 (within noise, chosen on
Acts 2), and strict settings hurt, because the 5–15% of breaks are costly
and Layer 2 already captures most of the same regularities. One chapter
cannot separate true onlies (spelling) from only-so-far onlies (`M→a` until
"Moses"). No fresh test was run: Acts 4–5 are spent and Acts 3 is not read.

Ways to contribute: decide which only states are *true* (defined, like
chemistry) versus *earned*; suggest where compatible joins and splits should
happen at the word and thought levels, where only states may matter more
than at the letter level.

#### Word-level only states: one-way and opposite

The next rung of the ladder (letters → words → thoughts) asks whether whole
words have only states. `neural_word_onlies.py` reads Acts 1 in order and
earns a **one-way** only when a word has been followed by exactly one next
word at least *support* times (2 or 3), and an **opposite** only when a word
has always been preceded by exactly one word. Later breaks are kept as
mutation variants, as at the letter level. During prediction the guidance
works on the current partial word: if the previous word has a one-way only,
letters off that path get a lower weight (0.2 or 0.5); opposite onlies lower
the weight of finishing a word whose required predecessor is missing. Nothing
is forbidden. Run `cacf05e4e24648ea83d77a879da0dd89` (76 s) found:

| Support | One-way | Opposite | Held on Acts 2 (one-way / opposite) |
|---|---|---|---|
| 2 | 7 (e.g. into→the, Holy→Spirit, were→looking/staying) | 15 (e.g. the←Holy, he←was) | 17% of 24 uses / 34% of 73 uses |
| 3 | 2 (into→the, Holy→Spirit) | 4 | 57% of 7 uses / 43% of 28 uses |

Validation loss on Acts 2 (nats per character, lower is better):

| Path | 10% | 25% | 50% | 100% |
|---|---|---|---|---|
| Standard counting (Kneser-Ney) | 2.628 | 2.303 | 2.177 | 1.974 |
| Layers 1/2 | 2.784 | 2.495 | 2.370 | 2.132 |
| Letter onlies + 1/2 | 2.799 | 2.494 | 2.371 | 2.128 |
| Word one-way + 1/2 (support 3, weight 0.2) | 2.784 | 2.495 | 2.370 | 2.132 |
| Word opposite + 1/2 (support 2, weight 0.2) | 2.786 | 2.497 | 2.382 | 2.151 |
| Letter + word both (support 3, weight 0.5) | 2.799 | 2.494 | 2.374 | 2.131 |

**What this shows:** in one chapter, words are far freer than letters. Only
fixed phrases lock ("Holy Spirit", "into the"); most earned word onlies are
coincidences of small data and break on the next chapter. Word guidance
changes loss by at most 0.0005 (noise), and opposite onlies slightly hurt.
**What it does not show:** that word onlies are useless. With far more text,
fixed phrases and names could earn real support, and phrase-level onlies may
be where "thoughts" begin. It also does not tell us anything fresh: Acts 2 is
reused development data.

Ways to contribute: list phrases you believe can only exist one way (names,
titles, set phrases), so we can test *defined* word onlies against *earned*
ones; and describe what makes two thoughts "compatible" to join or split.

#### Fixed patterns (only chains) for fast reading and generation

**The idea (user):** a pattern that can exist only one way does not need to be
predicted. When reading new text, the model should recognize it and skip it.
When generating, once the model chooses the start of the pattern, it should
write the rest at once instead of predicting one character at a time. This is
not a dictionary of unique words. A pattern can sit inside a word, cross a
space, or run into punctuation.

**What `neural_patterns.py` does:** for every block of 1–9 characters
(the *trigger*) seen at least *support* times in the learning text, it checks
whether every occurrence was followed by exactly the same characters. If so,
the longest shared run (up to 24 characters) is the *continuation*. A longer
trigger is kept only if it extends its shorter version. Reading uses the
longest trigger that matches: a correct continuation is skipped as one
decision; a wrong one costs one normal step. Generation uses counts
(Kneser-Ney) for normal steps and emits continuations whole.

**More source text:** Acts 6–27 were fetched from the same public-domain WEB
pages (eBible.org) with raw and text hashes. Verses the translation leaves
empty (8:37, 15:34, 24:7) are skipped and recorded. The rights page differed
from the earlier snapshot only in its generation date. Acts 3 is still never
read. Acts 28 was locked and fetched only after pre-registration.

**What happened, in order:**

1. The user chose: learn from Acts 6–27, lock 28, support 50, success = at
   least 25% coverage at 95% correct.
2. Development (learn 1, 2, 4–26 → check 27, run
   `4f84e153b69c4d38ac0d85486184f8d5`): only 76 patterns, 3.4% coverage at
   98.4% correct. Every support/exception setting tried stayed under 25% at 95%.
3. Before Acts 28 was fetched, the user revised: support 10, target ≥10%
   coverage at ≥95% correct. The revision is written into the pre-registration.
4. Development again (run `3744728e5ad743a394928115855cb3c6`): 1,127
   patterns, 12.9% coverage, 94.5% correct, 3.9% fewer decisions.
5. Locked test, once (learn 1, 2, 4–27, 117,198 characters → Acts 28, 4,259
   characters, 1,171 patterns):

| Measure | Acts 28 | Target |
|---|---|---|
| Characters patterns supplied correctly | 19.1% | ≥10% (met) |
| Correct when a pattern claimed characters | 94.1% (51 wrong of 863) | ≥95% (not met) |
| Whole patterns right | 95.0% (493 of 519) | — |
| Reading decisions saved | 6.9% | — |
| Generation decisions saved (5 samples) | 4.7–11.3% | — |

Examples that fired on Acts 28: `cerni→ng `, `tog→ether`, ` our f→ather`,
`hems→elves`, `e Ho→ly Spirit`, `peo→ple`, `¶Whe→n `. The longest learned:
`mandi→ng officer`, `hig→h priest`, `resu→rrection`.

**What this shows:** fixed patterns are real and cheap. Lookup is a dictionary
probe, not a prediction, and almost a fifth of an unseen chapter was supplied
by them. They are mostly word endings and set phrases ("Holy Spirit",
"commanding officer").
**What it does not show:** that patterns are safe to trust blindly. About 6%
of claimed characters were wrong, because "seen 10 times one way" is evidence,
not proof that a sequence can only exist one way. Decisions saved (6.9%) are
much lower than coverage (19.1%), because most continuations are only 1–3
characters and each still costs one decision. All text is one book, so this is
not general-language evidence. Acts 28 is now spent.

Ways to contribute: point out the wrong-claim cases you consider "really
fixed" versus coincidences. Suggest patterns you know are always the same
(names, titles, set phrases) so we can compare *defined* patterns with
*earned* ones. And decide whether a pattern should fire only when the model
is confident it has started that word or phrase, which would trade coverage
for accuracy.

#### Strict layer and light pattern layer

**The idea (user):** an only layer should be about 99.999% accurate: "if the
model sees this, then it is this." Lighter weighting belongs in another layer.

**Why the fixed patterns could not be that layer:** they guess the rest of a
word from its start. Every miss on Acts 27 was a new word that began like a
known one (`ffic→e` on "difficulty", `r I→ ` on "Italy"). Counting can't prove
99.999% either: n sightings with no exception only bound the error at about
3/n, so it would take roughly 300,000 sightings per pattern.

**What `neural_strict.py` does:**

- **Strict layer:** units are learned words, plus phrases of 2–4 words seen at
  least 5 times. When reading, a unit is claimed only when the text at a word
  start *is* the whole unit, ending at a word boundary. It is recognized with
  one decision and skipped; anything unknown goes to the other layers. When
  generating, a soft word-count layer chooses the next word (with a
  Layer-1-style rule: no word directly after a word, no double spaces), and the
  strict layer spells it out whole.
- **Light layer:** the earlier fixed patterns (support 10) only add weight
  `(support+1)/(support+2)` to their guessed character, mixed with Kneser-Ney
  counts. They are never certain and never forbid.

**Development result** (learn Acts 1, 2, 4–26 → Acts 27, run
`477fc8c644cc441bba970c2da375b424`; Acts 27 was already used, so this is not a
fresh test):

| Measure | Result |
|---|---|
| Strict wrong claims | 0 of 699 (by construction) |
| Letters recognized whole | 79.0% (1,054 characters came from multi-word phrases) |
| Reading decisions saved | 49.0% |
| New words handed on | 166 (names like Adramyttium, Euroclydon; new forms like Hoisting) |
| Generation: decisions saved, misspellings | 53–61%, 0 |
| Loss: counts only → counts + light patterns | 1.6787 → 1.6638 (rule of succession; 0.9 gives 1.6607) |

**What this shows:** a strict layer can be 100% right if it only *recognizes*
or *spells* units that have already been identified. It still removes about
half the work of reading, and spelling a chosen word costs one decision instead
of one per letter. The light patterns help counting a little when used as
weight instead of certainty.
**What it does not show:** that it understands or chooses well. Generated
word order is still nonsense; choosing the right word is the soft layers' job,
and they are weak. The 0 wrong claims prove the bookkeeping, not prediction.
New words (about 4% of words here) always need the other layers.

Ways to contribute: decide whether user-defined onlies (names, titles, set
phrases) should be added to the strict layer, and how much evidence a phrase
needs before it is a strict unit rather than a light weight.

#### Strict skeletons: hidden or missing letters that resolve one way

Idea (user): some patterns are not whole words, such as `th*t` or `a*d th*t`, yet
in practice resolve to only one thing. The strict version should be wrong
about 1 in 100,000 times or less. `neural_skeletons.py` tests this with three
skeleton kinds:
- **hidden1:** one letter hidden (`th*t`).
- **hidden2:** two letters hidden.
- **dropped1:** one letter missing (`tht`).

**Layer 0, the possible:** the SCOWL word list (permissive license, notice kept
in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)), plus every learning
word, plus regular forms of those words. A skeleton is **strict** only when:
- exactly one possible word fits it;
- that word was seen at least 5 times;
- the word is not a name;
- the first letter is visible.

A skeleton that only the learning text resolves (`sa*d`→said, even though
"sand" exists) is **light**: weight, not certainty. Phrase skeletons are light
unless every hidden letter is already forced.

**Data:** the whole public-domain WEB Bible, split by book before any analysis.
- Learn: 2.1M characters.
- Develop: 16 books, 1.0M characters.
- Locked: 15 books, 0.85M characters, read once.

Generated text must not be called WEB or scripture.

**What development taught (learn → develop):**
- Counting alone failed (`sa*d` was "sand" in the shipwreck chapter). The
  dictionary plus support rule gave 587 wrong in 1.16M checks (5e-4).
- Nearly every error was a **name** (`se*ah` Selah/Serah, `*aal` Baal/Gaal).
  Names are an open class that no list can complete. Fixes:
  - A word capitalized in most of its uses is never strict.
  - A skeleton showing a capital letter is handed on, not claimed.
  - A closing quote is no longer part of a word.
- Result: 12 wrong in 979k (1.2e-5). The rest were regular forms missing from
  the word list (revealer, scoffer, unstained) and names whose hidden first
  letter hid the capital (`*hav*` Ahava).
- Fix A: list regular forms (-s, -ed, -er, -ing, un-, re-, -ies…) as possible.
  These over-generate on purpose, so they can only veto a strict claim.
- Fix B: never claim when the first letter is hidden.
- With A+B: 0 wrong in 658k checks. The user chose A+B before the locked run.

**Locked result (pre-registered, run once, support 5):**

| | Development | Locked books |
|---|---|---|
| Strict checks | 658,347 | 560,404 |
| Wrong | 0 | 1 (`rbel`→rebel was the name Arbel) |
| Observed rate (K1 ≤ 1e-5) | 0 | 1.8e-6 — met |
| 95% upper bound (K2 ≤ 1e-5) | 4.6e-6 | 8.5e-6 — met |
| Words with a strict hidden-letter skeleton | 66,572 of 154,634 | 55,349 of 129,012 |

**What this shows:** in this text, a large set of partial patterns (about
13,900 one-hidden-letter skeletons for 3,200 words) can be treated as "if it
sees this, it is this" at the 1-in-100,000 level. That covers about 43% of word
occurrences, and the pre-registered test on unseen books agreed.
**What it does not show:**
- That it holds for general English, modern text, or other translations: all
  of this is one Bible translation.
- That it can never be wrong. One locked error remains, and the dropped-first-
  letter case hides names the same way a hidden first letter does. Fixing it
  now would need new untouched text, because the locked books are spent.
- Checks from the same word occurrence are related, so the word-level count (1
  of 129,012) is a fair second view.

Ways to contribute:
- Choose new, untouched text (with rights checked) for the next locked test.
- Decide whether a dropped first letter should also be handed on.
- Collect real "only one way" phrases you know, to test against the word rules.

#### Live skeleton units in neural training and generation

**Historical implementation, superseded:** this added neural word outputs,
which was not the user's intended recognition-only method. Existing checkpoints
remain loadable. See the corrected character-only path below.

The frozen experiment remains unchanged. `neural_skeleton_runtime.py build`
exports a checksummed reusable bank from **learn + develop only**, never the
locked books. It records source, input hashes, dictionary hash, and the frozen
experiment's code hash. The local export contains 42,984 hidden-letter patterns
for 3,201 words. Dropped-letter patterns are not used in this runtime.

On the Generation tab, enable **Use live skeleton units for training and
generation**. The existing review/rights form still applies. No historical
memory or other generator's corpus is automatically imported.

1. Select at most 128 bank-backed units actually present in the new approved
   training text. Held-out text cannot add units or output vocabulary.
2. Recognize a unit only after the **entire lowercase word matches exactly**.
   Do not guess missing letters or change the submitted text. Other text stays
   individual characters, preserving case, punctuation, and spaces.
3. Give the from-scratch MLP those units and characters as output tokens. It
   learns their embeddings and output weights through the existing backpropagation.
   A four-token context can now contain units, not just four characters.
4. The neural model chooses what to emit. A chosen unit is spelled out in one
   output step; character fallback remains available. Whole-unit boundaries are
   enforced, a partial prompt remains character-level, and a character budget
   never cuts a whole-unit emission in half.
5. Save the exact selected-unit map and bank provenance with the weights.
   `/api/neural/generate` and the saved-checkpoint UI control restore that map
   without retraining or requiring the current global bank.

**Mechanics check, not a fresh accuracy test:** an original repeated test fixture
had 240 input characters and 132 encoded targets (45% fewer, excluding the end
marker). After 300 updates, a saved model generated 120 characters with 66
decisions, including 18 complete units. Reloading reproduced the same output.
The output still contained bad word choices and character fragments.

**Limits:** fewer sequence positions are not proof of lower wall-clock cost:
the larger output vocabulary also costs work. Updates stay at the requested
step count. Mixed-token loss includes an end marker and is not directly
comparable to character-baseline loss. The bank already contains prior Bible
learning/development evidence, so a new held-out input is not automatically
unseen by the bank. Full-word confirmation proves lossless bookkeeping, not
the statistical accuracy of resolving arbitrary missing letters.

The default remains the character baseline. This is a working neural integration,
not replacement of the chat engine, permanent memory admission, online continual
learning, or a new general-English accuracy claim.

Ways to contribute: use the same newly approved training text in both modes,
inspect whole-unit emissions and unknown-character fallback, and suggest new
untouched evaluation text before comparing response quality or timed full passes.

#### Paired live-unit efficiency comparison

This first run measured the superseded whole-word-output architecture, not the
user's intended only-state recognition method.

`python3 neural_unit_benchmark.py --acknowledge-review` compares the actual
character MLP against the live unit MLP without comparing incompatible token
losses. It uses only approved staged Acts text, not historical memories.

Run `e9c480699e7c4d30a6b844c494ad05da`:
- 1,905 training characters from Acts 6; 1,494 evaluation characters from Acts 7.
- Three paired initialization/training seeds (17, 117, 217; shuffle seed +2).
- Same learning rate 0.1, four-token context, handwritten gradients and hidden
  dimensions. Vocabularies differ, so parameter counts and effective character
  contexts differ; this is a system comparison, not a fixed-parameter ablation.
- Complete shuffled passes, checkpoints at 1/2/4 passes and at 12,000 updates.
  Arm execution order alternates by seed; no concurrent training.
- 48 deterministic, evenly spaced next-word probes. Both models receive the
  identical original prefix and greedily predict an entire word with its
  ending boundary. Empty and unterminated outputs fail. Accuracy is exact-case
  whole-word match; normalized edit distance is a second common metric.
- CPU/wall timing covers neural updates, excluding evaluation. Source/bank
  loading and per-arm setup are reported separately. Earlier bank construction
  is **not included**. Estimated operation counts reuse the proof-runner helper;
  they are not hardware instruction counts.
- Inputs, hashes, protocol, all checkpoints' metrics, final weights, unit map,
  per-seed samples and report are saved under `.data/neural-unit-benchmark/`.
  The protocol was written before model updates. Settings were not tuned after
  seeing this result.

**Means across seeds:**

| Budget / metric | Character baseline | Live units |
|---|---:|---:|
| Parameters | 1,728 | 3,803 |
| Output vocabulary (including special tokens) | 48 | 131 |
| Targets per pass (unit path includes end marker) | 1,905 | 1,335 |
| Four passes: training CPU seconds | 3.50 | 5.03 |
| Four passes: estimated update work | 44.4M | 65.2M |
| Four passes: exact next-word accuracy | 0.7% | 2.1% |
| Four passes: normalized word edit distance | 0.836 | 0.849 |
| 12,000 updates: training CPU seconds | 5.50 | 11.29 |
| 12,000 updates: exact next-word accuracy | 4.9% | 0.7% |
| 12,000 updates: normalized word edit distance | 0.839 | 0.877 |

The pre-run loose similar-quality rule allowed at most 2 percentage points
lower accuracy and 0.02 higher edit distance at four passes. It technically
passed, **but both models are near the floor**: this does not establish useful
quality. The less-effort rule required CPU and estimated work each at most 80%
of baseline; it failed (CPU 144%, operations 147%). At equal updates, units were
about twice as expensive and worse on both shared quality measures.

The reason is visible in the measurements: units removed about 30% of input
positions, but more than doubled output vocabulary and parameters. Every
softmax and gradient update costs more. A shorter sequence alone was not enough.
This finding applies to the present implementation, not every possible unit model.

Generated samples at 12,000 updates contained more training-vocabulary words:
25% baseline versus 51% units, averaging the word fraction of each sample.
That is only a spelling/familiarity indicator. The actual samples contain
fragments and nonsense; neither demonstrates coherent answers. No automated
coherence verdict or human quality rating was invented.

**Scope:** the bank already saw both chapters, and these Acts chapters have
been used before. This is development evidence, not untouched held-out
generalization. The 48 probes are related and were not used for model selection;
no significance or universal speedup claim is made. There is also no ordinary
word-tokenizer control, so recognizable-word gains cannot be credited uniquely
to strict skeletons rather than generic tokenization.

**Next useful experiment:** reduce the cost of choosing among units (a small
candidate set or separate character/unit choice), then compare again on
development data without rewriting this result. Add an ordinary-tokenizer
control, and reserve new rights-reviewed text before any fresh quality claim.
The user can help by rating saved sample pairs without knowing their source.

#### Corrected character-only recognition and retest

The user clarified that expanded onlies should work like the bounded `a`/`I`
seed forms, not become additional neural vocabulary. `neural_character_onlies.py`
now supplies a rule layer around the unchanged character MLP:

- Recognize exact boundary-aligned forms from the existing bank, plus `a`, `A`,
  and `I`. Keep the original text and character contexts. Skip only internal
  spelling updates; keep the first-character and surrounding-boundary updates.
  Single-letter seeds have no interior characters to skip.
- No word embeddings, new output classes, mixed tokens, or end marker. Both
  paired models have the same vocabulary, initialization and parameter count.
- For generation, derive unique prefixes against the dictionary, learning words,
  and deliberately over-generated regular forms. Only an unambiguous prefix may
  supply the rest of its form without a neural forward pass. Every supplied
  character still advances the original character context. A following delimiter
  is chosen normally, restricted to keep the completed form bounded.
- Ambiguous prefixes such as `th` are not forced to `that`. The current possible
  set permits only 36 such prefix shortcuts, mostly contractions/possessives.
  Whole-word recognition covers 3,204 forms, but recognition coverage is not
  generation-shortcut coverage. This is conservative closed-list evidence, not
  proof against names, quotation ambiguities or new vocabulary.
- Save recognition rules alongside weights. The live UI now uses this path;
  legacy whole-word-token checkpoints retain their original decoder.

The paired development runner defaults to this corrected method. The old
approach is available only through `--legacy-units`. Each run preserves its own
protocol, input hashes, weights, measurements and samples. No locked books or
frozen skeleton experiment code are changed.

The retest uses the same Acts 6/7 prefixes, three seeds, four-pass and
12,000-update budgets and common next-word metrics. Model initialization hashes
are recorded so equality can be verified. Training timings exclude evaluation;
bank/rule construction is a separately amortized cost, not claimed free.
Skipping updates is not the same as learning meaning or absorbing semantic
content: the unchanged model sees character contexts at retained targets.

**Final corrected retest:** `11fef0167915435f9238b5b57e8a7e51`. An initial
corrected run (`731baec884544678b7c3f4e92f7bb713`) gave the same predictions.
The final run aligned the maximum next-word output length to 32 characters for
both methods and added saved initialization hashes; neither changed the result.
Both runs are preserved, not substituted for the old word-output measurements.

| Metric | Baseline | Recognition onlies |
|---|---:|---:|
| Neural outputs | 48 | 48 |
| Parameters | 1,728 | 1,728 |
| Four passes: updates | 7,620 | 5,336 |
| Four passes: CPU seconds | 3.30 | 2.32 |
| Four passes: estimated update work | 44.4M | 31.1M |
| Four passes: next-word accuracy | 0.7% | 3.5% |
| Four passes: normalized edit distance | 0.836 | 0.841 |
| 12,000 updates: CPU seconds | 5.20 | 5.20 |
| 12,000 updates: next-word accuracy | 4.9% | 3.5% |
| 12,000 updates: normalized edit distance | 0.839 | 0.856 |

This fixes the extra-output cost: at equal input passes it uses about 30% less
CPU and estimated update work, matching the skipped spelling work. At equal
updates there is no speed gain. Both loose development criteria passed, but
accuracy remains near the floor and edit distance is slightly worse. This is
not proof of reaching useful quality sooner.

None of the nine corrected generation samples triggered a unique-prefix rule.
Generation therefore did not show a speed gain; rule lookup adds some overhead.
Focused tests separately demonstrate suffix emission with zero neural calls
for those supplied characters, exact context advancement, apostrophes, and
safe fallback. More recognition patterns do not automatically mean more safely
forced generation prefixes.

Cold expanded recognition/prefix construction took 2.02 CPU seconds in a
separate preparation measurement. The cached run's shared setup took about
0.08 seconds. These preparation costs must be amortized over reuse; the
30% figure applies to neural update time, not the entire first-use workflow.

#### Larger recognition-only comparison

The user requested a significantly larger base-versus-new-trainer test. Run:

```bash
python3 neural_unit_benchmark.py --acknowledge-review --large
```

Run `41624f6d045f4a6b9d202ed2cb877b67` completed all ten arms:
- Training: complete Acts 6-12, **30,905 characters** (16.2 times the small run).
- Evaluation: complete Acts 13-14, **9,730 characters** (6.5 times the small run).
- Five paired seeds: 17, 117, 217, 317, 417. Both arms have identical saved
  initialization hashes, 63 neural outputs, 2,103 parameters, four-character
  context, learning rate 0.1, and the same original input text.
- Unchanged criteria from the earlier comparison. No tuning or selecting a
  favorable checkpoint after seeing results.
- Checkpoints at 1/2/4 complete source passes plus 120,000 gradient updates.
  At four passes the baseline has 123,620 updates; recognition has 87,680.
- 256 identical next-word probes. Additionally, 2,048 evenly spaced original
  character targets, **including skipped spelling**, are scored identically
  with each neural model, without recognition/generation rules. This common
  all-character loss is comparable, unlike each trainer's training-target loss.
- Sequential arms, alternating execution order across seeds. CPU/wall timings
  exclude evaluation; preparation is separately recorded. Cached shared setup
  was 0.084 CPU seconds; the earlier bank/rule construction cost is excluded.
- The run folder retains its pre-update protocol, input/codec snapshots,
  **40 milestone checkpoints**, ten final model files, predictions and samples.
  No locked-book evaluation was run.

**Means across five seeds:**

| Metric | Baseline | Recognition onlies |
|---|---:|---:|
| Four passes: training CPU seconds | 65.75 | 46.08 |
| Four passes: training wall seconds | 65.76 | 46.09 |
| Four passes: estimated update work | 862.7M | 611.9M |
| Four passes: next-word accuracy | 1.25% | 2.50% |
| Four passes: normalized word edit distance | 0.879 | 0.856 |
| Four passes: all-character probe loss (nats) | 2.685 | 2.959 |
| 120,000 updates: training CPU seconds | 63.85 | 63.08 |
| 120,000 updates: next-word accuracy | 2.81% | 2.73% |
| 120,000 updates: normalized word edit distance | 0.825 | 0.850 |
| 120,000 updates: all-character probe loss (nats) | 2.663 | 3.001 |

At four passes, per-seed CPU ranged from 64.97-66.43 seconds for baseline and
45.65-46.62 for recognition. The roughly **30% update-time reduction** repeats
across the larger run and follows the 29.1% skipped character targets. At
equal updates, CPU costs are essentially equal: the shortcut saves updates
per source pass, not the cost of each neural update.

The original loose word-quality and less-effort criteria both passed at four
passes. **The additional all-character probe reveals a tradeoff:** neural
character prediction is worse after skipping spelling updates. We therefore
cannot describe this as equal overall quality for less effort. Whole-word
accuracy remains very low and varies sharply between checkpoints; the tiny
model and fixed learning rate are not producing a stable useful language model.
No hyperparameter changes were made to improve this run after the fact.

Original sample entries are from each arm's last executed checkpoint, which is
four passes for baseline and 120,000 updates for recognition. Their old note
called them matched-update samples; that is inaccurate for this larger run.
To avoid comparing different budgets, a **separate post-training evaluation**
reloaded both arms' saved 120,000-update checkpoints and saved 30 sample outputs
under `matched-generation.json`, also exposed through `matched_generation`.
No model was retrained or selected. The runner now labels final-checkpoint
samples accurately via `sample_budgets`.

At equal updates, none of the 15 recognition samples used a unique-prefix
shortcut. Mean CPU per 120-character sample was 0.0192 seconds baseline versus
0.0204 recognition; these very short measurements do not show a generation
speed gain. Both sample sets are visibly fragmented/nonsensical. No automated
coherence verdict or invented human rating is reported.

**Limitations:** all text is reused public-domain Bible material already seen by
the bank, not new untouched general-English evidence. Character loss above tests
the neural model alone, whereas next-word output uses its deployed rules. These
are complementary metrics, not interchangeable scores. Probe counts and seeds
do not make this evidence of understanding, universal speed, or accuracy.

**Next step to discuss:** preserve the recognition shortcut while measuring
whether skipped spelling knowledge must be supplied by the rule layer more
often during generation, or whether some interior updates should remain.
Either changes the learning policy and needs a new recorded experiment; this
run must remain intact.

#### Baseline-plus-rules comparison

The user requested three arms: baseline alone, recognition with skipped
spelling updates, and baseline training plus the recognition/generation rules.
The third arm **retains every baseline training update**. It therefore has
exactly the baseline weights; recognition does not change those updates.

```bash
python3 neural_unit_benchmark.py --acknowledge-review --add-combined 41624f6d045f4a6b9d202ed2cb877b67
```

The runner restores all saved baseline checkpoints, applies the original saved
rule snapshot, and evaluates the same word probes and original character
positions. It verifies that the restored unassisted character loss is exactly
the recorded baseline loss. Training cost is inherited from the source run,
**not a fresh timing measurement or a training speedup**. Evaluation timings
are recorded separately. No new training, data selection, or rule tuning occurs.

Fresh generation samples cover all three arms at the same 120,000-update
budget, with the same prompts, length and random seed. CPU/wall timings,
rule matches and forced characters are recorded. Probe rule counts cover the
32-character continuations, not just the first scored word. The tiny samples
do not establish coherence or reliable timing differences.

Each comparison has a new run directory and pre-evaluation protocol, source
report hash, evaluation-code hashes and combined checkpoint weights. Original
training code hashes remain those of the source run; they are not relabeled as
the new evaluator. The original source reports and models remain unchanged.
The Generation panel displays all three arms and separates generation timing
from the inherited training measurements.

Run `ea3a9d10eae54a3ab52efadd5d7066a2` completed five seeds and all
20 baseline-plus-rules milestones. Source report hash and exact equality of
all combined/baseline checkpoint weights were verified.

| Mean measure | Baseline | Recognition (skip spelling) | Baseline + rules |
|---|---:|---:|---:|
| Four passes: training CPU seconds | 65.75 | 46.08 | 65.75 (inherited) |
| Four passes: next-word accuracy | 1.25% | 2.50% | 1.25% |
| Four passes: all-character loss | 2.685 | 2.959 | 2.685 |
| 120,000 updates: next-word accuracy | 2.81% | 2.73% | 2.81% |
| 120,000 updates: all-character loss | 2.663 | 3.001 | 2.663 |
| Fresh generation: CPU milliseconds/sample | 20.1 | 21.1 | 21.3 |

No rules activated across the 5,120 combined probe continuations or any of
the 45 fresh generation samples (15 per arm). Combined word predictions
therefore match baseline at every milestone. Its unassisted character loss
matches by construction. The combined version avoids skipped-update
degradation but demonstrates **no added quality or generation-speed benefit**.
The short timing measurements include lookup overhead and are descriptive,
not evidence of a stable percentage slowdown.

This tests whether the existing rules help **without removing baseline
learning**. It does not expand rule coverage or prove that partial patterns
are universally unique. A small contribution the user can make next is to
review the side-by-side samples for specific correct or incorrect completions;
that review is evaluation, not automatic new training data.

#### Atomic indexed-onlies test

The user clarified that a recognized only is an indexed reference handled like
a single character: one target, one context position, and one generation
selection emitting the entire pattern. This is different from skipping spelling
updates in character contexts. The existing `UnitCodec` already provides the
atomic mechanism, so it is reused, not duplicated. Earlier descriptions of
expanded outputs as categorically contrary to the intended method are
superseded by this clarification; earlier measurements remain unchanged.

```bash
python3 neural_unit_benchmark.py --acknowledge-review --indexed-onlies
```

Run `2379abc5da8b4edda60333b5a3c1c2cd` used the existing small paired profile:
1,905 training characters, 1,494 evaluation characters, three seeds, 48 shared
next-word probes, 1/2/4 source passes and 12,000 updates. No new corpus or
rule tuning. Recognition runs ahead of training; this does not benchmark
concurrent workers. The test selects up to 128 training-present forms backed
by the frozen bank, ranked by characters saved. Here 82 forms qualified.
Unrecognized text remains characters; the end marker is a separate output.
It is not a dictionary of arbitrary unique words or a new general gapped
pattern recognizer.

The index maps pattern identity to token ID, retains pattern evidence, and
preserves occurrence order in the encoded stream. Joining stream pieces
exactly recovers the source, including spacing. Each recognized occurrence
is one target; there are no internal spelling targets for that occurrence.
During generation the base selects the index ID directly; its literal spelling
is emitted without predicting each character, and the ID advances context once.
No unique-prefix trigger is needed.

| Mean measure | Character baseline | Base + indexed onlies |
|---|---:|---:|
| Output classes | 48 | 131 |
| Parameters | 1,728 | 3,803 |
| Four passes: training updates | 7,620 | 5,340 |
| Four passes: training CPU seconds | 3.59 | 5.30 |
| Four passes: exact next-word accuracy | 0.69% | 2.08% |
| Four passes: word edit distance (lower better) | 0.836 | 0.849 |
| 12,000 updates: training CPU seconds | 5.67 | 11.88 |
| 12,000 updates: exact next-word accuracy | 4.86% | 0.69% |
| 12,000 updates: word edit distance | 0.839 | 0.877 |

The representation reduces encoded reading positions by 30%, but the larger
dense output calculation raises estimated training work by 47% at four passes
and measured CPU by 48%. At equal updates, cost is approximately 2.1 times
baseline. Four-pass accuracy is slightly better but edit distance worse; equal
updates favor baseline. All accuracy remains very low. Unlike the
character-only comparison, parameter counts and effective character context
differ, and all-character-versus-unit losses are not claimed comparable.

Fresh matched-12,000-update samples use the same three prompts and random seed:
nine samples per arm, each producing 120 characters. Baseline needed 1,080
neural decisions; indexed generation needed 728, including **73 complete indexed
emissions**. Total CPU was 0.161s baseline versus 0.236s indexed. Thus atomic
emission activates, saves decisions, and preserves fixed spellings, but this
implementation has not demonstrated faster generation or better overall quality.
Both sample sets remain incoherent. These short generation timings are
descriptive, not stable universal speed ratios.

Per-run protocol/code/input hashes, saved token IDs, rule evidence, milestone
weights and matched-generation samples are retained. The source bank and locked
experiment are unchanged. Existing live character-only training is not silently
replaced. The next design question is the cost of choosing among indexed
patterns, not whether whole-pattern emission works.

#### Separate placement index and remainder base

The user clarified that extracted onlies must not become base training targets,
including reference-selection targets. Placement is a **separate learning job**.
`neural_index_pipeline.py` tests that architecture:

```bash
python3 neural_index_pipeline.py --acknowledge-review
```

- Reuse the bank-backed 82 training-present forms from the bounded atomic test.
  Remove each complete occurrence from the base's input, not just its internal
  characters. Keep its identity, original start/end offsets and stream position
  in a separate index record. Joining stream pieces recovers the original text.
- The base trains on the concatenated remainder only, including original
  whitespace. It sees neither the extracted spellings nor reference embeddings
  or reference targets. Both paired bases retain the full original **character**
  vocabulary for identical initial weights: 48 outputs and 1,728 parameters.
  Character classes are not pruned, but indexed output classes are never added.
- A separate standard-library context-count index learns a reference-or-base
  action at each original word boundary. It uses up to four preceding stream
  events, where a reference is one event, and backs off to shorter observed
  contexts and then global boundary counts. This is a deliberately simple
  first placement learner, not an additional neural vocabulary.
- Generation uses one ordered stream. At a boundary, the index may emit a
  whole pattern without a base forward pass, or defer to a base character.
  Indexed emissions advance the index context but **not** the base's
  remainder-character context. After an indexed word the next base character
  must delimit it. Unseen contexts use the recorded backoff policy; an
  unavailable action defers to the base.
- Fixed indexed spellings do not imply correct placement: placement counts
  are statistical predictions, not strict onlies. The selector can choose the
  wrong word. Patterns remain whole-word confirmations from the existing bank,
  not a new general recognizer for arbitrary missing-letter phrases.

Run `325fdc752672405a940f0ee62de9f037` used the existing small profile:
1,905 training characters, 1,494 evaluation characters, three paired seeds and
48 identical word probes. Extraction removed 113 occurrences / 684 characters,
leaving **1,221 base training characters**. All three initialization pairs are
identical. Full passes are shuffled with the same seed convention as prior
tests; no held-out text updates either learner.

| Four-pass mean | Baseline | Remainder base + placement index |
|---|---:|---:|
| Base updates | 7,620 | 4,884 |
| Base training CPU seconds | 3.358 | 2.178 |
| Placement learning CPU seconds | 0 | 0.0075 |
| Total training CPU seconds | 3.358 | 2.185 |
| Exact next-word accuracy | 0.69% | 1.39% |
| Normalized word edit distance (lower better) | 0.836 | 0.914 |

Total training CPU is about **35% lower**, counting placement-count updates.
Shared loading/bank validation/extraction/probe setup took 0.060 CPU seconds
separately; example preparation is separately recorded per arm. Earlier bank
construction is not included. The placement learner has 587 distinct context
rows, additional memory not counted as neural parameters. Its serialized
counts and per-pass count-update totals are saved. It learns once per pass;
the reported saving does not omit that learning.

At 1/2/4 passes, word accuracy was 0.69/1.39/0.69% for baseline versus
0/1.39/1.39% separated. Edit distance was 0.834/0.830/0.836 versus
0.989/0.975/0.914. The placement version improves from its first checkpoint,
but edit distance stays worse than baseline and accuracy remains near floor.
No equal-quality, coherent-language, or equal-update win is established.
Unassisted losses on different target streams are not compared.

After four passes, nine 120-character samples per arm produced 1,080 characters
each. Baseline used 1,080 neural calls and 0.154 total CPU seconds. Separated
generation used **691 neural calls, 243 index lookups and 70 indexed emissions**,
taking 0.103 total CPU seconds, including index lookup and decoding. This is
about 33% less CPU in these short samples, not a stable universal speed ratio.
Both sets remain incoherent; correct literal spelling is not correct meaning.

This is a serialized look-ahead prototype, not a concurrent-worker throughput
test. The new experiment is isolated: live training and old checkpoints are
unchanged. Each new run preserves a pre-training protocol, code/input/bank
hashes, original text, extracted stream/offsets/remainder, training-example
hashes, both components at 1/2/4 passes, evaluation rows and generation samples.
Historical and locked experiments are preserved. Reused Bible development
data already seen by the bank does not establish generalization.

Next, review wrong next-word predictions to distinguish index placement errors
from remainder-generation errors before scaling or changing either policy.
The user can help by identifying concrete misplaced patterns in the saved
samples; those ratings are evaluation, not automatic new training data.

##### Larger separated-index profile

```bash
python3 neural_index_pipeline.py --acknowledge-review --large
```

This profile uses complete Acts 6-12 for training and Acts 13-14 for evaluation,
five paired seeds (17, 117, 217, 317, 417), 256 shared next-word probes and
the same 1/2/4 source-pass milestones. It changes input size and seed/probe
counts, not the extraction or placement policy, model size settings, learning
rate, or 128-form selection cap. Training-present bank-backed forms are
selected anew from the larger training text, never from evaluation text.
All arms train sequentially, with alternating order across seeds. Both
components' costs and generation samples are reported using the same schema.
This is reused development evidence, not fresh untouched evaluation.

Run `6045eb140d8e47b1a5c6890199251039` completed all ten arms and
30 component checkpoints: **30,905 training / 9,730 evaluation characters**,
16.2/6.5 times the small profile. Both bases have 63 character outputs,
2,103 parameters and identical paired initialization. The 128 selected forms
account for 1,180 occurrences / 6,380 removed characters, leaving 24,525
remainder characters. Coverage is 20.6%, below the small run's 35.9% because
the selection cap stayed fixed while the corpus expanded.

| Four-pass mean | Baseline | Remainder base + placement index |
|---|---:|---:|
| Base updates | 123,620 | 98,100 |
| Base training CPU seconds | 64.813 | 51.264 |
| Placement learning CPU seconds | 0 | 0.124 |
| Total training CPU seconds | 64.813 | 51.388 |
| Exact next-word accuracy | 1.25% | 2.73% |
| Normalized word edit distance (lower better) | 0.879 | 0.888 |

Total training CPU is **20.7% lower**, including index learning. Baseline
per-seed CPU ranged from 63.75-66.34s; separated ranged from 50.81-51.87s.
Shared loading/validation/extraction/probe setup took 0.073 CPU seconds,
separate from per-arm examples setup. Prior bank construction remains excluded.
The placement learner has 3,148 distinct context rows in addition to the neural
weights; its memory and work are not treated as absent.

At 1/2/4 passes, mean exact-word accuracy was 1.56/1.17/1.25% baseline
versus 3.59/4.14/2.73% separated. Edit distance was 0.872/0.856/0.879
versus 0.905/0.897/0.888. The separated arm's exact accuracy is higher
at all measured milestones but falls between two and four passes; its edit
distance improves while remaining worse than baseline. These are mixed,
low-quality results, not stable convergence or an overall quality win.

Fifteen 120-character samples per arm produced 1,800 characters each.
Baseline used 1,800 neural calls / 0.289 total CPU seconds. Separated used
1,337 neural calls, 423 index lookups and **71 indexed emissions**, taking
0.228 total CPU seconds including index lookup and decoding (about 21% less).
Both sample sets remain incoherent. These short generation timings are
descriptive, not statistically established speed ratios. The larger run
confirms cost savings at this coverage, not equal-quality efficiency or
generalization. No policy, selection-cap or learning-rate tuning was performed
after inspecting results.

##### Different corpus: everyday prose

The user approved newly authored everyday prose, approximately 2,000 training
and 1,000 evaluation characters. `fixtures/neural-everyday-prose.json` contains
two separate workshop passages, **1,942 / 957 characters**, authored by the AI
assistant for this experiment, not copied from an external work. Provenance,
approval limitations and authorship are recorded. AI authorship is not a
guarantee of exclusive rights or commercial clearance. Both passages were
authored together before measurement: this is a controlled artificial
development corpus, not independently collected natural text or a locked test.

```bash
python3 neural_index_pipeline.py --acknowledge-review --corpus fixtures/neural-everyday-prose.json
```

The JSON input must include nonempty `title`, `provenance`, `rights_note`,
`limitations`, `training` and `evaluation` fields. Distinct passages of
64-50,000 characters are required, plus explicit command-line approval.
`--corpus` and the Acts `--large` profile are mutually exclusive. File and
passage hashes and the complete original record are saved before training.
Malformed or missing input raises an error, never falls back to Acts.

Run `acb0e2c7c63749f7b7fa2d47c363bd1b` used three paired seeds, 48
shared word probes and unchanged 1/2/4-pass training, learning rate, placement
policy and 128-form cap. The existing Bible-derived bank was **not expanded**;
85 bank-backed forms present in training qualified. It extracted 114
occurrences / 607 characters (31.3%), leaving 1,335 base training characters.
Both bases have 44 outputs / 1,628 parameters and identical paired initialization.
The evaluation passage never selects forms or updates either component.
Regression tests change evaluation text and verify identical saved base weights
and placement counts at every milestone.

| Four-pass mean | Baseline | Remainder base + placement index |
|---|---:|---:|
| Base updates | 7,768 | 5,340 |
| Base training CPU seconds | 3.232 | 2.215 |
| Placement learning CPU seconds | 0 | 0.0079 |
| Total training CPU seconds | 3.232 | 2.223 |
| Exact next-word accuracy | 5.56% | 2.78% |
| Normalized word edit distance (lower better) | 0.832 | 0.910 |

Total training CPU is **31.2% lower**, counting placement learning. Shared
loading/validation/extraction/probe setup took 0.063 CPU seconds separately;
prior bank construction remains excluded. The placement learner has 642
context rows in addition to the neural weights.

At 1/2/4 passes, accuracy was 5.56/11.81/5.56% baseline versus
1.39/4.17/2.78% separated. Edit distance was 0.810/0.757/0.832 versus
0.954/0.852/0.910. **Both quality measures favor baseline at every recorded
milestone.** The tiny models remain poor, but low absolute accuracy is not a
reason to dismiss this relative quality loss. No favorable checkpoint was
selected or policy tuned after observing it.

Nine 120-character samples per arm produced 1,080 characters each. Baseline
used 1,080 neural calls / 0.144 total CPU seconds. Separated generation used
701 neural calls, 289 index lookups and 71 indexed emissions / 0.096 total
CPU seconds (about 33% less in these short samples). Both samples remain
incoherent. The experiment demonstrates cost savings on a different small
corpus, **not quality-preserving transfer**. Removing base context and relying
on this simple placement mechanism has a predictive quality cost; this test
does not isolate which component contributes how much.

Original Bible reports, bank contents and frozen locked code remain unchanged.
Custom input changes the corpus, not the live training behavior or memory
authorization. A useful next user contribution is to review the saved incorrect
predictions for wrongly placed indexed forms versus malformed remainder words;
such evaluation does not authorize those reviews as new training data.

##### Four-arm candidate cascade

The user requested a small comparison of base alone, Layer 0 + base,
Layers 0-1 + base and Layers 0-2 + base, explicitly keeping base separate
from the layers. They selected both repeated 3-4-character blocks and hidden
patterns for Layer 2. Old Layer 1 exclusion rules are preserved for later
generation/input lab work, **not enabled in this comparison**.

```bash
python3 neural_index_pipeline.py --acknowledge-review \
  --corpus fixtures/neural-everyday-prose.json --candidate-layers
```

`neural_candidate_layers.py` provides an isolated reversible extraction codec
and a separate placement index; the historical codecs and Layer 0 constants
are unchanged:

- Layer 0 recognizes the existing seven exact seed configurations. Source-start
  forms are explicitly source-scoped; longer matches take precedence. `A`/`I`
  source prefixes are eligible observations, not declarations of complete words.
- Layer 1 uses the existing training-present bank-backed complete forms, capped
  at 128. It does not use the historical death/exclusion rules.
- Layer 2 discovers repeated alphabetic substrings of length 3 or 4 in
  unresolved spans, with support >=3 and a cap of 128, ranked by characters
  saved. It also considers up to 128 complete remaining words backed by the
  frozen hidden1/hidden2 bank. These are experimental candidates, not absolute
  admissions. Literal `*` is never interpreted as an unknown input character.
- No match can span an earlier reference. Within each stage, longest exact
  match wins with deterministic tie breaking. Typed reference IDs cannot collide
  with literal characters, and the original source offsets and spellings are
  preserved. Decoding the stream exactly reconstructs the source.
- The final base learns only concatenated remainder characters, with no indexed
  output classes, embeddings or targets. A separate four-event context-count
  placement learner sees reference identities and chooses a reference or base
  character. All layered arms learn placement at **every stream event**, not
  only word boundaries, so substring insertion is possible. This differs from
  the earlier boundary-only separated-index experiment; compare arms within
  this newly recorded run, not their timings as if the policies were identical.
- Reference emission advances only the placement history, not the base's
  remainder context. The old exclusions and extra word-delimiter guard are
  disabled consistently in all layered arms. Consequently a correctly spelled
  indexed form may still be inserted incorrectly or joined into an invalid word.
  Correct extraction is not correct placement.

Run `1bec5fa35edc4e83b316128801341c78` used the fixed workshop prose
(1,942 / 957 training/evaluation characters), three paired seeds, 48 identical
word probes and 1/2/4 source-pass milestones. All four bases have 44 outputs,
1,628 parameters and identical initialization within each seed. Execution order
rotates across arms. No text, policy or hyperparameters were tuned after seeing
results.

| Four-pass mean | Base | L0 + base | L01 + base | L012 + base |
|---|---:|---:|---:|---:|
| Remainder characters/pass | 1,942 | 1,891 | 1,284 | 943 |
| Base updates | 7,768 | 7,564 | 5,136 | 3,772 |
| Total training CPU seconds | 3.103 | 3.072 | 2.094 | 1.527 |
| Placement learning CPU seconds (included) | 0 | 0.0314 | 0.0232 | 0.0206 |
| Exact next-word accuracy | 5.56% | 5.56% | 3.47% | 1.39% |
| Word edit distance (lower better) | 0.832 | 0.792 | 0.864 | 0.946 |

Layer 0 removes 51 characters in 17 occurrences. Layer 1 adds 607 characters
in 114 complete-form occurrences. Layer 2 adds 341 characters in 108 repeated
block occurrences. **No hidden-backed complete forms add coverage**: the
earlier whole-form stage already handles those eligible forms in this corpus.
The implementation has a separate fixture test for the hidden-backed route,
but this training result does not measure a separate hidden-pattern benefit.
This is not a test of resolving actual missing-letter input such as `th?t`.

Layer 0 matches baseline accuracy and improves edit distance descriptively,
but training CPU changes only about 1%. Layers 0-1 save about 33% training CPU;
Layers 0-2 save about 51%. Both sacrifice word prediction quality in this run.
The combined setup/discovery/extraction/probe cost for all configurations is
0.269 CPU seconds, separately reported, not included in the update savings.
Prior bank construction remains excluded. Both base and placement learning
costs are included in training CPU. Extra placement context rows are persisted
and are not counted as neural parameters.

Nine 120-character samples per arm produced 1,080 characters each:

| Generation total | Base | L0 + base | L01 + base | L012 + base |
|---|---:|---:|---:|---:|
| CPU seconds | 0.138 | 0.146 | 0.103 | 0.070 |
| Base neural calls | 1,080 | 1,062 | 742 | 480 |
| Index lookups | 0 | 1,068 | 813 | 621 |
| Indexed emissions | 0 | 6 | 71 | 141 |

Layer 0's lookup overhead outweighs its small generation saving here.
Layers 0-2 reduce short-sample generation CPU by about 49%, but samples
remain incoherent and joins can be malformed. Timing samples are descriptive,
not stable universal speed ratios or quality-matched convergence.

The goal of >30% improvement across costs **and** quality has not been met.
Greater extraction coverage saves work but worsens reconstruction/prediction
in this simple placement architecture. All 36 component checkpoints,
pre-training protocol, code/input/bank hashes, typed references, stage offsets,
training-example hashes, candidate evidence, evaluation rows and samples are
preserved in a new run directory. The old Layer 0/1/2 benchmark implementations,
locked evidence, admission registry and live training remain unchanged.

##### Selective L1 trial

The user approved testing selective L1 between L0 alone and full L01:

```bash
python3 neural_index_pipeline.py --acknowledge-review \
  --corpus fixtures/neural-everyday-prose.json --candidate-layers --selective-l1
```

The gate was fixed before the run: >=2 training occurrences, >=6 potential
saved characters (`occurrences * (length - 1)`), and >=80% of occurrences
with a preceding up-to-four-character context seen with that word >=2 times
and >=90% next-word purity. Context alternatives include all complete
boundary-aligned words in training, not just bank words. This is a conservative
empirical placement heuristic, not a proof of uniqueness. Gate evaluation,
accepted/rejected words and context counts are saved in the pre-training
protocol. Evaluation text never participates in selection. Rejected forms stay
in the base training text. Layer 0, placement learning, model settings and
training data are unchanged; Layer 2 and old exclusions are not enabled.

Run `a951e691e1434ae9b945ba699fa46154` completed all four arms across
three paired seeds, with 36 component checkpoints and the same 48 word probes.
**Zero of 85 eligible L1 forms passed.** Even the repeated words did not have
contexts meeting the recorded repetition/purity requirement. The selective arm
therefore has exactly the same extraction, training examples, learned placement
counts, weights and generated text as L0 alone; minor timing differences are
measurement noise, not an algorithmic improvement.

| Four-pass mean | Base | L0 + base | Selective L01 + base | Full L01 + base |
|---|---:|---:|---:|---:|
| Total training CPU seconds | 3.132 | 3.094 | 3.092 | 2.116 |
| Base updates | 7,768 | 7,564 | 7,564 | 5,136 |
| Exact next-word accuracy | 5.56% | 5.56% | 5.56% | 3.47% |
| Word edit distance (lower better) | 0.832 | 0.792 | 0.792 | 0.864 |

Total training CPU includes placement learning. Combined setup/selection/
extraction took 0.157 CPU seconds separately; prior bank construction remains
excluded. Nine 120-character samples per arm used 1,080/1,062/1,062/742 neural
calls and 0.138/0.148/0.146/0.104 total CPU seconds, with 0/6/6/71 indexed
emissions. Short generation timings are descriptive.

This gate is **too restrictive on this corpus to test intermediate coverage**.
The trial neither establishes the proposed sweet spot nor rules one out.
It preserves the previously observed L0 quality and full-L1 cost/quality
tradeoff. Thresholds were not loosened after inspecting results, and the zero
selection outcome is retained. A future gate or larger training corpus would
be a separate development experiment, not a revision of this evidence.

##### Standalone L2 ablation

The user requested L2 on its own: **L2 + separate remainder base**, not removal
of the base. Use the same reviewed small prose corpus:

```bash
python3 neural_index_pipeline.py --acknowledge-review \
  --corpus fixtures/neural-everyday-prose.json --candidate-layers --l2-only
```

L0 and L1 extraction are disabled. The unchanged L2 rules discover repeated
alphabetic 3/4-character blocks (support >=3, cap 128) and training-present
hidden-backed complete forms (cap 128) directly from the full training text.
Longest exact match wins between those candidates; no literal missing-character
inference is enabled. Both placement learning and remainder training follow
the previous cascade policy. No old generation exclusions or delimiter guards
are enabled. No thresholds, data or learning rates were tuned after results.

Run `2e0ac7500b454704a967533ec608a814` used three paired seeds, 48
word probes and the fixed 1,942/957 training/evaluation characters. L2 extracts
1,040 characters in 251 occurrences: 138 repeated-block matches and 113
hidden-backed complete matches, leaving 902 remainder characters. All extracted
references belong to L2; neither foundation nor whole-form L1 entries exist
in this index. Both bases keep identical paired initializations and
44 character outputs / 1,628 parameters.

| Four-pass mean | Baseline | L2 + base |
|---|---:|---:|
| Base updates | 7,768 | 3,608 |
| Total training CPU seconds | 3.233 | 1.542 |
| Placement learning CPU seconds (included) | 0 | 0.0202 |
| Exact next-word accuracy | 5.56% | 0.69% |
| Word edit distance (lower better) | 0.832 | 0.953 |

Training CPU is about 52% lower, including placement learning. Combined setup/
discovery/extraction took 0.164 CPU seconds separately; prior bank construction
is excluded. L2 accuracy was 0/0/0.69% at 1/2/4 passes; the corresponding
edit distances were 0.970/0.970/0.953. Both quality measures favor baseline.

Nine 120-character samples per arm produced 1,080 characters each.
Generation CPU was 0.144s baseline versus 0.085s L2 (~41% less in these
short samples), with 1,080/582 base calls, 708 L2 lookups and 126 indexed
emissions. Timings are descriptive; samples remain incoherent. More extraction
again saves work without preserving quality. Hidden-backed coverage is now
nonzero, but because both L2 families are active together this does not isolate
the hidden family's independent contribution. It does not show generalization
or a quality-matched efficiency win.

New protocol, candidate snapshots, extraction offsets, code/input/bank hashes,
18 component checkpoints, probe rows and generated samples are saved without
overwriting earlier comparisons. Layer 0 constants, admissions, frozen code
and live training remain unchanged.

##### Intermediate L1 coverage

The earlier context-purity trial selected nothing. This separate experiment
keeps that result intact and tests both user-requested intermediate fractions:

```bash
python3 neural_index_pipeline.py --acknowledge-review \
  --corpus fixtures/neural-everyday-prose.json --candidate-layers --savings-l1
```

Only bank-selected complete forms occurring at least twice in training qualify.
Rank by occurrence count times (length minus one), breaking ties lexically;
select the first ceil(25% or 50% of the repeated-form count). These are fractions
of forms, not characters removed or the entire bank. Context purity does not
gate this experiment: repetition/savings do not prove reliable placement.
Selections and their evidence are saved before updates. Layer 0 is unchanged,
Layer 2 is absent, rejected forms remain base targets, and the base remains
separate. Existing placement rules and deferred generation exclusions are
unchanged. No thresholds were changed after evaluation.

Run `8656a80e546e45f7b7d723f96043f8df` used the fixed workshop prose
(1,942/957 training/evaluation characters), three paired seeds, 48 shared probes
and 1/2/4-pass milestones. Of 85 bank candidates, 18 repeat. The quarter arm
selects five: helper, looks, morning, something, table. The half arm selects
those five plus brings, needs, repair, that. Both bases retain 44 character
outputs and 1,628 parameters, with paired identical initializations.

| Four-pass mean | Base | L0 | L0 + 25% L1 | L0 + 50% L1 | Full L01 |
|---|---:|---:|---:|---:|---:|
| Remainder characters per pass | 1,942 | 1,891 | 1,806 | 1,751 | 1,284 |
| Training CPU seconds, including placement | 3.183 | 3.213 | 3.022 | 2.924 | 2.122 |
| Exact next-word accuracy | 5.56% | 5.56% | 5.56% | 4.17% | 3.47% |
| Word edit distance (lower better) | 0.832 | 0.792 | 0.822 | 0.825 | 0.864 |

The quarter arm saves about 5% training CPU relative to base with matching
word accuracy and slightly better edit distance at four passes. The half arm
saves about 8%, but loses word accuracy. Both have worse edit distance than L0.
Combined discovery/setup cost is separately 0.206 CPU seconds; bank construction
is excluded. There is no equal-update or quality-matched timing comparison.

| Word accuracy / edit distance | Base | L0 | Quarter | Half | Full L01 |
|---|---|---|---|---|---|
| 1 pass | 5.56% / .810 | 6.94% / .793 | 7.64% / .801 | 4.17% / .821 | 3.47% / .888 |
| 2 passes | 11.81% / .757 | 14.58% / .738 | 6.94% / .846 | 6.25% / .809 | 4.17% / .927 |
| 4 passes | 5.56% / .832 | 5.56% / .792 | 5.56% / .822 | 4.17% / .825 | 3.47% / .864 |

Nine 120-character samples per arm used 1,080/1,062/997/955/742 base calls
and 0/6/20/24/71 indexed emissions, respectively. Their total generation CPU
was 0.140/0.150/0.140/0.132/0.104 seconds. Quarter coverage did not measurably
improve generation CPU here. These short timings are descriptive, not a robust
speed guarantee; samples are still incoherent.

The extraction levels now differ, but a stable sweet spot is not established:
quarter coverage loses to base at two passes and both intermediate arms lose
to L0 on edit distance at every milestone. Reused artificial development data,
few probes and three seeds cannot establish generalization or significance.
All 45 checkpoints, selections, offsets, protocol and code/input hashes persist
alongside the earlier trials. To contribute a stronger next test, propose a
new short prose passage with provenance and a separately written evaluation
passage before running it; do not rewrite this fixture to improve the scores.

##### Corrected L1: unique fragments

The user clarified that L1 must identify a complete form from a discriminating
partial key, not simply index any bank-backed complete word. Earlier whole-word
trials do not test this corrected eligibility rule; their evidence remains
preserved, and their modes remain historical controls.

```bash
python3 neural_index_pipeline.py --acknowledge-review \
  --corpus fixtures/neural-everyday-prose.json --candidate-layers --unique-l1
```

For each complete training-present word, consider every nonempty consecutive
substring shorter than the whole word. Require exactly one distinct owner in
the expanded SCOWL 2020.12.07 inventory (644,145 normalized entries), union all
training words as additional ambiguity vetoes. Lowercase and normalize curly
apostrophes for comparison; preserve source casing/spelling in the payload.
Names, compounds, variants and inflections remain possible competitors.
Choose the shortest qualifying key, then lexical order for ties. No cap,
frequency gate, boundary/length hints or whole-word fallback. Evaluation words
do not enter selection. Save the inventory manifest, hashes and selected keys
in the protocol before learning. Missing or altered inventory fails explicitly.

The key is an identifier, not a completion prediction. Extraction still
confirms the exact complete bounded source spelling and removes its entire
payload, preserving offsets and earlier L0 references. Seeing a key embedded
in an unknown longer word does not automatically delete or replace that word.
Base trains only the concatenated remainder; the separate placement learner
is unchanged and remains statistical. Uniqueness does not guarantee correct
placement. No old generation exclusions or L2 are enabled in this trial.
The historical inventory is not all English or Oxford/Webster certification.

Run `0cb318e5564046c9953313c8a0d0e9a1` reused the fixed 1,942/957-character
workshop prose, three paired seeds, 48 shared next-word probes and 1/2/4 passes.
Only two training words qualify:

| Shortest key | Complete payload | Lengths |
|---|---|---|
| `eturns` | returns | 6 / 7 |
| `nplugge` | unplugged | 7 / 9 |

Both occur once. Corrected L1 extracts all 16 characters, not just the keys.
L0 still extracts 51 characters; corrected L01 extracts 67 total, leaving
1,875 remainder characters versus L0's 1,891. Every paired base keeps the
same initial weights, 44 character outputs and 1,628 parameters.

| Four-pass mean | Base | L0 | Corrected L01 |
|---|---:|---:|---:|
| Training CPU seconds, including placement | 3.137 | 3.099 | 3.074 |
| Base updates | 7,768 | 7,564 | 7,500 |
| Exact next-word accuracy | 5.56% | 5.56% | 15.97% |
| Word edit distance (lower better) | .832 | .792 | .727 |

The four-pass accuracy change is +10.42 percentage points versus base.
All three corrected seeds outperform their paired controls at four passes,
but the milestone curve does not show a consistent win:

| Word accuracy / edit distance | Base | L0 | Corrected L01 |
|---|---|---|---|
| 1 pass | 5.56% / .810 | 6.94% / .793 | 2.08% / .790 |
| 2 passes | 11.81% / .757 | 14.58% / .738 | 9.03% / .791 |
| 4 passes | 5.56% / .832 | 5.56% / .792 | 15.97% / .727 |

Training CPU decreases only about 2% versus base, while combined setup,
including the broad uniqueness scan, takes 3.828 CPU seconds separately.
Prior dictionary/bank construction is excluded. Nine 120-character samples
per arm took .139/.145/.147 CPU seconds, with 1,080/1,062/1,071 base calls
and 0/6/3 indexed emissions. No generation-speed improvement is demonstrated.
Removing two words also changes remainder contexts and shuffled update order;
this test does not isolate a causal explanation for the quality difference.

All 27 component checkpoints, 27 final samples, source/extraction hashes,
selected-key evidence and the preregistered protocol persist separately.
No thresholds, corpus, or checkpoints were selected after evaluating results.
The small reused artificial corpus is development evidence, not a general
language, coherence, statistical-significance or quality-matched speed claim.
For a next contribution, provide new prose and a separately authored evaluation
passage with provenance before measurements rather than altering this fixture.

##### Pretraining and protected reference injection

The user requested a fresh base trained only on L0/L1 before passage learning,
then chose a protected pretrained representation whenever a word is indexed.
This is not repeated spelling training or merging old gradients into current
weights. Prior models, corpora and experiment reports remain unchanged.

First, pretraining-only trial `de6da018f074421da88ac2b454e572e1` compared
base/L0/corrected-L01/pretrained-L01 on the same small prose, three seeds and
1/2/4 passes. Fixed pretraining used 200 shuffled passes (rate .1, seed+3)
over nine isolated literals: `" A "`, `" I "`, `" a "`, `"A"`, `"A "`,
`"I"`, `"I "`, `"returns"`, `"unplugged"`. Source-start notation is metadata,
not literal training characters. Context resets for each literal; no artificial
joins, evaluation text, partial-key targets or previous model weights.
There are 31 character targets per pretraining pass, 6,200 updates per seed.
No end-marker target is added; this teaches internal spelling, not termination
or which word to choose from an empty context.

Each checkpoint is saved and restored before remainder training. The latter
uses the same 1,875 remainder targets per pass as corrected L01, with no replay
of indexed occurrences. All arms use the common training/payload alphabet,
44 outputs and 1,628 trainable parameters; original seeded weights match.
Ordinary pretrained L01 reached 9.72% word accuracy / .770 edit distance at
four passes, versus 15.97% / .727 for cold corrected L01 and 5.56% / .832 for
base. Total training CPU was 5.735s versus 3.148s corrected L01 and 3.184s base,
including mean 2.541s pretraining. Setup was separately 3.931s.

Pretraining teacher-forced spelling accuracy reached 23/31 (74.2%) for all
three seeds. That is the maximum on these examples: several isolated patterns
have identical starting contexts but different targets. Given each L1 word's
first character, the pretrained base greedily completes `returns` and
`unplugged` exactly in all three seeds. After remainder training, unconditioned
teacher-forced accuracy fell to mean 8.6%. This does not meet a permanent
"learn once" objective; shared neural weights forget despite omitted targets.
The index still emits exact stored payloads independently.

Protected injection adds a fifth arm:

```bash
python3 neural_index_pipeline.py --acknowledge-review \
  --corpus fixtures/neural-everyday-prose.json --candidate-layers \
  --unique-l1 --pretrain-l01 --inject-l01
```

Protocol fixed before training:

- Each seed produces one fresh pretraining-only checkpoint, reused identically
  by ordinary-pretrained and protected-injection arms. Pretraining CPU is
  measured once and charged to both arms, explicitly marked as inherited
  timing for the second consumer. A new experiment does not reuse an old run.
- The frozen checkpoint encodes each complete reference payload from reset
  context. Average its 16 hidden activations after each character into a
  protected vector. This is a spelling-derived representation, not a proven
  semantic embedding or lossless neural decoder. Exact payloads and full
  frozen weights are separately retained.
- At each recognized training reference, use its vector for subsequent
  remainder targets until the next reference. No future references, indexed
  character targets, or evaluation-derived vectors enter training.
- Add the vector to the active hidden preactivation before tanh. It is an
  input, not a weight overwrite or stored gradient. No extra trainable
  parameters. The active base starts from the same pretrained weights as the
  ordinary pretrained arm and learns how to use this input on remainder only.
- Generation reconstructs the latest reference from each prompt and refreshes
  conditioning on every indexed emission. Each new prompt resets this state;
  no reference means zero conditioning. Index emission still needs no neural
  spelling forward. All vector setup and conditioned forward costs are counted.
- Frozen features are tuple-valued in memory and saved in each index snapshot;
  shape, finiteness, reference coverage and unchanged hashes are checked.
  Active weights may forget or ignore an unchanged vector: protected storage
  is not proof of preserved useful behavior.

Run `2576af579feb44968e59958ea4e1f1c4` kept the corpus, two corrected L1
words, 48 probes, seeds and all learning rates fixed. Same source extraction
in all three corrected-L01 arms; no post-result changes or checkpoint picking.

| Four-pass mean | Base | L0 | Corrected L01 | Pretrained L01 | Protected injection |
|---|---:|---:|---:|---:|---:|
| Total training CPU seconds | 3.184 | 3.128 | 3.077 | 5.693 | 5.695 |
| Word accuracy | 5.56% | 5.56% | 15.97% | 9.72% | 8.33% |
| Word edit distance (lower better) | .832 | .792 | .727 | .770 | .773 |

Mean pretraining cost is 2.524s per pretrained arm; injection feature setup is
another .0043s. Combined corpus/inventory setup is 3.953s separately; original
dictionary/bank construction is excluded. No amortized or equal-total-update
efficiency claim is made.

| Word accuracy / edit distance | Base | Corrected L01 | Pretrained L01 | Protected injection |
|---|---|---|---|---|
| 1 pass | 5.56% / .810 | 2.08% / .790 | 4.86% / .792 | 2.78% / .826 |
| 2 passes | 11.81% / .757 | 9.03% / .791 | 6.94% / .813 | 6.94% / .805 |
| 4 passes | 5.56% / .832 | 15.97% / .727 | 9.72% / .770 | 8.33% / .773 |

Injection conditions 1,837 of 1,875 remainder targets per pass. Across nine
120-character generation samples it actually conditions 348 of 1,050 neural
calls, with ten indexed emissions; total generation CPU .151s versus .138s
baseline. Ordinary pretraining used 1,077 calls/one emission/.152s;
cold corrected L01 1,071 calls/three emissions/.149s. Short noisy timings
are descriptive, and samples remain incoherent.

The protected checkpoint and vectors remain unchanged, but active unconditioned
spelling recall still falls. The experiment demonstrates persistence and use
of frozen features, not that they prevent all forgetting or improve placement.
Injection fails to beat ordinary pretraining or corrected L01 on four-pass
quality. This small, reused artificial corpus cannot establish generalization,
significance or broad architectural superiority. A useful next contribution is
to specify a concrete recall task (what context is supplied and what must be
returned) before testing another representation, rather than tuning on these
evaluation scores.

Saved evidence includes three frozen models, 45 component checkpoints,
45 final samples, per-seed vectors and hashes, exact initial/starting weights,
selections, source offsets, protocol and code hashes. Default neural behavior
remains unconditioned outside the explicit experimental path.

##### Atomic contextual L01

The user approved a clean contextual-token test without pretraining or frozen
injection. Corrected L1 recognition is unchanged: shortest unique fragments
against the expanded inventory plus training-word vetoes identify `returns`
and `unplugged`. The complete payload is represented by a typed reference,
not by a guessed completion or by internal spelling targets.

```bash
python3 neural_index_pipeline.py --acknowledge-review \
  --corpus fixtures/neural-everyday-prose.json --candidate-layers \
  --unique-l1 --atomic-l01
```

Three arms start from fresh seeded MLPs:

1. Character-only base, unchanged.
2. Corrected L01 remainder-only base plus separate context-count placement.
3. Atomic corrected L01: the original ordered stream contains references and
   characters together. Every reference takes one target and one context
   position; its input embedding and output selection weights are trainable.
   The neural model itself chooses references or characters, with no count
   placement learner. Fixed decoding emits an entire payload at once.

The existing UnitCodec cannot be used directly for this experiment: it
requires hidden-letter skeleton evidence and word-string token identities,
and its generation applies delimiter guards. The small AtomicCandidateIndex
adapter instead reuses the corrected CandidateCodec's typed identities and
lossless layer cascade, along with the same original NeuralModel/SGD.
It preserves the candidate control's source-start and remaining-length masks,
without introducing a new word-boundary guard. Malformed joins remain possible.
Each prompt encodes a final unfinished form conservatively, as in the control.
Atomic snapshots identify their mode explicitly and restore to the right
sampler, rejecting placement counts or injected features.

Protocol fixes three seeds, 48 common next-word probes, .1 learning rate,
context four, embedding eight, hidden sixteen and 1/2/4 source passes. No
end-marker target is added. No corpus, threshold, checkpoint or evaluation
probe tuning after results. All nine L0/L1 reference identities are retained,
including overlapping source-start forms. Context length is four events for
atomic, four remainder characters for the remainder control. This is an
intentional behavioral difference, not equivalent character windows.

Run `b08a426d36bb41e988ab8a2ebac12d84` reused the fixed 1,942 training /
957 evaluation characters. Both layered arms recognize the same 19 occurrences
(17 L0, two L1), covering 67 characters. Remainder-only has 1,875 targets/pass;
atomic has 1,894 (1,875 characters plus 19 references); base has 1,942.
Atomic uses 53 outputs and 1,853 parameters versus 44 / 1,628 in the controls.
Seeds are shared, but atomic's different matrix shapes mean initial weights
are not identical to the controls. No parameter-matched claim is made.

| Four-pass mean | Base | Remainder L01 | Atomic contextual L01 |
|---|---:|---:|---:|
| Training CPU seconds, including placement where used | 3.115 | 3.057 | 3.432 |
| Updates | 7,768 | 7,500 | 7,576 |
| Exact next-word accuracy | 5.56% | 15.97% | 5.56% |
| Word edit distance (lower better) | .832 | .727 | .788 |

Atomic reduces update count by about 2.5% versus base, but its wider dense
output costs more: training CPU is about 10% higher. Combined inventory/
recognition/probe setup is 3.814s separately; prior dictionary and bank
construction are excluded. There is no matched-update or quality-matched
efficiency conclusion.

| Word accuracy / edit distance | Base | Remainder L01 | Atomic L01 |
|---|---|---|---|
| 1 pass | 5.56% / .810 | 2.08% / .790 | 2.08% / .820 |
| 2 passes | 11.81% / .757 | 9.03% / .791 | 0% / .806 |
| 4 passes | 5.56% / .832 | 15.97% / .727 | 5.56% / .788 |

Usage metrics explicitly separate spelling from placement. A correct L1
selection requires a layer-one emission at output offset zero that equals the
target AND produces the exact next word (a wrong following join fails).
The same recent-reference subset is used for every arm: a recognized form
within the preceding four encoded prompt events. Missing support produces
null precision/accuracy rather than a perfect or zero-quality claim.

On this fixed fixture, all 48 probes have **zero L1 target opportunities and
zero recent-reference contexts**. No L1 reference is selected at the beginning
of a probe. Therefore the intended L1-usage and surrounding-word effects are
not measurable here, despite correct wiring. Nine 120-character samples per
arm use 1,080/1,071/1,038 neural calls and 0/3/21 reference emissions;
all emitted references are L0, not L1. Total generation CPU is
.136/.146/.149s, respectively. Short timings are descriptive; no coherence or
generation-speed improvement is established.

The test demonstrates atomic target/context training and exact decoding, not
better L1 word usage. The next useful evidence requires a new preregistered
corpus with enough qualifying words in training and independently authored
evaluation contexts that actually exercise them, not altering this evaluation
set after seeing its scores. Each current L1 payload appeared only once per
training pass. Historical results remain preserved; live training is unchanged.

All 27 component checkpoints, 27 final samples, source/extraction/code hashes,
atomic token vocabulary and per-probe usage evidence are saved. Tests verify
reference targets, reference-conditioned next-character contexts, complete
payload decoding, masks, restored atomic dispatch and correct-usage scoring.

##### Matched pretraining and occurrence reinforcement

The user chose equal pretraining for the full-character and indexed arms,
plus one extra character-training pass over every recognized literal in both.
This tests equal access to initial spelling knowledge, not no-retraining.

```bash
python3 neural_index_pipeline.py --acknowledge-review \
  --corpus fixtures/neural-everyday-prose.json --candidate-layers --unique-l1 \
  --pretrain-l01 --matched-reinforcement
```

Protocol: same fixed workshop prose, three seeds, 48 probes, rate .1 and
1/2/4 passage passes. Build one new L0/L1-only checkpoint per seed (200
shuffled isolated-literal passes, 6,200 updates) and restore the exact same
weights into both arms. Common alphabet, 44 outputs and 1,628 parameters.
No prior trained run, injection, atomic neural tokens or evaluation replay.
The indexed arm retains corrected unique-fragment recognition and separate
placement learning; the character base sees the full source.

Walk the original passage in source order. At each recognized L0/L1 span,
both arms first replay its full literal once with reset context. The base
then trains the span's ordinary character targets; indexed L01 skips those
targets and continues on its concatenated remainder. All other ordinary
contexts match their respective full/remainder stream exactly; isolated
refreshes do not contaminate the ordinary context. Both arms switch from the
historical shuffled ordinary updates to source order to make replay timing
literal and matched. Thus comparisons to old shuffled runs do not isolate
reinforcement alone. Earlier protocols and results remain preserved.

There are 19 nonoverlapping recognized occurrences per pass: 17 L0 plus
`returns` and `unplugged` once each. Their literals contain 67 characters:
both receive 67 extra character updates per passage pass. No additional
end-of-pass rehearsal. By four passes, ordinary/replay/pretraining updates are
7,768/268/6,200 for base (14,236 total) and 7,500/268/6,200 for indexed L01
(13,968 total). Equal replay is not equal total updates; the full base also
sees recognized characters in ordinary passage training.

Run `02a74582812e4a728d886b19a80d2268`:

| Four-pass mean | Matched character base | Matched indexed L01 |
|---|---:|---:|
| Total training CPU, including pretraining/replay | 5.785s | 5.698s |
| Pretraining CPU (included) | 2.479s | 2.479s |
| Literal replay CPU (included) | .110s | .110s |
| Word accuracy | 2.78% | 10.42% |
| Word edit distance (lower better) | .842 | .801 |

Pretraining is measured once per seed and charged equally to both arms;
the second consumer explicitly marks inherited timing. Schedule construction
is timed separately in each entry. Combined inventory/setup CPU is 3.878s,
excluding historical dictionary/bank construction. Timings are descriptive:
roughly 1.5% lower total training CPU is not a robust speed claim.

| Word accuracy / edit distance | Character base | Indexed L01 |
|---|---|---|
| 1 pass | 6.25% / .800 | 5.56% / .783 |
| 2 passes | 12.50% / .759 | 8.33% / .796 |
| 4 passes | 2.78% / .842 | 10.42% / .801 |

At four passes, indexed L01 improves word accuracy by 7.64 percentage points,
but the character base wins both quality measures at two passes. No stable
advantage or significance is established. Mean isolated teacher-forced spelling
recall after pretraining is 74.2% (23/31; ambiguous shared start contexts limit
the ceiling). After passage training it falls to 23.7% in base and 34.4% in
indexed L01. Occurrence replay does not permanently protect shared neural
weights. Exact indexed payload emission remains separate from neural recall.

Nine 120-character samples per arm take .141/.148 CPU seconds with
1,080/1,059 neural calls; no generation-speed win. The fixed evaluation still
does not directly test the two L1 payloads or their recent-reference contexts.
Reuse of a small artificial development corpus and differences in ordinary
context/update count limit conclusions. New independent L1-focused evaluation,
fixed before training, remains necessary.

Both schedule hashes, identical pretraining checkpoint hashes and starting
weight hashes, per-stage update/CPU counts, recall rows, 18 passage checkpoints
and 18 final samples are preserved. No live model reset or deletion occurs.

##### Whole Acts matched learning

The user explicitly requested training both models on the whole book of Acts.
This opts chapters 1-28, including former test chapters 3 and 28, into this new
training experiment. Historical locked reports remain unchanged; Acts cannot
be presented as unseen or locked evidence for these models.

```bash
python3 neural_index_pipeline.py --acknowledge-review --candidate-layers \
  --unique-l1 --pretrain-l01 --matched-reinforcement --all-acts
```

The loader reuses hash/rights-validated snapshots for Acts 1-3, 4-5, 6-27 and
28, records all four manifest hashes and assembles chapters in canonical order.
No new external text is fetched. Three paired seeds and 1/2/4 source passes
retain the matched pretraining/replay policy: both full-character and indexed
bases restore identical fresh pretraining weights and get the same isolated
literal refresh per original recognized occurrence. No prior run weights are
loaded into this experiment. Generated continuations are not scripture.

Before any weight updates, require at least 20 L1 occurrences with distinct
source offsets and 32 characters of preceding context. Select 20 evenly over
that occurrence list; merge with 256 evenly distributed general word probes,
deduplicating offsets. Twenty opportunities means occurrences, not twenty
unique words or multiplying support across seeds. Evaluate those focused
targets separately and save exact offsets and prefixes in the protocol/inputs.
All checks use training text: they measure fitting familiar material, not
generalization. Lack of a word prediction improvement is reported as such;
more training is not automatically more useful intelligence.

Run `59fbb1a0a36144e588ac1faae14d9654` has 124,762 training characters,
124 qualifying L1 forms and 131 isolated pretraining literals including L0.
The extraction contains 445 L0 and 264 L1 occurrences; 3,474 characters are
removed from ordinary indexed training, leaving 121,288 per pass. Both arms
replay those same 3,474 characters per pass, in 709 literal refresh events.
The 20 dedicated checks span 18 distinct L1 words. Full-source hashes,
opportunity offsets, training policy and source manifests were persisted before
training. Results are reported only after checkpoints finish and are verified.

The completed run uses 276 distinct merged probes (256 general plus 20
dedicated targets). Both models have 67 outputs and 2,203 parameters.
Pretraining comprises 1,063 literal-character targets per pass, 212,600
updates per seed, taking mean 115.809 CPU seconds, charged equally.

| Four-pass mean | Full-character base | Indexed L01 |
|---|---:|---:|
| Ordinary updates | 499,048 | 485,152 |
| Extra replay updates | 13,896 | 13,896 |
| Total training CPU, including pretraining/replay | 407.265s | 401.318s |
| Overall in-sample word accuracy | 0.12% | 0.48% |
| Overall word edit distance | .86515 | .86475 |
| Dedicated L1 target accuracy (mean across seeds) | 0% | 1.67% |

Focused checks: base gets 0/20 in all three seeds. Indexed gets 1/20 in seed
17 and 0/20 in seeds 117 and 217 at four passes; all are zero at one and two
passes. These are the same 20 source opportunities repeated across seeds, not
60 independent examples. Overall accuracy base/indexed at one/two/four passes
is .24/.60%, 0/.36%, .12/.48%; edit .879/.861, .875/.879, .865/.865.
This is very poor fitting even on training text, not evidence that the models
have become meaningfully smarter. No cause such as capacity, ordering,
forgetting or placement was isolated here; do not infer one from these scores.

Replay costs are 7.896/7.889 CPU seconds included in totals; indexed placement
learning adds 1.999s included. Setup is separately 5.379s, excluding historical
dictionary/bank construction. Training CPU differs by only about 1.5%, with
no convincing quality/speed claim. Nine generation samples per arm take
.191/.203s, use 1,080/1,071 neural calls, and indexed emits three references.
Final checkpoints contain the actual whole-book-trained weights and saved
placement counts, separate from the pretraining-only checkpoints. They are
available to restore, but are not automatically substituted into the live
generator. Neither this run nor the old locked claims establishes unseen
generalization for these models.

Recognition preserves earlier behavior while avoiding testing every reference
at every character: candidates are grouped by layer and first literal
character before exact matching. Reference order, longest-match ties, source
boundaries and lossless decoding are unchanged.

### Active layered neural chat

**Historical, retired:** layer-free New Base now replaces this path. This
heading is retained for existing documentation links; the description below
records the earlier activation and its restoration behavior.

App chat initially selected the final whole-Acts `layers_01` remainder
checkpoint from run `59fbb1a0a36144e588ac1faae14d9654`, seed 17, four passes.
This is the first protocol seed, not a selection optimized on the reported
scores. Its 2,203 neural parameters work alongside a learned placement index
and fixed L0/L1 literal payloads. It is not the atomic or frozen-feature variant.
Its poor in-sample results above still apply: making it active does not make
it a conversational assistant, a reliable source of facts, or scripture.
The Corinthians continuation described below was subsequently selected;
both checkpoints are now retained unchanged as retired experiments.

`neural_chat.py activate --run-id 59fbb1a0a36144e588ac1faae14d9654 --seed 17 --passes 4`
checks completed-run evidence, character/weight shapes, finite values and the
presence of both layers and placement counts. It preserves the original
experiment and copies the selected checkpoint into `.data/neural-chat/`,
using its SHA256 as the filename. `active.json` stores the selection and source
provenance. Every status/generation request checks that checkpoint's hash.
The SHA256 is
`b142bd910de91f198c9c1238aeba7750024af70ad53ff2c6a3377fa4a20e7a20`.
Future experiments do not follow through into active chat automatically.

`GET /api/chat-model` reports the selection. `/api/chat` restores the saved
neural weights and CandidateIndex, generates a continuation through the
existing layered sampler, and saves the exchange normally. The assistant
memory additionally records `model_provenance`, including run, seed, passes,
source hashes and checkpoint hash. Generated memories remain excluded from
automatic retrieval-model training targets. Neural replies have no retrieved
source ID or ngram candidate-selection score. Missing/corrupt active state
returns an API error; it never silently switches to the old generator.

Normal unmarked chat does not train either the pinned neural checkpoint or
the legacy generator. In this historical layered checkpoint, neural response
ratings/corrections were saved for review only; the current conversation
learner's explicit feedback-training behavior is documented above.
Optional local-lab comparisons still use the legacy generator independently;
they never choose the active response. The legacy generator, scorer and
training data remain available in Generation. If `active.json` is absent,
the original generator/retrieval chat behavior remains intact. Rollback means
renaming that selection to an unused backup filename, not deleting any data.

Validation compares live chat text with direct checkpoint sampling, verifies
stored per-reply provenance and unchanged checkpoint bytes, and tests hash
failure/missing-file rejection. An isolated HTTP test includes a ready legacy
generator and successful optional lab to prove neither overrides neural chat.
These checks prove routing and persistence, not answer quality.

A useful contribution is to try two or three short prompts and save a
corrected reply where needed. Those reviews are not automatically training
data: future learning and evaluation still require a separately agreed scope.

### Continuing active paired models on Corinthians

The user approved continuing both the active layered model and its paired
baseline for four additional passes through all of 1 and 2 Corinthians,
then activating the updated layered checkpoint. `neural_continuation.py`
implements this separate continuation rather than resetting the original
benchmark. It uses the active seed (17 for the first continuation), not all
three historical seeds and not a score-selected seed.

Source staging retains all 29 chapter HTML/text pairs, 694 verses and the
publisher's rights statement. The publisher identifies WEB text as public
domain, with a trademark restriction against labeling modified/generated
text as WEB. The loader checks exact book/chapter/verse inventories, raw,
text and rights hashes, and re-extracts each chapter. No chat memories or
external implementation/training material are added implicitly. The source
snapshot contains 80,757 characters. Generated output is not scripture.

Resume the saved final neural weights in both arms. Restore the layered
placement counts and stable typed reference IDs, then revalidate L1 over
the same expanded possibility inventory plus all prior/new words. Append
newly qualifying payloads; do not silently retain a now-ambiguous identifier.
Index counts carry forward and learn on each new pass. Both arms receive the
same isolated-literal refresh targets per original recognized occurrence,
interleaved with full-character or remainder-only targets respectively.
Initial isolated pretraining is not repeated and Acts is not replayed.

No characters absent from the Acts alphabet occur in this Corinthians
snapshot, so both models retain 2,203 parameters. A tested extension path
appends any future alphabet entries without renumbering old tokens or
resetting old parameters; new character rows copy each arm's unknown-token
embedding/output/bias. Additional output slots change probability
normalization even before learning, so expansion must not be described as
unchanged predictions.

Save starting, one-, two- and four-additional-pass checkpoints and 256 fixed
new-book/256 fixed Acts probes with source hashes, parent checkpoint hashes
and code hashes. Before/after checks use the same extended codec and probes.
These are in-sample learning/retention checks only: no held-out test is
created, no Acts retention is guaranteed, and one paired seed is not robust
multi-seed evidence. The earlier latest-benchmark report remains untouched.

Activation occurs only after both arms finish. The prior active hash must
still match the parent selection, preventing a long-running job from
overriding a deliberate model switch. Pin the completed layered checkpoint
through the same validated chat activation path; active metadata includes
parent lineage and describes passes as *additional continuation passes*.
Source history includes Acts followed by both Corinthians books. Original
parents remain intact.

Tests independently replay every tiny-fixture update in both arms to reproduce
all six saved checkpoints, their evaluations, and placement snapshots.
Additional checks cover raw/text tampering, missing/empty verses, unchanged
old parameter slices/token IDs, stable reference IDs/counts, newly eligible
payloads, ambiguity rejection, and persistent activation. These prove the
continuation mechanism, not the model's understanding.

For a small contribution, try a short prompt in app chat after activation,
then save a corrected response for review. Reviews do not automatically
change the pinned model's weights.

Completed run `1f7ed40a7b614dbba3c612a572782ca1` continues seed 17 from the
four-pass Acts parents. Extraction finds 491 L0 plus 185 L1 occurrences per
pass, removes 2,965 characters, and leaves 77,792 remainder characters.
L1 grows from 124 to 176 forms; parameter counts remain 2,203 in each arm.
Both arms perform 11,860 literal refresh updates over the four new passes.
Ordinary updates are 323,028 baseline / 311,168 layered. Total continuation
training CPU is 184.740s / 182.646s (including refresh and layered placement
learning; excluding source/index setup, evaluation and historical training).

| Fixed 256-probe check | Baseline before | Baseline after 4 | Layered before | Layered after 4 |
|---|---:|---:|---:|---:|
| Corinthians exact word accuracy | .78% (2/256) | 1.95% (5/256) | 1.17% (3/256) | 1.17% (3/256) |
| Corinthians word edit distance | .87042 | .85525 | .88432 | .88698 |
| Acts retention exact word accuracy | .39% (1/256) | 3.12% (8/256) | 1.17% (3/256) | .39% (1/256) |
| Acts retention word edit distance | .87017 | .85935 | .85311 | .89688 |

Baseline Corinthians accuracy at one/two/four added passes is 1.17/3.52/1.95%;
layered .39/1.56/1.17%. The protocol selected four passes before results;
do not replace it post hoc with a better-scoring intermediate checkpoint.
Layered new-book accuracy does not improve and Acts retention worsens on these
probes; modest CPU savings (~1.1%) do not establish a quality/speed gain.
Neither architecture is a useful conversational model on this evidence.

The requested final layered checkpoint is active with SHA256
`5baa5edf76d6308de27346c6611ba6cd09f28f9a356102fe0b5cbb3972d7f343`.
All six saved checkpoint evaluations and retention checks were restored
exactly; source/code hashes and unchanged parent hashes were verified.
Live API output equals direct sampling from this updated checkpoint, memory
records the new provenance, and chat leaves its weights unchanged. Browser
labels show Acts followed by both Corinthians books and distinguish the
additional pass count. The old latest-benchmark result remains historical.

### First real-text continuation run

The user explicitly authorized training on the staged Acts material after
source collection. `neural_rounds.py --acts --acknowledge-review` verifies
raw HTML, extracted-text and rights-record hashes; fixes chapters 1/2/3 as
training/validation/final test; and saves that provenance and authorization
with every round. No other corpus, memory, or developmental admission is reused.
Layer 2 training evidence permits up to 50,000 characters for this explicit
offline workflow; ordinary prompt analysis retains its 2,000-character limit.

Run `d097c4699617462b8f80f33a759c246f` completed 50 rounds/5,000 updates on
Acts 1 (3,411 characters), with Acts 2 validation (5,274) and Acts 3 testing
(3,279). It used 1,978 trainable parameters. Training loss fell from 4.0602 to
2.5788; validation from 4.0603 to 2.8241; final test loss was 2.7947.
Validation had 18 characters outside the training vocabulary; test had 7.
These are mapped to one unknown token, so loss does not distinguish them.
The final checkpoint was used without test-driven checkpoint selection.

The generated continuation remains garbled with some recognizable fragments.
This is genuine real-text training but **not** demonstrated understanding,
factual recall, coherent passage generation, or faithful scripture quotation.
No independent question-answering evaluation has been run. The old generator
and developmental registry were hash-verified unchanged.

The run took about 85 seconds on this machine, so it ran offline rather than
through the API's 30-second request budget. The browser reads saved progress
and polls every three seconds after it sees a running report. Round checkpoints
and samples persist exactly as for the synthetic workflow. Commands create
independent runs unless Acts continuation is explicitly requested with
`--resume-run` and `--additional-rounds`.

Source: eBible.org World English Bible (engwebp), publicly dedicated to the
public domain according to its rights page; original wording/rights snapshots
and preprocessing notes remain in `.data/acts-web-staged/manifest.json`.
The translation name is a trademark. Model-generated text must not be
presented as scripture or as the World English Bible.

The Acts baseline was subsequently continued for another 100 rounds in child
run `5823eaa380c043dfb2692da2f4aa7657`. It restored saved weights and replayed
training-example draws (not weight updates) to recover the original random
sampling position. Tests compare continuation weights exactly with an
uninterrupted run. Parent data/configuration/weight dimensions are validated;
its checkpoint is preserved and hashed in the child report.

After 15,000 cumulative updates, training loss was 2.5803 and validation
2.8446, versus 2.5788 and 2.8241 at round 50. Additional rounds under unchanged
settings did not improve the final result. All intermediate checkpoints are
available; no lower-loss intermediate checkpoint was silently substituted.
Acts 3 was not tested again; its prior score is historical and cannot serve
as an untouched test after inspection. Generated output remains garbled.
Investigate optimization and context/capacity limitations before scaling
unchanged rounds or claiming improved understanding.

### Controlled character-context comparison

`neural_context.py --acknowledge-review` compares fresh contexts 4, 16 and 32,
using the same approved, hash-verified Acts 1 training / Acts 2 validation.
It does not read Acts 3 text or evaluate a final test. `load_acts` can now omit
the final-test split without changing its existing default behavior.

Each model processes 5,000 identical target positions in identical order from
seed 19. Tokenization remains character-level, learning rate 0.1, batch size 1,
embedding size 8 and hidden size 16. Initialization seed 17 and sampling seed
23 are fixed. Losses are full-split per-character negative log probabilities
at steps 0, 500, ..., 5,000; all final checkpoints are retained without
validation-based selection. Fixed 120-character samples use two short prompts
and the first 64 characters of training text, identically across models.

For context C and vocabulary V, parameter count is `25V + 128C + 16`.
With the Acts vocabulary these are 1,978 / 3,514 / 5,562 parameters.
Increasing context changes both capacity and initialization of later weight
groups, so improvement would not prove a context-only effect. One seed and
reused validation provide a diagnostic, not independent generalization or
understanding evidence. Larger contexts still cannot see whole passages.

Models persist their context alongside weights, exact training/validation
texts and provenance in a separate comparison directory, every 500 steps
including step 0. The comparison report is restored and polled in Generation,
separately from the prior ordinary neural report. No old run, chat model,
developmental admission or Layer 0/1/2 calculation is replaced.

Tests check exact equality with independently replayed training for every
context, held-out isolation, finite-difference gradients, checkpoint content,
legacy four-character behavior, explicit failures, and successful comparison
with no final-test files present.

Run `e0901f8e75434d288d951222d649f09b` completed all three 5,000-update
models and saved 33 weight checkpoints:

| Context | Parameters | Final training loss | Final validation loss |
|---|---:|---:|---:|
| 4 characters | 1,978 | 2.5788 | 2.8241 |
| 16 characters | 3,514 | 2.8423 | 3.2384 |
| 32 characters | 5,562 | 3.0405 | 3.3811 |

The four-character weights exactly reproduce the original 50-round Acts
baseline. Saved losses and samples were recomputed from restored final
weights for all three models; the parent hash and ordinary latest report
were unchanged. Browser polling showed all three completed results.
All samples remain garbled. Longer context alone did not help under these
settings; this does not establish that longer context is inherently worse.
Training losses also worsened, making optimization a reasonable next
controlled investigation rather than assuming this is only overfitting.
Try learning rate/batch size separately before changing tokenization or
running more unchanged rounds. Further experiments require explicit approval.

### Controlled learning-rate comparison

The user approved the proposed rate comparison with "go for it".
`neural_context.py --learning-rates --acknowledge-review` reuses the controlled
comparison runner for rates 0.1, 0.03 and 0.01 at context 4. Every model has
1,978 parameters, identical initial weights and 5,000 identical target draws,
with batch size 1, character tokenization and all other settings unchanged.
Only the multiplier on the gradient changes; default neural training and
continuation still use rate 0.1.

The model's `update` accepts a finite positive learning rate; tests verify
each weight is exactly `old_weight - rate * gradient`, and its default gives
identical updates to explicit 0.1. Tests also independently replay every rate,
check held-out isolation and identical starting weights, and verify failures,
approval, no final-test files, and unchanged ordinary/context reports.
Every checkpoint stores its rate for reproducible further updates.

Acts 1/2 source, text and rights hashes are verified. Acts 3 text is not read
or evaluated. Generation restores/polls a separate rate report, samples and
loss histories; `latest-rate.json` is separate from `latest-context.json`
and the ordinary `latest.json`. Each run saves its own report and
`rate-R/step-XXXXX.json` weights/texts/provenance at steps 0, 500, ..., 5,000.
No automatic checkpoint selection or model installation occurs.

This is one seed and a fixed example budget. A smaller rate can learn more
slowly, so losing at 5,000 examples would not prove it cannot win later.
Acts 2 is reused tuning data, not an untouched test; generated samples are
model output, not scripture or evidence of understanding. Foundational layers,
developmental admissions, ordinary chat and historical runs remain unchanged.

Run `6466ad9b71384919bb62fe87beb35620` completed 5,000 updates per rate:

| Learning rate | Parameters | Final training loss | Final validation loss |
|---|---:|---:|---:|
| 0.1 | 1,978 | 2.5788 | 2.8241 |
| 0.03 | 1,978 | 2.5415 | 2.6582 |
| 0.01 | 1,978 | 2.8561 | 2.9080 |

Rate 0.03 produced the lowest final validation loss of the tested rates.
This supports smaller steps for this configuration and budget, not a
universal best rate. Rate 0.01 was still improving; this comparison does not
give it additional updates. All generated samples remain garbled.

Verification found 33 saved checkpoints with identical initial weights and
the same draw-sequence hash as the earlier context experiment. Restored final
weights reproduce full training/validation losses and all three samples at
every rate. Rate 0.1 exactly reproduces the original Acts baseline weights.
The ordinary/latest context reports and Acts parent hash remain unchanged.
Browser polling and reload restored all three completed results. All 84
targeted neural, layer, model, graph and HTTP tests passed; the build, editor
diagnostics, signature compatibility and Python syntax checks passed.

### Context comparison at learning rate 0.03

The user approved repeating the longer-context comparison at 0.03 with
"do it". Run `neural_context.py --learning-rate 0.03 --acknowledge-review`.
The context runner now accepts a validated finite positive rate and records
it in the report, model results, checkpoints and approval. The default remains
0.1. CLI flags reject combining fixed-rate context and varying-rate modes.

The three contexts remain 4/16/32 with 5,000 identical target draws, fresh
weights, fixed seeds and otherwise unchanged settings. The latest context
display/pointer updates, while earlier reports and weights stay in their
original run directories. The ordinary and varying-rate reports are untouched.
Acts 3 text is not read. This still compares context plus capacity at one
seed on reused validation, not a pure context effect or understanding.

Tests independently replay every context at 0.03, verify that the recorded
rate matches the actual updates, compare target-draw hashes and byte-check
all prior report/checkpoint files. Approval and invalid-rate rejection are
also tested. No default training rate or installed model changes.

Run `b63e174bbecc439da8dfdd09e4ec6fb4` completed:

| Context | Parameters | Training loss at 0.03 | Validation at 0.03 | Prior validation at 0.1 |
|---|---:|---:|---:|---:|
| 4 | 1,978 | 2.5415 | 2.6582 | 2.8241 |
| 16 | 3,514 | 2.6139 | 2.8046 | 3.2384 |
| 32 | 5,562 | 2.6331 | 2.9100 | 3.3811 |

Smaller steps improved all three configurations relative to their matched
0.1 runs, but context 4 remained best at 5,000 updates. These results do not
show that longer context is intrinsically harmful: the larger models may
need different optimization or more examples. All samples remain garbled;
no understanding claim follows from lower prediction loss.

Verification restored training/validation losses and all prompt samples
from final weights, found 33 checkpoints, matched initial weights against
each original context run and matched target-draw hashes across experiments.
Context 4 exactly reproduces the prior 0.03 rate-comparison weights.
Historical run files, the ordinary and rate latest reports, and the Acts
parent hash remain intact. Browser polling and reload displayed all results
with learning rate 0.03 and zero final-test evaluations.
All 14 focused neural tests passed; CLI conflict rejection, signature
compatibility, syntax, version and whitespace checks passed. No UI code
changed in this repeat.

### Fifty-round synthetic diagnostic

`neural_rounds.py` automates 50 rounds of 100 SGD updates using the existing
MLP and shared `update` implementation. It uses original synthetic passages
with explicit train/validation/test roles. Training alone determines vocabulary
and gradients. Tests verify that changing validation and final-test text leaves
weights identical. Test loss is absent from round 0–49 checkpoints and is first
computed after all updates. Nothing selects a checkpoint based on that test.

Layer 2 is counted once from the training passage; validation/test descriptive
scores never increment its counts. Generated samples are inspection outputs,
not new examples. This is the unconstrained neural baseline; no new layered
training or admission is implied. Evaluation shares synthetic patterns and
vocabulary, so neither data separation nor lower loss proves broad language
generalization.

Every round persists weights and cumulative loss history atomically.
Fixed prompts remain unchanged and use the same sampling seed. The UI shows
completed rounds, status, and final-test loss, with samples every ten rounds.
The UI polls saved running reports; automatic interrupted-run recovery and arbitrary
external dataset import are not implemented. Process termination can leave `running`; ordinary numeric/I/O
failures propagate and attempt to record `failed`. Previous finite round files
are retained.

The authorized first run `d742210851c749fd9fcb1d6592b0b9a7` completed 5,000
updates with 1,078 trainable parameters (1,070 changed). Training loss:
3.0901 -> 0.1627; validation: 3.0885 -> 0.8422; final test: 0.4282.
The output learned seed/pattern fragments but remained repetitive. The gap
between training and validation warrants inspection before scaling; this
is not a successful question-answering evaluation. Old generator and transcript
hashes were unchanged during the measured API run.

The existing chat lab's equal-length crop can be zero when one mode has no
living capture. Its cards then show available full outputs with an explicit
unequal-length label and give the empty modes' reasons. This display does not
invent captures or change the saved comparison, generation, or training.

`neural_experiment.py` adds an isolated character MLP using our own calculations:
four context tokens -> concatenated 8-dimensional embeddings -> 16 tanh units
-> next-character softmax. Two additional tokens represent source boundary and
unknown characters. Only training text defines the character vocabulary.

For vocabulary size V (including those two tokens), all trainable parameters
are `8V + (4 * 8 * 16) + 16 + 16V + V = 25V + 528`.
SGD updates embeddings, both weight matrices, and both biases using
cross-entropy gradients at learning rate 0.1. One step processes one sampled
context/target pair. Fixed initialization, training, and sampling seeds make
runs reproducible; UI runs start fresh rather than silently continuing a
prior corpus. The context boundary is a neural token, not admission to Layer 0.

Finite-difference tests check each gradient group, including accumulation when
a boundary embedding occurs repeatedly. A synthetic repeating-pattern test
requires both losses to fall below 30% of their initial values and checks that
parameters actually changed. This proves mechanics only: its held-out pattern
deliberately repeats the training pattern and is not independent evidence of
generalization.

For real text, the Generation tab requires explicit neural-use approval and a
provenance/rights note for new training and separate held-out text. Nothing is
backfilled from memory, the old corpus, or app prompts. Exact-copy/substring
held-out text is rejected, but that cannot establish semantic independence.
Held-out text never supplies gradients or vocabulary; changing it leaves
trained weights identical in tests. Unknown held-out characters map to the
unknown token and are counted in reports; loss on those tokens is not a
measure of distinguishing unseen characters.

The saved report includes exact parameter counts, changed count, processed
targets, steps, character counts/hashes, loss history, and before/after
continuations from one fixed prompt. Loss is average negative log probability
per character (natural logarithms). Lower loss on a truly independent set is
evidence of better prediction, not understanding. No early stopping or
model-selection procedure is implemented.

Checkpoints, including weights, exact texts, and declared provenance, persist
under `.data/neural-experiments/`; they are not committed. The UI restores the
latest report on refresh, while older run files remain intact. Training is
bounded to 2,000 training characters, 1,000 held-out characters, 128 distinct
training characters, and 1,000 SGD steps (default 300).

Layer 0/1 remain definite rules in their existing seed/lab surfaces. This
neural run is explicitly unconstrained: it does not claim that its samples
pass those rules. The next controlled experiment can compare baseline and
layer-assisted neural behavior with matched initialization, data, and budgets.
The existing generator still serves ordinary chat; no neural sample trains
another model or enters conversation memory automatically.

**A small contribution:** write two original short passages on the same topic,
reserve one for evaluation, and record their provenance. Inspect the loss and
fixed-prompt outputs rather than increasing parameter count first. A user
approval record does not resolve third-party rights.

## Lesson 1: Memory Is Not Learning

Memory keeps the text of an interaction. Training changes the numbers that
control a prediction. A larger transcript does not automatically mean a better
predictor.

In this first version:

1. Existing memories remain searchable by a fixed initial scoring rule.
2. You judge a memory relevant or not relevant to a particular query.
3. The judgment is stored separately from the transcript.
4. Only explicitly reviewed memories and queries can enter training.
5. A candidate must pass evaluation before replacing the saved learner.

The review gate is intentionally conservative. Old conversations are not
automatically approved. A review record means the user has reviewed the source;
it is not a legal clearance certificate.

## Lesson 2: Text Becomes Four Numbers

The `features` function in `original_learning.py` measures:

- **Word overlap:** similarity between word-count vectors.
- **Phrase overlap:** similarity between adjacent word-pair counts. This notices
  some word order, but does not understand grammar.
- **Query coverage:** the fraction of query words present in the memory.
- **Topic coverage:** the fraction present in the memory's existing topic tags.

For a question about apple pie, a recipe can score high on overlap and coverage;
a note about saving money can score low. Different wording and synonyms can
still be missed. The previous library-based latent representation is not used.

The reviewed vocabulary stored in the checkpoint is an inventory for inspection,
not a semantic embedding. The current feature calculation compares words and
phrases directly; it does not use that inventory to generate meaning.

## Lesson 3: Weights Control The Score

Each of the four measurements has an adjustable weight. A fifth adjustable
number, the intercept, controls the general tendency to score a memory highly.

The scoring rule adds the weighted measurements and the intercept. A smooth
function then maps that result between zero and one. A high score means the
model predicts relevance. It is not a probability that the memory is true and
has not been demonstrated to be statistically calibrated.

The search checkpoint starts with clearly specified heuristic weights. A new
training candidate starts at zero and learns its own weights. Neither start
comes from third-party pretrained weights.

## Lesson 4: Training Corrects Error

For each training example, the model makes a prediction and compares it with
your named label: **Direct answer** is 1.0, **Related context** is 0.5, and
**Unrelated** is 0.0. The `fit` function calculates
how each weight contributes to the error, then takes a small step to reduce it.
That calculation is gradient descent.

The middle target says connected background should score between a direct
answer and an unrelated memory. It is a design convention, not a 50 percent
chance that the memory is true. We still use one graded relevance score, not a
three-class language-understanding model. Adding this option does not by itself
teach the distinction; that requires reviewed examples and measured improvement.

We use an averaged update across the examples and a small penalty on large
weights. This reduces abrupt updates and discourages unnecessarily large
coefficients. The current prototype makes 600 passes through a batch, with a
step size of 0.3 and a weight penalty of 0.01. These are testable starting
settings, not universal constants.

## Safe First Experiment

From the Apps project directory, run:

```bash
python3 original_learning.py
```

This uses tiny illustrative examples constructed in code. It writes no files
and changes neither the graph nor your checkpoint. They are demonstration/test
fixtures, not a commercially approved training dataset.

Read the output as follows:

- `before_loss`: error before learning; smaller is better.
- `after_loss`: error after learning the same examples.
- `relevant_score`: should increase toward one.
- `context_score`: should stay between the direct and unrelated scores, near 0.5.
- `irrelevant_score`: should decrease toward zero.
- `weights`: the numbers training changed.

This experiment proves the calculation can learn those examples. It does not
prove it will handle new questions. That is why we need a separate test set.

## Lesson 5: Testing Must Use Unseen Queries

Real training requires at least 12 eligible judgments over six distinct queries,
with direct-answer and unrelated anchors in training and validation. Context
examples, when present, also contribute their 0.5 targets to loss. Whole queries
are kept out of training for validation. The same query cannot appear in both
partitions of a training run.

We compare candidate and current scores using average logarithmic loss. A wrong
confident prediction costs more than a hesitant wrong prediction. The candidate
must lower validation loss by more than 0.001 to be accepted. Otherwise, its
weights are rejected. The previous accepted checkpoint is retained for rollback.

Repeatedly using the same validation questions can still overfit them. A later
milestone needs a separate frozen benchmark and more varied examples. Passing
our tiny tests is not evidence of general intelligence.

## Your First Contribution
Visual growth is separate from training: domain size grows with the square
root of its message count and mean salience, capped at 180 graph units.
Individual memories grow with capped salience and a small consolidation bonus.
Recall and feedback can grow them; decay can shrink them. All sizes recompute
when the graph redraws, without changing membership or promotion thresholds.
AI memory remains central; solid shaded boundaries separate the other domains.

The [first teaching batch](FIRST_TEACHING_BATCH.md) proposes six clear pairs
from existing user-authored project statements. Its excerpts are verified,
but it remains an unapproved draft. Review the questions, labels, and sources
before any training; preparing the draft does not change model weights.

Choose one short memory you authored and one question it should help answer.
Then choose an unrelated memory you also authored for the same question. Tell
us why the first is helpful and the second is not. Begin with a clear case,
rather than a subtle or disputed one.

For the initial batch, expand this to six genuinely distinct questions and a
direct/unrelated pair for each, plus reviewed context examples where appropriate.
Do not manufacture unrelated query variants
just to reach the six-query threshold. Do not include secrets or material whose
source you cannot account for. We will review the examples before approving
them for training.

The graph's **Direct answer**, **Related context**, and **Unrelated** controls
save explicit named judgments. They do
not grant training approval. Helpful/correction/recall controls still change
graph salience, not training labels. Assistant replies cannot be approved as
training memories in this initial version.

## Data Review And Files

`python3 memory_learner.py review` reads a JSON object from stdin containing
`memory_ids`, `queries`, and `acknowledge: true`. Use it only after reviewing
the actual source text. IDs must be existing user-authored memories; questions
must be the actual questions associated with the judgments. Approval stores
content hashes, not a blanket approval of future content under the same ID.

After approval, resubmitting an existing relevance judgment triggers a training
attempt; it replaces the previous judgment rather than multiplying votes.
Editing a memory's text or topic tags invalidates its approval and old label.

- `.data/learner-graded.json`: graded model weights and named judgments.
- `.data/learner-graded.previous.json`: previous accepted graded checkpoint.
- `.data/learner-original.json`: old binary checkpoint, preserved unchanged;
  its judgments are carried forward as pending relabeling, not guessed labels.
- `.data/learning-review.json`: explicitly reviewed memory/query hashes.
- `.data/learner.json`: old library-based checkpoint, preserved but never loaded
  by the new learner.

The version-three graded checkpoint keeps weights between runs. It does not
reuse binary model weights or generation numbers. No third-party model
weights are imported or converted. All these files remain git-ignored local
data. Pause the server before manual rollback or file edits.

## Verify The Milestone

```bash
python3 -m unittest test_original_learning test_memory_learner test_memory_clusters -v
printf '{}' | python3 memory_learner.py status
```

Tests cover numerical stability, learning improvement, rejected updates,
query-separated validation, review invalidation, preservation of old files,
offline operation, and the prototype HTTP integration. Tests run in temporary
directories and never approve your historical memories.

## What Comes Next

First, collect and review real examples. Then evaluate whether this small
learner makes useful predictions on fresh questions. That evidence should guide
the next representation or model change, rather than adding size immediately.

The first language-generation milestone is a small character n-gram model. It
does not add semantic representation or neural layers, and it is not evidence
of understanding. Evaluate changes on reviewed, held-out samples before adding
complexity. The suitably licensed viewer may remain in use. Topic-cluster
promotion still follows its existing rules; the relevance learner does not
reorganize topics.

## English Seed Experiment

The separate `seed_life.py` experiment begins translating the concepts from
the neighboring LLM From Scratch project into this project. Layer 0 describes
the initial building blocks in which life is possible; possibility alone does
not mean life exists. Layer 1 only excludes known impossibilities; it does not
prove grammar or meaning. The combined assessment reports `alive`, `dead`, or
`maybe` and a separate `layer_zero_eligible` flag. These are names for model
states, not claims about biological life or consciousness.

The current Layer 0 seed list is `</>A`, `</>I`, `</>I `, `</>A `,
` a `, ` A `, and ` I `, with the displayed spaces significant. The `</>`
marker is carried over from the earlier project as a provisional boundary
representation; its precise meaning and punctuation rules still need
discussion. This is the agreed initial list, not a claim that every possible
punctuation primitive has been enumerated. The seven-form list and current
exclusions are deliberately limited for this experiment: outside forms cannot live in this
seed universe, even when collected for review. Review is not admission to
Layer 0. Additional layers are not a prerequisite for future integration.

The combined assessment recognizes the last five forms above as alive. The bare `</>A` and
`</>I` are `maybe`: on their own they are not alive, but more information may
make them viable. A lowercase ` i ` anywhere in the assessed text is dead, as
is any run of three identical consecutive characters, including whitespace.
An assessed response beginning `</>I.` is also dead, even if further text
follows: the current seed rule does not treat `I.` alone at the start as a
viable sentence or thought. ` I.` remains `maybe`, since it may appear at the
end of a thought such as `or I.`. The fragment ` A, ` remains `maybe`, since
it can occur in a larger sentence such as `vitamin A, which supports vision`.
Other punctuation, including the end of `area.` or `area. `, remains unresolved.
Character identity is exact, so mixed-case characters are not silently treated
as identical. A dead rule takes precedence; all other unrecognized input is
`maybe`, not automatically alive or dead. An alive result only recognizes an
exact seed form; it is not a verdict on a larger sentence.
For example, `hello` is `maybe` for extraction review but has
`layer_zero_eligible: false`, so it is not eligible for life. `</>I` is
`maybe` with `layer_zero_eligible: true`, but is not yet alive. ` I ` is
eligible and alive. Merely surviving the Layer 1 exclusions never admits
outside text or marks an incomplete seed alive.

Run `python3 seed_life.py ' I '` or `python3 seed_life.py '</>I'` to inspect
the result and reason. Run `python3 -m unittest test_seed_life` for the seed
examples. These tests check the agreed rules, not language understanding or
training improvement. A useful next contribution is checking the existing
seed cases and their eligibility without expanding either frozen layer.

The explicitly approved `seed-life-v4` addition excludes terminus + one English
consonant + literal space, in either case. Source offset zero supplies this
context; leading spaces retain their existing whitespace-boundary meaning.
Interior occurrences are not excluded by this rule. Initial `t.`, a consonant
without a following space, and multi-letter words are not covered. Layer 0 is
unchanged. This is a rule of the restrictive seed universe, not a universal
claim about initials, symbols, or all English writing.

The manual extraction experiment runs with
`python3 seed_extraction.py path/to/review-text.txt`. It reads a selected
nonempty UTF-8 text file of up to 50,000 characters, preserving case,
whitespace, and line endings, and examines every overlapping 1–3 character
window. It groups known living windows, known dead windows, and `maybe`
windows separately. Unknown windows are saved for review, not discarded or
approved for training. Dead windows are recorded separately and do not enter
the unknown review list. An occurrence's offset is a zero-based Python Unicode
character index into the exact source; the saved absolute source path and
SHA-256 of the original bytes identify its provenance. Repeating an extraction
of the same source content replaces that source's record; changed content is
kept as a separate record. Each pattern now includes `layer_zero_eligible`.
The `seed-life-v2` record identifies this output change; older ruleset records
are retained unchanged, and repeat extraction replaces only the matching
path, content hash, and ruleset. The queue is
`.data/seed-extraction-review.json` and may contain sensitive fragments and
paths. New records also retain the exact full source text so context can be
reconstructed; older records may contain only fragments. Assessment is per window; a dead larger
configuration does not automatically make each smaller window dead. The
`</>I.` rule is longer than three characters, so it is not detected by these
windows. Source offset zero now supplies terminus context for an initial `A`,
`I`, `A `, or `I `, matching the existing terminus seed forms without changing
the stored text or its offsets. An initial `A ` or `I ` is eligible and alive;
the bare initial letter is eligible but still `maybe`. Identical interior
fragments remain separately collected for review and independently ineligible.
Newlines and paragraph boundaries do not infer new termini. Records under
`seed-life-v3` distinguish this boundary-aware extraction; earlier records
remain unchanged.

Run `python3 -m unittest test_seed_life test_seed_extraction` to check seed
and extraction examples. These tests do not establish training performance.
Normal memory updates remain independent of extraction and training approval.
Manual extraction does not change the transcript, graph, or model weights.

### Chat Lab: Separate And Combined Comparisons

The chat comparison toggle requests a `model_lab.py` report before the current
prompt is trained. The chat displays exactly two extraction panels above the
generation comparison: old-model context/next-character captures and combined
layer captures. Both are visible by default; layer captures are grouped into
alive, review, and dead, with exact positions and eligibility. The downloadable
report retains the more detailed intermediate comparisons described below.
Extraction compares the same raw 1–3 character windows
unfiltered, through Layer 0 eligibility alone, through Layer 1 exclusions alone,
and through both. The actual old generator extraction is shown separately:
whitespace-normalized four-character contexts and next-character counts, not a
claim that the old model used the new window extractor. All unknown windows and
positions remain inspectable. Layer 1's lack of an exclusion is not proof of
life. The longer `</>I.` exclusion remains outside this window-sized comparison.

Generation compares the existing four-candidate, scorer-selected generator,
a rule sampler that concatenates complete living captures from this prompt,
and layers-first generation. The rule sampler is not a trained new model and
its concatenation is not certified as a valid sentence. Combined generation
protects the earliest living capture as the opening. For a source-start capture,
it looks up that opening with the existing start markers. For an interior seed
shorter than four characters, it samples a corpus context ending in that seed,
weighted by transition counts, then continues with normal four-character
transitions. This explicit suffix backoff only exists in the combined lab
mode. It does not change standalone generation. A missing capture, missing
context, or end marker is reported without an unrelated fallback.

All modes receive the same 320-character maximum. Their equal-length view is
cropped to the shortest actual output; no text is padded or repeated to pretend
the old model reached the budget. Full outputs, actual counts, and shortfalls
are shown alongside that view. No common output is claimed if any mode produces
none. The existing model's full output remains the ordinary chat response;
the lab does not replace it with an experimental continuation.

Each comparison is persisted under `.data/model-lab/` with exact prompt,
generation, checkpoint/scorer hashes, ruleset, deterministic random seed,
captures, and outputs, and can be downloaded from chat. This preserves the
pre-training snapshot even if opted-in training subsequently advances the
model. It does not remove prior exposure to earlier prompts. Reading or
generating a comparison does not train either model; recorded experimental
outputs are not automatically fed back into training. A comparison failure
is visibly reported independently of normal chat capture.

Tests check separate extraction shapes, exact source-start and interior
handoffs, protected prefixes, reproducibility, equal displayed lengths, honest
shortfalls, empty/unavailable modes, checkpoint preservation, persistence,
download routes, and pre-training ordering. These comparisons expose behavior;
they do not establish improvement, understanding, or a separately learned
seed model.

Lab version 3 replaces the version 2 append experiment with an aligned overlay
on the old-model response. A captured living ` I ` can correct a matching ` i `,
and ` A ` can correct ` a `. A living source-start `I ` or `A ` only supports
the corresponding lowercase response-start correction. Letters inside words,
different letters, punctuation boundaries, and unsupported cases are untouched.
This overlay policy does not add rules to either frozen layer. Each changed
character records its output offset, before/after values, and supporting exact
source capture and offset. Spaces and full output length remain unchanged,
including at the character budget. No matching correction or no old output is
explicitly reported; no blocks are appended. All four modes participate in
the equal-length view. The whole response is not certified as living or true.
Saved version 1 and 2 records retain their original modes and are not rewritten.

Lab version 4 extends the overlay only for that newly approved consonant-opening
exclusion. It uses the earliest living source-start `A ` or `I ` capture from
the current input to replace the dead initial consonant. It does not substitute
an interior ` I ` capture or invent a new opening. If no supporting capture
exists, it preserves the old response and visibly reports the dead opening as
unresolved. Replacement provenance includes the Layer 1 reason and exact source
offset. Output length, spaces, and all other old-model characters are preserved
apart from the existing supported case corrections. Version 3 records and
older extraction rulesets remain unchanged. The ordinary chat reply remains the
old model output; this repair is in the fourth lab comparison.

### Alongside Integration With Current Training

New approved generator documents and opted-in chat prompts now invoke the same
extractor before the generator checkpoint is updated. This is alongside
analysis, not a seed eligibility gate on the existing model: the generator
still learns from its full approved, whitespace-normalized text. Layer 0 and
Layer 1 uses only the explicitly agreed exclusions. No further layers, new learned representation, or
generation constraint were introduced.

The separate `.data/seed-training-review.json` queue records exact source text,
character positions, original content hash, and a `generator://documents/`
source identifier. Provenance includes the generator document ID, normalized
training hash, title, source, and rights note. Unknowns remain in review,
known dead windows are separate, and eligibility does not constitute training
approval. Approval of the original text for the current generator is not
approval of extracted patterns for a future seed model.

The API and Chat/Generation views report alongside counts for active corpus
sources and how many older documents have not been extracted. Existing
checkpoints remain readable without backfilling, retraining, or changing their
generation number. Removing a generator document removes its contribution from
active counts but preserves review records for provenance. A review record
alone does not prove a model update succeeded: it is written before the model.
If extraction fails, the error is surfaced and the model is not updated.
The chat endpoint reports training errors independently and continues saving
the exchange. Original submissions are limited to 50,000 characters, including
whitespace, so exact-text extraction has a bounded input.

Tests cover exact text and hashes, exclusion/review separation, unchanged
transition calculations, legacy checkpoints, failure reporting, and both HTTP
submission paths. They demonstrate wiring and preservation of behavior, not
better generation. A concrete next contribution is submitting a small original
text through Generation and checking its extraction counts against the queue.

The experiment does not import the earlier project's dictionary-derived
filters or datasets. Standalone extraction does not train or normalize text.
The dictionary's provenance and permissions require review before any reuse.
Before using the layers to restrict training, define how states compose, how
boundaries and punctuation work, what to do with `maybe`, and how to measure
results against the existing generator using appropriately reviewed examples.
The existing content-hashed review gate remains in place.

## Lesson 6: A Domain Can Grow Subtopics

Promotion out of Under evaluation moves the member user turns and their
associated replies into the promoted galaxy, without retaining evaluation as
their parent. The graph omits the evaluation-parent tether, not links between
individual memories. Memory pairs remain linked even if one endpoint is still
under evaluation: a sequence or similarity link is not galaxy membership.
Links to other galaxies and non-evaluation parent relationships remain.
Historical transcript links and the cluster's
origin metadata are preserved for provenance; regeneration applies the same
display separation without deleting memories or approving training.
Parent-child links are drawn as solid, thicker green lines between the circular
galaxy shells, rather than between their internal hubs. Hover text identifies
which galaxy grew from which. Cross-galaxy memory connections retain their
dashed styling; Under evaluation receives no parent connector. This is visual
only and does not change membership, merge galaxies, or train weights.

The graph can now look for related user contributions inside an existing domain,
such as AI memory. It removes broad parent words and conversational filler from
that comparison, so saying "memory model" repeatedly does not itself create an
AI-development topic. The learner's training approval gate is unrelated to this
graph-organization step; discovery does not approve data or train the learner.

A candidate needs cohesive wording and either evidence from multiple sessions
or sustained, distinct discussion in one recorded session. The initial
single-session rule requires eight meaningful contributions and a span of seven
transcript positions; the common cohesion gate is 0.30. See the README for the
complete thresholds. Assistant repetition and generic acknowledgments are not
votes. These are heuristics to test, not a human-level measure of topic meaning.

When promoted, a child topic receives its own hub and a link to its parent. A
parent link means "subtopic of"; it does not mean that every connected memory
belongs in the child core. Chronological subject-transition links remain
separate. Parent domains are preserved in saved metadata.

Our AI-development conversation is a useful real test: related wording can still
fragment into several lexical candidates. We should inspect that evidence rather
than silently manufacture a promotion. Better representation is a future
learning milestone, not something these rules pretend to provide.

## Lesson 7: Capture And Placement Are Different

A prompt can be successfully saved yet appear in the wrong visual region.
We found this with explicit "AI creation and dev" wording and short approvals
such as "sounds good let's do it": the keyword classifier placed them under
evaluation rather than the ongoing AI-memory conversation.

Explicit AI-development wording now selects the AI-memory parent. Clear short
follow-ups retain their own session's current subject; an explicit new subject
still wins, and assistant examples do not assign the user's topic. Ambiguous
new content is not automatically treated as a continuation. These are bounded
context rules, not semantic understanding.

Display synchronization matters too. Reclassifying an existing node must refresh
the open graph, not just update a file. Feedback saves preserve the canonical
graph's new nodes and membership metadata rather than overwriting them with a
reduced snapshot from an older tab. Capture, placement, rendering, and training
are separate paths and need separate checks.

## Lesson 8: The Learner Is A Component, Not A Memory

The visualization now includes two rectangular, visual-only nodes: the declared
AI-creation project and the Learner component. The project is not a claim that
the corresponding conversation topic has automatically promoted. Neither node
is stored as a memory, counted as a promotion vote, or included in training.

Directed system links show the project building the learner, the learner
retrieving memories, and reviewed examples flowing from memory to the learner.
These are architectural relationships, not inferred memory associations.

Select Learner or Inspect learner to view live generation, four feature weights,
the intercept, reviewed memories, recorded and eligible judgments, pending
review, and the last evaluation losses. Status refreshes periodically and after
feedback. Reading it does not train the model. A generation-zero learner displays
an initial scoring rule or an awaiting-review state, not a false learning claim.

When searching, amber dashed links show up to three predicted-relevant memories.
They disappear when the query is cleared or Relevance links is switched off.
They are not added to stored memory links and do not change topic membership.
At generation zero, predictions use the initial heuristic weights; an accepted
checkpoint uses the newly learned weights.

As new turns are captured, the memory count changes. Saving feedback changes
recorded judgment counts. Only reviewed, eligible examples passing held-out
evaluation can change the generation and coefficients. This lets us distinguish
growing memory, collected feedback, and demonstrated model updates.

## Lesson 9: Questions And Responses Can Become Experiences

The training view is now streamlined: Enter submits a prompt, and Shift+Enter
keeps a newline. The redundant submit button and manual memory-simulation
section are removed. First-use consent explains that subsequent grading clicks
review only the displayed memory/current prompt for training, not all stored
memories. Cancel skips saving that judgment. This replaces the repetitive review
checkbox without removing the underlying data-review gate.

Submitting a trainer question now retrieves earlier memories, produces a
source-linked memory-excerpt response, and saves both turns. The current question
is excluded from its own refreshed search, so it cannot be its own answer.
The latest question/response survives a page reload without being submitted
again. Questions may contain information, but recording them is not verification
of an assertion and does not approve them for training.

The relevance learner and language generator are separate. The first generator
is an original character-level four-character context model trained from text
explicitly added in the app's Generation tab and each new prompt submitted in
either in-app chat. Pressing Enter opts that prompt into future generator
training. It predicts character
transitions; it does not understand meaning, follow instructions, or produce
grounded answers. Its output may be incoherent, and it can reproduce sequences
from its small training corpus.

Each approved source records its title, provenance note, content hash, and the
user's review declaration. The title and additional provenance note are
optional; the source defaults to `Me`. Clicking **Approve text and train** is
the explicit approval action, not legal clearance. Removing a source rebuilds
the saved transition counts from the remaining approved corpus.
Historical conversations, retrieval approvals, and generated outputs are never
imported automatically. Generated responses are stored with a `local-generated`
source tag and excluded from generator training and the relevance learner's
representation. The
Generation view tracks fixed prompts, captures a deterministic sample at each
corpus generation, and records Keep/Edit/Reject feedback separately. Ratings
and corrections do not train on generated output.

The next response-selection component now has an isolated implementation in
`original_response_scorer.py`. Explicitly reviewed example pairs can fit four
scoring weights with the existing standard-library gradient descent. Synthetic
tests demonstrate lower loss on held-out queries, rating replacement, and
preservation of previous weights; they do not establish useful conversation or
truthfulness. Live chat now selects from four generated candidates with this
scorer and shows candidate count, scorer generation, and selection score.
Existing ratings remain untouched and unimported.

In the legacy ngram AI Chat, choose **Relevant and train** or **Needs correction and train** to
review a generated pair and update response-selection preferences. A nearby
reviewed pair can immediately adjust a candidate score by at most 0.25; broad
weight updates still require query-separated validation. This is a lexical
preference mechanism, not a check of meaning or truth. These judgments remain
separate from generator training: adding
your prompt teaches character patterns, not whether an answer was correct.
Watch the chat progress strip for corpus growth after each prompt, and use
**Compare generations** for fixed-prompt samples. Session rating counts reset
when the chat panel is reopened, but saved feedback is not deleted. More sources
or a higher generation number alone do not establish better answers.
Rating revisions replace the previous label for the same pair. Generated text
is not silently added to the generator corpus by rating it.
Empty generation candidates have no selection score. Chat preserves the memory
fallback and remains usable rather than trying to format a missing score.
AI Chat shows persistent scorer example/query counts, minimum-data progress
bars, and validation loss when available. These are readiness indicators, not
an accuracy score. **Needs correction** opens a field for your revised reply.
**Approve correction and train** explicitly approves that edited answer for
positive prompt-response scoring and generator character-pattern training.
The original reply remains a negative example and is not overwritten. Repeated
saves of the same correction do not multiply examples. Your approval is a review
declaration, not legal clearance or proof of truth. Fixed-prompt evaluation edits
remain evaluation-only.

This establishes a deliberately small, inspectable language-generation loop.
The next work is to test how its output changes with varied, reviewed documents
and to design evaluation before increasing model complexity. It is not evidence
of semantic understanding or factual reliability.