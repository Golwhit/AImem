# Push Bridge POC

A proof-of-concept PWA that sends a web push notification from a desktop browser to a phone browser, then lets the phone reply back to the same server state.

Ongoing work and discussion topics are tracked in the
[project backlog](docs/TASKS.md).
Application versions follow the [versioning policy](docs/VERSIONING.md);
the [versioned session log](docs/SESSION_LOG.md) records each checkpoint.
The memory graph displays its generated application version in a small
bottom-right badge inside the galaxy canvas. Regenerate the graph after a version bump.

## Working model: New Base, no layers

L0, L1 and L2 are **retired from the working path**. The layer-free
word/punctuation base with character fallback is the preserved prose baseline:
run `10255643d34e41749c6bcc1f68ee8b5c`, arm `new`, seed 17, four passes,
7,078 parameters. Seed/pass selection follows the protocol, not the best score.
This checkpoint trained on Acts 6-12; it is **not** the prior whole-Acts/
Corinthians continuation and does not inherit that model's training history.
It remains an experimental continuation model, not an assistant or factual source.
The persistent conversation-training path below supersedes it in chat after
explicit promotion; the prose checkpoint remains available for rollback.

The original context-4 character base is the permanent comparison control.
Its paired seed-17/four-pass checkpoint (2,103 parameters) is copied and
checksum-pinned beside the new checkpoint, with its own selection in
`.data/neural-chat/comparison-baseline.json`. Historical experiments, old
layered weights, source data and the previous selection are retained unchanged
for provenance, not used by active generation.

```sh
# Working training comparison: original versus new, without any layer setup.
python3 neural_base_comparison.py --acknowledge-review
# Explicitly select the established working checkpoint.
python3 neural_chat.py activate-base --run-id 10255643d34e41749c6bcc1f68ee8b5c --seed 17 --passes 4
python3 neural_chat.py status
# Same prompt through the active new base or the pinned original control.
printf '%s' '{"prompt":"The ","max_chars":64}' | python3 neural_chat.py generate
printf '%s' '{"prompt":"The ","max_chars":64}' | python3 neural_chat.py generate-original
```

Both use deterministic greedy decoding for repeatable comparisons; no layers
or online weight updates run. A new experiment never auto-activates. Missing
or corrupt pinned files fail explicitly. The Base Model Lab remains an
educational walkthrough of the **original** character base.
Layer descriptions and results below are historical. Reproducing retired
five-arm experiments requires `--include-retired-layers`.

### Permanent conversation training

[neural_training.py](neural_training.py) continues saved weights rather than
restarting for each round. Initial parent: the full-64, fresh-scaled seed-17
checkpoint from `ee7f3c1be14f477a82e2db03ae72ff71`, the first protocol seed,
not the highest test score. The separately preserved original character
control is unchanged. The conversation design has 15,508 parameters and
64 input slots; its weights were originally fresh, so it does not inherit
the prose model's Acts knowledge.

```sh
# Each invocation continues the last completed permanent checkpoint.
python3 neural_training.py --cpu-seconds 300 --rate 0.1 --activate-after
# Promote the completed replay-verified round to chat.
python3 neural_training.py --activate
# Restore the previous active selection without erasing learned progress.
python3 neural_training.py --rollback
```

`--activate-after` promotes only after a round finishes with status
`completed` and replays exactly. Rejected rounds are never activated, and
no approval flag is needed. Omit it to keep training progress separate from
the active model. Rounds use plain SGD on the **existing approved 64
exchanges only**. Default rate is .1; `--rate` lowers it. The recent
passing rounds used .02 with a 60 CPU-second budget. Each round is
guarded: hidden saturation above .6 or teaching exact below .9 rejects it.
Weights, exact inputs, parent hashes, source copies, metrics and reports
persist in `.data/neural-training/`; `progress.json` selects the most recent
completed round. A new target-draw seed is recorded per round. There is no
optimizer momentum to lose, no retraining from scratch and no indefinite
background daemon. Concurrent rounds fail explicitly rather than overwriting
one another. The CPU allowance covers updates, not evaluation or an
additional full exact replay.

Promotion copies checksum-pinned weights into the existing chat mechanism,
archives the previous selection and leaves the original comparison untouched.
Chat applies the teaching role format to the **current user message only**,
returns the first generated line (at most 96 characters), and reports empty
answers as errors. It does not claim full conversation history or retrieval-
conditioned generation. Normal memory retrieval, automatic app-chat capture,
development findings and graph refresh remain connected.

This is a narrow synthetic-task learner, not a grounded assistant.
**Unmarked replies do not train. Clicking Relevant trains that exact saved
prompt/answer immediately. Submitting Train correction now trains your
edited answer immediately.** A bare Needs correction click only opens the
editor. These explicit actions approve the exact content-hashed pair; there
is no additional review checkbox. The app verifies the learned answer and
activates saved weights before reporting success. Unsupported or unlearned
answers report errors, not a success-shaped review record.
User-approved pairs join future permanent training; a newer correction
replaces conflicting targets for that same prompt. See
[immediate feedback training](docs/FROM_SCRATCH.md#immediate-feedback-training).
Other new material still needs explicit content-hashed approval.
Evaluation answers never train weights or choose checkpoints. See
[permanent training details](docs/FROM_SCRATCH.md#permanent-conversation-training).
Initial permanent round: `57f452944e2c4767b5586c8320bfbad7`, 57,100 additional
updates / 300.28 update-CPU seconds, **64/64 teaching and 22/32 development
answers**, 10/16 fully correct pairs. Changed wording remains weak.
Exact state and prediction replay passed; rollback and the original control
are preserved. This is saved learning, not general conversational competence.

Memory capture remains connected throughout model development: development
conversation exchanges and important findings are recorded through the existing
transcript/graph workflow; internal app chat continues to append exchanges
automatically. Model-capability tests must not disable capture or stop memory
growth. Keep provenance and topic separation intact. Saving content to memory
does not approve it for neural training: the content-hashed review gate remains
separate from capture, retrieval and model-weight updates.

### Next milestone: stable learning and conversation context

The offline [stability diagnostic](neural_base_diagnostic.py) audits identical
inputs with conflicting targets, then fits two small already-approved training
snippets with three learning rates and seeds:

```sh
python3 neural_base_diagnostic.py --acknowledge-review
```

Run `2d557e54d04543efa9b8dc0fe2220d56` confirms learning but not near-perfect
tiny-sample fitting. Rate .1 fits faster but worsens old-sample token loss
after new-sample training in all three seeds. Lower rates reduce interference
at this short budget, not proof of better final quality. More importantly,
two draft questions differing only in an earlier red/blue fact produce
identical model features and masks, so this base cannot answer both correctly
with its deterministic decoder. Longer fact-bearing context is necessary
for this task; lowering the rate alone cannot fix it.

The [conversation teaching draft](fixtures/neural-conversation-draft.json)
contains eight proposed teaching examples and ten separate tests. It is
**pending review, not approved for training**, with no runtime answer lookup
or training integration. Review names, facts, corrections, missing-information
answers and split design before freezing content hashes.
See [diagnostic results and limitations](docs/FROM_SCRATCH.md#working-base-stability-diagnostic).

### First approved conversation-training trial

The user subsequently approved training on the eight draft teaching exchanges.
The exact file remains preserved; per-run `review.json` records its hash and
approval rather than rewriting the earlier pending-review artifact.
[neural_conversation_trial.py](neural_conversation_trial.py) continues isolated
copies of the actual active New Base, training assistant answers only.
The ten test answers are never optimizer targets or vocabulary sources.

Run `576f66ae430b4f0088fde6af5fc87e76`: 5,000 updates per rate (.1/.03/.01),
one parent and shared draw schedule. At .1, exact teaching answers **1/8**,
test answers **1/10**; the lower rates get zero on both. All three reduce
teacher-forced loss but fail useful conversation. The matched test answer
is "Jon." when he keeps the box; the same answer is incorrectly given when
he transfers it to Mira. Earlier facts fall outside observable context,
and ordinary text role labels consume some of that limited context.
Only one missing training character (straight apostrophe) was appended,
preserving old weights/IDs; test vocabulary was not added.
Active chat and original control are unchanged; no experimental model was
activated. See [training protocol and results](docs/FROM_SCRATCH.md#first-approved-conversation-training).

### Longer conversation context: information preserved, learning still weak

[neural_conversation_context.py](neural_conversation_context.py) compares
the previous short base, training-only word-identity expansion, and that same
expansion plus a 64-feature single context (up to 32 word/gap pairs, with
the original local fallback features). No layers or new word-output tokens.
Run `1fe67c6c59064ddab1e088843cbe1c60` uses the same approved teaching split,
parent and 5,000-update schedule. At rate .1, exact teaching/test answers:
1/8 and 1/10 short; 1/8 and 1/10 short+identities; **0/8 and 0/10 long**.
The longer variant distinguishes the red/blue facts but fails optimization;
all measured final teaching hidden activations exceed absolute .99.
Lower-rate follow-up selected .01 by teaching loss only: still 0/10 exact
test answers, 1/10 case-normalized, and only 4/10 terminated replies.
All five full model states and non-timing metrics replay exactly.
Active chat/control remain unchanged; findings and conversations remain in memory.
See [long-context evidence and limitations](docs/FROM_SCRATCH.md#longer-context-conversation-ablation).

### Initialization and normalization diagnostic

[neural_plasticity.py](neural_plasticity.py) tests inherited weights,
fresh scale-aware initialization and inherited weights with fixed pre-tanh
normalization. It keeps the approved teaching data and long-context
representation fixed, compares equal update-CPU allowances, and selects rates
using teaching loss before checking the existing developer-visible tests.
Experimental copies cannot activate chat. See
[protocol and validation](docs/FROM_SCRATCH.md#initialization-and-plasticity-diagnostic).
Run `29395ea2dc794a499c608b489959221f`: fresh initialization fits 4/8,
7/8 and 5/8 teaching answers; normalization fits 3/8, 4/8 and 4/8;
inherited weights fit none. Both interventions score only **1/10 test**
in every seed (the same blue-box answer), versus 0/10 inherited.
Learning on teaching examples improves, but useful conversation does not.
No relative-gain claim from a zero baseline, no activation or knowledge-retention claim.

### Larger balanced conversation meal

[neural_conversation_meal.py](neural_conversation_meal.py) compares an approved
64-exchange paired teaching set with its balanced 16-exchange subset.
Fresh and normalized designs use identical starting weights/vocabulary
across sizes, three seeds, rate .1 and 24 update-CPU seconds per copy.
The 32 separate evaluation cases include new combinations and changed
wording; answer and both-members-correct pair scores are reported.
No test answers become training input and no experimental model activates chat.
See [curriculum, exact approval and protocol](docs/FROM_SCRATCH.md#larger-balanced-conversation-meal).
Run `ee7f3c1be14f477a82e2db03ae72ff71`: fresh/64 fits 59/64, 58/64,
59/64 teaching answers and scores 21/32, 18/32, 15/32 on evaluation.
The fresh/16 control scores 20/32, 20/32, 19/32 at matched CPU:
**the bigger meal does not improve mean evaluation accuracy**.
Familiar question patterns work much better than changed wording.
These are narrow synthetic tasks, not useful general conversation.

## What it does

- Includes a separate **Base Model Lab** tab at `/?view=base-lab`: step through
  the real base neural model's construction and one character-training update.
  Diagrams and actual numeric values sit beside a short conceptual code sketch;
  a separate source view shows the real Python from line 1 with junior-friendly
  annotations. No L0/L1 or active chat training occurs in this teaching lab.
- Registers the phone as a push subscription.
- Sends one push from the desktop.
- Shows the notification on the phone via a service worker.
- Lets the phone send a response back to the backend.
- Polls the latest server state so the desktop can see the reply.

## Prerequisites

- Node.js 18 or newer.
- A browser with service worker and Push API support on the phone.
- An HTTPS URL for the phone to subscribe in a real test environment. Localhost works only on the same device; for cross-device testing, use a tunnel or deployment with HTTPS.

## Install

```bash
npm install
```

The original learning core needs Python 3.11+ and its standard library only.
The old `.venv` is preserved but is no longer needed by the learner. npm
dependencies remain necessary for the prototype viewer, not for the learning
calculations.

## Run locally

```bash
npm run dev
The original learning and generation cores need Python 3.11+ and its standard library only.

This starts the API server on `http://localhost:3001` and the Vite client on `http://localhost:5173`. The memory graph page can open directly from disk, but model retrieval and relevance feedback require the local API. No hosted AI service, AI API key, pretrained weights, or model downloads are used. The unrelated phone-push feature still communicates with browser push providers when invoked.

## Use it

1. Open the app on the phone and allow notifications.
2. Tap `Subscribe this phone`.
3. Open the app on the desktop and tap `Send push to phone`.
4. When the notification arrives on the phone, open the app and tap `Reply from phone`.

## Notes

- The server keeps one active subscription in memory for the demo.
- If you restart the server, the generated VAPID keys change unless you set `VAPID_PUBLIC_KEY` and `VAPID_PRIVATE_KEY`.
- For a more durable setup, store subscriptions and keys in a database or environment file.
- Each phone reply now appends a new memory exchange automatically, advances the cycle, regenerates `memory_graph.html`, and refreshes the backend graph-state snapshot.

## Local memory graph

### Base Model Lab: learn the mechanics first

For an offline quick multi-context comparison:
```sh
python3 neural_context_mix.py --acknowledge-review
```
This creates fresh base-only 4-character, 16-character, jointly trained
4+16 probability-mixture and approximately parameter-matched single-window
models. It uses three seeds, 1,500 identical target draws per arm/seed,
rate .03, and 512 fixed character checks each on Acts 1 training / reused
Acts 2 validation. The mixture learns a global blend weight, not attention
or an input-dependent selector. Equal target exposures are not equal compute.
Reports/checkpoints stay in `.data/neural-context-mix/<run-id>/`; active
chat, the teaching lab and historical latest-report pointers are unchanged.
First run: mixture character accuracy was higher (19.53% versus 17.51%
for context 4), but target loss was worse (3.134 versus 3.065, lower is
better) and training CPU about four times higher. This is mixed evidence,
not a multi-context advantage; see [full results](docs/FROM_SCRATCH.md#quick-multi-context-comparison).

Requested 5+12 versus 4-only variant:
```sh
python3 neural_context_mix.py --acknowledge-review --contexts 5 12 --pair-only
```
Same three seeds/1,500-target budget: mean character accuracy 19.27% mixed
versus 17.51% baseline, but loss 3.140 versus 3.065 (worse in all three
seeds). Mixture training CPU was ~3.6x baseline; no demonstrated overall win.

Triple variant: `python3 neural_context_mix.py --acknowledge-review --contexts 3 5 7 --pair-only`.
Three branches learn global softmax blend weights, initially equal.
Same short protocol: accuracy 20.64% versus 17.51% context-4; loss 3.176
versus 3.065 (worse in every seed), ~4.3x training CPU. Not an overall win.

Word/punctuation representation screening:
`python3 neural_word_context.py --acknowledge-review --known-targets-only`.
This is a **different, closed-64-word task**, not comparable to the character
accuracy above. Compare raw four-character context, two previous word IDs
without punctuation, and those IDs plus gap-punctuation signals, using the
same single MLP size, initial weights and 1,500 word-target draws per seed.
Unknown words remain in context; unknown target positions are excluded.
Three-seed known-word accuracy: 12.63/13.98/15.46%; loss
3.76434/3.75880/3.74443. Training CPU is approximately equal. Small promising
signal, not a 2x gain or demonstrated grammar/whole-vocabulary improvement.
The initial all-target version collapsed to predicting OTHER and is preserved.
See [screening limitations](docs/FROM_SCRATCH.md#word-identity-and-punctuation-screening).

Larger saved-corpus check:
`python3 neural_word_context.py --acknowledge-review --known-targets-only --larger --steps 5000`.
Exact archived Acts 6-12 training / 13-14 evaluation passages are verified
against the old benchmark hashes and inputs. Rerun matched word-task controls:
accuracy 18.38% raw4 / 19.01% words / 20.91% words+punctuation; loss
3.39667/3.44055/3.30765. Punctuation wins against both controls in all three
seeds on this closed-set task, with similar compute. Not directly comparable
to old character-generator accuracy and not a 2x gain. Active chat unchanged.

Historical full-generation five-arm comparison:
`python3 neural_base_comparison.py --acknowledge-review --include-retired-layers`.
Original character base versus a word/punctuation generator with character
fallback, then that new base with L0, L0+L1 and L0+L1+L2 extraction/placement.
Exact same larger saved passages, three seeds, 1/2/4 complete source passes,
and 256 unrestricted next-word probes; no unknown-target exclusion.
This is a new generative extension, not the earlier closed-word classifier.
Preparation/training/generation CPU and capacity are reported separately:
the new base is larger, not a punctuation-only matched-capacity control.
Experiments remain offline, with active chat unchanged.
See [protocol and limitations](docs/FROM_SCRATCH.md#full-generation-five-arm-comparison).
Run `10255643d34e41749c6bcc1f68ee8b5c`: four-pass accuracy
1.95/7.16/6.51/3.52/5.86%; preparation+training CPU
64.07/165.93/165.85/166.68/107.15s respectively.
L012 at one pass gives 5.99% in 30.96s versus original's best first-two-pass
mean 2.86% at a similar observed budget: promising average accuracy gain,
not an all-seed win. New-base fallback spelling remains poor; extra layers
do not consistently improve the new base. Restore all saved predictions with
`python3 neural_base_comparison.py --analyze-run 10255643d34e41749c6bcc1f68ee8b5c`.
All 60 states reproduce all 15,360 recorded continuations exactly.
Reduced L2 training work does not imply faster generation; restored inference
CPU is higher than both the original and the plain new base.

Open the **Base Model Lab** tab (production `http://localhost:3001/?view=base-lab`,
or the Vite client's same query path). It starts with authored text `ab ab ab `
and a fresh seed-17 instance of the real `NeuralModel`, not the active
Acts/Corinthians checkpoint.

Use **Next step** / **Back a step**, or the 11 stage buttons:
alphabet and token IDs → seeded parameters → context/target → embeddings →
hidden activations → logits → probabilities → target loss → backpropagation →
SGD weight update → prediction and whole-string loss after the update.
Hidden/output weighted sums can be expanded into their individual products.
The parameter inspector pages through every initial value or exact
before/gradient/change/after value in all five parameter groups.

**Train next character example** extends a reproducible training prefix by one
update (up to 32), cycling through the entered string in source order. One
update is not a sentence or a pass. **Build / reset teaching model** applies
new text/rate and returns to the first update. Step navigation reveals an
already computed trace; it does not trigger additional training. Tiny-input
limits are intentional: printable ASCII, at most 128 characters and 24
distinct characters. Inputs and teaching state are not persisted.

`POST /api/base-model-lab/trace` runs `neural_base_lab.py`, which calls the
shared base's actual forward, backpropagation and update methods, reads the
displayed source from that implementation, and includes its hash. It never
loads chat weights, reads transcript training data, saves model state or
changes memories. It is not the existing legacy extraction/model lab.
Diagrams are explicitly schematic; grids and tables contain the actual
values. Loss is measured on the tiny training string, not an unseen test.
See [the walkthrough notes](docs/FROM_SCRATCH.md#visual-base-model-walkthrough).

When a galaxy is promoted out of Under evaluation, its member memories and
associated replies move with it. Its parent tether to Under evaluation is
removed. Memory-to-memory links remain, including pairs whose other endpoint
is still under evaluation; those links do not confer galaxy membership.
Connections to other galaxies and original transcript provenance remain.
Promotion is permanent. A promoted galaxy never returns to Under evaluation, even
if its links weaken or a later similarity pass merges it into a larger group.
It keeps its own members, can be absorbed or split, and can be pulled into close
orbit by a sun; related memories can join it from the suns. Memories from a
merged group that were not in the galaxy are evaluated as separate clusters.
Parent-child galaxy relationships use solid green shell-to-shell connectors,
with hover labels identifying the child and parent. Cross-galaxy memory links
remain dashed. This styling does not change membership or learning weights.

When Under evaluation grows past its maximum drawn radius, the graph splits it
into two suns labeled Under evaluation and Under evaluation II. The split is
computed once at page load from the link structure: memories with stronger
mutual connections stay together, and the weakest-tied memories are pulled to
the second sun. Each sun is drawn at a fixed radius (`SUN_RADIUS` in
`generate_html_graph.py`) and then relaxed by a deterministic force layout so
connections pull related memories close and weaker links drag toward the new
sun. Membership is then refined so each memory sits with the sun that holds more of
its tie weight, keeping each sun between `SUN_BALANCE` and 1 − `SUN_BALANCE` of
the members; exact ties go to the smaller sun. Cross-sun links weaker than
`CROSS_SUN_MIN` are hidden, and the remaining ones are drawn with opacity that
follows strength, so decayed bonds fade. A memory outside evaluation draws its
link only to the sun it is most tied to. This is a display-only split; it does
not change stored weights.
The two suns are pushed apart by `BLAST_DISTANCE` and spread by `SUN_SPREAD` in
opposite directions, so they sit well apart rather than side by side. A promoted
galaxy with fewer than `BABY_INTERACTIONS` user interactions is drawn as a small
baby (`BABY_RADIUS`, currently 34px) just outside the sun it links to most strongly, on the side
facing the galaxies it links to most (weighted by link strength). It has no galaxy border or shading, so it reads as a standalone node. Hovering it shows its name, message count, and a list of its member memories, which hides on mouse-out. Hovering a row in that list shows that memory's details, and clicking a row selects it. Its member memories have no separate hover. It stays
promoted; only its drawing changes. Simple `related` connections are drawn between specific galaxy pairs (`RELATED_CLUSTER_PAIRS` in `memory_clusters.py`, currently Git / Aimem with Commit / Push). They link the `You` memories of the two galaxies, and the Git / Aimem memories also link to every AI Memory message that mentions "AImem". They are drawn as dashed yellow connectors, and they are rebuilt on every graph render. They do not merge galaxies or change membership. Within every galaxy and sun, each memory's depth comes from its own position relative to that galaxy's center: memories near the center sit at the front and are drawn larger and brighter, and those near the rim are drawn slightly smaller and dimmer. All memories stay opaque, so they read as stacked in front of the galaxy fill. A promoted galaxy takes its parent sun's color, so its memories match that sun's memories.

The conversation memory prototype stores its transcript in `chat_memory.json` and renders it in `memory_graph.html`. The browser view now keeps its learning overlay in a backend graph-state file served from the local API, so helpful, correction, and weak-link feedback survives refreshes without depending on browser `localStorage`. The graph page also polls that backend snapshot, so it refreshes itself when the stored memory state changes. Use the search box to rank memories by query and click a node to open the selected-memory panel. The zoom − and + buttons at the upper right of AI Memory scale the view. The feedback buttons teach the graph which memories should matter more. To add a conversation exchange and advance one persistent decay cycle:

```bash
python update_chat_memory.py \
  --user "What should be remembered?" \
  --assistant "A concise answer to remember." \
  --user-topic memory \
  --assistant-topic memory \
  --assistant-topic practical-use
```

This updates the local transcript and saved graph weights, then regenerates the HTML visualization and backend graph-state snapshot. Links combine turn order, manually supplied topic tags, sparse TF-IDF word similarity (up to two strongest text matches per message), and topic hub nodes that cluster related memories. The graph uses different line styles for those relationship types. The browser view also lets you reinforce or down-rank a memory interactively, and that feedback persists through the backend graph-state file. Search now uses our own word/phrase overlap calculations and a feedback-trained relevance model; if the server is unavailable, the page falls back to the viewer's lexical ranking.

## Locally Trained Miniature Model

### Neural weight lab

**Automated mechanics run:** **Run 50 synthetic rounds (5,000 updates)** uses
three original synthetic seed/pattern passages, never memories or the old
generator corpus. Each round is 100 SGD updates on training text only.
Validation is measured at round 0 and every round; fixed-prompt samples are
saved at rounds 0, 10, 20, 30, 40, and 50. The final test is evaluated once,
after round 50, with no early stopping or checkpoint selection. Related
synthetic patterns make this a mechanics check, not independent generalization.

Layer 2 counts are built **once** from training text and saved with descriptive
validation/final-test scores; repeating neural updates does not create new
Layer 2 observations. In this baseline runner they are not combined with the neural predictor;
the separate full-layer experiment below does connect them to learning.
Generated samples are never fed back into training. Layer 0/1 rules and
developmental admissions are unchanged.

The latest report is shown in Generation and restored on refresh. Each round
has a weights/report checkpoint under
`.data/neural-experiments/<run_id>/round-XX.json`; the usual run checkpoint
and latest report are also saved. Round 0 is the initial state. An ordinary
training failure surfaces explicitly and marks the latest report failed while
retaining earlier checkpoints. Abrupt process termination may leave a
`running` report; no automatic recovery/resume is implemented. Refresh to see
persisted progress if the request disconnects. The small fixed synthetic run
uses the existing 30-second API budget; it is not a general background trainer.

CLI equivalent: `python3 neural_rounds.py`. Without explicit continuation flags,
this starts a new independent 50-round run.

**Real-text Acts run:** with the locally staged dataset and explicit approval,
`python3 neural_rounds.py --acts --acknowledge-review` uses Acts 1 for training,
Acts 2 for validation, and Acts 3 for a single final test. It verifies source,
text, and rights-record hashes before training. The original source is the
public-domain World English Bible from eBible.org; its trademark conditions
are recorded in the manifest. Generated samples are model output, **not
scripture or the World English Bible**.

Real chapter runs exceed the API's 30-second budget, so run this command from
the terminal, not the synthetic training button. Generation displays the
latest saved result; after loading a running report it polls every three
seconds until completion/failure. Without `--resume-run`, each invocation starts fresh. Chapter splits
avoid direct passage reuse but share subject, vocabulary and style; prediction
loss is not a test of fact recall or understanding. Unknown validation/test
characters map to the unknown token and are reported, not silently added to
the training vocabulary. The staged manifest remains an immutable collection
record; each training checkpoint carries its own approval and provenance.

To continue a completed Acts run rather than restart:
`python3 neural_rounds.py --acts --acknowledge-review --resume-run <run_id> --additional-rounds 100`.
This validates the parent vocabulary, weights, dataset hashes and training
configuration, restores weights, and reconstructs the original SGD draw position.
A new child run preserves the parent unchanged, records its hash, and uses
cumulative round/step numbers. Up to 100 additional rounds are allowed per
invocation. Acts 3 is **not retested**: its earlier score is recorded as
historical, not a fresh final-test measurement. Validation remains monitoring
data. The UI shows the parent lineage and absence of a new final test.

**Controlled context comparison:** run
`python3 neural_context.py --acknowledge-review` offline to compare fresh
4-, 16-, and 32-character models on the same approved Acts 1/2 split.
All three process the same 5,000 target positions in the same order, with
character tokenization, batch size 1, learning rate 0.1, embedding size 8,
hidden size 16, and fixed initialization/training/sampling seeds.
Training/validation loss is measured initially and every 500 updates.
The three identical prompts include a 64-character training prefix so longer
contexts can use more prompt history; samples remain 120 characters at
temperature 1. Generated samples are never training data.

Longer contexts also add parameters and change initialization of later weight
groups, so this is a context-plus-capacity comparison, not a pure context-only
ablation. It uses one seed and reused validation, not proof of understanding.
Acts 3 text is neither read nor evaluated. No model is automatically selected
or installed, and Layer 0/1/2 policies and existing runs remain unchanged.
Generation shows a separate polling comparison table, samples and loss
histories. Reports persist at `.data/neural-experiments/latest-context.json`
and `<run_id>/report.json`; weights, exact texts and provenance are saved at
`<run_id>/context-N/step-XXXXX.json`. This does not overwrite the ordinary
`latest.json` report. Abrupt termination may leave `running`; ordinary numeric
or I/O failures propagate and attempt to record `failed`. There is no automatic
resume for comparisons.

To repeat the context comparison at a smaller fixed rate:
`python3 neural_context.py --learning-rate 0.03 --acknowledge-review`.
The rate is validated and saved in the report and every checkpoint; default
context comparisons still use 0.1. Do not combine `--learning-rate` with
`--learning-rates` (which varies the rate instead of context).
The latest-context pointer/display moves to the new run; previous reports and
weights remain unchanged under their original run IDs.

The repeat at 0.03 completed: validation loss was 2.6582 / 2.8046 / 2.9100
for contexts 4 / 16 / 32. All improved over their 0.1 counterparts, but the
four-character model still performed best at the fixed 5,000-update budget.
Outputs remain garbled; no model was automatically installed.

The first comparison completed 5,000 updates per model. Validation loss
(lower is better) was 2.8241 / 3.2384 / 3.3811 for contexts 4 / 16 / 32.
Larger context did not help under the unchanged settings; all outputs remain
garbled. This is not evidence that longer context is inherently worse.
See [the experiment record](docs/FROM_SCRATCH.md#controlled-character-context-comparison)
for training losses, limitations and verification.

**Controlled learning-rate comparison:** run
`python3 neural_context.py --learning-rates --acknowledge-review` to compare
0.1, 0.03 and 0.01 at a fixed four-character context. All three models have
1,978 parameters, identical starting weights and the same 5,000 target draws.
Only update size changes; batch size, tokenization, seeds and sample prompts
remain fixed. Losses, samples and checkpoints are saved every 500 steps.
Acts 3 text is neither read nor evaluated. Reused Acts 2 validation is tuning
data, not a fresh test, and one seed/short budget cannot identify the best
learning rate for every training duration or prove understanding.

Generation shows a separate polling learning-rate table alongside prior
context results and the ordinary run. Reports persist at
`.data/neural-experiments/latest-rate.json` and `<run_id>/report.json`;
checkpoints use `<run_id>/rate-R/step-XXXXX.json` and store the learning
rate with weights, texts and provenance. Existing reports/models are not
replaced or selected automatically. Comparisons share the same training
runner and failure/persistence behavior.

The first rate comparison completed: validation loss was 2.8241 / 2.6582 /
2.9080 for rates 0.1 / 0.03 / 0.01. Rate 0.03 improved prediction under this
fixed budget; all generated samples remain garbled. It is not automatically
installed as a new default. See [the rate experiment record](docs/FROM_SCRATCH.md#controlled-learning-rate-comparison).

**Full developmental learning and effort:** run
`python3 neural_efficiency.py --acknowledge-review` offline for the isolated
0/1/2 experiment. This is a new, explicit experimental numerical mapping of
the agreed layer roles, not a claim that earlier standalone neural runs used
the layers or that the mapping is proven.

- Layer 0 preserves all seven foundation forms and their existing maturity.
  Approved training observations introduce **unconfirmed** primitive/candidate
  blocks of 1–4 characters, shortest first. Longer blocks require both
  overlapping subblocks. A known foundation/candidate suffix gets availability
  weight 1; an unseen suffix retains 0.05 exploration, not a death verdict.
- Layer 1 applies the existing scoped exclusions before normalizing predictions.
- Layer 2 builds training-only attraction once. Its character probabilities
  retain 0.05 uniform exploration. These weights and Layer 0 availability
  multiply into neural probabilities **during cross-entropy and gradients**.
  This is not just post-generation filtering.

Controls are the unchanged neural baseline, full 0/1/2, each single layer
removed, and layers alone without neural weight updates. Neural arms use
identical initial weights and 5,000 identical target draws at context 4/rate
0.03. Removing Layer 1 also removes its veto on candidate construction.
Every arm reads all training text: zero gradient updates is not zero learning
work. The report counts candidate/context observations, layer state bytes,
preparation/update CPU and wall time, evaluation and checkpoint time, and
first measured match to this run's baseline final validation loss.
Repeated evaluations cache fixed priors; prediction timing alone is not
uncached serving latency. Samples compute priors at each character.

All original validation characters remain in evaluation. If a target is
forbidden, full-text loss is infinite (JSON `null`), with coverage and every
blocked offset reported; no filtered score can count as a threshold match.
Training rule conflicts are counted and skipped without source repair.
The complete texts, source hashes and candidate ledger remain in the run.
This tests prediction/effort, not agent competence or understanding.

The user explicitly approved isolated candidate growth. These candidates do
not become confirmed forms, change the permanent admission registry, or
approve other sources. Acts 3 text is not read. Acts 2 is reused monitoring
data, not an untouched test. Generated samples are never training evidence.
Neither ordinary chat nor any saved baseline is replaced.

Generation polls a separate full-layer report with quality/effort tables,
coverage, samples, raw history and exact policy. It persists at
`.data/neural-experiments/latest-layered.json` and `<run_id>/report.json`.
Each arm stores `evidence.json` with exact texts/ledger/provenance and
`step-XXXXX.json` with weights and results. Reports distinguish ordinary
numeric/I/O failures from completion; interruption can leave `running`.
There is no automatic resume or checkpoint selection.

First integrated run: full 0/1/2 reached the baseline's 2.6582 validation
target **before neural updates**, after building the layers. Its preparation
took about 0.452 s versus baseline preparation+5,000 updates at 2.586 s on
this machine. All paths still read the entire training chapter. Final full
loss was 2.6389; layers alone scored 2.6360. Coverage was 100%, not a score
improvement from hiding rejected text. Timing is descriptive and includes
additional layer construction, not a universal speed claim.

The ablations also found a limitation: removing the **current numerical
Layer 0 penalty** improved validation to 2.1276. That mapping needs review;
this is not evidence that the foundation concept should be removed.
Layer 1's effect was tiny on this corpus, and full-path neural updates added
little beyond the learned count prior. Output remains garbled, not a viable
agent. See [the full result and costs](docs/FROM_SCRATCH.md#first-integrated-result).

**Proof-shaping run:** `python3 neural_proof.py develop --acknowledge-review`,
then `preregister --run-id <id>`, `stage-fresh`, and `evaluate-fresh
--acknowledge-review`. It compares 10 paths (neural baseline, neural+layers,
counts-only layers, Layer 0 v1/v2, and a from-scratch standard Kneser-Ney
counting model) across 10/25/50/100% of Acts 1 and five seeds, with estimated
operation counts and generated rule-break rates. Criteria are frozen before
Acts 4–5 are retrieved into `.data/acts-fresh-staged/`; the fresh test runs once.
Results persist in `.data/neural-proof/` and
`.data/neural-experiments/latest-proof.json`; Generation shows them.

Result: Layer 2 counting matched and beat the neural baseline on fresh Acts 4–5
(2.095 vs 2.547) at about 0.07% of its estimated operations, and 25% of the
data was enough. Layer 1 removed generated rule breaks. But standard counting
beat every layered path (1.940), and Layer 0 (v1 and v2) still hurt loss. See
[the proof record](docs/FROM_SCRATCH.md#proof-shaping-run-size-curve-seeds-effort-rules-and-a-fresh-test).

**Layer 0 only states:** `python3 neural_onlies.py --acknowledge-review`
earns one-way configurations (2–9 characters, forward and backward) from
Acts 1, keeps broken ones as mutation variants, and compares them as
non-forbidding Layer 0 guidance. They held 85–96% on Acts 2 but improved loss
by only 0.004 at best; strict settings hurt. Saved in `.data/neural-onlies/`
and shown on Generation. See [the record](docs/FROM_SCRATCH.md#layer-0-only-states-one-way-configurations).

**Word-level only states:** `python3 neural_word_onlies.py --acknowledge-review`
earns one-way (word always followed by one word) and opposite (word always
preceded by one word) only states from Acts 1 and compares them as
non-forbidding guidance. Only 2–7 forward and 4–15 opposite onlies were
earned (e.g. "Holy→Spirit", "into→the"); they held just 17–57% on Acts 2 and
did not change loss meaningfully (best 2.1319 vs 2.1324). See
[the record](docs/FROM_SCRATCH.md#word-level-only-states-one-way-and-opposite).

**Fixed patterns (only chains):** `python3 neural_patterns.py stage|develop|preregister|stage-locked|evaluate-locked`
learns triggers of 1–9 characters that were followed by exactly the same
characters every time they appeared (at least 10 times) in public-domain
Acts 1, 2, 4–27. In reading it skips the characters a pattern supplies; in
generation it writes the whole continuation as one decision. On untouched
Acts 28, tested once, patterns correctly supplied 19.1% of the characters (target
10%, met), but only 94.1% of what they claimed was right (target 95%, not
met). Reading needed 6.9% fewer decisions. Source chapters, hashes and the
pre-registration are saved in `.data/acts-patterns-staged/`,
`.data/acts-patterns-locked/` and `.data/neural-patterns/`. See
[the record](docs/FROM_SCRATCH.md#fixed-patterns-only-chains-for-fast-reading-and-generation).

**Strict and light layers:** `python3 neural_strict.py --acknowledge-review`
splits that idea in two. The strict layer ("if it sees this, it is this")
claims a known word or phrase only after confirming the whole unit in the text,
and in generation spells a chosen word out whole. It never guesses a word from
its start, so it makes 0 wrong claims by construction. On Acts 27 (development)
it recognized 79% of letters and saved 49% of reading decisions; 166 new words
were handed to other layers. The light layer gives the 94–95% fixed patterns
weight only, which lowered Kneser-Ney loss from 1.679 to 1.664. See
[the record](docs/FROM_SCRATCH.md#strict-layer-and-light-pattern-layer).

**Strict skeletons:** `python3 neural_skeletons.py stage|develop|preregister|evaluate-locked`
tests partial patterns with hidden or missing letters (`th*t`→that) that only one
possible word fits. Layer 0's list of possible words is the SCOWL word list
(notice in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)), the learning words,
and their regular forms. Names and hidden first letters are handed to other layers.
On public-domain Bible books locked away until a single pre-registered run, strict
skeletons were wrong 1 time in 560,404 checks (95% upper bound 8.5e-6, under the
1-in-100,000 target). This holds for one Bible translation only. See
[the record](docs/FROM_SCRATCH.md#strict-skeletons-hidden-or-missing-letters-that-resolve-one-way).

**Broader fragment comparison (not active training):** a separate local inventory
in `.data/possible-words-expanded-scowl-2020.12.07/` unions all 358 SCOWL
2020.12.07 final lists through size 95 with the existing list: 644,145 distinct
lowercased entries, retaining compounds, names and variants. Source archive,
hashes, full notices and fragment match lists are preserved. This historical,
deliberately inclusive inventory is not all English or Oxford/Webster; its
largest lists include questionable entries. Most earlier small-list unique
keys fail here: `wev` matches 11 entries, `couldn` four, and `etz` 140.
`ven't` and `esn'` still match only `haven't` and `doesn't`, respectively.
No replacement of existing possibilities, model changes or new training occurs.
SCOWL/UKACD notice: Copyright (c) J Ross Beresford 1993-1999.
All Rights Reserved. Retain the full accompanying `Copyright` document for use
or redistribution; these are third-party lexical data, not original project data.

The **Generation** tab now includes a separate, from-scratch character neural
network: four-character context, 8-value embeddings, a 16-unit tanh hidden
layer, softmax predictions, cross-entropy loss, hand-written backpropagation,
and stochastic gradient descent. It uses only Python's standard library.
Parameter count depends on the training-only vocabulary and is shown exactly.
This is a tiny neural language-model experiment, not an LLM or a replacement
for the existing character-count generator.

**Expanded recognition (opt-in):** enable **Use expanded only-state recognition**
before training reviewed text. Both modes use the same character-only vocabulary,
parameters and four-character context. Exact recognized forms skip their internal
spelling updates; first characters and surrounding choices still train.
Generation supplies a suffix without neural predictions only when a prefix is
unique across the declared possible set. Ambiguous prefixes remain predictions;
boundaries around rule-completed forms are enforced.

Build the local reusable bank once from the already approved Bible learning and
development books (never the locked books):

```bash
python3 neural_skeleton_runtime.py build
```

The UI reports skipped updates and rule-supplied characters. **Generate from saved
neural checkpoint** restores the weights and recognition rules without the global
bank. The baseline stays available. This does not change neural tokenization,
replace chat or automatically train on memories. Earlier whole-word-output
checkpoints remain loadable as legacy experiments, not the intended method.
See [the correction](docs/FROM_SCRATCH.md#corrected-character-only-recognition-and-retest).

**Measured comparison:** `python3 neural_unit_benchmark.py --acknowledge-review`
runs three paired seeds on previously approved Acts 6/7 text, with equal-pass
and equal-update budgets and identical next-word questions. The earlier expanded-
output experiment was slower; it is retained with `--legacy-units`. The corrected
character-only path skips spelling work without adding output classes. Neither
comparison establishes coherent answers or general-English accuracy. Results
appear in Generation; see [the comparison record](docs/FROM_SCRATCH.md#paired-live-unit-efficiency-comparison).

**Larger retest:** add `--large` for complete Acts 6-12 training, Acts 13-14
evaluation, five paired seeds, 256 shared next-word questions, and equal-pass /
120,000-update budgets. The saved larger run used **30% less update CPU time**
at four passes (65.75s baseline, 46.08s recognition). However, its shared
all-character prediction loss was worse (2.685 baseline, 2.959 recognition).
This confirms skipped-update savings, not equal overall quality or coherent
answers. See [the larger-run results](docs/FROM_SCRATCH.md#larger-recognition-only-comparison).

**Three-way comparison:** add the baseline-plus-rules arm to those saved checkpoints:
`python3 neural_unit_benchmark.py --acknowledge-review --add-combined 41624f6d045f4a6b9d202ed2cb877b67`.
This keeps every baseline training update and applies the same only-state
generation rules. It reuses exact baseline weights and training timings rather
than retraining identical weights. A new report preserves the original evidence,
compares all three arms, and freshly measures equal-update generation samples.
The combined arm matched baseline quality at every saved milestone, but no
rules activated in its probes or the fresh samples: it avoids skipped-update
degradation without yet demonstrating an added benefit.
See [the combined comparison](docs/FROM_SCRATCH.md#baseline-plus-rules-comparison).

**Atomic indexed-onlies test:** `python3 neural_unit_benchmark.py --acknowledge-review --indexed-onlies`
tests the clarified representation: one indexed pattern is one training target,
context position and generation selection. It reuses the existing unit codec,
not the skip-spelling path; the live training checkbox is unchanged.
The three-seed test emitted 73 indexed patterns with fewer generation decisions,
but training was about 48% slower at equal passes and quality did not improve
overall. See [the atomic test](docs/FROM_SCRATCH.md#atomic-indexed-onlies-test).

**Separated placement index + remainder base:**
`python3 neural_index_pipeline.py --acknowledge-review` removes entire indexed
occurrences from the base's training text. A separate context-count mechanism
learns insertion; the base has no indexed outputs or embeddings. The small
three-seed test used about 35% less total training CPU, including index learning,
and fewer generation calls, but word edit distance worsened. This is an isolated
experiment, not a replacement for live training.
See [the separated test](docs/FROM_SCRATCH.md#separate-placement-index-and-remainder-base).
Add `--large` for complete Acts 6-12 training / Acts 13-14 evaluation,
five paired seeds and 256 shared word probes; the learning policy and
1/2/4-pass checkpoints remain unchanged.
The larger run used 30,905 training characters: total training CPU was
64.81s baseline / 51.39s separated (~21% less). Word accuracy was
1.25% / 2.73%, but edit distance remained slightly worse (0.879 / 0.888).
The unchanged 128-pattern cap removed 20.6% of the training characters.
Short generation samples also used ~21% less CPU; coherent output is not established.

**Different-corpus test:** run
`python3 neural_index_pipeline.py --acknowledge-review --corpus fixtures/neural-everyday-prose.json`
for newly AI-authored workshop prose (1,942 training / 957 evaluation characters),
with explicit provenance in the fixture. The existing pattern bank remains fixed.
At four passes, total CPU was 3.23s baseline / 2.22s separated (~31% less),
but both word accuracy (5.56% / 2.78%) and edit distance favored baseline.
This controlled artificial corpus does not establish real-world transfer.
See [the prose comparison](docs/FROM_SCRATCH.md#different-corpus-everyday-prose).

**Candidate cascade:** add `--candidate-layers` to the small prose command to
compare base alone, Layer 0 + base, Layers 0-1 + base, and Layers 0-2 + base.
Layer 0's seven seed configurations remain unchanged; Layer 1 indexes existing
whole-form candidates, and Layer 2 tests repeated alphabetic 3/4-character
blocks plus hidden-backed complete matches. No literal missing characters
are guessed, and the old generation exclusions remain deferred.
The first run cut training CPU ~51% with Layers 0-2, but worsened quality.
See [the four-arm comparison](docs/FROM_SCRATCH.md#four-arm-candidate-cascade).

**Corrected L1 (current experiment):** use `--candidate-layers --unique-l1`.
This replaces the previous whole-word eligibility rule with the shortest
unique contiguous fragment, checked against the expanded 644,145-entry
inventory plus all training words. No full-word fallback: each key must be
shorter than its complete payload and have exactly one normalized owner.
The entire exact bounded word is extracted; base learns only the remainder.
The old whole-word cascade/fraction flags remain historical controls, not
the corrected L1 definition. Base/L0/corrected-L01 small prose comparison
selected `eturns → returns` and `nplugge → unplugged`, removing 16 additional
characters. Four-pass accuracy was 5.56/5.56/15.97%, edit distance
.832/.792/.727, but corrected L1 lost on word accuracy at earlier milestones.
No stable quality or efficiency win is established. See
[the corrected trial](docs/FROM_SCRATCH.md#corrected-l1-unique-fragments).

**Atomic contextual L01:** use `--candidate-layers --unique-l1 --atomic-l01`
with the same corpus for three fresh arms: base, corrected remainder-only
L01, and corrected atomic L01. Each recognized reference becomes one target
and one context position; the neural model learns its placement and what
follows it. The fixed decoder supplies spelling. No pretraining, injection,
spelling replay or separate placement learner in the atomic arm.
Four-pass word accuracy was 5.56/15.97/5.56%, edit .832/.727/.788;
atomic training CPU was ~10% higher than base. Atomic size is 53 outputs /
1,853 parameters versus 44 / 1,628 for the controls. None of the 48 common
evaluation probes targets an L1 word or follows a recent reference; sample
references were L0 only. This does not demonstrate learned L1 usage.
See [the atomic trial](docs/FROM_SCRATCH.md#atomic-contextual-l01).

**Matched pretraining and refresh:** use `--candidate-layers --unique-l1
--pretrain-l01 --matched-reinforcement`. Both character and remainder-L01
arms restore identical per-seed pretrained weights and receive one extra
isolated literal pass for every recognized source occurrence (19 refreshes,
67 extra character updates per pass each). Both use source-order passage
updates to align replay timing, unlike earlier shuffled trials. Four-pass
accuracy was 2.78% character / 10.42% indexed; edit .842/.801; total CPU
5.785/5.698s including pretraining and replay. The character arm won at two
passes; no consistent improvement or permanent spelling retention established.
See [the matched trial](docs/FROM_SCRATCH.md#matched-pretraining-and-occurrence-reinforcement).

**Whole Acts learning run:** add `--all-acts` to the matched command (omit
`--corpus`). Both fresh arms train on all 28 chapters, including previously
held-out chapters, with identical per-seed pretraining and occurrence refreshes.
These are explicitly **in-sample learning checks, not held-out evaluation**.
The runner requires 20 distinct recognized L1 occurrence targets before
training and merges them with 256 general probes, deduplicating offsets.
Focused scores and their support are displayed separately. Historical locked
results are preserved, but Acts is not unseen evidence for this new run.
First whole-book result: 124,762 characters, 124 L1 forms, 20 dedicated targets
across 18 words. Four-pass in-sample accuracy was only .12% base / .48%
indexed; dedicated targets scored 0/20 in every base seed and 1/20, 0/20,
0/20 indexed. These poor scores do not demonstrate meaningful learning gains.

**Historical layered app chat (retired):** the whole-Acts layered remainder model was initially selected:
run `59fbb1a0a36144e588ac1faae14d9654`, arm `layers_01`, seed 17, four passes.
Seed 17 is the first protocol seed, not a best-score selection. This is the
2,203-parameter neural model **plus its learned placement index**, not the
atomic/injected variant or the pretraining-only checkpoint. Its output is
experimental continuation, not a conversationally trained answer or scripture.
The Corinthians continuation below later replaced it. Both are now retired
from active chat in favor of the layer-free New Base; their checkpoints remain preserved.

`neural_chat.py` stores a checksum-pinned local checkpoint and selection in
`.data/neural-chat/`; `GET /api/chat-model` reports the active identity.
New benchmark runs and Generation-panel training do not replace this selection.
Chat saves exchanges and per-reply checkpoint provenance in `chat_memory.json`,
but does not update neural weights/index or ingest prompts into the legacy
generator. Historical neural ratings were review-only; the current conversation
learner trains explicitly Relevant pairs and submitted corrections as described
above. Optional
model-lab comparisons still use the separate legacy generator and never
override the active neural response. Missing/corrupt active checkpoints cause
an explicit API error, not a silent fallback.

Historical rollback only (not the working model):
```sh
python3 neural_chat.py activate --run-id 59fbb1a0a36144e588ac1faae14d9654 --seed 17 --passes 4
python3 neural_chat.py status
```
The saved experiment is required for activation. Deployments without an active
selection retain the previous generator/retrieval behavior. To roll back,
rename `.data/neural-chat/active.json` to an unused backup filename; retain the
checkpoint and legacy generator data. See [the runtime notes](docs/FROM_SCRATCH.md#active-layered-neural-chat).

**Corinthians continuation:** `neural_continuation.py` continues both arms of
the active checkpoint's paired experiment on **all of 1 and 2 Corinthians**,
using the same publisher's public-domain WEB text (29 chapters, 694 verses).
Raw source, extracted verse text, publisher rights notice and hashes are kept
in `.data/corinthians-web-staged/`. The current approval covers both complete
books for four additional passes, comparison, and subsequent layered activation.

```sh
# Stage once; existing source snapshots are never overwritten.
python3 neural_continuation.py stage --acknowledge-review
python3 neural_continuation.py train --acknowledge-review --activate
```

Training resumes the selected layered seed **and its corresponding baseline**
from saved final weights, not fresh weights or initial pretraining. Existing
layer references and placement counts survive; L1 eligibility is rechecked
against the expanded dictionary plus all prior/new training words. Newly
eligible words are appended without renumbering references. Both arms receive
the same occurrence-triggered literal refreshes and source-order passes.
Old neural token IDs/weights are preserved if new characters need appending;
new rows copy the arm's unknown-token representation (which changes softmax
normalization). This Corinthians snapshot needs no new characters.

Checkpoints at one, two and four additional passes include **in-sample**
Corinthians word checks and Acts retention checks, before/after, using one
paired seed. Acts is not replayed; retaining old knowledge is measured, not
guaranteed. These are not held-out generalization or multi-seed results.
The prior chat checkpoint stays active until both arms complete. Activation
refuses to overwrite a selection changed during training. Original
checkpoints remain preserved; reports record hashes, parent lineage and code
versions. Omit `--activate` to retain the current selection after training.
The latest-benchmark panel remains historical; active chat reports its own
selection and cumulative source history separately.

Completed continuation `1f7ed40a7b614dbba3c612a572782ca1` was previously active
(layered seed 17, four **additional** passes). The neural model remains 2,203
parameters; L1 grows from 124 to 176 forms. On 256 fixed in-sample Corinthians
word probes, baseline accuracy changed .78% → 1.95%; layered 1.17% → 1.17%.
On 256 fixed Acts retention probes, baseline .39% → 3.12%; layered
1.17% → .39%. The layered model did not improve new-book accuracy and lost
retention on these probes; activating it is the user's choice, not a claim
that it outperformed the baseline or became meaningfully smarter.

**Pretraining and protected injection:** add `--pretrain-l01 --inject-l01` to
that corrected command. Each seed starts fresh, trains only the seven literal
L0 payloads and the two selected L1 words for 200 passes, and saves a reusable
pretraining-only checkpoint. Two extra arms start from those exact weights:
ordinary remainder training, and remainder training conditioned on frozen
16-value reference vectors. Indexed targets are not replayed. The frozen
model/vectors remain intact; active neural weights and their interpretation
can still change. Four-pass accuracy base/corrected/pretrained/injected was
5.56/15.97/9.72/8.33%; total training CPU includes ~2.52s pretraining per
pretrained arm. Protected storage worked, but injection did not improve quality
over either corrected L01 or ordinary pretraining. See
[the protected trial](docs/FROM_SCRATCH.md#pretraining-and-protected-reference-injection).

**Selective L1 trial:** add `--selective-l1` to that command to compare base,
L0, selective L01 and full L01. A fixed training-only gate requires repeated,
high-purity preceding contexts and potential character savings. On this small
corpus, none of 85 forms qualified; the selective arm reproduced L0 exactly.
This did not identify an intermediate speed/quality sweet spot.
See [the selective trial](docs/FROM_SCRATCH.md#selective-l1-trial).

**Intermediate L1 coverage:** use `--candidate-layers --savings-l1` with the
same prose corpus to compare base, L0, top 25%/50% of repeated L1 candidates,
and full L01. Training-only character savings rank the forms; counts round up.
Five/nine of 18 repeated forms were selected. At four passes the 25% arm
matched base word accuracy with ~5% less training CPU and slightly better
edit distance, but neither intermediate arm beat L0's edit distance.
The advantage was not consistent across milestones; no sweet spot established.
See [the fractional comparison](docs/FROM_SCRATCH.md#intermediate-l1-coverage).

**L2 alone:** use `--candidate-layers --l2-only` with the same prose corpus.
This disables L0/L1 and discovers L2 candidates directly from training.
The first paired run saved ~52% training CPU but worsened word accuracy
(0.69% vs 5.56% baseline) and edit distance (0.953 vs 0.832).
See [the standalone ablation](docs/FROM_SCRATCH.md#standalone-l2-ablation).

Start with **Run synthetic learning check**. This original repeating-pattern
fixture demonstrates weight updates, not generalization. For real experiments,
provide new training text, separate held-out text, a provenance/rights note,
and explicit approval for this neural use. Existing corpus approvals, memories,
and chat prompts are **not** automatically reused. Approval is a user
declaration, not legal clearance.

Each UI training run starts from deterministic fresh weights and saves its own checkpoint.
The report shows trainable/changed parameters, steps, processed target tokens,
initial/final samples, and training/held-out loss at step 0 and every 100 steps.
Held-out text does not update weights or determine the vocabulary; unknown
characters are reported. Do not select/tune against held-out text repeatedly
and then describe it as an untouched final test.

Full submitted text, provenance, hashes, weights, and report are saved locally
under `.data/neural-experiments/<run_id>.json`; `latest.json` holds the latest
report. These ignored files can contain sensitive text. The character baseline
does not apply or soften Layer 0/1 rules; the optional recognition path uses the
bank to avoid relearning confirmed spelling. Its loss covers only unskipped
targets, so it is not directly comparable to baseline all-character loss.
No word output classes or end marker are added. The bank contains prior Bible
learning/development evidence, so arbitrary held-out text is not necessarily
unseen by that bank. The existing four-mode chat lab is unchanged.

```bash
echo '{"synthetic":true}' | python3 neural_experiment.py train
python3 neural_experiment.py status
echo '{"run_id":"REPLACE_WITH_SAVED_RUN_ID","prompt":"and "}' | python3 neural_experiment.py generate
python3 -m unittest -q test_neural_experiment
```

### Layer 2: attraction evidence

New chat lab comparisons include a separate **Layer 2 attraction evidence**
section (introduced in lab schema v5). It measures exact overlapping 2–4-character patterns
in the current prompt, preserving spaces, case, and zero-based source offsets.
`am`, ` am`, `am `, and ` am ` are different observations. Three-character
contexts can distinguish word completion from continuation without declaring
a whole longer word understood.

For each pattern, count / context occurrences estimates forward likelihood.
Context occurrences at source end remain in the denominator as non-character
outcomes, rather than disappearing and inflating attraction. Source-start
offsets are explicit; no synthetic marker characters or cross-source windows
are inserted into this evidence.

Pairs also report forward and backward likelihoods: “a is always followed by
m” and “m is always preceded by a” are separate claims. Both must hold for a
mutual observed-always pair. This means **only in this source**, even at support
one, not a universal rule. Counts and Wilson 95% lower bounds are provided;
the bounds assume independent trials, which overlapping or repeated text does
not supply. Missing patterns are unknown, not impossible.

Lab schema v6 adds two experimental generation modes alongside the four
existing baselines:

- **Layer 2 alone + Layer 1 exclusions:** sample prompt-local character
  probabilities without requiring a captured living seed.
- **Old model + Layer 2 next-character scoring:** weight each available old
  four-character-context transition by its Layer 2 attraction before sampling.

Both operate **outside Layer 0 admission**, explicitly not an expansion of
its seven forms. Both reject every next character that would trigger a known
Layer 1 exclusion. If all options are excluded, they stop and say why, rather
than return forbidden text or silently fall back.

The Layer 2 distribution starts with prompt character frequencies and blends
one-, two-, then three-character contexts using `support / (support + 2)`.
Observed source ends are possible stop outcomes. Old-model scoring uses
`old_count * (0.05 + attraction_probability)`, preserving an explicit small
weight for old candidates unseen in the prompt. No matching old context means
stop, not an unrelated restart. New modes share seed/budget, with sampling
traces, candidate weights, rejected options, and stop reasons downloadable.

Each card reports equality to old output, prompt-fit negative log likelihood,
adjacency support, and detected Layer 1 violations. **These are descriptive,
not independent accuracy or improvement measures.** Lower prompt-fit loss may
mean imitation, not better answers. Baselines retain their algorithms; the
aligned overlay is a narrow A/I/initial-consonant edit, not reversed model
execution, so it will often equal old output. Identical results are not forced
to differ.

There is still no persistent attraction model or neural update. Reports persist
in downloadable lab records, not training checkpoints. Existing corpus
approvals do not authorize a new Layer 2 model. Ordinary chat still saves the
old response; the two new samples are lab-only. Older lab reports retain their
original modes.

### Seed rules and existing generator

#### Developmental review and composition

The original seven Layer 0 forms remain the small foundation, not a claim that
all other configurations are universally impossible. `seed-life-v5` retains
the old alive/dead/maybe extraction groups and adds separate developmental
metadata: **developing** for an incomplete observation, **unreviewed** for a
completed unknown, **confirmed** for an exact living foundation form, and
**excluded** for a known violation. Foundation membership is still a separate
field. Completing unknown text does not confirm it; frequency does not admit it.

Lab v7 replaces the old seed-block-only card with **Developmental layers:
blocks → exclusions → attraction**. It uses living captures from this prompt
and explicitly admitted blocks, checks the entire proposed join against Layer 1,
then ranks surviving blocks with Layer 2 prompt attraction. It preserves
start-only seed placement. Existing saved lab records are not rewritten.
The old-model continuation/overlay and the two outside-admission experimental
modes remain distinct comparisons, not promises of fully admitted output.

In a new Chat lab, open **Review a building block for lab generation**:

1. Select an exact 1–4-character pattern by zero-based Unicode character offset
   and length in the saved prompt. Review its exact preview.
2. Provide a rationale and choose defer, admit, or revoke.
3. Explicitly acknowledge lab-only use. The backend verifies the saved source
   hash, checks the candidate boundary and two characters of surrounding
   composition context on each side, and refuses admission on a known exclusion.
4. Run a new comparison. Admissions retain their observed source-start or
   interior placement; deferring or revoking removes them from future generation.
   The existing report remains a historical snapshot, not a live preview.

These are mechanical exclusion checks, not proof of meaning, safety, rights,
or universal composability. Each later generated join is checked again. A
revocation does not erase prior evidence or saved outputs.

The append-only decision history is local in `.data/seed-development.json`.
It retains exact source text/hash, saved lab ID, offsets, tests, rule version,
domain, rationale, and timestamps; treat it as potentially sensitive. Reports
save the registry snapshot/hash and selected block provenance for reproducibility.
Admissions **do not change Layer 0, approve training, feed the neural model,
or update ordinary chat/memory**. No patterns are admitted automatically.

Layer 1's scope is `english-seed-experiment`; source start, literal spaces,
and observation completion are explicit. It does not infer a new source from
a newline or sentence punctuation. Its catalog records rationale, boundary
conditions and examples; unsupported domains fail explicitly, rather than
silently inherit English seed exclusions. `t` while developing is not `t `
at source start. A source end, a word boundary, and a generation budget stop
do not establish grammatical completion.

An isolated English seed experiment explores Layer 0 possibilities and Layer 1
life/death rules before connecting them to training. It preserves exact
whitespace and returns `alive`, `dead`, or `maybe`:
Each result also reports `layer_zero_eligible`: a reviewable `maybe` outside
the frozen seven-form Layer 0 is not eligible for life. Layer 1 only excludes;
passing its exclusions does not confer life or training approval.

```bash
python3 seed_life.py ' I '
python3 seed_life.py ' i '
python3 seed_life.py '</>I'
```

See [Building Our Learning Core](docs/FROM_SCRATCH.md#english-seed-experiment)
for the agreed rules and open questions. This experiment does not change model
weights, memories, or the graph.

For a manual extraction experiment on a UTF-8 text file, run:

```bash
python3 seed_extraction.py path/to/review-text.txt
```

It saves unrecognized 1–3 character windows for review in
`.data/seed-extraction-review.json`, with source hashes and character offsets.
Known-dead windows are recorded separately, not added to the review queue.
Each saved pattern carries its Layer 0 eligibility separately from review
status. Older extraction records are preserved under their original ruleset.
At source offset zero, `A ` and `I ` use terminus context and are eligible;
identical interior fragments remain separately collected for review.
This does not train a model or interrupt normal memory updates. The queue
contains fragments and source paths, so treat it as potentially sensitive.

Seed extraction now also runs **alongside current generator training** for new
approved documents and opted-in chat prompts. Exact submitted text and pattern
positions are saved in `.data/seed-training-review.json`, with source metadata
and both original-text and normalized-training hashes. The Chat and Generation
views show extraction counts. Unknowns remain unapproved for seed training;
dead patterns are separate. The existing generator still trains on the full
approved text, including patterns outside the frozen seed universe.
Existing corpus documents are not automatically reprocessed. A queue-write
failure is reported and prevents that submission's model update; normal chat
memory capture still proceeds with the training error reported. Review records
are retained when a generator source is removed, but no longer counted as active
corpus extraction. These queues now retain full source text: treat them as
sensitive local data, not just pattern summaries.

### Chat comparison lab

Chat enables a pre-training lab comparison by default (toggle below the prompt).
It shows two extraction panels, visible above generation: the old generator's
normalized four-character context/next-character captures and the combined
Layer 0 + Layer 1 raw pattern captures. Layer captures show alive, review, and
dead groups with exact source positions and eligibility. Full downloadable
records also retain unfiltered windows and each layer's separate assessment.

Generation is shown separately for the existing generator, a rule-based
captured-seed block sampler (not a new trained model), and layers-first
generation. The combined mode protects the earliest living captured seed as
an opening and asks the existing generator for a continuation. Source-start
captures use start context; interior captures use an explicitly labeled
suffix-context backoff. No match or no capture is reported, not replaced with
unrelated generation. Only the opening is confirmed by the layers, not the
whole continuation or the concatenated layers-only output.

Each mode has a 320-character budget. The main comparison crops all four to
the shortest actual output so displayed lengths match without padding; actual
lengths, shortfalls, and full outputs remain visible. Zero common length is
reported when a mode cannot produce anything. In that case the main cards show
available full outputs at their actual lengths, explicitly not an equal-length
comparison; empty modes show their reason instead of hiding other results.
The ordinary chat reply remains
the existing generator's full response (or the existing retrieval fallback).
All comparisons precede training on that submission, but earlier submissions
may already be in the corpus. Exact comparison records are saved in
`.data/model-lab/` and downloadable from chat, including prompt, checkpoint and
scorer hashes, ruleset, seed, and generated outputs. Treat them as sensitive.
These records and experimental outputs are not automatically training data.

A fourth generation mode uses the old-model response as its base and overlays
only aligned corrections supported by living captures. Captured ` I ` can
correct ` i `; initial captured `I ` or `A ` can correct the corresponding
lowercase opening, and captured ` A ` can correct matching ` a `.
Other text, spaces, and output length remain unchanged. No seed blocks are
appended or forced into unrelated words. Each correction shows its output
offset and supporting input capture/offset. No match is explicitly reported.
This is not a trained new model or proof that the combined text is viable.
Older saved three-mode and append-mode records retain their historical display.

Layer 1 also excludes a terminus followed by one English consonant (either case)
and a literal space. Lab version 4 repairs that initial consonant in the overlay
using an actual living source-start `A ` or `I ` capture from the current input.
It preserves the remaining text and length and records the exclusion and capture.
Without that capture it reports the dead opening as unresolved, not repaired.
Interior consonants and initial `t.` are not covered by this rule. Extraction
uses the source-start context and records these exclusions under `seed-life-v4`;
older ruleset records remain intact. Layer 0's seven possibilities are unchanged.

**Generation is an early from-scratch prototype.** The app's Generation tab
trains a character n-gram model only from text you explicitly approve there.
Until at least one source is approved, chat keeps the stored-memory excerpt
fallback. Once ready, chat samples character continuations from the local model.
This first step is expected to be incoherent; it does not understand meaning or
ground its output in retrieved facts. Old saved responses remain historical
records; submit a new prompt to use the current generator.

Trainer submissions now save the question and its source-linked local response
as a memory exchange. A response uses one top-ranked excerpt capped at 600
source characters, and evaluations show at most three memories. Saved source
judgments display Evaluated instead of grading controls across refreshes;
use Revise judgment only when intentionally changing one. Response controls
hide after a successful rating and remain hidden on reload in the same tab.
Failed saves leave controls available.

Retrieval happens before capture; refresh searches exclude
the current question's memory ID. Reloading the page restores the latest
evaluation from tab-local session storage and does not submit it again. Each
intentional new submission creates a new exchange. A saved question is not a
verified factual assertion or automatic training approval.

The graph's prompt evaluation labels whether a response is a retrieved memory
excerpt or experimental local character output, and shows approved generator
source titles. Response useful/Needs correction and optional user corrections
are stored separately in `.data/response-feedback.json`, with response IDs and
content hashes. Unmarked answers do not train. In current conversation chat,
Relevant and submitted corrections explicitly approve immediate neural updates;
successful feedback records include the trained checkpoint hash.
Grade cited memories separately to train retrieval. Generated output is
never added to the legacy generator corpus automatically.

For the fastest in-app workflow, open `/graph`, enter a prompt in **Prompt
training**, and press Enter. Shift+Enter keeps a newline. The current question, results, and
latest judgment/evaluation are at the top of the sidebar. Evaluation runs on
submission, not on every keystroke. Training-mode results are user-authored
and assistant replies, with assistant text visibly marked unverified. On the
first grading action for your own message in a tab, a confirmation explains
that each named judgment also confirms review of its displayed memory/current
prompt for training. Cancel skips the judgment; accepting enables that scoped
workflow for later grading in the same tab. Then choose Direct answer, Related
context, or Unrelated. That click records review and feedback and attempts
training without CLI commands. It does not bulk-approve stored memories or the
teaching draft. Grades on assistant replies are saved pending separate review,
not automatically approved as truth or training data. The repeated review checkbox and manual recall/decay simulation
tools are removed from this training view; automatic capture and graph behavior
remain in place.
The learner still needs 12 eligible judgments across six distinct prompts and
direct/unrelated anchors in its training and held-out partitions before an
update can be evaluated. Newest outcomes stay visible above learner details.

Start with the [beginner walkthrough](docs/FROM_SCRATCH.md): it explains what
we are building, why, what each test demonstrates, and how you can contribute.

`original_learning.py` implements word and phrase overlap, logistic scoring,
error measurement, and gradient descent directly with standard-library Python.
`memory_learner.py` connects those calculations to the transcript, review
records, and checkpoints. No NumPy, scikit-learn, pretrained weights, or hosted
model is used by the active core. It is a small relevance learner, not a
conversational LLM. The former latent representation has been removed, so this
milestone does not supply semantic embeddings or synonym understanding.

Run `python3 original_learning.py` for a safe, nonpersistent demo. Training
reduces error on tiny illustrative examples; those examples are not a
commercially approved dataset or an independent benchmark.

Search the graph, then choose **Direct answer**, **Related context**, or
**Unrelated** beside a result. The original learner uses targets 1.0, 0.5, and
0.0 respectively. Context is connected background, not an unrelated or false
statement. These are graded relevance targets for one scoring model, not three
class probabilities or verified semantic understanding. This stores a judgment,
not training approval. The API and CLI require the named `label` field;
ambiguous legacy boolean requests are rejected instead of silently converted.
Only user-authored memories and questions explicitly reviewed through the
`review` CLI command can train this first version. Approval stores hashes of
the source text; editing a memory invalidates its approval and old label.
Review is a user declaration, not a legal certificate. Existing helpful,
correction, and recall controls change graph salience, not training labels.
Historical and assistant text may remain searchable without being trained on.
Locally generated retrieval replies are excluded from the search corpus.

Training begins after at least 12 reviewed labels covering six distinct queries. Both
training and validation partitions must include direct-answer and unrelated
anchors. Context examples use their 0.5 targets in both training and evaluation
when present; collecting context feedback is necessary to learn that distinction.
Entire queries are held out, so the same query cannot leak into both partitions.
A candidate replaces the current ranker only if its validation log loss improves
by more than 0.001. Old reviewed examples are replayed with new reviewed feedback.
Repeated query/ID judgments replace their labels rather than multiplying votes.
This small evaluation gate is not proof of general intelligence; repeated reuse
of a validation set can overfit it, and future releases need separate benchmarks.

New memories are searchable but do not automatically become training data.
Initial ranking weights are specified heuristics until training is accepted.
Topic promotion still follows the separate rules below. Semantic representation
remains future work.

Inspect `/api/learner-status`, or run:
The memory visualizer centers AI memory and arranges other domains on a
non-overlapping orbit. Solid circular boundaries and SVG shading give depth
without a 3D engine. Region radius is `min(180, 48 + 6*sqrt(message_count) +
20*mean_capped_salience)` in graph units. Memory radius is
`2.5 + 3.5*sqrt(min(1, salience))`, plus `0.75` when consolidated.
Sizes recompute on redraw, so feedback, recall, and decay affect them.
These bounded visual sizes do not measure intelligence or approve training.

```bash
printf '{}' | python3 memory_learner.py status
printf '{"query":"memory graph"}' | python3 memory_learner.py search
```

`/api/chat` uses a local character n-gram generator after at least one source is
explicitly approved in the **Generation** tab. Before that, it retains the
identified-memory-excerpt fallback. If all candidates are empty, Chat shows the
memory fallback without trying to format a missing selection score.
The first generator predicts one character from the previous four; it does not
understand meaning, follow instructions,
or produce grounded answers. Its output is experimental and may be incoherent.
If prompt-matched text ends immediately or is only a few characters long, the
generator falls back to sampling from the corpus start rather than returning a
blank response.

Text explicitly added in the Generation tab and each new prompt submitted in
either in-app chat enter this separate corpus. Pressing Enter is the user's
choice to add that prompt for future training; there is no separate checkbox.
Historical conversations and retrieval approvals are not imported. Generated
replies are never added automatically. Each source stores its title,
provenance, content hash, and review record.
The title and additional provenance note are optional; source defaults to `Me`.
Clicking **Approve text and train** is the explicit approval action. That
declaration is not legal clearance. Removing a source rebuilds the model without
it. Generated outputs are saved to chat memory with a `local-generated` source
tag, but are excluded from both the generation corpus and relevance learner
representation. Response feedback stays separate from generator text training
and the memory relevance learner.

AI Chat provides **Relevant and train** and **Needs correction and train** on
generated replies. Clicking either approves that exact saved prompt/reply pair
for response-scoring training, not as a factual target or generator document.
The `original_response_scorer.py` prototype learns response-selection
weights from explicitly reviewed prompt/reply ratings. It measures lexical
overlap, query coverage, and word diversity, not factual correctness. It needs
12 pairs across six distinct prompts and both ratings in query-separated
training/validation partitions. New weights require improved validation loss;
the previous accepted state is retained. Chat generates four candidates and
selects using this scorer. Before global training, nearby reviewed prompt/reply
pairs provide an immediate bounded preference adjustment (at most 0.25).
Word repetition is penalized separately. No existing feedback is auto-imported.
Run `python3 -m unittest test_original_response_scorer` for its isolated checks.
Its compact progress strip refreshes generator generation, source count, and
training-character count after each successful prompt. Rating totals cover only
the currently displayed chat session; the saved feedback records remain durable.
**Compare generations** opens the existing fixed-prompt evaluation workspace.
AI Chat also displays durable response-scorer counts, two minimum-data progress
bars, and the latest query-separated validation result. These counts are not
accuracy percentages; both labels must also be present in each partition.
After **Needs correction**, enter an edited reply and click **Approve correction
and train**. This explicitly approves your edited text: the original reply stays
negative, the prompt/correction pair becomes a positive scorer example, and the
correction enters the character generator corpus with user-correction provenance.
Exact repeated saves do not duplicate examples or documents. The original chat
record is preserved. This teaches text patterns and selection, not factual truth;
global scorer weights still require validation. Ordinary ratings and fixed-prompt
evaluation edits do not approve generator text. Corrections need 4 to 2000
characters; use only reviewed material you are entitled to train on.
Ratings are associated with the response ID and content hash, and can be revised.
They adjust selection preferences, not the character generator's transition
counts. New prompts still train its text
patterns independently of these ratings.

The **Progress checks** section tracks fixed prompts and saves one deterministic
sample per prompt for each generator generation. Corpus additions and removals
automatically capture the next comparison. Keep/Edit/Reject ratings and edits
are stored separately for evaluation; they do not train on generated output.

The generated checkpoint is `.data/original-generator.json` and is git-ignored.
Its transition counts are rebuilt from the approved local corpus when a source
is added or removed. Topic promotion still follows the separate rules below.
Semantic understanding and grounded language generation remain future work.

The new checkpoint is `.data/learner-graded.json`; the previous accepted
version is `.data/learner-graded.previous.json`. Review records are in
`.data/learning-review.json`. The binary `.data/learner-original.json` is
preserved unchanged. On first use of the graded model, its judgments are carried
forward with no named label and marked as needing relabeling; its weights and
generation are not reused. Relabeling the same query/ID replaces that pending
entry. The still older library-based `.data/learner.json` is not loaded.
All are JSON and git-ignored. Stop the
server before manual rollback. The reviewed vocabulary is an inspection
inventory, not a learned semantic representation.

The graph shows the memory network separately from a responsive system-flow
diagram. Reviewed source text feeds the local generator; chat exchanges are
saved in shared memory; eligible memories and explicit labels inform the
relevance learner. Generated replies are stored but are not automatically used
to train either model, and retrieved memories do not yet condition generation.
The learner panel shows live generation, all five parameters, review/eligibility
counts, and the latest evaluation. The learner status polls every five seconds
and refreshes after feedback; generation zero does not mean trained weights.
See Lesson 8 in the
[walkthrough](docs/FROM_SCRATCH.md) for how memory growth differs from learning.

## Commercial Use Boundaries

**Current boundary:** the model's scoring and training calculations remain from
scratch. The user accepts third-party dependencies for the viewer/UI and may
consider other dependencies later, subject to review. The existing viewer does
not need replacement merely because it uses libraries. Public availability
alone does not establish commercial-use or redistribution rights; retain
required notices and review license terms. This permission does not authorize
replacing the original learning core with a pretrained or hosted model.
No compliance or exclusive ownership is certified by this repository.

This design does not depend on a pretrained-model vendor's usage terms. The
new application code and locally fitted weights are separate from its reusable
libraries. No open-source license has been added to the original project code;
the owner's distribution policy is still a decision to make.

The preserved legacy virtual environment contains third-party packages, but the
active learner does not import them. Do not redistribute that environment as
part of the original core or remove its license notices. Existing npm
dependencies and the chosen runtime require their own license review.

Training from scratch does not automatically confer ownership of training data.
Existing Copilot responses, pasted documents, personal data, employer-owned
material, and any later datasets must be reviewed for rights, consent, and
applicable service terms before a commercial release. Avoid entering secrets;
local capture and feedback files retain interaction text. This is an engineering
design, not a legal determination of exclusive ownership or permission to sell.

The app is still a local prototype. Its permissive CORS policy, lack of
authentication, and npm audit's currently reported eight vulnerabilities need
review before exposing it to untrusted users or shipping it commercially.

To add newly inferred text-match links to existing messages without appending a conversation turn or advancing decay:

```bash
python update_chat_memory.py --reindex-links
```

## Automatic Cluster Growth

Subject ownership follows the user's message. The assistant's reply inherits
that subject, so discussing the food region of the memory graph is not itself a
food memory. Adjacent exchanges in different subjects retain a `transition`
link; that link does not confer core membership. Only newly created message
nodes pulse automatically, not topic hubs.

Candidate discovery now includes subtopics inside existing domains, not just
unknown subjects. Broad parent words (such as memory, model, and graph) and
generic acknowledgments are excluded from discovery. Candidates use the existing
local TF-IDF similarity code; only meaningful user contributions supply votes.
Assistant replies inherit the user's membership but do not supply evidence.

Promotion requires at least five user contributions and mean pairwise similarity
of 0.30, plus either three recorded conversation IDs or sustained discussion:
eight distinct meaningful contributions within one recorded session, spanning at
least seven transcript positions. Repeated acknowledgments and memories without
session provenance cannot satisfy that single-session path.

Promoted subtopics get their own region and a hub connected to their parent
domain by a green dashed parent link. This relationship does not turn all parent
or neighboring memories into child-topic core memories. Adjacent subject changes
remain separate transition links. Every message retains `parent_domain` for
provenance. Promoted IDs and labels remain stable; pending labels may improve as
evidence changes. Hover over a candidate memory to inspect its contribution
count, distinct count, sessions, span, and cohesion. Candidates can lose promotion
if evidence no longer meets the thresholds, without deleting memories.

The graph's **Mark important** toggle records an explicit vote on a selected
user-authored memory in `memory_graph_state.json`. A candidate cluster displays
a heuristic estimate: 25% distinct-interaction evidence (capped at 8), 20%
session evidence (capped at 3), 25% cohesion, 15% helpful marks (capped at 3),
and 15% explicit importance marks (capped at 3). This estimate is not trained,
does not alter memory salience, and does not change promotion rules.

Explicit AI creation/development wording is classified under the AI-memory
parent. Unambiguous short follow-ups (for example "sounds good let's do it" or
"so where are we at now?") retain the latest subject in their own recorded
session. Explicit new subjects override that context, and context is never
borrowed from another session. Assistant wording does not determine the user's
subject. These placement rules do not grant training approval or promotion votes
to generic acknowledgments.

Copilot's user-level stop hook records conversation IDs through
`update_chat_memory.py --session-id ... --source copilot`. App chat uses a
browser-tab conversation ID retained across refreshes and tab switches.
The Chat tab and Memory Graph trainer each retain their own ID in that tab's
`sessionStorage`; Copilot uses the ID supplied to its stop hook. Separate tabs
and Copilot sessions count as separate conversations, while reloading a tab
retains its ID.
Historical messages without provenance do not invent distinct conversations
and cannot alone satisfy the three-session gate. These thresholds are an initial
heuristic, not semantic understanding or a trained learning model. Existing
named domains remain configured parents. Differently worded AI-development
discussion can still split into separate lexical candidates; no named category
is forced simply because the user or assistant mentions it.

Run `python3 -m unittest test_memory_clusters -v` to check membership,
promotion, stability, and transition rules. Regenerating the visualization
updates cluster evidence without appending an exchange or advancing decay:
`python3 generate_html_graph.py`. When the API is running, the graph reloads
when its polled snapshot contains newly added nodes; a file-only view without
the API needs a manual refresh.

The live graph also reloads when existing nodes are reclassified. Viewer
feedback saves merge mutable salience/history/feedback into the canonical graph
rather than replacing its domain metadata, new nodes, parent links, or capture
cycle. This prevents an older open tab from erasing the latest graph membership.

## Memory Architecture Policy

This project is evolving from a chat-source memory into a queryable memory layer and eventually into a multi-agent system that can act from memory independently of the chat that created it. Keep these rules in mind:

- Preserve **provenance** for every memory so future agents can trace where it came from.
- Separate **domains** and **namespaces** so unrelated topics stay visible but not blended.
- Treat **evaluation/meta** as its own area rather than as core content memory.
- Allow **cluster promotion** only when a topic has enough weight, repetition, or importance.
- Prefer **stable references** and summaries for durable knowledge; keep transient turns lightweight.
- Support **retrieval-first behavior** so future agents can query memory instead of depending on chat context.
- Keep the graph and transcript compatible with later **agent-to-memory** workflows.
