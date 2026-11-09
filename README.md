# Push Bridge POC

A proof-of-concept PWA that sends a web push notification from a desktop browser to a phone browser, then lets the phone reply back to the same server state.

Ongoing work and discussion topics are tracked in the
[project backlog](docs/TASKS.md).
Application versions follow the [versioning policy](docs/VERSIONING.md);
the [versioned session log](docs/SESSION_LOG.md) records each checkpoint.
The memory graph displays its generated application version in a small
bottom-right badge. Regenerate the graph after a version bump.

## What it does

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

When a galaxy is promoted out of Under evaluation, its member memories and
associated replies move with it. Its parent tether to Under evaluation is
removed. Memory-to-memory links remain, including pairs whose other endpoint
is still under evaluation; those links do not confer galaxy membership.
Connections to other galaxies and original transcript provenance remain.
Parent-child galaxy relationships use solid green shell-to-shell connectors,
with hover labels identifying the child and parent. Cross-galaxy memory links
remain dashed. This styling does not change membership or learning weights.

The conversation memory prototype stores its transcript in `chat_memory.json` and renders it in `memory_graph.html`. The browser view now keeps its learning overlay in a backend graph-state file served from the local API, so helpful, correction, and weak-link feedback survives refreshes without depending on browser `localStorage`. The graph page also polls that backend snapshot, so it refreshes itself when the stored memory state changes. Use the search box to rank memories by query, click a node to open the selected-memory panel, hide that selected-memory section from the graph header if you want it out of the way, and hide the entire right menu with the button above the graph if you want the graph to use the full workspace. The feedback buttons teach the graph which memories should matter more. To add a conversation exchange and advance one persistent decay cycle:

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

The **Generation** tab now includes a separate, from-scratch character neural
network: four-character context, 8-value embeddings, a 16-unit tanh hidden
layer, softmax predictions, cross-entropy loss, hand-written backpropagation,
and stochastic gradient descent. It uses only Python's standard library.
Parameter count depends on the training-only vocabulary and is shown exactly.
This is a tiny neural language-model experiment, not an LLM or a replacement
for the existing character-count generator.

Start with **Run synthetic learning check**. This original repeating-pattern
fixture demonstrates weight updates, not generalization. For real experiments,
provide new training text, separate held-out text, a provenance/rights note,
and explicit approval for this neural use. Existing corpus approvals, memories,
and chat prompts are **not** automatically reused. Approval is a user
declaration, not legal clearance.

Each run starts from deterministic fresh weights and saves its own checkpoint.
The report shows trainable/changed parameters, steps, processed target tokens,
initial/final samples, and training/held-out loss at step 0 and every 100 steps.
Held-out text does not update weights or determine the vocabulary; unknown
characters are reported. Do not select/tune against held-out text repeatedly
and then describe it as an untouched final test.

Full submitted text, provenance, hashes, weights, and report are saved locally
under `.data/neural-experiments/<run_id>.json`; `latest.json` holds the latest
report. These ignored files can contain sensitive text. This baseline does not
apply or soften Layer 0/1 rules. Controlled layer-assisted neural comparisons
are a subsequent milestone; the existing four-mode chat lab is unchanged.

```bash
echo '{"synthetic":true}' | python3 neural_experiment.py train
python3 neural_experiment.py status
python3 -m unittest -q test_neural_experiment
```

### Seed rules and existing generator

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
content hashes. They remain unapproved and do not automatically train either
model. Grade cited memories separately to train retrieval. Generated output is
never added to the generator corpus automatically.

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
Inspect learner shows live generation, all five parameters, review/eligibility
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
