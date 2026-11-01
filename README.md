# Push Bridge POC

A proof-of-concept PWA that sends a web push notification from a desktop browser to a phone browser, then lets the phone reply back to the same server state.

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
```

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

**Language generation is not implemented.** The response path selects a matched
stored assistant reply and marks it as retrieved, not generated. If none of the
retrieved candidates is an assistant reply, it reports that limitation rather
than echoing the user's question as an answer. Relevance feedback trains ranking
only; a from-scratch token-prediction/decoding component is a separate next
milestone. Old saved responses remain historical records; submit a new prompt
to use the corrected response selection.

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

The response area currently shows ranked memory excerpts, not a neural language
generator. Response useful/Needs correction and optional user corrections are
stored separately in `.data/response-feedback.json`, with response IDs and
content hashes. They remain unapproved and do not automatically train the
relevance model or future text generator. Grade the cited memories separately
to train retrieval. Model output must never become its own trusted target merely
because it was generated; reviewed human corrections and independent checks can
supply evidence for future response training.

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
`/api/chat` returns identified excerpts, not invented free-form answers. Topic
promotion still follows the separate rules below. Semantic representation and
language generation remain future milestones.

Inspect `/api/learner-status`, or run:

```bash
printf '{}' | python3 memory_learner.py status
printf '{"query":"memory graph"}' | python3 memory_learner.py search
python3 -m unittest test_original_learning test_memory_learner test_memory_clusters -v
```

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

The graph also displays a visual-only AI-creation project and Learner component.
Inspect learner shows live generation, all five parameters, review/eligibility
counts, and the latest evaluation. Directed system links are architectural
relationships, not memories or automatically promoted topics. Query-specific
amber relevance links are temporary overlays and are never stored as memory
connections. The learner status polls every five seconds and refreshes after
feedback; generation zero does not mean trained weights. See Lesson 8 in the
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
