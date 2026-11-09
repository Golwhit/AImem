# Building Our Learning Core

## Goal And Boundary

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

## Neural Weight Milestone

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
runs reproducible; every run starts fresh rather than silently continuing a
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

In AI Chat, choose **Relevant and train** or **Needs correction and train** to
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