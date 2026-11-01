# Building Our Learning Core

## Goal And Boundary

We are building the learning calculations ourselves, rather than calling a
model service or using a machine-learning library. The first milestone is a
small relevance learner: given a question and a memory, estimate how useful
that memory is for answering the question.

This is not yet a language model, a reasoning engine, or a system that knows
whether a statement is true. It is a manageable first component whose behavior
we can inspect, test, and improve together.

The core uses Python's standard library. Third-party viewer/UI dependencies
such as React and Express are acceptable to the user when suitably licensed;
future dependencies can be considered separately. Python itself is also
third-party software. Public availability does not establish commercial rights.
Implementing our calculations does not certify that the whole product is
obligation-free or that AI-assisted code has exclusive ownership. Keep license
and provenance review separate from engineering progress.

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

Semantic representation, neural layers, and language generation are separate
later milestones. The suitably licensed viewer may remain in use. Topic-cluster
promotion still follows its existing rules; this learner does not yet reorganize
topics.

## Lesson 6: A Domain Can Grow Subtopics

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

Our present learner ranks memories; it does not predict new language tokens.
The response panel is therefore explicitly memory-based, not invented gibberish
presented as a new language model. A response rating or your correction is useful
future evidence and is stored separately with provenance, but it currently does
not change model weights. Retrieval judgments still follow the reviewed-example
and held-out-evaluation gates. A later response generator needs its own training
objective and evaluation rather than learning its own mistakes as facts.