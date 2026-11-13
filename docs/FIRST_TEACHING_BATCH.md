# First Teaching Batch: Draft For Review

## What This Is

These are proposed relevance judgments drawn from your existing messages. The
source statements are yours; the questions, labels, and explanations below were
proposed by the assistant. They have not been approved for training.

The private machine-readable draft is `.data/first-teaching-batch.json`. It has
six distinct questions and twelve candidate entries, two currently unlabeled.
Each source excerpt was checked
against `chat_memory.json`. No review records, training labels, or model weights
were changed while preparing it.

## How To Judge A Pair

The learner now accepts **Direct answer**, **Related context**, and **Unrelated**.
This packet preserves the original proposed binary entries for review; it has
not been automatically converted or approved. Every accepted entry needs an
explicit named label before submission. The disputed coaching/hardware entries
are candidates for context, not assumed unrelated examples. The new UI option
does not approve this draft or its sources.

Ask: **Does this memory help answer this particular question?**

Relevant is not the same as true, and not relevant is not the same as false.
A memory can be related to the project without directly answering a narrow
question. The user clarified that documentation/coaching and GPU requirements
are connected through tracking and creating the AI from scratch. The draft's
binary wording was too blunt for that distinction. Two entries are now
unlabeled; they must not be trained as unrelated. Other proposed negatives
still need review and are not assertions that two subjects share no connection.

## Proposed Pairs

### 1. GPU Lanes

Question: **Does each GPU need standalone full x16 rather than shared lanes?**

- Relevant, `You 47`: "no shared lane need full 16x on all gpus"
- Unlabeled, project-related, `You 76`: "good docs on what we're doing and why and coach me as we go"

The user clarified the requirement as standalone full x16, not shared lanes.
The first directly addresses that requirement. The second supports tracking and
creating the AI project; it is not a GPU-lane specification, but it is not
unrelated to the work. Its binary training label is disputed and has been removed.

### 2. Hosted AI

Question: **Should our AI connect to an outside hosted model?**

- Relevant, `You 70`: "no connecting with an outside model being hosted"
- Not relevant, `You 48`: "pricing for wrx90 boards"

The first states the restriction. Pricing does not answer it.

### 3. Viewer Dependencies

Question: **Are dependencies acceptable for the viewer?**

- Relevant, `You 77`: "I think view depenancies are okay"
- Not relevant, `You 47`: "no shared lane need full 16x on all gpus"

The first expresses your viewer preference. It is not a certificate that any
particular dependency's license is suitable.

### 4. Organic Growth

Question: **Should new memory topics grow organically from our interactions?**

- Relevant, `You 68`: "an organic model the grows on it's own based on our interactions"
- Not relevant, `You 48`: "pricing for wrx90 boards"

The first states the intended direction of growth. The second does not.

### 5. Coaching

Question: **Do I want documentation and coaching to understand the work?**

- Relevant, `You 76`: "coach me as we go so I can understand what's going on and help"
- Unlabeled, project-related, `You 47`: "no shared lane need full 16x on all gpus"

The first establishes the coaching preference. The second concerns hardware
for the same project. It does not directly answer the coaching question, but
we have removed the assumed unrelated label pending review.

### 6. Starting Small

Question: **Should we start with a viable miniature model built from scratch?**

- Relevant, `You 71`: "from scratch sounds good, but we need a viable mini enough for it to evolve"
- Not relevant, `You 48`: "pricing for wrx90 boards"

The first specifies the desired starting point. The second is not an answer
about the model's design.

## Your Review

Check that the questions preserve your meaning, the selected memories really
help or do not help, and the source text is suitable for training. You can
reject or change any pair. Other memories may also be relevant; these labels
do not claim that each positive is the only valid answer.

Once explicitly approved, the source memories and questions can be reviewed
through the existing review gate. Unlabeled entries must first be resolved or
replaced; the batch is not currently twelve binary judgments. Approved judgments can be submitted as a
batch. Entire questions are held out from training. A candidate is kept only
if it beats the current scoring rule on those held-out questions; otherwise,
the current model stays unchanged.

Approval is a user declaration, not legal clearance. These assistant-proposed
questions and labels also need review; merely reaching twelve judgments is
not approval. Never automatically import this draft into training.

## Review Decisions

Recorded from the user's review. Not yet submitted to training.

- GPU Lanes: approved. `You 47` relevant. `You 76` labeled Unrelated by the user.
- Hosted AI: rejected.
- Viewer Dependencies: rejected.
- Organic Growth: approved.
- Coaching: approved. `You 47` labeled Unrelated by the user.
- Starting Small: approved.

Approved questions: 4 (GPU Lanes, Organic Growth, Coaching, Starting Small).
Rejected: 2 (Hosted AI, Viewer Dependencies).

Submitted labels: GPU Lanes `You 47` related context, `You 76` unrelated;
Organic Growth `You 68` direct answer, `You 48` unrelated; Coaching `You 76`
direct answer, `You 47` unrelated; Starting Small `You 71` direct answer,
`You 48` unrelated. All eight were recorded in the review gate.

Training outcome: the held-out candidate was **rejected** (validation loss
0.7348 vs. baseline 0.6615). The active model is unchanged at generation 0.

## Phrasing Variant Submission Outcome

Submitted through the review gate on the user's instruction. The eight variants
were reviewed with their base memories' labels, one pair at a time. The gate
re-ran after each pair, and this is not the batch submission the draft describes.

- Intermediate runs: three candidates were accepted during submission, then the
  later ones were rejected or superseded.
- Final state: accepted at generation 3. Validation loss 0.5582 against a
  baseline of 0.5596, an improvement of 0.0015 (the gate needs more than 0.001).
- Training examples: 45. Validation examples: 15 across 7 held-out queries.
- Feedback entries: 114.
- Rolled back on the user's instruction to generation 0: initial weights
  `[3.0, 0.5, 1.0, 0.5]`, intercept -2.0. All 114 judgments were kept. The
  generation-3 state is backed up privately in `.data/learner-graded.gen3-backup.json`.

## Phrasing Variants (User Verified)

Assistant-drafted paraphrases of the four approved questions. The user verified
all eight. Each keeps its base question's meaning and label. They were submitted
to the review gate; see the outcome above.

- GPU Lanes (base: standalone full x16 rather than shared lanes)
  - "Do the GPUs each need their own full x16 link, not shared lanes?"
  - "Will every GPU need a dedicated x16 slot instead of splitting lanes?"
- Organic Growth (base: grow organically from our interactions)
  - "Should memory topics emerge over time from what we talk about, rather than being fixed up front?"
  - "Do we want new memory topics to form on their own from our conversations?"
- Coaching (base: documentation and coaching to understand the work)
  - "Would you like the docs and coaching so you can follow what the work is?"
  - "Should I explain the work as we go so you understand it?"
- Starting Small (base: viable miniature model built from scratch)
  - "Should we begin with a small working model that we build from scratch?"
  - "Is a viable, miniature from-scratch model the right place to start?"

For how the calculations work, see [Building Our Learning Core](FROM_SCRATCH.md).