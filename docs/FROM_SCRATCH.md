# Building Our Learning Core

## Goal And Boundary

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