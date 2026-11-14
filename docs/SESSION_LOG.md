# Versioned Session Log

Newest application checkpoint first. Entries map delivered versions to the
session that produced them; historical conversation memory remains separate.
See [versioning policy](VERSIONING.md) for bump and release rules.

## 0.7.0 - 2026-11-13

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Since the 0.6.0 checkpoint, refined the memory graph's galaxy presentation
  and relation links, recorded user-reviewed teaching material, and added
  guarded activation and continuation from active conversation weights.
  These features were committed previously, including `99ced75` and
  `9acbced`; this version synchronizes the previously unversioned work.
- Added the [remote training handoff](REMOTE_TRAINING_HANDOFF.md), documenting
  the observed round-24 checkpoint, private transfer boundaries, replay and
  rollback requirements, and GPU and partial-resume work that remains to be
  designed. Updated the backlog's stale active-selection summary.
- Validation: `npm run version:check`, `npm run test:version`, and
  `npm run build` passed. The 12 targeted active-training,
  permanent-training, and correction tests passed before this documentation
  checkpoint. Checkpoint, report, fixture, and immediate rollback hashes
  were verified locally; no transfer or GPU test was run.

## 0.6.0 - 2026-11-10

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- User clarified immediate learning on Relevant and explicitly submitted
  corrections, never unmarked replies. Added `neural_correction.py`:
  validate saved conversation-model pair, hash exact approved content,
  preserve vocabulary IDs while appending approved symbols, train actual
  weights with approved-only rehearsal, verify exact generated answer and
  full replay before permanent/active promotion. Latest same-prompt target
  supersedes conflicts and joins future rounds. Bounded10CPU/2000updates;
  unsupported or unlearned answers fail explicitly without promotion.
  UI now uses Train correction now and immediate Relevant training, with
  progress/success/errors; no second review checkbox. Backend feedback
  verifies checkpoint/pair evidence before marking training approval.
  Bare Needs correction and unmarked inference do not update weights.
  Previous checkpoints and original control preserved; no lookup answers
  or generalization/retention guarantee. Connected memory remains intact.
  Validation:40affected tests passed, then6focused tests passed including
  a new optimizer-input check proving a correction supersedes the
  conflicting base target in the next permanent round (41distinct tests).
  Production build, Node syntax and diff checks passed. Pylance's syntax
  request timed out; Python compilation and runtime tests succeeded.
  Restarted local production API and verified browser Relevant:
  already-approved teaching prompt "Mira has a red cup. What color is the cup?"
  / "Red." learned through50real updates, .456updateCPU seconds,
  exact replay and saved/active checkpoint agreement. Unmarked generation
  left the initial checkpoint unchanged; next generation returned "Red.".
  Initial feedback checkpoint:
  `4b26d71d3ecb599f43a00352092a5cededa224690850286ef2cf710debfa9d74`.
  Subsequent explicit user feedback may advance this pointer; the log
  records the test, not a promise that this remains the latest checkpoint.
- User approved permanent learning, five update-CPU minutes and chat
  activation. Added bounded resumable `neural_training.py`: continue actual
  full64 fresh seed17 checkpoint, preserve parent/inputs/source/evidence,
  exact replay before advancing checksum-pinned progress. Round
  `57f452944e2c4767b5586c8320bfbad7`: 57,100 additional updates,
  300.28CPU seconds,64/64 teaching,22/32 development (16/16 familiar
  combinations,6/16 changed wording),10/16 pairs. Entire weights,
  teaching milestones and evaluation continuations replay exactly.
  Active checkpointSHA
  `c7dec17d83b210de79293e78d814056ef03dcc9e31c7579932c38a0413e878d9`.
  Added conversation restore/formatting/first-line decoding, correct API/UI/
  graph labels and training-round provenance; preserve old New Base and
  original character control. Activation is idempotent; rollback validates
  target before switching and leaves training progress intact. Optional
  activate-after promotes verified future rounds without repeating review
  for identical approved bytes. No automatic chat/self-training, new data
  or dependency changes. Thirty-six focused tests, compilation/Node syntax,
  production build and live API/browser capture checks pass. Production
  server restarted. Saved weights resume without retraining from scratch;
  changed wording remains weak, no general-intelligence/retention claim.
- Approved larger original teaching meal, exact SHA256
  `ce196cbef192c0f497afca080c25881136f21d748229061addb3152e7ce73461`.
  Run `ee7f3c1be14f477a82e2db03ae72ff71`: 64 paired teaching exchanges,
  eight skills, separate 32-case combination/wording evaluation. Compare
  balanced 16-example subset with full 64 under identical initial
  weights/vocabulary, 15,508 parameters, fresh/normalized designs,
  .1 rate, three seeds, 24 update-CPU seconds per copy. No observed
  teaching token-input collisions (105/420 targets); all 32 first-context
  pairs distinguish their facts. Fresh/16 fits all16 each seed; fresh/64
  fits59,58,59 of64. Test fresh/16 20,20,19 of32; fresh/64 21,18,15.
  Normalized test16:14,15,14; test64:7,9,12. Larger meal does not improve
  overall evaluation at equal CPU. Familiar phrasing far stronger than
  changed wording; six changed-wording identities absent from teaching.
  All12 endpoints saved before test access; no test-guided tuning or
  activation. Thirty-four targeted tests and Python compilation pass.
  Original synthetic provenance, exact approval/source archives, full
  checkpoints/continuations and paired scores retained. No dependencies
  or changes to active chat; memory remains connected. All12 full final
  states, evaluation continuations and non-timing teaching milestones
  independently replay exactly; all seven active/control hashes unchanged.
- Approved initialization/normalization diagnostic `29395ea2dc794a499c608b489959221f`:
  frozen 64-slot context, 15,308 parameters and eight approved teaching
  exchanges; inherited, fresh scale-aware reset and fixed pre-tanh
  normalization. Three seeds, .01/.1, 12 update-CPU seconds per rate,
  teaching checks at 3/6/12 seconds. All nine rate selections saved before
  test access. Inherited selects .01, teaching 0/8 in all seeds; fresh
  selects .1, teaching 4/8,7/8,5/8; normalized selects .1, teaching
  3/8,4/8,4/8. Both interventions test 1/10 in each seed, same blue case;
  inherited zero. Mean selected losses 1.807/.216/.492; CPU
  12.29/12.26/12.36s. Final hidden saturation 97.66/30.15/.17%.
  Better fitting, not useful conversation or a 30% qualified win.
  Thirty targeted tests pass, including finite-difference normalization
  gradients and unchanged ordinary tanh equations; Python compilation passes.
  Pylance syntax calls timed out. No active model, data or dependency changes;
  no retained-knowledge claim after reset/normalization. All 18 full trained
  states and non-timing teaching milestones replay exactly, as do all nine
  selected test assessments. All seven working/control JSON hashes unchanged.
  Exact sources and raw/derived evidence retained; findings captured in memory.
- Approved longer-context ablation `1fe67c6c59064ddab1e088843cbe1c60`:
  reproduce short control, add 21 teaching-only word identities, then add
  60 older word/gap slots to the existing four local features. Copy actual
  parent weights; older connections start zero. Same vocabulary/masks and
  initial distributions in short/long identity arms; older connections learn.
  At .1/5,000 updates: short/identity/long teaching exact 1/8,1/8,0/8;
  test 1/10,1/10,0/10, CPU 8.67/9.25/25.19s. Context now distinguishes
  red/blue facts, but long hidden values saturate on all teaching targets.
  Separate .03/.01 long follow-up selects .01 using teaching loss only,
  then tests once: 0/10 exact, 1/10 normalized, 4/10 terminated.
  No useful conversation improvement. All five weight sets and non-timing
  metrics independently replay exactly; short control matches prior trial.
  Nineteen targeted tests pass; editor diagnostics clear. Historical reports,
  active chat and original comparison unchanged; memory stays connected.
- User approved eight-example conversation teaching trial, content-hash pinned
  separately from preserved draft. Run `576f66ae430b4f0088fde6af5fc87e76`
  continues isolated copies of actual active New Base at .1/.03/.01 for
  5,000 answer-only updates each; shared draw schedule, no test-guided tuning.
  Initial missing straight-apostrophe target failed before updates; reuse
  existing alphabet expansion to append that training-only character while
  preserving learned weights/IDs (7,103 experimental parameters). No new
  word vocabulary or architecture; role labels ordinary text.
  Loss 3.733 -> .600/.761/1.121, exact teaching 1/8,0/8,0/8; separate
  test 1/10,0/10,0/10. Replies conflate changed facts; role suffix consumes
  already limited context. No useful conversational understanding shown.
  Independently replayed all three full states and non-timing predictions/
  milestone metrics exactly. Active/control unchanged, memory connected,
  no experimental activation. Full protocol and raw answers preserved.
- Added offline working-base stability/representation diagnostic and an
  eight-example conversation teaching draft with ten proposed tests, explicitly
  pending review and never trained. Run `2d557e54d04543efa9b8dc0fe2220d56`
  uses two small approved Acts6-12 training snippets, fixed full-training
  vocabulary, three seeds, rates .1/.03/.01, 1,500 first-sample and 500
  second-sample updates. Mean first-sample token loss 4.324 -> 2.589/3.098/3.386;
  after second-sample training, first loss changes +.354/-.028/-.086.
  Learning occurs but stable near-perfect fitting is not established.
  Exact feature/mask audit gives 60.67% empirical full-training token ceiling;
  this is not a word/intelligence bound. Red/blue draft fact questions have
  identical current inputs, demonstrating missing fact-bearing context.
  Independently replayed all nine final weight sets and every non-timing
  checkpoint/continuation metric exactly. Active/control hashes unchanged;
  memory capture remains connected. No layers, activation or draft-data training.
- Retired L0/L1/L2 from the working model at the user's request. Activated
  layer-free `new` checkpoint from `10255643d34e41749c6bcc1f68ee8b5c`,
  seed 17/four passes, 7,078 parameters. Copied and checksum-pinned its
  paired 2,103-parameter original character checkpoint as comparison control;
  added `generate-original`. Preserve former selection, layered weights,
  datasets and all reports. Explicit provenance: current training is Acts
  6-12, not inherited whole-Acts/Corinthians training. Working training
  defaults to original/new only, without bank/dictionary or layer setup;
  historical five-arm mode now requires `--include-retired-layers`.
  Live chat, viewer and graph label new replies as layer-free; original
  teaching lab explicitly distinguished from the working new base.
  Greedy repeatable generation, review-only prompts/ratings, explicit
  corruption errors and no automatic experiment activation retained.
  Validation: 11 targeted Python tests including isolated HTTP, production
  build/version tests, server/Python syntax and whitespace checks passed.
  Live API and browser show the layer-free selection and pinned original;
  legacy layer comparison is off by default. Direct new/control generation
  matches saved states without changing weights. Memory graph refreshed.
- Full-generation five-arm run `10255643d34e41749c6bcc1f68ee8b5c`: exact
  archived Acts 6-12/13-14, all 256 probes, three seeds, 1/2/4 source passes
  at .1. Original context-4 character MLP versus a new causal word/gap base
  with exact surface outputs and character fallback, then existing L0/L01/L012
  extraction/placement. Queue word payloads characterwise so layers can act
  inside them; extracted spans bypass neural targets. Same new-arm weights
  and vocabulary; no evaluation vocabulary/target filtering, no activation.
  Four-pass accuracy 1.95/7.16/6.51/3.52/5.86%, setup+training CPU
  64.07/165.93/165.85/166.68/107.15s. New has 7,078 parameters versus 2,103
  original: quality gain is not a matched-capacity punctuation-only result.
  L012 one-pass 5.99%/30.96s versus original's best first-two-pass mean
  2.86% at similar observed cost is promising, not an all-seed win.
  Report full curves/earliest original crossings, not speed versus a
  collapsed endpoint. New fallback judgments at four passes: 0/348.
  Add explicit offline restoration/derived analysis without overwriting raw
  reports/checkpoints. 27 focused tests pass, including exact miniature
  training replay, queued placement, causal features and failed-state checks.
  Restored all 60 initial/trained states: all 15,360 full continuations and
  their word/accuracy/edit metrics match exactly. Raw report/source hashes
  and active chat selection/weight hashes verified. Inference is slower,
  not faster: 2.77/8.18/9.44/10.14/14.03s CPU per 256 restored probes.
- Larger word-context run `0baa5fd9d2384a3784538b85703bc371` reuses exact
  archived Acts 6-12/13-14 passages (30,905/9,730 characters), verifying
  saved baseline hashes and inputs before training. Matched word-task
  controls rerun, not compared numerically to prior character generation.
  Three seeds/5,000 eligible word targets at .03, 4,753 parameters each,
  263/512 known evaluation positions. Accuracy raw4/words/punctuation
  18.38/19.01/20.91%, loss 3.39667/3.44055/3.30765. Punctuation better
  against both controls in every seed; modest gain at comparable compute,
  not 2x or a generalization result. All nine weights independently replay
  exactly; evaluations match. Five focused tests passed. Chat unchanged.
- Brief word/punctuation screening with one matched 4,153-parameter MLP per
  arm: raw4, two prior words, words plus punctuation-gap signals. Common
  training-only 64-word vocabulary, identical initialization/draws/rate .03,
  three seeds/1,500 word targets. Initial all-target run
  `f5a091dcf6b043838ae6f3b6a2c0544b` collapsed to OTHER (zero known words).
  Preserved it and narrowed explicitly to closed known-word targets in
  `7b65fe2ad4e4493c92c19ca38d8dad78`: 248/512 validation positions scored.
  Accuracy 12.63/13.98/15.46%, loss 3.76434/3.75880/3.74443, CPU
  1.476/1.471/1.482s. Punctuation helps loss versus words-only in all seeds,
  versus raw in two of three. Small exploratory signal, not 2x or grammar
  improvement; different task from prior character tests. Four tests passed;
  all nine final follow-up weights replay exactly and evaluations match.
  Original punctuation/case/whitespace retained in source; internal word
  apostrophes/hyphens preserved. Active chat/visual lab unchanged.
- Triple 3+5+7 versus context-4 run `3bec0fc1df4f46268b80d7279ead98df`:
  three seeds, same 1,500 target draws/rate .03 and reused Acts 2 checks.
  Learned global softmax blend; exact responsibility/gate gradients.
  Accuracy 20.64/17.51%, loss 3.17557/3.06480 (triple worse in every seed),
  training CPU 3.118/.726s. More parameters/compute, no demonstrated overall
  benefit. Four focused tests passed; all six final models independently
  replayed weight-for-weight and both split metrics reproduced. Chat unchanged.
- Requested quick mixed 5+12 versus context-4 run
  `66afdebcd2824735b2a44bd9aed5c233`: three seeds, same 1,500 draws/rate .03,
  reused Acts 2 validation. Accuracy 19.27% mixture / 17.51% baseline;
  loss 3.13958 / 3.06480, mixture worse in all seeds. CPU 2.601/.716s.
  Added explicit context-pair and pair-only runner options while preserving
  original defaults/results. All six restored evaluations match; baseline
  exactly reproduces prior run. Three targeted tests passed; chat unchanged.
- Added an isolated quick multi-context experiment: fresh context-4,
  context-16, jointly trained 4+16 global probability mixture, and an
  approximately parameter-matched context-31 control. Original core,
  teaching lab, active chat and historical latest-report pointers unchanged.
  Exact mixture-loss branch responsibilities and sigmoid-gate gradients
  checked numerically. Three seeds, 1,500 same-target draws per arm at .03;
  Acts 1 training / reused Acts 2 validation, 512 character checks per split.
  Run `d9c11456613f4c0aaa3e3685fe1026ec`: validation loss means
  3.06480/3.08575/3.13436/3.11671; character accuracy
  17.51/16.21/19.53/14.00%. Mixture loss worse than both smaller models in all
  three seeds; roughly 4x context-4 training CPU, no demonstrated advantage.
  Thirteen regression tests passed. Restored all 12 final checkpoint metrics
  exactly; source/code hashes checked. Not a new untouched generalization test.
- Added a separate Base Model Lab tab (`/?view=base-lab`) focused only on
  the shared neural base, without L0/L1 or active-chat training. Eleven
  stages pair real numeric calculations/diagrams with compact sketches and
  actual runtime-inspected Python annotated for a junior engineer, beginning
  at source line 1. Inspect initialization, weighted sums, every gradient and
  before/after weight, target probabilities/loss and a whole-string loss curve.
  Disposable deterministic character-by-character updates, reset/custom text
  and explicit bounds/errors; no checkpoint/transcript writes. Core replay,
  source equality and numerical-gradient checks plus isolated HTTP passed.
  Browser verified all stages, navigation, another update, reset, invalid
  input, product inspection and narrow layout; live file hashes unchanged.
  Final validation: 12 targeted Python tests, production build/version tests,
  server syntax and whitespace checks passed; Pylance reports no diagnostics
  for the teaching trace/tests. Browser left at the first construction step.
- Added explicit `--all-acts` matched learning mode: validate all staged
  chapter/rights hashes, train chapters 1-28, and label all scoring in-sample.
  Require 20 distinct L1 occurrence checks before training; persist focused
  scores separately from merged general probes. Prior test chapters are
  training material only in this newly authorized mode.
- Added `--matched-reinforcement`: full-character and corrected remainder
  arms restore identical fresh pretraining checkpoints, then both replay
  each recognized literal once per occurrence. Source-order ordinary updates
  in both arms align refresh timing; costs and update counts saved separately.
- Added `--atomic-l01` corrected contextual-token comparison: base, remainder
  L01 and atomic L01. Typed references are trainable targets and context
  positions with fixed payload decoding; no pretraining, injection or separate
  placement counts in atomic. Persist atomic mode and usage metrics.
- Added fresh L0/L1-only pretraining checkpoints and opt-in
  `--pretrain-l01 --inject-l01` comparison. Protected pooled hidden-state
  vectors condition the active remainder base on the latest reference,
  without spelling replay, vector gradients or weight merging. Defaults
  remain unconditioned; prior comparisons remain intact.
- Corrected experimental L1 via `--unique-l1`: shortest unique partial keys
  against the expanded inventory plus training-word vetoes replace whole-word
  eligibility. Extract the complete exact payload; no full-word fallback,
  frequency gate or L2. Historical whole-word controls/results remain intact.
- Added `--savings-l1` five-arm intermediate-coverage comparison, preserving
  the old zero-selection purity trial. Select ceil(25%/50%) of training-only
  repeated L1 forms ranked by potential savings, with lexical ties. Save
  selections before training; no L2 or changes to live training.
- Added `--l2-only` candidate ablation: baseline versus standalone L2 + separate
  base, disabling all L0/L1 extraction. L2 discovers existing repeated-block
  and hidden-backed complete candidates directly from training. No policy
  changes to historical comparisons or live training.
- Added fixed training-only selective-L1 gate and `--selective-l1` cascade
  ablation: base/L0/selective-L01/full-L01. Preserve every rejected candidate
  in remainder training and save context evidence/thresholds before updates.
  Selection does not promote candidates or change Layer 0.
- Added isolated four-arm `--candidate-layers` small-run comparison: base,
  L0 + base, L01 + base and L012 + base. Base remains separate. Layer 0
  constants unchanged; Layer 1 indexes existing complete-form candidates;
  Layer 2 discovers repeated alphabetic 3/4-character blocks and checks
  hidden-backed complete forms. Typed lossless references preserve offsets.
  Separate placement learns at each stream event; no old generation exclusions
  or delimiter guard. No permanent admissions or changes to live training.
- Added reviewed `--corpus` JSON input for the separated experiment, preserving
  passages, authorship/rights declarations and file/content hashes before
  training. Reject invalid corpora and conflicting Acts large mode. Added
  AI-authored everyday workshop fixture and corpus-aware Generation labeling.
  Existing only-pattern bank and all learning policies stay fixed.
- Added `--large` to the separated-index experiment: complete Acts 6-12
  training / Acts 13-14 evaluation, five paired seeds, 256 common probes,
  unchanged extraction/placement policy and 1/2/4-pass milestones.
- Added isolated `neural_index_pipeline.py`: extract entire onlies and their
  positions, train the unchanged character base only on the concatenated
  remainder, and train a separate backoff context-count placement index.
  Coordinated generation emits either an indexed pattern or a base character;
  no indexed outputs or embeddings are added to the base. Both component
  costs, snapshots, examples hashes and new Generation report are recorded.
- Added explicit `--indexed-onlies` test mode for the clarified atomic-reference
  representation. Reuses existing UnitCodec; no internal spelling targets,
  one context position and one selection per indexed pattern. Saved token-ID
  evidence and fresh matched-update samples; live skip-spelling mode unchanged.
- Added a three-arm comparison: baseline, recognition with skipped spelling,
  and baseline plus the same rules while retaining every baseline update.
  `--add-combined RUN_ID` restores exact source weights, saves a separate
  protocol/report, and freshly measures equal-update samples for all arms.
  Combined training timings are explicitly inherited; no retraining or
  training-speed claim. Generation now displays the third arm and rule usage.
- Expanded the recognition benchmark with `--large`: complete Acts 6-12
  training / Acts 13-14 evaluation, five paired seeds, 256 common word probes,
  2,048 shared all-character probes, and a 120,000-update matched budget.
- Persisted all milestone weights; UI now supports the selected budget/seed
  count, common character loss, and separate equal-update generation samples.
- Corrected the active path to the user's recognition-only method:
  `neural_character_onlies.py` skips recognized interior spelling updates with
  original character contexts, no new output classes, and rule-supplied suffixes
  only for prefixes unique in the declared possible set. Legacy word-token
  checkpoints and earlier negative experiment remain preserved.
- Added the paired `neural_unit_benchmark.py` development runner: fixed
  three-seed protocol, equal-pass and equal-update budgets, common next-word
  accuracy/edit-distance metrics, measured CPU/wall time and estimated work.
- Reused the runtime generation probability/boundary policy for evaluation.
  Preserved frozen skeleton experiment code and locked result.
- Added `unit_benchmark` status reporting, Generation comparison panel,
  saved per-seed weights, source hashes and generated samples.

### Evidence and limitations

- User approved continuing the paired active models on complete 1 and 2
  Corinthians for four additional passes and activating the final layered arm.
  Staged 29 WEB chapters/694 verses, 80,757 characters, with raw/extracted
  text, publisher rights notice and content hashes. New continuation runner
  retains old weights, tokens, typed references and placement counts;
  revalidates/extends L1 with the same inventory and prior/new words.
  Identical occurrence refreshes in both arms; no Acts replay or repeated
  initial pretraining. One paired seed (17), not a new multi-seed claim.
  Saves before/after Corinthians learning and Acts retention checks, explicitly
  in-sample. Prior chat checkpoint remains selected until completion.
  24 targeted tests and production build/version checks passed; tiny-fixture
  checkpoints/indices/evaluations independently replayed exactly.
  Completed `1f7ed40a7b614dbba3c612a572782ca1` and activated layered seed 17,
  four additional passes. L1 124→176; 491 L0/185 L1 occurrences, 77,792
  remainder characters. Both arms refresh 11,860 characters; ordinary updates
  323,028 baseline / 311,168 layered; continuation CPU 184.740/182.646s.
  Fixed 256-probe Corinthians accuracy baseline .78→1.95%, layered 1.17→1.17%;
  Acts retention baseline .39→3.12%, layered 1.17→.39%. Layered did not improve
  new-book accuracy and lost measured retention, not a smarter-model claim.
  Restored all six checkpoint evaluations/retention checks exactly; verified
  source/code/unchanged-parent hashes and live API generation/provenance.
  Active checkpoint SHA256
  `5baa5edf76d6308de27346c6611ba6cd09f28f9a356102fe0b5cbb3972d7f343`.
  Browser shows cumulative source history and additional-pass semantics.
- Activated whole-Acts `layers_01`, seed 17, four passes for app chat. First
  protocol seed chosen deterministically, not based on its focused score.
  `neural_chat.py` validates/restores final neural weights and CandidateIndex,
  copies a checksum-pinned checkpoint, and persists the selection separately
  from latest experiment reports. SHA256
  `b142bd910de91f198c9c1238aeba7750024af70ad53ff2c6a3377fa4a20e7a20`.
  Chat records model/run/checkpoint provenance with each assistant memory;
  no automatic neural or legacy generator training occurs for these prompts.
  Legacy generator and optional lab remain separate and preserved. Active
  checkpoint failures return errors, not old-model output. React/graph label
  neural continuations experimental, not grounded answers or scripture.
  Live API text exactly matched direct pinned sampling; saved reply provenance
  and unchanged checkpoint bytes verified. 25 targeted tests passed, including
  HTTP routing with a ready legacy generator and optional lab. Broader
  learner/cluster run: 32/33 passed; an existing test references `result`
  before assignment (`test_context_feedback_is_explicit_and_replaces_previous_judgment`).
- Whole Acts run `59fbb1a0a36144e588ac1faae14d9654`: all 28 chapters,
  124,762 characters, 124 L1 forms, 264 L1 occurrences. Exactly 20 distinct
  focused target offsets spanning 18 words are included in 276 merged checks.
  Four-pass base/indexed in-sample accuracy .12/.48%, edit .86515/.86475;
  focused base 0/20 each seed, indexed 1/20, 0/20, 0/20. Poor fitting, not a
  meaningful intelligence or generalization gain. Mean total training CPU
  407.265/401.318s includes identical 115.809s pretraining and ~7.89s replay.
  Setup 5.379s separately. Whole-book trained checkpoints persist; not loaded
  automatically into the live generator. 51 targeted tests and build passed.
  Verified all 18 restored evaluations, focused target counts, spelling recall,
  18 generated samples, identical pretrained starts, replay schedules/costs,
  protocol/API and executed/frozen source hashes. Syntax, diagnostics, caller
  compatibility and version checks passed. Browser explicitly shows training
  overlap and the 20-opportunity support; graph refreshed.
- Matched run `02a74582812e4a728d886b19a80d2268`: 19 refreshes / 67
  extra character updates per passage pass per arm, same 6,200-update
  pretraining checkpoint per seed. Four-pass base/indexed accuracy 2.78/10.42%,
  edit .842/.801, total CPU 5.785/5.698s including 2.479s pretraining and
  .110s replay. Source order differs from old shuffled experiments; base wins
  at two passes. Mean final spelling recall 23.7/34.4%, not permanent retention.
  49 targeted tests, build/version/syntax checks passed. Independently replayed
  every update to reproduce all 18 checkpoint weights, evaluations, recall
  and 18 samples; verified matched starting weights, refresh schedules, cost
  accounting, protocol/API and code hashes. Browser shows both warm arms
  and replay costs; graph refreshed.
- Atomic run `b08a426d36bb41e988ab8a2ebac12d84`: three arms, three seeds,
  fixed workshop prose. Four-pass accuracy base/remainder/atomic
  5.56/15.97/5.56%, edit .832/.727/.788, CPU 3.115/3.057/3.432s.
  Atomic is ~10% slower than base, 53 outputs/1,853 parameters instead of
  44/1,628; context uses events, not remainder characters. No identical
  initial-weight or matched-parameter claim. None of the 48 probes targets
  an L1 payload or follows a recent reference; L1 usage cannot be established.
  Samples emit only L0 references. 48 targeted regressions and build passed.
  Verified all 27 restored checkpoint evaluations and 27 samples, exact
  atomic target/context examples, same corrected codec/extraction, explicit
  atomic restore and absent placement counts. Syntax, diagnostics, caller
  compatibility, version and whitespace checks passed. Browser/API show
  three arms and unsupported usage probes explicitly; graph refreshed.
- Pretraining-only run `de6da018f074421da88ac2b454e572e1`: 200 passes,
  6,200 isolated literal updates per seed; checkpoint saved/restored before
  passage training. Word accuracy 9.72% at four passes, below cold corrected
  L01's 15.97%. Active spelling recall fell from 74.2% to mean 8.6%; no
  permanent retention. Total CPU includes mean 2.541s pretraining.
- Protected run `2576af579feb44968e59958ea4e1f1c4`: five arms, fixed
  small prose, three seeds. Four-pass base/L0/corrected/pretrained/injected
  accuracy 5.56/5.56/15.97/9.72/8.33%; edit .832/.792/.727/.770/.773.
  Total training CPU 3.184/3.128/3.077/5.693/5.695s including 2.524s
  pretraining for each warm arm. Setup 3.953s separately. Frozen vectors
  condition 1,837 targets/pass and 348 sample neural calls, remaining
  unchanged. Storage/use verified, not a quality, no-forgetting or speed win.
- 164 regression tests passed, including original neural gradients and
  conditioned finite-difference checks, reference/target alignment, prompt
  resets, shared pretraining checkpoints and cost accounting. Build passed.
  Final emission-conditioning/gradient checks passed (17 targeted tests).
  Verified all 45 restored evaluations and 45 final samples, exact feature
  reconstruction from frozen weights, unchanged checkpoint/vector hashes,
  paired fresh starts and shared pretrained starts. Syntax, diagnostics,
  callers and version checks passed; browser/API show five arms, conditioned
  call counts and retention. Graph refreshed.
- Corrected run `0cb318e5564046c9953313c8a0d0e9a1`: only `eturns → returns`
  and `nplugge → unplugged` qualify; L1 removes 16 additional characters.
  Four-pass base/L0/corrected-L01 CPU 3.137/3.099/3.074s, accuracy
  5.56/5.56/15.97%, edit .832/.792/.727. Earlier accuracy milestones favor
  base over corrected L1. Combined setup 3.828s; no generation-speed win.
  39 targeted tests and production build passed. No stable quality/efficiency
  or generalization conclusion from this reused small development corpus.
  Verified all 27 restored evaluations and 27 samples, exact unique owners
  and absence of shorter keys, whole-payload extraction, paired initialization,
  protocol and executed/frozen hashes. Syntax/version checks passed; browser/API
  show two corrected keys, nine milestone rows and three generation rows.
- Fractional run `8656a80e546e45f7b7d723f96043f8df`: five/nine of 18
  repeated forms selected from 85 bank candidates; 45 saved checkpoints.
  Four-pass base/L0/quarter/half/full CPU 3.183/3.213/3.022/2.924/2.122s;
  accuracy 5.56/5.56/5.56/4.17/3.47%; edit .832/.792/.822/.825/.864.
  Quarter saves ~5% CPU versus base at matching four-pass word accuracy,
  but loses at two passes and never beats L0's edit distance. No stable
  sweet spot, quality-matched speed claim or evaluation-based retuning.
- Fractional validation: 38 targeted tests passed; production build, version
  checks and Python syntax checks passed. Restored all 45 checkpoint evaluations
  and 45 final samples exactly; verified paired initialization, training-only
  selections, remainder targets, saved protocol and executed/frozen source
  hashes. Browser/API show both selection counts, four coverage rows, 15
  milestone rows and five generation rows. Graph refreshed.
- Standalone L2 run `2e0ac7500b454704a967533ec608a814`: three seeds,
  18 checkpoints; extracts 1,040/1,942 characters (138 block occurrences,
  113 hidden-backed complete matches), leaving 902. Four-pass total CPU
  3.233/1.542s baseline/L2 (~52% less); accuracy 5.56/0.69%, edit distance
  0.832/0.953. Short generation CPU 0.144/0.085s (~41% less), 1,080/582
  base calls and 126 L2 emissions. Quality worsens; no independent hidden-only
  effect or quality-preserving efficiency claim. Combined setup 0.164s.
- Standalone validation: 36 targeted candidate/index/runtime/benchmark tests
  passed, including absence of L0/L1 references, full-training L2 discovery,
  literal-mask preservation and two-arm runner wiring. Build passed.
- Selective run `a951e691e1434ae9b945ba699fa46154`: zero of 85 eligible
  forms qualified under the predeclared repeated-context/purity/savings gate.
  Selective L01 exactly reproduces L0; no intermediate coverage or sweet spot
  established. Four-pass CPU base/L0/selective/full: 3.132/3.094/3.092/2.116s;
  accuracy 5.56/5.56/5.56/3.47%, edit distance 0.832/0.792/0.792/0.864.
  Thresholds were not retuned after results. Combined setup 0.157 CPU seconds.
- Selective regression: 157 targeted tests passed, including repeated-context
  purity, singleton/low-savings rejection, unchanged foundations, saved gate
  evidence and four-arm selective wiring. Production build passed.
  Browser/API confirmed zero of 85 accepted and four-arm result tables.
  Verified all 36 checkpoints, training-only gate evidence, exact L0/selective
  weights/counts, restored samples, code hashes and frozen source. Version,
  syntax, diagnostics and caller checks passed; graph refreshed.
- Cascade run `1bec5fa35edc4e83b316128801341c78`: three seeds on the
  unchanged workshop prose, four arms, 36 checkpoints. Four-pass CPU:
  3.103/3.072/2.094/1.527s base/L0/L01/L012, counting placement learning.
  Accuracy 5.56/5.56/3.47/1.39%; edit distance 0.832/0.792/0.864/0.946.
  L012 removes 999 characters total (51 L0, 607 L1, 341 L2), saving ~51%
  training CPU and ~49% short-sample generation CPU but worsening quality.
  Layer 2 adds 108 repeated blocks; hidden-backed forms add no coverage
  beyond Layer 1 in this corpus. No hidden-inference benefit claimed.
  Combined setup is separately measured at 0.269 CPU seconds.
- Cascade validation: 156 targeted tests passed, including unchanged seed forms,
  source-scoped references, lossless offsets, no cross-reference matching,
  literal-mask preservation, hidden-backed complete-match route, remainder-only
  targets, four-arm initialization equality and saved generation restore.
  Build/version tests and Python/editor diagnostics passed. Browser/API confirmed
  three coverage rows, twelve training rows and four generation arms. Verified
  all 36 checkpoints, stage offsets, example/code hashes and frozen source;
  memory graph refreshed.
- Prose run `acb0e2c7c63749f7b7fa2d47c363bd1b`: 1,942 training / 957
  evaluation characters, three seeds, 48 probes. 85 existing bank forms remove
  607 characters (114 occurrences), leaving 1,335. Identical paired bases:
  44 outputs / 1,628 parameters. Four-pass total CPU 3.232/2.223s baseline/
  separated (~31% lower), but accuracy 5.56/2.78% and edit distance 0.832/0.910
  both favor baseline; the same quality ordering holds at all three milestones.
  Short generation CPU 0.144/0.096s; 1,080/701 base calls, 71 indexed emissions.
  Artificial authorship-controlled corpus, not natural-text generalization.
- Prose regression validation: 151 targeted tests passed, including custom
  provenance, invalid-input rejection, no Acts loading, training-only bank
  selection, and unchanged weights/index when evaluation text changes.
  Build/version tests, syntax and caller checks passed. Browser/API confirmed
  corpus identity, sizes and results. Verified fixture hash, all 18 checkpoints,
  paired initialization, remainder examples, restored samples, executed-code
  hashes and unchanged frozen experiment. Graph refreshed.
- Larger separated run `6045eb140d8e47b1a5c6890199251039`: all ten arms,
  30 checkpoints; 30,905 training / 9,730 evaluation characters, five paired
  seeds and 256 common probes. Unchanged 128-form cap extracts 1,180 occurrences
  / 6,380 characters, leaving 24,525 (20.6% removed). Both bases: 63 outputs,
  2,103 parameters. Four-pass total CPU 64.813/51.388s (~21% less), including
  0.124s placement learning. Accuracy 1.25/2.73%, but edit distance 0.879/0.888
  remains worse. Fifteen samples/arm: 1,800/1,337 neural calls, 71 indexed
  emissions, total generation CPU 0.289/0.228s. Savings persist; quality
  remains mixed and samples incoherent. No post-result tuning.
- Larger-profile regression validation: 150 targeted Python tests passed,
  including complete chapter selection, five seeds, 256-probe request and
  persistence of all larger-profile checkpoints.
  Build/version tests, Python syntax/diagnostics and caller checks passed.
  Browser/API confirmed larger input counts, six milestone rows and generation
  totals. All 30 checkpoints, five paired initializations, remainder-example
  hashes, restored samples, executed-code hashes and frozen source verified.
- Separated run `325fdc752672405a940f0ee62de9f037`: three seeds; 113 onlies
  removed (684 characters), leaving 1,221 of 1,905 training characters.
  Both bases: 48 outputs / 1,728 parameters and identical initial weights.
  Four-pass total CPU 3.358s baseline / 2.185s separated, including 0.0075s
  placement learning (~35% lower). Word accuracy 0.69/1.39%, but edit distance
  worse at 0.836/0.914. No equal-quality claim. Short generation samples:
  1,080/691 base calls, 70 indexed emissions; total CPU 0.154/0.103s.
  Recognition runs ahead, not concurrently. Placement counts add memory and
  predict statistically; fixed spelling does not guarantee correct insertion.
- Separated validation: 149 targeted Python tests passed, including full
  extraction/offsets, remainder-only targets/contexts, no indexed base outputs,
  separate placement learning, no base-context advance on indexed emissions,
  delimiter constraints, held-out immutability and saved component restore.
  Production build passed.
  Browser/API confirmed six training rows and separate generation totals.
  Verified all 18 checkpoints, paired initializations, remainder-example hashes,
  restored samples, executed-code hashes and unchanged frozen experiment.
  Python syntax/editor diagnostics, version tests and graph refresh passed.
- Atomic run `2379abc5da8b4edda60333b5a3c1c2cd`: three seeds, 82 indexed
  forms; 48/131 outputs and 1,728/3,803 parameters. Four-pass CPU 3.59/5.30s
  baseline/indexed; accuracy 0.69/2.08%, edit distance 0.836/0.849. Equal
  12,000 updates favor baseline quality and cost. Atomic generation activates:
  73 indexed emissions, 728 versus 1,080 decisions for 1,080 output characters
  per arm, but total generation CPU 0.236 versus 0.161s. Fewer decisions do
  not offset dense output cost; no overall quality or speed gain established.
- Atomic validation: 144 targeted tests passed, including one-target,
  one-context-position and one-forward-pass emission, index persistence and
  final-versus-matched checkpoint preservation. Build passed.
  Browser/API confirmed atomic budget rows and 728/1,080 generation decisions.
  Verified all 24 milestone checkpoints, token-ID mappings, executed code hashes,
  unchanged frozen experiment, Python syntax/caller checks and version test.
- Three-arm run `ea3a9d10eae54a3ab52efadd5d7066a2`: verified exact
  baseline weights at all 20 combined milestones; source report unchanged.
  Combined predictions match baseline throughout. No rules fired in 5,120
  combined probe continuations or the 45 fresh three-arm samples.
  Combined avoids skipped-update degradation but adds no demonstrated benefit.
  Mean generation CPU/sample: 20.1ms baseline, 21.1ms recognition, 21.3ms
  combined; these short measurements do not establish a stable speed ratio.
- Three-arm validation: 142 targeted Python tests passed, including no-retraining
  checkpoint reuse, source preservation, three-arm summaries and rule counters.
  Production build and Python syntax validation passed.
  Browser/API confirmed six budget rows, three generation arms, and the
  inherited-training-time caveat; version tests and caller compatibility passed.
- Larger run `41624f6d045f4a6b9d202ed2cb877b67`: 30,905 training / 9,730
  evaluation characters. All ten arms completed; identical initialization,
  vocabulary and parameter count within every pair.
- Four-pass mean CPU: 65.75s baseline versus 46.08s recognition (~30% less).
  Next-word accuracy 1.25% versus 2.50%, but shared all-character loss is worse
  (2.685 versus 2.959). Overall equal-quality efficiency is not established.
- Equal 120,000-update CPU is essentially equal; accuracy 2.81% versus 2.73%.
  Separate matched-update samples are incoherent and trigger no spelling
  shortcuts. Preserve prior reports and frozen locked experiment.
- Larger-test validation: 140 targeted Python tests passed; build, version
  metadata test and Pylance checks passed. Browser/API confirmed five seeds,
  large-budget rows and the explicit worse-character-loss caveat. All 40
  milestone checkpoints and paired initialization equality were verified.
- Corrected retest `11fef0167915435f9238b5b57e8a7e51`: identical 48 outputs,
  1,728 parameters and paired initialization hashes. At four source passes,
  CPU 3.30s baseline versus 2.32s recognition (30% lower); estimated work
  44.4M versus 31.1M. Accuracy 0.7% versus 3.5%, both near floor.
- Equal 12,000 updates: CPU effectively equal, accuracy 4.9% versus 3.5%.
  Saved generation samples triggered no shortcuts: generation speedup is
  not demonstrated. Cached rule/bank construction is excluded from training.
- One-time expanded recognition/prefix-rule construction measured 2.02 CPU
  seconds (bank construction remains a separate historical cost).
- Corrected UI/API smoke check trained a character-only checkpoint and restored
  generation successfully (HTTP 200); legacy checkpoints retain their decoder.
- Final correction validation: 139 targeted Python tests passed, including
  prefix ambiguity, character-only vocabulary, skipped-offset context alignment,
  rule-supplied generation, persistence and existing HTTP routes. Build,
  version test and Pylance checks passed.
- Run `e9c480699e7c4d30a6b844c494ad05da`: 30% fewer encoded positions,
  but 44% more training CPU and 47% more estimated work at four passes.
- Both shared next-word accuracies near floor (0.7% baseline, 2.1% units).
  Loose similar-quality rule met; less-effort rule failed. Equal-update
  comparison favored baseline on both quality metrics.
- Development only: reused Bible chapters already seen by the bank; no
  ordinary-tokenizer control, no coherence or generalization claim.
- 133 targeted Python tests passed; production build, version metadata test,
  and Pylance checks passed. Browser/API confirmed all four comparison rows
  and the negative less-effort verdict. Graph regenerated.
- Existing transcript/graph changes from memory capture were preserved.

## 0.5.0 - 2026-11-10

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Added `neural_skeleton_runtime.py`: checksummed learn/develop-only bank,
  exact-confirmed unit encoding, and boundary-safe unit/character generation.
  Frozen experiment code and locked evaluation are unchanged.
- Wired opt-in units into the actual MLP training, vocabulary, embeddings,
  gradient updates, samples, checkpoints, and Generation UI. Character baseline
  remains default; no memory/corpus reuse or chat-engine replacement.
- Added saved-checkpoint generation via `/api/neural/generate` and the UI.
  Checkpoints carry selected units and bank provenance, so reload does not
  depend on the mutable global bank.

### Validation and limitations

- 128 targeted Python tests passed, including HTTP routes; production build,
  version metadata test, server syntax check, and Pylance syntax checks passed.
- Regenerated the memory graph from the preserved transcript.
- Focused runtime tests cover lossless encoding, fallback, word boundaries,
  no held-out vocabulary/weight influence, bank integrity and learn/develop-only
  construction, and generation after removing the global bank.
- Browser/API smoke check: a saved model generated 120 characters with 66
  decisions, including 18 complete units; invalid run IDs return HTTP 400.
- Original repeated-fixture check: 45% fewer encoded input targets, not a
  wall-clock speed or response-quality claim. Mixed-token loss is not directly
  comparable to character loss; arbitrary held-out text may overlap bank evidence.
- Runtime excludes dropped-letter claims. Known units are full-word-confirmed,
  not speculative prefix completions or proof of general-English onlies.

## 0.4.0 - 2026-11-10

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Added strict skeletons (`neural_skeletons.py`, `test_neural_skeletons.py`):
  hidden/missing-letter patterns that only one possible word fits. Layer 0 is
  SCOWL plus learning words plus regular forms, with names and hidden first
  letters handed on. A single pre-registered locked run on WEB Bible books gave
  1 wrong in 560,404 checks (bound 8.5e-6, target 1e-5 met). Added
  `THIRD_PARTY_NOTICES.md`, `licenses/`, the `skeletons_comparison` status key
  and a UI panel.
- Added strict and light layers (`neural_strict.py`, `test_neural_strict.py`):
  the strict layer claims only whole, confirmed words/phrases and spells chosen
  words whole (0 wrong claims by construction, 79% of letters, 49% fewer reading
  decisions on Acts 27). Fixed patterns moved to a light weighting layer (loss
  1.6787 → 1.6638). Run `477fc8c644cc441bba970c2da375b424`, shown via `strict_comparison`.
- Added fixed patterns (`neural_patterns.py`, `test_neural_patterns.py`):
  1–9 character triggers with forced continuations, used for skip-reading and
  whole-span generation. Fetched public-domain Acts 6–27 (learning) and Acts 28
  (locked, after pre-registration). After a development check the user revised
  the setup to support 10 and ≥10% coverage at ≥95% correct. On Acts 28, tested once:
  19.1% coverage (met) and 94.1% correct (not met); 6.9% fewer reading decisions.
  Shown on Generation via `patterns_comparison`.

### Validation and limitations

- 122 targeted Python tests passed; production build and version check passed.
- Frozen skeleton code matches the pre-registration hash. Both recorded locked
  criteria met; checks within a word are correlated, and this is one translation,
  not general-English or runtime speed evidence.
- The new panels report experiments; live training/generation integration is next.

## 0.3.0 - 2026-11-09

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Added word-level only states (`neural_word_onlies.py`): one-way and opposite
  word onlies with mutation variants; run `cacf05e4e24648ea83d77a879da0dd89`
  found 2–7 one-way and 4–15 opposite onlies holding 17–57% on Acts 2 and no
  meaningful loss change. Shown on Generation via `word_onlies_comparison`.
- Added Layer 0 only states (`neural_onlies.py`): earned one-way 2–9 character
  configurations with mutation variants; run `7a445dd9bd814f20a6be10b6bf8fbd88`
  found ~1,160 states holding 85–96% on Acts 2 but no meaningful loss gain.
- Added `neural_proof.py`, `neural_counting.py` and Layer 0 v2 composition:
  standard Kneser-Ney comparison, 10/25/50/100% training curve, five seeds,
  estimated operation counts, generated rule-break rates, frozen criteria and
  one-time fresh evaluation on newly staged Acts 4–5. Development run
  `ac6c58288c0f447ebf5cdad9844c3775`. Layer 2 met similar-result, effort, and
  data-efficiency criteria; standard counting beat all layered paths and
  Layer 0 did not help. Generation shows the proof table and verdicts.
- Added evidence-only Layer 2 attraction analysis to each new chat lab report:
  exact overlapping 2-, 3-, and 4-character patterns, counts, source offsets,
  directional conditional likelihood, source-end alternatives, and support.
- Pair reports distinguish always-followed, always-preceded, and mutual
  observed-always relationships. Finite observations never establish a law.
- Lab schema v5 introduced evidence; v6 adds Layer 2 generation and old-model
  next-character scoring as two experimental modes alongside four baselines.
  Known Layer 1 exclusions are enforced; these new modes are explicitly
  outside Layer 0 admission. Reports preserve prompt hashes and candidate traces.
- Added support-weighted context blending, source-end stop outcomes, per-mode
  prompt-fit metrics, detected violations, and equality-to-old indicators.
  The existing overlay is now explicitly labeled a narrow edit, not reversed
  execution. Different outputs are not forced or claimed as improvements.
- Added explicit scoped developmental assessments, separate from foundation
  membership and legacy alive/dead/maybe extraction. Ruleset v5 preserves
  incomplete versus completed-unreviewed observations; no new exclusions.
- Lab v7 replaces the original seed sampler with foundation/admitted blocks →
  whole-join Layer 1 checks → Layer 2 ranking. Review UI and API support explicit
  defer/admit/revoke with exact saved evidence, mechanical checks and rationale.
  Registry snapshots/hashes and selected block provenance stay in each report.
  Admission is reversible lab-generation permission only, never training approval.
- Added and executed 50 synthetic neural rounds of 100 updates, with round
  checkpoints, validation each round, fixed-prompt samples each ten rounds,
  and one final-test evaluation after round 50. Layer 2 counted once from
  training only; generated text is never used as training data.
- First run `d742210851c749fd9fcb1d6592b0b9a7`: 1,078 trainable parameters,
  5,000 updates; training loss 3.0901 -> 0.1627, validation 3.0885 -> 0.8422,
  final test 0.4282. Repetitive fragments show mechanics, not understanding.
  The old generator and transcript hashes were unchanged during execution.
- After automation: all 78 targeted Python tests, versioning test/check, and
  production build passed. Browser verified completed 50/50 rounds, 51 loss
  rows, persisted final-test score and report restoration.
- User authorized real-text training on staged public-domain Acts chapters.
  Added explicit approval and source/rights/text hash checks to the offline
  50-round runner, and live polling of saved progress in the neural panel.
- Acts run `d097c4699617462b8f80f33a759c246f` completed 5,000 updates with
  1,978 parameters: training loss 4.0602 -> 2.5788; Acts 2 validation
  4.0603 -> 2.8241; one final Acts 3 test 2.7947. Unknown character counts:
  validation 18, test 7. Generated text remains garbled, not scripture or
  evidence of understanding. Existing generator/admissions stayed unchanged.
- Real-text run uses the terminal because its roughly 85-second runtime exceeds
  the API request budget. No persistent model-training service was introduced.
- Acts integration validation: 79 targeted Python tests, versioning test/check
  and production build passed. Browser verified the real-text label, 50/50
  rounds, 51 loss rows, final score and explicit non-scripture disclaimer.
- Continued Acts run for 100 additional rounds from saved weights and original
  SGD draw position, preserving parent `d097c4699617462b8f80f33a759c246f`.
  Child `5823eaa380c043dfb2692da2f4aa7657` completed round 150 / 15,000
  cumulative updates, with 101 child checkpoints including initial round 50.
- Training loss 2.5788 -> 2.5803 and validation 2.8241 -> 2.8446: no final
  improvement from the extension. Output remains garbled. Acts 3 was not
  reevaluated because its score was already inspected. Continuation tests
  verify exact weight equality to uninterrupted training and unchanged parent.
- Added a controlled offline 4/16/32-character context comparison, with
  5,000 identical training-target draws per model, fixed hyperparameters and
  seeds, and losses/checkpoints every 500 steps. Acts 3 text is not read.
  Reports and Generation comparison display are separate from existing runs.
- Context is now saved per comparison model; legacy training retains context 4.
  Parameter counts are 1,978 / 3,514 / 5,562, so this compares context plus
  capacity, not context alone. No automatic selection or layered-model changes.
- All 11 focused neural tests and production build passed, including finite
  differences at all contexts, exact training replay, held-out isolation,
  unchanged ordinary latest report and absent final-test files.
- Comparison `e0901f8e75434d288d951222d649f09b` completed: contexts 4/16/32
  have final training losses 2.5788 / 2.8423 / 3.0405 and validation losses
  2.8241 / 3.2384 / 3.3811. Larger context did not improve this configuration;
  all samples remain garbled. No understanding claim or automatic replacement.
- Verified all 33 saved checkpoints, restored final losses/samples, exact
  reproduction of the original 4-character Acts weights, preserved parent
  hash, and completed browser polling/table. An additional 74 related layer,
  model, graph and HTTP regression tests passed (82 unique tests total across
  both commands). Editor diagnostics and Python syntax checks found no errors.
- Added a matched 0.1/0.03/0.01 learning-rate comparison at fixed context 4,
  reusing the offline comparison runner. Models share 1,978 parameters,
  identical starting weights and the same 5,000 target draws. Only update
  size differs. Acts 3 text is not read; no automatic model selection.
- Added a separate persisted/polled rate table and samples in Generation,
  preserving both context comparison and ordinary latest reports. Checkpoints
  include the learning rate. Default neural updates remain exactly unchanged.
- All 13 focused neural tests and production build passed, including explicit
  rate validation, exact gradient scaling, independent training replay for
  each rate, identical starts, held-out isolation and failed-run reporting.
- Rate run `6466ad9b71384919bb62fe87beb35620` completed 5,000 updates per
  model: training losses 2.5788 / 2.5415 / 2.8561 and validation losses
  2.8241 / 2.6582 / 2.9080 at rates 0.1 / 0.03 / 0.01. Rate 0.03 performed
  best at this budget; all samples remain garbled, not demonstrated understanding.
- Verified 33 saved checkpoints, identical starting weights and draw hash,
  restored full-split losses/all sample prompts, exact original-baseline
  reproduction at 0.1, and preserved ordinary/context reports and parent hash.
  Browser verified live completion and restoration after reload.
- Another 71 related layer, model, graph and HTTP tests passed (84 total).
  Editor diagnostics, signature compatibility and syntax checks passed.
  Default training rate remains 0.1; no automatic installation or further run.
- Includes the pending 0.2.1 badge-placement fix documented below; that
  checkpoint was not separately committed/tagged.
- Added fixed `--learning-rate` for context comparisons, preserving default
  0.1 and rejecting conflicting comparison modes. Approved repeat at 0.03
  uses fresh 4/16/32-character models and the same 5,000 target draws.
- All 14 focused neural tests passed, including independent 0.03 replay at
  each context, actual/report/checkpoint rate consistency, invalid rates,
  explicit approval and byte-identical archived runs. Signature compatibility
  and Python syntax checks passed.
- Context repeat `b63e174bbecc439da8dfdd09e4ec6fb4` completed at 0.03:
  training losses 2.5415 / 2.6139 / 2.6331 and validation losses
  2.6582 / 2.8046 / 2.9100 for contexts 4 / 16 / 32.
  All improved over their matched 0.1 runs, but context 4 remains best at
  5,000 updates. Output remains garbled; no demonstrated understanding.
- Verified 33 checkpoints, restored losses/all samples, matching starts/draw
  hashes, exact prior 0.03 weights at context 4, intact historical runs and
  unchanged ordinary/rate reports and parent hash. Browser polling/reload
  verified completed results and zero final-test evaluations.

### Validation and limitations

- Implemented the first isolated full Layer 0/1/2 neural-learning experiment
  after explicit approval of evidence-driven candidate growth. Foundation
  possibilities/maturity stay unchanged; candidate primitives and compositions
  remain unconfirmed, local to the experiment, never permanent admissions.
- Layer 0 availability and training-only Layer 2 attraction enter probabilities
  before cross-entropy/backpropagation; existing Layer 1 exclusions mask
  forbidden next characters. Tests verify gradients against finite differences
  and that this changes weight updates, not merely generated output.
- Six matched controls use context 4/rate 0.03: baseline, full_012, without_0,
  without_1, without_2 and layers_only. All raw targets remain in evaluation;
  blocked targets give infinite loss (JSON null), explicit offsets/coverage,
  and cannot count as a threshold match. Source conflicts remain saved.
- Run `48790f42b24b4bc2b865f5c437edb024` completed, baseline target 2.6582.
  Final validation losses: baseline 2.6582; full 2.6389; without_0 2.1276;
  without_1 2.6389; without_2 2.3301; layers_only 2.6360.
  All six scored all 5,274 validation characters; no rule conflicts were hidden.
- Full path first matched at zero neural updates after 0.452 seconds of
  measured training preparation; baseline required 5,000 updates and 2.586
  seconds preparation+updates. Layers-only matched at zero updates/0.405
  seconds preparation. Every arm still read all 3,411 training characters.
  These shared-machine timings are descriptive, not a general speed claim.
- Full-path setup created 3,696 unconfirmed candidates; its serialized
  candidate/attraction audit evidence is about 1.68 MB, not a peak-memory
  measurement. The current Layer 0 unseen-suffix penalty hurt validation
  relative to removing it. This critiques the numerical mapping, not the
  foundation concept. Layer 1 changed this corpus's loss very little.
  Full-path neural updates did not improve on its initial count-based prior.
- Verified all 56 checkpoints, reconstructed every final predictor's exact
  losses/samples/first-match records, and reproduced original baseline weights.
  Prior ordinary/context/rate reports and original generator hashes stayed
  unchanged; the permanent admission registry stayed absent. Browser polling
  displayed all six completed quality/effort paths. Acts 3 was not read.
  Corrected layers-only checkpoint completion metadata without changing any
  weights or measurements; the completion ordering now has a regression test.
- Final validation: all 91 targeted Python tests and the production build
  passed. Editor diagnostics, signature compatibility, Python syntax and
  version/whitespace checks passed. Browser reload restored six completed
  paths plus both prior comparison panels.

- Tests cover ` am ` versus `amount`, literal whitespace/case, both pair
  directions, terminal occurrences, support bounds, and unchanged generation.
- All 75 targeted Python tests and production build passed after developmental
  integration, including isolated HTTP admission/revocation and unchanged model,
  memory, and saved lab data. Version checks match this pending 0.3.0 checkpoint.
- Earlier versioning tests/check and browser rendering checks
  passed. Browser rendering was verified using a saved read-only lab record
  with chat submission intercepted to avoid training or memory side effects,
  including all six cards on a prompt without living seed captures.
- Developmental review form, Unicode-offset preview, explicit acknowledgment,
  request payload and visible error handling were checked in the browser with
  the write intercepted. The production registry remains empty; real
  admission/revocation persistence was verified in isolated HTTP tests.
- Layer 2 uses only the current exact prompt. No persistent attraction training,
  corpus backfill, neural updates, or ordinary-chat response changes.
- Prompt-fit losses are descriptive, not held-out evaluation or proof of
  improved answer quality. Layer 1 candidate exhaustion stops explicitly.
- Wilson lower bounds assume independent trials; overlapping/repeated text is
  correlated and cannot justify strong certainty or universal exclusions.
- Layer 0 possibilities and Layer 1 exclusions remain unchanged. Observing a
  pattern does not admit it to Layer 0 or approve it for training.
- No automatic developmental admissions are provided. Boundary/join checks do
  not prove universal composability or meaning; whole outputs are rechecked
  during generation. Unsupported domains fail explicitly.

## 0.2.1 - 2026-11-09

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: patch

### Changes

- Moved the subtle version badge from the page viewport to the bottom-right
  inside the galaxy canvas. It stays with the canvas when scrolling or resizing,
  without intercepting graph interactions.

### Validation and limitations

- Targeted graph-generation tests, version consistency check, and production
  build pass. Browser geometry verifies the badge is inside the canvas.
- No galaxy membership, memory, or model-training behavior changed.

## 0.2.0 - 2026-11-09

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Reconciled the stale 0.1.0 package version with the feature-bearing current
  application. Established patch/minor/major rules and build-time manifest/log
  checks; future checkpoint commits must update this log and the version.
- Added a muted bottom-right memory-graph version badge sourced from the
  application manifest when the graph is generated.
- Integrated Layer 0 possibilities and explicitly agreed Layer 1 exclusions,
  exact-text extraction with provenance, alongside existing approved training.
- Added two extraction panels and four chat generation comparisons, including
  captured-seed overlays and initial consonant repair. Available outputs remain
  visible when an empty mode prevents an equal-length comparison.
- Added isolated from-scratch neural character MLP training, new-text approval,
  persisted run checkpoints, and weight/loss/before-after progress reporting.
- Preserved promoted memory pairings and individual links; established-parent
  galaxies use solid shell connectors, while Under evaluation promotions do not.
- Recorded backlog topics including memory consolidation and thought progression.

### Earlier checkpoints in this session

- `706f71a`: seed layers, extraction/generation lab, galaxy relationships.
- `1577f05`: neural weight lab and zero-common-output display fix.
- These commits were pushed without synchronized application version bumps;
  this entry reconciles them rather than retroactively relabeling them.

### Validation and limitations

- Previous feature checkpoint: 61 targeted Python tests and production build
  passed; neural training/report persistence and saved chat-lab rendering were
  checked in the browser.
- Versioning checks/tests and production build are validated in this checkpoint.
- Graph-generation fixture includes its application manifest and verifies the
  rendered badge; all 61 targeted Python tests pass with that integration.
- Synthetic neural run: 653 trainable parameters, 300 steps, training loss
  1.5904 -> 0.0106, repeating-pattern held-out loss 1.5908 -> 0.0205.
  This verifies mechanics, not independent generalization.
- Neural training remains isolated from ordinary chat and historical memories.
  Layer-assisted neural training is not implemented. Local model checkpoints
  remain ignored and are not included in Git release history.
- The unrelated pre-existing full-suite test failure documented during this
  session was not fixed; targeted results do not claim a clean full suite.
