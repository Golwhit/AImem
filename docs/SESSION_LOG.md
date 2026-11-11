# Versioned Session Log

Newest application checkpoint first. Entries map delivered versions to the
session that produced them; historical conversation memory remains separate.
See [versioning policy](VERSIONING.md) for bump and release rules.

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
