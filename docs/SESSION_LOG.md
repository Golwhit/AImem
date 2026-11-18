# Versioned Session Log

Newest application checkpoint first. Entries map delivered versions to the
session that produced them; historical conversation memory remains separate.
See [versioning policy](VERSIONING.md) for bump and release rules.

## 0.48.0 - 2026-11-18

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Injected memories now get topics from the user's topics merged with
  vocabulary matches from `DOMAIN_TERMS` (`extract_topics`). Extracted and
  user topics are stored the same way and count toward cohesion.
- Each injection creates a deterministic `Generated N` memory (role `AI`,
  origin `ai-generated`) with topics, nearest memories, a domain hint, and a
  suggestion. It is linked to its source with a two-way `generated` bond.
- Injected and generated memories are drawn as standalone stars outside the
  evaluation split. They join a galaxy only if absorbed later; at creation they
  are unassigned.

### Validation

- `python3 -m unittest -q test_memory_learner test_memory_clusters`: 56 tests,
  1 failure, the pre-existing `test_http_routes_use_local_learner_and_save_excerpts`.
- Temp-copy graph regeneration succeeded and the embedded script passes
  `node --check`.

### Limitations

- Browser star rendering was not verified: the static server had no API, so the
  graph showed zero nodes. The `generated` link surviving the snapshot merge is
  unverified.
- Generated notes are deterministic, not LLM output.
- Injected memories still take the `domain` from `update_clusters`; galaxy
  absorption is not implemented.

## 0.47.3 - 2026-11-18

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: patch

### Changes

- Moved the memory edit and add-memory controls from the Selected memory panel
  into Prompt training, using the panel's existing `query-input`, `section-label`,
  and `graph-actions` styles.

### Validation

- `python3 -m unittest -q test_memory_clusters` passes.
- Browser check: the editor appears in Prompt training and loads the bare text of `You 8`.

## 0.47.2 - 2026-11-18

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: patch

### Changes

- Memory editor now strips the `You: ` role prefix that the graph adds to node
  content, so saving no longer writes the prefix into `chat_memory.json`.
- Validation: browser save on the real transcript changed only `You 8` (period
  added, `edit_history` recorded, approval reset). A pre-save backup is at
  `.data/backups/chat_memory.pre-browser-save-20261118-072059.json`.

### Limitations

- The save returned a transient 500 in the browser console, but the write and
  graph regeneration completed. Not reproduced since.

## 0.47.1 - 2026-11-18

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: patch

### Changes

- Memory edit and inject routes now return the learner's plain error message
  instead of a JSON string nested inside `error`.
- Validation: the restarted server returns 400 with `text is required` for an
  empty inject and an empty edit. Learner edit and inject were exercised against
  a copy of the transcript.

## 0.47.0 - 2026-11-18

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Memory review and editing. Clicking a user memory in the graph opens an
  editable text and topics box. Saving records the previous text and topics in
  `edit_history` and clears approval, because approval is keyed to the memory
  hash. Only user-authored (`You`) memories are editable.
- Manual injection. The "Add a memory" form adds a `You` memory with
  `origin: "user-injected"` and an id such as `Injected 1`. It enters the review
  queue unapproved and does not go through the chat stop hook.
- The learner status now reports `edited_memories`, `injected_memories`, and
  `reapproval_needed`.
- Validation: 18 learner tests (one pre-existing HTTP failure remains) and 36
  graph tests pass. The server passes a syntax check. Live browser editing and
  injection were not exercised, to avoid writing test data to the transcript.
- Limitations: search still uses the existing graph search, not a new partial
  match for memories. Asking the AI about a memory does not yet open that memory.
  Injected memories have no links until the graph re-clusters them.

## 0.46.2 - 2026-11-18

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: patch

### Changes

- Renamed evaluation areas to seeders. Pairs of reciprocal verified exchanges
  with no link of at least 0.36 to an outside memory now float free of the
  seeders. Other pairs are pulled together at 1.5x weight. Escaped pairs move
  one damped step toward their goal on each render (no timer), so they settle
  over turns rather than animating continuously.
- Validation: 36 focused graph and cluster tests passed. The live page loaded
  with no page errors and showed the escaped pairs in place.
- Limitations: the README and FROM_SCRATCH docs do not yet describe escape,
  the 0.36 cutoff, pair pull, or drift.

## 0.46.1 - 2026-11-17

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: patch

### Changes

- Group verified evaluation exchanges before dividing and relaxing the suns,
  keeping both turns close without merging them or altering their stored
  connections. Preserve paired memories' external links through the sun
  display filters; zero-strength links do not create layout attraction.
- Validation: 35 focused graph and cluster tests passed. Live browser checks
  found 599 verified pairs in the same sun and all 7,379 paired external
  connections present as drawn elements, with 57 short dots retained and no
  overlaps among ordinary pairs. Python syntax, version tests, whitespace
  checks, and the Node 22 production build passed. Links behind galaxy fills
  may still be hard to see; ambiguous older turns are not guessed. The grouping
  does not approve data for training.

## 0.46.0 - 2026-11-17

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Automatically show eligible short under-evaluation turns as selectable
  amber dots within their detailed exchange partners. The internal link is
  hidden in the view only; original messages, feedback, provenance, domain
  membership and links remain intact. Each partner can be inspected.
- Add provisional lexical context associations for matching short turns with
  related detail, without duplicating an existing cross-link or forcing
  unmatched pairs together. These derived associations do not supply
  promotion evidence or training approval. Matching words and long replies
  do not prove semantic agreement.
- Validation: 32 graph clustering tests passed, including pairing,
  association, promotion isolation, regeneration and preservation cases.
  Python syntax checks, the version test and production build passed. The
  live browser showed 57 paired dots and their partner controls. A broader
  learner run passed 46 of 47 tests; its unrelated HTTP test expects 320
  repeated characters, but the existing repetition guard emits two.
  The prior trial found repeated test prompts and misleading contexts;
  a fallback response and generic "next" associations are excluded.

## 0.45.0 - 2026-11-17

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Assessed the 40,600-update character candidate against the live
  37,700-update base. Teaching improved 155/280 to 164/280, but 16 previously
  correct teaching answers and four older development answers were lost.
  Feedback retention and replayed midpoint/final saturation passed. The
  user chose to keep the existing live base.
- Created the private ignored `portable-model/` folder with the full
  candidate weights, live-state anchor, source files, reviewed meals,
  integrity manifest, placeholder environment names, and Windows transfer
  instructions. No actual secrets or chat transcript are included; model
  weights may nevertheless reflect training data.
- Added isolated reviewed-meal CPU training, exact replay, and return staging
  that checks the live anchor and hash-linked round chain without activation.
  The existing 50/50 rehearsal protocol is not implemented by this runner.
  Full independent return replay, live promotion and CUDA acceleration remain
  unimplemented; Windows and NVIDIA hardware have not been tested.
- Validation: the bundle manifest and model restoration passed, a one-update
  CPU round replayed exactly, mismatched review hashes were rejected, and
  return staging against the actual live base succeeded without activation.
  Altered bundle files were rejected, the version test and production build
  passed with Node 22, and the graph was regenerated. These checks do not
  prove GPU behavior or independent generalization.

## 0.44.0 - 2026-11-17

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Selected the 37,700-update character-only conversation checkpoint as the
  live working base after the user explicitly accepted its measured
  non-regression failures as a one-time exception. Selection round 40 is
  administrative and adds no training updates. The previous live selection,
  checkpoint and progress remain saved for rollback.
- Made character-only output persist through model snapshots, chat generation,
  feedback training and replay. Existing word/gap, spacing, partial-word
  context, generation filtering and later training safeguards remain in use.
  The new base was not created by combining separate model weights.
- Saved an independent, untrained experimental copy of the exact selected
  state under `universe-lab/experiments/character-base-f668ee33396a42958bbb08f04e6997d7/`.
  The original character comparison remains available.
- The candidate scored 155/280 on approved teaching versus the old live
  model's 204/280, losing 51 previously correct answers, including three
  approved feedback answers; paired development fell from 10/32 to 9/32.
  The reused checks are not independent generalization evidence. User
  approval is not legal clearance.
- Validation: 30 focused model, training and selection tests plus 13 lab
  tests passed; the actual candidate was selected, generated, copied and
  rolled back in a temporary environment. Restored live-compatible word
  scores matched all 64 measured fresh cases. The actual live model,
  training progress and experimental copy match; live generation and
  the archived rollback selection were verified. A character-only correction
  was trained and replayed in isolation, preserving its output mode. The
  version test and production build passed with Node 22, and the graph was
  regenerated. Vite reported existing deprecation warnings.

## 0.43.3 - 2026-11-17

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: patch

### Changes

- Documented a fourth 2,900-update mixed-meal round reaching 37,700 updates.
  Fresh development remained 32/64 first words, 4/12 second words and 30/64
  complete answers, with two previously correct cases lost. Old teaching
  rose 152/280 to 155/280 but lost 20 previously correct answers.
- Compared the candidate's character-only output with the actual live
  hybrid model using the live approved teaching, development and feedback.
  Candidate teaching was 155/280 versus live 204/280, with 51 previously
  correct answers lost, including three approved feedback answers;
  development paired accuracy was 9/32 versus live 10/32. Replayed hidden
  saturation was 0.222 halfway and 0.201 at the end, below the .6 limit.
  Retention and paired non-regression checks failed. These reused checks
  are not independent generalization evidence; review is not legal clearance.
- Preserved a checksum-verified copy of the active selection, training
  progress, checkpoint, and original comparison; live selection and
  progress remained unchanged. No activation or guard override was performed.
- Validation: 20 focused Python tests and private-runner compilation passed;
  the round replayed exactly, live comparison verified input hashes, and
  milestone replay reached the exact candidate state. Rollback file hashes,
  version tests, version check, build, and whitespace check passed. The
  build used isolated Node 22 and its optional native binding. The memory
  graph was regenerated.

## 0.43.2 - 2026-11-17

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: patch

### Changes

- Documented a third isolated 2,900-update round on the same reviewed meal
  and rehearsal stream, reaching 34,800 updates. It drew 1,430 fresh and
  1,470 rehearsal positions, reproduced the parent scores and replayed
  exactly.
- Fresh development first words rose 30/64 to 32/64 and complete answers
  28/64 to 30/64; second words fell 6/12 to 4/12. Contrasting pairs with
  both answers exact rose 10/32 to 12/32. The changed-wording subset stayed
  10/32 while losing two previously correct cases. Fresh owner and transfer
  cases were both 0/8.
- Older teaching fell 154/280 to 152/280 while losing 24 previously
  correct cases; previous-meal development stayed 40/64 but lost four.
  Reused, developer-visible cases do not prove generalization and no
  same-budget fresh-only control isolates rehearsal's effect. Full
  activation and mid-round safeguards were not run; active chat remained
  unchanged. User approval is not legal clearance.
- Validation: 20 focused Python tests and private-runner compilation passed.
  The prior word, paired and exact-answer scores reproduced; replay and
  protected-state, source and reviewed-input checks passed. Saved checkpoint
  hash, version tests, version check, build, and whitespace check passed.
  The build used isolated Node 22 and its optional native binding. The
  memory graph was regenerated.

## 0.43.1 - 2026-11-17

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: patch

### Changes

- Documented two isolated 2,900-update rounds using a newly generated,
  exact-hash user-approved 256-exchange meal and pooled previously approved
  conversation rehearsal at 50/50 sampling probability per character
  position. They reached 31,900 updates and replayed exactly. Draws were
  1,426 fresh / 1,474 rehearsal, then 1,421 fresh / 1,479 rehearsal.
- Fresh development first words rose 16/64 to 26/64 to 30/64; second words
  0/12 to 2/12 to 6/12; complete answers 14/64 to 24/64 to 28/64; both
  answers correct in a contrasting pair 2/32 to 6/32 to 10/32. The
  changed-wording subset rose 6/32 to 10/32 and stayed at 10/32, with four
  previously correct answers lost in the second round.
- Old teaching improved 108/280 to 144/280 to 154/280, with six previously
  correct cases lost in round one and none in round two. Previous-meal
  development stayed 40/64 but lost four previously correct cases in round
  one. The developer-visible diagnostics are not independent generalization
  evidence; no fresh-only control isolates rehearsal's effect. User review
  is not legal clearance. Full activation and mid-round safeguards were not
  run; active chat remained unchanged.
- Validation: 20 focused Python tests and private-runner compilation passed.
  Prior scores reproduced; both rounds replayed exactly and protected-state,
  source and reviewed-input checks passed. Saved checkpoint hashes and
  score continuity, version tests, version check, build, and whitespace
  check passed. The build used isolated Node 22 and its optional native
  binding. The memory graph was regenerated.

## 0.43.0 - 2026-11-17

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Extended the new-meal runner to resume a validated checkpoint using the
  same reviewed meal and deterministic sampling stream. The next 2,900
  updates reached 26,100; prior word and exact-answer scores reproduced
  and replay was exact.
- New-meal development remained 44/64 first words, 8/12 second words and
  40/64 complete answers, with none of its correct answers lost. Second-word
  third-letter matches rose 0 to 4 of four eligible targets, but fourth
  letters remained 0/4. Old teaching fell 115/280 to 108/280 with eight
  previously correct answers lost; old meal teaching fell 118/256 to
  109/256 with 14 lost. Old development rose 23/64 to 25/64 but lost two
  previously correct answers.
- These repeatedly used, template-related checks do not prove generalization;
  no same-budget rehearsal control isolates the cause of forgetting. Review
  is a user declaration, not legal clearance. Full activation and mid-round
  safeguards were not run; the active model was not changed.
- Validation: 20 focused Python tests and compilation passed, as did
  Pylance syntax checks, version tests, version check, build, and whitespace
  check. The build used isolated Node 22 and its optional native binding.
  The run reproduced previous scores, replayed exactly and verified
  protected state; the memory graph was regenerated.

## 0.42.0 - 2026-11-17

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Added an isolated, exact-hash-reviewed new-meal runner with deterministic
  replay and previous-checkpoint, input, and protected-state checks. A
  2,900-update new-meal-only run reached 23,200 updates.
- On the new 64-case development set, first words rose 12 to 44, second
  words 0 to 8/12, and complete answers 12 to 40. On old development,
  complete answers fell 24 to 23 with two previously correct answers lost.
  Old teaching fell 135/280 to 115/280 with 28 lost; old meal teaching
  fell 142/256 to 118/256 with 28 lost. The new evaluation shares training
  templates; no same-budget rehearsal control establishes the cause of
  forgetting. Review is a user declaration, not legal clearance.
- Validation: 18 focused Python tests and compilation passed; exact replay,
  prior word-score reproduction and protected-state checks passed. Version
  tests, version check, build, and whitespace check passed. The build used
  an isolated Node 22 runtime and missing optional native binding because
  the default Node 18 environment could not build. The memory graph was
  regenerated. Full activation and mid-round safeguards were not run; the
  active model was not changed.

## 0.41.1 - 2026-11-16

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: patch

### Changes

- Documented a final isolated 2,900-update continuation using the approved
  256-exchange meal and prior conversation rehearsal, reaching 20,300 total
  updates. The 1,422 meal and 1,478 rehearsal draws replayed exactly.
- On the same 64 conversation cases, first words increased 21 to 24 and
  complete answers 19 to 24; one previously correct case was lost in each
  measure. Second words remained 6/8, all with the first word correct. All
  eight two-word targets start `I do`, limiting the interpretation.
  Rehearsal complete answers increased 118/280 to 135/280 with no lost
  previously correct cases; meal development increased 13/32 to 15/32,
  also without losses. Both text groups remained at zero exact first and
  second words.
- These are reused checks, not independent generalization evidence. Full
  activation and mid-round safeguards were not run; the checkpoint was not
  activated and earlier checkpoints remain preserved.
- Validation: 16 focused Python tests, version tests, production build and
  documentation whitespace checks passed. Active and progress still select
  round 39; their hashes, the live universe and the parent checkpoint were
  unchanged. The memory graph was regenerated.

## 0.41.0 - 2026-11-16

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Added isolated character-model training on the previously approved,
  exact-hash 256-exchange meal with equal-probability approved conversation
  rehearsal, exact replay, input/protocol checks, and retained sampling
  position for later meal continuations. The selected meal contains 169
  prompts not present in the earlier teaching and 87 overlaps; no selected
  meal training prompts overlap the previous conversation development set.
- The 14,500-update experimental checkpoint received 2,900 updates
  (1,406 meal and 1,494 rehearsal positions). On reused conversation cases,
  first-word accuracy rose 15/64 to 21/64 and complete answers 12/64 to
  19/64; second words stayed 6/8. No previously correct complete answers
  were lost. Rehearsal complete answers increased 101/280 to 118/280 and
  meal development answers 10/32 to 13/32; neither lost previously correct
  answers. The weights and draw counts replayed exactly.
- These are reused checks with a small second-word sample. No same-budget
  no-new-meal control isolates the effect of the meal, and full activation
  and mid-round safeguards were not run. The checkpoint remains isolated.
- Validation: 16 focused Python tests, including split meal-stream replay,
  Python compilation, version tests, and production build passed. Active
  and progress still select round 39; their hashes and the live universe
  hash were unchanged. The memory graph was regenerated.

## 0.40.2 - 2026-11-16

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: patch

### Changes

- Documented another isolated 2,900-update conversation-only continuation,
  from 11,600 to 14,500 total updates. The round replayed exactly.
- On the same 64 conversation cases, exact first words rose 13 to 15 and
  complete answers 7 to 12. One previously correct case was lost in each
  measure. Exact second words stayed 6/8; all six also had the first word
  correct. One OANC first word matched; the remaining text-group exact
  first- and second-word counts were zero.
- These are reused checks, with only eight second-word targets, not
  independent generalization evidence. Full activation and mid-round
  safeguards were not run; the checkpoint was not activated.
- Validation: 13 focused tests, the version test, production build, and
  documentation whitespace checks passed. Active and progress still select
  round 39; their hashes and the live universe hash were unchanged. The
  memory graph was regenerated.

## 0.40.1 - 2026-11-16

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: patch

### Changes

- Documented a further 2,900-update conversation-only continuation of the
  8,700-update experimental checkpoint. The resulting 11,600-update model
  replayed exactly and remained separate from active chat.
- On the same 64 conversation cases, exact first words stayed 13/64 and
  complete answers stayed 7/64, but each measure lost six previously correct
  cases and gained six others. Exact second words increased from 5/8 to
  6/8; all six cases also had the first word correct. Both text-continuation
  groups remained at zero exact first and second words.
- These checks are reused, with only eight second-word targets; they are
  not independent evidence of generalization. Full activation and
  mid-round safeguards were not run.
- Validation: 13 focused tests, the version test, production build, and
  documentation whitespace checks passed. Active and progress still select
  round 39; their hashes and the live universe hash were unchanged. The
  memory graph was regenerated.

## 0.40.0 - 2026-11-16

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Added hash-checked continuation of isolated character checkpoints with
  reconstruction of the appropriate conversation-only or mixed random
  sampling stream. The saved checkpoint, data, cases, scoring dependencies
  and previous scores must match before a continuation starts.
- Continued the 2,900-update conversation-only checkpoint for two more
  2,900-update rounds. Each round sampled only approved conversation
  characters and replayed exactly. On the same 64 conversation cases,
  exact first words increased 2 to 7 to 13; exact second words increased
  0 to 0 to 5 of eight cases with second targets. The 8,700-update
  checkpoint got both words correct in five cases and the full expected
  answer correct in seven. No previously correct first word or full answer
  was lost between measured stages.
- Both text groups remained at zero exact first and second words. The
  conversation evaluation is reused, with only eight second-word targets,
  and is not independent generalization evidence. Full activation and
  mid-round safeguards were not run. The active model and previous
  checkpoints remained unchanged.
- Validation: 13 focused tests, including split versus uninterrupted
  training, Python compilation, version tests, and production build
  passed. Active and progress still select round 39; their hashes, the live
  universe, and the original character checkpoint were unchanged. The
  memory graph was regenerated.

## 0.39.0 - 2026-11-16

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Added conversation-only character training and optional hash-pinned
  comparison against a prior character trial. Comparison verifies the
  same fresh initial weights, reviewed inputs, update count, evaluation
  cases, initial scores, and unchanged scoring dependencies.
- Compared 2,900 conversation-only updates with 2,900 updates using the
  earlier 75% conversation / 25% corpus mix. Training draws and final
  weights replayed exactly. Conversation first letters improved 14/64 to
  19/64 and exact first words 0/64 to 2/64, but second letters fell from
  10/56 to 1/56. Both exact words were `I`; 56/64 first outputs were `B`.
  Both runs missed all eight expected second words.
- In the two text groups, first-word missing counts fell from 50/64 to
  3/64 for Bible and from 58/64 to 4/64 for OANC, but neither run got
  any complete first or second words correct in those groups. One seed,
  reused development cases, nonidentical sampled training positions, and
  the small text sample limit conclusions. Full activation safeguards were
  not run; both experiments remain isolated from active chat.
- Validation: 12 focused tests, Python compilation, version tests, and
  production build passed. Active and progress still select round 39;
  their hashes, the live universe, and the mixed-run checkpoint were
  unchanged. The memory graph was regenerated.

## 0.38.0 - 2026-11-16

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Added an isolated fresh-weight, character-output training and evaluation
  trial. It compares conversation prompts against known answers and
  preceding text against remaining words, without supplying the first
  target letter. It reports exact first/second words, first four letter
  positions, missing and unfinished predictions, and one-target cases
  excluded from second-word denominators. Expected but missing words count
  wrong. Longer words continue to their boundary or the 64-character cap.
- On the same reviewed conversation and corpus inputs, 2,900 updates
  replayed exactly. Of 64 reused conversation development cases, first
  letters improved 0/64 to 14/64, but exact first words remained 0/64 and
  all 64 generated first words began with `B`. All eight expected second
  words were missing after training. On each of
  64 Bible and 64 OANC continuation cases, exact first and second words
  were zero; 50 and 58 first words were missing after training.
- This is one seed and a short run. Source cases use the first 32 eligible
  held-out documents per source, with one- and two-target cases from each
  selected segment; near duplicates may cross the split. Context groups
  have different target distributions and are not an apples-to-apples
  comparison. Reused development cases are not independent generalization
  evidence, and full activation safeguards were not run. No model was
  activated or prior experimental checkpoint replaced.
- Validation: 11 focused tests and Python compilation passed; Pylance
  syntax checks timed out. The version test and production build passed.
  Active and progress still select round 39; their hashes and the live
  universe hash were unchanged. The memory graph was regenerated.

## 0.37.2 - 2026-11-16

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: patch

### Changes

- Documented three replay-verified branches from the isolated 34,800-update
  experimental checkpoint, using the same reviewed data and continuing
  random stream, at 725, 1,450, and 2,900 additional updates.
- On the reused 1,604,277-pair test, second-initial availability changed
  from 51.53% at the start to 6.59%, 12.56%, and 0%, respectively;
  accuracy changed from 3.58% to 0.38%, 0.67%, and 0%. Exact pair counts
  were 47, 1, 44, and 0. One greedy prediction per starting letter is
  reused for many pairs; these are not independent sentence-generation
  tests. At 2,900 updates all 52 tested initials ended the first line
  after one word.
- Teaching scores were 162, 167, 167, and 171 out of 280. Relative to the
  starting checkpoint, the 725-update branch lost two previously correct
  teaching answers, while the other branches lost none. Final saturation
  was 0.2547, 0.2485, 0.2447, and 0.3023, respectively; the first collapse
  occurred without an increase. Full activation safeguards were not run.
  Active chat and the preserved experimental parent were not changed.
- Validation: eight focused continuation/comparison tests, version tests,
  production build, and documentation whitespace checks passed. Active and
  progress still select round 39; their hashes, the live universe, and the
  experimental parent checkpoint were unchanged. The graph was regenerated.

## 0.37.1 - 2026-11-16

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: patch

### Changes

- Documented another isolated 17,400-update continuation of the saved
  34,800-update experimental checkpoint, using the same reviewed inputs.
  The resulting 52,200-update weights replayed exactly; updates took
  166.42 CPU seconds, excluding replay and evaluation.
- On the same 1,604,277 held-out pairs, second-initial accuracy fell from
  3.58% to 0.81%, availability from 51.53% to 17.12%, correct second
  words from 754 to 99, and exact pairs from 47 to 33. Teaching increased
  162/280 to 180/280, development 20/64 to 25/64, paired development
  8/32 to 9/32, and feedback 3/24 to 4/24. Nevertheless, 17 previously
  correct teaching answers and one feedback answer were lost. Final
  saturation increased from 0.2547 to 0.3067.
- The checks are reused, the held-out pair restrictions and split
  limitations remain, and full activation safeguards were not run. Both
  experimental checkpoints remain isolated; the active model was unchanged.
- Validation: eight focused continuation/comparison tests, version tests,
  production build, and documentation whitespace checks passed. Active and
  progress still select round 39; active selection, progress, and live
  universe hashes were unchanged. The memory graph was regenerated.

## 0.37.0 - 2026-11-16

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Added isolated experimental-model continuation using the saved
  17,400-update fresh checkpoint, the same reviewed inputs and reconstructed
  random stream. Another 17,400 updates took 179.62 update-CPU seconds,
  excluding replay and evaluation. The starting checkpoint matched and the
  continued weights replayed exactly; total updates are 34,800.
- On the same 1,604,277 held-out pairs, second-initial accuracy rose from
  0.74% to 3.58%, and exact two-word matches from 0 to 47. Second-initial
  availability fell from 100% to 51.53%. Second-word-only accuracy was
  0.047%, versus 0.341% for the previously measured mixture. Only the
  actual first initial is supplied; actual first words are at most four
  letters, and all second-word lengths are included. Pairs can cross
  punctuation and near-duplicate content can cross the split.
- On reused checks, teaching improved 104/280 to 162/280, development
  16/64 to 20/64, paired development 1/32 to 8/32, and feedback 2/24 to
  3/24. One previously correct teaching answer was lost; no previously
  correct feedback answer was lost. Final saturation was 0.2547; full
  activation safeguards were not run. The experimental checkpoint was not
  activated, and active round 39 remained unchanged.
- Validation: 73 focused Python tests, three pattern tests, Python
  compilation, version tests, and production build passed. Active and
  permanent progress still agree at round 39; the live universe hash is
  unchanged. The memory graph was regenerated.

## 0.36.0 - 2026-11-16

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Added an isolated fresh-weight comparison using the active architecture,
  lexicon and symbol inventory, the reviewed corpus and approved conversation
  rehearsal. Starting from the same fresh seed, 5,800- and 17,400-update
  runs used the same initial sample order; both final weights replayed
  exactly. The 5,800 updates match the rejected continuation's last round;
  update CPU was 59.07 and 181.96 seconds, excluding replay and evaluation.
- On 1,604,277 held-out two-word pairs, the short fresh run never reached
  the second word; the longer reached it on all cases and predicted its
  initial 0.74% of the time. Both got zero exact pairs. The previously
  measured active/candidate mixture scored 3.18% on the initial and five
  exact pairs. Only the first initial is supplied; actual first words are
  at most four letters. Pairs can cross punctuation and repeated or
  near-duplicate content can cross the split.
- On reused checks, short / long fresh scored 0/280 / 104/280 teaching,
  0/64 / 16/64 development and 0/24 / 2/24 feedback; the earlier mixture
  scored 226/280, 27/64 and 11/24. The longer fresh run ended at 0.1747
  saturation, but intermediate saturation and full promotion safeguards
  were not checked. One seed and two budgets are not a general verdict;
  the mixture has a longer training history and an unresolved rejected
  branch. Neither fresh run was activated.
- Validation: 71 focused Python tests, three pattern tests, Python
  compilation, version tests, and production build passed. Active and
  permanent progress still agree at round 39; the live universe hash is
  unchanged. The memory graph was regenerated.

## 0.35.0 - 2026-11-16

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Added an isolated two-word comparison of active round 39, its rejected
  corpus candidate, and their fixed equal-weight output-probability
  mixture. The individual checkpoints and live selection were unchanged.
- On 1,604,277 held-out pairs, active / candidate / mixture correctly
  predicted the second initial 1.01% / 4.37% / 3.18% of the time; exact
  two-word counts were 0 / 25 / 5. Only the first initial is supplied,
  actual first words are at most four letters, and all second-word lengths
  are included. Pairs can cross punctuation; repeated words and
  near-duplicate documents can cross the split.
- On reused assessments the mixture scored 226/280 teaching, 27/64
  development, 11/32 paired development and 11/24 feedback, with no
  previously correct teaching or feedback answers lost relative to
  active. Active scored 204/280, 26/64, 10/32 and 6/24; the rejected
  candidate scored 236/280, 29/64, 12/32 and 10/24. This is a fixed blend,
  not a trained integration or a fresh-model comparison. The rejected
  branch's saturation failure remains unresolved.
- Validation: 69 focused Python tests, three pattern tests, Python
  compilation, version tests, and production build passed. Active and
  permanent progress still agree at round 39; the live universe hash is
  unchanged. The memory graph was regenerated.

## 0.34.0 - 2026-11-16

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- At the user's request, ran a guarded 60-update-CPU-second continuation
  from active round 39 on the exact reviewed user corpus and approved
  conversation rehearsal. Run `7a67711a46ef4a609a04d92d1b7ee2a1` made
  5,800 updates, replayed exactly and was rejected for saturation plus
  loss of one previously correct teaching item and one feedback item.
  Active round 39 and its checkpoint remained unchanged.
- Overall teaching improved 204/280 to 236/280, development 26/64 to
  29/64, paired development 10/32 to 12/32, feedback 6/24 to 10/24, and
  teacher-forced corpus positions 214/536 to 278/536. Saturation was
  0.6019 and 0.6064 at the two milestones, above the 0.6 corpus limit.
- Added an isolated held-out three-word evaluator. On 708,090 fragments,
  the original active model did not reach a third word. The rejected
  candidate reached one on 544,360 (76.88%), predicted its initial on
  21,193 (2.99% of all cases), and exactly predicted zero third words or
  three-word sequences. The first initial is supplied; actual first and
  second words are at most four letters. The greedy model/decoder differs
  from the earlier index experiment; this is not a conversation-quality test.
  Bible/OANC fragments were evaluation only; the user corpus was the
  reviewed corpus-training input.
- Validation: 65 focused Python tests, three pattern tests, Python
  compilation, version tests, and the production build passed. The memory
  graph was regenerated. Active and permanent progress still agree at
  round 39, and the live universe hash is unchanged.

## 0.33.0 - 2026-11-16

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Added an isolated two-word initial/prefix index, with blended one-word
  fallback, and a matched comparison at nested 25%, 50%, and 100% training
  document samples. Both generated arms share the same first two words;
  another diagnostic supplies actual first and second words.
- On 708,090 held-out fragments at full size, known-context third-initial
  accuracy was 19.96% / 20.96% for one / two preceding words, and complete
  third-word accuracy was 9.13% / 10.30%. In the generated chain the same
  initial scores were 9.49% / 7.84%, word scores 2.11% / 1.32%, and exact
  three-word scores 0.269% / 0.306%. More documents did not consistently
  improve two-word performance. No optimizer or active model was trained.
- The 512 indexed two-word contexts covered 44.12% of actual test pairs;
  others fall back to one-word context. Only the first initial is supplied
  in the chain, the first two actual words are short, and longer third
  words cannot be completed. One-word training may cross punctuation;
  indexed triples and test fragments do not. Split reuse and mapping
  calibration limitations from the earlier experiment still apply.
- Validation: 63 focused Python tests, three pattern tests, compilation,
  version tests, and production build passed. Full-data one-word scores
  reproduced the previous run. Pylance signature checking timed out; the
  existing call sites and focused tests were checked separately. Memory
  graph regenerated; live universe, active selection, and checkpoint hashes
  were unchanged.

## 0.32.0 - 2026-11-16

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Added an isolated three-word continuation comparison, reusing the joint
  search with a generated preceding word; preserved the two-word context
  index as a future review idea, not an implemented feature.
- On 708,090 held-out fragments, beam versus greedy third-initial accuracy
  was 9.49% versus 5.70%, and complete three-word accuracy 0.27% versus
  0.12%. On the 495,756 cases where greedy reached the third initial,
  complete sequence accuracy was 0.26% versus 0.17%. A separate diagnostic
  given the actual second word scored 19.96% on the third initial.
- Both actual preceding words are at most four letters, only the first
  initial is supplied to the chains, and longer third words cannot be
  completed. Fragments do not cross punctuation or documents, but training
  pairs can cross punctuation. Repeated words and near duplicates can cross
  the split; mapping calibration included broader corpus frequencies.
  The run used approved hash-pinned private inputs and did not compare
  conversational quality against the active model.
- Validation: 57 focused Python tests, three pattern tests, version tests,
  Python compilation, and the production build passed. The previous joint
  experiment's aggregate counts and scores were exactly reproduced after
  the refactor. Pylance's syntax request timed out; Python compilation and
  execution succeeded. The memory graph was regenerated, and the universe
  state, active selection, and checkpoint hashes matched their pre-run values.

## 0.31.0 - 2026-11-15

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Added an isolated joint next-word search that marginalizes up to 20
  preceding-word candidates across five or 20 next-word paths. Exposed
  shared initial and contextual probability weights while preserving the
  existing greedy comparison. No next initial is supplied.
- On the 244,430 four-letter targets where the earlier preceding-word
  generator stopped, the best 20-path word scored 4.78% versus 4.17%
  for the prior mixture; its top-five candidate coverage was 11.48%.
  Across all 1,402,110 matched pairs, exact-word accuracy was 8.21% versus
  8.15%, but initial accuracy was 15.47% versus 15.89%.
- Both arms receive the preceding word's true initial and assume it is at
  most four letters. Non-ASCII-letter words are excluded, sentence
  punctuation may be crossed, words can recur across the split, and mapping
  calibration included broader corpus frequencies. Approved hash-pinned
  private inputs were used; active weights and the universe were unchanged.
- Fifty-three focused Python tests, three pattern tests, the version tests,
  Python compilation, and the production build passed. The memory graph
  was regenerated.

## 0.30.0 - 2026-11-15

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Added an isolated next-initial and word-generation comparison that
  supplies only the preceding word's true initial. The 5/20-path arms
  combine finished preceding-word candidates to predict the next initial,
  then use that prediction to select context and generate the next word.
  The existing greedy arm and a known-previous-word oracle provide controls.
- On the 1,402,110 held-out pairs where greedy generation stopped, the
  20-path mixture predicted initials at 15.89% versus 14.76% greedy and
  15.14% most-common; complete-word accuracy was 8.15% versus 6.87%.
  For four-letter targets, complete-word accuracy was 4.17% versus 3.82%.
  Longer targets count as incorrect in full-pair word scores.
- Previous words are assumed to be at most four letters, and their true
  initials are supplied; the next initial is not. ASCII-only filtering,
  sentence-punctuation crossings, repeated words across splits, and mapping
  calibration on broader source frequencies limit interpretation. Approved
  hash-pinned inputs were used; active weights and the universe were unchanged.
- Forty-nine focused Python tests, three pattern tests, the version tests,
  Python compilation, and the production build passed. The memory graph
  was regenerated.

## 0.29.0 - 2026-11-15

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Added an isolated 5- and 20-path preceding-word beam comparison, including
  paths that end or continue beyond four characters. Finished candidates
  can be re-ranked using the supplied current-word initial. It uses the
  previously approved, hash-pinned private inputs and held-out windows.
- On the 749,650 windows where greedy generation stopped, the 20-path
  re-ranked beam improved exact preceding-word accuracy from 35.36% to
  42.82%, current-word accuracy from 29.69% to 34.28%, and following-initial
  accuracy from 8.19% to 9.57%. For four-letter current words, the matched
  score rose from 13.01% to 15.86%; the five-path beam scored 16.39%.
- Evaluation assumes the preceding word is at most four letters. Both true
  first letters are supplied, and the current initial is used in re-ranking,
  not predicted. Common words can recur across the split; non-ASCII-letter
  words are excluded, sentence punctuation may be crossed, and mapping
  calibration included broader corpus frequencies. Active weights and the
  live universe were unchanged.
- Forty-five focused Python tests, three pattern tests, the version tests,
  Python compilation, and the production build passed. The memory graph
  was regenerated.

## 0.28.0 - 2026-11-15

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Added an isolated short-word generator comparison using the 64 most
  frequent training-only preceding words and a ten-observation unconditional
  prior. It compares actual versus generated preceding words against the
  existing prefix generator on identical held-out three-word windows.
- On 873,368 windows with the current word's true initial supplied, the
  actual preceding word improved complete-word accuracy from 30.85% to
  43.21%; on four-letter targets, from 12.45% to 22.50%. Next-initial
  accuracy on all windows was 11.61% versus 12.11%.
- The preceding word generated from its true initial stopped on 85.83%
  of windows and was exact on 35.36% of that subset. On the same subset,
  generated-context versus unconditional current-word accuracy was
  29.69% versus 30.48%, and next-initial accuracy 8.19% versus 11.41%.
  Both first letters are supplied; this is not unaided generation.
- The test uses approved hash-pinned private inputs, excludes non-ASCII
  words, and may cross sentence punctuation. Repeated words may cross the
  document/book split, and mapping calibration used broader corpus
  frequencies. No active weights or live universe were changed.
- Forty-two focused Python tests, three pattern tests, the version tests,
  Python compilation, and the production build passed. The memory graph
  was regenerated.

## 0.27.0 - 2026-11-15

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Added an isolated next-initial comparison after completed one- to
  four-letter words. The known-word arm receives the actual prior word;
  the generated arm receives its true initial and uses the existing
  prefix generator and stop decision. Both are refitted solely on the
  approved, hash-pinned training inputs.
- On 1,604,277 held-out pairs, the known-word arm scored 20.42% versus a
  14.51% global baseline. The generator stopped within four letters on
  87.40% of pairs and scored 14.76% of those; unavailable predictions
  counted as incorrect yield 12.90% overall. On the stopped subset, the
  known-word arm scored 20.14% and baseline 15.14%.
- For 460,982 pairs following actual four-letter words, the known-word
  arm scored 22.28%; the generator stopped in 77.99% and scored 12.85%
  of those. ASCII-only token filtering, repeated words crossing splits,
  sentence-punctuation crossings, and mapping calibration on broader source
  frequencies limit interpretation. No active weights or live universe
  were changed.
- Thirty-eight focused Python tests, three pattern tests, the version tests,
  Python compilation, and the production build passed. The memory graph
  was regenerated.

## 0.26.0 - 2026-11-15

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Added an isolated pattern-family prefix ranker and stop decision for
  prefixes one through four characters long. Missing prefixes fall back to
  same-position last-letter counts, then position-wide counts. It uses the
  previously approved, hash-pinned private inputs and retains its queryable
  derived prefix report outside Git.
- On held-out occurrences, the top next letter or stop was correct 54.86%
  after two letters and 70.95% after three; same-position last-letter
  baselines scored 33.30% and 35.94%. From only the actual first letter,
  greedy generation produced and correctly ended 64,325/473,497 four-letter
  occurrences (13.59%). Correct prefixes were supplied for next-step scores,
  but not after the first letter in generation.
- The test excludes non-ASCII-letter words, including contractions;
  repeated words can appear across the split, and the pattern mapping saw
  corpus-wide initial frequencies. The isolated experiment did not change
  active weights or the live universe.
- Thirty-six focused Python tests, three pattern tests, the version tests,
  Python compilation, and the production build passed. The private `str`
  lookup returned `a`, `e`, and `u` as its top three choices. The memory
  graph was regenerated.

## 0.25.0 - 2026-11-15

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Added an isolated, queryable second-letter ranker using fixed pattern
  families and the previously approved, hash-pinned Bible/OANC inputs.
  Pattern variants share a letter family; missing first-letter families
  fall back to the training-wide ranking. The private report lists all 52
  case-sensitive possibilities without publishing corpus text.
- On 2,781,847 held-out word occurrences, the top guess scored 42.56%
  and top three 73.62%; training-wide baselines scored 15.42% and 42.31%.
  This is occurrence-weighted and teacher-forced; repeated words may cross
  splits, and the pattern mapping saw corpus-wide initial frequencies.
  One-letter words, word endings, full-word generation and quark bonding
  are not tested. No active weights or live universe were changed.
- Thirty-three focused Python tests, three pattern tests, the version check,
  the private report lookup, and the production build passed. The memory
  graph was regenerated. Pylance's syntax-check request timed out.

## 0.24.0 - 2026-11-15

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Added an isolated short-meal comparison against the frozen active
  conversation model. The sidecar uses 41 pairs from ten sentences in the
  previously approved exact-hash user corpus; neither neural weights nor the
  live universe are changed. Its fixed pattern labels are reversible to
  ordinary initials and unknown contexts preserve the model's prior.
- On 44 newly written diagnostic word starts, the model scored 3/44,
  sidecar 6/44, and a most-common-meal-initial baseline 3/44. Probabilities
  are grouped by first output character, including nonletters. These small,
  teacher-forced results do not prove generalization, chat quality, or a
  quark-specific benefit. Mapping calibration used broader corpus frequencies.
- Twenty-nine focused Python tests, three pattern tests, the version check,
  Python compilation, and the production build passed. Pylance syntax-check
  requests timed out.

## 0.23.0 - 2026-11-15

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Added a private, standalone word-start experiment using preceding-word
  counts and a fixed many-pattern-per-letter encoding. It does not modify
  the live universe, chat weights, or memory. OANC documents are separated
  by content hash and Bible learning books by book; old locked Bible books
  are not used. Pattern selection and decoding are deterministic.
- The user approved private use of both sources, but this declaration is not
  legal clearance. The OANC archive and derived mapping stay outside Git.
  Mapping construction previously saw initial frequencies in the test text,
  so the evaluation is held out from predictor fitting, not mapping creation.
  Near-duplicate documents may cross the split.
- On 2,906,608 test word starts the contextual count baseline scored 21.10%
  versus 13.73% for always guessing `t`. Encoding then decoding gave the
  identical answers by construction, not a pattern-learning improvement.
  Twenty-six focused universe and experiment tests, three pattern tests,
  version tests, and the production build passed.

## 0.22.0 - 2026-11-15

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Automatic runs now process up to 50 recorded phases per batch, targeting
  50 phases per second. The graph refreshes and Pause takes effect between
  batches; each batch is saved atomically and stops at the end of the last
  available prediction cycle. Individual stepping and old runs remain supported.
- On a copy of the user's run, a browser test with real Python subprocesses
  measured batch-start intervals of 1.321, 1.000, and 1.000 seconds, with no
  overlapping requests. It reached zero energy in the predict phase; a separate
  in-flight Pause test completed one batch and started no more. These are not
  guarantees for larger histories or slower systems. Twenty-four universe
  tests, three pattern tests, version tests, Python diagnostics, and the
  production build passed.
  The live universe was not advanced for the tests.

## 0.21.1 - 2026-11-15

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: patch

### Changes

- Automatic runs now target four phase steps per second without overlapping
  requests. Slow responses and larger histories can reduce the actual rate.
- A simulated browser run with 80 ms responses measured 250 ms between
  eight step requests, finished at zero energy in the predict phase, and
  confirmed that pausing an in-flight request starts no further steps.
  The 22 Python universe tests, three pattern tests, version tests, and
  production build passed. The user's live universe was not used for the
  browser test.

## 0.21.0 - 2026-11-15

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Moved the three-quark experiment into the main universe graph and its
  persisted step history. New runs cap groups at three, perform one internal
  round, and freeze candidates; older saved runs retain their unrestricted
  rules. Positions separate groups visually, not through simulated physics.
- Twenty-two universe tests, three isolated comparison tests, version checks,
  and the production build passed,
  including candidate separation, frozen quarks, save/replay, and corrupt
  history. Python diagnostics were clear. A simulated browser verified the
  spark-only main graph and a later candidate with both bond widths.
- Saved and verified the previous live run as "Before three-quark main reset
  - 2026-11-15", then reset to the spark with 24 starting energy (23 after
  the spark). A live read confirmed one seeder and no bonds. Candidates are
  not letters, do not bond to each other, and do not affect prediction or
  training.

## 0.20.0 - 2026-11-15

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Added a separate, fixed-seed three-quark experiment. Eligible groups can
  join only up to size three; each complete group gets one internal round
  and then freezes. Its graph displays separate candidates and incomplete
  groups, with chain and triangle shapes and existing bond colors.
- Three focused JavaScript tests, 20 original universe tests, version checks,
  and the production build passed. A
  read-only live-browser check found seven candidates, one incomplete group,
  no overlapping group boxes, and correctly weighted bond lines. The main
  universe state was unchanged. This is a fixed sample with decorative
  spacing, not physical repulsion, assigned letters, letter-to-letter bonds,
  or training; it uses a separate random stream from the main universe.

## 0.19.1 - 2026-11-15

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: patch

### Changes

- Made one-way bond lines 1 px and two-way lines 0.5 px, retaining both
  visibility and the strength distinction. A read-only live browser check
  confirmed both widths across the existing bond colors; build and version
  tests passed. The saved universe and bond calculations were not changed.

## 0.19.0 - 2026-11-15

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Every pop-out tries to connect on each phase, excluding itself and the
  seeder. Attempts and their outcomes are saved, existing bonds stay intact,
  and older snapshots continue under the new rules. The graph displays all
  recorded bonds and counts current two-quark bond types as observations,
  not letters or learned probabilities.
- Twenty universe tests, version checks, and the production build passed,
  including repeated attempts, existing bonds, old history, and save/replay;
  Pylance reported no syntax errors or Python diagnostics. Simulated browser checks covered updated graph/counts and
  expandable attempts; a read-only live check kept the state hash unchanged.
  An in-memory 100-reveal run recorded 20,000 attempts and about 2.2 MB of
  JSON: larger runs can become slow and storage-heavy. This does not yet
  model letter formation, letter probabilities, or bond evolution.

## 0.18.1 - 2026-11-15

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: patch

### Changes

- Made quarks smaller and scattered their positions deterministically for
  display only. Replaced arrowheads with solid lines whose widths show the
  recorded strength in one direction; exact direction remains in the tooltip
  and activity log. No partner choices or learning calculations changed.
- Sixteen universe tests, version checks, and the production build passed. A simulated browser
  checked 1,001 quarks, all bond colors and widths, absent arrowheads, stable
  positions after adding a quark and reloading, and follow/manual scrolling.
  The live state hash remained unchanged. This does not implement physical
  movement, bond evolution, or arrow-based accessibility.

## 0.18.0 - 2026-11-15

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Unified the Universe Lab graph around individual quarks and recorded
  directional bonds. The first pop-out floats; each later one selects an
  earlier pop-out uniformly, never the seeder, and selects between the four
  approved equally likely starting outcomes. Same-bit bonds are supported.
- Choices and the random seed persist and replay with saved experiments.
  Earlier quarks remain present without fabricated historical bonds. Bond
  evolution, attraction, repulsion, and bond-based prediction are not yet
  implemented; the existing seeder and predictor remain separate.
- Sixteen isolated universe tests passed, including first-floater behavior,
  all four outcomes, same-bit bonds, corrupt bonds, and save/replay; four
  earlier-lab tests, Pylance checks, build, and version tests passed. A
  simulated browser checked 90 distinct quarks, colors and directions,
  adding a quark, and following the newest one. A read-only live check
  showed the 24 existing quarks without invented bonds; the real run was
  not advanced.

## 0.17.2 - 2026-11-15

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: patch

### Changes

- Made connection updates easier to see with counts on the arrows and a
  description of the latest revealed connection and next phase. Counts
  still change only when the seeder reveals a bit; no learning rules changed.
- A copy of the user's run verified that the next reveal advances `0 -> 1`
  from 11 to 12. A simulated browser verified the displayed connection and
  arrow counts change on reveal, not measurement; a read-only live check
  verified the new display. Build, version tests, and ten universe tests
  passed. The live universe was not stepped by these checks.

## 0.17.1 - 2026-11-15

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: patch

### Changes

- Placed shared step, run, and pause controls beside both Universe Lab
  visualizations while retaining the existing controls at the top. All
  copies operate the same current run.
- The build, version test, and ten isolated universe tests passed. A
  simulated browser API verified stepping, cross-panel pause, and
  zero-energy controls; a read-only live check confirmed both new button
  groups were enabled. The live universe was not advanced by testing.

## 0.17.0 - 2026-11-15

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Added a directional connection view for observed bit transitions. Each
  reveal increases the appropriate direction's count; predictions do not
  count. Observations are reconstructed from event history, preserving old
  saves and the existing state and predictor. Arrow thickness has a visual
  cap; the displayed counts do not.
- The connections describe the supplied seeder's history, not learned
  weights; the predictor does not use them. Letter and later stages remain
  unimplemented.
- Ten isolated universe tests and four earlier-lab tests passed; the build,
  version tests, and Pylance syntax/diagnostic checks passed. A simulated
  browser API showed all four connection states and repeated counts.
  A read-only live check showed seven transitions in each direction and the
  same original state-file fingerprint; no live run was reset or advanced.

## 0.16.0 - 2026-11-15

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Added local named Universe Lab saves, safe reset and restore choices, and
  confirmed deletion. Complete mid-cycle state is saved; stale browser
  revisions are rejected. Saved copies are Git-ignored, not backed up.
- Eight isolated universe tests and four earlier-lab tests passed, including
  protected replacement, stale requests, archive integrity, and mid-cycle
  restoration. A browser test with a simulated API checked the controls and
  confirmation cancellation. Pylance reported no syntax errors or
  diagnostics for the universe code and tests; build and version tests passed.
- After the user-approved restart, both services responded and the updated
  API reported the original live universe unchanged: 58 events, zero energy,
  and `predict` next. No destructive tests used that universe. The browser
  interaction test used a simulated API, not the live experiment.

## 0.15.0 - 2026-11-15

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Added an isolated Universe Lab in the Base Model Lab view. A spark creates
  the first `1`; a user-designed seeder supplies growing batches of `01`
  pairs. A small from-scratch predictor shows its prediction before each
  reveal, then its error and weight update as separate steps. Energy is
  finite, adjustable, and persisted along with the event history.
- The new page shows connected bit events, plain-language source
  annotations, and the actual activity log. The earlier base teaching trace
  remains accessible; the active chat model and memory are not used by the
  experiment. The supplied seeder is a toy environment, not evidence of
  spontaneous emergence or language generation.
- Four universe tests and four earlier-lab tests passed; the build, version
  test, and Pylance syntax/diagnostic checks passed. An isolated browser
  test verified spark, each learning phase, graph, pause, and saved-state
  reload. After the approved app restart, both services responded, the
  experiment remained empty, and chat remained on round 39.

## 0.14.0 - 2026-11-15

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Added per-answer retention gates to immediate corrections and permanent,
  active-authored, and corpus continuation. An improved overall score no
  longer hides a previously correct approved answer becoming incorrect.
  An explicitly corrected input supersedes its old target; unrelated
  approved examples remain protected. The correction report records
  before/after retention and rejected attempts retain their evidence.
- The feedback API timeout is ten minutes because retention evaluation
  adds generation work. No alphabet, corpus item, or spacing encoding was
  changed. The safeguard protects exact approved answers, not arbitrary
  paraphrases or every teacher-forced corpus position.
- In an isolated replay from round 38, the earlier `Name?` → `Mira`
  correction was rejected because it lost 30 previously correct approved
  answers; the isolated parent remained unchanged. This check is not
  retroactive: the live correction checkpoint remains active.
- All 29 training and correction tests and 12 chat/context tests passed;
  the production build and version test passed. The corpus hash remained
  unchanged. After the user approved a restart, both local services were
  responsive and the chat API still selected round 39.

## 0.13.1 - 2026-11-15

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: patch

### Changes

- With user approval, raised only the immediate chat-feedback saturation
  limit from 0.6 to 0.65; ordinary corpus-training limits remain unchanged.
  Successful correction reports record the measured saturation and limit.
- Three focused tests passed, including acceptance at 0.65 and rejection
  immediately above it. The submitted `Name?` → `Mira` correction replayed,
  activated in a separate round 39, and returned `Mira` from live weights;
  its correction-set saturation was 0.6424. The previously rejected corpus
  candidate remains rejected, and round 38 is the rollback selection.
- All 14 immediate-correction tests passed, as did the production build and
  version check/test. The chat API reported the correction checkpoint active.
- The higher limit does not guarantee retention of other answers, nor does
  an exact response establish generalization. Correction and corpus
  saturation are measured over different sets.
- A later baseline compared round 39's active correction checkpoint with
  round 38 on the same recorded cases: teaching fell 232/279 to 203/279,
  development 29/64 to 26/64, and prior feedback 11/23 to 5/23. Five
  exploratory name prompts returned `Mira` or `Mira.`; these prompts were
  evaluation only, not added to training.
- An isolated 0.01-rate corpus continuation from round 39 made 6,100
  updates in 60.75 update-CPU seconds and replayed exactly. Teaching rose
  204/280 to 236/280, development 26/64 to 29/64, complete pairs 10/32 to
  12/32, feedback 6/24 to 10/24, and teacher-forced corpus positions
  214/536 to 286/536. It was rejected because saturation reached 0.6091
  and the previously correct `Name?` answer changed from `Mira` to `Mira.`.
  The isolated trial did not change live weights; these reused cases and
  training positions are not independent generalization evidence.

## 0.13.0 - 2026-11-15

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Expanded the approved corpus with the user's new text, the approved
  correction from “runds” to “runs,” three fragments, and seven historical
  Layer 0 seeds with source-boundary metadata. Layer 0 itself remains
  unchanged; the marker `</>` is not taught as ordinary output text.
- Added a case-sensitive three-identical-character output constraint for
  live neural chat, including whitespace and whole-word emissions.
  Continuation and correction evaluations use the same rule; corrections
  containing such runs fail explicitly. The original comparison control is
  unchanged.
- The chat, active training, permanent training, and conversation context
  suites passed 24 tests; two focused correction tests passed. Real live-model
  generations for three prompts had no triple repeats. Pylance reported no
  diagnostics for the changed core modules checked. The production build
  and version test passed.
- This constrains output rather than proving learned grammar. The 30 corpus
  records have 12 conflicting teacher-forced input groups across records;
  no individual record had a conflict. The training data is not an
  independent generalization test.
- The expanded-corpus continuation made 6,200 updates in 60.28 update-CPU
  seconds and replayed exactly, but was rejected: the 1,200-update hidden
  saturation was 0.607 (limit 0.6), teaching fell 227/279 to 222/279, and
  a previously correct feedback answer was lost. Feedback fell 10/23 to
  9/23; development rose 29/64 to 32/64 and paired development 13 to 14.
  Teacher-forced corpus accuracy rose 75/536 to 147/536 after vocabulary
  expansion, not a free-running answer test. The fox/wizards answer remained
  incorrect, and round 34 stayed live.
- After those losses were disclosed, the user requested training the
  rejected candidate in. Both milestones, complete weights, development,
  and feedback outputs independently replayed; isolated promotion and
  rollback passed. The specific candidate was then activated as round 35
  under a separate user-authorization record, preserving the original
  rejected report and round 34 as rollback. This does not relax later
  continuation safeguards.
- A guarded repeat made 5,800 updates in 60.99 update-CPU seconds and
  replayed exactly, but was rejected for saturation, teaching and paired
  development regressions, and losing a previously correct feedback answer.
  Teaching fell 222/279 to 219/279, development 32/64 to 29/64, and
  development pairs 14 to 11; feedback increased 9/23 to 10/23 overall.
  Teacher-forced corpus accuracy improved 147/536 to 194/536, but the
  fox/wizards response remained wrong. Round 35 stayed live.
- After these regressions were disclosed, the user authorized promoting
  that specific rejected repeat. Both milestones, complete weights,
  development, and feedback replayed independently; isolated promotion and
  rollback passed. It became round 36, retaining the original rejection
  and round 35 rollback.
- The next guarded corpus continuation made 6,500 updates in 60.64
  update-CPU seconds, replayed exactly, passed the normal safeguards, and
  activated as round 37 with round 36 as rollback. Teaching improved
  219/279 to 227/279, approved feedback 10/23 to 11/23, and teacher-forced
  corpus positions 194/536 to 227/536. Reused development remained 29/64
  with 11/32 fully correct pairs. The fox/wizards answer changed from
  `Orare.` to `Sere.`, still incorrect. The corpus score measures training
  positions, not free-running generalization; evaluation and replay take
  additional elapsed time.
- The next guarded candidate made 6,500 updates in 60.17 update-CPU
  seconds and replayed exactly, but was rejected because final hidden
  saturation reached 0.6025 (limit 0.6). Teaching rose 227/279 to 232/279
  and teacher-forced corpus positions 227/536 to 245/536; development
  remained 29/64, fully correct pairs improved 11/32 to 12/32, and feedback
  remained 11/23. The fox/wizards prediction remained `Sere.`. Round 37
  stayed active with round 36 as rollback.
- The user then explicitly authorized activation despite the disclosed
  saturation. Both milestones, complete weights, development, and feedback
  independently replayed; isolated promotion and rollback passed. The
  candidate was activated as round 38, preserving its original rejection,
  all 23 approved records, and round 37 as rollback. Future training keeps
  the normal safeguards.
- A further guarded run made 6,200 updates in 60.82 update-CPU seconds
  and replayed exactly, but was rejected for saturation (0.6157), teaching
  regression, and loss of previously correct feedback. Teaching fell
  232/279 to 228/279, development 29/64 to 28/64, and feedback 11/23 to
  10/23, with three previously correct feedback answers lost. Complete
  development pairs remained 12/32 and teacher-forced corpus positions
  rose 245/536 to 266/536. The fox/wizards prediction changed from `Sere.`
  to `Sure.`, still incorrect. Round 38 stays active, with round 37 rollback.

## 0.12.0 - 2026-11-14

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Added the six user-supplied pangram, alphabet, and digit items as a
  provenance-recorded, review-gated corpus. The separate corpus trainer
  uses a fixed input-only text boundary, expands approved character,
  identity, and gap symbols, and mixes corpus positions with approved
  conversation rehearsal. Ordinary continuation does not automatically
  rehearse the corpus.
- A focused test verified review-hash enforcement, distinguishable digit
  positions, deterministic updates, activation, rejection, and rollback.
  Pylance reported no diagnostics for the corpus training module. Corpus
  measurements are training-set results, not independent generalization
  evidence; user approval is not legal clearance.
- The active-training and conversation-context suites passed 16 tests, and
  the production build and version test passed.
- The 60-second corpus continuation made 6,000 updates and replayed exactly,
  but was rejected: teaching fell 227/279 to 223/279, development pairs
  fell 13 to 12, and a previously correct feedback answer was lost.
  Individual development exact answers rose 29/64 to 30/64; feedback stayed
  10/23 overall. Teacher-forced corpus token accuracy rose 15/181 to 85/181
  after alphabet expansion, which measures training examples rather than
  free-running recall. Round 34 remained live and the rejected report was
  retained.

## 0.11.1 - 2026-11-14

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: patch

### Changes

- Fixed a measured same-input/different-target collision in the explicitly
  reviewed fox/wizards pair by adding two partial-word-length context slots.
  The new connections begin at zero; previous checkpoints still restore.
  Lengths through 4095 are represented; longer partial words fail explicitly.
- In isolated and live migration checks, the pair's conflicting inputs
  decreased from one to zero. Teaching remained 212/256, development 32/64,
  and approved feedback 8/23, with no changed predictions. Migration replay
  and isolated rollback passed; the live previous checkpoint was preserved.
  The migration itself performed no training updates.
- The context, active training, permanent training, and chat suites passed
  22 tests; the correction suite passed 12 tests. Pylance reported no
  diagnostics for the changed migration, context, training, and correction
  modules. A focused correction test also passed on the new 130-slot
  checkpoint. The production build and version test passed; the running chat
  API reported the migrated checkpoint.
- A guarded continuation from round 33 activated as round 34 after 6,300
  updates and 60.07 update-CPU seconds, with exact weight and milestone
  replay. Teaching improved 220/279 to 227/279 and feedback 8/23 to 10/23;
  individual development exact answers fell 32/64 to 29/64, while fully
  correct development pairs rose 12 to 13. The fox/wizards answer remained
  incorrect. The gate checks paired development accuracy, not the individual
  exact count, and protects previously correct approved feedback.
- This fixes the measured representation conflict, not every possible
  collision or a guarantee of producing the requested answer. Development
  examples are reused and are not an independent generalization test.

## 0.11.0 - 2026-11-14

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Added a reviewed, replayable structural migration for the 32 approved
  word identities, uppercase `Q` and `X`, and learnable space-count features
  for the represented word gaps. No new examples are approved or trained by
  this migration. Previous weights and token IDs are retained and rollback
  is preserved.
- Space counts are distinct through 4095 and larger gaps fail explicitly.
  The feature does not distinguish all arrangements of spaces, tabs, and
  punctuation and does not guarantee learned spacing or word meaning.
- A synthetic correction containing two spaces after a question mark trained
  and replayed on the migrated architecture. The initial 50-call timing
  probe showed about 1.7 times the parent inference CPU. Four focused
  migration, correction, and activation/rollback tests passed. In an isolated
  copy of round 25, exact answers stayed 179/256 teaching, 26/64 development,
  and 3/19 feedback, with no changed predictions. Migration activation and
  rollback both worked in that copy. The development cases are reused.
- The production build and version tests passed, as did focused chat,
  continuation, correction, and context tests. Pylance reported no
  diagnostics for the changed core files.
- A later 108-character correction regression failed on the migrated
  architecture within the existing 30-second update-CPU budget; the
  original architecture learned the same synthetic answer in 5,250 updates.
  The first live migration attempt was stopped before promotion, leaving
  active chat and progress matched at round 25 during investigation.
  With the update-CPU budget raised to 60 seconds in an isolated test, the
  migrated model learned the same answer in 4,900 updates and replayed it.
  A masked-output calculation optimization alone did not resolve the
  30-second failure and was removed.
- With user approval, the new spacing architecture now has a 60-second
  update-CPU budget (older checkpoints retain 30), and the feedback API
  timeout is four minutes. The local app was restarted and both backend
  and frontend responded. Longer correction fitting remains unguaranteed.
- After 32 targeted regression tests and the isolated HTTP route test passed,
  the migration's live checks again preserved 179/256 teaching, 26/64
  development, and 3/19 feedback answers, with no changed predictions.
  It activated as round 26 without optimizer updates. Active and permanent
  checkpoint hashes match; the round-25 rollback selection and progress
  checkpoint verified, and the latest approved correction reproduced exactly.
  The restarted backend reports round 26 and the frontend responds.

## 0.10.0 - 2026-11-14

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Raised the conversation correction limit to 319 characters and generation
  limit to 320; evaluation leaves room for the newline on longer answers.
  Longer corrections use learning rate .06 rather than .03.
- Added a backward-compatible, checkpointed spacing representation when a
  correction has conflicting training inputs, preserving the old embeddings
  at migration and rejecting unresolved conflicts explicitly.
- Preserved full contextual feedback hashes in later authored continuation.
- The user's 35-character correction was successfully trained and replayed
  in an isolated checkpoint copy, after its whitespace collision was removed.
  In that copy, previously approved feedback stayed 2/18 exact and the
  existing development examples changed from 25/64 to 26/64. A synthetic
  108-character correction also trained, replayed and generated exactly.
  These checks do not prove general fitting, retention, or chat accuracy.
- Validation: 27 targeted neural, continuation, and isolated HTTP tests
  passed, along with the production build, version check, graph-template
  check, and Pylance diagnostics for the changed core files. After the app
  service stopped, it was restored and the previously submitted correction
  was retried through the local API. It trained in 1,750 updates, replayed
  exactly, and activated as round 25; the active and progress checkpoints
  match and inference with the original context reproduces the exact answer.

## 0.9.0 - 2026-11-14

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Added opt-in experimental follow-up context for the active conversation
  model. It supplies the preceding visible exchange, not retrieved memories;
  single-turn chat remains the default.
- Saved the exact model input turns as response provenance and included them
  in explicitly approved feedback training, review hashes, and replay.
  Existing single-turn feedback retains its original hash and behavior.
- The current model is not a grounded assistant and context is not guaranteed
  to improve its replies. On 16 previously inspected multi-turn development
  cases, the active checkpoint scored 10/16 with full context and 12/16 with
  only the final turn. Four additional context-dependent paired examples scored
  5/8 with context and 0/8 without; neither set establishes general ability.
- Validation: 14 focused neural-chat, correction, and isolated HTTP tests
  passed, as did six training-continuation tests and the production build.
  An isolated browser mock displayed the option off by default and showed
  the contextual-reply notice after a follow-up; it did not use live memory.
  With the user's approval, restarted the development app and verified that
  the live backend advertises the capability and the checkbox appears. No live
  weights or training inputs were changed by these checks.

## 0.8.0 - 2026-11-14

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Made within-galaxy connectors more visible and colorful without moving
  ordinary cross-galaxy links in front of fills. Strengthened the simulated
  per-galaxy depth of memories with radial lighting and size scaling.
  Graph membership, saved search, and learned weights are unchanged.
- Made the additional AI Memory strength gradient slightly darker and
  independent of age and display depth; the existing depth lighting remains.
  Selected, recalled, important, and search-matched
  memories retain their existing emphasis. Strength comes from the existing
  learned score, which reflects decay and feedback.
- Strengthened the main galaxy shells' highlights, rim, and shading without
  covering internal memories or connections.
- Made base memory sizes relative to each galaxy using learned-score rank
  and total connection-strength rank, while retaining the depth and
  consolidation effects. This changes presentation only.
- Made ordinary links within all galaxies fade with their connection
  strength, while preserving created and recalled highlights and existing
  cross-galaxy behavior. This does not remove or change stored links.
- Added AI Memory candidate previews without changing their membership, and
  replaced session-based promotion gates with five distinct user contributions
  plus cohesion. Candidate evidence combines topic similarity and current
  stored connection strengths. Session provenance remains recorded.
- Removed candidate preview borders and moved them closer to AI Memory, with
  the same hover panel as baby galaxies. Reduced the Chat heading by 20px and
  navigation-button height by 15px at the measured desktop size.
- Added candidate-edge placement for unclaimed AI Memory messages connected
  only within the parent. The strongest candidate link chooses their placement;
  ties remain in the parent. They do not vote toward promotion, retain their
  links and parent provenance, and become sticky child members on promotion.
- Preserved a saved message-link strength when promotion changes its rendered
  kind between sequence and transition.
- Validation of candidate-edge behavior: 28 memory-cluster tests, the version
  check, JavaScript syntax check, and production build passed. A regeneration
  in a temporary copy retained transcript content, provenance, source links,
  saved node values and link strengths, query, and sidebar state. In the live
  graph two eligible messages appeared on candidate edges and retained AI
  Memory as their parent. No AI Memory child was promoted in this live check;
  promotion and sticky retention were exercised by tests instead. The existing
  child-parent link remains at strength 0.5, not a newly increased weight.
- Validation: 24 memory-cluster tests, the version check, JavaScript syntax
  check, and production build passed. The live graph rendered seven AI Memory
  candidate previews containing 22 memories, all retaining their parent
  domain; preview circles did not overlap other galaxy circles. Regeneration
  preserved transcript text, provenance, links, saved node values, link
  strengths, query, and sidebar state. After the latest appearance changes,
  all seven candidates have no dashed boundary, sit five graph units from the
  AI Memory boundary, and provide the shared hover panel; hovering a candidate
  opens its member list and leaving closes it. At the observed desktop width,
  the title changed from 57.5px to 37.5px and navigation buttons from 34px to
  19px. These checks do not confirm the user's visual preference. Previously, in the live graph,
  AI Memory's 10th-to-90th percentile base-radius spread changed from
  4.49-5.82 to 3.40-6.44 graph units; 367 AI Memory nodes and nine galaxy
  layout groups were rendered in that check. A later live check showed
  5,184 ordinary internal links with strength-based opacity (median 0.33),
  and the created/recalled highlight was unaffected. AI Memory's radius was
  171.13 out of 180 graph units. The rendered graph matched its template
  while retaining its embedded graph data. These checks confirm rendering
  behavior, not the user's visual preference.

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
