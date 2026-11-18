"""Compare conversation and text contexts with isolated character-only training."""

import argparse
from collections import Counter
from copy import copy
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fresh_pair_comparison import training_rng
from neural_base_comparison import digest, generate
from neural_chat import CONVERSATION_MODEL, load_active
from neural_conversation_meal import AUTHORED_FIXTURE, LARGE_FIXTURE, validate_curriculum
from neural_conversation_trial import assess, canonical, lost_correct_ids, prompt_for
from neural_corpus_training import APPROVED_SHA256, CORPUS, CORPUS_RATE, corpus_records
from neural_plasticity import PlasticityContext, initialize
from neural_training import load_progress
from neural_word_context import WORD
from original_generator import atomic_json
from second_letter_experiment import BIBLE_MANIFEST_SHA256, OANC_SHA256, require_hash
from word_start_experiment import documents

ROUND = ROOT / ".data/neural-training/rounds/7a67711a46ef4a609a04d92d1b7ee2a1"
ROUND_REPORT_SHA256 = "ed2ea5e912e9ea0ad4a0871e631258fc059ec6716ed4cf641ef538b2b4afb202"
SOURCE_LIMIT = 32
MEAL_SHA256 = "3c8720173eda1b53b2f3cb7cd6916f6e12e2f65ebcb51bf19c2230965932c725"


def character_model(model):
    result = copy(model)
    result.set_output_mode("character_only")
    return result


def character_examples(model, records):
    examples = []
    for boundary, text in records:
        prefix = boundary
        for char in text + "\n":
            token = model.model.tokens.get("char:" + char)
            if token is None:
                raise ValueError(f"Training character is absent from the model: {char!r}")
            prior = model.prior(prefix)
            if not prior[token]:
                raise ValueError("Character target is excluded by the output mask")
            examples.append((model.context(prefix), token, prior))
            prefix += char
    if not examples:
        raise ValueError("Training examples cannot be empty")
    return examples


def text_cases(docs, limit=SOURCE_LIMIT):
    counts = Counter()
    cases = []
    for source, partition, text in docs:
        if partition != "test" or source not in ("bible", "oanc") or counts[source] >= limit:
            continue
        for segment in re.finditer(r"[^.!?\r\n]+", text):
            line = segment.group()
            words = list(WORD.finditer(line))
            if len(words) < 3:
                continue
            for target_count in (1, 2):
                targets = words[-target_count:]
                prompt = "text: " + line[:targets[0].start()]
                cases.append((source, prompt, [match.group() for match in targets]))
            counts[source] += 1
            break
        if all(counts[source] == limit for source in ("bible", "oanc")):
            break
    if any(counts[source] != limit for source in ("bible", "oanc")):
        raise ValueError(f"Not enough eligible held-out documents: {dict(counts)}")
    return cases


def generated_words(output):
    line, separator, _ = output.partition("\n")
    matches = list(WORD.finditer(line))[:2]
    return [(match.group(), match.end() < len(line) or bool(separator)) for match in matches]


def score_cases(model, cases):
    totals = Counter({key: 0 for key in (
        "cases", "no_second_target", "word_1_targets", "word_1_missing", "word_1_unfinished",
        "word_1_exact", "word_2_targets", "word_2_missing", "word_2_unfinished", "word_2_exact",
    )})
    letters = [Counter({"targets": 0, "correct": 0}) for _ in range(8)]
    for _, prompt, expected in cases:
        if not 1 <= len(expected) <= 2:
            raise ValueError("Each case must have one or two target words")
        prediction = generated_words(generate(model, prompt, length=64, no_triple_repeat=True)["text"])
        totals["cases"] += 1
        totals["no_second_target"] += len(expected) == 1
        for index, target in enumerate(expected):
            word = prediction[index][0] if index < len(prediction) else None
            complete = prediction[index][1] if index < len(prediction) else False
            label = f"word_{index + 1}"
            totals[label + "_targets"] += 1
            totals[label + "_missing"] += word is None
            totals[label + "_unfinished"] += word is not None and not complete
            totals[label + "_exact"] += word == target and complete
            wanted = [char for char in target if char.isalpha()]
            got = [char for char in word if char.isalpha()] if word is not None else []
            for position in range(min(4, len(wanted))):
                counts = letters[index * 4 + position]
                counts["targets"] += 1
                counts["correct"] += position < len(got) and got[position] == wanted[position]
    return {**dict(totals), "letters": {
        f"word_{i // 4 + 1}_letter_{i % 4 + 1}": dict(counts)
        for i, counts in enumerate(letters)
    }}


def train_characters(model, conversations, corpus, *, updates, seed, rate, conversation_only, offset=0):
    if type(offset) is not int or offset < 0:
        raise ValueError("Training offset must be a nonnegative integer")
    rng = training_rng(seed, 0 if conversation_only else offset, len(conversations), len(corpus))
    if conversation_only:
        for _ in range(offset):
            rng.random()
            rng.randrange(len(conversations))
    draws = Counter({"conversation": 0, "corpus": 0})
    for _ in range(updates):
        choose_corpus = rng.random() < CORPUS_RATE and not conversation_only
        pool = corpus if choose_corpus else conversations
        draws["corpus" if choose_corpus else "conversation"] += 1
        context, target, prior = pool[rng.randrange(len(pool))]
        model.model.update(context, target, output_prior=prior, learning_rate=rate)
    return dict(draws)


def train_meal(model, rehearsal, meal, *, updates, seed, rate, offset=0):
    if not rehearsal or not meal or type(offset) is not int or offset < 0:
        raise ValueError("Both reviewed meal and rehearsal require examples")
    rng = random.Random(seed)
    draws = Counter({"rehearsal": 0, "meal": 0})
    for _ in range(offset):
        pool = meal if rng.random() < 0.5 else rehearsal
        rng.randrange(len(pool))
    for _ in range(updates):
        chosen = "meal" if rng.random() < 0.5 else "rehearsal"
        pool = meal if chosen == "meal" else rehearsal
        draws[chosen] += 1
        context, target, prior = pool[rng.randrange(len(pool))]
        model.model.update(context, target, output_prior=prior, learning_rate=rate)
    return dict(draws)


def assessment_summary(before, after):
    return {"examples": before["examples"],
            "before_exact": sum(row["exact"] for row in before["rows"]),
            "after_exact": sum(row["exact"] for row in after["rows"]),
            "previously_correct_lost": len(lost_correct_ids(before, after))}


def run(args):
    bible_hash = require_hash(args.bible_dir / "manifest.json", BIBLE_MANIFEST_SHA256)
    oanc_hash = require_hash(args.oanc, OANC_SHA256)
    report_bytes = (ROUND / "report.json").read_bytes()
    if digest(report_bytes) != ROUND_REPORT_SHA256:
        raise ValueError("Reviewed training report hash mismatch")
    training = json.loads(report_bytes)
    inputs = json.loads((ROUND / "inputs.json").read_bytes())
    fixture = (ROOT / AUTHORED_FIXTURE).read_bytes()
    corpus = (ROOT / CORPUS).read_bytes()
    records = corpus_records(inputs["corpus"])
    if (training["status"] != "rejected" or digest(fixture) != APPROVED_SHA256
            or inputs["curriculum"] != json.loads(fixture)
            or digest(corpus) != training["review"]["corpus_file_sha256"]
            or inputs["corpus"] != json.loads(corpus)
            or digest(canonical({"conversation": inputs["teaching"], "corpus": records}))
            != training["review"]["train_sha256"]):
        raise ValueError("Training inputs do not match the reviewed records")
    active_file = ROOT / ".data/neural-chat/active.json"
    progress_file = ROOT / ".data/neural-training/progress.json"
    universe_file = ROOT / "universe-lab/state.json"
    protected = {path: path.read_bytes() for path in (active_file, progress_file, universe_file)}
    active_record, active, index = load_active(ROOT)
    progress, _ = load_progress(ROOT)
    if (active_record["model"] != CONVERSATION_MODEL or not isinstance(active, PlasticityContext)
            or index is not None or active_record["checkpoint_sha256"] != progress["checkpoint_sha256"]
            or active_record["checkpoint_sha256"] != training["parent_checkpoint_sha256"]):
        raise ValueError("Active model and reviewed training parent do not match")
    fresh = initialize(active, "fresh_scaled", seed=17)
    if fresh.snapshot()["weights"] == active.snapshot()["weights"]:
        raise AssertionError("Fresh model did not receive new weights")
    previous = None
    offset = 0
    if args.previous_run:
        previous_report = args.previous_run / "report.json"
        require_hash(previous_report, args.previous_report_sha256)
        previous = json.loads(previous_report.read_bytes())
        offset = previous.get("total_updates", previous["updates"])
        if (previous["kind"] != "private_character_context_trial"
                or not all(previous["validation"].values())
                or type(offset) is not int or offset < 1
                or previous["training_seed"] != training["seed"]
                or previous["initialization_seed"] != 17 or previous["rate"] != training["rate"]
                or previous["output_mode"] != "character_only"
                or (previous["training"].get("meal_sha256") is not None
                    and (not args.meal_256 or previous["training"]["meal_sha256"] != MEAL_SHA256))
                or previous["training"]["conversation_probability"] != (1 if args.conversation_only else 1 - CORPUS_RATE)
                or previous["training"]["reviewed_training_sha256"] != training["review"]["train_sha256"]
                or previous["data"]["active_parent_checkpoint_sha256"] != active_record["checkpoint_sha256"]
                or previous["data"]["bible_manifest_sha256"] != bible_hash
                or previous["data"]["oanc_sha256"] != oanc_hash
                or previous["data"]["corpus_file_sha256"] != digest(corpus)
                or previous["data"]["fixture_sha256"] != digest(fixture)):
            raise ValueError("Previous checkpoint does not match the reviewed training protocol")
        previous_checkpoint = args.previous_run / "trained.json"
        require_hash(previous_checkpoint, previous["trained_sha256"])
        saved = json.loads(previous_checkpoint.read_bytes())
        if saved["output_mode"] != "character_only":
            raise ValueError("Previous checkpoint does not use character-only output")
        model = character_model(PlasticityContext.restore(saved["state"]))
    else:
        model = character_model(fresh)
    conversations = character_examples(model, [(prompt_for(row), row["answer"]) for row in inputs["teaching"]])
    corpus_examples = character_examples(model, records)
    meal_raw = None
    meal_data = None
    meal_examples = None
    if args.meal_256:
        meal_path = ROOT / LARGE_FIXTURE
        require_hash(meal_path, MEAL_SHA256)
        meal_raw = meal_path.read_bytes()
        meal_data = json.loads(meal_raw)
        validate_curriculum(meal_data)
        prior_answers = {canonical(row["turns"]): row["answer"] for row in inputs["teaching"]}
        prior_tests = {canonical(row["turns"]) for row in inputs["curriculum"]["test"]}
        if (any(canonical(row["turns"]) in prior_tests for row in meal_data["train"])
                or any(canonical(row["turns"]) in prior_answers
                       and prior_answers[canonical(row["turns"])] != row["answer"] for row in meal_data["train"])
                or any(canonical(row["turns"]) in prior_answers for row in meal_data["test"])):
            raise ValueError("Meal conflicts with approved rehearsal or contaminates evaluation")
        meal_examples = character_examples(model, [(prompt_for(row), row["answer"]) for row in meal_data["train"]])
    conversation_cases = []
    for row in inputs["curriculum"]["test"]:
        expected = [match.group() for match in WORD.finditer(row["answer"])][:2]
        if not expected:
            raise ValueError("Conversation evaluation answer has no words")
        conversation_cases.append(("conversation", prompt_for(row), expected))
    sources = text_cases(documents(args.bible_dir, args.oanc))
    groups = {"conversation": conversation_cases}
    groups.update({source: [case for case in sources if case[0] == source] for source in ("bible", "oanc")})
    args.run_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    source_dir = args.run_dir / "source"
    source_dir.mkdir(mode=0o700)
    source_files = [Path(__file__), ROOT / "universe-lab/fresh_pair_comparison.py",
                    ROOT / "universe-lab/word_start_experiment.py",
                    ROOT / "universe-lab/second_letter_experiment.py"]
    source_files.extend(ROOT / name for name in (
        "neural_base_comparison.py", "neural_chat.py", "neural_conversation_trial.py",
        "neural_conversation_context.py", "neural_corpus_training.py", "neural_experiment.py",
        "neural_plasticity.py", "neural_training.py", "neural_word_context.py",
    ))
    source_hashes = {}
    for path in source_files:
        raw = path.read_bytes()
        (source_dir / path.name).write_bytes(raw)
        source_hashes[path.name] = digest(raw)
    initial = json.loads(json.dumps(model.snapshot(), allow_nan=False))
    atomic_json(args.run_dir / "initial.json", {"state": initial, "output_mode": "character_only"})
    before = {name: score_cases(model, cases) for name, cases in groups.items()}
    case_hashes = {name: digest(canonical(cases)) for name, cases in groups.items()}
    if previous is not None and (
            previous["trained_sha256"] != digest((args.run_dir / "initial.json").read_bytes())
            or previous["case_sha256"] != case_hashes or previous["after"] != before
            or any(previous["source_sha256"][name] != value for name, value in source_hashes.items()
                   if name != Path(__file__).name)):
        raise ValueError("Previous checkpoint, scores, cases, or scoring dependencies differ")
    before_assessments = None
    if meal_data is not None:
        before_assessments = {
            "rehearsal": assess(model, inputs["teaching"], no_triple_repeat=True),
            "conversation_development": assess(model, inputs["curriculum"]["test"], no_triple_repeat=True),
            "meal_teaching": assess(model, meal_data["train"], no_triple_repeat=True),
            "meal_development": assess(model, meal_data["test"], no_triple_repeat=True),
        }
    comparison = None
    if args.comparison_report:
        require_hash(args.comparison_report, args.comparison_sha256)
        comparison = json.loads(args.comparison_report.read_bytes())
        if (comparison["kind"] != "private_character_context_trial"
                or not all(comparison["validation"].values())
                or comparison["updates"] != args.updates
                or comparison["training_seed"] != training["seed"]
                or comparison["initialization_seed"] != 17
                or comparison["rate"] != training["rate"]
                or comparison["output_mode"] != "character_only"
                or comparison["data"]["active_parent_checkpoint_sha256"] != active_record["checkpoint_sha256"]
                or comparison["training"]["reviewed_training_sha256"] != training["review"]["train_sha256"]
                or comparison["initial_sha256"] != digest((args.run_dir / "initial.json").read_bytes())
                or comparison["case_sha256"] != case_hashes or comparison["before"] != before
                or any(comparison["source_sha256"][name] != value for name, value in source_hashes.items()
                       if name != Path(__file__).name)):
            raise ValueError("Comparison does not match the initial model, inputs, cases, or scoring dependencies")
    print(f"Character model: {offset} prior updates, training {args.updates} more; "
          f"evaluating {sum(map(len, groups.values()))} cases.", flush=True)
    meal_start = (previous["training"].get("meal_start_updates", previous["starting_updates"])
                  if previous is not None and previous["training"].get("meal_sha256") else offset)
    meal_seed = (previous["training"]["meal_seed"]
                 if previous is not None and previous["training"].get("meal_sha256")
                 else training["seed"] + offset)
    if type(meal_start) is not int or not 0 <= meal_start <= offset:
        raise ValueError("Invalid meal training stream position")
    meal_offset = offset - meal_start
    if previous is not None and previous["training"].get("meal_sha256"):
        if (meal_data is None or meal_seed != training["seed"] + meal_start
                or previous["training"]["meal_probability"] != 0.5
                or previous["training"]["rehearsal_probability"] != 0.5
                or previous["training"]["meal_train_sha256"] != digest(canonical(meal_data["train"]))
                or previous["training"]["meal_test_sha256"] != digest(canonical(meal_data["test"]))):
            raise ValueError("Previous meal training protocol does not match")
    if meal_examples is not None:
        draws = train_meal(model, conversations, meal_examples, updates=args.updates,
                           seed=meal_seed, rate=training["rate"], offset=meal_offset)
    else:
        draws = train_characters(model, conversations, corpus_examples, updates=args.updates,
                                 seed=training["seed"], rate=training["rate"],
                                 conversation_only=args.conversation_only, offset=offset)
    final = json.loads(json.dumps(model.snapshot(), allow_nan=False))
    replay = character_model(PlasticityContext.restore(json.loads(json.dumps(initial))))
    replay_conversations = character_examples(replay, [(prompt_for(row), row["answer"]) for row in inputs["teaching"]])
    replay_corpus = character_examples(replay, records)
    if replay_conversations != conversations or replay_corpus != corpus_examples:
        raise AssertionError("Character training examples changed during replay")
    if meal_data is not None:
        replay_meal = character_examples(replay, [(prompt_for(row), row["answer"]) for row in meal_data["train"]])
        if replay_meal != meal_examples:
            raise AssertionError("Meal training examples changed during replay")
        replay_draws = train_meal(replay, replay_conversations, replay_meal, updates=args.updates,
                                  seed=meal_seed, rate=training["rate"], offset=meal_offset)
    else:
        replay_draws = train_characters(replay, replay_conversations, replay_corpus, updates=args.updates,
                                        seed=training["seed"], rate=training["rate"],
                                        conversation_only=args.conversation_only, offset=offset)
    if replay.snapshot() != final or replay_draws != draws:
        raise AssertionError("Character training did not replay exactly")
    after = {name: score_cases(model, cases) for name, cases in groups.items()}
    retention = None
    if meal_data is not None and before_assessments is not None:
        rows = {"rehearsal": inputs["teaching"], "conversation_development": inputs["curriculum"]["test"],
                "meal_teaching": meal_data["train"], "meal_development": meal_data["test"]}
        retention = {name: assessment_summary(before_assessments[name],
                                             assess(model, items, no_triple_repeat=True))
                     for name, items in rows.items()}
    if (any(path.read_bytes() != content for path, content in protected.items())
            or (ROOT / AUTHORED_FIXTURE).read_bytes() != fixture or (ROOT / CORPUS).read_bytes() != corpus
            or (ROUND / "report.json").read_bytes() != report_bytes):
        raise RuntimeError("Active state or reviewed training inputs changed")
    require_hash(args.bible_dir / "manifest.json", bible_hash)
    require_hash(args.oanc, oanc_hash)
    if meal_raw is not None:
        require_hash(ROOT / LARGE_FIXTURE, MEAL_SHA256)
    if args.comparison_report:
        require_hash(args.comparison_report, args.comparison_sha256)
    if previous is not None:
        require_hash(args.previous_run / "report.json", args.previous_report_sha256)
        require_hash(args.previous_run / "trained.json", previous["trained_sha256"])
    atomic_json(args.run_dir / "trained.json", {"state": final, "output_mode": "character_only"})
    report = {
        "kind": "private_character_context_trial",
        "updates": args.updates, "starting_updates": offset, "total_updates": offset + args.updates,
        "training_seed": training["seed"], "initialization_seed": 17,
        "rate": training["rate"], "output_mode": "character_only",
        "data": {"bible_manifest_sha256": bible_hash, "oanc_sha256": oanc_hash,
                 "corpus_file_sha256": digest(corpus), "fixture_sha256": digest(fixture),
                 "reviewed_round_report_sha256": ROUND_REPORT_SHA256,
                 "active_parent_checkpoint_sha256": active_record["checkpoint_sha256"]},
        "training": {"conversation_probability": 1 if args.conversation_only else 1 - CORPUS_RATE,
                     "corpus_probability": 0 if args.conversation_only else CORPUS_RATE,
                     "draws": draws, "reviewed_training_sha256": training["review"]["train_sha256"]},
        "cases": {"conversation": len(conversation_cases), "bible": len(groups["bible"]),
                  "oanc": len(groups["oanc"])},
        "case_sha256": case_hashes,
        "initial_sha256": digest((args.run_dir / "initial.json").read_bytes()),
        "trained_sha256": digest((args.run_dir / "trained.json").read_bytes()),
        "source_sha256": source_hashes,
        "validation": {"fresh_weights": True, "exact_replay": True, "protected_unchanged": True},
        "before": before, "after": after,
        "retention": retention,
        "comparison": {"report_sha256": args.comparison_sha256, "after": comparison["after"]}
        if comparison is not None else None,
        "previous": {"report_sha256": args.previous_report_sha256, "trained_sha256": previous["trained_sha256"]}
        if previous is not None else None,
        "limitations": [
            "One initialization and a short run cannot establish which context is generally better.",
            "Conversation cases reuse the approved fixture's development answers and are not independent generalization evidence.",
            "Text cases use the first eligible 32 held-out documents per source and select the final one and final two words from one sentence segment per document.",
            "One- and two-target text cases share source segments, and near-duplicate documents may cross the split.",
            "Prompt types and target distributions differ, so their accuracy scores are not an apples-to-apples causal comparison.",
            "Case-sensitive character scores cover the first four alphabetic letters in each target; missing output counts wrong.",
            "A target with no second word is excluded from all second-word denominators. An expected but missing second word counts wrong.",
            "The 64-character generation limit and first-line parsing can leave output unfinished; this is reported, not activation safety.",
            "The architecture and symbol inventory are reused, while all weights are fresh and training and generation select individual characters only.",
            "Historical and external data approval is not legal clearance; no checkpoint was activated.",
        ],
    }
    if meal_data is not None:
        report["training"].update(meal_sha256=MEAL_SHA256, meal_probability=0.5,
                                  rehearsal_probability=0.5, meal_seed=meal_seed,
                                  meal_start_updates=meal_start,
                                  meal_train_sha256=digest(canonical(meal_data["train"])),
                                  meal_test_sha256=digest(canonical(meal_data["test"])))
        report["limitations"].append(
            "The meal overlaps prior rehearsal, and both its development and the previous conversation cases are reused checks.")
    atomic_json(args.run_dir / "report.json", report)
    print(json.dumps({"cases": report["cases"], "before": before, "after": after}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bible-dir", required=True, type=Path)
    parser.add_argument("--oanc", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--updates", type=int, default=2900)
    parser.add_argument("--conversation-only", action="store_true",
                        help="Train only on reviewed conversation-answer characters")
    parser.add_argument("--comparison-report", type=Path)
    parser.add_argument("--comparison-sha256")
    parser.add_argument("--previous-run", type=Path)
    parser.add_argument("--previous-report-sha256")
    parser.add_argument("--meal-256", action="store_true",
                        help="Mix the approved 256-example meal evenly with approved rehearsal")
    args = parser.parse_args()
    if args.updates < 1:
        parser.error("--updates must be positive")
    if bool(args.comparison_report) != bool(args.comparison_sha256):
        parser.error("--comparison-report and --comparison-sha256 must be supplied together")
    if bool(args.previous_run) != bool(args.previous_report_sha256):
        parser.error("--previous-run and --previous-report-sha256 must be supplied together")
    if args.previous_run and args.comparison_report:
        parser.error("--previous-run and --comparison-report cannot be combined")
    if args.meal_256 and (not args.previous_run or not args.conversation_only):
        parser.error("--meal-256 requires --previous-run and --conversation-only")
    run(args)


if __name__ == "__main__":
    main()
