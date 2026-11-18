"""Train an isolated character checkpoint on one explicitly reviewed new meal."""

import argparse
import copy
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from character_context_trial import (
    MEAL_SHA256, ROUND, ROUND_REPORT_SHA256, assessment_summary, character_examples,
    character_model, score_cases, text_cases,
)
from neural_base_comparison import digest
from neural_chat import CONVERSATION_MODEL, load_active
from neural_conversation_meal import AUTHORED_FIXTURE, LARGE_FIXTURE, validate_curriculum
from neural_conversation_trial import assess, canonical, prompt_for
from neural_corpus_training import APPROVED_SHA256, CORPUS, corpus_records
from neural_plasticity import PlasticityContext
from neural_training import load_progress
from neural_word_context import WORD
from original_generator import atomic_json
from second_letter_experiment import BIBLE_MANIFEST_SHA256, OANC_SHA256, require_hash
from word_start_experiment import documents


def train_new_meal(model, examples, *, updates, seed, rate, offset=0):
    if (not examples or type(updates) is not int or updates < 1
            or type(offset) is not int or offset < 0):
        raise ValueError("A nonempty meal, positive update count and nonnegative offset are required")
    rng = random.Random(seed)
    for _ in range(offset):
        rng.randrange(len(examples))
    for _ in range(updates):
        context, target, prior = examples[rng.randrange(len(examples))]
        model.model.update(context, target, output_prior=prior, learning_rate=rate)


def word_cases(rows, source):
    cases = []
    for row in rows:
        expected = [match.group() for match in WORD.finditer(row["answer"])][:2]
        if not expected:
            raise ValueError("Evaluation answer has no words")
        cases.append((source, prompt_for(row), expected))
    return cases


def run(args):
    if not args.acknowledge_review:
        raise ValueError("Explicit approval of the exact meal hash is required")
    require_hash(args.meal, args.reviewed_sha256)
    meal_raw = args.meal.read_bytes()
    meal = json.loads(meal_raw)
    validate_curriculum(meal)
    require_hash(args.previous_run / "report.json", args.previous_report_sha256)
    previous_raw = (args.previous_run / "report.json").read_bytes()
    previous = json.loads(previous_raw)
    require_hash(args.previous_run / "trained.json", previous["trained_sha256"])
    parent_raw = (args.previous_run / "trained.json").read_bytes()
    parent = json.loads(parent_raw)
    continuing = previous["kind"] == "private_new_conversation_meal_trial"
    if not previous["validation"] or not all(previous["validation"].values()) or parent["output_mode"] != "character_only":
        raise ValueError("Expected a validated character-only checkpoint")
    if continuing:
        if args.origin_run is None:
            raise ValueError("Continuation requires the original character meal run")
        base_hash = previous.get("base_report_sha256", previous["parent_report_sha256"])
        require_hash(args.origin_run / "report.json", base_hash)
        base_raw = (args.origin_run / "report.json").read_bytes()
        base = json.loads(base_raw)
        if (previous["training"]["meal_sha256"] != args.reviewed_sha256
                or not previous["training"]["new_meal_only"]):
            raise ValueError("Continuation requires the same reviewed meal and training protocol")
    else:
        base_hash = args.previous_report_sha256
        base = previous
    if (base["kind"] != "private_character_context_trial"
            or not base["validation"] or not all(base["validation"].values())
            or base["training"].get("meal_sha256") != MEAL_SHA256):
        raise ValueError("Expected the validated original character meal checkpoint")
    require_hash(ROUND / "report.json", ROUND_REPORT_SHA256)
    training = json.loads((ROUND / "report.json").read_bytes())
    inputs = json.loads((ROUND / "inputs.json").read_bytes())
    fixture_raw = (ROOT / AUTHORED_FIXTURE).read_bytes()
    corpus_raw = (ROOT / CORPUS).read_bytes()
    records = corpus_records(inputs["corpus"])
    require_hash(ROOT / LARGE_FIXTURE, MEAL_SHA256)
    old_meal_raw = (ROOT / LARGE_FIXTURE).read_bytes()
    old_meal = json.loads(old_meal_raw)
    if (digest(fixture_raw) != APPROVED_SHA256 or inputs["curriculum"] != json.loads(fixture_raw)
            or digest(corpus_raw) != training["review"]["corpus_file_sha256"]
            or inputs["corpus"] != json.loads(corpus_raw)
            or digest(canonical({"conversation": inputs["teaching"], "corpus": records}))
            != training["review"]["train_sha256"]
            or base["training"]["reviewed_training_sha256"] != training["review"]["train_sha256"]):
        raise ValueError("Previous reviewed input evidence does not match")
    old_rows = inputs["teaching"] + inputs["curriculum"]["test"] + old_meal["train"] + old_meal["test"]
    old_prompts = {canonical(row["turns"]) for row in old_rows}
    if any(canonical(row["turns"]) in old_prompts for row in meal["train"] + meal["test"]):
        raise ValueError("New meal overlaps previous conversation inputs or tests")
    protected_paths = (ROOT / ".data/neural-chat/active.json", ROOT / ".data/neural-training/progress.json",
                       ROOT / "universe-lab/state.json")
    protected = {path: path.read_bytes() for path in protected_paths}
    active_record, active, index = load_active(ROOT)
    progress, _ = load_progress(ROOT)
    if (active_record["model"] != CONVERSATION_MODEL or not isinstance(active, PlasticityContext)
            or index is not None or progress["checkpoint_sha256"] != active_record["checkpoint_sha256"]
            or base["data"]["active_parent_checkpoint_sha256"] != active_record["checkpoint_sha256"]):
        raise ValueError("Active parent and permanent progress do not match previous experiment")
    require_hash(args.bible_dir / "manifest.json", BIBLE_MANIFEST_SHA256)
    require_hash(args.oanc, OANC_SHA256)
    model = character_model(PlasticityContext.restore(copy.deepcopy(parent["state"])))
    examples = character_examples(model, [(prompt_for(row), row["answer"]) for row in meal["train"]])
    groups = {"conversation": word_cases(inputs["curriculum"]["test"], "conversation")}
    source_cases = text_cases(documents(args.bible_dir, args.oanc))
    groups.update({source: [case for case in source_cases if case[0] == source] for source in ("bible", "oanc")})
    if {name: digest(canonical(cases)) for name, cases in groups.items()} != base["case_sha256"]:
        raise ValueError("Previous evaluation cases have changed")
    assessments = {"old_teaching": inputs["teaching"], "old_development": inputs["curriculum"]["test"],
                   "old_meal_teaching": old_meal["train"], "old_meal_development": old_meal["test"],
                   "new_meal_teaching": meal["train"], "new_meal_development": meal["test"]}
    args.run_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    source_dir = args.run_dir / "source"
    source_dir.mkdir(mode=0o700)
    source_paths = [Path(__file__), ROOT / "universe-lab/character_context_trial.py"]
    source_paths += [ROOT / "universe-lab" / name for name in (
        "fresh_pair_comparison.py", "word_start_experiment.py", "second_letter_experiment.py")]
    source_paths += [ROOT / name for name in (
        "neural_base_comparison.py", "neural_chat.py", "neural_conversation_trial.py",
        "neural_conversation_context.py", "neural_corpus_training.py", "neural_experiment.py",
        "neural_plasticity.py", "neural_training.py", "neural_word_context.py")]
    sources = {}
    for path in source_paths:
        raw = path.read_bytes()
        (source_dir / path.name).write_bytes(raw)
        sources[path.name] = digest(raw)
    for name in previous["source_sha256"]:
        if name not in ("character_context_trial.py", "new_meal_trial.py") and name in sources and previous["source_sha256"][name] != sources[name]:
            raise ValueError(f"Previous evaluation dependency changed: {name}")
    if continuing and previous["source_sha256"]["character_context_trial.py"] != sources["character_context_trial.py"]:
        raise ValueError("Previous character evaluation dependency changed")
    atomic_json(args.run_dir / "initial.json", {"state": parent["state"], "output_mode": "character_only"})
    atomic_json(args.run_dir / "approved-meal.json", meal)
    initial = copy.deepcopy(model.snapshot())
    before_words = {name: score_cases(model, cases) for name, cases in groups.items()}
    if before_words != previous["after"]:
        raise ValueError("Prior word scores could not be reproduced")
    before_new_words = score_cases(model, word_cases(meal["test"], "new_meal"))
    before = {name: assess(model, rows, no_triple_repeat=True) for name, rows in assessments.items()}
    if continuing:
        if before_new_words != previous["new_meal_words"]["after"] or any(
                sum(row["exact"] for row in result["rows"]) != previous["retention"][name]["after_exact"]
                for name, result in before.items()):
            raise ValueError("Prior new meal or retention scores could not be reproduced")
        seed = previous["seed"]
        offset = previous.get("stream_offset", 0) + previous["additional_updates"]
    else:
        seed = previous["training_seed"] + previous["total_updates"]
        offset = 0
    print(f"Starting from {previous['total_updates']} updates; training {args.updates} on new meal only.", flush=True)
    train_new_meal(model, examples, updates=args.updates, seed=seed, rate=previous["rate"], offset=offset)
    final = copy.deepcopy(model.snapshot())
    replay = character_model(PlasticityContext.restore(copy.deepcopy(initial)))
    replay_examples = character_examples(replay, [(prompt_for(row), row["answer"]) for row in meal["train"]])
    if replay_examples != examples:
        raise AssertionError("Meal training examples changed during replay")
    train_new_meal(replay, replay_examples, updates=args.updates, seed=seed, rate=previous["rate"], offset=offset)
    if replay.snapshot() != final:
        raise AssertionError("New meal training failed exact replay")
    after_words = {name: score_cases(model, cases) for name, cases in groups.items()}
    after_new_words = score_cases(model, word_cases(meal["test"], "new_meal"))
    retention = {name: assessment_summary(before[name], assess(model, rows, no_triple_repeat=True))
                 for name, rows in assessments.items()}
    if (any(path.read_bytes() != content for path, content in protected.items())
            or args.meal.read_bytes() != meal_raw
            or (args.previous_run / "report.json").read_bytes() != previous_raw
            or (args.previous_run / "trained.json").read_bytes() != parent_raw
            or (continuing and (args.origin_run / "report.json").read_bytes() != base_raw)
            or (ROOT / AUTHORED_FIXTURE).read_bytes() != fixture_raw
            or (ROOT / CORPUS).read_bytes() != corpus_raw
            or (ROOT / LARGE_FIXTURE).read_bytes() != old_meal_raw):
        raise RuntimeError("Protected model or reviewed input changed during experiment")
    atomic_json(args.run_dir / "trained.json", {"state": final, "output_mode": "character_only"})
    report = {
        "kind": "private_new_conversation_meal_trial",
        "starting_updates": previous["total_updates"], "additional_updates": args.updates,
        "total_updates": previous["total_updates"] + args.updates, "rate": previous["rate"], "seed": seed,
        "stream_offset": offset, "base_report_sha256": base_hash,
        "training": {"new_meal_only": True, "positions": len(examples),
                     "meal_sha256": args.reviewed_sha256, "meal_train_sha256": digest(canonical(meal["train"])),
                     "meal_test_sha256": digest(canonical(meal["test"])),
                     "approval": "User approved the exact meal hash and selected new meal only."},
        "parent_report_sha256": args.previous_report_sha256,
        "parent_checkpoint_sha256": previous["trained_sha256"],
        "trained_sha256": digest((args.run_dir / "trained.json").read_bytes()),
        "source_sha256": sources,
        "validation": {"previous_scores_reproduced": True, "exact_replay": True, "protected_unchanged": True},
        "before": before_words, "after": after_words,
        "new_meal_words": {"before": before_new_words, "after": after_new_words},
        "retention": retention,
        "limitations": [
            "The new evaluation uses new names and objects but the same generated templates as training.",
            "Old conversation development cases are repeatedly reused and are not independent generalization evidence.",
            "Training used only the newly approved meal, with no rehearsal; exact replay is not an activation safeguard.",
            "Full activation and mid-round saturation safeguards were not run; no checkpoint was activated.",
            "Review approval is a user declaration, not legal clearance.",
        ],
    }
    atomic_json(args.run_dir / "report.json", report)
    print(json.dumps({"conversation_before": before_words["conversation"],
                      "conversation_after": after_words["conversation"],
                      "new_meal_words": report["new_meal_words"], "retention": retention}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meal", required=True, type=Path)
    parser.add_argument("--reviewed-sha256", required=True)
    parser.add_argument("--acknowledge-review", action="store_true")
    parser.add_argument("--previous-run", required=True, type=Path)
    parser.add_argument("--previous-report-sha256", required=True)
    parser.add_argument("--origin-run", type=Path,
                        help="Original character meal run when continuing a new-meal run")
    parser.add_argument("--bible-dir", required=True, type=Path)
    parser.add_argument("--oanc", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--updates", type=int, default=2900)
    args = parser.parse_args()
    if args.updates < 1:
        parser.error("--updates must be positive")
    run(args)


if __name__ == "__main__":
    main()
