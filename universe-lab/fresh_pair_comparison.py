"""Train a fresh compatible model and compare it with the isolated mixture."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from active_candidate_pair_experiment import evaluate_pairs, summary
from neural_base_comparison import digest
from neural_chat import CONVERSATION_MODEL, load_active
from neural_conversation_meal import AUTHORED_FIXTURE, paired_scores
from neural_conversation_trial import answer_examples, assess, canonical, feedback_hash, lost_correct_ids
from neural_corpus_training import APPROVED_SHA256, CORPUS, CORPUS_RATE, advance, corpus_examples, corpus_records, corpus_score
from neural_plasticity import PlasticityContext, diagnostics, initialize
from neural_training import load_progress
from original_generator import atomic_json
from second_letter_experiment import BIBLE_MANIFEST_SHA256, OANC_SHA256, require_hash
from unsupplied_next_word_experiment import collect
from word_start_experiment import documents

INITIALIZATION_SEED = 17


def training_rng(seed, offset, conversation_count, corpus_count):
    if (type(seed) is not int or type(offset) is not int or offset < 0
            or conversation_count < 1 or corpus_count < 1):
        raise ValueError("Invalid training random-stream position")
    rng = random.Random(seed)
    for _ in range(offset):
        count = corpus_count if rng.random() < CORPUS_RATE else conversation_count
        rng.randrange(count)
    return rng


def train_and_replay(model, teaching, records, *, updates, seed, rate, offset=0):
    if (type(updates) is not int or updates < 1 or type(seed) is not int
            or type(offset) is not int or offset < 0 or not math.isfinite(rate) or rate <= 0):
        raise ValueError("Invalid training protocol")
    initial = json.loads(json.dumps(model.snapshot(), allow_nan=False))
    conversations = answer_examples(model, teaching)
    corpus = corpus_examples(model, records)
    if not conversations or not corpus:
        raise ValueError("Training requires both reviewed conversation and corpus examples")
    start = time.process_time()
    advance(model, conversations, corpus, training_rng(seed, offset, len(conversations), len(corpus)), updates, rate)
    update_cpu_seconds = time.process_time() - start
    state = json.loads(json.dumps(model.snapshot(), allow_nan=False))
    replayed = PlasticityContext.restore(json.loads(json.dumps(initial, allow_nan=False)))
    replay_conversations = answer_examples(replayed, teaching)
    replay_corpus = corpus_examples(replayed, records)
    if replay_conversations != conversations or replay_corpus != corpus:
        raise AssertionError("Training examples changed during replay")
    advance(replayed, replay_conversations, replay_corpus,
            training_rng(seed, offset, len(conversations), len(corpus)), updates, rate)
    if replayed.snapshot() != state:
        raise AssertionError("Fresh model weights did not replay exactly")
    return initial, state, conversations, corpus, update_cpu_seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bible-dir", required=True, type=Path)
    parser.add_argument("--oanc", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--training-report-sha256", required=True)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--comparison-sha256", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--updates", type=int, help="Override the matched continuation's update count")
    args = parser.parse_args()
    hashes = {
        "bible_manifest_sha256": require_hash(args.bible_dir / "manifest.json", BIBLE_MANIFEST_SHA256),
        "oanc_sha256": require_hash(args.oanc, OANC_SHA256),
        "candidate_sha256": require_hash(args.candidate, args.candidate_sha256),
        "training_report_sha256": require_hash(args.candidate.parent / "report.json", args.training_report_sha256),
        "comparison_sha256": require_hash(args.comparison, args.comparison_sha256),
    }
    training = json.loads((args.candidate.parent / "report.json").read_text())
    inputs = json.loads((args.candidate.parent / "inputs.json").read_text())
    comparison = json.loads(args.comparison.read_text())
    active_path = ROOT / ".data/neural-chat/active.json"
    progress_path = ROOT / ".data/neural-training/progress.json"
    before_active = active_path.read_bytes()
    before_progress = progress_path.read_bytes()
    active_record, active, index = load_active(ROOT)
    progress, _ = load_progress(ROOT)
    if (training["status"] != "rejected" or active_record["model"] != CONVERSATION_MODEL
            or index is not None or not isinstance(active, PlasticityContext)
            or active.snapshot()["plasticity_arm"] != "plain"
            or training["parent_checkpoint_sha256"] != active_record["checkpoint_sha256"]
            or progress["checkpoint_sha256"] != active_record["checkpoint_sha256"]
            or comparison["active_checkpoint_sha256"] != active_record["checkpoint_sha256"]
            or comparison["candidate_run_id"] != training["run_id"]
            or comparison["data"] != {key: hashes[key] for key in comparison["data"]}
            or not training["validation"]["full_weights_and_milestones_replay_exact"]
            or not training["validation"]["test_replay_exact"]):
        raise ValueError("Active parent, replay-validated candidate, and comparison must match")
    fixture = (ROOT / AUTHORED_FIXTURE).read_bytes()
    corpus_file = (ROOT / CORPUS).read_bytes()
    records = corpus_records(inputs["corpus"])
    feedback = [item["row"] for item in progress["approved_corrections"]]
    if (digest(fixture) != APPROVED_SHA256 or inputs["curriculum"] != json.loads(fixture)
            or digest(corpus_file) != training["review"]["corpus_file_sha256"]
            or inputs["corpus"] != json.loads(corpus_file)
            or digest(canonical({"conversation": inputs["teaching"], "corpus": records}))
            != training["review"]["train_sha256"]
            or any(item["sha256"] != feedback_hash(item["row"]["turns"], item["row"]["answer"])
                   for item in progress["approved_corrections"])):
        raise ValueError("Training inputs or feedback do not match review evidence")
    reference_updates = training["checkpoints"][-1]["updates"]
    updates = reference_updates if args.updates is None else args.updates
    if updates < 1:
        raise ValueError("Update count must be positive")
    seed, rate = training["seed"], training["rate"]
    if json.loads((args.candidate.parent / "initial.json").read_text())["state"] != active.snapshot():
        raise ValueError("Candidate initial state does not match the active parent")
    fresh = initialize(active, "fresh_scaled", seed=INITIALIZATION_SEED)
    if fresh.snapshot()["weights"] == active.snapshot()["weights"]:
        raise AssertionError("Fresh initialization did not replace the active weights")
    if (answer_examples(fresh, inputs["teaching"]) != answer_examples(active, inputs["teaching"])
            or corpus_examples(fresh, records) != corpus_examples(active, records)):
        raise AssertionError("Fresh and inherited models do not have matching training examples")
    args.run_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    source_files = (
        Path(__file__), ROOT / "neural_corpus_training.py", ROOT / "neural_plasticity.py",
        ROOT / "neural_experiment.py", ROOT / "neural_conversation_trial.py",
        ROOT / "neural_conversation_context.py", ROOT / "neural_base_comparison.py",
        ROOT / "neural_word_context.py", ROOT / "seed_life.py",
        ROOT / "universe-lab" / "active_candidate_pair_experiment.py",
        ROOT / "universe-lab" / "active_three_word_evaluation.py",
        ROOT / "universe-lab" / "unsupplied_next_word_experiment.py",
        ROOT / "universe-lab" / "word_start_experiment.py",
    )
    source_dir = args.run_dir / "source"
    source_dir.mkdir(mode=0o700)
    source_hashes = {}
    for path in source_files:
        raw = path.read_bytes()
        (source_dir / path.name).write_bytes(raw)
        source_hashes[path.name] = hashlib.sha256(raw).hexdigest()
    atomic_json(args.run_dir / "protocol.json", {
        "status": "running", "data": hashes, "parent_checkpoint_sha256": active_record["checkpoint_sha256"],
        "training_sha256": training["review"]["train_sha256"], "initialization_seed": INITIALIZATION_SEED,
        "seed": seed, "rate": rate, "updates": updates, "sources": source_hashes,
    })
    print(f"Training fresh model for {updates} updates on the reviewed data; exact replay follows.", flush=True)
    initial, state, conversations, corpus, update_cpu = train_and_replay(
        fresh, inputs["teaching"], records, updates=updates, seed=seed, rate=rate)
    atomic_json(args.run_dir / "initial.json", {"state": initial})
    atomic_json(args.run_dir / "trained.json", {"state": state})
    print("Fresh weights replayed exactly; scoring held-out pairs and retention.", flush=True)
    _, _, test_pairs, _ = collect(doc for doc in documents(args.bible_dir, args.oanc) if doc[1] == "test")
    fresh_scores = evaluate_pairs(fresh, test_pairs)
    if fresh_scores["combined"]["pairs"] != comparison["pair_scores"]["mixture"]["combined"]["pairs"]:
        raise ValueError("Held-out pair counts differ from the earlier comparison")
    fresh_teaching = assess(fresh, inputs["teaching"], no_triple_repeat=True)
    fresh_test = assess(fresh, inputs["curriculum"]["test"], no_triple_repeat=True)
    fresh_feedback = assess(fresh, feedback, no_triple_repeat=True)
    measurements = {
        "teaching": summary(fresh_teaching),
        "development": summary(fresh_test),
        "development_pairs": paired_scores(fresh_test)["both_exact"],
        "feedback": summary(fresh_feedback),
        "active_correct_teaching_not_matched": len(lost_correct_ids(training["before"]["training"], fresh_teaching)),
        "active_correct_feedback_not_matched": len(lost_correct_ids(training["before"]["feedback"], fresh_feedback)),
        "corpus": corpus_score(fresh, corpus),
        "diagnostics": diagnostics(fresh, conversations + corpus),
    }
    if active_path.read_bytes() != before_active or progress_path.read_bytes() != before_progress:
        raise RuntimeError("The live model or training progress changed during evaluation")
    report = {
        "kind": "private_fresh_vs_active_candidate_mixture",
        "data": hashes,
        "parent_checkpoint_sha256": active_record["checkpoint_sha256"],
        "initialization_seed": INITIALIZATION_SEED,
        "training_seed": seed,
        "rate": rate,
        "updates": updates,
        "reference_continuation_updates": reference_updates,
        "update_cpu_seconds": update_cpu,
        "initial_sha256": digest((args.run_dir / "initial.json").read_bytes()),
        "trained_sha256": digest((args.run_dir / "trained.json").read_bytes()),
        "source_sha256": source_hashes,
        "validation": {"fresh_final_weights_replay_exact": True, "live_selection_unchanged": True},
        "pair_scores": {"fresh": fresh_scores, **comparison["pair_scores"]},
        "retention": {"fresh": measurements, **comparison["retention"]},
        "limitations": [
            "The fresh model reuses the active model's architecture, symbol inventory and lexicon but starts with independently initialized weights.",
            "The sample order matches the rejected continuation up to the shorter run's update count; the active model has a much longer prior training history.",
            "One initialization and one training budget cannot establish whether a fresh model is ultimately better.",
            "The mixture is an untrained equal-weight blend containing a candidate that failed saturation; no model was promoted.",
            "The same previously used development and feedback cases are not independent generalization evidence.",
            "The first actual word is at most four letters and its first initial is supplied; held-out pairs may cross punctuation and near duplicates may cross the split.",
            "Only the final fresh weights were replayed and final saturation measured; activation safeguards were not run.",
        ],
    }
    atomic_json(args.run_dir / "report.json", report)
    print(json.dumps({"fresh_pairs": fresh_scores["combined"], "fresh_retention": measurements,
                      "mixture_pairs": comparison["pair_scores"]["mixture"]["combined"],
                      "mixture_retention": comparison["retention"]["mixture"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
