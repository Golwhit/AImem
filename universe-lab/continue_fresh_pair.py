"""Continue a reviewed, isolated fresh model and repeat the two-word tests."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from active_candidate_pair_experiment import evaluate_pairs, summary
from fresh_pair_comparison import train_and_replay
from neural_base_comparison import digest
from neural_chat import CONVERSATION_MODEL, load_active
from neural_conversation_meal import AUTHORED_FIXTURE, paired_scores
from neural_conversation_trial import assess, canonical, feedback_hash, lost_correct_ids
from neural_corpus_training import APPROVED_SHA256, CORPUS, corpus_records, corpus_score
from neural_plasticity import PlasticityContext, diagnostics
from neural_training import load_progress
from original_generator import atomic_json
from second_letter_experiment import BIBLE_MANIFEST_SHA256, OANC_SHA256, require_hash
from unsupplied_next_word_experiment import collect
from word_start_experiment import documents

CORE_SOURCES = (
    "neural_corpus_training.py", "neural_plasticity.py", "neural_experiment.py",
    "neural_conversation_trial.py", "neural_conversation_context.py",
    "neural_base_comparison.py", "neural_word_context.py", "seed_life.py",
)
TEST_SOURCES = (
    "active_candidate_pair_experiment.py", "active_three_word_evaluation.py",
    "unsupplied_next_word_experiment.py", "word_start_experiment.py",
)


def previous_details(report):
    validation = report["validation"]
    if report["kind"] == "private_fresh_vs_active_candidate_mixture":
        if not (validation["fresh_final_weights_replay_exact"] and validation["live_selection_unchanged"]):
            raise ValueError("Previous fresh-model run was not replay-validated and isolated")
        return (report["parent_checkpoint_sha256"], report["updates"],
                report["pair_scores"]["fresh"], report["retention"]["fresh"])
    if report["kind"] == "private_fresh_model_continuation":
        if not (validation["continued_weights_replay_exact"] and validation["starting_state_matches_saved_checkpoint"]
                and validation["previous_scores_reproduced"] and validation["live_selection_unchanged"]):
            raise ValueError("Previous continuation was not replay-validated and isolated")
        return (report["active_checkpoint_sha256"], report["total_updates"],
                report["continued"]["pair_scores"], report["continued"]["retention"])
    raise ValueError("Unsupported previous experimental run")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bible-dir", required=True, type=Path)
    parser.add_argument("--oanc", required=True, type=Path)
    parser.add_argument("--previous-run", required=True, type=Path)
    parser.add_argument("--previous-report-sha256", required=True)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--additional-updates", type=int, help="Defaults to the previous run's update count")
    args = parser.parse_args()
    require_hash(args.previous_run / "report.json", args.previous_report_sha256)
    previous = json.loads((args.previous_run / "report.json").read_text())
    previous_active, offset, previous_scores, previous_retention = previous_details(previous)
    hashes = {
        "bible_manifest_sha256": require_hash(args.bible_dir / "manifest.json", BIBLE_MANIFEST_SHA256),
        "oanc_sha256": require_hash(args.oanc, OANC_SHA256),
        "candidate_sha256": require_hash(args.candidate, previous["data"]["candidate_sha256"]),
        "training_report_sha256": require_hash(
            args.candidate.parent / "report.json", previous["data"]["training_report_sha256"]),
        "comparison_sha256": require_hash(args.comparison, previous["data"]["comparison_sha256"]),
        "previous_report_sha256": args.previous_report_sha256,
        "parent_checkpoint_sha256": require_hash(args.previous_run / "trained.json", previous["trained_sha256"]),
    }
    if any(hashes[key] != previous["data"][key]
           for key in ("bible_manifest_sha256", "oanc_sha256", "candidate_sha256",
                       "training_report_sha256", "comparison_sha256")):
        raise ValueError("Inputs differ from the previous experiment")
    sources = [ROOT / name for name in CORE_SOURCES] + [ROOT / "universe-lab" / name for name in TEST_SOURCES]
    for path in sources:
        if digest(path.read_bytes()) != previous["source_sha256"][path.name]:
            raise ValueError(f"Experiment source changed since the previous run: {path.name}")
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
            or training["parent_checkpoint_sha256"] != active_record["checkpoint_sha256"]
            or previous_active != active_record["checkpoint_sha256"]
            or progress["checkpoint_sha256"] != active_record["checkpoint_sha256"]
            or comparison["active_checkpoint_sha256"] != active_record["checkpoint_sha256"]
            or not training["validation"]["full_weights_and_milestones_replay_exact"]
            or not training["validation"]["test_replay_exact"]
            or previous["training_seed"] != training["seed"] or previous["rate"] != training["rate"]):
        raise ValueError("Active parent and original reviewed training protocol must match")
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
    last_round_updates = previous.get("additional_updates", offset)
    updates = last_round_updates if args.additional_updates is None else args.additional_updates
    if type(offset) is not int or offset < 1 or type(updates) is not int or updates < 1:
        raise ValueError("Previous and additional update counts must be positive")
    parent_state = json.loads((args.previous_run / "trained.json").read_text())["state"]
    model = PlasticityContext.restore(json.loads(json.dumps(parent_state, allow_nan=False)))
    args.run_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    source_dir = args.run_dir / "source"
    source_dir.mkdir(mode=0o700)
    source_hashes = {}
    for path in [Path(__file__), ROOT / "universe-lab" / "fresh_pair_comparison.py", *sources]:
        raw = path.read_bytes()
        (source_dir / path.name).write_bytes(raw)
        source_hashes[path.name] = hashlib.sha256(raw).hexdigest()
    atomic_json(args.run_dir / "protocol.json", {
        "status": "running", "data": hashes, "training_sha256": training["review"]["train_sha256"],
        "starting_updates": offset, "additional_updates": updates, "seed": previous["training_seed"],
        "rate": previous["rate"], "sources": source_hashes,
    })
    print(f"Continuing the saved {offset}-update model for {updates} more updates; replay follows.", flush=True)
    initial, state, conversations, corpus, update_cpu = train_and_replay(
        model, inputs["teaching"], records, updates=updates, seed=previous["training_seed"],
        rate=previous["rate"], offset=offset)
    if initial != parent_state:
        raise AssertionError("Continuation did not start at the saved experimental checkpoint")
    atomic_json(args.run_dir / "initial.json", {"state": initial})
    atomic_json(args.run_dir / "trained.json", {"state": state})
    print("Continuation replayed exactly; scoring the same pairs and retention checks.", flush=True)
    _, _, test_pairs, _ = collect(doc for doc in documents(args.bible_dir, args.oanc) if doc[1] == "test")
    scores = evaluate_pairs(model, test_pairs)
    if any(scores[key]["pairs"] != prior["pairs"]
           for key, prior in previous_scores.items()):
        raise ValueError("Held-out pair counts differ from the previous run")
    parent = PlasticityContext.restore(json.loads(json.dumps(parent_state, allow_nan=False)))
    before_teaching = assess(parent, inputs["teaching"], no_triple_repeat=True)
    before_feedback = assess(parent, feedback, no_triple_repeat=True)
    if (summary(before_teaching) != previous_retention["teaching"]
            or summary(before_feedback) != previous_retention["feedback"]):
        raise AssertionError("Previous experimental checkpoint did not reproduce its recorded scores")
    after_teaching = assess(model, inputs["teaching"], no_triple_repeat=True)
    after_test = assess(model, inputs["curriculum"]["test"], no_triple_repeat=True)
    after_feedback = assess(model, feedback, no_triple_repeat=True)
    results = {
        "teaching": summary(after_teaching),
        "development": summary(after_test),
        "development_pairs": paired_scores(after_test)["both_exact"],
        "feedback": summary(after_feedback),
        "previously_correct_teaching_lost": len(lost_correct_ids(before_teaching, after_teaching)),
        "previously_correct_feedback_lost": len(lost_correct_ids(before_feedback, after_feedback)),
        "corpus": corpus_score(model, corpus),
        "diagnostics": diagnostics(model, conversations + corpus),
    }
    if (active_path.read_bytes() != before_active or progress_path.read_bytes() != before_progress
            or (ROOT / AUTHORED_FIXTURE).read_bytes() != fixture or (ROOT / CORPUS).read_bytes() != corpus_file):
        raise RuntimeError("The live model, progress, or reviewed training inputs changed during evaluation")
    require_hash(args.previous_run / "report.json", args.previous_report_sha256)
    require_hash(args.previous_run / "trained.json", previous["trained_sha256"])
    report = {
        "kind": "private_fresh_model_continuation",
        "data": hashes,
        "active_checkpoint_sha256": active_record["checkpoint_sha256"],
        "training_seed": previous["training_seed"], "rate": previous["rate"],
        "starting_updates": offset, "additional_updates": updates, "total_updates": offset + updates,
        "update_cpu_seconds": update_cpu,
        "initial_sha256": digest((args.run_dir / "initial.json").read_bytes()),
        "trained_sha256": digest((args.run_dir / "trained.json").read_bytes()),
        "source_sha256": source_hashes,
        "validation": {"continued_weights_replay_exact": True, "starting_state_matches_saved_checkpoint": True,
                       "previous_scores_reproduced": True, "live_selection_unchanged": True},
        "previous": {"pair_scores": previous_scores, "retention": previous_retention},
        "continued": {"pair_scores": scores, "retention": results},
        "mixture": {"pair_scores": comparison["pair_scores"]["mixture"],
                    "retention": comparison["retention"]["mixture"]},
        "limitations": [
            "The experimental model remains isolated and no activation safeguards or mid-round saturation checks were run.",
            "Only one initialization and one continuing update schedule were tested.",
            "Training reused the same reviewed corpus and approved feedback, not new examples.",
            "The actual first word is at most four letters and its first initial is supplied; held-out pairs can cross punctuation and near duplicates can cross the split.",
            "Previously used teaching, development and feedback cases are not independent generalization evidence.",
            "The earlier mixture has much longer inherited training and contains a rejected branch with unresolved saturation.",
        ],
    }
    atomic_json(args.run_dir / "report.json", report)
    print(json.dumps({"previous_pairs": previous_scores["combined"],
                      "previous_retention": previous_retention,
                      "continued_pairs": scores["combined"], "continued_retention": results,
                      "mixture_pairs": comparison["pair_scores"]["mixture"]["combined"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
