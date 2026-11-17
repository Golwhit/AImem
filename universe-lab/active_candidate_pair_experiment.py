"""Compare active, rejected, and combined models on two-word continuations."""

import argparse
from collections import Counter
from copy import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from active_three_word_evaluation import OUTPUT_LIMIT, words_from_output
from neural_base_comparison import WordBase, digest, generate
from neural_chat import CONVERSATION_MODEL, load_active
from neural_conversation_meal import AUTHORED_FIXTURE, paired_scores
from neural_conversation_trial import assess, canonical, lost_correct_ids
from neural_corpus_training import APPROVED_SHA256, corpus_records
from neural_plasticity import PlasticityContext
from neural_training import load_progress
from second_letter_experiment import BIBLE_MANIFEST_SHA256, LETTERS, OANC_SHA256, require_hash
from unsupplied_next_word_experiment import collect
from word_prefix_experiment import MAX_PREFIX
from word_start_experiment import documents


class MeanBackend:
    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.tokens = left.tokens
        self.context = left.context
        self.size = left.size

    def forward(self, context, output_prior=None):
        left, hidden, left_probabilities = self.left.forward(context, output_prior=output_prior)
        right_probabilities = self.right.forward(context, output_prior=output_prior)[2]
        if (len(left_probabilities) != len(right_probabilities)
                or any(not math.isfinite(value) or value < 0
                       for value in left_probabilities + right_probabilities)):
            raise ArithmeticError("Mixture branches returned invalid probabilities")
        return left, hidden, [(a + b) / 2 for a, b in zip(left_probabilities, right_probabilities)]


def mixture(active, candidate):
    if (not isinstance(active, WordBase) or not isinstance(candidate, WordBase)
            or type(active) is not type(candidate)
            or type(active.model) is not type(candidate.model)
            or active.model.characters != candidate.model.characters
            or active.model.context != candidate.model.context
            or active.lexicon != candidate.lexicon
            or active.gap_encoding != candidate.gap_encoding
            or active.payloads != candidate.payloads):
        raise ValueError("Mixture requires compatible model architectures, symbols, and payloads")
    combined = copy(active)
    combined.model = MeanBackend(active.model, candidate.model)
    return combined


def evaluate_pairs(model, pairs):
    predictions = {
        initial: words_from_output(initial, generate(
            model, "text: " + initial, length=OUTPUT_LIMIT, no_triple_repeat=True)["text"])
        for initial in LETTERS
    }
    results = {}
    for source, items in pairs.items():
        groups = {source: Counter(), "combined": results.setdefault("combined", Counter()),
                  "short_second_word_only": results.setdefault("short_second_word_only", Counter()),
                  "four_letter_second_word_only": results.setdefault("four_letter_second_word_only", Counter())}
        for (first, second), count in items.items():
            predicted = predictions[first[0]]
            first_word = predicted[0][1] if predicted else None
            second_initial, second_word = predicted[1] if len(predicted) > 1 else (None, None)
            keys = [source, "combined"]
            if len(second) <= MAX_PREFIX:
                keys.append("short_second_word_only")
            if len(second) == MAX_PREFIX:
                keys.append("four_letter_second_word_only")
            for key in keys:
                scores = groups[key]
                scores["pairs"] += count
                scores["first_word_correct"] += count * (first_word == first)
                scores["second_initial_available"] += count * (second_initial is not None)
                scores["second_initial_correct"] += count * (second_initial == second[0])
                scores["second_word_completed"] += count * (second_word is not None)
                scores["second_word_correct"] += count * (second_word == second)
                scores["both_words_correct"] += count * (first_word == first and second_word == second)
                scores["second_initial_correct_when_first_correct"] += count * (
                    first_word == first and second_initial == second[0])
        results[source] = groups[source]
    if not results.get("combined", {}).get("pairs"):
        raise ValueError("No held-out word pairs")
    return {key: dict(value) for key, value in sorted(results.items())}


def summary(assessment):
    return {"examples": assessment["examples"],
            "exact": sum(row["exact"] for row in assessment["rows"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bible-dir", required=True, type=Path)
    parser.add_argument("--oanc", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--training-report-sha256", required=True)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    hashes = {
        "bible_manifest_sha256": require_hash(args.bible_dir / "manifest.json", BIBLE_MANIFEST_SHA256),
        "oanc_sha256": require_hash(args.oanc, OANC_SHA256),
        "candidate_sha256": require_hash(args.candidate, args.candidate_sha256),
        "training_report_sha256": require_hash(args.candidate.parent / "report.json", args.training_report_sha256),
    }
    training = json.loads((args.candidate.parent / "report.json").read_text())
    inputs = json.loads((args.candidate.parent / "inputs.json").read_text())
    active_path = ROOT / ".data/neural-chat/active.json"
    progress_path = ROOT / ".data/neural-training/progress.json"
    before_active = active_path.read_bytes()
    before_progress = progress_path.read_bytes()
    active_record, active, index = load_active(ROOT)
    progress, _ = load_progress(ROOT)
    if (training["status"] != "rejected" or active_record["model"] != CONVERSATION_MODEL
            or index is not None or not isinstance(active, PlasticityContext)
            or training["parent_checkpoint_sha256"] != active_record["checkpoint_sha256"]
            or progress["checkpoint_sha256"] != active_record["checkpoint_sha256"]
            or not training["validation"]["full_weights_and_milestones_replay_exact"]
            or not training["validation"]["test_replay_exact"]):
        raise ValueError("Expected the active parent and its replay-validated rejected candidate")
    fixture = (ROOT / AUTHORED_FIXTURE).read_bytes()
    if (digest(fixture) != APPROVED_SHA256 or inputs["curriculum"] != json.loads(fixture)
            or digest(canonical({"conversation": inputs["teaching"],
                                 "corpus": corpus_records(inputs["corpus"])})) != training["review"]["train_sha256"]):
        raise ValueError("Training inputs do not match their review evidence")
    candidate = PlasticityContext.restore(json.loads(args.candidate.read_bytes())["state"])
    combined = mixture(active, candidate)
    _, _, test_pairs, _ = collect(doc for doc in documents(args.bible_dir, args.oanc) if doc[1] == "test")
    start = time.perf_counter()
    pair_scores = {}
    for name, model in (("active", active), ("candidate", candidate), ("mixture", combined)):
        pair_scores[name] = evaluate_pairs(model, test_pairs)
        print(f"Finished {name} two-word evaluation", flush=True)
    pair_seconds = time.perf_counter() - start
    start = time.perf_counter()
    mix_teaching = assess(combined, inputs["teaching"], no_triple_repeat=True)
    mix_test = assess(combined, inputs["curriculum"]["test"], no_triple_repeat=True)
    feedback = [item["row"] for item in progress["approved_corrections"]]
    mix_feedback = assess(combined, feedback, no_triple_repeat=True)
    retention = {
        "active": {"teaching": summary(training["before"]["training"]),
                   "development": summary(training["before"]["test"]),
                   "development_pairs": paired_scores(training["before"]["test"])["both_exact"],
                   "feedback": summary(training["before"]["feedback"])},
        "candidate": {"teaching": summary(training["checkpoints"][-1]["training"]),
                      "development": summary(training["test"]),
                      "development_pairs": training["test_pairs"]["both_exact"],
                      "feedback": summary(training["feedback"])},
        "mixture": {"teaching": summary(mix_teaching), "development": summary(mix_test),
                    "development_pairs": paired_scores(mix_test)["both_exact"],
                    "feedback": summary(mix_feedback),
                    "previously_correct_teaching_lost": len(lost_correct_ids(training["before"]["training"], mix_teaching)),
                    "previously_correct_feedback_lost": len(lost_correct_ids(training["before"]["feedback"], mix_feedback))},
    }
    if active_path.read_bytes() != before_active or progress_path.read_bytes() != before_progress:
        raise RuntimeError("The live model or training progress changed during evaluation")
    report = {
        "kind": "private_active_candidate_two_word_mixture",
        "data": hashes,
        "active_checkpoint_sha256": active_record["checkpoint_sha256"],
        "candidate_run_id": training["run_id"],
        "candidate_rejection": training["rejection"],
        "method": "Compare the active model, its rejected candidate, and their fixed equal-weight probability mixture, without changing either model. Use the same input-only text: prefix and actual first initial, greedy output, live three-repeat filter and 64-character cap. Parse up to the first newline. Evaluate actual preceding words at most four letters and all next-word lengths; no actual next initial is supplied. Short and four-letter next-word subsets are reported separately.",
        "pair_scores": pair_scores,
        "retention": retention,
        "pair_evaluation_seconds": pair_seconds,
        "retention_evaluation_seconds": time.perf_counter() - start,
        "limitations": [
            "The mixture is one fixed, untrained equal-weight blend, not a weight merge or a new active model.",
            "Held-out pairs may cross sentence punctuation; near-duplicate documents and repeated words may cross the split.",
            "The actual first word is at most four letters and its first initial is supplied; unavailable or unfinished outputs count as wrong.",
            "Development and feedback cases were reused and are not an independent conversation-quality test.",
            "Per-model inference times and memory were not measured separately, and no new-from-scratch model was tested.",
            "Candidate and approved feedback were not promoted or altered.",
        ],
    }
    with args.report.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps({"pairs": {name: scores["combined"] for name, scores in pair_scores.items()},
                      "retention": retention}, indent=2), flush=True)


if __name__ == "__main__":
    main()
