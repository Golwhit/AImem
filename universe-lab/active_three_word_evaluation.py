"""Compare active-model checkpoints on held-out three-word fragments."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neural_base_comparison import generate
from neural_chat import CONVERSATION_MODEL, load_selection
from neural_plasticity import PlasticityContext
from second_letter_experiment import BIBLE_MANIFEST_SHA256, LETTERS, OANC_SHA256, require_hash
from three_word_experiment import collect
from word_prefix_experiment import MAX_PREFIX
from word_start_experiment import WORD, documents

OUTPUT_LIMIT = 64


def words_from_output(initial, continuation):
    line, separator, _ = (initial + continuation).partition("\n")
    words = []
    position = 0
    for _ in range(3):
        match = WORD.match(line, position)
        if match is None or any(char not in LETTERS for char in match.group()):
            break
        word = match.group()
        ended = match.end() < len(line) or bool(separator)
        words.append((word[0], word if ended else None))
        if not ended:
            break
        position = match.end()
        gap_start = position
        while position < len(line) and line[position].isspace():
            position += 1
        if position == gap_start or position == len(line):
            break
    return words


def evaluate(model, triples):
    guesses = {
        initial: words_from_output(
            initial, generate(model, "text: " + initial, length=OUTPUT_LIMIT, no_triple_repeat=True)["text"]
        )
        for initial in LETTERS
    }
    results = {}
    for source, items in triples.items():
        groups = {source: Counter(), "combined": results.setdefault("combined", Counter()),
                  "short_third_word_only": results.setdefault("short_third_word_only", Counter()),
                  "four_letter_third_word_only": results.setdefault("four_letter_third_word_only", Counter())}
        for (first, second, third), count in items.items():
            predicted = guesses[first[0]]
            first_word = predicted[0][1] if predicted else None
            second_initial, second_word = predicted[1] if len(predicted) > 1 else (None, None)
            third_initial, third_word = predicted[2] if len(predicted) > 2 else (None, None)
            keys = [source, "combined"]
            if len(third) <= MAX_PREFIX:
                keys.append("short_third_word_only")
            if len(third) == MAX_PREFIX:
                keys.append("four_letter_third_word_only")
            for key in keys:
                scores = groups[key]
                scores["fragments"] += count
                scores["first_word_completed"] += count * (first_word is not None)
                scores["first_word_correct"] += count * (first_word == first)
                scores["second_initial_correct"] += count * (second_initial == second[0])
                scores["second_word_completed"] += count * (second_word is not None)
                scores["second_word_correct"] += count * (second_word == second)
                scores["first_two_correct"] += count * (first_word == first and second_word == second)
                scores["third_initial_available"] += count * (third_initial is not None)
                scores["third_initial_correct"] += count * (third_initial == third[0])
                scores["third_word_completed"] += count * (third_word is not None)
                scores["third_word_correct"] += count * (third_word == third)
                scores["phrase_correct"] += count * (
                    first_word == first and second_word == second and third_word == third)
        results[source] = groups[source]
    if not results.get("combined", {}).get("fragments"):
        raise ValueError("No held-out fragments")
    return {key: dict(value) for key, value in sorted(results.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bible-dir", required=True, type=Path)
    parser.add_argument("--oanc", required=True, type=Path)
    parser.add_argument("--before-checkpoint", required=True)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    hashes = {
        "bible_manifest_sha256": require_hash(args.bible_dir / "manifest.json", BIBLE_MANIFEST_SHA256),
        "oanc_sha256": require_hash(args.oanc, OANC_SHA256),
    }
    report_bytes = (args.candidate.parent / "report.json").read_bytes()
    candidate_bytes = args.candidate.read_bytes()
    source_report = json.loads(report_bytes)
    if (source_report["status"] not in ("completed", "rejected")
            or source_report["parent_checkpoint_sha256"] != args.before_checkpoint
            or not source_report["validation"]["full_weights_and_milestones_replay_exact"]
            or not source_report["validation"]["test_replay_exact"]):
        raise ValueError("Candidate must have matching parent and exact replay evidence")
    _, before, _ = load_selection(ROOT, {"model": CONVERSATION_MODEL, "arm": "conversation",
                                         "checkpoint_sha256": args.before_checkpoint})
    candidate = PlasticityContext.restore(json.loads(candidate_bytes)["state"])
    _, _, tests, _ = collect(doc for doc in documents(args.bible_dir, args.oanc) if doc[1] == "test")
    report = {
        "kind": "private_active_model_three_word_continuation",
        "data": {**hashes, "corpus_train_sha256": source_report["review"]["corpus_file_sha256"]},
        "before_checkpoint_sha256": args.before_checkpoint,
        "candidate_run_id": source_report["run_id"],
        "candidate_file_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "training_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "candidate_status": source_report["status"],
        "candidate_rejection": source_report.get("rejection", []),
        "method": "For each actual first initial, greedily generate up to 64 characters from the input-only 'text: ' prefix and initial, using the live three-repeat filter. Stop parsing at the first newline. Score adjacent ASCII words separated by whitespace, counting unfinished or unavailable words as wrong. The same held-out fragments as the short-word index test are used; actual first and second words are at most four letters. All target lengths are included.",
        "before": evaluate(before, tests),
        "candidate": evaluate(candidate, tests),
        "limitations": [
            "The active model is a different architecture and decoding method from the earlier isolated index.",
            "The true first initial is supplied; neither the actual second nor third word is supplied.",
            "Generation stops at the first newline, and the 64-character output cap can leave words unfinished.",
            "The reviewed training corpus and rehearsal differ from the Bible/OANC text used for held-out scoring.",
            "Fragments are not necessarily sentences; common text and near-duplicate documents may cross the earlier split.",
            "This is a fixed held-out generation test, not a measure of conversational quality.",
        ],
    }
    with args.report.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps({key: report[key] for key in ("candidate_status", "candidate_rejection", "before", "candidate")},
                     indent=2))


if __name__ == "__main__":
    main()
