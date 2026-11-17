"""Evaluate three-word continuations from a single supplied first initial."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from contextual_word_experiment import ContextualGenerator, select_contexts, train_contexts
from joint_word_beam_experiment import JointWordBeam
from next_word_after_generation import NextInitialRanker
from second_letter_experiment import (
    BIBLE_MANIFEST_SHA256, LETTERS, MAPPING_SHA256, OANC_SHA256, require_hash,
)
from word_beam_experiment import beam_words
from word_prefix_experiment import MAX_PREFIX, PrefixRanker
from word_start_experiment import WORD, documents, load_mapping

WIDTH = 20


def collect(docs):
    training_words = Counter()
    training_pairs = Counter()
    tests = defaultdict(Counter)
    totals = defaultdict(Counter)
    for source, partition, text in docs:
        if partition not in ("train", "test"):
            raise ValueError(f"Unknown partition: {partition}")
        totals[source, partition]["documents"] += 1
        previous = None
        history = []
        last_end = None
        for match in WORD.finditer(text):
            word = match.group()
            if partition == "test" and last_end is not None and not text[last_end:match.start()].isspace():
                history = []
            last_end = match.end()
            if not all(char in LETTERS for char in word):
                previous = None
                history = []
                totals[source, partition]["excluded_words"] += 1
                continue
            totals[source, partition]["words"] += 1
            if partition == "train":
                training_words[word[:MAX_PREFIX + 1]] += 1
                if previous is not None and len(previous) <= MAX_PREFIX:
                    training_pairs[previous, word[0]] += 1
                previous = word
            else:
                if len(history) == 2 and all(len(item) <= MAX_PREFIX for item in history):
                    tests[source][history[0], history[1], word] += 1
                    totals[source, partition]["triples"] += 1
                history = (history + [word])[-2:]
    return training_words, training_pairs, tests, totals


class PhrasePredictor:
    def __init__(self, base, contextual, ranker, joint):
        self.base = base
        self.contextual = contextual
        self.ranker = ranker
        self.joint = joint

    def greedy(self, first_initial):
        first, ended = self.base.generate_up_to_four(first_initial)
        if not ended:
            return None, None, None, None, None
        second_initial = self.ranker.rank(first)[0][0]
        second, ended, _ = self.contextual.generate_up_to_four(first, second_initial)
        if not ended:
            return first, second_initial, None, None, None
        third_initial = self.ranker.rank(second)[0][0]
        third, ended, _ = self.contextual.generate_up_to_four(second, third_initial)
        return first, second_initial, second, third_initial, third if ended else None

    def beam(self, first_initial):
        candidates = beam_words(self.base, first_initial, WIDTH)
        if not candidates:
            return None, None, None, None, None
        first = candidates[0][0]
        second_paths = self.joint.search_candidates([(first, 1.0)], WIDTH)
        if not second_paths:
            raise ValueError("No second-word paths for a finished first word")
        second, _, state = second_paths[0]
        if state != "ended":
            return first, second[0], None, None, None
        third_paths = self.joint.search_candidates([(second, 1.0)], WIDTH)
        if not third_paths:
            raise ValueError("No third-word paths for a finished second word")
        third, _, state = third_paths[0]
        return first, second[0], second, third[0], third if state == "ended" else None

    def known_second(self, second):
        initial = self.ranker.rank(second)[0][0]
        word, ended, _ = self.contextual.generate_up_to_four(second, initial)
        return initial, word if ended else None


def evaluate(predictor, triples):
    scores = Counter()
    for arm in ("greedy", "beam"):
        for field in ("first_correct", "second_initial_correct", "second_correct",
                      "first_two_correct", "third_available", "third_initial_correct",
                      "third_correct", "phrase_correct", "third_initial_correct_when_first_two_correct",
                      "third_correct_when_first_two_correct", "third_initial_correct_when_greedy_available",
                      "third_correct_when_greedy_available", "phrase_correct_when_greedy_available"):
            scores[arm + "_" + field] = 0
    predicted = {}
    oracle = {}
    for (first, second, third), count in triples.items():
        scores["triples"] += count
        if first[0] not in predicted:
            predicted[first[0]] = {
                "greedy": predictor.greedy(first[0]),
                "beam": predictor.beam(first[0]),
            }
        if second not in oracle:
            oracle[second] = predictor.known_second(second)
        initial, word = oracle[second]
        scores["known_second_initial_correct"] += count * (initial == third[0])
        scores["known_second_word_correct"] += count * (word == third)
        greedy_available = predicted[first[0]]["greedy"][3] is not None
        for arm, (first_guess, second_initial, second_guess, third_initial, third_guess) in predicted[first[0]].items():
            first_ok = first_guess == first
            second_ok = second_guess == second
            third_ok = third_guess == third
            scores[arm + "_first_correct"] += count * first_ok
            scores[arm + "_second_initial_correct"] += count * (second_initial == second[0])
            scores[arm + "_second_correct"] += count * second_ok
            scores[arm + "_first_two_correct"] += count * (first_ok and second_ok)
            scores[arm + "_third_available"] += count * (third_initial is not None)
            scores[arm + "_third_initial_correct"] += count * (third_initial == third[0])
            scores[arm + "_third_correct"] += count * third_ok
            scores[arm + "_phrase_correct"] += count * (first_ok and second_ok and third_ok)
            scores[arm + "_third_initial_correct_when_first_two_correct"] += count * (
                first_ok and second_ok and third_initial == third[0])
            scores[arm + "_third_correct_when_first_two_correct"] += count * (
                first_ok and second_ok and third_ok)
            if greedy_available:
                scores[arm + "_third_initial_correct_when_greedy_available"] += count * (third_initial == third[0])
                scores[arm + "_third_correct_when_greedy_available"] += count * third_ok
                scores[arm + "_phrase_correct_when_greedy_available"] += count * (first_ok and second_ok and third_ok)
    if not scores["triples"]:
        raise ValueError("No held-out three-word fragments")
    return dict(scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bible-dir", required=True, type=Path)
    parser.add_argument("--oanc", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    hashes = {
        "bible_manifest_sha256": require_hash(args.bible_dir / "manifest.json", BIBLE_MANIFEST_SHA256),
        "oanc_sha256": require_hash(args.oanc, OANC_SHA256),
        "mapping_sha256": require_hash(args.mapping, MAPPING_SHA256),
    }
    mapping = load_mapping(args.mapping)
    training_words, training_pairs, tests, totals = collect(documents(args.bible_dir, args.oanc))
    contexts = train_contexts(documents(args.bible_dir, args.oanc), select_contexts(training_pairs))
    base = PrefixRanker(mapping, training_words)
    contextual = ContextualGenerator(base, contexts)
    ranker = NextInitialRanker(mapping, training_pairs)
    predictor = PhrasePredictor(base, contextual, ranker, JointWordBeam(base, contextual, ranker))
    if not tests:
        raise ValueError("No held-out sources")
    all_test = sum(tests.values(), Counter())
    short_test = Counter({triple: count for triple, count in all_test.items() if len(triple[2]) <= MAX_PREFIX})
    four_test = Counter({triple: count for triple, count in all_test.items() if len(triple[2]) == MAX_PREFIX})
    results = {source: evaluate(predictor, triples) for source, triples in sorted(tests.items())}
    results["combined"] = evaluate(predictor, all_test)
    results["short_third_word_only"] = evaluate(predictor, short_test)
    results["four_letter_third_word_only"] = evaluate(predictor, four_test)
    report = {
        "kind": "private_three_word_short_context",
        "data": {**hashes, "bible": "Staged World English Bible learning books only; Acts 3 and earlier development/locked books excluded.",
                 "oanc": "Previously approved OANC archive, private use.",
                 "rights": "User approval is not legal clearance; OANC package and underlying rights require review."},
        "method": "Supply only the actual first word's first letter. Compare greedy continuation against a 20-path search that commits to each selected word before proceeding; next-word context is only the immediately preceding generated word. For comparison, a separate oracle receives the actual second word and greedily predicts the third. Evaluation uses adjacent ASCII words separated by whitespace, with first and second words at most four letters. No-stop or unavailable guesses count as incorrect in full-fragment scores.",
        "width": WIDTH,
        "counts": {f"{source}_{partition}": dict(values) for (source, partition), values in sorted(totals.items())},
        "test": results,
        "limitations": [
            "The first word's true initial is supplied and the first two actual words are restricted to at most four letters. Beam selection of the first word considers only completed candidates.",
            "The third word's true initial is never supplied. Longer third words cannot be completed by the four-letter generator.",
            "Fragments are not necessarily complete sentences. Test words are separated by whitespace; training pairs may cross sentence punctuation.",
            "Only completely ASCII-letter words are included. Common words and near-duplicate documents may cross the split, and mapping calibration used corpus-wide initial frequencies.",
            "This uses one previous word, not the two-word index retained for future review.",
        ],
    }
    with args.report.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps({key: report[key] for key in ("counts", "test")}, indent=2))


if __name__ == "__main__":
    main()
