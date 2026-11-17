"""Predict a next-word initial and word without supplying that initial."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from contextual_word_experiment import ContextualGenerator, select_contexts, train_contexts
from next_word_after_generation import NextInitialRanker
from second_letter_experiment import (
    BIBLE_MANIFEST_SHA256, LETTERS, MAPPING_SHA256, OANC_SHA256, require_hash,
)
from word_beam_experiment import beam_words, next_initial_distribution, select_with_next_initial
from word_prefix_experiment import MAX_PREFIX, PrefixRanker
from word_start_experiment import WORD, documents, load_mapping

WIDTHS = (5, 20)


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
        for match in WORD.finditer(text):
            word = match.group()
            if not all(char in LETTERS for char in word):
                previous = None
                totals[source, partition]["excluded_words"] += 1
                continue
            totals[source, partition]["words"] += 1
            if partition == "train":
                training_words[word[:MAX_PREFIX + 1]] += 1
                if previous is not None and len(previous) <= MAX_PREFIX:
                    training_pairs[previous, word[0]] += 1
            elif previous is not None and len(previous) <= MAX_PREFIX:
                tests[source][previous, word] += 1
                totals[source, partition]["pairs"] += 1
            previous = word
    return training_words, training_pairs, tests, totals


def mixed_initial_ranking(candidates, ranker):
    if not candidates:
        return ()
    total_weight = sum(probability for _, probability in candidates)
    if total_weight <= 0 or any(probability <= 0 for _, probability in candidates):
        raise ValueError("Beam probabilities must be positive")
    probabilities = Counter()
    for word, weight in candidates:
        for letter, probability in next_initial_distribution(word, ranker).items():
            probabilities[letter] += weight / total_weight * probability
    return tuple(sorted(LETTERS, key=lambda letter: (-probabilities[letter], letter)))


def proposals(base, ranker, first):
    previous, ended = base.generate_up_to_four(first)
    result = {"greedy": (previous, ranker.rank(previous)[0]) if ended else None}
    for width in WIDTHS:
        candidates = beam_words(base, first, width)
        if candidates:
            ranking = mixed_initial_ranking(candidates, ranker)
            selected = select_with_next_initial(candidates, ranking[0], ranker)
            if selected is None:
                raise ValueError("A nonempty beam did not select a previous word")
            result[f"beam{width}_mixture"] = selected, ranking
        else:
            result[f"beam{width}_mixture"] = None
        if width == max(WIDTHS):
            result["beam20_single"] = (
                (candidates[0][0], ranker.rank(candidates[0][0])[0]) if candidates else None
            )
    return result


def evaluate(base, contextual, ranker, pairs):
    scores = Counter()
    for arm in ("greedy", "beam5_mixture", "beam20_mixture", "beam20_single", "known_previous"):
        for field in ("available", "previous_correct", "initial_correct", "initial_top3_correct",
                      "word_stopped", "word_correct", "initial_correct_when_greedy_available",
                      "word_correct_when_greedy_available"):
            scores[arm + "_" + field] = 0
    proposals_cache = {}
    oracle_cache = {}
    generation_cache = {}
    baseline = ranker.baseline()
    for (previous, actual), count in pairs.items():
        scores["pairs"] += count
        if previous[0] not in proposals_cache:
            proposals_cache[previous[0]] = proposals(base, ranker, previous[0])
        options = proposals_cache[previous[0]]
        if previous not in oracle_cache:
            oracle_cache[previous] = previous, ranker.rank(previous)[0]
        options_for_pair = {**options, "known_previous": oracle_cache[previous]}
        greedy_available = options["greedy"] is not None
        scores["greedy_available_pairs"] += count * greedy_available
        scores["global_initial_correct"] += count * (baseline[0] == actual[0])
        if greedy_available:
            scores["global_initial_correct_when_greedy_available"] += count * (baseline[0] == actual[0])
        for arm, proposal in options_for_pair.items():
            if proposal is None:
                continue
            predicted_previous, ranking = proposal
            initial = ranking[0]
            key = predicted_previous, initial
            if key not in generation_cache:
                generation_cache[key] = contextual.generate_up_to_four(predicted_previous, initial)
            generated, ended, _ = generation_cache[key]
            scores[arm + "_available"] += count
            scores[arm + "_previous_correct"] += count * (predicted_previous == previous)
            scores[arm + "_initial_correct"] += count * (initial == actual[0])
            scores[arm + "_initial_top3_correct"] += count * (actual[0] in ranking[:3])
            scores[arm + "_word_stopped"] += count * ended
            scores[arm + "_word_correct"] += count * (ended and generated == actual)
            if greedy_available:
                scores[arm + "_initial_correct_when_greedy_available"] += count * (initial == actual[0])
                scores[arm + "_word_correct_when_greedy_available"] += count * (ended and generated == actual)
    if not scores["pairs"]:
        raise ValueError("No held-out word pairs")
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
    selected = select_contexts(training_pairs)
    contexts = train_contexts(documents(args.bible_dir, args.oanc), selected)
    base = PrefixRanker(mapping, training_words)
    contextual = ContextualGenerator(base, contexts)
    ranker = NextInitialRanker(mapping, training_pairs)
    if not tests:
        raise ValueError("No held-out sources")
    all_test = sum(tests.values(), Counter())
    short_test = Counter({pair: count for pair, count in all_test.items() if len(pair[1]) <= MAX_PREFIX})
    four_test = Counter({pair: count for pair, count in all_test.items() if len(pair[1]) == MAX_PREFIX})
    results = {source: evaluate(base, contextual, ranker, pairs) for source, pairs in sorted(tests.items())}
    results["combined"] = evaluate(base, contextual, ranker, all_test)
    results["short_target_only"] = evaluate(base, contextual, ranker, short_test)
    results["four_letter_target_only"] = evaluate(base, contextual, ranker, four_test)
    report = {
        "kind": "private_unsupplied_next_initial_and_word",
        "data": {**hashes, "bible": "Staged World English Bible learning books only; Acts 3 and earlier development/locked books excluded.",
                 "oanc": "Previously approved OANC archive, private use.",
                 "rights": "User approval is not legal clearance; OANC package and underlying rights require review."},
        "method": "Supply only the actual previous word's first letter. Compare a greedy previous word, a single highest-prior beam word, and mixtures over finished 5/20-path beams. Mixtures combine each previous-word path probability with its add-one-smoothed training distribution for the next initial. Select context using the predicted initial, then generate the next word. An oracle arm receives the actual previous word, but no arm receives the actual next initial. Evaluate the identical held-out pairs; no-stop cases count as incorrect in full-pair scores.",
        "counts": {f"{source}_{partition}": dict(values) for (source, partition), values in sorted(totals.items())},
        "test": results,
        "limitations": [
            "The actual previous initial is supplied and the evaluation assumes the previous word is at most four letters. The beam selects among finished candidates.",
            "The target initial is predicted, not supplied. Word generation stops after at most four letters; longer targets cannot be generated exactly.",
            "Only completely ASCII-letter words are included. Pairs do not skip excluded words or cross documents, but can cross sentence punctuation.",
            "Common words and near-duplicate documents may cross the split. The pattern mapping used corpus-wide initial frequencies including test sources.",
            "The fixed pattern families encode letters and do not simulate quark bonding.",
        ],
    }
    with args.report.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps({key: report[key] for key in ("counts", "test")}, indent=2))


if __name__ == "__main__":
    main()
