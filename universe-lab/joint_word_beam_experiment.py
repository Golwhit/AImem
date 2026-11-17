"""Predict a next word by keeping alternatives through its letters."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from contextual_word_experiment import ContextualGenerator, select_contexts, train_contexts
from next_word_after_generation import NextInitialRanker
from second_letter_experiment import (
    BIBLE_MANIFEST_SHA256, LETTERS, MAPPING_SHA256, OANC_SHA256, require_hash,
)
from unsupplied_next_word_experiment import collect, proposals
from word_beam_experiment import beam_words, next_initial_distribution
from word_prefix_experiment import END, MAX_PREFIX, PrefixRanker
from word_start_experiment import documents, load_mapping

WIDTHS = (5, 20)
PREVIOUS_WIDTH = 20


class JointWordBeam:
    def __init__(self, base, contextual, ranker, initial_distribution=next_initial_distribution):
        self.base = base
        self.contextual = contextual
        self.ranker = ranker
        self.initial_distribution = initial_distribution
        self.previous_cache = {}
        self.distribution_cache = {}

    def distribution(self, previous, prefix):
        key = previous, prefix
        if key not in self.distribution_cache:
            scores, _ = self.contextual.weights(previous, prefix)
            total = sum(scores.values())
            if total <= 0:
                raise ValueError("A prefix must have positive next-character mass")
            self.distribution_cache[key] = {label: score / total for label, score in scores.items()}
        return self.distribution_cache[key]

    def search(self, first, width):
        if len(first) != 1 or first not in LETTERS or width < 1:
            raise ValueError("Search needs one ASCII initial and a positive width")
        if first not in self.previous_cache:
            self.previous_cache[first] = beam_words(self.base, first, PREVIOUS_WIDTH)
        return self.search_candidates(self.previous_cache[first], width)

    def search_candidates(self, previous, width):
        if width < 1:
            raise ValueError("Search needs a positive width")
        if not previous:
            return []
        if any(not 1 <= len(word) <= MAX_PREFIX or any(char not in LETTERS for char in word)
               or not math.isfinite(weight) or weight <= 0 for word, weight in previous):
            raise ValueError("Preceding-word candidates must be short ASCII words with positive finite weights")
        total = sum(weight for _, weight in previous)
        if not math.isfinite(total) or total <= 0:
            raise ValueError("Preceding-word candidates must have positive mass")
        distributions = [self.initial_distribution(word, self.ranker) for word, _ in previous]
        nodes = []
        for letter in LETTERS:
            weights = tuple(weight / total * distribution[letter]
                            for (_, weight), distribution in zip(previous, distributions))
            nodes.append((letter, weights, "active"))
        nodes = sorted(nodes, key=lambda node: (-sum(node[1]), node[0], node[2]))[:width]
        for _ in range(MAX_PREFIX):
            expanded = []
            for prefix, weights, state in nodes:
                if state != "active":
                    expanded.append((prefix, weights, state))
                    continue
                transitions = {}
                for index, ((word, _), weight) in enumerate(zip(previous, weights)):
                    if not weight:
                        continue
                    for label, probability in self.distribution(word, prefix).items():
                        if label not in transitions:
                            transitions[label] = [0.0] * len(previous)
                        transitions[label][index] += weight * probability
                for label, values in transitions.items():
                    if label == END:
                        expanded.append((prefix, tuple(values), "ended"))
                    elif len(prefix) < MAX_PREFIX:
                        expanded.append((prefix + label, tuple(values), "active"))
                    else:
                        expanded.append((prefix + label, tuple(values), "continues"))
            nodes = sorted(expanded, key=lambda node: (-sum(node[1]), node[0], node[2]))[:width]
        return [(word, sum(weights), state) for word, weights, state in nodes]


def evaluate(base, contextual, ranker, joint, pairs):
    scores = Counter()
    for arm in ("joint5", "joint20"):
        for field in ("available", "winner_stopped", "initial_correct", "word_correct",
                      "top5_word_correct", "top20_word_correct",
                      "initial_correct_when_greedy_available", "word_correct_when_greedy_available",
                      "top5_word_correct_when_greedy_available"):
            scores[arm + "_" + field] = 0
    for field in ("available", "initial_correct", "word_correct",
                  "initial_correct_when_greedy_available", "word_correct_when_greedy_available"):
        scores["baseline_" + field] = 0
    options = {}
    baseline_cache = {}
    paths_cache = {}
    for (previous, actual), count in pairs.items():
        scores["pairs"] += count
        first = previous[0]
        if first not in options:
            options[first] = proposals(base, ranker, first)
            paths_cache[first] = {width: joint.search(first, width) for width in WIDTHS}
        greedy_available = options[first]["greedy"] is not None
        scores["greedy_available_pairs"] += count * greedy_available
        baseline = options[first]["beam20_mixture"]
        if baseline is not None:
            context, ranking = baseline
            key = context, ranking[0]
            if key not in baseline_cache:
                baseline_cache[key] = contextual.generate_up_to_four(*key)
            predicted, stopped, _ = baseline_cache[key]
            scores["baseline_available"] += count
            scores["baseline_initial_correct"] += count * (ranking[0] == actual[0])
            scores["baseline_word_correct"] += count * (stopped and predicted == actual)
            if greedy_available:
                scores["baseline_initial_correct_when_greedy_available"] += count * (ranking[0] == actual[0])
                scores["baseline_word_correct_when_greedy_available"] += count * (stopped and predicted == actual)
        for width, paths in paths_cache[first].items():
            if not paths:
                continue
            word, _, state = paths[0]
            arm = f"joint{width}"
            scores[arm + "_available"] += count
            scores[arm + "_winner_stopped"] += count * (state == "ended")
            scores[arm + "_initial_correct"] += count * (word[0] == actual[0])
            scores[arm + "_word_correct"] += count * (state == "ended" and word == actual)
            scores[arm + "_top5_word_correct"] += count * any(
                candidate == actual and candidate_state == "ended" for candidate, _, candidate_state in paths[:5])
            scores[arm + "_top20_word_correct"] += count * any(
                candidate == actual and candidate_state == "ended" for candidate, _, candidate_state in paths[:20])
            if greedy_available:
                scores[arm + "_initial_correct_when_greedy_available"] += count * (word[0] == actual[0])
                scores[arm + "_word_correct_when_greedy_available"] += count * (state == "ended" and word == actual)
                scores[arm + "_top5_word_correct_when_greedy_available"] += count * any(
                    candidate == actual and candidate_state == "ended" for candidate, _, candidate_state in paths[:5])
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
    joint = JointWordBeam(base, contextual, ranker)
    if not tests:
        raise ValueError("No held-out sources")
    all_test = sum(tests.values(), Counter())
    short_test = Counter({pair: count for pair, count in all_test.items() if len(pair[1]) <= MAX_PREFIX})
    four_test = Counter({pair: count for pair, count in all_test.items() if len(pair[1]) == MAX_PREFIX})
    results = {source: evaluate(base, contextual, ranker, joint, pairs) for source, pairs in sorted(tests.items())}
    results["combined"] = evaluate(base, contextual, ranker, joint, all_test)
    results["short_target_only"] = evaluate(base, contextual, ranker, joint, short_test)
    results["four_letter_target_only"] = evaluate(base, contextual, ranker, joint, four_test)
    report = {
        "kind": "private_joint_next_word_beam",
        "data": {**hashes, "bible": "Staged World English Bible learning books only; Acts 3 and earlier development/locked books excluded.",
                 "oanc": "Previously approved OANC archive, private use.",
                 "rights": "User approval is not legal clearance; OANC package and underlying rights require review."},
        "method": "Supply the true preceding-word first letter, then keep up to 20 finished preceding-word candidates. Marginalize their weighted distributions for the next initial and subsequent characters, updating their weights along each word-prefix path. Search five and twenty next-word paths including end and over-four continuations. Compare with the prior 20-previous-path mixture using one predicted initial and greedy current-word continuation. All decisions use training-only counts.",
        "widths": WIDTHS, "previous_width": PREVIOUS_WIDTH,
        "counts": {f"{source}_{partition}": dict(values) for (source, partition), values in sorted(totals.items())},
        "test": results,
        "limitations": [
            "The true first letter of the previous word is supplied, and the previous word is assumed no longer than four letters. The next initial is not supplied.",
            "Continuing and finished paths compete for beam positions. A top path continuing beyond four is not counted as a complete word; top-five and top-twenty include all path types.",
            "Only completely ASCII-letter words are included. Pairs do not skip excluded words or cross documents, but can cross sentence punctuation.",
            "Common words and near-duplicate documents may cross the split. The pattern mapping used corpus-wide initial frequencies including test sources.",
            "The fixed pattern families encode letters and do not simulate quark bonding.",
        ],
    }
    with args.report.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps({key: report[key] for key in ("widths", "counts", "test")}, indent=2))


if __name__ == "__main__":
    main()
