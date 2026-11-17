"""Compare next-word initials after known and generated short words."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from second_letter_experiment import (
    BIBLE_MANIFEST_SHA256, LETTERS, MAPPING_SHA256, OANC_SHA256, require_hash,
)
from word_prefix_experiment import MAX_PREFIX, PrefixRanker
from word_start_experiment import WORD, documents, load_mapping


def collect(docs):
    words = defaultdict(Counter)
    pairs = defaultdict(Counter)
    totals = defaultdict(Counter)
    for source, partition, text in docs:
        if partition not in ("train", "test"):
            raise ValueError(f"Unknown partition: {partition}")
        totals[source, partition]["documents"] += 1
        previous = None
        for match in WORD.finditer(text):
            word = match.group()
            if not all(char in LETTERS for char in word):
                totals[source, partition]["excluded_words"] += 1
                previous = None
                continue
            words[source, partition][word] += 1
            totals[source, partition]["words"] += 1
            if previous is not None and len(previous) <= MAX_PREFIX:
                pairs[source, partition][previous, word[0]] += 1
                totals[source, partition]["short_word_pairs"] += 1
            previous = word
    return words, pairs, totals


class NextInitialRanker:
    def __init__(self, mapping, pairs):
        self.representatives = {letter: identities[0] for letter, identities in mapping.items()}
        self.labels = {identity: letter for letter, identities in mapping.items() for identity in identities}
        if set(self.representatives) != set(LETTERS) or len(self.labels) != sum(map(len, mapping.values())):
            raise ValueError("Mapping must assign each pattern uniquely and include all 52 letters")
        self.contexts = defaultdict(Counter)
        self.overall = Counter()
        for (previous, initial), count in pairs.items():
            if (not 1 <= len(previous) <= MAX_PREFIX or any(char not in LETTERS for char in previous)
                    or initial not in LETTERS or len(initial) != 1 or count <= 0):
                raise ValueError("Invalid training pair")
            context = tuple(self.representatives[char] for char in previous)
            target = self.representatives[initial]
            self.contexts[context][target] += count
            self.overall[target] += count
        if not self.overall:
            raise ValueError("No training word pairs")

    def rank(self, previous):
        if not 1 <= len(previous) <= MAX_PREFIX or any(char not in LETTERS for char in previous):
            raise ValueError("Previous word must contain one to four ASCII letters")
        context = tuple(self.representatives[char] for char in previous)
        counts = self.contexts.get(context)
        return self._order(counts or self.overall), not bool(counts)

    def _order(self, counts):
        return tuple(self.labels[pattern] for pattern in
                     sorted(counts, key=lambda pattern: (-counts[pattern], self.labels[pattern])))

    def baseline(self):
        return self._order(self.overall)


def evaluate(ranker, generator, pairs):
    baseline = ranker.baseline()
    generated = {}
    known_predictions = {}
    generated_predictions = {}
    result = Counter()
    for (previous, actual), count in pairs.items():
        if previous not in known_predictions:
            known_predictions[previous] = ranker.rank(previous)
        known, fallback = known_predictions[previous]
        if previous[0] not in generated:
            generated[previous[0]] = generator.generate_up_to_four(previous[0])
        predicted_word, ended = generated[previous[0]]
        result["pairs"] += count
        result["known_word_top1_correct"] += count * (known[0] == actual)
        result["known_word_top3_correct"] += count * (actual in known[:3])
        result["known_word_fallback"] += count * fallback
        result["baseline_top1_correct"] += count * (baseline[0] == actual)
        result["baseline_top3_correct"] += count * (actual in baseline[:3])
        if ended:
            if predicted_word not in generated_predictions:
                generated_predictions[predicted_word] = ranker.rank(predicted_word)
            predicted, generated_fallback = generated_predictions[predicted_word]
            result["generated_end_available"] += count
            result["generated_previous_exact"] += count * (predicted_word == previous)
            result["generated_top1_correct"] += count * (predicted[0] == actual)
            result["generated_top3_correct"] += count * (actual in predicted[:3])
            result["generated_word_fallback"] += count * generated_fallback
            result["known_word_top1_correct_when_generated_end"] += count * (known[0] == actual)
            result["baseline_top1_correct_when_generated_end"] += count * (baseline[0] == actual)
    if not result["pairs"]:
        raise ValueError("No held-out short-word pairs")
    return dict(result)


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
    words, pairs, totals = collect(documents(args.bible_dir, args.oanc))
    training_words = sum((values for (_, partition), values in words.items() if partition == "train"), Counter())
    training_pairs = sum((values for (_, partition), values in pairs.items() if partition == "train"), Counter())
    generator = PrefixRanker(mapping, training_words)
    ranker = NextInitialRanker(mapping, training_pairs)
    sources = sorted({source for source, partition in pairs if partition == "test"})
    if not sources:
        raise ValueError("No held-out sources")
    all_test = sum((pairs[source, "test"] for source in sources), Counter())
    four_test = Counter({pair: count for pair, count in all_test.items() if len(pair[0]) == MAX_PREFIX})
    results = {source: evaluate(ranker, generator, pairs[source, "test"]) for source in sources}
    results["combined"] = evaluate(ranker, generator, all_test)
    results["four_letter_previous_only"] = evaluate(ranker, generator, four_test)
    report = {
        "kind": "private_next_word_after_short_generation",
        "data": {**hashes, "bible": "Staged World English Bible learning books only; Acts 3 and earlier development/locked books excluded.",
                 "oanc": "Previously approved OANC archive, private use.",
                 "rights": "User approval is not legal clearance; OANC package and underlying rights require review."},
        "method": "Predict the next word's case-sensitive initial after a completed ASCII word of one to four letters. Compare the correct previous word with the greedy four-character generator seeded with the correct previous initial; only predict after its stop signal. Count no-stop cases as unavailable, not correct. Refit both arms only on the training documents; Bible learning books split by book and OANC documents by content hash.",
        "counts": {f"{source}_{partition}": dict(values) for (source, partition), values in sorted(totals.items())},
        "baseline_top3": ranker.baseline()[:3],
        "test": results,
        "limitations": [
            "The known-word arm receives the actual previous word; the generated arm receives its actual first letter and must generate and stop independently.",
            "This predicts only the next initial, not the next complete word. No-stop cases cannot produce a boundary within four characters.",
            "Only completely ASCII-letter words are included; pairs do not skip excluded words or cross documents, but can cross sentence punctuation.",
            "Common words and near-duplicate documents may cross the split. The pattern mapping used corpus-wide initial frequencies including test sources.",
            "The fixed pattern families encode letters and do not simulate quark bonding.",
        ],
    }
    with args.report.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps({key: report[key] for key in ("counts", "baseline_top3", "test")}, indent=2))


if __name__ == "__main__":
    main()
