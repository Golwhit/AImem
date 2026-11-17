"""Compare short-word generation with and without preceding-word context."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from next_word_after_generation import NextInitialRanker
from second_letter_experiment import (
    BIBLE_MANIFEST_SHA256, LETTERS, MAPPING_SHA256, OANC_SHA256, require_hash,
)
from word_prefix_experiment import END, MAX_PREFIX, PrefixRanker
from word_start_experiment import WORD, documents, load_mapping

MAX_CONTEXT_WORDS = 64
PRIOR_STRENGTH = 10


def collect(docs):
    training_words = Counter()
    training_pairs = Counter()
    windows = defaultdict(Counter)
    totals = defaultdict(Counter)
    for source, partition, text in docs:
        if partition not in ("train", "test"):
            raise ValueError(f"Unknown partition: {partition}")
        history = []
        totals[source, partition]["documents"] += 1
        for match in WORD.finditer(text):
            word = match.group()
            if not all(char in LETTERS for char in word):
                history = []
                totals[source, partition]["excluded_words"] += 1
                continue
            totals[source, partition]["words"] += 1
            if partition == "train":
                # Only the first five characters can affect a decision through position four.
                training_words[word[:MAX_PREFIX + 1]] += 1
                if history and len(history[-1]) <= MAX_PREFIX:
                    training_pairs[history[-1], word[0]] += 1
            elif (len(history) == 2 and len(history[0]) <= MAX_PREFIX
                  and len(history[1]) <= MAX_PREFIX):
                windows[source][history[0], history[1], word[0]] += 1
                totals[source, partition]["short_word_windows"] += 1
            history = (history + [word])[-2:]
    return training_words, training_pairs, windows, totals


def select_contexts(pairs):
    counts = Counter()
    for (previous, _), count in pairs.items():
        counts[previous] += count
    return set(sorted(counts, key=lambda word: (-counts[word], word))[:MAX_CONTEXT_WORDS])


def train_contexts(docs, selected):
    contexts = defaultdict(Counter)
    for _, partition, text in docs:
        if partition != "train":
            continue
        previous = None
        for match in WORD.finditer(text):
            word = match.group()
            if not all(char in LETTERS for char in word):
                previous = None
                continue
            if previous in selected:
                for length in range(1, min(len(word), MAX_PREFIX) + 1):
                    target = word[length] if length < len(word) else END
                    contexts[previous, word[:length]][target] += 1
            previous = word
    if not contexts:
        raise ValueError("No preceding-word context observations")
    return contexts


class ContextualGenerator:
    def __init__(self, base, contexts):
        self.base = base
        self.contexts = contexts
        self.cache = {}

    def weights(self, previous, prefix):
        _, total, _, base_counts = self.base.rank(prefix)
        observed = self.contexts.get((previous, prefix))
        scores = Counter(observed or {})
        for pattern, count in base_counts.items():
            label = END if pattern is None else self.base.labels[pattern]
            scores[label] += PRIOR_STRENGTH * count / total if observed else count
        return scores, bool(observed)

    def predict(self, previous, prefix):
        key = previous, prefix
        if key not in self.cache:
            scores, used = self.weights(previous, prefix)
            self.cache[key] = (min(scores, key=lambda label: (-scores[label], label)), used)
        return self.cache[key]

    def generate_up_to_four(self, previous, initial):
        prefix = initial
        used_context = False
        while True:
            target, used = self.predict(previous, prefix)
            used_context |= used
            if target == END:
                return prefix, True, used_context
            if len(prefix) == MAX_PREFIX:
                return prefix, False, used_context
            prefix += target


def evaluate(base, candidate, next_ranker, windows):
    scores = Counter()
    cache = {}
    previous_cache = {}
    next_cache = {}

    def next_correct(predicted, actual):
        if predicted not in next_cache:
            next_cache[predicted] = next_ranker.rank(predicted)[0][0]
        return next_cache[predicted] == actual

    for (previous, actual, following), count in windows.items():
        scores["windows"] += count
        key = previous, actual[0]
        if key not in cache:
            cache[key] = (base.generate_up_to_four(actual[0]),
                          candidate.generate_up_to_four(previous, actual[0]))
        control, contextual = cache[key]
        for arm, (predicted, ended) in (
            ("control", control), ("known_previous", contextual[:2]),
        ):
            scores[arm + "_stopped"] += count * ended
            scores[arm + "_word_correct"] += count * (ended and predicted == actual)
            scores[arm + "_next_initial_correct"] += count * (
                ended and next_correct(predicted, following))
        scores["known_previous_used_context"] += count * contextual[2]
        if previous[0] not in previous_cache:
            previous_cache[previous[0]] = base.generate_up_to_four(previous[0])
        generated_previous, previous_ended = previous_cache[previous[0]]
        if previous_ended:
            scores["generated_previous_available"] += count
            scores["generated_previous_exact"] += count * (generated_previous == previous)
            scores["control_word_correct_same_subset"] += count * (control[1] and control[0] == actual)
            scores["control_next_correct_same_subset"] += count * (
                control[1] and next_correct(control[0], following))
            key = generated_previous, actual[0]
            if key not in cache:
                cache[key] = (base.generate_up_to_four(actual[0]),
                              candidate.generate_up_to_four(generated_previous, actual[0]))
            predicted, ended, used = cache[key][1]
            scores["generated_previous_used_context"] += count * used
            scores["generated_previous_word_correct"] += count * (ended and predicted == actual)
            scores["generated_previous_next_initial_correct"] += count * (
                ended and next_correct(predicted, following))
    if not scores["windows"]:
        raise ValueError("No held-out short-word windows")
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
    training_words, training_pairs, windows, totals = collect(documents(args.bible_dir, args.oanc))
    selected = select_contexts(training_pairs)
    contexts = train_contexts(documents(args.bible_dir, args.oanc), selected)
    base = PrefixRanker(mapping, training_words)
    candidate = ContextualGenerator(base, contexts)
    next_ranker = NextInitialRanker(mapping, training_pairs)
    if not windows:
        raise ValueError("No held-out sources")
    all_test = sum(windows.values(), Counter())
    four_test = Counter({window: count for window, count in all_test.items() if len(window[1]) == MAX_PREFIX})
    results = {source: evaluate(base, candidate, next_ranker, values) for source, values in sorted(windows.items())}
    results["combined"] = evaluate(base, candidate, next_ranker, all_test)
    results["four_letter_target_only"] = evaluate(base, candidate, next_ranker, four_test)
    report = {
        "kind": "private_contextual_short_word_generation",
        "data": {**hashes, "bible": "Staged World English Bible learning books only; Acts 3 and earlier development/locked books excluded.",
                 "oanc": "Previously approved OANC archive, private use.",
                 "rights": "User approval is not legal clearance; OANC package and underlying rights require review."},
        "method": "Select the 64 most frequent one- to four-letter preceding words from training pairs. For each generated prefix, combine its contextual next-character counts with 10 observations of the unconditional prefix distribution; otherwise use the unchanged prefix model. Compare a true preceding word, a generated preceding word, and the unconditional model on identical held-out short-word windows. The true first letter of the target word is provided to every arm.",
        "context_limit": MAX_CONTEXT_WORDS,
        "prior_strength": PRIOR_STRENGTH,
        "selected_context_count": len(selected),
        "counts": {f"{source}_{partition}": dict(values) for (source, partition), values in sorted(totals.items())},
        "test": results,
        "limitations": [
            "The first letters of both previous and target words are supplied; the generated-previous arm does not generate continuous text without assistance.",
            "No-stop cases count as incorrect in full-window scores. Generated-previous scores are additionally reported over the subset where the previous word stopped.",
            "Only completely ASCII-letter words are included. Windows do not skip excluded words or cross documents, but can cross sentence punctuation.",
            "Common words and near-duplicate documents may cross the split. The pattern mapping used corpus-wide initial frequencies including test sources.",
            "The fixed pattern families encode letters and do not simulate quark bonding.",
        ],
    }
    with args.report.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps({key: report[key] for key in ("context_limit", "prior_strength", "counts", "test")}, indent=2))


if __name__ == "__main__":
    main()
