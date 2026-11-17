"""Compare one- and two-word context at increasing training-data sizes."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

from contextual_word_experiment import ContextualGenerator, select_contexts, train_contexts
from joint_word_beam_experiment import JointWordBeam
from next_word_after_generation import NextInitialRanker
from second_letter_experiment import (
    BIBLE_MANIFEST_SHA256, LETTERS, MAPPING_SHA256, OANC_SHA256, require_hash,
)
from three_word_experiment import WIDTH, collect
from word_beam_experiment import beam_words, next_initial_distribution
from word_prefix_experiment import END, MAX_PREFIX, PrefixRanker
from word_start_experiment import WORD, documents, load_mapping

FRACTIONS = (25, 50, 100)
MAX_CONTEXTS = 512
PRIOR_STRENGTH = 10


def sampled(docs, percent):
    if percent not in FRACTIONS:
        raise ValueError("Unsupported training fraction")
    for source, partition, text in docs:
        value = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        if partition == "test" or value * 100 < percent * (1 << 64):
            yield source, partition, text


def training_triples(docs):
    for _, partition, text in docs:
        if partition != "train":
            continue
        history = []
        last_end = None
        for match in WORD.finditer(text):
            if last_end is not None and not text[last_end:match.start()].isspace():
                history = []
            last_end = match.end()
            word = match.group()
            if not all(char in LETTERS for char in word):
                history = []
                continue
            if len(history) == 2 and all(len(item) <= MAX_PREFIX for item in history):
                yield (history[0], history[1]), word
            history = (history + [word])[-2:]


def build_index(docs, selected):
    initials = defaultdict(Counter)
    prefixes = defaultdict(Counter)
    for pair, word in training_triples(docs):
        if pair not in selected:
            continue
        initials[pair][word[0]] += 1
        for length in range(1, min(len(word), MAX_PREFIX) + 1):
            target = word[length] if length < len(word) else END
            prefixes[pair, word[:length]][target] += 1
    return initials, prefixes


def blend(prior, observed):
    if not observed:
        return prior.copy()
    total = sum(prior.values())
    if total <= 0:
        raise ValueError("Context prior must have positive mass")
    scores = Counter(observed)
    for key, value in prior.items():
        scores[key] += PRIOR_STRENGTH * value / total
    return scores


class TwoWordContext:
    def __init__(self, base, initials, prefixes):
        self.base = base
        self.initials = initials
        self.prefixes = prefixes

    def initial_distribution(self, first, second, ranker):
        prior = next_initial_distribution(second, ranker)
        scores = blend(prior, self.initials.get((first, second)))
        total = sum(scores.values())
        return {key: value / total for key, value in scores.items()}

    def weights(self, first, second, prefix):
        prior, _ = self.base.weights(second, prefix)
        observed = self.prefixes.get(((first, second), prefix))
        return blend(prior, observed), bool(observed)


class BoundContext:
    def __init__(self, context, first):
        self.context = context
        self.first = first

    def weights(self, second, prefix):
        return self.context.weights(self.first, second, prefix)


def greedy_third(first, second, ranker, contextual, two=None):
    probabilities = (two.initial_distribution(first, second, ranker) if two
                     else next_initial_distribution(second, ranker))
    initial = min(probabilities, key=lambda letter: (-probabilities[letter], letter))
    prefix = initial
    while True:
        scores, _ = (two.weights(first, second, prefix) if two else contextual.weights(second, prefix))
        target = min(scores, key=lambda label: (-scores[label], label))
        if target == END:
            return initial, prefix
        if len(prefix) == MAX_PREFIX:
            return initial, None
        prefix += target


def generated_predictions(base, contextual, ranker, two):
    joint = JointWordBeam(base, contextual, ranker)
    predictions = {}
    for initial in LETTERS:
        first_paths = beam_words(base, initial, WIDTH)
        if not first_paths:
            predictions[initial] = (None, None, None, None)
            continue
        first = first_paths[0][0]
        second_paths = joint.search_candidates([(first, 1.0)], WIDTH)
        if not second_paths:
            raise ValueError("No second-word paths for a finished first word")
        second, _, state = second_paths[0]
        if state != "ended":
            predictions[initial] = (first, None, None, None)
            continue
        two_joint = JointWordBeam(
            base, BoundContext(two, first), ranker,
            lambda word, current_ranker: two.initial_distribution(first, word, current_ranker),
        )
        guesses = []
        for search in (joint, two_joint):
            paths = search.search_candidates([(second, 1.0)], WIDTH)
            if not paths:
                raise ValueError("No third-word paths for a finished second word")
            word, _, status = paths[0]
            guesses.append((word[0], word if status == "ended" else None))
        predictions[initial] = (first, second, *guesses)
    return predictions


def evaluate(base, contextual, ranker, two, triples):
    predictions = generated_predictions(base, contextual, ranker, two)
    totals = defaultdict(Counter)
    known = {}
    for source, items in triples.items():
        for (first, second, third), count in items.items():
            groups = [source, "combined"]
            if len(third) <= MAX_PREFIX:
                groups.append("short_third_word_only")
            if len(third) == MAX_PREFIX:
                groups.append("four_letter_third_word_only")
            pair = first, second
            if pair not in known:
                known[pair] = (
                    greedy_third(first, second, ranker, contextual),
                    greedy_third(first, second, ranker, contextual, two),
                )
            generated_first, generated_second, *generated = predictions[first[0]]
            first_two_correct = generated_first == first and generated_second == second
            for group in groups:
                scores = totals[group]
                scores["fragments"] += count
                scores["known_pair_indexed"] += count * (pair in two.initials)
                scores["generated_first_two_correct"] += count * first_two_correct
                scores["generated_third_available"] += count * (generated_second is not None)
                for index, name in enumerate(("one_word", "two_word")):
                    known_initial, known_word = known[pair][index]
                    scores[name + "_known_initial_correct"] += count * (known_initial == third[0])
                    scores[name + "_known_word_correct"] += count * (known_word == third)
                    guess = generated[index] if generated_second is not None else None
                    if guess is not None:
                        guessed_initial, guessed_word = guess
                        scores[name + "_generated_initial_correct"] += count * (guessed_initial == third[0])
                        scores[name + "_generated_word_correct"] += count * (guessed_word == third)
                        scores[name + "_phrase_correct"] += count * (first_two_correct and guessed_word == third)
                        scores[name + "_initial_correct_when_first_two_correct"] += count * (
                            first_two_correct and guessed_initial == third[0])
    if not totals.get("combined"):
        raise ValueError("No held-out three-word fragments")
    return {key: dict(scores) for key, scores in sorted(totals.items())}


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
    runs = {}
    for percent in FRACTIONS:
        start = time.perf_counter()
        def input_docs():
            return sampled(documents(args.bible_dir, args.oanc), percent)
        training_words, training_pairs, tests, counts = collect(input_docs())
        pair_counts = Counter(pair for pair, _ in training_triples(input_docs()))
        selected = set(sorted(pair_counts, key=lambda pair: (-pair_counts[pair], pair))[:MAX_CONTEXTS])
        initials, prefixes = build_index(input_docs(), selected)
        contexts = train_contexts(input_docs(), select_contexts(training_pairs))
        base = PrefixRanker(mapping, training_words)
        contextual = ContextualGenerator(base, contexts)
        ranker = NextInitialRanker(mapping, training_pairs)
        two = TwoWordContext(contextual, initials, prefixes)
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        scores = evaluate(base, contextual, ranker, two, tests)
        runs[str(percent)] = {
            "counts": {f"{source}_{partition}": dict(values) for (source, partition), values in sorted(counts.items())},
            "selected_pair_contexts": len(selected),
            "indexed_prefix_contexts": len(prefixes),
            "train_seconds": train_seconds,
            "evaluation_seconds": time.perf_counter() - start,
            "test": scores,
        }
        print(f"Finished {percent}% training: {scores['combined']}", flush=True)
    report = {
        "kind": "private_two_word_context_training_sizes",
        "data": {**hashes, "rights": "Private use approved by user, not legal clearance; OANC package and underlying rights require review."},
        "method": "Train on nested 25%, 50%, and 100% samples of approved training documents, chosen by document-content hash. Compare one preceding word with up to 512 most frequent two-word contexts; their initial and prefix counts are blended with the one-word distribution using prior strength 10, with full fallback for unseen contexts. Both arms use identical data and first/second generated words. Known-context diagnostics supply both actual words; autonomous chains receive only the actual first initial. Third initials are never supplied. Short test fragments require whitespace between words and first two words at most four letters.",
        "width": WIDTH, "prior_strength": PRIOR_STRENGTH,
        "runs": runs,
        "limitations": [
            "Increasing data size is not additional optimizer training or repeat passes over the same examples.",
            "Only the 512 most frequent two-word contexts are indexed; other contexts use the one-word fallback.",
            "Known-context diagnostics use greedy decoding; generated-chain scores use a 20-path beam with completed first-word selection.",
            "Third words longer than four characters cannot be completed, and unavailable predictions count as wrong.",
            "One-word training pairs may cross punctuation; two-word indexed triples and test fragments do not.",
            "Common words or near-duplicate documents may cross the split, and mapping calibration used corpus-wide initial frequencies.",
            "Neither the active chat model nor live universe state is used or modified.",
        ],
    }
    with args.report.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
