"""Test pattern-family word prefixes and word endings in an isolated experiment."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from second_letter_experiment import (
    BIBLE_MANIFEST_SHA256, LETTERS, MAPPING_SHA256, OANC_SHA256, require_hash,
)
from word_start_experiment import WORD, documents, load_mapping

END = "<END>"
MAX_PREFIX = 4


def collect(docs):
    words = defaultdict(Counter)
    totals = defaultdict(Counter)
    for source, partition, text in docs:
        if partition not in ("train", "test"):
            raise ValueError(f"Unknown partition: {partition}")
        totals[source, partition]["documents"] += 1
        for match in WORD.finditer(text):
            word = match.group()
            if all(char in LETTERS for char in word):
                words[source, partition][word] += 1
                totals[source, partition]["words"] += 1
            else:
                totals[source, partition]["excluded_non_ascii_or_nonletter"] += 1
    return words, totals


class PrefixRanker:
    def __init__(self, mapping, training):
        self.representatives = {letter: identities[0] for letter, identities in mapping.items()}
        self.labels = {identity: letter for letter, identities in mapping.items() for identity in identities}
        if set(self.representatives) != set(LETTERS) or len(self.labels) != sum(map(len, mapping.values())):
            raise ValueError("Mapping must assign each pattern uniquely and include all 52 letters")
        self.contexts = defaultdict(Counter)
        self.last = defaultdict(Counter)
        self.position = defaultdict(Counter)
        for word, count in training.items():
            if not word or any(char not in LETTERS for char in word) or count <= 0:
                raise ValueError("Invalid training word")
            patterns = tuple(self.representatives[char] for char in word[:MAX_PREFIX])
            for length in range(1, min(len(word), MAX_PREFIX) + 1):
                target = self.representatives[word[length]] if length < len(word) else None
                self.contexts[patterns[:length]][target] += count
                self.last[length, patterns[length - 1]][target] += count
                self.position[length][target] += count
        if not self.position:
            raise ValueError("No training words")
        self.cache = {}

    def _label(self, pattern):
        return END if pattern is None else self.labels[pattern]

    def rank(self, prefix, *, last_only=False):
        if not 1 <= len(prefix) <= MAX_PREFIX or any(char not in LETTERS for char in prefix):
            raise ValueError("Prefix must contain one to four case-sensitive ASCII letters")
        cache_key = prefix, last_only
        if cache_key not in self.cache:
            patterns = tuple(self.representatives[char] for char in prefix)
            context = None if last_only else self.contexts.get(patterns)
            last = self.last.get((len(prefix), patterns[-1]))
            counts = context or last or self.position.get(len(prefix))
            if not counts:
                raise ValueError(f"No training evidence for prefix length {len(prefix)}")
            fallback = "prefix" if context else "last_letter" if last else "position"
            order = sorted(counts, key=lambda pattern: (-counts[pattern], self._label(pattern)))
            self.cache[cache_key] = (tuple(self._label(pattern) for pattern in order),
                                     sum(counts.values()), fallback, counts)
        return self.cache[cache_key]

    def generate_up_to_four(self, initial, *, last_only=False):
        prefix = initial
        while True:
            target = self.rank(prefix, last_only=last_only)[0][0]
            if target == END:
                return prefix, True
            if len(prefix) == MAX_PREFIX:
                return prefix, False
            prefix += target

    def saved_counts(self):
        def encode(counter):
            return {self._label(pattern): count for pattern, count in counter.items()}
        return {
            "prefixes": {"".join(self.labels[pattern] for pattern in prefix): encode(counts)
                         for prefix, counts in self.contexts.items()},
            "last_letters": {f"{length}:{self.labels[pattern]}": encode(counts)
                             for (length, pattern), counts in self.last.items()},
            "positions": {str(length): encode(counts) for length, counts in self.position.items()},
        }


def evaluate(ranker, words):
    steps = {str(length): Counter() for length in range(1, MAX_PREFIX + 1)}
    generation = Counter()
    initial_predictions = {}
    for word, count in words.items():
        for length in range(1, min(len(word), MAX_PREFIX) + 1):
            prefix = word[:length]
            actual = word[length] if length < len(word) else END
            ordered, _, fallback, _ = ranker.rank(prefix)
            baseline = ranker.rank(prefix, last_only=True)[0]
            metrics = steps[str(length)]
            metrics["observations"] += count
            metrics["top1_correct"] += count * (ordered[0] == actual)
            metrics["top3_correct"] += count * (actual in ordered[:3])
            metrics["last_letter_top1_correct"] += count * (baseline[0] == actual)
            metrics["last_letter_top3_correct"] += count * (actual in baseline[:3])
            metrics["fallback"] += count * (fallback != "prefix")
            metrics["actual_end"] += count * (actual == END)
            metrics["predicted_end"] += count * (ordered[0] == END)
            metrics["correct_end"] += count * (ordered[0] == actual == END)
        if word[0] not in initial_predictions:
            initial_predictions[word[0]] = (
                ranker.generate_up_to_four(word[0]),
                ranker.generate_up_to_four(word[0], last_only=True),
            )
        for name, (predicted, ended) in zip(("prefix", "last_letter"), initial_predictions[word[0]]):
            if len(word) == 4:
                generation[f"{name}_four_letter_targets"] += count
                generation[f"{name}_correct_complete_four"] += count * (ended and predicted == word)
                generation[f"{name}_correct_first_four"] += count * (predicted == word)
            if len(predicted) == 4 and ended:
                generation[f"{name}_generated_four"] += count
                generation[f"{name}_generated_four_correct"] += count * (predicted == word)
    if not words:
        raise ValueError("No held-out words")
    return {"steps": {length: dict(values) for length, values in steps.items()},
            "generation": dict(generation)}


def lookup(saved, prefix):
    if not 1 <= len(prefix) <= MAX_PREFIX or any(char not in LETTERS for char in prefix):
        raise ValueError("Prefix must contain one to four case-sensitive ASCII letters")
    context = saved["prefixes"].get(prefix)
    last = saved["last_letters"].get(f"{len(prefix)}:{prefix[-1]}")
    counts = context or last or saved["positions"].get(str(len(prefix)))
    if not counts:
        raise ValueError(f"No training evidence for prefix length {len(prefix)}")
    total = sum(counts.values())
    return {"prefix": prefix, "fallback": "prefix" if context else "last_letter" if last else "position",
            "observations": sum(context.values()) if context else 0,
            "ranking": [{"next": target, "count": count, "probability": count / total}
                        for target, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bible-dir", type=Path)
    parser.add_argument("--oanc", type=Path)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--from-report", type=Path)
    parser.add_argument("--prefix")
    args = parser.parse_args()
    if args.from_report is not None:
        if args.prefix is None or any(value is not None for value in (args.bible_dir, args.oanc, args.mapping, args.report)):
            parser.error("--from-report requires --prefix and cannot be combined with training inputs")
        report = json.loads(args.from_report.read_text(encoding="utf-8"))
        if report["kind"] != "private_word_prefix_pattern_ranking":
            raise ValueError("Not a word-prefix ranking report")
        print(json.dumps(lookup(report["model"], args.prefix), indent=2))
        return
    if args.prefix is not None or any(value is None for value in (args.bible_dir, args.oanc, args.mapping, args.report)):
        parser.error("Training requires --bible-dir, --oanc, --mapping, and --report")
    hashes = {
        "bible_manifest_sha256": require_hash(args.bible_dir / "manifest.json", BIBLE_MANIFEST_SHA256),
        "oanc_sha256": require_hash(args.oanc, OANC_SHA256),
        "mapping_sha256": require_hash(args.mapping, MAPPING_SHA256),
    }
    mapping = load_mapping(args.mapping)
    groups, totals = collect(documents(args.bible_dir, args.oanc))
    training = sum((words for (source, partition), words in groups.items() if partition == "train"), Counter())
    ranker = PrefixRanker(mapping, training)
    sources = sorted({source for source, partition in groups if partition == "test"})
    if not sources:
        raise ValueError("No held-out sources")
    all_test = sum((groups[source, "test"] for source in sources), Counter())
    results = {source: evaluate(ranker, groups[source, "test"]) for source in sources}
    results["combined"] = evaluate(ranker, all_test)
    saved = ranker.saved_counts()
    report = {
        "kind": "private_word_prefix_pattern_ranking",
        "data": {**hashes, "bible": "Staged World English Bible learning books only; Acts 3 and earlier development/locked books excluded.",
                 "oanc": "Previously approved OANC archive, private use.",
                 "rights": "User approval is not legal clearance; OANC package and underlying rights require review."},
        "method": "Count next characters or end-of-word after case-sensitive word prefixes of one to four letters. Prefixes use pattern families; unseen prefixes fall back to the last letter at the same position, then position-wide counts. Bible learning books split by book; OANC documents split by content hash.",
        "counts": {f"{source}_{partition}": dict(values) for (source, partition), values in sorted(totals.items())},
        "test": results,
        "examples": {prefix: lookup(saved, prefix) for prefix in ("st", "str")},
        "model": saved,
        "limitations": [
            "Only words entirely composed of ASCII letters are included; apostrophes, accented letters and other word characters are excluded.",
            "The fixed pattern families encode letters; this does not simulate or train quark bonding.",
            "The pattern mapping was calibrated using corpus-wide first-letter frequencies including test sources; prefix counts use only training documents.",
            "Repeated words and near-duplicate documents may cross the split, so this is not a test of entirely new vocabulary or sources.",
            "Next-character scores use the correct preceding prefix. Generation starts with the actual first letter and uses greedy choices through the first four letters and a stop decision.",
        ],
    }
    with args.report.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps({"counts": report["counts"], "test": results,
                      "examples": {prefix: {**value, "ranking": value["ranking"][:5]}
                                   for prefix, value in report["examples"].items()}}, indent=2))


if __name__ == "__main__":
    main()
