"""Rank second letters from pattern-labeled first letters in a private offline test."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import string

from word_start_experiment import WORD, documents, load_mapping

LETTERS = string.ascii_letters
OANC_SHA256 = "1a5c37b0227c176d2c5a00f23721e70e7489782e31ba14e58e47f56eff04d24f"
BIBLE_MANIFEST_SHA256 = "a7961064bb9d05f7680a7e8925ad7c5dc415f9da8ef4ca157f596a85544877fc"
MAPPING_SHA256 = "c2c6b7f5dfc8a161a900cc06ad8f3095b5241da6590ec3582db65868f4ffef58"


def file_hash(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def require_hash(path, expected):
    actual = file_hash(path)
    if actual != expected:
        raise ValueError(f"Unapproved or changed input: {path} (sha256 {actual})")
    return actual


def collect(docs):
    groups = defaultdict(Counter)
    totals = defaultdict(Counter)
    for source, partition, text in docs:
        if partition not in ("train", "test"):
            raise ValueError(f"Unknown partition: {partition}")
        totals[source, partition]["documents"] += 1
        for match in WORD.finditer(text):
            word = match.group()
            if len(word) < 2:
                totals[source, partition]["one_character_words"] += 1
            elif word[0] not in LETTERS or word[1] not in LETTERS:
                totals[source, partition]["non_ascii_or_nonletter_pairs"] += 1
            else:
                groups[source, partition][word[0], word[1]] += 1
                totals[source, partition]["pairs"] += 1
    return groups, totals


class PatternRanker:
    def __init__(self, mapping, training_pairs):
        self.representatives = {letter: identities[0] for letter, identities in mapping.items()}
        self.labels = {identity: letter for letter, identities in mapping.items() for identity in identities}
        if set(self.representatives) != set(LETTERS) or len(self.labels) != sum(map(len, mapping.values())):
            raise ValueError("Mapping must assign each pattern uniquely and include all 52 letters")
        self.contexts = defaultdict(Counter)
        self.overall = Counter()
        for (first, second), count in training_pairs.items():
            if first not in self.representatives or second not in self.representatives or count <= 0:
                raise ValueError("Invalid training pair")
            first_pattern = self.representatives[first]
            second_pattern = self.representatives[second]
            self.contexts[first_pattern][second_pattern] += count
            self.overall[second_pattern] += count
        if not self.overall:
            raise ValueError("No second-letter training pairs")

    def ranking(self, first_pattern):
        if first_pattern not in self.labels:
            raise ValueError("Unassigned first-letter pattern")
        first_letter = self.labels[first_pattern]
        context = self.contexts.get(self.representatives[first_letter])
        counts = context if context else self.overall
        ordered = sorted(LETTERS, key=lambda letter: (-counts[self.representatives[letter]], letter))
        total = sum(counts.values())
        return {
            "first": first_letter,
            "observations": sum(context.values()) if context else 0,
            "uses_global_fallback": not bool(context),
            "ranking": [{"letter": letter, "count": counts[self.representatives[letter]],
                         "probability": counts[self.representatives[letter]] / total}
                        for letter in ordered],
        }

    def baseline(self):
        return sorted(LETTERS, key=lambda letter: (-self.overall[self.representatives[letter]], letter))


def evaluate(ranker, pairs):
    baseline = ranker.baseline()
    baseline_top3 = [letter for letter in baseline
                     if ranker.overall[ranker.representatives[letter]] > 0][:3]
    rankings = {first: ranker.ranking(ranker.representatives[first]) for first in LETTERS}
    result = Counter()
    for (first, actual), count in pairs.items():
        ordered = [item["letter"] for item in rankings[first]["ranking"] if item["count"] > 0]
        result["pairs"] += count
        result["unknown_first"] += count * rankings[first]["uses_global_fallback"]
        result["top1_correct"] += count * (actual == ordered[0])
        result["top3_correct"] += count * (actual in ordered[:3])
        result["baseline_top1_correct"] += count * (actual == baseline[0])
        result["baseline_top3_correct"] += count * (actual in baseline_top3)
    if not result["pairs"]:
        raise ValueError("No held-out second-letter pairs")
    return dict(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bible-dir", type=Path)
    parser.add_argument("--oanc", type=Path)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--from-report", type=Path)
    parser.add_argument("--first")
    args = parser.parse_args()
    if args.from_report is not None:
        if args.first is None or len(args.first) != 1 or args.first not in LETTERS:
            parser.error("--from-report requires a single case-sensitive ASCII --first letter")
        if any(value is not None for value in (args.bible_dir, args.oanc, args.mapping, args.report)):
            parser.error("--from-report cannot be combined with training inputs or --report")
        report = json.loads(args.from_report.read_text(encoding="utf-8"))
        if report["kind"] != "private_second_letter_pattern_ranking":
            raise ValueError("Not a second-letter ranking report")
        print(json.dumps(report["first_letter_rankings"][args.first], indent=2))
        return
    if args.first is not None or any(value is None for value in (args.bible_dir, args.oanc, args.mapping, args.report)):
        parser.error("Training requires --bible-dir, --oanc, --mapping, and --report; use --first only with --from-report")
    hashes = {
        "bible_manifest_sha256": require_hash(args.bible_dir / "manifest.json", BIBLE_MANIFEST_SHA256),
        "oanc_sha256": require_hash(args.oanc, OANC_SHA256),
        "mapping_sha256": require_hash(args.mapping, MAPPING_SHA256),
    }
    mapping = load_mapping(args.mapping)
    groups, totals = collect(documents(args.bible_dir, args.oanc))
    training = sum((pairs for (source, partition), pairs in groups.items() if partition == "train"), Counter())
    ranker = PatternRanker(mapping, training)
    sources = sorted({source for source, partition in groups if partition == "test"})
    if not sources:
        raise ValueError("No held-out sources")
    all_test = sum((groups[source, "test"] for source in sources), Counter())
    results = {source: evaluate(ranker, groups[source, "test"]) for source in sources}
    results["combined"] = evaluate(ranker, all_test)
    report = {
        "kind": "private_second_letter_pattern_ranking",
        "data": {**hashes, "bible": "Staged World English Bible learning books only; Acts 3 and prior development/locked books excluded.",
                 "oanc": "Previously approved OANC archive, private use.",
                 "rights": "User approval is not legal clearance; OANC package and underlying rights require review."},
        "method": "Count case-sensitive first/second-letter pairs from training documents; rank pattern-labeled second letters given the first. Unknown first families use the training-wide second-letter ranking. Bible learning books are split by book; OANC documents by content hash.",
        "counts": {f"{source}_{partition}": dict(values) for (source, partition), values in sorted(totals.items())},
        "baseline_order": ranker.baseline(),
        "test": results,
        "first_letter_rankings": {letter: ranker.ranking(ranker.representatives[letter]) for letter in LETTERS},
        "limitations": [
            "This only predicts a second letter when one exists; one-character words, word-end decisions, and pairs with non-ASCII or nonletter characters are excluded.",
            "One representative pattern per letter is counted; all variants for that letter share its ranking. This is equivalent to ordinary letter-pair counting, not learned quark bonding.",
            "The pattern mapping was calibrated using corpus-wide first-letter frequencies including test material; the second-letter counts themselves use only the training split.",
            "Exact duplicate OANC documents share a split, but near duplicates and common words can appear in both splits.",
            "Teacher-forced second-letter scores do not establish accurate four-character words, complete word generation, chat quality, or independent-source generalization.",
        ],
    }
    with args.report.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps({"counts": report["counts"], "baseline_top3": report["baseline_order"][:3],
                      "test": results,
                      "examples": {letter: report["first_letter_rankings"][letter]["ranking"][:5]
                                   for letter in ("t", "T", "s", "S")}}, indent=2))


if __name__ == "__main__":
    main()
