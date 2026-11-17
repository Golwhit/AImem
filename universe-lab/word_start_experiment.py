"""Private word-start comparison; no chat model or universe state is changed."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import string
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neural_skeletons import chapter_text

WORD = re.compile(r"[^\W\d_]+(?:['\u2019][^\W\d_]+)*", re.UNICODE)
LETTERS = set(string.ascii_letters)


def load_mapping(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    patterns = {}
    seen = set()
    for row in data["mapping"]:
        letter = row["letter"]
        identity = (row["bits"], tuple(row["bonds"]))
        if letter not in LETTERS or len(letter) != 1 or identity in seen:
            raise ValueError("Invalid or duplicate pattern mapping")
        seen.add(identity)
        patterns.setdefault(letter, []).append(identity)
    if set(patterns) != LETTERS:
        raise ValueError("Pattern mapping must include all 52 letters")
    return {letter: sorted(identities) for letter, identities in patterns.items()}


def split(identity):
    return "test" if int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") % 5 == 0 else "train"


def documents(bible_dir, oanc_path):
    manifest = json.loads((bible_dir / "manifest.json").read_text(encoding="utf-8"))
    bible_path = bible_dir / "engwebp_readaloud.zip"
    with bible_path.open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != manifest["zip_sha256"]:
            raise ValueError("Bible archive hash mismatch")
    with zipfile.ZipFile(bible_path) as archive:
        for source in manifest["sources"]:
            if source["group"] != "learn":
                continue
            text = chapter_text(archive.read(source["file"]))
            if hashlib.sha256(text.encode()).hexdigest() != source["text_sha256"]:
                raise ValueError(f"Bible chapter hash mismatch: {source['file']}")
            yield "bible", split(f'bible-book:{source["book"]}'.encode()), text
    with zipfile.ZipFile(oanc_path) as archive:
        for name in sorted(archive.namelist()):
            if not name.startswith("OANC/data/") or not name.endswith(".txt"):
                continue
            raw = archive.read(name)
            yield "oanc", split(b"oanc-document:" + hashlib.sha256(raw).digest()), raw.decode("utf-8-sig")


def observations(text):
    previous = None
    for match in WORD.finditer(text):
        word = match.group()
        initial = word[0]
        if initial in LETTERS and previous is not None:
            yield previous, initial
        previous = word.casefold()


def most_common(counts):
    return min(counts, key=lambda letter: (-counts[letter], letter))


def train(docs):
    contexts = defaultdict(Counter)
    initials = Counter()
    counts = Counter()
    for source, partition, text in docs:
        if partition != "train":
            continue
        counts[source + "_documents"] += 1
        for context, letter in observations(text):
            contexts[context][letter] += 1
            initials[letter] += 1
            counts[source + "_observations"] += 1
    if not initials:
        raise ValueError("No training observations")
    predictions = {context: most_common(letters) for context, letters in contexts.items()}
    return predictions, most_common(initials), counts


def evaluate(docs, predictions, baseline, patterns):
    totals = defaultdict(Counter)
    inverse = {pattern: letter for letter, identities in patterns.items() for pattern in identities}
    used = defaultdict(set)
    for source, partition, text in docs:
        if partition != "test":
            continue
        totals[source]["documents"] += 1
        for context, actual in observations(text):
            known = context in predictions
            predicted = predictions.get(context, baseline)
            choices = patterns[predicted]
            index = int.from_bytes(hashlib.sha256(context.encode()).digest()[:8], "big") % len(choices)
            pattern = choices[index]
            decoded = inverse[pattern]
            used[source].add(pattern)
            totals[source]["observations"] += 1
            totals[source]["known_contexts"] += known
            totals[source]["baseline_correct"] += actual == baseline
            totals[source]["context_correct"] += actual == predicted
            totals[source]["decoded_pattern_correct"] += actual == decoded
    if not totals or any(not item["observations"] for item in totals.values()):
        raise ValueError("No test observations for an evaluated source")
    for source in totals:
        totals[source]["distinct_patterns_used"] = len(used[source])
    return {source: dict(counts) for source, counts in sorted(totals.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bible-dir", type=Path, required=True)
    parser.add_argument("--oanc", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    patterns = load_mapping(args.mapping)
    predictions, baseline, train_counts = train(documents(args.bible_dir, args.oanc))
    results = evaluate(documents(args.bible_dir, args.oanc), predictions, baseline, patterns)
    with args.oanc.open("rb") as file:
        oanc_hash = hashlib.file_digest(file, "sha256").hexdigest()
    report = {
        "method": "Predict the next case-sensitive initial from the preceding case-folded word, using training counts; unseen contexts fall back to the training-set most frequent initial. Encode each prediction with one of its assigned four-quark patterns, selected deterministically from the context, then decode it. Documents split by content hash; Bible learning books split by book. The mapping was previously fitted using all OANC and Bible learning-split initial frequencies, including this test text: the evaluation is held out from predictor fitting, not independent of mapping construction.",
        "data": {"bible": "World English Bible staged learning books only; Acts 3 and earlier development/locked books excluded",
                 "oanc_sha256": oanc_hash, "oanc_source": "https://anc.org/data/oanc/download/",
                 "mapping_sha256": hashlib.sha256(args.mapping.read_bytes()).hexdigest(),
                 "rights": "User approved private use of both sources on 2026-11-15; that declaration is not legal clearance. OANC package license and source rights require review before release."},
        "training": {**dict(train_counts), "unique_contexts": len(predictions), "baseline": baseline},
        "test": results,
        "limitations": ["One word of context; no neural model or letter-bond simulation.",
                        "Selecting an assigned pattern guarantees label round-tripping and does not test whether quark formation predicts letters.",
                        "Exact duplicate OANC documents share a split; near duplicates may cross it.",
                        "Tokenization is word-based and omits first words of each document; non-ASCII initials are not targets.",
                        "Uncertainty and performance on new sources are not established."],
    }
    with args.report.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps({"report": str(args.report), "training": report["training"], "test": results}, indent=2))


if __name__ == "__main__":
    main()
