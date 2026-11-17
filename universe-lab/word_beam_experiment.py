"""Compare several preceding-word hypotheses on an isolated short-word task."""

import argparse
from collections import Counter
import json
from pathlib import Path

from contextual_word_experiment import (
    ContextualGenerator, collect, select_contexts, train_contexts,
)
from next_word_after_generation import NextInitialRanker
from second_letter_experiment import (
    BIBLE_MANIFEST_SHA256, LETTERS, MAPPING_SHA256, OANC_SHA256, require_hash,
)
from word_prefix_experiment import END, MAX_PREFIX, PrefixRanker
from word_start_experiment import documents, load_mapping

BEAM_WIDTHS = (5, 20)


def beam_words(base, initial, width):
    if len(initial) != 1 or initial not in LETTERS or width < 1:
        raise ValueError("Beam needs one ASCII initial and a positive width")
    paths = [(initial, 1.0, "active")]
    for _ in range(MAX_PREFIX):
        expanded = []
        for prefix, probability, state in paths:
            if state != "active":
                expanded.append((prefix, probability, state))
                continue
            _, total, _, counts = base.rank(prefix)
            for pattern, count in counts.items():
                label = END if pattern is None else base.labels[pattern]
                score = probability * count / total
                if label == END:
                    expanded.append((prefix, score, "ended"))
                elif len(prefix) < MAX_PREFIX:
                    expanded.append((prefix + label, score, "active"))
                else:
                    expanded.append((prefix + label, score, "continues"))
        paths = sorted(expanded, key=lambda item: (-item[1], item[0], item[2]))[:width]
    return [(word, probability) for word, probability, state in paths if state == "ended"]


def next_initial_distribution(word, ranker):
    context = tuple(ranker.representatives[char] for char in word)
    counts = ranker.contexts.get(context) or ranker.overall
    denominator = sum(counts.values()) + len(LETTERS)
    return {letter: (counts[ranker.representatives[letter]] + 1) / denominator for letter in LETTERS}


def select_with_next_initial(candidates, initial, ranker):
    if len(initial) != 1 or initial not in LETTERS:
        raise ValueError("Next initial must be one ASCII letter")
    scored = []
    for word, probability in candidates:
        likelihood = next_initial_distribution(word, ranker)[initial]
        scored.append((word, probability * likelihood))
    return min(scored, key=lambda item: (-item[1], item[0]))[0] if scored else None


def evaluate(base, contextual, next_ranker, windows):
    scores = Counter()
    prior_beams = {}
    choices = {}
    current_cache = {}
    next_cache = {}
    for (previous, actual, following), count in windows.items():
        scores["windows"] += count
        first = previous[0]
        current_initial = actual[0]
        if first not in prior_beams:
            prior_beams[first] = {width: beam_words(base, first, width) for width in BEAM_WIDTHS}
        key = first, current_initial
        if key not in choices:
            greedy, ended = base.generate_up_to_four(first)
            choices[key] = {"greedy": greedy if ended else None}
            for width in BEAM_WIDTHS:
                beam = prior_beams[first][width]
                choices[key][f"beam{width}_prior"] = beam[0][0] if beam else None
                choices[key][f"beam{width}_next_initial"] = select_with_next_initial(
                    beam, current_initial, next_ranker)
        for width in BEAM_WIDTHS:
            scores[f"beam{width}_contains_previous"] += count * any(
                word == previous for word, _ in prior_beams[first][width])
        greedy_available = choices[key]["greedy"] is not None
        scores["greedy_available_windows"] += count * greedy_available
        for arm, predicted_previous in choices[key].items():
            if predicted_previous is None:
                continue
            scores[arm + "_previous_available"] += count
            scores[arm + "_previous_correct"] += count * (predicted_previous == previous)
            current_key = predicted_previous, current_initial
            if current_key not in current_cache:
                current_cache[current_key] = contextual.generate_up_to_four(*current_key)
            predicted, stopped, _ = current_cache[current_key]
            scores[arm + "_current_stopped"] += count * stopped
            scores[arm + "_current_correct"] += count * (stopped and predicted == actual)
            if greedy_available:
                scores[arm + "_previous_correct_when_greedy_available"] += count * (predicted_previous == previous)
                scores[arm + "_current_correct_when_greedy_available"] += count * (stopped and predicted == actual)
            if stopped:
                if predicted not in next_cache:
                    next_cache[predicted] = next_ranker.rank(predicted)[0][0]
                scores[arm + "_following_correct"] += count * (next_cache[predicted] == following)
                if greedy_available:
                    scores[arm + "_following_correct_when_greedy_available"] += count * (
                        next_cache[predicted] == following)
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
    contextual = ContextualGenerator(base, contexts)
    next_ranker = NextInitialRanker(mapping, training_pairs)
    if not windows:
        raise ValueError("No held-out sources")
    all_test = sum(windows.values(), Counter())
    four_target = Counter({window: count for window, count in all_test.items() if len(window[1]) == MAX_PREFIX})
    four_previous = Counter({window: count for window, count in all_test.items() if len(window[0]) == MAX_PREFIX})
    results = {source: evaluate(base, contextual, next_ranker, values)
               for source, values in sorted(windows.items())}
    results["combined"] = evaluate(base, contextual, next_ranker, all_test)
    results["four_letter_target_only"] = evaluate(base, contextual, next_ranker, four_target)
    results["four_letter_previous_only"] = evaluate(base, contextual, next_ranker, four_previous)
    report = {
        "kind": "private_short_word_beam",
        "data": {**hashes, "bible": "Staged World English Bible learning books only; Acts 3 and earlier development/locked books excluded.",
                 "oanc": "Previously approved OANC archive, private use.",
                 "rights": "User approval is not legal clearance; OANC package and underlying rights require review."},
        "method": "Compare greedy previous-word generation with beams of 5 and 20 paths. Beam priors multiply next-character and end probabilities. Finished, active, and continuing-beyond-four paths compete for beam slots; only finished words can supply context. Compare the most probable completed word with the completed word re-ranked using the supplied current-word initial and add-one-smoothed training transition counts. Generate the current word using that chosen context, then predict the following initial if the current word ends.",
        "beam_widths": BEAM_WIDTHS,
        "counts": {f"{source}_{partition}": dict(values) for (source, partition), values in sorted(totals.items())},
        "test": results,
        "limitations": [
            "The actual first letters of both previous and current words are supplied; next-initial re-ranking uses information unavailable until the current initial is known.",
            "Widths count active, ended and over-four paths, not guaranteed numbers of finished candidate words. Unavailable predictions count as incorrect in full-window scores.",
            "The evaluated previous words are known to have at most four letters; selecting only finished beam candidates uses that length restriction.",
            "Only completely ASCII-letter words are included. Windows do not skip excluded words or cross documents, but can cross sentence punctuation.",
            "Common words and near-duplicate documents may cross the split. The pattern mapping used corpus-wide initial frequencies including test sources.",
            "The fixed pattern families encode letters and do not simulate quark bonding.",
        ],
    }
    with args.report.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps({key: report[key] for key in ("beam_widths", "counts", "test")}, indent=2))


if __name__ == "__main__":
    main()
