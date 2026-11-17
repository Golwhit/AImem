"""Small offline word-start comparison against the active conversation model."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import string
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neural_chat import CONVERSATION_MODEL, load_active
from neural_plasticity import PlasticityContext
from word_start_experiment import load_mapping

CORPUS = ROOT / "fixtures/neural-user-corpus-2026-11-14.json"
APPROVED_SHA256 = "af5e5dd0b1f95630a3220e6f4e40b35b417c7c3d039b3db79edb376bd65dfdee"
MEAL_INDICES = (0, 1, 2, 10, 11, 12, 13, 14, 15, 16)
ALPHABET = string.ascii_letters
WORD = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*")
SMOOTHING = 10.0
TEST_TEXTS = (
    "The bird rests nearby.",
    "A dog follows the cat.",
    "Alice finds a book.",
    "Bob holds a cup.",
    "The cat sees Bob.",
    "A quiet fox runs away.",
    "Is the dog sleeping?",
    "Yes, the dog is resting.",
    "No, the bird is flying.",
    "The box holds five cups.",
    "Pack the bag with books.",
    "The brown dog sees a fox.",
)


def pairs(text):
    words = list(WORD.finditer(text))
    for previous, current in zip(words, words[1:]):
        yield previous.group()[0], current.group()[0], "text: " + text[:current.start()]


def meal_counts(texts):
    by_previous = defaultdict(Counter)
    overall = Counter()
    for text in texts:
        for previous, actual, _ in pairs(text):
            by_previous[previous][actual] += 1
            overall[actual] += 1
    if not overall:
        raise ValueError("The reviewed meal contains no word-start pairs")
    return by_previous, overall


def factors(previous, counts, overall):
    """Shrink the small context sample toward the meal's smoothed global prior."""
    context = counts.get(previous)
    if not context:
        return {letter: 1.0 for letter in ALPHABET}
    global_total = sum(overall.values()) + len(ALPHABET)
    context_total = sum(context.values())
    result = {}
    for letter in ALPHABET:
        global_probability = (overall[letter] + 1) / global_total
        conditional = (context[letter] + SMOOTHING * global_probability) / (context_total + SMOOTHING)
        result[letter] = conditional / global_probability
    return result


def first_mass(model, probabilities):
    result = Counter()
    for token, payload in model.payloads.items():
        if payload:
            result[payload[0]] += probabilities[token]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    raw = CORPUS.read_bytes()
    if hashlib.sha256(raw).hexdigest() != APPROVED_SHA256:
        raise ValueError("The approved corpus file hash does not match")
    corpus = json.loads(raw)
    meal = [corpus["training"][index] for index in MEAL_INDICES]
    if set(meal) & set(TEST_TEXTS):
        raise ValueError("Meal and evaluation sentences must be separate")
    mapping = load_mapping(args.mapping)
    representative = {letter: identities[0] for letter, identities in mapping.items()}
    inverse = {pattern: letter for letter, pattern in representative.items()}
    if len(inverse) != 52:
        raise ValueError("Representatives must map uniquely to all 52 letters")
    counts, overall = meal_counts(meal)
    active_path = ROOT / ".data/neural-chat/active.json"
    before_active = active_path.read_bytes()
    record, model, index = load_active(ROOT)
    if record["model"] != CONVERSATION_MODEL or not isinstance(model, PlasticityContext) or index is not None:
        raise ValueError("Expected the pinned conversation model without an index")
    checkpoint_path = ROOT / ".data/neural-chat" / (record["checkpoint_sha256"] + ".json")
    before_checkpoint = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if before_checkpoint != record["checkpoint_sha256"]:
        raise ValueError("Active model checkpoint hash mismatch")

    scores = {"control": Counter(), "pattern_sidecar": Counter()}
    seconds = Counter()
    examples = 0
    unavailable = 0
    used = Counter()
    baseline = min(overall, key=lambda letter: (-overall[letter], letter))
    baseline_correct = 0
    for text in TEST_TEXTS:
        for previous, actual, prefix in pairs(text):
            pattern = representative[previous]
            family = inverse[pattern]
            multiplier = factors(family, counts, overall)
            prior = model.prior(prefix)
            start = time.perf_counter()
            probabilities = model.model.forward(model.context(prefix), output_prior=prior)[2]
            seconds["control"] += time.perf_counter() - start
            start = time.perf_counter()
            candidate_prior = prior[:]
            for token, payload in model.payloads.items():
                if payload and payload[0] in ALPHABET:
                    candidate_prior[token] *= multiplier[payload[0]]
            candidate = model.model.forward(model.context(prefix), output_prior=candidate_prior)[2]
            seconds["pattern_sidecar"] += time.perf_counter() - start
            masses = {"control": first_mass(model, probabilities), "pattern_sidecar": first_mass(model, candidate)}
            if not any(payload[0] == actual for payload in model.payloads.values() if payload):
                unavailable += 1
            for arm, mass in masses.items():
                chosen = min(mass, key=lambda first: (-mass[first], first))
                scores[arm]["correct"] += chosen == actual
                scores[arm]["negative_log_probability"] += -math.log(max(mass[actual], 1e-300))
            baseline_correct += baseline == actual
            used["known_family"] += family in counts
            examples += 1
    if not examples:
        raise ValueError("No held-out word starts to evaluate")
    if active_path.read_bytes() != before_active or hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != before_checkpoint:
        raise RuntimeError("The active model changed during the comparison")
    arms = {arm: {"correct": int(values["correct"]), "accuracy": values["correct"] / examples,
                  "mean_initial_log_loss": values["negative_log_probability"] / examples,
                  "forward_wall_seconds": seconds[arm]} for arm, values in scores.items()}
    report = {
        "kind": "offline_pattern_sidecar_short_meal",
        "model_checkpoint_sha256": before_checkpoint,
        "meal": {"corpus_file_sha256": APPROVED_SHA256, "indices": MEAL_INDICES,
                 "sentences": len(meal), "word_start_pairs": sum(overall.values()),
                 "approval": "Previously approved exact user-supplied corpus file; user requested a short isolated meal."},
        "pattern_mapping_sha256": hashlib.sha256(args.mapping.read_bytes()).hexdigest(),
        "evaluation": {"provenance": "New AI-assisted developer-written diagnostic sentences, not used to fit the sidecar.",
                       "sentences": len(TEST_TEXTS), "word_starts": examples,
                       "starts_with_unavailable_initial": unavailable, "known_previous_families": used["known_family"]},
        "arms": arms,
        "meal_most_common_baseline": {"initial": baseline, "correct": baseline_correct,
                                      "accuracy": baseline_correct / examples},
        "limitations": [
            "The sidecar maps previous initials to representative pattern families, then aggregates by letter; it does not simulate or train quark bonding.",
            "The mapping is reversible here: the sidecar is equivalent to directly counting preceding letters, so this cannot establish a benefit from the patterns themselves.",
            "The active neural weights are frozen in both arms; only sidecar frequency counts are fitted.",
            "The new evaluation sentences are a tiny developer-visible diagnostic, not independent generalization evidence.",
            "The fixed many-to-one mapping was fitted using broader corpus frequencies; it was not learned from this meal.",
            "This is a teacher-forced word-start diagnostic, not a chat-quality or generated-word test.",
            "All output payloads are grouped by their first character; a winning non-letter counts as wrong and unavailable target initials remain in the denominator.",
        ],
        "active_model_unchanged": True,
    }
    with args.report.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps({key: report[key] for key in ("meal", "evaluation", "arms", "meal_most_common_baseline", "active_model_unchanged")}, indent=2))


if __name__ == "__main__":
    main()
