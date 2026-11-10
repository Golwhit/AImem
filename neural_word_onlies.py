"""Word-level only states: words that are always followed (forward) or preceded (backward) by one word.

Builds on the character only states in neural_onlies.py. Earned in reading order from approved
training text; later different neighbours become compatible variants, never deleting the original
path. Guidance only: off-path characters keep a penalty weight and are never forbidden.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from uuid import uuid4

from neural_counting import KneserNey
from neural_experiment import NeuralModel
from neural_layers import DevelopmentalPrior, evaluate
from neural_onlies import OnlyStates
from neural_rounds import load_acts
from original_generator import atomic_json


WORD = re.compile(r"[A-Za-z’']+")
HISTORY = 80
SUPPORTS = (2, 3)
PENALTIES = (0.2, 0.5)
FRACTIONS = (0.1, 0.25, 0.5, 1.0)
CHARACTER_PENALTY = 0.5


def earn_pairs(pairs: list[tuple[str, str]], min_support: int) -> dict[str, dict]:
    seen: dict[str, Counter] = {}
    states: dict[str, dict] = {}
    for at, (block, outcome) in enumerate(pairs):
        state = states.get(block)
        if state is not None:
            state["support"] += 1
            if outcome != state["path"] and outcome not in state["variants"]:
                state["variants"][outcome] = at
            continue
        counts = seen.setdefault(block, Counter())
        counts[outcome] += 1
        if len(counts) == 1 and counts[outcome] >= min_support:
            states[block] = {"path": outcome, "variants": {}, "support": counts[outcome], "earned_at_word": at}
            del seen[block]
    return states


class WordOnlies:
    def __init__(self, source: str, *, min_support: int = 2, forward: bool = True, backward: bool = True):
        words = WORD.findall(source)
        if len(words) < 2 or min_support < 1:
            raise ValueError("Word only states need at least two words and positive support")
        self.min_support = min_support
        self.use_forward = forward
        self.use_backward = backward
        self.vocabulary = sorted(set(words))
        self.forward_states = earn_pairs(list(zip(words, words[1:])), min_support)
        self.backward_states = earn_pairs(list(zip(words[1:], words)), min_support)
        self.operations = Counter(build_word_pairs=2 * (len(words) - 1), build_tokenized_characters=len(source))

    @staticmethod
    def allowed(state: dict) -> set[str]:
        return {state["path"], *state["variants"]}

    def split(self, prefix: str) -> tuple[str | None, str]:
        """(previous complete word, current partial word)."""
        partial = re.search(r"[A-Za-z’']*$", prefix).group()
        words = WORD.findall(prefix[:len(prefix) - len(partial)])
        return (words[-1] if words else None), partial

    def multipliers(self, prefix: str, model: NeuralModel, penalty: float) -> list[float]:
        weights = [1.0] * model.size
        previous, partial = self.split(prefix[-HISTORY:])
        if previous is None:
            return weights
        if self.use_forward and (state := self.forward_states.get(previous)):
            self.operations["predict_forward_word_checks"] += len(model.tokens)
            targets = [word for word in self.allowed(state) if word.startswith(partial)]
            if targets:
                favoured = {word[len(partial)] for word in targets if len(word) > len(partial)}
                ends = any(word == partial for word in targets)
                for char, token in model.tokens.items():
                    if not (char in favoured or (ends and not WORD.fullmatch(char))):
                        weights[token] *= penalty
        if self.use_backward:
            self.operations["predict_backward_word_checks"] += len(self.vocabulary)
            candidates = [word for word in self.vocabulary if word.startswith(partial)]
            for char, token in model.tokens.items():
                if WORD.fullmatch(char):
                    reachable = [word for word in candidates if len(word) > len(partial) and word[len(partial)] == char]
                else:
                    reachable = [partial] if partial in self.vocabulary else []
                if reachable and all(
                    (state := self.backward_states.get(word)) and previous not in self.allowed(state)
                    for word in reachable
                ):
                    weights[token] *= penalty
        return weights

    def summary(self) -> dict:
        def describe(states: dict[str, dict]) -> dict:
            mutated = sum(bool(state["variants"]) for state in states.values())
            strongest = sorted(states.items(), key=lambda item: -item[1]["support"])[:12]
            return {"count": len(states), "pure": len(states) - mutated, "mutated": mutated,
                    "strongest": [{"word": word, **state} for word, state in strongest]}
        return {"vocabulary": len(self.vocabulary), "min_support": self.min_support,
                "forward": describe(self.forward_states), "backward": describe(self.backward_states)}

    def ledger(self) -> dict:
        return {"forward": self.forward_states, "backward": self.backward_states,
                "state": "unconfirmed candidates; never confirmed forms or permanent admissions"}


def hold_rates(states: WordOnlies, text: str) -> dict:
    words = WORD.findall(text)
    result = {}
    for name, table, pairs in (("forward", states.forward_states, zip(words, words[1:])),
                               ("backward", states.backward_states, zip(words[1:], words))):
        counts = Counter()
        for block, outcome in pairs:
            if state := table.get(block):
                counts["path" if outcome == state["path"] else "variant" if outcome in state["variants"] else "new"] += 1
        total = sum(counts.values())
        result[name] = {**counts, "uses": total, "held_rate": (counts["path"] + counts["variant"]) / total if total else None}
    return result


# name: (character onlies, word forward, word backward)
ARMS = {
    "layers_12": (False, False, False),
    "char_onlies_12": (True, False, False),
    "word_forward_12": (False, True, False),
    "word_backward_12": (False, False, True),
    "word_both_12": (False, True, True),
    "char_and_word_both_12": (True, True, True),
}


def priors(name: str, train: str, text: str, model: NeuralModel, support: int, penalty: float):
    characters, forward, backward = ARMS[name]
    base = DevelopmentalPrior(train, layer_zero=False, layer_one=True, layer_two=True)
    character = OnlyStates(train) if characters else None
    words = WordOnlies(train, min_support=support, forward=forward, backward=backward) if forward or backward else None
    rows = []
    for offset in range(len(text)):
        weights = base.prior(text[max(0, offset - 4):offset], model)
        if character:
            factors = character.multipliers(text[max(0, offset - 8):offset], model, CHARACTER_PENALTY)
            weights = [w * f for w, f in zip(weights, factors)]
        if words:
            factors = words.multipliers(text[max(0, offset - HISTORY):offset], model, penalty)
            weights = [w * f for w, f in zip(weights, factors)]
        rows.append(weights)
    operations = Counter(base.operations)
    for part in (character, words):
        if part:
            operations.update(part.operations)
    return rows, operations


def run(root: Path, *, acknowledge_review: bool = False) -> dict:
    dataset = load_acts(root, acknowledge_review, include_final_test=False)
    train_all, held = dataset["training"], dataset["validation"]
    run_id = uuid4().hex
    discovery = {str(s): WordOnlies(train_all, min_support=s) for s in SUPPORTS}
    report = {
        "version": 1, "kind": "word-only-states", "run_id": run_id, "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_sha256": hashlib.sha256(train_all.encode()).hexdigest(),
        "held_out_sha256": hashlib.sha256(held.encode()).hexdigest(),
        "supports": SUPPORTS, "penalties": PENALTIES, "fractions": FRACTIONS,
        "character_only_penalty": CHARACTER_PENALTY,
        "discovery": {s: states.summary() for s, states in discovery.items()},
        "hold_rates_on_validation": {s: hold_rates(states, held) for s, states in discovery.items()},
        "approval": "User: continue the onlies strategy with the identified character onlies plus one-way (forward) and opposite (backward) onlies at word level: 'let's do it'.",
        "policy": {
            "words": "Words are runs of letters and apostrophes, case preserved; punctuation and spaces separate words. Word only states are earned in reading order; later different neighbours become variants.",
            "forward": "After a word with a forward only state, characters that keep the current partial word on the path (or a variant), or end it when complete, keep weight 1; others get the penalty.",
            "backward": "A character whose every reachable training word requires a different previous word gets the penalty. Unknown words are untouched.",
            "selection": "Development comparison on Acts 2 (reused monitoring data). Acts 3 never read; Acts 4-5 already spent. Any claim needs a frozen test on new text.",
            "honesty": "Guidance only, never zero probability. Lower loss is not understanding. Registry unchanged.",
        },
        "results": [],
    }
    started = time.perf_counter()
    for fraction in FRACTIONS:
        train = train_all[:round(len(train_all) * fraction)]
        model = NeuralModel(sorted(set(train)))
        validation = model.examples(held)
        kn = KneserNey(train)
        metrics = evaluate(model, validation, [kn.prior(held[max(0, i - 3):i], model) for i in range(len(held))], neural=False)
        report["results"].append({"arm": "counting_kn", "support": None, "penalty": None, "fraction": fraction,
                                  "validation_loss": metrics["loss"], "coverage": metrics["coverage"]})
        for name, (_, forward, backward) in ARMS.items():
            settings = [(s, p) for s in SUPPORTS for p in PENALTIES] if forward or backward else [(None, None)]
            for support, penalty in settings:
                try:
                    rows, operations = priors(name, train, held, model, support or 2, penalty or 1.0)
                except ValueError as error:
                    report["results"].append({"arm": name, "support": support, "penalty": penalty,
                                              "fraction": fraction, "validation_loss": None, "error": str(error)})
                    continue
                metrics = evaluate(model, validation, rows, neural=False)
                report["results"].append({
                    "arm": name, "support": support, "penalty": penalty, "fraction": fraction,
                    "validation_loss": metrics["loss"], "coverage": metrics["coverage"],
                    "blocked_targets": metrics["blocked_targets"], "operations": dict(operations),
                })
                print(f"{name} s={support} p={penalty} f={fraction}: {metrics['loss']}", file=sys.stderr, flush=True)
    report["wall_seconds"] = time.perf_counter() - started
    report["status"] = "completed"
    directory = root / ".data/neural-onlies" / run_id
    atomic_json(directory / "word-ledger.json", {s: states.ledger() for s, states in discovery.items()})
    atomic_json(directory / "report.json", report)
    atomic_json(root / ".data/neural-experiments/latest-word-onlies.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acknowledge-review", action="store_true")
    args = parser.parse_args()
    output = run(Path(__file__).resolve().parent, acknowledge_review=args.acknowledge_review)
    print(json.dumps({"run_id": output["run_id"], "discovery": output["discovery"],
                      "hold_rates": output["hold_rates_on_validation"]}, indent=1, ensure_ascii=False))
