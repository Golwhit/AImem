"""Isolated developmental priors for neural training, not permanent admissions."""

from collections import Counter
from functools import lru_cache
import math
import random

from layer_two import Attraction
from neural_experiment import NeuralModel
from seed_life import ALIVE_SEEDS, DOMAIN, LAYER_ZERO_SEEDS, TERMINUS, boundary_text, exclusion_reason


EXPLORATION = 0.05
LAYER_ZERO_MODES = ("v1", "v2")


def attraction_operations(attraction: Attraction, prefix: str) -> int:
    """Table entries touched by Attraction.distribution, mirroring its interpolation loop."""
    keys = set(attraction.unigrams)
    operations = len(keys)
    for size in (1, 2, 3):
        counts = attraction.contexts.get(prefix[-size:]) if len(prefix) >= size else None
        if counts:
            keys |= counts.keys()
            operations += len(keys)
    return operations


def local_exclusion(prefix: str, character: str) -> str | None:
    text = prefix + character if len(prefix) < 2 else prefix[-2:] + character
    return exclusion_reason(boundary_text(text, len(prefix) < 2))


class DevelopmentalPrior:
    def __init__(
        self, source: str, *, layer_zero: bool = True,
        layer_one: bool = True, layer_two: bool = True, layer_zero_mode: str = "v1",
    ):
        if layer_zero_mode not in LAYER_ZERO_MODES:
            raise ValueError("Layer 0 mode must be v1 or v2")
        self.layer_zero_mode = layer_zero_mode
        self.operations: Counter[str] = Counter()
        self.layer_zero = layer_zero
        self.layer_one = layer_one
        self.layer_two = layer_two
        self.candidates: dict[tuple[bool, str], dict] = {}
        self.observations = 0
        self.conflicts = [
            {"offset": offset, "text": source[max(0, offset - 2):offset + 1], "reason": reason}
            for offset, char in enumerate(source)
            if (reason := local_exclusion(source[max(0, offset - 3):offset], char))
        ] if layer_one else []
        if layer_zero:
            for size in range(1, 5):
                for offset in range(len(source) - size + 1):
                    self.observations += 1
                    text = source[offset:offset + size]
                    at_start = offset == 0
                    if layer_one and exclusion_reason(boundary_text(text, at_start)):
                        continue
                    if size > 1 and not (
                        self.available(text[:-1], at_start) and self.available(text[1:], False)
                    ):
                        continue
                    if boundary_text(text, at_start) in LAYER_ZERO_SEEDS:
                        continue
                    key = (at_start, text)
                    if key not in self.candidates:
                        self.candidates[key] = {
                            "text": text, "at_source_start": at_start,
                            "state": "candidate", "confirmed": False, "first_offset": offset,
                            "support": 0, "components": [text[:-1], text[1:]] if size > 1 else [],
                        }
                    self.candidates[key]["support"] += 1
        self.attraction = Attraction(source, max_characters=50_000) if layer_two else None
        self.possible = lru_cache(maxsize=None)(self._possible)
        self.operations["build_layer_zero_window_checks"] = self.observations
        self.operations["build_layer_one_source_checks"] = len(source) if layer_one else 0
        self.operations["build_layer_two_table_updates"] = (
            len(source) + sum(max(0, len(source) - size) for size in (1, 2, 3))
            + sum(max(0, len(source) - size + 1) for size in (2, 3, 4))
        ) if layer_two else 0

    @staticmethod
    def confirmed(text: str, at_start: bool) -> bool:
        return boundary_text(text, at_start) in ALIVE_SEEDS

    def available(self, text: str, at_start: bool) -> bool:
        return boundary_text(text, at_start) in LAYER_ZERO_SEEDS or (at_start, text) in self.candidates

    def _possible(self, text: str, at_start: bool) -> bool:
        """Layer 0 v2: a block is possible when known, or composed from overlapping possible pieces.

        Single characters and pairs must be observed foundation/candidate blocks; three- and
        four-character blocks may be new combinations of possible overlapping pieces.
        """
        if self.available(text, at_start):
            return True
        return len(text) > 2 and self.possible(text[:-1], at_start) and self.possible(text[1:], False)

    def prior(self, prefix: str, model: NeuralModel) -> list[float]:
        weights = [1.0] * model.size
        if self.attraction is not None:
            self.operations["predict_layer_two_table_entries"] += attraction_operations(self.attraction, prefix) + 2 * model.size
            distribution = self.attraction.distribution(prefix)
            mass = sum(distribution.get(char, 0) for char in model.characters)
            if mass <= 0:
                raise ArithmeticError("Training attraction has no character probability mass")
            weights = [EXPLORATION / model.size] * model.size
            for char, token in model.tokens.items():
                weights[token] += (1 - EXPLORATION) * distribution.get(char, 0) / mass
        if self.layer_zero:
            weights[0] *= EXPLORATION
            weights[1] *= EXPLORATION
            check = self.available if self.layer_zero_mode == "v1" else self.possible
            self.operations["predict_layer_zero_checks"] += len(model.tokens)
            for char, token in model.tokens.items():
                text = (prefix + char)[-4:]
                if not check(text, len(prefix) < 4):
                    weights[token] *= EXPLORATION
        if self.layer_one:
            self.operations["predict_layer_one_checks"] += len(model.tokens)
            for char, token in model.tokens.items():
                if local_exclusion(prefix, char):
                    weights[token] = 0.0
        return weights

    def evidence(self) -> dict:
        return {
            "domain": DOMAIN, "layer_zero": self.layer_zero, "layer_zero_mode": self.layer_zero_mode,
            "layer_one": self.layer_one,
            "layer_two": self.layer_two, "foundation": list(LAYER_ZERO_SEEDS),
            "confirmed_forms": sorted(ALIVE_SEEDS),
            "candidates": list(self.candidates.values()),
            "candidate_observations": self.observations,
            "source_rule_conflicts": self.conflicts,
            "attraction_evidence": self.attraction.report if self.attraction else None,
            "exploration": EXPLORATION, "permanent_admissions_changed": False,
        }


def probabilities(
    model: NeuralModel, context: list[int], prior: list[float] | None, *, neural: bool,
) -> list[float]:
    if neural:
        return model.forward(context, output_prior=prior)[2]
    if prior is None or not any(prior):
        raise ValueError("Layers-only prediction requires a nonempty prior")
    total = sum(prior)
    return [weight / total for weight in prior]


def evaluate(
    model: NeuralModel, examples: list[tuple[list[int], int]],
    priors: list[list[float]] | None, *, neural: bool,
) -> dict:
    if priors is not None and len(priors) != len(examples):
        raise ValueError("Evaluation priors must align with every original target")
    loss = 0.0
    blocked = []
    for offset, (context, target) in enumerate(examples):
        prior = priors[offset] if priors is not None else None
        if prior is not None and prior[target] == 0:
            blocked.append(offset)
            continue
        probability = probabilities(model, context, prior, neural=neural)[target]
        if probability <= 0 or not math.isfinite(probability):
            raise ArithmeticError("Nonpositive or nonfinite probability for an unblocked target")
        loss -= math.log(probability)
    return {
        "loss": None if blocked else loss / len(examples),
        "blocked_targets": len(blocked), "blocked_offsets": blocked,
        "scored_targets": len(examples),
        "coverage": (len(examples) - len(blocked)) / len(examples),
        "loss_note": "Any blocked original target makes full-text loss infinite (JSON null), never a filtered success.",
    }


def sample(model: NeuralModel, prior: DevelopmentalPrior | None, prompt: str, *, neural: bool) -> str:
    rng = random.Random(23)
    text = prompt
    for _ in range(120):
        context = ([0] * model.context + [model.tokens.get(char, 1) for char in text])[-model.context:]
        weights = probabilities(
            model, context, prior.prior(text[-4:], model) if prior is not None else None, neural=neural,
        )
        token = rng.choices(range(2, model.size), weights=weights[2:])[0]
        text += model.characters[token - 2]
    return text[len(prompt):]
