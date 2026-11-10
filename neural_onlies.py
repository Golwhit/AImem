"""Layer 0 "only states": configurations that, so far, continue or are entered in exactly one way.

Discovery reads approved training text in order. A context of 1-8 characters becomes an
only state once it has been seen MIN_SUPPORT times with a single outcome. If later text shows
another outcome, the original path is kept and the new outcome is recorded as a compatible
variant (a mutation), never deleting the original. Only states are unconfirmed candidates:
they guide likelihood but never make an unseen character impossible.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

from neural_counting import KneserNey
from neural_experiment import NeuralModel
from neural_layers import DevelopmentalPrior, evaluate
from neural_rounds import load_acts
from original_generator import atomic_json


MAX_CONTEXT = 8
MIN_SUPPORT = 3
PENALTIES = (0.05, 0.2, 0.5)
FRACTIONS = (0.1, 0.25, 0.5, 1.0)


class OnlyStates:
    def __init__(self, source: str, *, max_context: int = MAX_CONTEXT, min_support: int = MIN_SUPPORT,
                 backward: bool = True):
        if not source or type(max_context) is not int or not 1 <= max_context <= 16 or min_support < 1:
            raise ValueError("Only-state discovery needs text, a 1-16 context and positive support")
        self.max_context = max_context
        self.min_support = min_support
        self.backward = backward
        self.forward_states = self._discover(source, forward=True)
        self.backward_states = self._discover(source, forward=False) if backward else {}
        self.operations = Counter(build_observations=2 * len(source) * max_context if backward else len(source) * max_context)

    def _discover(self, source: str, *, forward: bool) -> dict[str, dict]:
        """forward: block -> next character. backward: block -> character before it."""
        seen: dict[str, Counter] = {}
        states: dict[str, dict] = {}
        for size in range(1, self.max_context + 1):
            for offset in range(len(source) - size):
                if forward:
                    block, outcome, at = source[offset:offset + size], source[offset + size], offset + size
                else:
                    block, outcome, at = source[offset + 1:offset + size + 1], source[offset], offset
                state = states.get(block)
                if state is not None:
                    state["support"] += 1
                    if outcome != state["path"] and outcome not in state["variants"]:
                        state["variants"][outcome] = at
                    continue
                counts = seen.setdefault(block, Counter())
                counts[outcome] += 1
                if len(counts) == 1 and counts[outcome] >= self.min_support:
                    states[block] = {"path": outcome, "variants": {}, "support": counts[outcome], "earned_at": at}
                    del seen[block]
        return states

    def allowed(self, state: dict) -> set[str]:
        return {state["path"], *state["variants"]}

    def multipliers(self, prefix: str, model: NeuralModel, penalty: float) -> list[float]:
        """Off-path characters keep `penalty` weight; on-path and variant characters keep 1."""
        weights = [1.0] * model.size
        for size in range(min(self.max_context, len(prefix)), 0, -1):
            self.operations["predict_forward_lookups"] += 1
            state = self.forward_states.get(prefix[-size:])
            if state:
                allowed = self.allowed(state)
                weights = [penalty] * model.size
                for char in allowed:
                    if char in model.tokens:
                        weights[model.tokens[char]] = 1.0
                break
        if self.backward:
            for char, token in model.tokens.items():
                for size in range(1, min(self.max_context, len(prefix)) + 1):
                    self.operations["predict_backward_lookups"] += 1
                    block = prefix[len(prefix) - size + 1:] + char if size > 1 else char
                    state = self.backward_states.get(block)
                    if state and prefix[-size] not in self.allowed(state):
                        weights[token] *= penalty
                        break
        return weights

    def summary(self) -> dict:
        def describe(states: dict[str, dict]) -> dict:
            lengths = Counter(len(block) + 1 for block in states)
            mutated = [block for block, state in states.items() if state["variants"]]
            return {"count": len(states), "pure": len(states) - len(mutated), "mutated": len(mutated),
                    "by_configuration_length": dict(sorted(lengths.items()))}
        return {"forward": describe(self.forward_states), "backward": describe(self.backward_states),
                "max_context": self.max_context, "min_support": self.min_support}

    def ledger(self) -> dict:
        return {
            "forward": {block: state for block, state in self.forward_states.items()},
            "backward": {block: state for block, state in self.backward_states.items()},
            "state": "unconfirmed candidates; never confirmed forms or permanent admissions",
        }


def hold_rates(states: OnlyStates, text: str) -> dict:
    """How often earned forward/backward only states took their path or a known variant in new text."""
    rows = {}
    for name, table, forward in (("forward", states.forward_states, True), ("backward", states.backward_states, False)):
        by_length: dict[int, Counter] = {}
        for size in range(1, states.max_context + 1):
            for offset in range(len(text) - size):
                block = text[offset:offset + size] if forward else text[offset + 1:offset + size + 1]
                outcome = text[offset + size] if forward else text[offset]
                state = table.get(block)
                if state:
                    by_length.setdefault(size + 1, Counter())[
                        "path" if outcome == state["path"] else "variant" if outcome in state["variants"] else "new"
                    ] += 1
        rows[name] = {
            str(length): {**counts, "held_rate": (counts["path"] + counts["variant"]) / sum(counts.values())}
            for length, counts in sorted(by_length.items())
        }
    return rows


def arm_priors(name: str, train: str, text: str, model: NeuralModel, penalty: float | None):
    if name == "counting_kn":
        prior = KneserNey(train)
        return [prior.prior(text[max(0, i - prior.order):i], model) for i in range(len(text))], prior.operations
    zero_v2 = name.endswith("_v2")
    base = DevelopmentalPrior(train, layer_zero=zero_v2, layer_one=True, layer_two=True, layer_zero_mode="v2")
    onlies = None
    if name.startswith("onlies"):
        onlies = OnlyStates(train, backward="forward" not in name)
    priors = []
    for offset in range(len(text)):
        weights = base.prior(text[max(0, offset - 4):offset], model)
        if onlies:
            factors = onlies.multipliers(text[max(0, offset - MAX_CONTEXT):offset], model, penalty)
            weights = [weight * factor for weight, factor in zip(weights, factors)]
        priors.append(weights)
    operations = Counter(base.operations)
    if onlies:
        operations.update(onlies.operations)
    return priors, operations


ARMS = ("counting_kn", "layers_12", "layers_012_v2", "onlies_forward_12", "onlies_12", "onlies_012_v2")


def run_onlies(root: Path, *, acknowledge_review: bool = False) -> dict:
    dataset = load_acts(root, acknowledge_review, include_final_test=False)
    train_all, held = dataset["training"], dataset["validation"]
    run_id = uuid4().hex
    directory = root / ".data/neural-onlies" / run_id
    discovered = OnlyStates(train_all)
    report = {
        "version": 1, "kind": "layer-zero-only-states", "run_id": run_id, "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_sha256": hashlib.sha256(train_all.encode()).hexdigest(),
        "held_out_sha256": hashlib.sha256(held.encode()).hexdigest(),
        "max_context": MAX_CONTEXT, "min_support": MIN_SUPPORT, "penalties": PENALTIES, "fractions": FRACTIONS,
        "discovery": discovered.summary(), "hold_rates_on_validation": hold_rates(discovered, held),
        "results": [],
        "approval": "User approved 'go' on only states in Layer 0 (with mutation variants) and rerunning the comparison: 'both'.",
        "policy": {
            "only_states": "Earned in reading order from approved training text only. A context seen 3 times with one outcome becomes an only state; later different outcomes are kept as compatible variants (mutations), never deleting the original path.",
            "prediction": "Longest matching forward only state: path and variants keep weight 1, other characters keep the penalty weight. Backward only state: a character whose required predecessor is absent is multiplied by the penalty. Never zero; Layer 1 exclusions remain the only zeros.",
            "selection": "Development only. Penalties are compared on Acts 2, which is reused monitoring data. Acts 3 is never read; Acts 4-5 were spent by the pre-registered proof test and are not reused. A frozen test on new untouched text is still required before any claim.",
            "honesty": "Lower next-character loss is not understanding. Only states are unconfirmed candidates; the permanent admission registry is unchanged.",
        },
    }
    started = time.perf_counter()
    for fraction in FRACTIONS:
        train = train_all[:round(len(train_all) * fraction)]
        model = NeuralModel(sorted(set(train)))
        validation = model.examples(held)
        for name in ARMS:
            for penalty in (PENALTIES if name.startswith("onlies") else (None,)):
                cpu = time.process_time()
                priors, operations = arm_priors(name, train, held, model, penalty)
                metrics = evaluate(model, validation, priors, neural=False)
                report["results"].append({
                    "arm": name, "penalty": penalty, "fraction": fraction, "training_characters": len(train),
                    "validation_loss": metrics["loss"], "coverage": metrics["coverage"],
                    "blocked_targets": metrics["blocked_targets"], "operations": dict(operations),
                    "cpu_seconds_including_validation": time.process_time() - cpu,
                })
                print(f"{name} penalty={penalty} f={fraction}: {metrics['loss']}", file=sys.stderr, flush=True)
    report["wall_seconds"] = time.perf_counter() - started
    report["status"] = "completed"
    atomic_json(directory / "ledger.json", {"run_id": run_id, "training_sha256": report["training_sha256"],
                                            "only_states": discovered.ledger()})
    atomic_json(directory / "report.json", report)
    atomic_json(root / ".data/neural-experiments/latest-onlies.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acknowledge-review", action="store_true")
    args = parser.parse_args()
    output = run_onlies(Path(__file__).resolve().parent, acknowledge_review=args.acknowledge_review)
    print(json.dumps({key: output[key] for key in ("run_id", "discovery", "status")}, indent=1))
