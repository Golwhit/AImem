"""Small offline comparison of single contexts and a learned two-context mixture."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
from uuid import uuid4

from neural_experiment import NeuralModel
from neural_rounds import load_acts
from original_generator import atomic_json


class ContextMix:
    """Two independent base MLPs with a learned global probability-mixture weight."""

    def __init__(self, characters: list[str], seed: int, *, contexts: tuple[int, int] = (4, 16)):
        if (len(contexts) != 2 or any(type(c) is not int for c in contexts)
                or not 1 <= contexts[0] < contexts[1] <= 128):
            raise ValueError("Mixture contexts must be two increasing integers from 1 to 128")
        self.short = NeuralModel(characters, seed=seed, context=contexts[0])
        self.long = NeuralModel(characters, seed=seed + 1, context=contexts[1])
        self.gate = 0.0

    def weight(self) -> float:
        return 1 / (1 + math.exp(-self.gate))

    def probabilities(self, context: list[int]) -> list[float]:
        short = self.short.forward(context[-self.short.context:])[2]
        long = self.long.forward(context[-self.long.context:])[2]
        weight = self.weight()
        return [(1 - weight) * a + weight * b for a, b in zip(short, long)]

    def update(self, context: list[int], target: int, rate: float) -> None:
        short = self.short.forward(context[-self.short.context:])[2][target]
        long = self.long.forward(context[-self.long.context:])[2][target]
        weight = self.weight()
        probability = (1 - weight) * short + weight * long
        if not math.isfinite(probability) or probability <= 0:
            raise ArithmeticError("Mixture target probability must be finite and positive")
        # Chain-rule responsibilities give the exact mixture-loss gradient in each branch.
        short_share = (1 - weight) * short / probability
        long_share = weight * long / probability
        gate_gradient = weight * (1 - weight) * (short - long) / probability
        self.short.update(context[-self.short.context:], target, learning_rate=rate * short_share)
        self.long.update(context[-self.long.context:], target, learning_rate=rate * long_share)
        self.gate -= rate * gate_gradient

    def snapshot(self) -> dict:
        return {"short": self.short.weights, "long": self.long.weights, "gate": self.gate}


class ContextTriple(ContextMix):
    """Three independent branches with learned global softmax mixture weights."""

    def __init__(self, characters: list[str], seed: int, *, contexts: tuple[int, ...]):
        if (len(contexts) != 3 or any(type(c) is not int or not 1 <= c <= 128 for c in contexts)
                or list(contexts) != sorted(set(contexts))):
            raise ValueError("Triple contexts must be three increasing integers from 1 to 128")
        self.branches = [NeuralModel(characters, seed=seed + i, context=context)
                         for i, context in enumerate(contexts)]
        self.gates = [0.0] * 3

    def branch_weights(self) -> list[float]:
        values = [math.exp(value - max(self.gates)) for value in self.gates]
        return [value / sum(values) for value in values]

    def probabilities(self, context: list[int]) -> list[float]:
        rows = [branch.forward(context[-branch.context:])[2] for branch in self.branches]
        weights = self.branch_weights()
        return [sum(weight * row[k] for weight, row in zip(weights, rows))
                for k in range(self.branches[0].size)]

    def update(self, context: list[int], target: int, rate: float) -> None:
        targets = [branch.forward(context[-branch.context:])[2][target] for branch in self.branches]
        weights = self.branch_weights()
        probability = sum(weight * target_probability for weight, target_probability in zip(weights, targets))
        if not math.isfinite(probability) or probability <= 0:
            raise ArithmeticError("Mixture target probability must be finite and positive")
        responsibilities = [weight * target_probability / probability
                            for weight, target_probability in zip(weights, targets)]
        for branch, share in zip(self.branches, responsibilities):
            branch.update(context[-branch.context:], target, learning_rate=rate * share)
        self.gates = [gate - rate * (weight - share)
                      for gate, weight, share in zip(self.gates, weights, responsibilities)]

    def snapshot(self) -> dict:
        return {"branches": [branch.weights for branch in self.branches], "gates": self.gates}


def metrics(model: NeuralModel | ContextMix, examples: list) -> dict:
    losses, correct = [], 0
    for context, target in examples:
        probabilities = (model.probabilities(context) if isinstance(model, ContextMix)
                         else model.forward(context[-model.context:])[2])
        losses.append(-math.log(max(probabilities[target], 1e-300)))
        correct += max(range(len(probabilities)), key=probabilities.__getitem__) == target
    return {"loss": statistics.fmean(losses), "character_accuracy": correct / len(examples),
            "targets": len(examples)}


def compare(root: Path, *, acknowledge_review: bool, steps: int = 1500,
            contexts: tuple[int, ...] = (4, 16), pair_only: bool = False) -> dict:
    if acknowledge_review is not True:
        raise ValueError("Explicit approval is required")
    if type(steps) is not int or not 1 <= steps <= 5000:
        raise ValueError("Steps must be an integer from 1 to 5000")
    triple = len(contexts) == 3
    if triple and not pair_only:
        raise ValueError("Triple comparison requires --pair-only (mixture versus baseline)")
    if triple:
        ContextTriple(["a"], 17, contexts=contexts)
    else:
        ContextMix(["a"], 17, contexts=contexts)
    dataset = load_acts(root, True, include_final_test=False)
    train, held = dataset["training"], dataset["validation"]
    characters = sorted(set(train))
    size = len(characters) + 2
    # One single MLP's hidden input matrix grows by 128 parameters per context character.
    mixed_parameters = len(contexts) * (25 * size + 16) + 128 * sum(contexts) + (3 if triple else 1)
    matched_context = max(1, min(128, round((mixed_parameters - (25 * size + 16)) / 128)))
    mix_name = "mixed" + "_".join(map(str, contexts))
    arms = ({"context4": 4, mix_name: None} if pair_only else
            {f"context{contexts[0]}": contexts[0], f"context{contexts[1]}": contexts[1],
             mix_name: None, "single_capacity_control": matched_context})
    context = max(4, max(contexts), matched_context if not pair_only else 4)
    source_model = NeuralModel(characters, context=context)
    training = source_model.examples(train)
    validation = source_model.examples(held)
    # Fixed evenly spaced checks; training draws are independent of these selectors.
    select = lambda values: [values[i * len(values) // min(512, len(values))]
                             for i in range(min(512, len(values)))]
    train_checks, checks = select(training), select(validation)
    run_id = uuid4().hex
    directory = root / ".data/neural-context-mix" / run_id
    report = {
        "run_id": run_id, "status": "running", "steps_per_arm": steps,
        "seeds": [17, 117, 217], "learning_rate": 0.03,
        "mixture_contexts": list(contexts), "pair_only": pair_only,
        "training_sha256": hashlib.sha256(train.encode()).hexdigest(),
        "validation_sha256": hashlib.sha256(held.encode()).hexdigest(),
        "manifest_sha256": dataset["manifest_sha256"],
        "training_characters": len(train), "validation_characters": len(held),
        "unknown_validation_characters": sum(c not in source_model.tokens for c in held),
        "source": "Existing authorized WEB Acts 1 training / Acts 2 reused validation. Acts 3 not loaded.",
        "policy": "Fresh base models, no layers or pretrained/chat checkpoints. "
                  "Same sampled target positions per seed/arm. Mixture uses independent MLP branches and "
                  "learned GLOBAL blend weights (sigmoid for two; three softmax logits for three, "
                  "with one redundant degree of freedom), not attention or an input-dependent selector. "
                  "Branches are jointly trained on mixture cross-entropy using exact responsibility "
                  "gradients. Equal target exposures, NOT equal operations: mixture updates multiple branches "
                  "and performs extra forward calls. When included, single-window control approximately matches parameter "
                  "count, not compute, connectivity or initialization. 512 fixed character checks per split. "
                  "Three seeds and short runs are exploratory, not proof of generalization or convergence. "
                  "Acts 2 is reused development text and was trained on by other historical models. "
                  "No activation, no latest experiment pointers changed.",
        "code_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                        for name in ("neural_context_mix.py", "neural_experiment.py")},
        "runs": [],
    }
    atomic_json(directory / "protocol.json", report)
    atomic_json(directory / "inputs.json", {"training": train, "validation": held,
                                          "training_checks": train_checks, "validation_checks": checks})
    for position, seed in enumerate(report["seeds"]):
        rng = random.Random(seed + 2)
        draws = [rng.randrange(len(training)) for _ in range(steps)]
        names = list(arms)
        names = names[position:] + names[:position]
        for arm in names:
            if arm == mix_name:
                model = (ContextTriple(characters, seed, contexts=contexts) if triple
                         else ContextMix(characters, seed, contexts=contexts))
            else:
                model = NeuralModel(characters, seed=seed, context=arms[arm])
            parameters = (mixed_parameters if isinstance(model, ContextMix)
                          else sum(map(len, model.weights.values())))
            entry = {"seed": seed, "arm": arm, "context": arms[arm], "parameters": parameters,
                     "draws_sha256": hashlib.sha256(json.dumps(draws).encode()).hexdigest(),
                     "before": metrics(model, checks), "history": []}
            milestones = {max(1, steps // 2), steps}
            cpu = 0.0
            for step, draw in enumerate(draws, 1):
                selected_context, target = training[draw]
                start = time.process_time()
                if isinstance(model, ContextMix):
                    model.update(selected_context, target, 0.03)
                else:
                    model.update(selected_context[-model.context:], target, learning_rate=0.03)
                cpu += time.process_time() - start
                if step in milestones:
                    entry["history"].append({"step": step, "training": metrics(model, train_checks),
                                             "validation": metrics(model, checks), "training_cpu_seconds": cpu})
            if isinstance(model, ContextTriple):
                entry["branch_weights"] = model.branch_weights()
            elif isinstance(model, ContextMix):
                entry["long_context_weight"] = model.weight()
            atomic_json(directory / f"{seed}-{arm}.json", {
                "characters": characters, "state": model.snapshot() if isinstance(model, ContextMix) else model.weights,
                "entry": entry})
            report["runs"].append(entry)
            atomic_json(directory / "report.json", report)
            final = entry["history"][-1]
            print(f"{seed} {arm}: parameters={parameters}; loss={final['validation']['loss']:.5f}; "
                  f"accuracy={final['validation']['character_accuracy']:.4f}; CPU={cpu:.2f}s", flush=True)
    report["summary"] = {
        arm: {key: statistics.fmean(
            row["history"][-1]["validation"][key] for row in report["runs"] if row["arm"] == arm)
              for key in ("loss", "character_accuracy")}
        for arm in arms}
    mixed = {row["seed"]: row["history"][-1]["validation"]["loss"]
             for row in report["runs"] if row["arm"] == mix_name}
    report["paired_loss_differences_mix_minus_control"] = {
        arm: [mixed[seed] - next(row for row in report["runs"] if row["arm"] == arm and row["seed"] == seed)
              ["history"][-1]["validation"]["loss"] for seed in report["seeds"]]
        for arm in arms if arm != mix_name}
    report["status"] = "completed"
    atomic_json(directory / "report.json", report)
    print(json.dumps({"run_id": run_id, "summary": report["summary"],
                      "paired": report["paired_loss_differences_mix_minus_control"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acknowledge-review", action="store_true")
    parser.add_argument("--contexts", nargs="+", type=int, default=(4, 16), metavar="CONTEXT")
    parser.add_argument("--pair-only", action="store_true",
                        help="Compare the requested mixture against context 4 only")
    args = parser.parse_args()
    compare(Path(__file__).resolve().parent, acknowledge_review=args.acknowledge_review,
            contexts=tuple(args.contexts), pair_only=args.pair_only)
