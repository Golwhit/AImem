"""Isolated character MLP: standard-library forward pass, backpropagation and SGD."""

import argparse
import hashlib
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from original_generator import atomic_json


CONTEXT = 4
EMBEDDING = 8
HIDDEN = 16
RATE = 0.1
SYNTHETIC_TRAIN = "ab " * 80
SYNTHETIC_HELD = "ab " * 20


class NeuralModel:
    def __init__(self, characters: list[str], seed: int = 17):
        self.characters = characters
        self.tokens = {char: index + 2 for index, char in enumerate(characters)}
        self.size = len(characters) + 2  # Boundary and unknown are distinct tokens.
        rng = random.Random(seed)
        self.weights = {
            "embedding": [rng.uniform(-0.2, 0.2) for _ in range(self.size * EMBEDDING)],
            "hidden": [rng.uniform(-0.2, 0.2) for _ in range(CONTEXT * EMBEDDING * HIDDEN)],
            "hidden_bias": [0.0] * HIDDEN,
            "output": [rng.uniform(-0.2, 0.2) for _ in range(HIDDEN * self.size)],
            "output_bias": [0.0] * self.size,
        }

    def examples(self, text: str) -> list[tuple[list[int], int]]:
        tokens = [0] * CONTEXT + [self.tokens.get(char, 1) for char in text]
        return [(tokens[i - CONTEXT:i], tokens[i]) for i in range(CONTEXT, len(tokens))]

    def forward(self, context: list[int]) -> tuple[list[float], list[float], list[float]]:
        w = self.weights
        x = [w["embedding"][token * EMBEDDING + d] for token in context for d in range(EMBEDDING)]
        hidden = [
            math.tanh(w["hidden_bias"][j] + sum(value * w["hidden"][i * HIDDEN + j] for i, value in enumerate(x)))
            for j in range(HIDDEN)
        ]
        logits = [
            w["output_bias"][k] + sum(hidden[j] * w["output"][j * self.size + k] for j in range(HIDDEN))
            for k in range(self.size)
        ]
        maximum = max(logits)
        exponentials = [math.exp(value - maximum) for value in logits]
        total = sum(exponentials)
        return x, hidden, [value / total for value in exponentials]

    def loss_and_gradients(self, context: list[int], target: int) -> tuple[float, dict[str, list[float]]]:
        x, hidden, probabilities = self.forward(context)
        loss = -math.log(max(probabilities[target], 1e-300))
        delta = probabilities[:]
        delta[target] -= 1
        w = self.weights
        gradients = {name: [0.0] * len(values) for name, values in w.items()}
        gradients["output_bias"] = delta
        hidden_delta = [
            sum(w["output"][j * self.size + k] * delta[k] for k in range(self.size)) * (1 - hidden[j] ** 2)
            for j in range(HIDDEN)
        ]
        gradients["hidden_bias"] = hidden_delta
        for j in range(HIDDEN):
            for k in range(self.size):
                gradients["output"][j * self.size + k] = hidden[j] * delta[k]
        for i, value in enumerate(x):
            for j in range(HIDDEN):
                gradients["hidden"][i * HIDDEN + j] = value * hidden_delta[j]
            token = context[i // EMBEDDING]
            gradients["embedding"][token * EMBEDDING + i % EMBEDDING] += sum(
                w["hidden"][i * HIDDEN + j] * hidden_delta[j] for j in range(HIDDEN)
            )
        return loss, gradients

    def loss(self, examples: list[tuple[list[int], int]]) -> float:
        return sum(-math.log(max(self.forward(context)[2][target], 1e-300)) for context, target in examples) / len(examples)

    def sample(self, prompt: str, length: int = 120) -> str:
        rng = random.Random(23)
        context = ([0] * CONTEXT + [self.tokens.get(char, 1) for char in prompt])[-CONTEXT:]
        output = []
        for _ in range(length):
            probabilities = self.forward(context)[2]
            token = rng.choices(range(2, self.size), weights=probabilities[2:])[0]
            output.append(self.characters[token - 2])
            context = context[1:] + [token]
        return "".join(output)


def train_experiment(payload: dict, root: Path) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Neural training request must be a JSON object")
    synthetic = payload.get("synthetic") is True
    train = SYNTHETIC_TRAIN if synthetic else payload.get("training_text")
    held = SYNTHETIC_HELD if synthetic else payload.get("held_out_text")
    source = "Original synthetic repeating-pattern diagnostic" if synthetic else payload.get("source")
    if not synthetic and payload.get("acknowledge_review") is not True:
        raise ValueError("Explicit approval for this neural experiment is required")
    if not isinstance(source, str) or not source.strip() or len(source) > 500:
        raise ValueError("Provide a provenance/rights note of 1–500 characters")
    for label, text, limit in [("Training", train, 2000), ("Held-out", held, 1000)]:
        if not isinstance(text, str) or not text.strip() or not 8 <= len(text) <= limit:
            raise ValueError(f"{label} text must contain 8–{limit} characters")
    if not synthetic and (train in held or held in train):
        raise ValueError("Use separate held-out text, not an exact copy or substring of training text")
    if len(set(train)) > 128:
        raise ValueError("This tiny experiment supports at most 128 distinct training characters")
    steps = payload.get("steps", 300)
    if type(steps) is not int or not 1 <= steps <= 1000:
        raise ValueError("Training steps must be an integer from 1 to 1000")
    model = NeuralModel(sorted(set(train)))
    training = model.examples(train)
    validation = model.examples(held)
    rng = random.Random(19)
    history = []
    initial = {name: values[:] for name, values in model.weights.items()}
    initial_sample = model.sample(train[:CONTEXT])

    def measure(step: int) -> None:
        history.append({"step": step, "training_loss": model.loss(training), "held_out_loss": model.loss(validation)})

    measure(0)
    for step in range(1, steps + 1):
        context, target = rng.choice(training)
        _, gradients = model.loss_and_gradients(context, target)
        for name, values in model.weights.items():
            for i, gradient in enumerate(gradients[name]):
                values[i] -= RATE * gradient
        if step % 100 == 0 or step == steps:
            measure(step)
    if not all(math.isfinite(value) for values in model.weights.values() for value in values):
        raise ArithmeticError("Non-finite neural weights; checkpoint not saved")
    if not all(math.isfinite(row[key]) for row in history for key in ("training_loss", "held_out_loss")):
        raise ArithmeticError("Non-finite neural loss; checkpoint not saved")
    report = {
        "version": 1, "run_id": uuid4().hex, "model": "original-character-mlp-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "synthetic": synthetic, "source": source, "legal_clearance": False,
        "parameter_count": sum(map(len, model.weights.values())),
        "trainable_parameters": sum(map(len, model.weights.values())),
        "changed_parameters": sum(a != b for name in initial for a, b in zip(initial[name], model.weights[name])),
        "context": CONTEXT, "embedding": EMBEDDING, "hidden": HIDDEN,
        "vocabulary": model.size, "steps": steps, "processed_target_tokens": steps,
        "training_characters": len(train), "held_out_characters": len(held),
        "held_out_unknown_characters": sum(char not in model.tokens for char in held),
        "training_sha256": hashlib.sha256(train.encode()).hexdigest(),
        "held_out_sha256": hashlib.sha256(held.encode()).hexdigest(),
        "history": history, "sample_prompt": train[:CONTEXT],
        "initial_sample": initial_sample, "sample": model.sample(train[:CONTEXT]),
        "learning_rate": RATE, "initialization_seed": 17, "training_seed": 19,
        "layer_policy": "Unconstrained neural baseline; Layer 0/1 rules are unchanged and not applied here.",
        "evaluation_note": "Synthetic repetition checks mechanics only, not generalization." if synthetic else
        "Held-out text receives no gradient updates. Author must ensure meaningful separation; substring checks do not prove independence.",
    }
    checkpoint = {
        "report": report, "characters": model.characters, "weights": model.weights,
        "training_text": train, "held_out_text": held,
        "approval": "synthetic diagnostic" if synthetic else "explicit user declaration for this neural experiment",
    }
    directory = root / ".data" / "neural-experiments"
    atomic_json(directory / f"{report['run_id']}.json", checkpoint)
    atomic_json(directory / "latest.json", report)
    return report


def status(root: Path) -> dict:
    path = root / ".data" / "neural-experiments" / "latest.json"
    return {"ready": path.exists(), "report": json.loads(path.read_text()) if path.exists() else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["status", "train"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    try:
        result = status(root) if args.command == "status" else train_experiment(json.load(sys.stdin), root)
    except (ValueError, ArithmeticError, OSError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
