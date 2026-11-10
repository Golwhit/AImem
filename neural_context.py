"""Offline Acts context or learning-rate comparison; never reads final-test text."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from datetime import datetime, timezone
from uuid import uuid4

from neural_experiment import EMBEDDING, HIDDEN, RATE, NeuralModel
from neural_rounds import load_acts
from original_generator import atomic_json


CONTEXTS = (4, 16, 32)
LEARNING_RATES = (0.1, 0.03, 0.01)
STEPS = 5000
MEASURE_EVERY = 500


def compare_contexts(
    root: Path, *, acknowledge_review: bool = False, learning_rate: float = RATE,
) -> dict:
    if type(learning_rate) not in (int, float) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be a finite positive number")
    return _compare(
        root, acknowledge_review=acknowledge_review,
        configurations=[(context, learning_rate) for context in CONTEXTS], vary_rate=False,
    )


def compare_learning_rates(root: Path, *, acknowledge_review: bool = False) -> dict:
    return _compare(
        root, acknowledge_review=acknowledge_review,
        configurations=[(4, rate) for rate in LEARNING_RATES], vary_rate=True,
    )


def _compare(
    root: Path, *, acknowledge_review: bool,
    configurations: list[tuple[int, float]], vary_rate: bool,
) -> dict:
    dataset = load_acts(root, acknowledge_review, include_final_test=False)
    train = dataset["training"]
    held = dataset["validation"]
    rng = random.Random(19)
    draws = [rng.randrange(len(train)) for _ in range(STEPS)]
    prompts = ("The ", "They", train[:64])
    run_id = uuid4().hex
    directory = root / ".data" / "neural-experiments"
    run_path = directory / run_id
    axis = "rate" if vary_rate else "context"
    report = {
        "version": 2, "kind": f"{axis}-comparison", "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "running",
        "contexts": sorted({context for context, _ in configurations}), "steps_per_model": STEPS,
        "measurement_interval": MEASURE_EVERY, "batch_size": 1,
        "learning_rates": [rate for _, rate in configurations],
        "learning_rate": None if vary_rate else configurations[0][1],
        "embedding": EMBEDDING, "hidden": HIDDEN, "tokenization": "character",
        "initialization_seed": 17, "training_seed": 19, "sampling_seed": 23,
        "sample_length": 120, "sampling_temperature": 1,
        "training_characters": len(train), "held_out_characters": len(held),
        "training_sha256": hashlib.sha256(train.encode()).hexdigest(),
        "held_out_sha256": hashlib.sha256(held.encode()).hexdigest(),
        "training_draws_sha256": hashlib.sha256(json.dumps(draws).encode()).hexdigest(),
        "data_provenance": {
            "manifest": dataset["manifest"], "manifest_sha256": dataset["manifest_sha256"],
            "approval": (
                "User approved the proposed controlled learning-rate comparison on staged Acts 1/2 with 'go for it'."
                if vary_rate else
                "Explicitly approved controlled context comparison on staged Acts 1/2 "
                f"at learning rate {configurations[0][1]:g}; command requires --acknowledge-review."
            ),
        },
        "legal_clearance": False, "final_test_evaluations": 0, "models": [],
        "evaluation_note": (
            (
                "Fresh four-character models differ only in learning rate (0.1/0.03/0.01); "
                "initial weights, parameter count and target draw order are identical. "
                "One seed and a fixed 5,000-example budget do not establish which rate is best "
                "after longer training. "
                if vary_rate else
                "Fresh 4/16/32-character models see identical target positions in identical order. "
                "Longer contexts also add parameters and change initialization of later weight groups; "
                "this is a context-plus-capacity comparison, not proof of a context-only effect. "
            ) +
            "Acts 1 training only determines vocabulary and gradients; Acts 2 is reused validation, "
            "not an untouched test. Acts 3 text is neither read nor evaluated. "
            "This is not evidence of understanding. No checkpoint selection or automatic model replacement. "
            "Generated samples are model output, not scripture or the World English Bible. "
            "Layer 0/1/2 and developmental admissions do not participate in this neural baseline."
        ),
    }

    def save_report() -> None:
        atomic_json(run_path / "report.json", report)
        atomic_json(directory / f"latest-{axis}.json", report)

    save_report()
    try:
        for context_size, learning_rate in configurations:
            model = NeuralModel(sorted(set(train)), context=context_size)
            training = model.examples(train)
            validation = model.examples(held)
            result = {
                "context": context_size, "learning_rate": learning_rate, "status": "running", "steps": 0,
                "parameter_count": sum(map(len, model.weights.values())),
                "vocabulary": model.size,
                "held_out_unknown_characters": sum(char not in model.tokens for char in held),
                "history": [], "initial_samples": [
                    {"prompt": prompt, "text": model.sample(prompt)} for prompt in prompts
                ], "samples": [],
            }
            report["models"].append(result)

            def measure(step: int) -> None:
                row = {
                    "step": step, "training_loss": model.loss(training),
                    "held_out_loss": model.loss(validation),
                }
                if not all(math.isfinite(row[key]) for key in ("training_loss", "held_out_loss")):
                    raise ArithmeticError("Non-finite comparison loss")
                if not all(math.isfinite(v) for values in model.weights.values() for v in values):
                    raise ArithmeticError("Non-finite comparison weights")
                result["history"].append(row)
                result["steps"] = step
                result["samples"] = [{"prompt": prompt, "text": model.sample(prompt)} for prompt in prompts]
                model_id = f"rate-{learning_rate:g}" if vary_rate else f"context-{context_size}"
                atomic_json(run_path / model_id / f"step-{step:05}.json", {
                    "model": "original-character-mlp-v1", "context": model.context,
                    "learning_rate": learning_rate,
                    "characters": model.characters, "weights": model.weights,
                    "training_text": train, "held_out_text": held,
                    "result": result, "comparison_run_id": run_id,
                    "data_provenance": report["data_provenance"],
                })
                save_report()
                print(
                    f"Context {context_size}, rate {learning_rate:g}, step {step}/{STEPS}: "
                    f"train={row['training_loss']:.4f} validation={row['held_out_loss']:.4f}",
                    file=sys.stderr, flush=True,
                )

            measure(0)
            for step, index in enumerate(draws, 1):
                model.update(*training[index], learning_rate=learning_rate)
                if step % MEASURE_EVERY == 0 or step == STEPS:
                    if step == STEPS:
                        result["status"] = "completed"
                    measure(step)
        report["status"] = "completed"
        save_report()
    except (ArithmeticError, OSError) as error:
        report.update(status="failed", error=str(error))
        save_report()
        raise
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acknowledge-review", action="store_true")
    parser.add_argument("--learning-rates", action="store_true", help="Compare rates 0.1/0.03/0.01 at context 4 instead of varying context")
    parser.add_argument("--learning-rate", type=float, help="Fixed rate for context comparison (default: 0.1)")
    args = parser.parse_args()
    if args.learning_rates and args.learning_rate is not None:
        parser.error("--learning-rate cannot be combined with --learning-rates")
    root = Path(__file__).resolve().parent
    result = (
        compare_learning_rates(root, acknowledge_review=args.acknowledge_review)
        if args.learning_rates else
        compare_contexts(
            root, acknowledge_review=args.acknowledge_review,
            learning_rate=RATE if args.learning_rate is None else args.learning_rate,
        )
    )
    print(json.dumps(result, allow_nan=False))
