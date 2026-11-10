"""Fixed 50-round training on synthetic diagnostics or explicitly approved Acts text."""

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import random
import re
import sys
from uuid import uuid4

from layer_two import Attraction
from neural_experiment import CONTEXT, EMBEDDING, HIDDEN, RATE, NeuralModel
from original_generator import atomic_json


TRAIN = "I am a seed. A seed can grow. I can learn a pattern. A pattern can guide a seed."
VALIDATION = "A seed can learn. I can grow a pattern. A pattern can guide."
TEST = "I can guide a seed. A pattern can grow. I am a seed."
PROMPTS = ("I am", "A se", "I ca")
ROUNDS = 50
UPDATES = 100


def load_acts(root: Path, acknowledge_review: bool, *, include_final_test: bool = True) -> dict:
    if acknowledge_review is not True:
        raise ValueError(
            "Explicit approval of Acts 1 training / Acts 2 validation"
            + (" / Acts 3 testing" if include_final_test else "") + " required"
        )
    directory = root / ".data" / "acts-web-staged"
    manifest_bytes = (directory / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if hashlib.sha256((directory / "rights.html").read_bytes()).hexdigest() != manifest["rights_sha256"]:
        raise ValueError("Acts rights record hash mismatch")
    texts = {}
    roles = [("training", 1), ("validation", 2)]
    if include_final_test:
        roles.append(("final_test", 3))
    for role, chapter in roles:
        sources = [item for item in manifest["sources"] if item["role"] == role]
        if len(sources) != 1 or sources[0]["chapter"] != chapter:
            raise ValueError("Unexpected Acts chapter split")
        source = sources[0]
        raw = (directory / f"acts-{chapter:02}.html").read_bytes()
        if hashlib.sha256(raw).hexdigest() != source["raw_sha256"]:
            raise ValueError(f"Acts {chapter} raw source hash mismatch")
        # Fixed names avoid treating manifest text as an arbitrary file path.
        text = (directory / f"{role}.txt").read_bytes().decode("utf-8")
        if hashlib.sha256(text.encode()).hexdigest() != source["text_sha256"]:
            raise ValueError(f"Acts {chapter} extracted text hash mismatch")
        if not 8 <= len(text) <= 50_000 or len(text) != source["characters"]:
            raise ValueError(f"Acts {chapter} text length mismatch or unsupported size")
        texts[role] = text
    if len(set(texts["training"])) > 128:
        raise ValueError("Acts training vocabulary exceeds this experiment's 128-character limit")
    return {
        **texts, "manifest": manifest,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "approval": "User explicitly authorized staged Acts 1 training, Acts 2 validation, Acts 3 final test with 'run it for real'. Not approval for other sources or admissions.",
    }


def run_rounds(
    root: Path, *, acts: bool = False, acknowledge_review: bool = False,
    resume_run: str | None = None, additional_rounds: int = 100,
) -> dict:
    dataset = load_acts(root, acknowledge_review) if acts else None
    train_text = dataset["training"] if dataset else TRAIN
    validation_text = dataset["validation"] if dataset else VALIDATION
    test_text = dataset["final_test"] if dataset else TEST
    prompts = ("The ", "They", "Pete") if acts else PROMPTS
    model = NeuralModel(sorted(set(train_text)))
    training = model.examples(train_text)
    validation = model.examples(validation_text)
    attraction = Attraction(train_text, max_characters=50_000)
    attraction.report["source"] = "Acts 1 training split" if acts else "synthetic training split"
    rng = random.Random(19)
    initial = {name: values[:] for name, values in model.weights.items()}
    run_id = uuid4().hex
    directory = root / ".data" / "neural-experiments"
    rounds_path = directory / run_id
    report = {
        "version": 3, "run_id": run_id, "model": "original-character-mlp-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "synthetic": not acts, "automated_synthetic": not acts, "automated_rounds": True, "status": "running",
        "source": "Public-domain World English Bible: Acts 1 train / Acts 2 validation / Acts 3 final test" if acts else
        "Original synthetic seed/pattern diagnostic passages; user authorized mechanics run",
        "data_provenance": {
            "manifest": dataset["manifest"], "manifest_sha256": dataset["manifest_sha256"],
            "approval": dataset["approval"],
        } if dataset else None,
        "legal_clearance": False, "rounds_planned": ROUNDS, "rounds_completed": 0,
        "updates_per_round": UPDATES, "steps": 0, "processed_target_tokens": 0,
        "parameter_count": sum(map(len, model.weights.values())),
        "trainable_parameters": sum(map(len, model.weights.values())),
        "changed_parameters": 0, "context": CONTEXT, "embedding": EMBEDDING, "hidden": HIDDEN,
        "vocabulary": model.size, "training_characters": len(train_text),
        "held_out_characters": len(validation_text),
        "held_out_unknown_characters": sum(char not in model.tokens for char in validation_text),
        "test_characters": len(test_text),
        "test_unknown_characters": sum(char not in model.tokens for char in test_text),
        "training_sha256": hashlib.sha256(train_text.encode()).hexdigest(),
        "held_out_sha256": hashlib.sha256(validation_text.encode()).hexdigest(),
        "test_sha256": hashlib.sha256(test_text.encode()).hexdigest(),
        "learning_rate": RATE, "initialization_seed": 17, "training_seed": 19,
        "sample_prompt": prompts[0], "initial_sample": model.sample(prompts[0]),
        "sample": "", "history": [], "fixed_prompt_samples": [],
        "layer_policy": "Unconstrained neural baseline; Layer 0/1 unchanged. Layer 2 counted once from training only, not fed into neural updates.",
        "evaluation_note": (
            "Real-text continuation training. Related chapters are held out by source but share language/style. "
            "Loss includes unknown-token targets and is not a test of fact recall, question-answering or understanding. "
            "Generated text is model output, not scripture or the World English Bible. "
            if acts else "Synthetic related passages share vocabulary and patterns. This checks training mechanics, not independent generalization or question-answering. "
        ) + "Final test evaluated once after round 50; no checkpoint selection.",
        "layer_two": {
            "training_evidence": attraction.report,
            "validation_metrics": attraction.metrics(validation_text),
            "count_builds": 1,
        },
    }
    start_round = 0
    end_round = ROUNDS
    if resume_run is not None:
        if not acts or not re.fullmatch(r"[a-f0-9]{32}", resume_run):
            raise ValueError("Continuation requires an Acts run ID and explicit Acts approval")
        if type(additional_rounds) is not int or not 1 <= additional_rounds <= 100:
            raise ValueError("Additional rounds must be an integer from 1 to 100")
        parent_path = directory / f"{resume_run}.json"
        parent_bytes = parent_path.read_bytes()
        parent = json.loads(parent_bytes)
        previous = parent["report"]
        if previous["status"] != "completed" or previous["synthetic"]:
            raise ValueError("Only completed real-text checkpoints can be continued")
        for key in ("training_sha256", "held_out_sha256", "test_sha256",
                    "context", "embedding", "hidden", "learning_rate", "updates_per_round"):
            if previous[key] != report[key]:
                raise ValueError(f"Continuation configuration mismatch: {key}")
        if parent["characters"] != model.characters or set(parent["weights"]) != set(model.weights):
            raise ValueError("Continuation vocabulary or weight groups mismatch")
        for name, values in parent["weights"].items():
            if len(values) != len(model.weights[name]) or not all(math.isfinite(value) for value in values):
                raise ValueError("Invalid continuation weights")
        model.weights = parent["weights"]
        initial = {name: values[:] for name, values in model.weights.items()}
        start_round = previous["rounds_completed"]
        if previous["steps"] != start_round * UPDATES:
            raise ValueError("Parent steps do not match completed rounds")
        # Older checkpoints predate saved RNG state. Replay draws, not gradient updates.
        for _ in range(previous["steps"]):
            rng.choice(training)
        end_round = start_round + additional_rounds
        report.update(
            parent_run_id=resume_run, parent_checkpoint_sha256=hashlib.sha256(parent_bytes).hexdigest(),
            rounds_planned=end_round, rounds_completed=start_round,
            steps=previous["steps"], processed_target_tokens=previous["steps"],
            additional_rounds=additional_rounds, initial_sample=model.sample(prompts[0]),
            sample=model.sample(prompts[0]),
            historical_test_loss=previous.get("final_test_loss", previous.get("historical_test_loss")),
            final_test_evaluations=0,
            evaluation_note=(
                "Continuation from saved Acts weights with original SGD draw sequence. "
                "Acts 3 was already evaluated in the parent lineage; it is not retested here. "
                "Validation is monitoring data, not an untouched final test. "
                "Generated text is model output, not scripture or the World English Bible. "
                "Prediction loss is not evidence of understanding."
            ),
        )

    def measure(round_number: int) -> None:
        row = {
            "round": round_number, "step": round_number * UPDATES,
            "training_loss": model.loss(training), "held_out_loss": model.loss(validation),
        }
        if not all(math.isfinite(row[key]) for key in ("training_loss", "held_out_loss")):
            raise ArithmeticError("Non-finite round loss; run halted")
        report["history"].append(row)
        if round_number % 10 == 0:
            report["fixed_prompt_samples"].append({
                "round": round_number,
                "samples": [{"prompt": prompt, "text": model.sample(prompt)} for prompt in prompts],
            })

    def save(round_number: int) -> None:
        if not all(math.isfinite(value) for values in model.weights.values() for value in values):
            raise ArithmeticError("Non-finite round weights; run halted")
        checkpoint = {
            "report": report, "characters": model.characters, "weights": model.weights,
            "training_text": train_text, "held_out_text": validation_text, "test_text": test_text,
            "approval": dataset["approval"] if dataset else "explicit synthetic mechanics experiment only",
        }
        atomic_json(rounds_path / f"round-{round_number:02}.json", checkpoint)
        atomic_json(directory / f"{run_id}.json", checkpoint)
        atomic_json(directory / "latest.json", report)

    measure(start_round)
    save(start_round)
    try:
        for round_number in range(start_round + 1, end_round + 1):
            for _ in range(UPDATES):
                context, target = rng.choice(training)
                model.update(context, target)
            report.update(
                rounds_completed=round_number, steps=round_number * UPDATES,
                processed_target_tokens=round_number * UPDATES,
                changed_parameters=sum(
                    a != b for name in initial for a, b in zip(initial[name], model.weights[name])
                ),
            )
            measure(round_number)
            report["sample"] = model.sample(prompts[0])
            save(round_number)
            print(f"Round {round_number}/{end_round}: train={report['history'][-1]['training_loss']:.4f} validation={report['history'][-1]['held_out_loss']:.4f}", file=sys.stderr, flush=True)
        if resume_run is None:
            report["final_test_loss"] = model.loss(model.examples(test_text))
            if not math.isfinite(report["final_test_loss"]):
                raise ArithmeticError("Non-finite final test loss")
            report["layer_two"]["final_test_metrics"] = attraction.metrics(test_text)
            report["final_test_evaluations"] = 1
        report["status"] = "completed"
        save(end_round)
    except (ArithmeticError, OSError) as error:
        # Keep the last finite checkpoint; expose failure rather than label it completed.
        failed = {**report, "status": "failed", "error": str(error)}
        atomic_json(directory / "latest.json", failed)
        raise
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acts", action="store_true", help="Use staged, hash-verified Acts chapter split")
    parser.add_argument("--acknowledge-review", action="store_true", help="Explicitly approve staged Acts split for this run")
    parser.add_argument("--resume-run", help="Completed Acts run ID to continue without retesting Acts 3")
    parser.add_argument("--additional-rounds", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(run_rounds(
        Path(__file__).resolve().parent, acts=args.acts, acknowledge_review=args.acknowledge_review,
        resume_run=args.resume_run, additional_rounds=args.additional_rounds,
    ), allow_nan=False))
