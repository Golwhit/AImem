"""Matched Acts baseline/full-layer/ablation efficiency experiment."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from uuid import uuid4

from neural_experiment import NeuralModel
from neural_layers import DevelopmentalPrior, evaluate, sample
from neural_rounds import load_acts
from original_generator import atomic_json
from seed_life import boundary_text, exclusion_reason


STEPS = 5000
MEASURE_EVERY = 500
RATE = 0.03
ARMS = (
    ("baseline", False, False, False, True),
    ("full_012", True, True, True, True),
    ("without_0", False, True, True, True),
    ("without_1", True, False, True, True),
    ("without_2", True, True, False, True),
    ("layers_only", True, True, True, False),
)


def first_match(history: list[dict], target: float) -> dict | None:
    return next((
        row for row in history
        if row["validation"]["loss"] is not None and row["validation"]["loss"] <= target
    ), None)


def run_efficiency(root: Path, *, acknowledge_review: bool = False) -> dict:
    verification_started = time.perf_counter()
    dataset = load_acts(root, acknowledge_review, include_final_test=False)
    verification_seconds = time.perf_counter() - verification_started
    train, held = dataset["training"], dataset["validation"]
    rng = random.Random(19)
    draws = [rng.randrange(len(train)) for _ in range(STEPS)]
    run_id = uuid4().hex
    directory = root / ".data/neural-experiments"
    run_path = directory / run_id
    report = {
        "version": 1, "kind": "layered-efficiency", "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "running",
        "steps_per_neural_arm": STEPS, "measurement_interval": MEASURE_EVERY,
        "learning_rate": RATE, "context": 4, "batch_size": 1,
        "initialization_seed": 17, "training_seed": 19, "sampling_seed": 23,
        "training_characters": len(train), "held_out_characters": len(held),
        "training_sha256": hashlib.sha256(train.encode()).hexdigest(),
        "held_out_sha256": hashlib.sha256(held.encode()).hexdigest(),
        "training_draws_sha256": hashlib.sha256(json.dumps(draws).encode()).hexdigest(),
        "shared_verification_seconds": verification_seconds,
        "final_test_evaluations": 0, "models": [], "target_validation_loss": None,
        "data_provenance": {
            "manifest": dataset["manifest"], "manifest_sha256": dataset["manifest_sha256"],
            "approval": "User approved isolated evidence-driven candidate growth and full Layer 0/1/2 efficiency testing. Candidates are not confirmed forms or permanent admissions.",
        },
        "policy": {
            "layer_zero": "Keep the seven foundation forms. Grow separate 1-4-character candidate blocks from approved training observations, shortest first, requiring both overlapping subblocks for compositions. Confirmed forms never expand automatically. At prediction, known foundation/candidate suffixes have availability weight 1; unseen suffixes retain 0.05 exploration, not a declaration of impossibility.",
            "layer_one": "Existing scoped exclusions zero the probability of forbidden next characters before normalization and gradient calculation. Conflicting original targets are reported, not repaired or silently removed from evaluation.",
            "layer_two": "Build existing 1-3-context attraction once from training. Renormalize its character mass, reserve 0.05 uniform mass over neural tokens, and multiply into neural probabilities before normalization and cross-entropy/backpropagation.",
            "growth": "Candidate primitive characters come from approved training only; longer blocks compose from known subblocks. This operational experiment is not a new universal law or automatic confirmation.",
            "work": "All arms read the same full training source. Fewer gradient updates do not imply fewer source characters. Charge source scans, layer construction and cached training priors to training preparation; time updates separately. Evaluation includes validation-prior preparation, full-split prediction and samples. Repeated evaluation reuses fixed prior caches; validation prediction time alone is not uncached serving latency. Samples compute priors at each character. End-to-end includes evaluation and prior checkpoint writes, with current-write completion tracked separately.",
        },
        "evaluation_note": (
            "Compare raw original next-character targets, including unknowns. Any forbidden target "
            "makes full-text loss infinite (JSON null); coverage and blocked offsets remain visible. "
            "The target is this run's baseline final validation loss; first match is at sampled "
            "checkpoints, not an exact crossing. Acts 2 is reused monitoring data, not a fresh test. "
            "Acts 3 text is neither read nor evaluated. One seed and shared-machine timings cannot "
            "establish a general speedup or understanding. Generated output is not scripture or "
            "the World English Bible. No installed model, foundation, or admission registry changes."
        ),
    }

    def save_report() -> None:
        atomic_json(run_path / "report.json", report)
        atomic_json(directory / "latest-layered.json", report)

    save_report()
    try:
        for name, zero, one, two, neural in ARMS:
            started = time.perf_counter()
            prepare_cpu = time.process_time()
            model = NeuralModel(sorted(set(train)))
            training = model.examples(train)
            prior = DevelopmentalPrior(train, layer_zero=zero, layer_one=one, layer_two=two) if zero or one or two else None
            training_priors = [
                prior.prior(train[max(0, offset - 4):offset], model)
                for offset in range(len(train))
            ] if prior else None
            preparation_cpu = time.process_time() - prepare_cpu
            preparation_seconds = time.perf_counter() - started
            evidence = prior.evidence() if prior else None
            evidence_bytes = len(json.dumps(evidence).encode()) if evidence else 0
            atomic_json(run_path / name / "evidence.json", {
                "training_text": train, "held_out_text": held,
                "training_sha256": report["training_sha256"], "evidence": evidence,
                "data_provenance": report["data_provenance"],
            })
            evaluation_started = time.perf_counter()
            validation = model.examples(held)
            validation_priors = [
                prior.prior(held[max(0, offset - 4):offset], model)
                for offset in range(len(held))
            ] if prior else None
            result = {
                "name": name, "status": "running", "layers": [i for i, on in enumerate((zero, one, two)) if on],
                "neural": neural, "parameter_count": sum(map(len, model.weights.values())) if neural else 0,
                "layer_state_json_bytes": evidence_bytes,
                "source_characters_read": len(train), "source_character_observations": len(train),
                "layer_zero_window_observations": prior.observations if prior else 0,
                "layer_one_source_checks": len(train) if one else 0,
                "layer_two_source_characters": len(train) if two else 0,
                "layer_two_context_observations": sum(len(train) - size + 1 for size in (1, 2, 3)) if two else 0,
                "layer_two_analysis_windows": sum(len(train) - size + 1 for size in (2, 3, 4)) if two else 0,
                "candidate_count": len(prior.candidates) if prior else 0,
                "confirmed_form_count": 5 if zero else 0,
                "training_preparation_seconds": preparation_seconds,
                "training_preparation_cpu_seconds": preparation_cpu,
                "update_seconds": 0.0, "update_cpu_seconds": 0.0,
                "evaluation_seconds": time.perf_counter() - evaluation_started,
                "validation_prior_preparation_seconds": time.perf_counter() - evaluation_started,
                "checkpoint_seconds": 0.0, "history": [], "samples": [],
                "attempted_targets": 0, "gradient_updates": 0, "skipped_rule_conflicts": 0,
                "held_out_unknown_characters": sum(char not in model.tokens for char in held),
                "first_match": None,
            }
            report["models"].append(result)

            def measure(step: int) -> None:
                evaluation_started = time.perf_counter()
                training_metrics = evaluate(model, training, training_priors, neural=neural)
                validation_started = time.perf_counter()
                validation_metrics = evaluate(model, validation, validation_priors, neural=neural)
                validation_seconds = time.perf_counter() - validation_started
                result["samples"] = []
                for prompt in ("The ", "They", train[:64]):
                    text = sample(model, prior, prompt, neural=neural)
                    result["samples"].append({
                        "prompt": prompt, "text": text,
                        "layer_one_violation": exclusion_reason(boundary_text(prompt + text, True)),
                    })
                result["evaluation_seconds"] += time.perf_counter() - evaluation_started
                row = {
                    "step": step, "gradient_updates": result["gradient_updates"],
                    "training": training_metrics, "validation": validation_metrics,
                    "training_work_seconds": preparation_seconds + result["update_seconds"],
                    "training_work_cpu_seconds": preparation_cpu + result["update_cpu_seconds"],
                    "validation_prediction_seconds": validation_seconds,
                    "end_to_end_seconds": time.perf_counter() - started + verification_seconds,
                }
                result["history"].append(row)
                if name == "baseline" and step == STEPS:
                    report["target_validation_loss"] = validation_metrics["loss"]
                target = report["target_validation_loss"]
                if target is not None:
                    result["first_match"] = first_match(result["history"], target)
                checkpoint_started = time.perf_counter()
                atomic_json(run_path / name / f"step-{step:05}.json", {
                    "model": "original-character-mlp-v1" if neural else "developmental-layers-only-v1",
                    "context": model.context, "learning_rate": RATE,
                    "characters": model.characters, "weights": model.weights if neural else None,
                    "comparison_run_id": run_id, "result": result,
                    "evidence_file": "evidence.json",
                })
                save_report()
                result["checkpoint_seconds"] += time.perf_counter() - checkpoint_started
                print(
                    f"{name} step {step}/{STEPS}: validation={validation_metrics['loss']} "
                    f"coverage={validation_metrics['coverage']:.4f} updates={result['gradient_updates']}",
                    file=sys.stderr, flush=True,
                )

            if not neural:
                result["status"] = "completed"
            measure(0)
            if neural:
                for step, index in enumerate(draws, 1):
                    context, target = training[index]
                    output_prior = training_priors[index] if training_priors is not None else None
                    result["attempted_targets"] += 1
                    if output_prior is not None and output_prior[target] == 0:
                        result["skipped_rule_conflicts"] += 1
                    else:
                        update_started, cpu_started = time.perf_counter(), time.process_time()
                        model.update(context, target, learning_rate=RATE, output_prior=output_prior)
                        result["update_seconds"] += time.perf_counter() - update_started
                        result["update_cpu_seconds"] += time.process_time() - cpu_started
                        result["gradient_updates"] += 1
                    if step % MEASURE_EVERY == 0 or step == STEPS:
                        if not all(math.isfinite(value) for values in model.weights.values() for value in values):
                            raise ArithmeticError("Nonfinite neural weights")
                        if step == STEPS:
                            result["status"] = "completed"
                        measure(step)
            result["status"] = "completed"
            result["end_to_end_seconds"] = time.perf_counter() - started + verification_seconds
            save_report()
        report["status"] = "completed"
        save_report()
    except (ValueError, ArithmeticError, OSError) as error:
        report.update(status="failed", error=str(error))
        save_report()
        raise
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acknowledge-review", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_efficiency(
        Path(__file__).resolve().parent, acknowledge_review=args.acknowledge_review,
    ), allow_nan=False))
