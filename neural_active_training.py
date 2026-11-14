"""Continue the active chat weights on the approved authored curriculum."""

import argparse
import fcntl
import json
import math
from pathlib import Path
import time
from uuid import uuid4

from neural_base_comparison import digest
from neural_chat import CONVERSATION_MODEL, DIRECTORY as CHAT_DIRECTORY, load_active
from neural_continuation import extend_alphabet
from neural_conversation_meal import AUTHORED_FIXTURE, paired_scores, validate_curriculum
from neural_conversation_trial import assess, canonical, prompt_for
from neural_plasticity import PlasticityContext, fit, replay
from neural_training import DIRECTORY, MAX_SATURATION, activate, load_progress
from neural_word_context import WORD
from original_generator import atomic_json


APPROVED_SHA256 = "c60c4e3e6971cec59e5bdd9704e1c47584174dd34b4f8eeb058e56fbc2b8f99e"


def verify_meal_parent(root: Path, active: dict, model: PlasticityContext, data: dict) -> None:
    source = active["source"]
    run_id = source["run_id"]
    if (len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id)
            or source["subset"] != "meal256" or source["arm"] != "fresh_scaled"):
        raise ValueError("Active model is not a reviewed authored meal checkpoint")
    directory = root / ".data/neural-conversation-meal" / run_id
    report = json.loads((directory / "report.json").read_text())
    validation = json.loads((directory / "validation.json").read_text())
    if (report["status"] != "completed"
            or report["review"]["file_sha256"] != APPROVED_SHA256
            or report["review"]["canonical_file_sha256"] != digest(canonical(data))
            or report["review"]["train_sha256"] != source["training_sha256"]
            or len(report["trials"]) != 12
            or validation["exact_weight_and_test_replays"] != 12
            or not validation["protected_checkpoints_unchanged"]):
        raise ValueError("Active meal checkpoint lacks matching reviewed, replay-validated evidence")
    trial = directory / f"{source['arm']}-{active['seed']}-meal256-final.json"
    if json.loads(trial.read_text())["state"] != model.snapshot():
        raise ValueError("Active weights differ from the validated meal checkpoint")


def round_once(root: Path, *, cpu_seconds: float = 60.0, rate: float = .02) -> dict:
    if (not math.isfinite(cpu_seconds) or not 1 <= cpu_seconds <= 900
            or not math.isfinite(rate) or not 0 < rate <= .1):
        raise ValueError("Invalid training budget or learning rate")
    raw = (root / AUTHORED_FIXTURE).read_bytes()
    if digest(raw) != APPROVED_SHA256:
        raise ValueError("Authored curriculum differs from the approved exact file")
    data = json.loads(raw)
    validate_curriculum(data)
    directory = root / DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Training is busy; retry when the current round finishes") from error
        active_path = root / CHAT_DIRECTORY / "active.json"
        active_bytes = active_path.read_bytes()
        active, model, index = load_active(root)
        if active["model"] != CONVERSATION_MODEL or not isinstance(model, PlasticityContext) or index is not None:
            raise ValueError("Active chat must use a layer-free conversation model")
        pointer_path = directory / "progress.json"
        previous_pointer = json.loads(pointer_path.read_text()) if pointer_path.exists() else None
        if previous_pointer is not None:
            previous_pointer, _ = load_progress(root)
        if previous_pointer and previous_pointer["checkpoint_sha256"] == active["checkpoint_sha256"]:
            if (previous_pointer.get("fixture") != AUTHORED_FIXTURE
                    or previous_pointer["reviewed_file_sha256"] != APPROVED_SHA256):
                raise ValueError("Active permanent progress uses a different approved curriculum")
        else:
            verify_meal_parent(root, active, model, data)
        corrections = previous_pointer.get("approved_corrections", []) if previous_pointer else []
        for item in corrections:
            row = item["row"]
            if (len(row["turns"]) != 1 or row["turns"][0]["role"] != "user"
                    or item["sha256"] != digest(canonical(
                        {"prompt": row["turns"][0]["content"], "answer": row["answer"]}))):
                raise ValueError("Approved feedback evidence is invalid")
        corrected = {canonical(item["row"]["turns"]) for item in corrections}
        teaching = [row for row in data["train"] if canonical(row["turns"]) not in corrected]
        feedback = [item["row"] for item in corrections]
        teaching += feedback
        parent_state = json.loads(json.dumps(model.snapshot(), allow_nan=False))
        before_train = assess(model, teaching)
        before_test = assess(model, data["test"])
        before_feedback = assess(model, feedback) if feedback else None
        symbols = set()
        for row in feedback:
            text = prompt_for(row) + row["answer"] + "\n"
            symbols.update("char:" + char for char in text)
            symbols.update("identity:" + match.group().lower().replace("’", "'") for match in WORD.finditer(text))
        expanded = extend_alphabet(model.model, sorted(symbols))
        state = model.snapshot()
        state.update(symbols=expanded.characters, weights=expanded.weights)
        model = PlasticityContext.restore(state)
        initial = json.loads(json.dumps(model.snapshot(), allow_nan=False))
        round_number = (previous_pointer["round"] if previous_pointer else 0) + 1
        seed = 1000000 + round_number
        run_id = uuid4().hex
        target = directory / "rounds" / run_id
        report = {
            "run_id": run_id, "round": round_number, "status": "running",
            "kind": "active_authored_continuation",
            "parent_checkpoint_sha256": active["checkpoint_sha256"],
            "previous_progress": previous_pointer, "fixture": AUTHORED_FIXTURE,
            "reviewed_file_sha256": APPROVED_SHA256,
            "review": {"train_sha256": digest(canonical(teaching)),
                       "test_sha256": digest(canonical(data["test"])),
                       "scope": "Approved authored teaching split and explicitly approved feedback only",
                       "provenance": data["provenance"], "record_type": "User declaration, not legal clearance"},
            "seed": seed, "rate": rate, "cpu_seconds": cpu_seconds,
            "before": {"training": before_train, "test": before_test, "feedback": before_feedback},
            "policy": "Continue active weights; approved feedback supersedes matching curriculum prompts. "
                      "Expand vocabulary only for approved feedback. Evaluation is development-only and used "
                      "as a non-regression gate, not independent evidence. Stop on any rejected round.",
        }
        atomic_json(target / "protocol.json", report)
        atomic_json(target / "inputs.json", {"curriculum": data, "teaching": teaching})
        atomic_json(target / "parent.json", {"state": parent_state, "selection": active})
        atomic_json(target / "initial.json", {"state": initial})
        source_root = Path(__file__).resolve().parent
        for name in ("neural_active_training.py", "neural_training.py", "neural_correction.py",
                     "neural_plasticity.py", "neural_experiment.py", "neural_conversation_trial.py",
                     "neural_conversation_context.py", "neural_conversation_meal.py",
                     "neural_base_comparison.py", "neural_continuation.py", "neural_word_context.py"):
            path = target / "source" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((source_root / name).read_bytes())
        print(f"Active continuation round {round_number}: {run_id}, {cpu_seconds} update-CPU seconds", flush=True)
        points = fit(model, teaching, seed=seed, rate=rate, cpu_checkpoints=(cpu_seconds / 5, cpu_seconds))
        state = json.loads(json.dumps(model.snapshot(), allow_nan=False))
        atomic_json(target / "trained.json", {"state": state, "checkpoints": points})
        after_test = assess(model, data["test"])
        after_feedback = assess(model, feedback) if feedback else None
        replayed = replay(initial, teaching, seed=seed, rate=rate, checkpoints=points)
        if replayed.snapshot() != state or assess(replayed, data["test"]) != after_test:
            raise AssertionError("Continuation weights or evaluation differ on replay")
        before_pairs, after_pairs = paired_scores(before_test), paired_scores(after_test)
        failures = []
        if any(point["diagnostics"]["hidden_abs_over_099_fraction"] > MAX_SATURATION for point in points):
            failures.append("hidden saturation exceeds limit")
        if points[-1]["training"]["exact"] < before_train["exact"]:
            failures.append("teaching accuracy regressed")
        if after_pairs["both_exact"] < before_pairs["both_exact"]:
            failures.append("development paired accuracy regressed")
        if before_feedback and after_feedback:
            if any(before["exact"] and not after["exact"]
                   for before, after in zip(before_feedback["rows"], after_feedback["rows"])):
                failures.append("previously correct approved feedback regressed")
        report.update(checkpoints=points, test=after_test, test_pairs=after_pairs,
                      feedback=after_feedback,
                      validation={"full_weights_and_milestones_replay_exact": True, "test_replay_exact": True})
        if failures:
            report.update(status="rejected", rejection=failures)
            atomic_json(target / "report.json", report)
            raise RuntimeError("Continuation rejected: " + "; ".join(failures))
        if active_path.read_bytes() != active_bytes:
            raise RuntimeError("Active model changed during training; candidate was not promoted")
        report["status"] = "completed"
        atomic_json(target / "report.json", report)
        evidence = {"run_id": run_id, "round": round_number,
                    "parent_checkpoint_sha256": active["checkpoint_sha256"],
                    "reviewed_file_sha256": APPROVED_SHA256, "fixture": AUTHORED_FIXTURE,
                    "report_sha256": digest((target / "report.json").read_bytes()),
                    "approved_corrections": corrections}
        encoded = canonical({"state": state, "evidence": evidence})
        checksum = digest(encoded)
        (directory / f"{checksum}.json").write_bytes(encoded)
        atomic_json(pointer_path, {**evidence, "checkpoint_sha256": checksum})
        try:
            activate(root)
        except (ValueError, KeyError, TypeError, ArithmeticError, OSError):
            if previous_pointer is not None:
                atomic_json(pointer_path, previous_pointer)
            else:
                pointer_path.unlink()
            if active_path.read_bytes() != active_bytes:
                current = json.loads(active_path.read_text())
                if current.get("checkpoint_sha256") == checksum:
                    atomic_json(active_path, active)
            raise
        print(f"Activated continuation {run_id}: {checksum}; "
              f"development pairs {before_pairs['both_exact']} -> {after_pairs['both_exact']}", flush=True)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-seconds", type=float, default=60.0)
    parser.add_argument("--rate", type=float, default=.02)
    parser.add_argument("--continuous", action="store_true",
                        help="Continue successive verified rounds until interrupted or a guard rejects a round")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    while True:
        round_once(root, cpu_seconds=args.cpu_seconds, rate=args.rate)
        if not args.continuous:
            break
        print("Waiting 15 seconds before the next round; feedback training can run during this pause.", flush=True)
        time.sleep(15)


if __name__ == "__main__":
    main()
