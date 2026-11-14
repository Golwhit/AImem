"""Bounded permanent conversation-weight rounds on explicitly reviewed teaching bytes."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import time
from uuid import uuid4

from neural_base_comparison import digest
from neural_conversation_meal import FIXTURE, paired_scores, validate_curriculum
from neural_conversation_trial import assess, canonical
from neural_plasticity import PlasticityContext, fit, replay
from original_generator import atomic_json


DIRECTORY = ".data/neural-training"
INITIAL_RUN = "ee7f3c1be14f477a82e2db03ae72ff71"


def load_progress(root: Path) -> tuple[dict, PlasticityContext]:
    directory = root / DIRECTORY
    pointer = json.loads((directory / "progress.json").read_text())
    run_id = pointer["run_id"]
    if (not isinstance(run_id, str) or len(run_id) != 32
            or any(c not in "0123456789abcdef" for c in run_id)
            or type(pointer["round"]) is not int or pointer["round"] < 1):
        raise ValueError("Invalid permanent training round selection")
    checksum = pointer["checkpoint_sha256"]
    if not isinstance(checksum, str) or len(checksum) != 64 or any(
            c not in "0123456789abcdef" for c in checksum):
        raise ValueError("Invalid permanent training checkpoint hash")
    raw = (directory / f"{checksum}.json").read_bytes()
    if digest(raw) != checksum:
        raise ValueError("Permanent training checkpoint hash mismatch")
    checkpoint = json.loads(raw)
    if checkpoint["evidence"] != {key: value for key, value in pointer.items()
                                  if key != "checkpoint_sha256"}:
        raise ValueError("Permanent training checkpoint evidence mismatch")
    return pointer, PlasticityContext.restore(checkpoint["state"])


def activate(root: Path) -> dict:
    from neural_chat import CONVERSATION_MODEL, DIRECTORY as CHAT_DIRECTORY, status, validate_base_vocabulary

    pointer, model = load_progress(root)
    directory = root / DIRECTORY
    validate_base_vocabulary(model.snapshot())
    report_bytes = (directory / "rounds" / pointer["run_id"] / "report.json").read_bytes()
    if digest(report_bytes) != pointer["report_sha256"]:
        raise ValueError("Permanent round report hash mismatch")
    report = json.loads(report_bytes)
    verified = report["validation"]
    if (report["status"] != "completed"
            or not verified.get("full_weights_and_milestones_replay_exact")
            or not (verified.get("test_replay_exact") or verified.get("correction_replay_exact"))):
        raise ValueError("Only completed replay-verified permanent rounds may activate")
    chat = root / CHAT_DIRECTORY
    previous_path = chat / "active.json"
    previous = previous_path.read_bytes()
    prior = json.loads(previous)
    if prior.get("checkpoint_sha256") == pointer["checkpoint_sha256"]:
        return status(root)
    raw = (directory / f"{pointer['checkpoint_sha256']}.json").read_bytes()
    record = {
        "model": CONVERSATION_MODEL, "arm": "conversation", "run_id": pointer["run_id"],
        "seed": 17, "passes": pointer["round"], "training_round": pointer["round"],
        "checkpoint_sha256": pointer["checkpoint_sha256"],
        "parameters": sum(map(len, model.model.weights.values())),
        "context": model.model.context, "vocabulary": model.model.size,
        "label": "Conversation learning model (persistent weights; reviewed teaching and feedback)",
        "source": {"training": ("256 approved teaching exchanges plus explicitly approved feedback"
                                if pointer.get("fixture") else
                                "64 approved teaching exchanges plus explicitly Relevant answers and submitted corrections"),
                   "training_sha256": report["review"]["train_sha256"]},
        "comparison_baseline": prior.get("comparison_baseline"),
        "rollback_selection_sha256": digest(previous),
        "retired_layers": [0, 1, 2],
        "selection_policy": "User-authorized completed, replay-verified permanent round. "
                            "Only explicitly Relevant answers and submitted corrections train from chat.",
        "note": "Experimental narrow-task model, not a grounded assistant. Raw user message receives the "
                "training role format; first generated line only, at most96 characters. Single-turn input "
                "only; retrieval/capture remain connected but are not injected into neural context. "
                "Saved weights resume without retraining from scratch. No claim retained Acts knowledge.",
    }
    (chat / f"retired-selection-{digest(previous)}.json").write_bytes(previous)
    (chat / f"{pointer['checkpoint_sha256']}.json").write_bytes(raw)
    atomic_json(previous_path, record)
    return status(root)


def rollback(root: Path) -> dict:
    from neural_chat import DIRECTORY as CHAT_DIRECTORY, load_active, load_selection, status

    record, _, _ = load_active(root)
    checksum = record.get("rollback_selection_sha256")
    if not isinstance(checksum, str) or len(checksum) != 64 or any(
            c not in "0123456789abcdef" for c in checksum):
        raise ValueError("Active model has no valid rollback selection")
    directory = root / CHAT_DIRECTORY
    raw = (directory / f"retired-selection-{checksum}.json").read_bytes()
    if digest(raw) != checksum:
        raise ValueError("Rollback selection hash mismatch")
    restored = json.loads(raw)
    load_selection(root, restored)
    atomic_json(directory / "active.json", restored)
    return status(root)


MAX_SATURATION = .6
MIN_TEACHING_EXACT = .9


def health_problem(points: list[dict]) -> str | None:
    if not points:
        return "no checkpoints recorded"
    for point in points:
        saturation = point["diagnostics"]["hidden_abs_over_099_fraction"]
        if saturation > MAX_SATURATION:
            return f"hidden saturation {saturation:.3f} above {MAX_SATURATION} at update {point['updates']}"
    final = points[-1]["training"]["exact"]
    if final < MIN_TEACHING_EXACT:
        return f"teaching exact {final:.3f} below {MIN_TEACHING_EXACT}"
    return None


def run(root: Path, *, cpu_seconds: float = 300.0, rate: float = .1) -> dict:
    raw = (root / FIXTURE).read_bytes()
    reviewed_sha256 = digest(raw)
    if type(cpu_seconds) not in (int, float) or not 1 <= cpu_seconds <= 900:
        raise ValueError("Round budget must be 1-900 CPU seconds")
    if type(rate) not in (int, float) or not 0 < rate <= .1:
        raise ValueError("Round learning rate must be in (0, 0.1]")
    data = json.loads(raw)
    validate_curriculum(data)
    directory = root / DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another permanent training round is running") from error
        if (directory / "progress.json").exists():
            parent, model = load_progress(root)
            parent_hash = parent["checkpoint_sha256"]
            round_number = parent["round"] + 1
            if parent["reviewed_file_sha256"] != reviewed_sha256:
                raise ValueError("New curriculum requires a separately designed vocabulary/approval migration")
        else:
            source = root / ".data/neural-conversation-meal" / INITIAL_RUN
            source_report = json.loads((source / "report.json").read_text())
            validation = json.loads((source / "validation.json").read_text())
            if (source_report["status"] != "completed"
                    or validation["exact_weight_and_test_replays"] != 12
                    or source_report["review"]["file_sha256"] != reviewed_sha256):
                raise ValueError("Initial reviewed, replay-verified meal checkpoint required")
            source_path = source / "fresh_scaled-17-meal64-final.json"
            source_raw = source_path.read_bytes()
            source_checkpoint = json.loads(source_raw)
            evidence = source_checkpoint["trial"]
            if (evidence["arm"], evidence["seed"], evidence["subset"]) != ("fresh_scaled", 17, "meal64"):
                raise ValueError("Permanent initialization must use first protocol seed and full teaching split")
            model = PlasticityContext.restore(source_checkpoint["state"])
            parent_hash = digest(source_raw)
            parent = {"source": str(source_path.relative_to(root)), "checkpoint_sha256": parent_hash,
                      "selection": "First protocol seed17, fresh_scaled, meal64; not best test score"}
            round_number = 1
        corrections = parent.get("approved_corrections", [])
        corrected_prompts = {canonical(item["row"]["turns"]) for item in corrections}
        teaching = [row for row in data["train"] if canonical(row["turns"]) not in corrected_prompts]
        teaching += [item["row"] for item in corrections]
        initial = json.loads(json.dumps(model.snapshot(), allow_nan=False))
        run_id = uuid4().hex
        target = directory / "rounds" / run_id
        seed = 1000000 + round_number
        report = {
            "run_id": run_id, "round": round_number, "status": "running",
            "parent": parent, "parent_checkpoint_sha256": parent_hash,
            "reviewed_file_sha256": reviewed_sha256,
            "review": {"authorization": "Standing user authorization: every round trains automatically on the "
                                        "recorded fixture bytes; no per-round approval",
                       "scope": "Exact approved 64 teaching exchanges plus explicitly approved feedback pairs; "
                                "no unmarked chat training",
                       "train_sha256": digest(canonical(teaching)),
                       "test_sha256": digest(canonical(data["test"])),
                       "provenance": data["provenance"], "record_type": "User declaration, not legal clearance"},
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "rate": rate, "seed": seed, "cpu_seconds": cpu_seconds,
            "policy": "Continue saved weights, never fresh initialization on resume. New deterministic target-draw "
                      "seed per round; no optimizer momentum to restore (plain SGD). Fixed rate and final endpoint, "
                      "no test-guided selection. Existing test set is development-only and not new evidence. "
                      "Persist checkpoints before evaluating; promotion requires exact replay and finite reload. "
                      "Only explicitly Relevant or corrected chat pairs join rehearsal. No unmarked chat training. "
                      "No continuous daemon. "
                      "No prior-corpus retention claim. Active chat is a separate checksum-pinned selection.",
        }
        atomic_json(target / "protocol.json", report)
        atomic_json(target / "inputs.json", data)
        atomic_json(target / "initial.json", {"state": initial})
        for name in ("neural_training.py", "neural_plasticity.py", "neural_experiment.py",
                     "neural_conversation_trial.py", "neural_conversation_context.py",
                     "neural_base_comparison.py"):
            path = target / "source" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((root / name).read_bytes())
        print(f"Permanent round {round_number}: {run_id}, {cpu_seconds} update-CPU seconds, rate {rate}", flush=True)
        points = fit(model, teaching, seed=seed, rate=rate,
                     cpu_checkpoints=(cpu_seconds / 5, cpu_seconds))
        state = json.loads(json.dumps(model.snapshot(), allow_nan=False))
        atomic_json(target / "trained.json", {"state": state, "checkpoints": points})
        restored = PlasticityContext.restore(json.loads(json.dumps(state)))
        report["checkpoints"] = points
        report["status"] = "trained"
        atomic_json(target / "report.json", report)
        start = time.process_time()
        report["test"] = assess(restored, data["test"])
        report["test_pairs"] = paired_scores(report["test"])
        report["evaluation_cpu_seconds"] = time.process_time() - start
        replayed = replay(initial, teaching, seed=seed, rate=rate, checkpoints=points)
        if replayed.snapshot() != state or assess(replayed, data["test"]) != report["test"]:
            raise AssertionError("Permanent round full weights or test outputs differ on replay")
        report["validation"] = {"full_weights_and_milestones_replay_exact": True,
                                "test_replay_exact": True}
        problem = health_problem(points)
        if problem:
            # Rejected rounds keep their evidence but never become the parent or the active model.
            report["status"] = "rejected"
            report["rejection"] = problem
            atomic_json(target / "report.json", report)
            print(f"Rejected permanent round {round_number}: {problem}; parent unchanged", flush=True)
            return report
        report["status"] = "completed"
        atomic_json(target / "report.json", report)
        # Evidence excludes its own content hash; the pointer binds the serialized checkpoint.
        evidence = {"run_id": run_id, "round": round_number,
                    "parent_checkpoint_sha256": parent_hash, "reviewed_file_sha256": reviewed_sha256,
                    "report_sha256": digest((target / "report.json").read_bytes())}
        if corrections:
            evidence["approved_corrections"] = corrections
        checkpoint = {"state": state, "evidence": evidence}
        encoded = canonical(checkpoint)
        checksum = digest(encoded)
        (directory / f"{checksum}.json").write_bytes(encoded)
        pointer = {**evidence, "checkpoint_sha256": checksum}
        atomic_json(directory / "progress.json", pointer)
        print(f"Saved permanent round {round_number}: {checksum}; "
              f"teaching={points[-1]['training']['exact']:.3f}; "
              f"test={report['test']['exact']:.3f}", flush=True)
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-seconds", type=float, default=300.0)
    parser.add_argument("--rate", type=float, default=.1, help="Plain SGD rate for this round (0, 0.1]")
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--activate-after", action="store_true",
                        help="Promote after this training round completes and replays exactly")
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if (args.activate and args.rollback) or (args.activate_after and (args.activate or args.rollback)):
        parser.error("Choose one activation/rollback operation")
    if args.activate:
        print(json.dumps(activate(root)))
    elif args.rollback:
        print(json.dumps(rollback(root)))
    else:
        report = run(root, cpu_seconds=args.cpu_seconds, rate=args.rate)
        if args.activate_after:
            if report["status"] != "completed":
                parser.exit(1, f"Round {report['round']} was {report['status']}; not activating\n")
            print(json.dumps(activate(root)))
