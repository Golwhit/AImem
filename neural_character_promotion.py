"""Select a reviewed character-only checkpoint with an explicit regression exception."""

import argparse
import fcntl
import json
from pathlib import Path
from uuid import uuid4

from neural_base_comparison import digest
from neural_chat import CONVERSATION_MODEL, DIRECTORY as CHAT_DIRECTORY, load_active
from neural_conversation_trial import canonical, feedback_hash
from neural_plasticity import PlasticityContext
from neural_training import DIRECTORY, MAX_SATURATION, activate, load_progress
from original_generator import atomic_json


def pinned(path: Path, expected: str) -> bytes:
    if (not isinstance(expected, str) or len(expected) != 64
            or any(char not in "0123456789abcdef" for char in expected)):
        raise ValueError("A valid exact SHA-256 is required")
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError(f"Hash mismatch: {path.name}")
    return raw


def select(root: Path, *, candidate_dir: Path, report_sha256: str, assessment_sha256: str,
           fresh_meal: Path, previous_meal: Path, accept_regression: bool) -> dict:
    if accept_regression is not True:
        raise ValueError("Explicit acceptance of the measured regression is required")
    candidate_report_raw = pinned(candidate_dir / "report.json", report_sha256)
    assessment_raw = pinned(candidate_dir / "activation-assessment.json", assessment_sha256)
    candidate_report, assessment = json.loads(candidate_report_raw), json.loads(assessment_raw)
    candidate_raw = pinned(candidate_dir / "trained.json", candidate_report["trained_sha256"])
    candidate = json.loads(candidate_raw)
    if (candidate_report["kind"] != "private_fresh_meal_rehearsal_continuation"
            or not candidate_report["validation"] or not all(value is True for value in candidate_report["validation"].values())
            or candidate["output_mode"] != "character_only"
            or assessment["candidate_report_sha256"] != report_sha256
            or assessment["candidate_checkpoint_sha256"] != digest(candidate_raw)
            or not assessment["validation"] or not all(value is True for value in assessment["validation"].values())
            or not assessment["activation_check_failures"]
            or assessment["hidden_saturation"]["limit"] != MAX_SATURATION
            or assessment["hidden_saturation"]["midpoint"] > MAX_SATURATION
            or assessment["hidden_saturation"]["final"] > MAX_SATURATION):
        raise ValueError("Candidate or regression assessment is not valid")
    fresh_raw = pinned(fresh_meal, candidate_report["training"]["fresh_meal_sha256"])
    previous_raw = pinned(previous_meal, candidate_report["training"]["previous_meal_sha256"])
    sources = {}
    for name, checksum in candidate_report["source_sha256"].items():
        if Path(name).name != name or not name.endswith(".py"):
            raise ValueError("Invalid source snapshot name")
        sources[name] = pinned(candidate_dir / "source" / name, checksum)
    state = {**candidate["state"], "output_mode": "character_only"}
    model = PlasticityContext.restore(state)
    if model.output_mode != "character_only":
        raise ValueError("Candidate output mode was not restored")
    directory = root / DIRECTORY
    with (directory / "training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Training is busy; retry selection when it finishes") from error
        active_path = root / CHAT_DIRECTORY / "active.json"
        progress_path = directory / "progress.json"
        active_raw, progress_raw = active_path.read_bytes(), progress_path.read_bytes()
        active, _, index = load_active(root)
        progress, _ = load_progress(root)
        if (active["model"] != CONVERSATION_MODEL or index is not None
                or active["checkpoint_sha256"] != progress["checkpoint_sha256"]
                or active["checkpoint_sha256"] != assessment["live_checkpoint_sha256"]
                or digest(active_raw) != assessment["live_selection_sha256"]
                or digest(progress_raw) != assessment["progress_sha256"]
                or progress["reviewed_file_sha256"] != assessment["reviewed_fixture_sha256"]):
            raise ValueError("Live model or training progress changed since the comparison")
        pinned(root / progress["fixture"], progress["reviewed_file_sha256"])
        corrections = progress.get("approved_corrections", [])
        for item in corrections:
            if item["sha256"] != feedback_hash(item["row"]["turns"], item["row"]["answer"]):
                raise ValueError("Approved feedback hash mismatch")
        run_id = uuid4().hex
        round_number = progress["round"] + 1
        target = directory / "rounds" / run_id
        target.mkdir(mode=0o700, parents=False, exist_ok=False)
        for name, raw in (("candidate.json", candidate_raw), ("candidate-report.json", candidate_report_raw),
                          ("assessment.json", assessment_raw), ("fresh-meal.json", fresh_raw),
                          ("previous-meal.json", previous_raw)):
            (target / name).write_bytes(raw)
        source_dir = target / "source"
        source_dir.mkdir(mode=0o700)
        for name, raw in sources.items():
            (source_dir / name).write_bytes(raw)
        report = {
            "kind": "user_authorized_character_base", "status": "completed", "run_id": run_id,
            "round": round_number, "parent_checkpoint_sha256": active["checkpoint_sha256"],
            "previous_selection_sha256": digest(active_raw), "candidate_sha256": digest(candidate_raw),
            "candidate_report_sha256": report_sha256, "assessment_sha256": assessment_sha256,
            "review": {"train_sha256": candidate_report["training"]["reviewed_training_sha256"],
                       "fresh_meal_sha256": candidate_report["training"]["fresh_meal_sha256"],
                       "previous_meal_sha256": candidate_report["training"]["previous_meal_sha256"]},
            "exception": {"user_accepted_regression": True,
                          "failed_checks": assessment["activation_check_failures"],
                          "scope": "This selection only; later training retains its normal safeguards."},
            "validation": {"candidate_replay_exact": True, "milestone_replay_exact": True},
            "note": "The candidate starts from separately initialized weights, not merged live weights. "
                    "Approved feedback records are retained, but some answers were not retained by the candidate.",
        }
        atomic_json(target / "report.json", report)
        evidence = {"run_id": run_id, "round": round_number,
                    "parent_checkpoint_sha256": active["checkpoint_sha256"],
                    "reviewed_file_sha256": progress["reviewed_file_sha256"], "fixture": progress["fixture"],
                    "report_sha256": digest((target / "report.json").read_bytes()),
                    "approved_corrections": corrections}
        encoded = canonical({"state": state, "evidence": evidence})
        checksum = digest(encoded)
        (directory / f"{checksum}.json").write_bytes(encoded)
        experiment = root / "universe-lab" / "experiments" / f"character-base-{run_id}"
        experiment.mkdir(mode=0o700, parents=False, exist_ok=False)
        atomic_json(experiment / "initial.json", {
            "state": state, "output_mode": "character_only", "base_checkpoint_sha256": checksum,
            "note": "Independent starting copy of the selected base; no additional training.",
        })
        copied = json.loads((experiment / "initial.json").read_bytes())
        if copied["state"] != state or copied["base_checkpoint_sha256"] != checksum:
            raise AssertionError("Experimental base copy differs from the selection")
        if active_path.read_bytes() != active_raw or progress_path.read_bytes() != progress_raw:
            raise RuntimeError("Live state changed while preparing the selection")
        atomic_json(progress_path, {**evidence, "checkpoint_sha256": checksum})
        try:
            selected = activate(root)
            _, restored, _ = load_active(root)
            if selected["checkpoint_sha256"] != checksum or restored.snapshot() != state:
                raise AssertionError("Selected model differs from the experimental base copy")
        except (ValueError, KeyError, TypeError, ArithmeticError, OSError, AssertionError):
            atomic_json(progress_path, progress)
            if active_path.read_bytes() != active_raw:
                current = json.loads(active_path.read_bytes())
                if current.get("checkpoint_sha256") == checksum:
                    atomic_json(active_path, active)
            raise
    return {"selection": selected, "experimental_copy": str(experiment / "initial.json"),
            "report": str(target / "report.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--assessment-sha256", required=True)
    parser.add_argument("--fresh-meal", required=True, type=Path)
    parser.add_argument("--previous-meal", required=True, type=Path)
    parser.add_argument("--accept-regression", action="store_true")
    args = parser.parse_args()
    result = select(Path(__file__).resolve().parent, candidate_dir=args.candidate_dir,
                    report_sha256=args.report_sha256, assessment_sha256=args.assessment_sha256,
                    fresh_meal=args.fresh_meal, previous_meal=args.previous_meal,
                    accept_regression=args.accept_regression)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
