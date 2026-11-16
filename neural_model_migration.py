"""Review-gated structural migration of the active conversation checkpoint."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
from uuid import uuid4

from neural_active_training import APPROVED_SHA256
from neural_base_comparison import digest
from neural_chat import CONVERSATION_MODEL, DIRECTORY as CHAT_DIRECTORY, load_active
from neural_continuation import extend_alphabet
from neural_conversation_context import CONTEXT, SPACING_CONTEXT, SPACING_DIGITS, WORD_POSITION_CONTEXT
from neural_conversation_meal import AUTHORED_FIXTURE, validate_curriculum
from neural_conversation_trial import answer_examples, assess, canonical, feedback_hash, prompt_for
from neural_base_diagnostic import ambiguity
from neural_experiment import EMBEDDING, HIDDEN
from neural_plasticity import PlasticityContext
from neural_training import DIRECTORY, activate, load_progress
from neural_word_context import WORD
from original_generator import atomic_json


WORDS = (
    "people", "understand", "love", "rainbow", "learn", "help", "answer", "ask",
    "explain", "reason", "memory", "remember", "conversation", "follow", "question",
    "friend", "family", "name", "thought", "idea", "fact", "time", "place", "weather",
    "light", "dark", "green", "orange", "yellow", "purple", "black", "white",
)
CHARACTERS = ("Q", "X")
WORD_POSITION_REVIEW = {"encoding": "word-length-base64-v1", "maximum": 4095}


def migrate(parent: PlasticityContext) -> PlasticityContext:
    if parent.model.context != CONTEXT:
        raise ValueError("This migration requires the original 64-slot conversation checkpoint")
    additions = (["identity:" + word for word in WORDS]
                 + ["char:" + char for char in CHARACTERS]
                 + [f"gap:space-{part}:{digit}" for part in ("low", "high")
                    for digit in range(SPACING_DIGITS)])
    if any(symbol in parent.model.tokens for symbol in additions if symbol.startswith("gap:")):
        raise ValueError("The parent already contains spacing-count features")
    expanded = extend_alphabet(parent.model, additions)
    other = parent.model.tokens["identity:<other>"]
    for word in WORDS:
        symbol = "identity:" + word
        if symbol not in parent.model.tokens:
            target = expanded.tokens[symbol]
            expanded.weights["embedding"][target * EMBEDDING:(target + 1) * EMBEDDING] = (
                parent.model.weights["embedding"][other * EMBEDDING:(other + 1) * EMBEDDING]
            )
    for part in ("low", "high"):
        for digit in range(SPACING_DIGITS):
            target = expanded.tokens[f"gap:space-{part}:{digit}"]
            expanded.weights["embedding"][target * EMBEDDING:(target + 1) * EMBEDDING] = (
                [1.0, digit / (SPACING_DIGITS - 1)]
                + [float((digit >> bit) & 1) for bit in range(EMBEDDING - 2)]
            )
    state = parent.snapshot()
    state.update(symbols=expanded.characters, weights=expanded.weights, context=SPACING_CONTEXT,
                 spacing_features="word-gap-space-counts-base64-v1")
    state["weights"]["hidden"] += [0.0] * ((SPACING_CONTEXT - CONTEXT) * EMBEDDING * HIDDEN)
    return PlasticityContext.restore(state)


def migrate_word_position(parent: PlasticityContext) -> PlasticityContext:
    if parent.model.context != SPACING_CONTEXT:
        raise ValueError("This migration requires the 128-slot spacing checkpoint")
    state = json.loads(json.dumps(parent.snapshot(), allow_nan=False))
    state["context"] = WORD_POSITION_CONTEXT
    state["partial_word_features"] = WORD_POSITION_REVIEW["encoding"]
    state["weights"]["hidden"] += [0.0] * ((WORD_POSITION_CONTEXT - SPACING_CONTEXT) * EMBEDDING * HIDDEN)
    return PlasticityContext.restore(state)


def pair_ambiguity(model: PlasticityContext, row: dict) -> dict:
    text = prompt_for(row) + row["answer"] + "\n"
    symbols = {"char:" + char for char in text}
    symbols.update("identity:" + match.group().lower().replace("’", "'") for match in WORD.finditer(text))
    expanded = extend_alphabet(model.model, sorted(symbols))
    state = model.snapshot()
    state.update(symbols=expanded.characters, weights=expanded.weights)
    return ambiguity(answer_examples(PlasticityContext.restore(state), [row]))


def run_word_position(root: Path, *, acknowledge_review: bool, reviewed_sha256: str,
                      target_pair_sha256: str) -> dict:
    if acknowledge_review is not True or reviewed_sha256 != digest(canonical(WORD_POSITION_REVIEW)):
        raise ValueError("Explicit approval of the exact word-position feature hash is required")
    directory = root / DIRECTORY
    with (directory / "training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Training is busy; retry the migration later") from error
        active_path = root / CHAT_DIRECTORY / "active.json"
        progress_path = directory / "progress.json"
        active_bytes, progress_bytes = active_path.read_bytes(), progress_path.read_bytes()
        active, parent, index = load_active(root)
        progress, _ = load_progress(root)
        if (active["model"] != CONVERSATION_MODEL or not isinstance(parent, PlasticityContext)
                or index is not None or active["checkpoint_sha256"] != progress["checkpoint_sha256"]):
            raise ValueError("Active model and permanent progress must match")
        raw = (root / AUTHORED_FIXTURE).read_bytes()
        if (progress.get("fixture") != AUTHORED_FIXTURE
                or progress.get("reviewed_file_sha256") != APPROVED_SHA256
                or digest(raw) != APPROVED_SHA256):
            raise ValueError("The approved authored curriculum must match the parent checkpoint")
        data = json.loads(raw)
        validate_curriculum(data)
        corrections = progress.get("approved_corrections", [])
        for item in corrections:
            if item["sha256"] != feedback_hash(item["row"]["turns"], item["row"]["answer"]):
                raise ValueError("Approved feedback hash mismatch")
        targets = [item["row"] for item in corrections if item["sha256"] == target_pair_sha256]
        if len(targets) != 1:
            raise ValueError("The target must identify exactly one approved pair")
        candidate = migrate_word_position(parent)
        state = json.loads(json.dumps(candidate.snapshot(), allow_nan=False))
        if migrate_word_position(PlasticityContext.restore(
                json.loads(json.dumps(parent.snapshot())))).snapshot() != state:
            raise AssertionError("Word-position migration did not replay exactly")
        before_ambiguity = pair_ambiguity(parent, targets[0])
        after_ambiguity = pair_ambiguity(candidate, targets[0])
        run_id = uuid4().hex
        target = directory / "rounds" / run_id
        report = {
            "run_id": run_id, "round": progress["round"] + 1, "status": "running",
            "kind": "reviewed_word_position_migration",
            "parent_checkpoint_sha256": progress["checkpoint_sha256"],
            "review": {
                "feature": WORD_POSITION_REVIEW, "feature_sha256": reviewed_sha256,
                "target_pair_sha256": target_pair_sha256,
                "train_sha256": active["source"]["training_sha256"],
                "authorization": "User requested fixing the reviewed pair's representation collision.",
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            },
            "representation": {"before": before_ambiguity, "after": after_ambiguity},
            "policy": "No optimizer updates. Append two zero-weight partial-word-length features using "
                      "existing digit embeddings; retain all old weights and token IDs. Lengths 1–4095 "
                      "are distinguished; longer partial words fail explicitly. This resolves the "
                      "reviewed pair's measured collision, not every possible representation collision.",
        }
        atomic_json(target / "protocol.json", report)
        atomic_json(target / "parent.json", {"state": parent.snapshot(), "selection": active})
        atomic_json(target / "candidate.json", {"state": state})
        for name in ("neural_model_migration.py", "neural_training.py", "neural_chat.py",
                     "neural_plasticity.py", "neural_experiment.py", "neural_conversation_trial.py",
                     "neural_conversation_context.py", "neural_base_comparison.py",
                     "neural_continuation.py", "neural_word_context.py"):
            path = target / "source" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((Path(__file__).parent / name).read_bytes())
        results, regressions = {}, []
        for label, rows in (("teaching", data["train"]), ("development", data["test"]),
                            ("feedback", [item["row"] for item in corrections])):
            before, after = assess(parent, rows), assess(candidate, rows)
            lost = [old["id"] for old, new in zip(before["rows"], after["rows"])
                    if old["exact"] and not new["exact"]]
            changed = sum(old["prediction"] != new["prediction"]
                          or old["terminated"] != new["terminated"]
                          for old, new in zip(before["rows"], after["rows"]))
            results[label] = {"before": before, "after": after, "changed_predictions": changed,
                              "lost_correct_ids": lost}
            regressions.extend(f"{label}:{identifier}" for identifier in lost)
            print(f"{label}: {sum(r['exact'] for r in before['rows'])}/{len(rows)} -> "
                  f"{sum(r['exact'] for r in after['rows'])}/{len(rows)}; "
                  f"lost={len(lost)}, changed={changed}", flush=True)
        report["results"] = results
        report["validation"] = {
            "structural_migration_replay_exact": True,
            "existing_correct_answers_preserved": not regressions,
            "target_pair_collision_free": after_ambiguity["conflicting_inputs"] == 0,
        }
        failures = regressions + (["target pair still has conflicting inputs"]
                                  if after_ambiguity["conflicting_inputs"] else [])
        if failures:
            report.update(status="rejected", rejection=failures)
            atomic_json(target / "report.json", report)
            raise RuntimeError("Word-position migration rejected: " + "; ".join(failures))
        if active_path.read_bytes() != active_bytes or progress_path.read_bytes() != progress_bytes:
            raise RuntimeError("Active selection or progress changed during migration; candidate not activated")
        report["status"] = "completed"
        atomic_json(target / "report.json", report)
        evidence = {"run_id": run_id, "round": report["round"],
                    "parent_checkpoint_sha256": progress["checkpoint_sha256"],
                    "reviewed_file_sha256": progress["reviewed_file_sha256"],
                    "fixture": AUTHORED_FIXTURE, "approved_corrections": corrections,
                    "report_sha256": digest((target / "report.json").read_bytes())}
        encoded = canonical({"state": state, "evidence": evidence})
        checksum = digest(encoded)
        (directory / f"{checksum}.json").write_bytes(encoded)
        atomic_json(progress_path, {**evidence, "checkpoint_sha256": checksum})
        try:
            activate(root)
        except (ValueError, KeyError, TypeError, ArithmeticError, OSError):
            atomic_json(progress_path, progress)
            if active_path.read_bytes() != active_bytes:
                current = json.loads(active_path.read_text())
                if current.get("checkpoint_sha256") == checksum:
                    atomic_json(active_path, active)
            raise
        return report


def run(root: Path, *, acknowledge_review: bool, reviewed_sha256: str) -> dict:
    approval = {"words": list(WORDS), "characters": list(CHARACTERS)}
    if acknowledge_review is not True or reviewed_sha256 != digest(canonical(approval)):
        raise ValueError("Explicit approval of the exact vocabulary hash is required")
    directory = root / DIRECTORY
    with (directory / "training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Training is busy; retry the migration later") from error
        active_path = root / CHAT_DIRECTORY / "active.json"
        progress_path = directory / "progress.json"
        active_bytes, progress_bytes = active_path.read_bytes(), progress_path.read_bytes()
        active, parent, index = load_active(root)
        progress, _ = load_progress(root)
        if (active["model"] != CONVERSATION_MODEL or not isinstance(parent, PlasticityContext)
                or index is not None or active["checkpoint_sha256"] != progress["checkpoint_sha256"]):
            raise ValueError("Active model and permanent progress must be the same conversation checkpoint")
        raw = (root / AUTHORED_FIXTURE).read_bytes()
        if (progress.get("fixture") != AUTHORED_FIXTURE
                or progress.get("reviewed_file_sha256") != APPROVED_SHA256
                or digest(raw) != APPROVED_SHA256):
            raise ValueError("The approved authored curriculum must match the parent checkpoint")
        data = json.loads(raw)
        validate_curriculum(data)
        corrections = progress.get("approved_corrections", [])
        for item in corrections:
            if item["sha256"] != feedback_hash(item["row"]["turns"], item["row"]["answer"]):
                raise ValueError("Approved feedback hash mismatch")
        feedback = [item["row"] for item in corrections]
        candidate = migrate(parent)
        state = json.loads(json.dumps(candidate.snapshot(), allow_nan=False))
        if migrate(PlasticityContext.restore(json.loads(json.dumps(parent.snapshot())))).snapshot() != state:
            raise AssertionError("Structural migration did not replay exactly")
        if PlasticityContext.restore(state).snapshot() != state:
            raise AssertionError("Migrated checkpoint did not restore exactly")
        run_id = uuid4().hex
        target = directory / "rounds" / run_id
        review = {
            "vocabulary": approval, "vocabulary_sha256": reviewed_sha256,
            "authorization": "User approved these exact word identities and Q/X and requested live activation.",
            "train_sha256": active["source"]["training_sha256"],
            "scope": "Vocabulary and representation only; no new training examples or optimizer updates.",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        report = {"run_id": run_id, "round": progress["round"] + 1, "status": "running",
                  "kind": "reviewed_structural_migration",
                  "parent_checkpoint_sha256": progress["checkpoint_sha256"], "review": review,
                  "policy": "No optimizer updates. Old weights and token IDs are retained; new word "
                            "embeddings copy the unknown-word embedding, distinct bounded spacing-count "
                            "embeddings are attached through zero-initialized weights, "
                            "and Q/X output rows copy the unknown-token row. Space counts cover the "
                            "32 represented word gaps up to 4095 spaces; larger gaps fail explicitly. "
                            "Mixed punctuation/whitespace "
                            "layouts with equal space counts can still collide. The new features do not "
                            "teach meanings or spacing without approved training. Development cases are "
                            "reused and are not an independent generalization test."}
        atomic_json(target / "protocol.json", report)
        atomic_json(target / "parent.json", {"state": parent.snapshot(), "selection": active})
        atomic_json(target / "candidate.json", {"state": state})
        for name in ("neural_model_migration.py", "neural_training.py", "neural_chat.py",
                     "neural_plasticity.py", "neural_experiment.py", "neural_conversation_trial.py",
                     "neural_conversation_context.py", "neural_base_comparison.py",
                     "neural_continuation.py", "neural_word_context.py"):
            path = target / "source" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((Path(__file__).parent / name).read_bytes())
        results, regressions = {}, []
        for label, rows in (("teaching", data["train"]), ("development", data["test"]),
                            ("feedback", feedback)):
            if not rows:
                continue
            before, after = assess(parent, rows), assess(candidate, rows)
            lost = [old["id"] for old, new in zip(before["rows"], after["rows"])
                    if old["exact"] and not new["exact"]]
            changed = sum(old["prediction"] != new["prediction"]
                          or old["terminated"] != new["terminated"]
                          for old, new in zip(before["rows"], after["rows"]))
            results[label] = {"before": before, "after": after, "changed_predictions": changed,
                              "lost_correct_ids": lost}
            regressions.extend(f"{label}:{identifier}" for identifier in lost)
            print(f"{label}: {sum(r['exact'] for r in before['rows'])}/{len(rows)} -> "
                  f"{sum(r['exact'] for r in after['rows'])}/{len(rows)}; "
                  f"lost={len(lost)}, changed={changed}", flush=True)
        report["results"] = results
        report["validation"] = {"structural_migration_replay_exact": True,
                                "existing_correct_answers_preserved": not regressions}
        if regressions:
            report.update(status="rejected", rejection=regressions)
            atomic_json(target / "report.json", report)
            raise RuntimeError("Migration rejected: previously correct answers regressed; active weights unchanged")
        if active_path.read_bytes() != active_bytes or progress_path.read_bytes() != progress_bytes:
            raise RuntimeError("Active selection or progress changed during migration; candidate not activated")
        report["status"] = "completed"
        atomic_json(target / "report.json", report)
        evidence = {"run_id": run_id, "round": report["round"],
                    "parent_checkpoint_sha256": progress["checkpoint_sha256"],
                    "reviewed_file_sha256": progress["reviewed_file_sha256"],
                    "fixture": AUTHORED_FIXTURE, "approved_corrections": corrections,
                    "report_sha256": digest((target / "report.json").read_bytes())}
        encoded = canonical({"state": state, "evidence": evidence})
        checksum = digest(encoded)
        (directory / f"{checksum}.json").write_bytes(encoded)
        atomic_json(progress_path, {**evidence, "checkpoint_sha256": checksum})
        try:
            activate(root)
        except (ValueError, KeyError, TypeError, ArithmeticError, OSError):
            atomic_json(progress_path, progress)
            if active_path.read_bytes() != active_bytes:
                current = json.loads(active_path.read_text())
                if current.get("checkpoint_sha256") == checksum:
                    atomic_json(active_path, active)
            raise
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acknowledge-review", action="store_true")
    parser.add_argument("--reviewed-sha256", required=True)
    parser.add_argument("--word-position-target-sha256")
    args = parser.parse_args()
    if args.word_position_target_sha256:
        result = run_word_position(Path(__file__).resolve().parent, acknowledge_review=args.acknowledge_review,
                                   reviewed_sha256=args.reviewed_sha256,
                                   target_pair_sha256=args.word_position_target_sha256)
    else:
        result = run(Path(__file__).resolve().parent, acknowledge_review=args.acknowledge_review,
                     reviewed_sha256=args.reviewed_sha256)
    print(f"Activated reviewed structural migration round {result['round']}", flush=True)
