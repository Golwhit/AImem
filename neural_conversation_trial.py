"""Answer-only continuation of the pinned New Base on reviewed teaching exchanges."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import statistics
import time
from uuid import uuid4

from neural_base_comparison import WordBase, digest, generate
from neural_base_diagnostic import ambiguity
from neural_chat import BASE_MODEL, load_active
from neural_continuation import extend_alphabet
from original_generator import atomic_json


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()


def prompt_for(row: dict) -> str:
    if not row["turns"]:
        raise ValueError("Conversation needs at least one turn")
    pieces = []
    for turn in row["turns"]:
        if turn["role"] not in ("user", "assistant") or not turn["content"].strip():
            raise ValueError("Invalid conversation turn")
        pieces.append(f"{turn['role']}: {turn['content']}\n")
    return "".join(pieces) + "assistant: "


def answer_examples(model: WordBase, rows: list[dict]) -> list:
    result = []
    for row in rows:
        prefix = prompt_for(row)
        if not row["answer"].strip() or "\n" in row["answer"]:
            raise ValueError("Teaching answers must be nonempty single lines")
        for token in model.encode(row["answer"] + "\n"):
            prior = model.prior(prefix)
            if not prior[token]:
                raise ValueError("Answer target is excluded by causal output mask")
            result.append((model.context(prefix), token, prior))
            prefix += model.payloads[token]
    return result


def prepare(model: WordBase, rows: list[dict]) -> tuple[WordBase, list[str]]:
    teaching_text = "".join(prompt_for(row) + row["answer"] + "\n" for row in rows)
    added = sorted({"char:" + char for char in teaching_text} - set(model.model.characters))
    expanded = extend_alphabet(model.model, added)
    state = {"lexicon": model.lexicon, "symbols": expanded.characters, "weights": expanded.weights}
    return WordBase.restore(state), added


def assess(model: WordBase, rows: list[dict], *, max_chars: int = 96) -> dict:
    measured = []
    for row in rows:
        continuation = generate(model, prompt_for(row), length=max_chars)["text"]
        terminated = "\n" in continuation
        answer = continuation.split("\n", 1)[0]
        measured.append({
            "id": row["id"], "skill": row["skill"], "pair": row.get("pair"),
            "expected": row["answer"], "prediction": answer, "terminated": terminated,
            "exact": terminated and answer == row["answer"],
            "normalized_exact": terminated and answer.strip().casefold() == row["answer"].strip().casefold(),
            "continuation": continuation,
        })
    if not measured:
        raise ValueError("Assessment requires examples")
    return {
        "examples": len(measured),
        "exact": statistics.fmean(row["exact"] for row in measured),
        "normalized_exact": statistics.fmean(row["normalized_exact"] for row in measured),
        "termination": statistics.fmean(row["terminated"] for row in measured),
        "rows": measured,
    }


def token_loss(model: WordBase, examples: list) -> float:
    losses = []
    for context, target, prior in examples:
        probabilities = model.model.forward(context, output_prior=prior)[2]
        if not all(math.isfinite(value) for value in probabilities):
            raise ArithmeticError("Nonfinite conversation probabilities")
        losses.append(-math.log(max(probabilities[target], 1e-300)))
    return statistics.fmean(losses)


def train(model: WordBase, rows: list[dict], *, rate: float, seed: int,
          checkpoints=(500, 2000, 5000)) -> list[dict]:
    if (not math.isfinite(rate) or rate <= 0 or not checkpoints
            or any(type(n) is not int or n <= 0 for n in checkpoints)
            or list(checkpoints) != sorted(set(checkpoints))):
        raise ValueError("Invalid training protocol")
    examples = answer_examples(model, rows)
    if not examples:
        raise ValueError("Training requires answer tokens")
    rng = random.Random(seed)
    updates, cpu, points = 0, 0.0, []
    for endpoint in checkpoints:
        start = time.process_time()
        while updates < endpoint:
            context, target, prior = examples[rng.randrange(len(examples))]
            model.model.update(context, target, output_prior=prior, learning_rate=rate)
            updates += 1
        cpu += time.process_time() - start
        points.append({"updates": updates, "training_cpu_seconds": cpu,
                       "teacher_forced_loss": token_loss(model, examples),
                       "training": assess(model, rows)})
    return points


def run(root: Path, *, acknowledge_review: bool, reviewed_sha256: str) -> dict:
    path = root / "fixtures/neural-conversation-draft.json"
    raw = path.read_bytes()
    checksum = digest(raw)
    if acknowledge_review is not True or checksum != reviewed_sha256:
        raise ValueError("Explicit approval and the exact reviewed curriculum file hash are required")
    data = json.loads(raw)
    ids = [row["id"] for split in ("train", "test") for row in data[split]]
    if len(ids) != len(set(ids)) or not data["train"] or not data["test"]:
        raise ValueError("Teaching and test IDs must be unique and both splits nonempty")
    if {canonical(row["turns"]) for row in data["train"]} & {
        canonical(row["turns"]) for row in data["test"]
    }:
        raise ValueError("Teaching and test prompts overlap")
    record, parent, index = load_active(root)
    if record["model"] != BASE_MODEL or not isinstance(parent, WordBase) or index is not None:
        raise ValueError("Trial requires the pinned layer-free New Base")
    protected = {str(p.relative_to(root)): digest(p.read_bytes())
                 for p in (root / ".data/neural-chat").glob("*.json")}
    original_parent = json.loads(json.dumps(parent.snapshot()))
    parent, added = prepare(parent, data["train"])
    examples = answer_examples(parent, data["train"])
    run_id = uuid4().hex
    directory = root / ".data/neural-conversation-trial" / run_id
    review = {
        "reviewed_file_sha256": checksum, "train_sha256": digest(canonical(data["train"])),
        "test_sha256": digest(canonical(data["test"])),
        "approval": "User input: \"let's train and see what we get\"",
        "scope": "Eight proposed teaching exchanges only; ten separate test answers are evaluation only.",
        "review_record_type": "User declaration, not legal clearance",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "provenance": data["provenance"],
    }
    report = {
        "run_id": run_id, "status": "running", "review": review, "parent": record,
        "rates": [.1, .03, .01], "shuffle_seed": 17, "updates": [500, 2000, 5000],
        "added_training_character_symbols": added,
        "parameters": sum(map(len, parent.model.weights.values())),
        "answer_tokens": len(examples), "representation": ambiguity(examples),
        "policy": "Continue identical copies of the ACTUAL active seed17 four-pass New Base. "
                  "No fresh word identities or architecture, layers or parent resets. Missing character "
                  "symbols from teaching prompts/answers/role formatting only are appended using existing "
                  "extend_alphabet: preserve old IDs/parameters, copy unknown embeddings/output rows for "
                  "new characters. This changes softmax size and is reported; test data cannot add symbols. "
                  "Three rates and one shared random target schedule, not three independent seeds. "
                  "Train only assistant-answer tokens plus newline terminator; supplied user and prior "
                  "assistant turns are causal context, not optimizer targets. Role labels are ordinary "
                  "text features, NOT added role tokens. Word vocabulary is frozen from parent Acts "
                  "training; absent input characters map to unknown. Encode targets strictly: absent "
                  "target characters raise errors. No test inputs/answers build vocabulary or updates. "
                  "Prompt format is role: content, newline, ending assistant: space. "
                  "Greedy96-character continuation; first newline terminates answer. No stop inferred "
                  "from expected answer length; nonterminated responses fail exact scoring. "
                  "Evaluate all training checkpoints; test split only before training and after fixed "
                  "final endpoints, no test-guided tuning or best-rate activation. Exact single-reference "
                  "and whitespace/case-normalized scores are lexical, not semantic judgments; inspect "
                  "raw replies and paired facts. Developer-visible proposed tests are not locked unseen "
                  "generalization evidence. No corpus replay; this is a narrow adaptation feasibility "
                  "trial, not a retention study. No lookup replies, memory disconnection or activation.",
        "protected_before": protected,
        "code_sha256": {name: digest((root / name).read_bytes()) for name in (
            "neural_conversation_trial.py", "neural_base_comparison.py", "neural_experiment.py",
            "neural_continuation.py")},
        "before": {"training": assess(parent, data["train"]), "test": assess(parent, data["test"]),
                   "teacher_forced_loss": token_loss(parent, examples)},
        "trials": [],
    }
    atomic_json(directory / "review.json", review)
    atomic_json(directory / "inputs.json", data)
    atomic_json(directory / "parent.json", {"state": original_parent})
    atomic_json(directory / "prepared-parent.json", {"state": parent.snapshot()})
    atomic_json(directory / "protocol.json", report)
    print(f"Conversation trial {run_id}; {len(examples)} answer-token examples", flush=True)
    for rate in report["rates"]:
        model = WordBase.restore(json.loads(json.dumps(parent.snapshot())))
        points = train(model, data["train"], rate=rate, seed=report["shuffle_seed"])
        final_test = assess(model, data["test"])
        state = json.loads(json.dumps(model.snapshot(), allow_nan=False))
        restored = WordBase.restore(state)
        if assess(restored, data["test"]) != final_test:
            raise AssertionError("Restored conversation predictions differ")
        trial = {"rate": rate, "checkpoints": points, "test": final_test,
                 "restoration_verified": True}
        atomic_json(directory / f"rate-{rate}.json", {"state": state, "evidence": trial})
        report["trials"].append(trial)
        atomic_json(directory / "report.json", report)
        print(f"rate={rate} train exact={points[-1]['training']['exact']:.3f}; "
              f"test exact={final_test['exact']:.3f}; loss={points[-1]['teacher_forced_loss']:.3f}",
              flush=True)
    report["protected_after"] = {str(p.relative_to(root)): digest(p.read_bytes())
                                for p in (root / ".data/neural-chat").glob("*.json")}
    if protected != report["protected_after"]:
        raise RuntimeError("Pinned working/control files changed during trial")
    report["status"] = "completed"
    atomic_json(directory / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acknowledge-review", action="store_true")
    parser.add_argument("--reviewed-sha256", required=True)
    args = parser.parse_args()
    run(Path(__file__).resolve().parent, acknowledge_review=args.acknowledge_review,
        reviewed_sha256=args.reviewed_sha256)
