"""Controlled, offline word-identity and longer-context conversation ablation."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from uuid import uuid4

from neural_base_comparison import WordBase, digest
from neural_base_diagnostic import ambiguity
from neural_chat import BASE_MODEL, load_active
from neural_continuation import extend_alphabet
from neural_conversation_trial import (
    answer_examples, assess, canonical, prepare, prompt_for, token_loss, train,
)
from neural_experiment import EMBEDDING, HIDDEN, NeuralModel
from neural_word_context import WORD, gap_symbol
from original_generator import atomic_json


ARMS = ("short_original", "short_identities", "long_identities")
CONTEXT = 64


class ConversationContext(WordBase):
    """Thirty older word/gap pairs followed by the unchanged four local features."""

    def context(self, prefix: str) -> list[int]:
        if self.model.context == 4:
            return super().context(prefix)
        if self.model.context != CONTEXT:
            raise ValueError("Conversation context must contain 4 or 64 feature slots")
        words = list(WORD.finditer(prefix))
        partial = words[-1] if words and words[-1].end() == len(prefix) else None
        complete = words[:-1] if partial else words
        older = complete[:-1] if partial else complete[:-2]
        features = []
        for position, word in enumerate(older):
            following = words[position + 1].start()
            identity = word.group().lower().replace("’", "'")
            features.extend([
                self.model.tokens.get("identity:" + identity, self.model.tokens["identity:<other>"]),
                self.model.tokens.get(gap_symbol(prefix[word.end():following]), 1),
            ])
        local = super().context(prefix)
        return ([0] * (CONTEXT - 4) + features)[-(CONTEXT - 4):] + local

    def snapshot(self) -> dict:
        return {**super().snapshot(), "context": self.model.context,
                "representation": "older-word-gap-pairs-plus-original-local-four-v1"}

    @classmethod
    def restore(cls, state: dict) -> "ConversationContext":
        if state["context"] not in (4, CONTEXT):
            raise ValueError("Invalid saved conversation context")
        model = cls("", 17, lexicon=state["lexicon"], symbols=state["symbols"])
        model.model = NeuralModel(state["symbols"], context=state["context"])
        if set(state["weights"]) != set(model.model.weights) or any(
            not isinstance(state["weights"][name], list)
            or len(state["weights"][name]) != len(values)
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   for value in state["weights"][name])
            for name, values in model.model.weights.items()
        ):
            raise ValueError("Conversation weights do not match context shape")
        model.model.weights = state["weights"]
        return model


def build(parent: WordBase, teaching: list[dict], arm: str) -> tuple[ConversationContext, dict]:
    if arm not in ARMS:
        raise ValueError("Unknown conversation ablation arm")
    copied = WordBase.restore(json.loads(json.dumps(parent.snapshot())))
    prepared, characters = prepare(copied, teaching)
    identities = []
    if arm != "short_original":
        text = "".join(prompt_for(row) + row["answer"] + "\n" for row in teaching)
        identities = sorted({
            "identity:" + match.group().lower().replace("’", "'") for match in WORD.finditer(text)
        } - set(prepared.model.characters))
        prepared.model = extend_alphabet(prepared.model, identities)
    context = CONTEXT if arm == "long_identities" else 4
    old = prepared.model
    network = NeuralModel(old.characters, context=context)
    network.weights = {name: values[:] for name, values in old.weights.items()}
    network.weights["hidden"] = [0.0] * ((context - 4) * EMBEDDING * HIDDEN) + old.weights["hidden"][:]
    state = {"lexicon": prepared.lexicon, "symbols": old.characters, "context": context,
             "weights": network.weights}
    return ConversationContext.restore(state), {
        "added_training_characters": characters, "added_training_identities": identities,
        "context_slots": context, "parameters": sum(map(len, network.weights.values())),
    }


def run(root: Path, *, acknowledge_review: bool, reviewed_sha256: str) -> dict:
    raw = (root / "fixtures/neural-conversation-draft.json").read_bytes()
    if acknowledge_review is not True or digest(raw) != reviewed_sha256:
        raise ValueError("Explicit approval of the exact curriculum hash is required")
    data = json.loads(raw)
    identifiers = [row["id"] for split in ("train", "test") for row in data[split]]
    if (not data["train"] or not data["test"] or len(set(identifiers)) != len(identifiers)
            or {canonical(row["turns"]) for row in data["train"]} &
            {canonical(row["turns"]) for row in data["test"]}):
        raise ValueError("Conversation splits must be nonempty and disjoint")
    record, parent, index = load_active(root)
    if record["model"] != BASE_MODEL or not isinstance(parent, WordBase) or index is not None:
        raise ValueError("Experiment requires the layer-free working base")
    protected = {str(path.relative_to(root)): digest(path.read_bytes())
                 for path in (root / ".data/neural-chat").glob("*.json")}
    run_id = uuid4().hex
    directory = root / ".data/neural-conversation-context" / run_id
    report = {
        "run_id": run_id, "status": "running", "parent": record,
        "review": {"file_sha256": digest(raw), "train_sha256": digest(canonical(data["train"])),
                   "test_sha256": digest(canonical(data["test"])),
                   "approval": "User input: \"go for it\"",
                   "scope": "Longer-context trial on previously approved eight teaching exchanges; "
                            "ten test answers remain evaluation only. User declaration, not legal clearance.",
                   "recorded_at": datetime.now(timezone.utc).isoformat()},
        "rate": .1, "shuffle_seed": 17, "updates": [500, 2000, 5000],
        "arms": list(ARMS), "protected_before": protected,
        "policy": "Same actual active parent copied per arm. short_original reproduces prior .1 "
                  "conversation trial. short_identities appends training-only normalized word identity "
                  "symbols using unknown-row copies; word output vocabulary and spelling fallback "
                  "unchanged. long_identities uses exactly the same vocabulary and output masks, "
                  "with 60 older word/gap slots plus the original four local features (64 total). "
                  "Older hidden connections initialize to zero, existing connections/weights preserved. "
                  "Long context distinguishes earlier facts structurally, but does not guarantee learned "
                  "use or generalization. Normalized case/apostrophe identities, no hash embeddings, "
                  "test word identities or new word-output tokens. Unknown test names still share OTHER. "
                  "Role labels remain text but teaching user/assistant identities are distinct. "
                  "One single fixed context, not parallel branches, RNN, transformer or retired layers. "
                  "Same 97 answer targets/draws/.1 rate/checkpoints and greedy96 newline-terminated "
                  "scoring as prior trial; only training metrics at checkpoints, test before/final. "
                  "One parent/draw seed, no independent-seed uncertainty claim. Same updates NOT same "
                  "compute/capacity; report preparation/training/evaluation CPU and parameters. "
                  "No post-test tuning, best-score deployment, replay or retention claims. "
                  "Developer-visible tiny tests are not independent generalization evidence. "
                  "Memory capture remains connected; active chat and original comparison untouched.",
        "code_sha256": {name: digest((root / name).read_bytes()) for name in (
            "neural_conversation_context.py", "neural_conversation_trial.py",
            "neural_base_comparison.py", "neural_experiment.py", "neural_continuation.py")},
        "trials": [],
    }
    atomic_json(directory / "inputs.json", data)
    atomic_json(directory / "parent.json", {"state": parent.snapshot()})
    atomic_json(directory / "review.json", report["review"])
    atomic_json(directory / "protocol.json", report)
    print(f"Context trial {run_id}", flush=True)
    for arm in ARMS:
        start = time.process_time()
        model, metadata = build(parent, data["train"], arm)
        examples = answer_examples(model, data["train"])
        setup = time.process_time() - start
        atomic_json(directory / f"{arm}-initial.json", {"state": model.snapshot()})
        before = {"training": assess(model, data["train"]), "test": assess(model, data["test"]),
                  "teacher_forced_loss": token_loss(model, examples)}
        points = train(model, data["train"], rate=report["rate"], seed=report["shuffle_seed"],
                       checkpoints=tuple(report["updates"]))
        start = time.process_time()
        final_test = assess(model, data["test"])
        evaluation_cpu = time.process_time() - start
        state = json.loads(json.dumps(model.snapshot(), allow_nan=False))
        restored = ConversationContext.restore(state)
        if assess(restored, data["test"]) != final_test:
            raise AssertionError("Context checkpoint restoration changed test outputs")
        trial = {"arm": arm, **metadata, "setup_cpu_seconds": setup, "before": before,
                 "answer_targets": len(examples), "representation": ambiguity(examples),
                 "checkpoints": points, "test": final_test,
                 "final_test_cpu_seconds": evaluation_cpu, "restoration_verified": True}
        atomic_json(directory / f"{arm}-final.json", {"state": state, "evidence": trial})
        report["trials"].append(trial)
        atomic_json(directory / "report.json", report)
        print(f"{arm}: parameters={metadata['parameters']}; train "
              f"{points[-1]['training']['exact']:.3f}; test {final_test['exact']:.3f}; "
              f"loss={points[-1]['teacher_forced_loss']:.3f}; "
              f"CPU={points[-1]['training_cpu_seconds']:.2f}s", flush=True)
    report["protected_after"] = {str(path.relative_to(root)): digest(path.read_bytes())
                                 for path in (root / ".data/neural-chat").glob("*.json")}
    if report["protected_after"] != protected:
        raise RuntimeError("Pinned working/control files changed during context trial")
    report["status"] = "completed"
    atomic_json(directory / "report.json", report)
    return report


def lower_rate_screen(root: Path, run_id: str, *, acknowledge_review: bool,
                      reviewed_sha256: str) -> dict:
    if len(run_id) != 32 or any(char not in "0123456789abcdef" for char in run_id):
        raise ValueError("Invalid context run ID")
    directory = root / ".data/neural-conversation-context" / run_id
    raw = (root / "fixtures/neural-conversation-draft.json").read_bytes()
    report_bytes = (directory / "report.json").read_bytes()
    report = json.loads(report_bytes)
    if (acknowledge_review is not True or digest(raw) != reviewed_sha256
            or report["review"]["file_sha256"] != reviewed_sha256 or report["status"] != "completed"):
        raise ValueError("Completed context run and unchanged approved curriculum are required")
    data = json.loads((directory / "inputs.json").read_text())
    if canonical(data) != canonical(json.loads(raw)):
        raise ValueError("Saved and approved curriculum differ")
    protected = {str(path.relative_to(root)): digest(path.read_bytes())
                 for path in (root / ".data/neural-chat").glob("*.json")}
    initial = json.loads((directory / "long_identities-initial.json").read_text())["state"]
    result = {
        "parent_run_id": run_id, "parent_report_sha256": digest(report_bytes),
        "reviewed_file_sha256": reviewed_sha256, "rates": [.03, .01],
        "policy": "Follow-up prompted by .1 optimization failure on training and inspected saturated "
                  "hidden states; same long initial checkpoint and 5000-update schedule, lower rates. "
                  "Choose lowest final TEACHING token loss only. No test access during training/selection; "
                  "evaluate selected model once after selection. Previously inspected developer-visible "
                  "tests are not fresh locked evidence. Initial .1 failure remains unchanged.",
        "code_sha256": digest(Path(__file__).read_bytes()),
        "protected_before": protected, "trials": [],
    }
    if (directory / "lower-rate-screen.json").exists():
        raise ValueError("Lower-rate screen already exists; preserve completed evidence")
    atomic_json(directory / "lower-rate-protocol.json", result)
    for rate in result["rates"]:
        model = ConversationContext.restore(json.loads(json.dumps(initial)))
        points = train(model, data["train"], rate=rate, seed=report["shuffle_seed"],
                       checkpoints=tuple(report["updates"]))
        state = json.loads(json.dumps(model.snapshot(), allow_nan=False))
        atomic_json(directory / f"long-rate-{rate}.json", {"state": state, "checkpoints": points})
        result["trials"].append({"rate": rate, "checkpoints": points})
        print(f"long lower-rate={rate}: teaching loss={points[-1]['teacher_forced_loss']:.4f}; "
              f"exact={points[-1]['training']['exact']:.3f}", flush=True)
    chosen = min(result["trials"], key=lambda trial: trial["checkpoints"][-1]["teacher_forced_loss"])
    result["selected_rate_by_training_loss"] = chosen["rate"]
    atomic_json(directory / "lower-rate-selection.json", result)
    checkpoint = json.loads((directory / f"long-rate-{chosen['rate']}.json").read_text())
    model = ConversationContext.restore(checkpoint["state"])
    result["selected_test"] = assess(model, data["test"])
    result["protected_after"] = {str(path.relative_to(root)): digest(path.read_bytes())
                                 for path in (root / ".data/neural-chat").glob("*.json")}
    if result["protected_after"] != protected:
        raise RuntimeError("Pinned chat/control files changed during lower-rate screen")
    atomic_json(directory / "lower-rate-screen.json", result)
    print(f"selected={chosen['rate']}; test exact={result['selected_test']['exact']:.3f}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acknowledge-review", action="store_true")
    parser.add_argument("--reviewed-sha256", required=True)
    parser.add_argument("--lower-rate-screen", help="Completed context run to screen lower rates using teaching loss")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.lower_rate_screen:
        lower_rate_screen(root, args.lower_rate_screen, acknowledge_review=args.acknowledge_review,
                          reviewed_sha256=args.reviewed_sha256)
    else:
        run(root, acknowledge_review=args.acknowledge_review, reviewed_sha256=args.reviewed_sha256)
