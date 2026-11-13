"""Pinned experimental neural checkpoints for app chat; no online training."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

from neural_candidate_layers import CandidateIndex, AtomicCandidateIndex
from neural_experiment import NeuralModel
from neural_base_comparison import WordBase, generate as generate_base
from original_generator import atomic_json


DIRECTORY = ".data/neural-chat"
MODEL = "original-layered-character-mlp-chat-v1"
BASE_MODEL = "original-word-punctuation-mlp-chat-v1"
CONVERSATION_MODEL = "original-conversation-mlp-chat-v1"


def validate_base_vocabulary(state: dict) -> None:
    symbols, lexicon = state["symbols"], state["lexicon"]
    if (not isinstance(symbols, list) or not symbols
            or any(not isinstance(symbol, str) or not symbol for symbol in symbols)
            or len(set(symbols)) != len(symbols)
            or not isinstance(lexicon, list) or not lexicon
            or any(not isinstance(word, str) or not word for word in lexicon)
            or len(set(lexicon)) != len(lexicon)
            or "identity:<other>" not in symbols
            or any("identity:" + word not in symbols for word in lexicon)
            or not any(symbol.startswith("char:") and len(symbol) == 6 for symbol in symbols)
            or any(not symbol.startswith(("char:", "emit:", "identity:", "gap:")) for symbol in symbols)
            or any(symbol.startswith("char:") and len(symbol) != 6 for symbol in symbols)
            or any(symbol.startswith("emit:") and len(symbol) <= 5 for symbol in symbols)):
        raise ValueError("Working base checkpoint has invalid vocabulary")


def restore_base(checkpoint: dict) -> WordBase:
    if checkpoint.get("index") is not None:
        raise ValueError("Working base checkpoint must not contain layer placement state")
    state = checkpoint["state"]
    validate_base_vocabulary(state)
    symbols, lexicon = state["symbols"], state["lexicon"]
    model = WordBase("", 17, lexicon=lexicon, symbols=symbols)
    weights = state["weights"]
    if set(weights) != set(model.model.weights) or any(
        not isinstance(weights[name], list) or len(weights[name]) != len(expected)
        or any(type(value) not in (int, float) or not math.isfinite(value) for value in weights[name])
        for name, expected in model.model.weights.items()
    ):
        raise ValueError("Working base checkpoint has invalid neural weights")
    model.model.weights = weights
    return model


def restore_original(checkpoint: dict) -> NeuralModel:
    if checkpoint.get("index") is not None:
        raise ValueError("Original comparison checkpoint must be layer-free")
    state = checkpoint["state"]
    characters = state["characters"]
    if (not isinstance(characters, list) or not characters
            or any(not isinstance(char, str) or len(char) != 1 for char in characters)
            or len(set(characters)) != len(characters)):
        raise ValueError("Original comparison checkpoint has invalid alphabet")
    model = NeuralModel(characters)
    weights = state["weights"]
    if set(weights) != set(model.weights) or any(
        not isinstance(weights[name], list) or len(weights[name]) != len(expected)
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in weights[name])
        for name, expected in model.weights.items()
    ):
        raise ValueError("Original comparison checkpoint has invalid weights")
    model.weights = weights
    return model


def activate_base(root: Path, run_id: str, seed: int = 17, passes: int = 4) -> dict:
    if (len(run_id) != 32 or any(char not in "0123456789abcdef" for char in run_id)
            or type(seed) is not int or type(passes) is not int or passes not in (1, 2, 4)):
        raise ValueError("Invalid base experiment checkpoint selection")
    source = root / ".data/neural-base-comparison" / run_id
    report = json.loads((source / "report.json").read_text())
    if report["status"] != "completed":
        raise ValueError("Select a completed full-generation base comparison")
    inputs = json.loads((source / "inputs.json").read_text())
    for label in ("training", "evaluation"):
        if hashlib.sha256(inputs[label].encode()).hexdigest() != report[label + "_sha256"]:
            raise ValueError("Base comparison source hash mismatch")
    directory = root / DIRECTORY
    records = {}
    copies = {}
    for arm in ("new", "original"):
        entries = [entry for entry in report["runs"] if entry["arm"] == arm and entry["seed"] == seed]
        if len(entries) != 1:
            raise ValueError("Base checkpoint is not present in the selected report")
        points = [point for point in entries[0]["checkpoints"] if point["passes"] == passes]
        if len(points) != 1:
            raise ValueError("Base pass count is not present in the selected report")
        raw = (source / f"{seed}-{arm}-passes-{passes}.json").read_bytes()
        checkpoint = json.loads(raw)
        if checkpoint["evidence"] != points[0] or checkpoint.get("index") is not None:
            raise ValueError("Base checkpoint evidence or layer state does not match the report")
        if arm == "new":
            restored = restore_base(checkpoint)
            network = restored.model
        else:
            network = restore_original(checkpoint)
            if network.characters != sorted(set(inputs["training"])):
                raise ValueError("Original comparison alphabet differs from training")
        checksum = hashlib.sha256(raw).hexdigest()
        copies[checksum] = raw
        records[arm] = {
            "model": BASE_MODEL if arm == "new" else "original-character-mlp-comparison-v1",
            "run_id": run_id, "arm": arm, "seed": seed, "passes": passes,
            "checkpoint_sha256": checksum, "parameters": sum(map(len, network.weights.values())),
            "context": network.context, "vocabulary": network.size,
            "source": {"training": "Acts 6-12 complete", "evaluation": "Acts 13-14 complete (reused development)",
                       "training_sha256": report["training_sha256"],
                       "evaluation_sha256": report["evaluation_sha256"]},
        }
    record = {
        **records["new"], "label": "New Base (word/punctuation + character fallback; no layers)",
        "retired_layers": [0, 1, 2], "comparison_baseline": records["original"],
        "selection_policy": f"Explicit seed {seed}, {passes}-pass endpoint; defaults to first protocol seed "
                            "and four passes, not best-score selection.",
        "note": "Experimental text continuation, not a conversationally trained assistant or factual answer. "
                "L0-L2 retired from the working path. Original character base preserved as comparison control. "
                "Pinned weights; prompts and ratings are review-only. Training alphabet limits fallback spelling. "
                "Generated text is not scripture or the World English Bible.",
    }
    directory.mkdir(parents=True, exist_ok=True)
    previous = directory / "active.json"
    if previous.exists():
        raw = previous.read_bytes()
        (directory / f"retired-selection-{hashlib.sha256(raw).hexdigest()}.json").write_bytes(raw)
    for checksum, raw in copies.items():
        (directory / f"{checksum}.json").write_bytes(raw)
    atomic_json(directory / "comparison-baseline.json", records["original"])
    atomic_json(directory / "active.json", record)
    return status(root)


def restore(checkpoint: dict) -> tuple[NeuralModel, CandidateIndex]:
    characters = checkpoint["characters"]
    if (not isinstance(characters, list) or not characters
            or any(not isinstance(char, str) or len(char) != 1 for char in characters)
            or len(set(characters)) != len(characters)):
        raise ValueError("Chat checkpoint requires a distinct character alphabet")
    model = NeuralModel(characters)
    weights = checkpoint["weights"]
    if set(weights) != set(model.weights) or any(
        not isinstance(weights[name], list) or len(weights[name]) != len(expected)
        or any(type(value) not in (int, float) or not math.isfinite(value) for value in weights[name])
        for name, expected in model.weights.items()
    ):
        raise ValueError("Chat checkpoint has invalid neural weights")
    model.weights = weights
    index = CandidateIndex.restore(checkpoint["index"])
    if isinstance(index, AtomicCandidateIndex):
        raise ValueError("This chat runtime requires the layered remainder model")
    if not {0, 1}.issubset({entry["layer"] for entry in index.codec.units.values()}):
        raise ValueError("Chat checkpoint must include L0 and L1")
    if not index.counts:
        raise ValueError("Chat checkpoint is missing trained placement counts")
    return model, index


def activate(root: Path, run_id: str, seed: int = 17, passes: int = 4) -> dict:
    if len(run_id) != 32 or any(char not in "0123456789abcdef" for char in run_id):
        raise ValueError("Invalid layered experiment run ID")
    if type(seed) is not int or type(passes) is not int or passes not in (1, 2, 4):
        raise ValueError("Invalid checkpoint seed or pass count")
    source = root / ".data/neural-unit-benchmark" / run_id
    report = json.loads((source / "report.json").read_text())
    if report["status"] != "completed" or not report["settings"].get("matched_reinforcement"):
        raise ValueError("Select a completed matched-reinforcement layered run")
    entries = [entry for entry in report["runs"] if entry["arm"] == "layers_01" and entry["seed"] == seed]
    if len(entries) != 1 or not any(row["passes"] == passes for row in entries[0]["checkpoints"]):
        raise ValueError("Layered checkpoint is not present in the selected report")
    raw = (source / f"{seed}-layers_01-passes-{passes}.json").read_bytes()
    checkpoint = json.loads(raw)
    model, index = restore(checkpoint)
    expected = next(row for row in entries[0]["checkpoints"] if row["passes"] == passes)
    if checkpoint["evidence"] != expected:
        raise ValueError("Layered checkpoint evidence does not match the saved report")
    # Keep the source checkpoint intact; activation is independent of the latest experiment pointer.
    directory = root / DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    checksum = hashlib.sha256(raw).hexdigest()
    target = directory / f"{checksum}.json"
    if not target.exists():
        target.write_bytes(raw)
    record = {
        "model": MODEL, "run_id": run_id, "arm": "layers_01", "seed": seed, "passes": passes,
        "checkpoint_sha256": checksum, "parameters": sum(map(len, model.weights.values())),
        "context": model.context, "vocabulary": model.size,
        "l1_forms": sum(entry["layer"] == 1 for entry in index.codec.units.values()),
        "source": report["source"],
        "note": "Experimental text continuation, not a conversationally trained assistant or factual answer. "
                "Pinned checkpoint; chat prompts/ratings do not update its weights or placement counts. "
                "Generated text is not scripture or the World English Bible.",
    }
    if "continuation" in report:
        record["continuation"] = report["continuation"]
        record["note"] += " Passes refers to the latest continuation, in addition to parent training."
    atomic_json(directory / "active.json", record)
    return status(root)


def load_active(root: Path) -> tuple[dict, NeuralModel | WordBase, CandidateIndex | None]:
    record = json.loads((root / DIRECTORY / "active.json").read_text())
    return load_selection(root, record)


def load_selection(root: Path, record: dict) -> tuple[dict, NeuralModel | WordBase, CandidateIndex | None]:
    checksum = record["checkpoint_sha256"]
    if (record["model"] not in (MODEL, BASE_MODEL, CONVERSATION_MODEL)
            or not isinstance(checksum, str) or len(checksum) != 64
            or any(char not in "0123456789abcdef" for char in checksum)):
        raise ValueError("Invalid active neural chat selection")
    raw = (root / DIRECTORY / f"{checksum}.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != checksum:
        raise ValueError("Active neural chat checkpoint hash mismatch")
    if record["model"] == BASE_MODEL:
        if record["arm"] != "new":
            raise ValueError("Active working base must select the layer-free new arm")
        return record, restore_base(json.loads(raw)), None
    if record["model"] == CONVERSATION_MODEL:
        from neural_plasticity import PlasticityContext

        checkpoint = json.loads(raw)
        if record["arm"] != "conversation" or checkpoint.get("index") is not None:
            raise ValueError("Conversation checkpoint must be layer-free")
        validate_base_vocabulary(checkpoint["state"])
        model = PlasticityContext.restore(checkpoint["state"])
        return record, model, None
    model, index = restore(json.loads(raw))
    return record, model, index


def status(root: Path) -> dict:
    if not (root / DIRECTORY / "active.json").exists():
        return {"ready": False, "model": None}
    record, _, _ = load_active(root)
    return {"ready": True, **record}


def generate(root: Path, payload: dict, *, original: bool = False) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Chat generation request must be an object")
    prompt = payload.get("prompt")
    length = payload.get("max_chars", 320)
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 2000:
        raise ValueError("Chat prompt must contain 1-2000 characters")
    if "\x00" in prompt:
        raise ValueError("Chat prompt contains a reserved reference marker")
    if type(length) is not int or not 1 <= length <= 500:
        raise ValueError("Chat output length must be 1-500 characters")
    if original:
        record = json.loads((root / DIRECTORY / "comparison-baseline.json").read_text())
        checksum = record["checkpoint_sha256"]
        if (record["arm"] != "original" or not isinstance(checksum, str) or len(checksum) != 64
                or any(char not in "0123456789abcdef" for char in checksum)):
            raise ValueError("Invalid original comparison selection")
        raw = (root / DIRECTORY / f"{checksum}.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != checksum:
            raise ValueError("Original comparison checkpoint hash mismatch")
        model, index = restore_original(json.loads(raw)), None
    else:
        record, model, index = load_active(root)
    if not original and record["model"] == CONVERSATION_MODEL:
        from neural_conversation_trial import prompt_for

        formatted = prompt_for({"turns": [{"role": "user", "content": prompt}]})
        result = generate_base(model, formatted, length=min(length, 96))
        continuation = result["text"]
        result = {**result, "text": continuation.split("\n", 1)[0],
                  "terminated": "\n" in continuation}
        if not result["text"].strip():
            raise ArithmeticError("Conversation model generated an empty first answer")
    elif isinstance(model, WordBase):
        result = generate_base(model, prompt, length=length)
    elif original:
        result = generate_base(model, prompt, length=length)
    elif index is not None:
        result = index.sample(model, prompt, length=length)
    else:
        raise ValueError("Layered checkpoint is missing its placement index")
    return {**result, "model": record["model"], "generation": record["passes"],
            "chat_model": record,
            "training_sources": [{"title": record["source"]["training"],
                                  "sha256": record["source"]["training_sha256"]}]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["status", "generate", "generate-original", "activate", "activate-base"])
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--passes", type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    try:
        if args.command in ("activate", "activate-base"):
            if not args.run_id:
                raise ValueError("Activation requires --run-id")
            activation = activate_base if args.command == "activate-base" else activate
            result = activation(root, args.run_id, args.seed, args.passes)
        elif args.command == "status":
            result = status(root)
        else:
            result = generate(root, json.load(sys.stdin), original=args.command == "generate-original")
    except (ValueError, KeyError, TypeError, ArithmeticError, OSError) as error:
        print(f"Neural chat error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
