"""Offline word-identity/punctuation representation screening, not chat training."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import time
from uuid import uuid4

from neural_experiment import NeuralModel
from neural_rounds import load_acts
from neural_patterns import load_chapters, STAGED_DIRECTORY
from original_generator import atomic_json


WORD = re.compile(r"[A-Za-z]+(?:['’\-][A-Za-z]+)*")
ARMS = ("raw4", "words_only", "words_punctuation")


def parse(text: str) -> dict:
    rows = []
    end = 0
    for match in WORD.finditer(text):
        rows.append({"word": match.group(), "identity": match.group().lower().replace("’", "'"),
                     "gap": text[end:match.start()], "start": match.start(), "end": match.end()})
        end = match.end()
    return {"words": rows, "tail": text[end:]}


def gap_symbol(gap: str) -> str:
    punctuation = "".join(char for char in gap if not char.isspace())
    return "gap:" + punctuation + (":newline" if "\n" in gap or "\r" in gap else ":inline")


def vocabulary(training: str) -> list[str]:
    words = parse(training)["words"]
    counts = Counter(row["identity"] for row in words)
    if not counts:
        raise ValueError("Word-context screening requires training words")
    lexicon = sorted(counts, key=lambda word: (-counts[word], word))[:64]
    return (["word:<other>"] + ["word:" + word for word in lexicon]
            + sorted({"char:" + char for char in training})
            + sorted({"gap:<ignored>"} | {gap_symbol(row["gap"]) for row in words}))


def examples(text: str, tokens: dict[str, int], arm: str) -> list[dict]:
    if arm not in ARMS:
        raise ValueError("Unknown word-context screening arm")
    rows = parse(text)["words"]
    result = []
    for position, row in enumerate(rows):
        target = tokens.get("word:" + row["identity"], tokens["word:<other>"])
        if arm == "raw4":
            prefix = text[max(0, row["start"] - 4):row["start"]]
            context = ([0] * 4 + [tokens.get("char:" + char, 1) for char in prefix])[-4:]
        else:
            context = []
            for previous in (position - 2, position - 1):
                if previous < 0:
                    context.extend([0, 0])
                else:
                    context.extend([
                        tokens.get("word:" + rows[previous]["identity"], tokens["word:<other>"]),
                        tokens.get(gap_symbol(rows[previous + 1]["gap"]), 1)
                        if arm == "words_punctuation" else tokens["gap:<ignored>"],
                    ])
        result.append({"context": context, "target": target, "known": target != tokens["word:<other>"],
                       "punctuation_boundary": any(not c.isspace() for c in row["gap"]),
                       "offset": row["start"], "word": row["identity"]})
    return result


def measure(model: NeuralModel, rows: list[dict], prior: list[float]) -> dict:
    measured = []
    allowed = [i for i, value in enumerate(prior) if value]
    for row in rows:
        if not prior[row["target"]]:
            continue
        probabilities = model.forward(row["context"], output_prior=prior)[2]
        predicted = max(allowed, key=probabilities.__getitem__)
        measured.append({**row, "prediction": predicted,
                         "loss": -math.log(max(probabilities[row["target"]], 1e-300)),
                         "correct": predicted == row["target"]})
    def summary(selected):
        return {"targets": len(selected),
                "loss": statistics.fmean(row["loss"] for row in selected) if selected else None,
                "accuracy": statistics.fmean(row["correct"] for row in selected) if selected else None}
    known = [row for row in measured if row["known"]]
    return {"all_scored_targets": summary(measured), "known_words": summary(known),
            "excluded_targets": len(rows) - len(measured),
            "known_after_punctuation": summary([row for row in known if row["punctuation_boundary"]]),
            "known_without_punctuation": summary([row for row in known if not row["punctuation_boundary"]]),
            "rows": measured}


def compare(root: Path, *, acknowledge_review: bool, steps: int = 1500,
            known_targets_only: bool = False, larger: bool = False) -> dict:
    if acknowledge_review is not True:
        raise ValueError("Explicit approval of reused Acts screening is required")
    if type(steps) is not int or not 1 <= steps <= 5000:
        raise ValueError("Steps must be an integer from 1 to 5000")
    if larger:
        texts, manifest_hash = load_chapters(root, STAGED_DIRECTORY)
        text = "\n".join(texts[chapter] for chapter in range(6, 13))
        held = "\n".join(texts[chapter] for chapter in (13, 14))
        parent_id = "6045eb140d8e47b1a5c6890199251039"
        parent = root / ".data/neural-unit-benchmark" / parent_id
        raw_report = (parent / "report.json").read_bytes()
        archived = json.loads(raw_report)
        saved = json.loads((parent / "inputs.json").read_text())
        for value, key, saved_key in ((text, "training_sha256", "training"),
                                      (held, "evaluation_sha256", "evaluation")):
            if hashlib.sha256(value.encode()).hexdigest() != archived["source"][key] or saved[saved_key] != value:
                raise ValueError("Larger corpus does not exactly match the saved baseline passages")
        data = {"training": text, "validation": held, "manifest_sha256": manifest_hash}
    else:
        data = load_acts(root, True, include_final_test=False)
    text, held = data["training"], data["validation"]
    symbols = vocabulary(text)
    template = NeuralModel(symbols)
    prior = [float(symbol.startswith("word:")) for symbol in ["", ""] + symbols]
    if known_targets_only:
        prior[template.tokens["word:<other>"]] = 0.0
    setup = time.process_time()
    training = {arm: examples(text, template.tokens, arm) for arm in ARMS}
    validation = {}
    for arm in ARMS:
        rows = examples(held, template.tokens, arm)
        validation[arm] = [rows[i * len(rows) // min(512, len(rows))]
                           for i in range(min(512, len(rows)))]
    setup_cpu = time.process_time() - setup
    if not training["raw4"] or not validation["raw4"]:
        raise ValueError("Screening requires word targets in both splits")
    eligible = [i for i, row in enumerate(training["raw4"]) if prior[row["target"]]]
    run_id = uuid4().hex
    directory = root / ".data/neural-word-context" / run_id
    report = {
        "run_id": run_id, "status": "running", "steps_per_arm": steps, "learning_rate": 0.03,
        "seeds": [17, 117, 217], "symbols": symbols, "parameters": sum(map(len, template.weights.values())),
        "known_targets_only": known_targets_only, "eligible_training_targets": len(eligible),
        "larger": larger, "training_characters": len(text), "validation_characters": len(held),
        "setup_cpu_seconds": setup_cpu, "manifest_sha256": data["manifest_sha256"],
        "training_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "validation_sha256": hashlib.sha256(held.encode()).hexdigest(),
        "source": ("Exact saved larger baseline: Acts 6-12 training / Acts 13-14 reused evaluation."
                   if larger else "Existing authorized WEB Acts 1 train / Acts 2 reused validation. Acts 3 not loaded."),
        "policy": "SCREENING ONLY. Same base MLP, input/output vocabulary, initialization, word targets, "
                  "draw order, rate and parameter count in all arms. Predict normalized next-word identity "
                  "at known word boundaries, not characters: NOT comparable to earlier character accuracy. "
                  "Training-only top-64 word lexicon; all other words use an OTHER bucket for targets AND "
                  "context. Primary known-word metrics exclude OTHER, with support reported separately. "
                  "No future words in context. Raw4 sees last four raw characters; words_only sees two "
                  "previous word IDs with ignored-gap slots; words_punctuation adds training-derived gap "
                  "IDs (exact non-whitespace punctuation sequence plus newline/inline). "
                  "Case, spacing and exact punctuation remain losslessly in source records, but case and "
                  "whitespace details are not fully represented in neural features. Internal apostrophes/"
                  "hyphens remain part of word identity. Unseen gap IDs use unknown. "
                  "All outputs masked except the common 64-word-plus-OTHER target set. "
                  "Single shared network per arm, no extra branch. Three seeds, short reused validation "
                  "and several prior trials mean exploratory evidence, not untouched generalization. "
                  "No spelling generation, punctuation generation, grammar evaluation or activation.",
        "code_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                        for name in ("neural_word_context.py", "neural_experiment.py")},
        "runs": [],
    }
    if larger:
        report["archived_corpus"] = {"run_id": parent_id,
                                    "report_sha256": hashlib.sha256(raw_report).hexdigest(),
                                    "note": "Same exact passages, different word-target task; rerun matched "
                                            "controls, do not compare accuracy numerically to old character task."}
    if known_targets_only:
        report["policy"] += (
            " CLOSED-SET FOLLOW-UP: OTHER excluded from both output mask and training target draws; "
            "validation scores only known-word positions. Unknown context words still use OTHER. "
            "Target eligibility is oracle-filtered: not an all-word prediction or end-to-end generator test. "
            "Follow-up motivated by initial OTHER-bucket collapse, not an untouched preregistered result.")
    atomic_json(directory / "protocol.json", report)
    atomic_json(directory / "inputs.json", {"training": text, "validation": held,
        "training_records": parse(text), "validation_records": parse(held),
        "training_examples": training, "validation_examples": validation, "output_prior": prior})
    for seed_position, seed in enumerate(report["seeds"]):
        rng = random.Random(seed + 2)
        draws = [rng.choice(eligible) for _ in range(steps)]
        for arm in ARMS[seed_position:] + ARMS[:seed_position]:
            model = NeuralModel(symbols, seed=seed)
            entry = {"seed": seed, "arm": arm, "history": [],
                     "initial_weights_sha256": hashlib.sha256(json.dumps(model.weights).encode()).hexdigest(),
                     "draws_sha256": hashlib.sha256(json.dumps(draws).encode()).hexdigest(),
                     "before": measure(model, validation[arm], prior)}
            cpu = 0.0
            for step, draw in enumerate(draws, 1):
                row = training[arm][draw]
                start = time.process_time()
                model.update(row["context"], row["target"], learning_rate=0.03, output_prior=prior)
                cpu += time.process_time() - start
                if step in {max(1, steps // 2), steps}:
                    entry["history"].append({"step": step, "training_cpu_seconds": cpu,
                                            "validation": measure(model, validation[arm], prior)})
            atomic_json(directory / f"{seed}-{arm}.json", {"symbols": symbols, "weights": model.weights,
                                                         "entry": entry})
            report["runs"].append(entry)
            atomic_json(directory / "report.json", report)
            print(f"{seed} {arm}: {entry['history'][-1]['validation']['known_words']}; CPU={cpu:.3f}s", flush=True)
    report["summary"] = {
        arm: {group: {metric: statistics.fmean(
            row["history"][-1]["validation"][group][metric] for row in report["runs"] if row["arm"] == arm)
                      for metric in ("loss", "accuracy", "targets")}
              for group in ("all_scored_targets", "known_words", "known_after_punctuation", "known_without_punctuation")
              if all(row["history"][-1]["validation"][group]["targets"] for row in report["runs"] if row["arm"] == arm)}
        for arm in ARMS}
    report["status"] = "completed"
    atomic_json(directory / "report.json", report)
    print(json.dumps({"run_id": run_id, "summary": report["summary"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acknowledge-review", action="store_true")
    parser.add_argument("--known-targets-only", action="store_true")
    parser.add_argument("--larger", action="store_true", help="Reuse exact Acts 6-12 / 13-14 benchmark passages")
    parser.add_argument("--steps", type=int, default=1500)
    args = parser.parse_args()
    compare(Path(__file__).resolve().parent, acknowledge_review=args.acknowledge_review,
            known_targets_only=args.known_targets_only, larger=args.larger, steps=args.steps)
