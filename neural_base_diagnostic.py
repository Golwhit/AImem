"""Offline stability and representation diagnostic; never modifies active models."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import statistics
import time
from uuid import uuid4

from neural_base_comparison import WordBase, digest, generate, load_inputs
from neural_word_context import WORD
from original_generator import atomic_json


def ambiguity(examples: list) -> dict:
    groups = defaultdict(Counter)
    for context, target, prior in examples:
        groups[(tuple(context), tuple(i for i, value in enumerate(prior) if value))][target] += 1
    total = len(examples)
    if not total:
        raise ValueError("Ambiguity audit needs nonempty examples")
    unavoidable = sum(sum(counts.values()) - max(counts.values()) for counts in groups.values())
    return {
        "examples": total, "distinct_observable_inputs": len(groups),
        "conflicting_inputs": sum(len(counts) > 1 for counts in groups.values()),
        "unavoidable_token_errors": unavoidable,
        "deterministic_token_accuracy_ceiling": 1 - unavoidable / total,
        "minimum_mean_token_cross_entropy": sum(
            -count * math.log(count / sum(counts.values()))
            for counts in groups.values() for count in counts.values()
        ) / total,
        "scope": "Exact teacher-forced token targets given current features AND output masks. "
                 "This is an empirical representation bound, not a word-generation or semantic score.",
    }


def word_probes(text: str, limit: int = 16) -> list[dict]:
    words = list(WORD.finditer(text))
    chosen = words[:limit]
    return [{"prefix": text[:word.start()], "target": word.group(), "offset": word.start()}
            for word in chosen]


def measure(model: WordBase, examples: list, probes: list[dict]) -> dict:
    losses, correct = [], []
    for context, target, prior in examples:
        probabilities = model.model.forward(context, output_prior=prior)[2]
        if not all(math.isfinite(value) for value in probabilities):
            raise ArithmeticError("Diagnostic produced nonfinite probabilities")
        allowed = [i for i, value in enumerate(prior) if value]
        predicted = max(allowed, key=lambda token: (probabilities[token], -token))
        losses.append(-math.log(max(probabilities[target], 1e-300)))
        correct.append(predicted == target)
    rows = []
    for probe in probes:
        text = generate(model, probe["prefix"], length=32)["text"]
        match = WORD.match(text)
        prediction = match.group() if match and match.end() < len(text) else ""
        rows.append({**probe, "prediction": prediction, "continuation": text,
                     "requires_fallback": "emit:" + probe["target"] not in model.model.tokens})
    return {"token_loss": statistics.fmean(losses), "token_accuracy": statistics.fmean(correct),
            "word_accuracy": statistics.fmean(row["prediction"] == row["target"] for row in rows),
            "word_probes": len(rows), "rows": rows}


def train_trial(model: WordBase, first: str, second: str, *, seed: int, rate: float,
                checkpoints=(250, 750, 1500), second_updates: int = 500) -> tuple[dict, WordBase]:
    if (not checkpoints or any(type(n) is not int or n <= 0 for n in checkpoints)
            or list(checkpoints) != sorted(set(checkpoints))
            or type(second_updates) is not int or second_updates < 1
            or not math.isfinite(rate) or rate <= 0):
        raise ValueError("Invalid diagnostic training protocol")
    first_examples, second_examples = model.examples(first), model.examples(second)
    if not first_examples or not second_examples:
        raise ValueError("Both diagnostic phases require examples")
    first_probes, second_probes = word_probes(first), word_probes(second)
    before = {"first": measure(model, first_examples, first_probes),
              "second": measure(model, second_examples, second_probes)}
    rng = random.Random(seed + 2)
    rows, elapsed, updates = [], 0.0, 0
    for endpoint in checkpoints:
        start = time.process_time()
        while updates < endpoint:
            context, target, prior = first_examples[rng.randrange(len(first_examples))]
            model.model.update(context, target, learning_rate=rate, output_prior=prior)
            updates += 1
        elapsed += time.process_time() - start
        rows.append({"updates": updates, "training_cpu_seconds": elapsed,
                     **measure(model, first_examples, first_probes)})
    learned_second_before = measure(model, second_examples, second_probes)
    start = time.process_time()
    for _ in range(second_updates):
        context, target, prior = second_examples[rng.randrange(len(second_examples))]
        model.model.update(context, target, learning_rate=rate, output_prior=prior)
    elapsed += time.process_time() - start
    after_first = measure(model, first_examples, first_probes)
    after_second = measure(model, second_examples, second_probes)
    state = model.snapshot()
    restored = WordBase.restore(json.loads(json.dumps(state, allow_nan=False)))
    if (measure(restored, first_examples, first_probes) != after_first
            or measure(restored, second_examples, second_probes) != after_second):
        raise AssertionError("Diagnostic checkpoint restoration changed results")
    return {
        "seed": seed, "rate": rate, "before": before, "first_phase": rows,
        "second_before": learned_second_before, "after_second_first": after_first,
        "after_second_second": after_second, "second_updates": second_updates,
        "training_cpu_seconds": elapsed,
        "retention_loss_change": after_first["token_loss"] - rows[-1]["token_loss"],
        "retention_word_accuracy_change": after_first["word_accuracy"] - rows[-1]["word_accuracy"],
        "restoration_verified": True,
    }, model


def run(root: Path, *, acknowledge_review: bool) -> dict:
    if acknowledge_review is not True:
        raise ValueError("Diagnostic requires approval to reuse the existing authorized corpus")
    protected = {str(path.relative_to(root)): digest(path.read_bytes())
                 for path in (root / ".data/neural-chat").glob("*.json")}
    training, _, _, provenance = load_inputs(root)
    # Both phases come from approved training, never the evaluation chapters.
    boundary = training.find(" ", 384)
    end = training.find(" ", boundary + 384)
    if boundary < 0 or end < 0:
        raise ValueError("Approved training source is too short for the diagnostic")
    first, second = training[:boundary + 1], training[boundary + 1:end + 1]
    template = WordBase(training, 17)
    run_id = uuid4().hex
    directory = root / ".data/neural-base-diagnostic" / run_id
    report = {
        "run_id": run_id, "status": "running", "source": provenance,
        "training_sha256": digest(training.encode()),
        "phase_a_sha256": digest(first.encode()), "phase_b_sha256": digest(second.encode()),
        "phase_a_characters": len(first), "phase_b_characters": len(second),
        "seeds": [17, 117, 217], "rates": [.1, .03, .01],
        "checkpoints": [250, 750, 1500], "second_updates": 500,
        "parameters": sum(map(len, template.model.weights.values())),
        "policy": "Fresh offline working-base instances, full approved Acts6-12 training vocabulary "
                  "fixed across phases/rates/seeds. No evaluation vocabulary. Phase A/B are adjacent "
                  "approximately384-character slices of TRAINING. Same random draws per seed across "
                  "rates. Phase A receives 1500 replacement-sampled updates; B receives 500, no replay. "
                  "Tiny in-sample fit and interference stress test, not dialogue/generalization evidence. "
                  "Retention decline may reflect competing/aliased targets, not proof of catastrophic "
                  "forgetting. Token loss/accuracy and 16 complete-word probes are different tasks; "
                  "no 30-percent word-intelligence claims. No layers, activation, online training or "
                  "conversation ingestion. Historical corpus approval reused; draft curriculum not loaded.",
        "protected_before": protected, "code_sha256": {
            name: digest((root / name).read_bytes())
            for name in ("neural_base_diagnostic.py", "neural_base_comparison.py", "neural_experiment.py")},
        "representation": {"full_training": ambiguity(template.examples(training)),
                           "phase_a": ambiguity(template.examples(first)),
                           "phase_b": ambiguity(template.examples(second)),
                           "combined_independent_phases": ambiguity(
                               template.examples(first) + template.examples(second))},
        "trials": [],
    }
    atomic_json(directory / "inputs.json", {"phase_a": first, "phase_b": second,
                                          "vocabulary": template.snapshot()["symbols"],
                                          "lexicon": template.lexicon})
    atomic_json(directory / "protocol.json", report)
    print(f"Diagnostic {run_id}; A={len(first)} B={len(second)} characters", flush=True)
    for seed in report["seeds"]:
        for rate in report["rates"]:
            model = WordBase("", seed, lexicon=template.lexicon, symbols=template.model.characters)
            trial, model = train_trial(model, first, second, seed=seed, rate=rate)
            atomic_json(directory / f"{seed}-{rate}.json", {"state": model.snapshot(), "trial": trial})
            report["trials"].append(trial)
            atomic_json(directory / "report.json", report)
            print(f"seed={seed} rate={rate}: token loss "
                  f"{trial['before']['first']['token_loss']:.3f} -> "
                  f"{trial['first_phase'][-1]['token_loss']:.3f}; "
                  f"retention change={trial['retention_loss_change']:+.3f}", flush=True)
    report["protected_after"] = {str(path.relative_to(root)): digest(path.read_bytes())
                                 for path in (root / ".data/neural-chat").glob("*.json")}
    if report["protected_after"] != protected:
        raise RuntimeError("Pinned chat/control files changed during diagnostic")
    report["summary"] = {
        str(rate): {
            key: statistics.fmean(trial[key] for trial in report["trials"] if trial["rate"] == rate)
            for key in ("training_cpu_seconds", "retention_loss_change", "retention_word_accuracy_change")
        } | {
            "first_phase_final_loss": statistics.fmean(trial["first_phase"][-1]["token_loss"]
                                                      for trial in report["trials"] if trial["rate"] == rate),
            "first_phase_final_word_accuracy": statistics.fmean(trial["first_phase"][-1]["word_accuracy"]
                                                               for trial in report["trials"] if trial["rate"] == rate),
        } for rate in report["rates"]}
    report["status"] = "completed"
    atomic_json(directory / "report.json", report)
    print(json.dumps({"run_id": run_id, "representation": report["representation"],
                      "summary": report["summary"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acknowledge-review", action="store_true")
    args = parser.parse_args()
    run(Path(__file__).resolve().parent, acknowledge_review=args.acknowledge_review)
