"""Paired development comparison of the live unit MLP and character baseline.

Reuses approved Acts text and prior bank evidence: not a fresh locked test.
No token losses are compared. Next-word predictions use identical text prefixes.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
from uuid import uuid4

from neural_experiment import NeuralModel, RATE
from neural_character_onlies import CharacterOnlies
from neural_patterns import STAGED_DIRECTORY, load_chapters
from neural_proof import neural_update_operations
from neural_skeleton_runtime import END, UnitCodec, load_bank, word_boundaries, word_character
from neural_skeletons import WORD
from original_generator import atomic_json


DIRECTORY = ".data/neural-unit-benchmark"
SEEDS = (17, 117, 217)
PASSES = (1, 2, 4)
MATCHED_UPDATES = 12_000
PROBES = 48
CRITERIA = {
    "similar_quality": "At four complete input passes, mean exact next-word accuracy is no more than 0.02 lower "
                       "and normalized word edit distance no more than 0.02 higher than the character baseline.",
    "less_effort": "At four passes, mean training CPU time and estimated neural update operations are each "
                   "at most 80% of baseline. Evaluation excluded; setup/bank costs reported separately.",
    "scope": "Development only. Reused Bible text; bank already saw both chapters. Not proof of generalization, "
             "coherence, unique benefit beyond ordinary tokenization, or quality-matched convergence.",
}


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def edit_distance(left: str, right: str) -> int:
    row = list(range(len(right) + 1))
    for i, first in enumerate(left, 1):
        following = [i]
        for j, second in enumerate(right, 1):
            following.append(min(following[-1] + 1, row[j] + 1, row[j - 1] + (first != second)))
        row = following
    return row[-1]


def probes(text: str, count: int) -> list[dict]:
    candidates = [m for m in WORD.finditer(text)
                  if m.start() >= 32 and word_boundaries(text, m.start(), m.end())]
    if not candidates or count < 1:
        raise ValueError("Next-word evaluation needs positive count and eligible words")
    chosen = [candidates[i * len(candidates) // min(count, len(candidates))]
              for i in range(min(count, len(candidates)))]
    return [{"prefix": text[:m.start()], "target": m.group(), "offset": m.start()} for m in chosen]


def next_word(model: NeuralModel, codec: UnitCodec | CharacterOnlies | None, prefix: str,
              *, trace: dict | None = None) -> str:
    if isinstance(codec, CharacterOnlies):
        generated = codec.sample(model, prefix, length=32, greedy=True)
        if trace is not None:
            trace.update(generated)
        output = generated["text"]
        match = WORD.match(output)
        return match.group() if match and match.end() < len(output) else ""
    pieces = codec.encode(prefix, open_tail=True) if codec else list(prefix)
    context = ([0] * model.context + [model.tokens.get(piece, 1) for piece in pieces])[-model.context:]
    output = ""
    after_unit = False
    for _ in range(32):
        probabilities = (codec.distribution(model, context, prefix + output,
                                            after_unit=after_unit, remaining=64 - len(output))
                         if codec else model.forward(context)[2])
        token = max(range(2, model.size), key=lambda k: (probabilities[k], -k))
        piece = model.characters[token - 2]
        if piece == END:
            return output if WORD.fullmatch(output) else ""
        if not all(word_character(char) for char in piece):
            return output if WORD.fullmatch(output) else ""
        output += piece
        after_unit = codec is not None and piece in codec.units
        context = context[1:] + [token]
    return ""


def quality(model: NeuralModel, codec: UnitCodec | CharacterOnlies | None, items: list[dict]) -> dict:
    rows = []
    traces = []
    for item in items:
        trace = {}
        prediction = (next_word(model, codec, item["prefix"], trace=trace)
                      if isinstance(codec, CharacterOnlies) else next_word(model, codec, item["prefix"]))
        rows.append({"offset": item["offset"], "target": item["target"], "prediction": prediction})
        traces.append(trace)
    return {
        "probes": len(rows), "exact_next_word_accuracy": statistics.fmean(
            row["target"] == row["prediction"] for row in rows),
        "generation_rule_matches": sum(trace.get("unit_occurrences", 0) for trace in traces),
        "generation_forced_characters": sum(trace.get("forced_characters", 0) for trace in traces),
        "normalized_edit_distance": statistics.fmean(
            edit_distance(row["target"], row["prediction"]) / max(len(row["target"]), len(row["prediction"]), 1)
            for row in rows),
        "rows": rows,
    }


def character_probe_examples(model: NeuralModel, text: str, count: int = 2048) -> list:
    if not text or type(count) is not int or count < 1:
        raise ValueError("Character evaluation needs text and a positive probe count")
    all_examples = model.examples(text)
    size = min(count, len(all_examples))
    return [all_examples[i * len(all_examples) // size] for i in range(size)]


def summarize(runs: list[dict], *, matched_updates: int = MATCHED_UPDATES) -> dict:
    result = {}
    for budget in ("passes-4", f"updates-{matched_updates}"):
        arms = {}
        for arm in dict.fromkeys(run["arm"] for run in runs):
            rows = [next(row for row in run["checkpoints"] if budget in row["budgets"])
                    for run in runs if run["arm"] == arm]
            arms[arm] = {
                key: statistics.fmean(row[key] for row in rows)
                for key in ("training_cpu_seconds", "training_wall_seconds", "estimated_update_operations",
                            "updates", "exact_next_word_accuracy", "normalized_edit_distance")
            }
            if all("all_character_probe_loss" in row for row in rows):
                arms[arm]["all_character_probe_loss"] = statistics.fmean(
                    row["all_character_probe_loss"] for row in rows)
        base, units = arms["characters"], arms["skeleton_units"]
        result[budget] = {
            "arms": arms,
            "accuracy_difference": units["exact_next_word_accuracy"] - base["exact_next_word_accuracy"],
            "edit_distance_difference": units["normalized_edit_distance"] - base["normalized_edit_distance"],
            "cpu_ratio": units["training_cpu_seconds"] / base["training_cpu_seconds"],
            "wall_ratio": units["training_wall_seconds"] / base["training_wall_seconds"],
            "operation_ratio": units["estimated_update_operations"] / base["estimated_update_operations"],
        }
    primary = result["passes-4"]
    result["verdicts"] = {
        "similar_quality": primary["accuracy_difference"] >= -0.02 and primary["edit_distance_difference"] <= 0.02,
        "less_effort": primary["cpu_ratio"] <= 0.8 and primary["operation_ratio"] <= 0.8,
    }
    return result


def add_combined(root: Path, source_run_id: str, *, acknowledge_review: bool) -> dict:
    """Evaluate baseline weights with rules; retain every original training update."""
    if acknowledge_review is not True:
        raise ValueError("Explicit approval to reuse the approved Acts text is required")
    if len(source_run_id) != 32 or any(c not in "0123456789abcdef" for c in source_run_id):
        raise ValueError("Source run must be a lowercase 32-character hexadecimal ID")
    source = root / DIRECTORY / source_run_id
    original = json.loads((source / "report.json").read_text())
    if original["status"] != "completed" or original["method"] != "character_recognition_onlies":
        raise ValueError("Combined comparison requires a completed character-recognition run")
    if any(entry["arm"] == "baseline_plus_rules" for entry in original["runs"]):
        raise ValueError("Source comparison already contains a combined arm")
    inputs = json.loads((source / "inputs.json").read_text())
    for name in ("training", "evaluation"):
        if text_hash(inputs[name]) != original["source"][f"{name}_sha256"]:
            raise ValueError(f"Saved {name} text does not match the source report")
    codec = CharacterOnlies.restore(inputs["unit_codec"])
    report = deepcopy(original)
    report.pop("summary", None)
    report.pop("matched_generation", None)
    report.pop("sample_budget_note", None)
    report.update({
        "run_id": uuid4().hex, "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run_id": source_run_id,
        "combined_note": "Baseline plus rules reuses exact baseline checkpoint weights and all training updates. "
                         "Training time is inherited, not a new measurement. Rule-assisted evaluation and "
                         "generation are measured separately. No training-speed saving is claimed for this arm.",
        "evaluation_code_hashes": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                   for name in ("neural_unit_benchmark.py", "neural_character_onlies.py",
                                                "neural_experiment.py")},
        "source_report_sha256": hashlib.sha256((source / "report.json").read_bytes()).hexdigest(),
    })
    directory = root / DIRECTORY / report["run_id"]
    directory.mkdir(parents=True)
    atomic_json(directory / "protocol.json", report)
    atomic_json(directory / "inputs.json", inputs)
    latest = root / ".data/neural-experiments/latest-unit-benchmark.json"
    matched_budget = f"updates-{report['settings']['matched_updates']}"
    report["matched_generation"] = {"budget": matched_budget, "runs": []}
    report["sample_budget_note"] = "Fresh samples from all three saved equal-update checkpoints; no retraining."
    for entry in original["runs"]:
        arm = entry["arm"]
        for checkpoint in entry["checkpoints"]:
            saved = json.loads((source / f"{entry['seed']}-{arm}-step-{checkpoint['updates']}.json").read_text())
            model = NeuralModel(saved["characters"], seed=entry["seed"])
            model.weights = saved["weights"]
            if arm == "characters":
                start_cpu, start_wall = time.process_time(), time.perf_counter()
                measured = quality(model, codec, inputs["probes"])
                measured["all_character_probe_loss"] = model.loss(
                    character_probe_examples(model, inputs["evaluation"]))
                row = {**deepcopy(checkpoint), **measured,
                       "evaluation_cpu_seconds": time.process_time() - start_cpu,
                       "evaluation_wall_seconds": time.perf_counter() - start_wall}
                if row["all_character_probe_loss"] != checkpoint["all_character_probe_loss"]:
                    raise ValueError("Restored baseline loss differs from original; comparison is not equivalent")
                if checkpoint is entry["checkpoints"][0]:
                    combined = {**deepcopy(entry), "arm": "baseline_plus_rules", "checkpoints": [],
                                "samples": [], "training_reused_from": source_run_id}
                    report["runs"].append(combined)
                combined["checkpoints"].append(row)
                atomic_json(directory / f"{entry['seed']}-baseline_plus_rules-step-{row['updates']}.json", {
                    "characters": model.characters, "weights": model.weights,
                    "unit_codec": codec.snapshot(), "evidence": row,
                    "baseline_weights_sha256": text_hash(json.dumps(model.weights, sort_keys=True)),
                })
                print(f"{entry['seed']} baseline_plus_rules {row['budgets']}: "
                      f"accuracy={row['exact_next_word_accuracy']:.3f} "
                      f"rule_matches={row['generation_rule_matches']}", flush=True)
            if matched_budget in checkpoint["budgets"]:
                sample_arms = ("characters", "baseline_plus_rules") if arm == "characters" else (arm,)
                for sample_arm in sample_arms:
                    samples = []
                    for prompt in ("The ", "and ", "They "):
                        start_cpu, start_wall = time.process_time(), time.perf_counter()
                        generated = ({"text": model.sample(prompt, length=120), "decisions": 120,
                                      "unit_occurrences": 0, "forced_characters": 0}
                                     if sample_arm == "characters" else codec.sample(model, prompt, length=120))
                        samples.append({"prompt": prompt, **generated,
                                        "cpu_seconds": time.process_time() - start_cpu,
                                        "wall_seconds": time.perf_counter() - start_wall})
                    report["matched_generation"]["runs"].append(
                        {"seed": entry["seed"], "arm": sample_arm, "samples": samples})
                    if sample_arm == "baseline_plus_rules":
                        combined["samples"] = samples
                        combined["sample_budgets"] = [matched_budget]
        atomic_json(directory / "report.json", report)
        atomic_json(latest, report)
    report["summary"] = summarize(report["runs"], matched_updates=report["settings"]["matched_updates"])
    report["status"] = "completed"
    atomic_json(directory / "matched-generation.json", report["matched_generation"])
    atomic_json(directory / "report.json", report)
    atomic_json(latest, report)
    return report


def run(root: Path, *, acknowledge_review: bool, legacy_units: bool = False, large: bool = False,
        indexed_onlies: bool = False) -> dict:
    if acknowledge_review is not True:
        raise ValueError("Explicit approval to reuse the approved Acts text is required")
    if legacy_units and indexed_onlies:
        raise ValueError("Select indexed onlies or legacy reproduction, not both")
    atomic_units = legacy_units or indexed_onlies
    setup_wall, setup_cpu = time.perf_counter(), time.process_time()
    texts, manifest_hash = load_chapters(root, STAGED_DIRECTORY)
    if large:
        train = "\n".join(texts[chapter] for chapter in range(6, 13))
        held = "\n".join(texts[chapter] for chapter in (13, 14))
    else:
        train = texts[6][:3000].rsplit(" ", 1)[0] + " "
        held = texts[7][:1500].rsplit(" ", 1)[0] + " "
    seeds = (17, 117, 217, 317, 417) if large else SEEDS
    matched_updates = 120_000 if large else MATCHED_UPDATES
    bank = load_bank(root)
    codec = UnitCodec.from_bank(bank, train) if atomic_units else CharacterOnlies.cached(root)
    items = probes(held, 256 if large else PROBES)
    run_id = uuid4().hex
    directory = root / DIRECTORY / run_id
    directory.mkdir(parents=True)
    report = {
        "kind": "unit-efficiency-development", "run_id": run_id, "status": "running",
        "method": ("indexed_onlies" if indexed_onlies else
                   "legacy_word_outputs" if legacy_units else "character_recognition_onlies"),
        "profile": "large" if large else "small",
        "created_at": datetime.now(timezone.utc).isoformat(), "criteria": CRITERIA,
        "settings": {"seeds": list(seeds), "passes": list(PASSES), "matched_updates": matched_updates,
                     "rate": RATE, "context_tokens": 4, "probes": len(items),
                     "all_character_probes": min(2048, len(held)) if not atomic_units else 0},
        "source": {"training": "Acts 6-12 complete" if large else "Acts 6 prefix",
                   "evaluation": "Acts 13-14 complete" if large else "Acts 7 prefix",
                   "training_characters": len(train), "evaluation_characters": len(held),
                   "training_sha256": text_hash(train), "evaluation_sha256": text_hash(held),
                   "manifest_sha256": manifest_hash, "bank": codec.provenance,
                   "approval": "User approved the paired comparison on already approved public-domain text."},
        "code_hashes": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
                        ("neural_unit_benchmark.py", "neural_skeleton_runtime.py", "neural_experiment.py",
                         "neural_character_onlies.py")},
        "reading": codec.reading(train),
        "units": len(codec.units) if isinstance(codec, UnitCodec) else len(codec.words), "runs": [],
        "setup_cpu_seconds": time.process_time() - setup_cpu,
        "setup_wall_seconds": time.perf_counter() - setup_wall,
        "bank_build_included": False,
        "bank_build_note": "Reusable bank and prefix rules were built earlier; their construction cost is not included in training.",
    }
    if indexed_onlies:
        report["indexed_policy"] = {
            "selection": "Up to 128 bank-backed forms present in training, ranked by characters saved. "
                         "Same existing UnitCodec as the legacy reproduction, now explicitly testing atomic references.",
            "training": "One target and one context position per indexed occurrence; train placement, not internal spelling. "
                        "Unrecognized material remains characters. A reserved end marker terminates the stream.",
            "generation": "The base selects a token ID. Indexed IDs emit their entire fixed pattern with one forward pass. "
                          "No unique-prefix requirement or internal character predictions.",
            "cost": "Additional output classes/parameters and longer effective character context are measured, not hidden. "
                    "Recognition is performed ahead of training, not a concurrent-worker speed test.",
        }
    atomic_json(directory / "protocol.json", report)
    atomic_json(directory / "inputs.json", {"training": train, "evaluation": held, "probes": items,
                                          "unit_codec": codec.snapshot()})
    latest = root / ".data/neural-experiments/latest-unit-benchmark.json"
    atomic_json(latest, report)
    for index, seed in enumerate(seeds):
        order = ("characters", "skeleton_units") if index % 2 == 0 else ("skeleton_units", "characters")
        for arm in order:
            active = codec if arm == "skeleton_units" else None
            start_cpu, start_wall = time.process_time(), time.perf_counter()
            vocabulary = set(train)
            if active and isinstance(codec, UnitCodec):
                vocabulary |= set(codec.units) | {END}
            model = NeuralModel(sorted(vocabulary), seed=seed)
            examples = active.examples(model, train) if active else model.examples(train)
            character_probes = character_probe_examples(model, held) if not atomic_units else None
            setup = {"cpu_seconds": time.process_time() - start_cpu,
                     "wall_seconds": time.perf_counter() - start_wall}
            milestones = {len(examples) * passes: [f"passes-{passes}"] for passes in PASSES}
            milestones.setdefault(matched_updates, []).append(f"updates-{matched_updates}")
            rng = random.Random(seed + 2)
            sequence = []
            update = 0
            cpu_seconds = wall_seconds = 0.0
            entry = {"seed": seed, "arm": arm, "targets_per_pass": len(examples),
                     "initial_weights_sha256": text_hash(json.dumps(model.weights, sort_keys=True)),
                     "parameters": sum(map(len, model.weights.values())),
                     "vocabulary": model.size, "setup": setup, "checkpoints": []}
            if indexed_onlies:
                entry["index"] = {word: {"token_id": model.tokens[word], **evidence}
                                  for word, evidence in codec.units.items()} if active else {}
            for milestone in sorted(milestones):
                start_cpu, start_wall = time.process_time(), time.perf_counter()
                while update < milestone:
                    if not sequence:
                        sequence = list(range(len(examples)))
                        rng.shuffle(sequence)
                    context, target = examples[sequence.pop()]
                    model.update(context, target, learning_rate=RATE)
                    update += 1
                cpu_seconds += time.process_time() - start_cpu
                wall_seconds += time.perf_counter() - start_wall
                evaluation_cpu, evaluation_wall = time.process_time(), time.perf_counter()
                measured = quality(model, active, items)
                if character_probes is not None:
                    measured["all_character_probe_loss"] = model.loss(character_probes)
                row = {
                    "budgets": milestones[milestone], "updates": update, "passes": update / len(examples),
                    "training_cpu_seconds": cpu_seconds, "training_wall_seconds": wall_seconds,
                    "estimated_update_operations": update * neural_update_operations(model, prior=False),
                    **measured, "evaluation_cpu_seconds": time.process_time() - evaluation_cpu,
                    "evaluation_wall_seconds": time.perf_counter() - evaluation_wall,
                }
                entry["checkpoints"].append(row)
                atomic_json(directory / f"{seed}-{arm}-step-{update}.json", {
                    "characters": model.characters, "weights": model.weights,
                    "unit_codec": active.snapshot() if active else None, "evidence": row,
                })
                print(f"{seed} {arm} {row['budgets']}: accuracy={measured['exact_next_word_accuracy']:.3f} "
                      f"edit={measured['normalized_edit_distance']:.3f} cpu={cpu_seconds:.2f}s", flush=True)
            entry["samples"] = []
            entry["sample_budgets"] = milestones[max(milestones)]
            known_words = set(WORD.findall(train))
            for prompt in ("The ", "and ", "They "):
                generation_cpu, generation_wall = time.process_time(), time.perf_counter()
                generated = active.sample(model, prompt, length=120) if active else {
                    "text": model.sample(prompt, length=120), "decisions": 120, "unit_occurrences": 0}
                words = WORD.findall(generated["text"])
                entry["samples"].append({
                    "prompt": prompt, **generated, "cpu_seconds": time.process_time() - generation_cpu,
                    "wall_seconds": time.perf_counter() - generation_wall,
                    "training_vocabulary_word_fraction": sum(w in known_words for w in words) / len(words) if words else None,
                    "note": "Samples at final executed checkpoint (see sample_budgets). Vocabulary overlap is not coherence or correctness. "
                            "Generated experiment text, not scripture.",
                })
            if indexed_onlies:
                matched = json.loads((directory / f"{seed}-{arm}-step-{matched_updates}.json").read_text())
                model.weights = matched["weights"]
                matched_samples = []
                for prompt in ("The ", "and ", "They "):
                    generation_cpu, generation_wall = time.process_time(), time.perf_counter()
                    generated = active.sample(model, prompt, length=120) if active else {
                        "text": model.sample(prompt, length=120), "decisions": 120, "unit_occurrences": 0}
                    matched_samples.append({
                        "prompt": prompt, **generated,
                        "cpu_seconds": time.process_time() - generation_cpu,
                        "wall_seconds": time.perf_counter() - generation_wall,
                    })
                report.setdefault("matched_generation", {
                    "budget": f"updates-{matched_updates}", "runs": [],
                })["runs"].append({"seed": seed, "arm": arm, "samples": matched_samples})
                report["sample_budget_note"] = "Samples reloaded at the same update count, with identical prompts and RNG seed."
                model.weights = json.loads((directory / f"{seed}-{arm}-step-{max(milestones)}.json").read_text())["weights"]
            atomic_json(directory / f"{seed}-{arm}.json", {
                "characters": model.characters, "weights": model.weights,
                "unit_codec": active.snapshot() if active else None, "evidence": entry,
            })
            report["runs"].append(entry)
            atomic_json(directory / "report.json", report)
            if indexed_onlies:
                atomic_json(directory / "matched-generation.json", report["matched_generation"])
            atomic_json(latest, report)
    report["summary"] = summarize(report["runs"], matched_updates=matched_updates)
    report["status"] = "completed"
    atomic_json(directory / "report.json", report)
    atomic_json(latest, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acknowledge-review", action="store_true")
    parser.add_argument("--legacy-units", action="store_true", help="Reproduce the superseded expanded-output approach")
    parser.add_argument("--indexed-onlies", action="store_true",
                        help="Test atomic indexed patterns: one target/context position/emission per recognized form")
    parser.add_argument("--large", action="store_true", help="Seven complete training chapters, two evaluation chapters, five seeds")
    parser.add_argument("--add-combined", metavar="RUN_ID",
                        help="Add baseline training plus rules using a completed run's exact saved checkpoints")
    args = parser.parse_args()
    if args.add_combined and (args.large or args.legacy_units or args.indexed_onlies):
        parser.error("--add-combined uses saved source settings; do not combine with training flags")
    if args.indexed_onlies and args.legacy_units:
        parser.error("--indexed-onlies and --legacy-units are mutually exclusive")
    root = Path(__file__).resolve().parent
    report = (add_combined(root, args.add_combined, acknowledge_review=args.acknowledge_review)
              if args.add_combined else run(root, acknowledge_review=args.acknowledge_review,
                                           legacy_units=args.legacy_units, large=args.large,
                                           indexed_onlies=args.indexed_onlies))
    print(json.dumps({"run_id": report["run_id"], "summary": report["summary"]}, indent=2))
