"""Five-arm full-text generation comparison; offline and never auto-activated."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
from uuid import uuid4

from neural_candidate_layers import CandidateIndex
from neural_experiment import NeuralModel
from neural_patterns import load_chapters, STAGED_DIRECTORY
from neural_skeleton_runtime import load_bank
from neural_unit_benchmark import edit_distance
from neural_word_context import WORD, gap_symbol
from original_generator import atomic_json


ARMS = ("original", "new", "new_l0", "new_l01", "new_l012")
PASSES = (1, 2, 4)
SEEDS = (17, 117, 217)
RATE = 0.1
ARCHIVE = "6045eb140d8e47b1a5c6890199251039"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class WordBase:
    """Same base MLP, causal word/gap features, fixed word payloads and character fallback."""

    def __init__(self, training: str, seed: int, *, lexicon: list[str] | None = None,
                 symbols: list[str] | None = None):
        counts = Counter(match.group().lower().replace("’", "'") for match in WORD.finditer(training))
        self.lexicon = lexicon if lexicon is not None else sorted(counts, key=lambda w: (-counts[w], w))[:64]
        self.known = set(self.lexicon)
        if symbols is None:
            surfaces = {match.group() for match in WORD.finditer(training)
                        if match.group().lower().replace("’", "'") in self.known}
            gaps = self.gaps(training)
            symbols = (sorted("char:" + char for char in set(training))
                       + sorted("emit:" + word for word in surfaces)
                       + sorted("identity:" + word for word in self.known)
                       + ["identity:<other>"]
                       + sorted(set(gaps)))
        self.model = NeuralModel(symbols, seed=seed)
        self.payloads = {token: symbol[5:]
                         for symbol, token in self.model.tokens.items()
                         if symbol.startswith(("char:", "emit:"))}
        self.word_tokens = {token for symbol, token in self.model.tokens.items() if symbol.startswith("emit:")}
        self.character_tokens = set(self.payloads) - self.word_tokens

    @staticmethod
    def gaps(text: str) -> list[str]:
        values = []
        end = 0
        for match in WORD.finditer(text):
            gap = text[end:match.start()]
            values.extend(gap_symbol(gap[:i]) for i in range(len(gap) + 1))
            end = match.end()
        gap = text[end:]
        values.extend(gap_symbol(gap[:i]) for i in range(len(gap) + 1))
        return values

    def encode(self, text: str) -> list[int]:
        tokens = []
        offset = 0
        for match in WORD.finditer(text):
            tokens.extend(self.model.tokens["char:" + char] for char in text[offset:match.start()])
            surface = "emit:" + match.group()
            if surface in self.model.tokens:
                tokens.append(self.model.tokens[surface])
            else:
                tokens.extend(self.model.tokens["char:" + char] for char in match.group())
            offset = match.end()
        tokens.extend(self.model.tokens["char:" + char] for char in text[offset:])
        if "".join(self.payloads[token] for token in tokens) != text:
            raise AssertionError("Word base tokenization changed the source")
        return tokens

    def context(self, prefix: str) -> list[int]:
        # Reverse matching stops after three words without scanning all preceding text.
        matches = []
        cursor = len(prefix)
        letter = lambda char: char.isascii() and char.isalpha()
        while cursor and len(matches) < 3:
            while cursor and not letter(prefix[cursor - 1]):
                cursor -= 1
            if not cursor:
                break
            end = cursor
            while cursor and letter(prefix[cursor - 1]):
                cursor -= 1
            while cursor >= 2 and prefix[cursor - 1] in "'’-" and letter(prefix[cursor - 2]):
                cursor -= 1
                while cursor and letter(prefix[cursor - 1]):
                    cursor -= 1
            match = WORD.match(prefix, cursor, end)
            if match is None:
                raise AssertionError("Backward word scan disagrees with word parser")
            matches.append(match)
        matches.reverse()
        partial = matches[-1] if matches and matches[-1].end() == len(prefix) else None
        completed = matches[:-1] if partial else matches
        def identity(match):
            word = match.group().lower().replace("’", "'")
            return self.model.tokens.get("identity:" + word, self.model.tokens["identity:<other>"])
        if partial:
            previous = completed[-1] if completed else None
            gap = prefix[previous.end() if previous else 0:partial.start()]
            tail = partial.group()[-2:]
            return [identity(previous) if previous else 0,
                    self.model.tokens.get(gap_symbol(gap), 1),
                    *([0] * (2 - len(tail))), *(self.model.tokens.get("char:" + c, 1) for c in tail)]
        context = []
        for i in (-2, -1):
            if len(completed) < -i:
                context.extend([0, 0])
            else:
                match = completed[i]
                following_start = completed[i + 1].start() if i == -2 else len(prefix)
                gap = prefix[match.end():following_start]
                context.extend([identity(match), self.model.tokens.get(gap_symbol(gap), 1)])
        return context

    def prior(self, prefix: str, remaining: int | None = None) -> list[float]:
        # Whole-word payloads cannot start inside an unfinished word. Character fallback
        # remains possible everywhere, including for unseen spellings; no OTHER output.
        inside = bool(prefix and prefix[-1].isascii() and prefix[-1].isalpha())
        allowed = self.character_tokens if inside else set(self.payloads)
        return [float(token in allowed and (remaining is None or len(self.payloads[token]) <= remaining))
                for token in range(self.model.size)]

    def examples(self, text: str) -> list[tuple[list[int], int, list[float]]]:
        prefix = ""
        result = []
        for token in self.encode(text):
            prior = self.prior(prefix)
            if not prior[token]:
                raise ValueError("Hybrid tokenizer produced a word token inside another word")
            result.append((self.context(prefix), token, prior))
            prefix += self.payloads[token]
        return result

    def snapshot(self) -> dict:
        return {"lexicon": self.lexicon, "symbols": self.model.characters, "weights": self.model.weights}

    @classmethod
    def restore(cls, state: dict) -> "WordBase":
        model = cls("", 17, lexicon=state["lexicon"], symbols=state["symbols"])
        model.model.weights = state["weights"]
        return model


def generate(model: NeuralModel | WordBase, prompt: str, index: CandidateIndex | None = None,
             *, length: int = 64) -> dict:
    if type(length) is not int or not 1 <= length <= 500:
        raise ValueError("Generation length must be an integer from 1 to 500 characters")
    if "\x00" in prompt:
        raise ValueError("Prompt contains the reserved layer namespace")
    extracted = index.extract(prompt, open_tail=True) if index else None
    history = extracted["pieces"] if extracted else list(prompt)
    remainder = extracted["remainder"] if extracted else prompt
    output = ""
    pending = ""
    calls = lookups = emissions = 0
    while len(output) < length:
        if index:
            choices = index.choices(history, length - len(output))
            reference = min(choices, key=lambda key: (-choices[key], key))
            lookups += 1
            if reference:
                payload = index.codec.units[reference]["text"]
                output += payload
                history.append(reference)
                emissions += 1
                continue
        if pending:
            payload = pending[0]
            pending = pending[1:]
        elif isinstance(model, WordBase):
            probabilities = model.model.forward(model.context(remainder),
                output_prior=model.prior(remainder, length - len(output)))[2]
            if (not all(math.isfinite(value) and value >= 0 for value in probabilities)
                    or not any(probabilities[token] > 0 for token in model.payloads)):
                raise ArithmeticError("Hybrid generation has no finite payload probability mass")
            token = max(model.payloads, key=lambda k: (probabilities[k], -k))
            pending = model.payloads[token]
            payload, pending = pending[0], pending[1:]
            calls += 1
        else:
            context = ([0] * model.context +
                       [model.tokens.get(c, 1) for c in remainder[-model.context:]])[-model.context:]
            probabilities = model.forward(context)[2]
            if (not all(math.isfinite(value) and value >= 0 for value in probabilities)
                    or not any(probabilities[2:])):
                raise ArithmeticError("Character generation has no finite output probability mass")
            token = max(range(2, model.size), key=lambda k: (probabilities[k], -k))
            payload = model.characters[token - 2]
            calls += 1
        output += payload
        remainder += payload
        history.extend(payload)
    return {"text": output, "neural_calls": calls, "index_lookups": lookups, "indexed_emissions": emissions}


def evaluate(model, items: list[dict], index=None) -> dict:
    rows = []
    start = time.process_time()
    for item in items:
        sample = generate(model, item["prefix"], index)
        match = WORD.match(sample["text"])
        prediction = match.group() if match and match.end() < len(sample["text"]) else ""
        rows.append({"offset": item["offset"], "target": item["target"], "prediction": prediction,
                     **sample})
    return {"accuracy": statistics.fmean(row["prediction"] == row["target"] for row in rows),
            "edit": statistics.fmean(edit_distance(row["prediction"], row["target"]) /
                                    max(len(row["prediction"]), len(row["target"]), 1) for row in rows),
            "probes": len(rows), "evaluation_cpu_seconds": time.process_time() - start,
            "rows": rows}


def load_inputs(root: Path) -> tuple[str, str, list[dict], dict]:
    texts, manifest_hash = load_chapters(root, STAGED_DIRECTORY)
    train = "\n".join(texts[c] for c in range(6, 13))
    held = "\n".join(texts[c] for c in (13, 14))
    archive = root / ".data/neural-unit-benchmark" / ARCHIVE
    report_bytes = (archive / "report.json").read_bytes()
    report = json.loads(report_bytes)
    inputs = json.loads((archive / "inputs.json").read_text())
    if (train != inputs["training"] or held != inputs["evaluation"]
            or digest(train.encode()) != report["source"]["training_sha256"]
            or digest(held.encode()) != report["source"]["evaluation_sha256"]):
        raise ValueError("Five-arm corpus must exactly match saved larger baseline")
    # Keep the archived offsets, but determine the full word with one common
    # apostrophe/hyphen-aware parser for all arms; no 64-word target filtering.
    items = []
    for probe in inputs["probes"]:
        match = WORD.match(held, probe["offset"])
        if match is None:
            raise ValueError("Archived probe is not a word start")
        items.append({"offset": probe["offset"], "target": match.group(),
                      "prefix": held[:probe["offset"]]})
    return train, held, items, {"archive_run": ARCHIVE, "archive_report_sha256": digest(report_bytes),
                               "manifest_sha256": manifest_hash}


def summarize(report: dict) -> None:
    rows = report["runs"]
    arms = tuple(arm for arm in ARMS if any(row["arm"] == arm for row in rows))
    report["summary"] = {
        str(passes): {
            arm: {key: statistics.fmean(
                checkpoint[key] for row in rows if row["arm"] == arm
                for checkpoint in row["checkpoints"] if checkpoint["passes"] == passes)
                  for key in ("accuracy", "edit", "training_cpu_seconds", "total_cpu_seconds",
                              "updates", "evaluation_cpu_seconds")}
            for arm in arms}
        for passes in PASSES}
    report["time_to_original_accuracy"] = []
    report["time_to_original_best_accuracy"] = []
    report["equal_or_lower_cpu_checks"] = []
    for seed in report["seeds"]:
        original = next(row for row in rows if row["seed"] == seed and row["arm"] == "original")
        baseline = original["checkpoints"][-1]
        original_points = [original["before"], *original["checkpoints"]]
        for row in (r for r in rows if r["seed"] == seed):
            points = [row["before"], *row["checkpoints"]]
            for destination, threshold in (
                ("time_to_original_accuracy", baseline["accuracy"]),
                ("time_to_original_best_accuracy", max(point["accuracy"] for point in original_points)),
            ):
                first_original = next(point for point in original_points if point["accuracy"] >= threshold)
                crossing = next((point for point in points if point["accuracy"] >= threshold), None)
                reduction = (1 - crossing["total_cpu_seconds"] / first_original["total_cpu_seconds"]
                             if crossing and first_original["total_cpu_seconds"] > 0 and threshold > 0 else None)
                report[destination].append({
                    "seed": seed, "arm": row["arm"], "threshold": threshold,
                    "crossing_pass": crossing["passes"] if crossing else None,
                    "total_cpu_seconds": crossing["total_cpu_seconds"] if crossing else None,
                    "original_earliest_pass": first_original["passes"],
                    "original_earliest_total_cpu_seconds": first_original["total_cpu_seconds"],
                    "time_reduction_vs_earliest_original": reduction,
                    "time_reduction_30pct": reduction is not None and reduction >= .3,
                    "meaningful_threshold": threshold > 0,
                    "original_four_pass_below_earlier_peak": baseline["accuracy"] <
                        max(point["accuracy"] for point in original_points),
                })
            for point in row["checkpoints"]:
                affordable = [p for p in original_points
                              if p["total_cpu_seconds"] <= point["total_cpu_seconds"]]
                best = max((p["accuracy"] for p in affordable), default=None)
                gain = point["accuracy"] / best - 1 if best is not None and best > 0 else None
                report["equal_or_lower_cpu_checks"].append({
                    "seed": seed, "arm": row["arm"], "passes": point["passes"],
                    "accuracy": point["accuracy"], "original_best_affordable_accuracy": best,
                    "relative_accuracy_gain": gain, "accuracy_30pct": gain is not None and gain >= .3})
    original_mean = report["summary"]["4"]["original"]
    report["endpoint_criteria"] = {
        arm: {"relative_accuracy_gain": (values["accuracy"] / original_mean["accuracy"] - 1
                                        if original_mean["accuracy"] else None),
              "accuracy_30pct": values["accuracy"] >= 1.3 * original_mean["accuracy"]
                               if original_mean["accuracy"] else False,
              "compute_ratio": values["total_cpu_seconds"] / original_mean["total_cpu_seconds"],
              "note": "Accuracy threshold alone is not a compute-qualified win. Inspect per-seed results, "
                      "setup-inclusive CPU, nonzero baseline and measured crossing support."}
        for arm, values in report["summary"]["4"].items()}


def analyze_saved_run(directory: Path) -> dict:
    report_bytes = (directory / "report.json").read_bytes()
    report = json.loads(report_bytes)
    if report["status"] != "completed":
        raise ValueError("Only completed comparisons can be analyzed")
    inputs = json.loads((directory / "inputs.json").read_text())
    for label in ("training", "evaluation"):
        if digest(inputs[label].encode()) != report[label + "_sha256"]:
            raise ValueError(f"Saved {label} hash mismatch")
    verified = []
    for entry in report["runs"]:
        for point in (0, *report["passes"]):
            name = (f"{entry['seed']}-{entry['arm']}-initial.json" if not point else
                    f"{entry['seed']}-{entry['arm']}-passes-{point}.json")
            checkpoint_bytes = (directory / name).read_bytes()
            saved = json.loads(checkpoint_bytes)
            state = saved["state"]
            if entry["arm"] == "original":
                model = NeuralModel(state["characters"], seed=entry["seed"])
                model.weights = state["weights"]
            else:
                model = WordBase.restore(state)
            index_state = (saved.get("index") if point else inputs["indices"].get(entry["arm"]))
            index = CandidateIndex.restore(index_state) if index_state else None
            measured = evaluate(model, inputs["probes"], index)
            expected = (saved["evidence"] if point else entry["before"])
            for key in ("accuracy", "edit", "rows", "probes"):
                if measured[key] != expected[key]:
                    raise AssertionError(f"Saved {name}: {key} mismatch")
            stored = entry["before"] if not point else next(
                row for row in entry["checkpoints"] if row["passes"] == point)
            stored["evaluation_cpu_seconds_at_training"] = stored["evaluation_cpu_seconds"]
            stored["evaluation_cpu_seconds"] = measured["evaluation_cpu_seconds"]
            verified.append({"file": name, "sha256": digest(checkpoint_bytes),
                             "seed": entry["seed"], "arm": entry["arm"], "passes": point})
    if report["protected_chat_sha256_before"] != report["protected_chat_sha256_after"]:
        raise AssertionError("Chat isolation hashes differ")
    validation = {
        "run_id": report["run_id"], "verified_checkpoints_including_initial": len(verified),
        "verified_predictions": len(verified) * len(inputs["probes"]), "files": verified,
        "method": "Restore each saved state; regenerate every full greedy continuation and "
                  "compare all rows/metrics exactly. No full-size training replay claimed.",
        "report_sha256": digest(report_bytes),
        "analysis_code_sha256": digest(Path(__file__).read_bytes()),
        "analysis_note": "Raw training report/checkpoints remain unchanged. Derived analysis uses "
                         "earliest baseline crossings and also reports its best observed quality; "
                         "a collapsed final baseline is not a speed shortcut. Inference CPU is "
                         "remeasured on restored states using bounded character context lookup; "
                         "original instrumentation timings are retained. Output rows must match exactly. "
                         "Training schedules, weights, preparation and training CPU are unchanged.",
    }
    summarize(report)
    report["restoration_validation"] = validation
    atomic_json(directory / "validation.json", validation)
    atomic_json(directory / "analysis.json", report)
    return report


def run(root: Path, *, acknowledge_review: bool, seeds=SEEDS, passes=PASSES,
        include_retired_layers: bool = False) -> dict:
    if acknowledge_review is not True:
        raise ValueError("Explicit approval of full five-arm generation comparison is required")
    if tuple(passes) != PASSES or not seeds:
        raise ValueError("Protocol requires 1/2/4 source passes and at least one seed")
    protected = {str(path.relative_to(root)): digest(path.read_bytes())
                 for path in (root / ".data/neural-chat").glob("*.json")}
    start = time.process_time()
    train, held, items, provenance = load_inputs(root)
    source_cpu = time.process_time() - start
    arms = ARMS if include_retired_layers else ARMS[:2]
    bank, inventory, possibilities = {}, {}, set()
    bank_cpu = inventory_cpu = 0.0
    if include_retired_layers:
        start = time.process_time()
        bank = load_bank(root)
        bank_cpu = time.process_time() - start
        start = time.process_time()
        inventory_dir = root / ".data/possible-words-expanded-scowl-2020.12.07"
        inventory = json.loads((inventory_dir / "manifest.json").read_text())
        raw_words = (inventory_dir / "words.txt").read_bytes()
        if digest(raw_words) != inventory["words_sha256"]:
            raise ValueError("Expanded possibility inventory hash mismatch")
        possibilities = set(raw_words.decode().splitlines())
        if len(possibilities) != inventory["entries"]:
            raise ValueError("Possibility inventory entry count mismatch")
        inventory_cpu = time.process_time() - start
    candidates, extraction, prep = {}, {}, {}
    for depth, arm in enumerate(arms[2:]):
        start = time.process_time()
        candidates[arm] = CandidateIndex.build(bank, train, depth, possibilities=possibilities if depth else None)
        extraction[arm] = candidates[arm].extract(train)
        prep[arm] = time.process_time() - start + source_cpu
        if depth:
            prep[arm] += inventory_cpu
        if depth == 2:
            prep[arm] += bank_cpu
    run_id = uuid4().hex
    directory = root / ".data/neural-base-comparison" / run_id
    report = {
        "run_id": run_id, "status": "running", "seeds": list(seeds), "passes": list(passes),
        "arms": list(arms), "retired_layers_enabled": include_retired_layers,
        "rate": RATE, "source": provenance,
        "training_characters": len(train), "evaluation_characters": len(held),
        "training_sha256": digest(train.encode()), "evaluation_sha256": digest(held.encode()),
        "inventory": inventory, "preparation_cpu_seconds": prep, "source_cpu_seconds": source_cpu,
        "bank_load_cpu_seconds": bank_cpu, "inventory_load_cpu_seconds": inventory_cpu,
        "bank_sha256": digest(json.dumps(bank, sort_keys=True).encode()),
        "protected_chat_sha256_before": protected,
        "policy": "Fresh ORIGINAL unconditioned character MLP, context4, embedding8, hidden16, rate .1. "
                  "NEW is an experimental full-generation extension of the word/punctuation screen: "
                  "top64 normalized training word identities, exact case-preserving surface output tokens, "
                  "all training characters as fallback outputs, two-word/gap causal context; unfinished "
                  "fallback words use last-word/gap/two-character features. Original Unicode text preserved. "
                  "All outputs are generated, no oracle lexicon filtering, target lookup, forced word choice, "
                  "or future context. Words outside training alphabet map to unknown as inputs; cannot "
                  "emit characters absent from training. No universal arbitrary-character guarantee. "
                  "NEW vocabulary is shared across new arms, learned from full training text only. "
                  "Neural examples for layered arms are built from concatenated remainder (not whole "
                  "source) after exact CandidateIndex extraction. Existing L0 seeds, corrected unique L1, "
                  "and existing L2 repeated-block/hidden-backed payloads; placement counts trained each pass. "
                  "Recognized spans receive no neural spelling/word-target updates; placement learns them. "
                  "No initial pretraining, occurrence replay or protected injection in this comparison. "
                  "This changes both tokenization and neural features versus original; not punctuation-only. "
                  "Layers may split words; remainder is retokenized losslessly, including resulting fragments. "
                  "Neural word payloads are queued and emitted one character per placement event: the "
                  "existing selector can intervene inside them without consuming queued remainder. "
                  "All models train 1/2/4 complete source passes, shuffling their own examples seed+2 then "
                  "popping; target counts differ by encoding/extraction. Same seed does NOT mean original "
                  "and new share parameter values/shapes. New arms share identical starting weights. "
                  "Same archived evaluation offsets, common word parser, greedy 64-character continuations. "
                  "All vocabulary/OOV targets scored; exact case and punctuation-sensitive source retained. "
                  "CPU includes feature construction, training and selector updates; setup-inclusive totals "
                  "also charge source/index/inventory/bank loads. Historical bank/dictionary construction "
                  "excluded and explicitly not free; bank previously saw some evaluation text, so not an "
                  "unseen/generalization claim. Word probes/Acts reused development text. Three seeds, "
                  "fixed endpoints; 30% relative accuracy at comparable compute or 30% less time to "
                  "original four-pass accuracy are goals, not assumed outcomes. First measured crossings "
                  "only, coarse checkpoints, no interpolation or accuracy monotonicity assumption. "
                  "Generation CPU reported separately. No chat model or historical report pointer modified.",
        "code_sha256": {name: digest((Path(__file__).parent / name).read_bytes())
                        for name in ("neural_base_comparison.py", "neural_experiment.py",
                                     "neural_candidate_layers.py", "neural_word_context.py",
                                     "neural_index_pipeline.py", "neural_skeleton_runtime.py")},
        "runs": [],
    }
    if not include_retired_layers:
        report["policy"] = ("Working comparison: layer-free New Base versus preserved original character base. "
                            "No L0/L1/L2 construction, extraction, placement, dictionary or bank loading. "
                            "The layer clauses below describe only the optional historical mode. "
                            + report["policy"])
    atomic_json(directory / "protocol.json", report)
    atomic_json(directory / "inputs.json", {"training": train, "evaluation": held, "probes": items,
                                          "indices": {arm: index.snapshot() for arm, index in candidates.items()}})
    print(f"{len(arms)}-arm run {run_id}; {len(train)} train characters; {len(items)} unrestricted probes", flush=True)
    for seed_position, seed in enumerate(seeds):
        rotation = seed_position % len(arms)
        order = arms[rotation:] + arms[:rotation]
        for arm in order:
            setup = time.process_time()
            if arm == "original":
                model = NeuralModel(sorted(set(train)), seed=seed)
            else:
                model = WordBase(train, seed)
            index = CandidateIndex.restore(candidates[arm].snapshot()) if arm in candidates else None
            remainder = extraction[arm]["remainder"] if index else train
            examples = model.examples(remainder)
            preparation_cpu = time.process_time() - setup + prep.get(arm, source_cpu)
            snapshot = model.snapshot() if isinstance(model, WordBase) else {
                "characters": model.characters, "weights": model.weights}
            network = model.model if isinstance(model, WordBase) else model
            entry = {"seed": seed, "arm": arm, "parameters": sum(map(len, network.weights.values())),
                     "initial_weights_sha256": digest(json.dumps(network.weights, sort_keys=True).encode()),
                     "preparation_cpu_seconds": preparation_cpu,
                     "targets_per_pass": len(examples), "remainder_characters": len(remainder),
                     "layer_forms": Counter(str(v["layer"]) for v in index.codec.units.values()) if index else {},
                     "layer_occurrences": Counter(str(row["layer"]) for row in extraction[arm]["occurrences"])
                                          if index else {},
                     "checkpoints": []}
            atomic_json(directory / f"{seed}-{arm}-initial.json", {"state": snapshot})
            entry["before"] = {"passes": 0, "total_cpu_seconds": preparation_cpu,
                               **evaluate(model, items, index)}
            rng = random.Random(seed + 2)
            cpu = 0.0
            placement_updates = 0
            report["runs"].append(entry)
            for epoch in range(1, max(passes) + 1):
                start = time.process_time()
                if index:
                    placement_updates += index.learn(extraction[arm]["pieces"])
                sequence = list(range(len(examples)))
                rng.shuffle(sequence)
                while sequence:
                    example = examples[sequence.pop()]
                    if isinstance(model, WordBase):
                        context, target, prior = example
                        network.update(context, target, learning_rate=RATE, output_prior=prior)
                    else:
                        network.update(*example, learning_rate=RATE)
                cpu += time.process_time() - start
                if epoch not in passes:
                    continue
                row = {"passes": epoch, "training_cpu_seconds": cpu,
                       "total_cpu_seconds": preparation_cpu + cpu, "updates": epoch * len(examples),
                       "placement_count_updates": placement_updates,
                       "index_state_bytes": len(json.dumps(index.snapshot()).encode()) if index else 0,
                       **evaluate(model, items, index)}
                entry["checkpoints"].append(row)
                atomic_json(directory / f"{seed}-{arm}-passes-{epoch}.json", {
                    "state": model.snapshot() if isinstance(model, WordBase) else {
                        "characters": model.characters, "weights": model.weights},
                    "index": index.snapshot() if index else None, "evidence": row})
                atomic_json(directory / "report.json", report)
                print(f"{seed} {arm} pass {epoch}: accuracy={row['accuracy']:.5f}; "
                      f"CPU={row['total_cpu_seconds']:.2f}s; params={entry['parameters']}", flush=True)
    summarize(report)
    report["protected_chat_sha256_after"] = {
        str(path.relative_to(root)): digest(path.read_bytes())
        for path in (root / ".data/neural-chat").glob("*.json")}
    if report["protected_chat_sha256_after"] != protected:
        raise RuntimeError("Active chat files changed during the offline comparison")
    report["status"] = "completed"
    atomic_json(directory / "report.json", report)
    print(json.dumps({"run_id": run_id, "summary": report["summary"],
                      "endpoint": report["endpoint_criteria"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acknowledge-review", action="store_true")
    parser.add_argument("--analyze-run", help="Restore/verify completed run and derive time-to-quality analysis")
    parser.add_argument("--include-retired-layers", action="store_true",
                        help="Historical experiments only: include retired L0/L1/L2 arms")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.analyze_run:
        if len(args.analyze_run) != 32 or any(char not in "0123456789abcdef" for char in args.analyze_run):
            parser.error("--analyze-run requires a 32-character lowercase hex run ID")
        analysis = analyze_saved_run(root / ".data/neural-base-comparison" / args.analyze_run)
        print(json.dumps({"run_id": analysis["run_id"], "summary": analysis["summary"],
                          "validation": analysis["restoration_validation"]}))
    else:
        run(root, acknowledge_review=args.acknowledge_review,
            include_retired_layers=args.include_retired_layers)
