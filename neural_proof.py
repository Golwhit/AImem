"""Proof-shaping experiments for the developmental layers.

Development phase (Acts 1 training / Acts 2 validation only):
  training-size curve x seeds x arms, operation counts, generated-output rule-break rate.
Then: write pre-registered success criteria, stage Acts 4-5 as untouched data, evaluate once.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from uuid import uuid4

from neural_counting import KneserNey
from neural_experiment import EMBEDDING, HIDDEN, NeuralModel
from neural_layers import DevelopmentalPrior, evaluate, local_exclusion, probabilities
from neural_rounds import load_acts
from original_generator import atomic_json


FRACTIONS = (0.1, 0.25, 0.5, 1.0)
SEEDS = ((17, 19, 23), (117, 119, 123), (217, 219, 223), (317, 319, 323), (417, 419, 423))
STEPS = 5000
MEASURE_EVERY = 500
RATE = 0.03
CONTEXT = 4
SAMPLE_PROMPTS = ("The ", "They", "And ", "He s", "of t")
SAMPLE_LENGTH = 200
# name, kind, layer_zero, layer_one, layer_two, layer_zero_mode, neural
ARMS = (
    ("baseline", "layers", False, False, False, "v1", True),
    ("neural_12", "layers", False, True, True, "v1", True),
    ("neural_012_v1", "layers", True, True, True, "v1", True),
    ("neural_012_v2", "layers", True, True, True, "v2", True),
    ("counting_kn", "kn", False, False, False, "v1", False),
    ("layers_2", "layers", False, False, True, "v1", False),
    ("layers_12", "layers", False, True, True, "v1", False),
    ("layers_02_v2", "layers", True, False, True, "v2", False),
    ("layers_012_v1", "layers", True, True, True, "v1", False),
    ("layers_012_v2", "layers", True, True, True, "v2", False),
)
LAYERED = tuple(arm[0] for arm in ARMS if arm[0] not in ("baseline", "counting_kn"))
FRESH_CHAPTERS = ((4, 37), (5, 42))
FRESH_DIRECTORY = ".data/acts-fresh-staged"
PROOF_DIRECTORY = ".data/neural-proof"
CRITERIA = {
    "selection": "Primary layered configuration = the layered arm (any arm using Layer 0, 1 or 2, neural or counts-only) with the fewest mean estimated training operations among those whose mean Acts 2 validation loss at 100% data is within 0.02 nats of the best layered arm. Selected before any fresh text exists locally.",
    "selection_revision": "Revised after seeing development results but before fresh data existed: the first draft rule (lowest validation loss) selected neural_12, only ~0.005 nats better than counts-only paths at ~1,000x the operations, so C2 would have failed by construction rather than by evidence. The first-draft pick is evaluated and reported under the same criteria as a secondary arm.",
    "C1_similar_result": "Primary arm mean fresh-test loss <= baseline neural mean fresh-test loss + 0.02 nats (similar or better).",
    "C2_less_effort": "Primary arm mean estimated training operations <= 10% of baseline neural training operations AND mean training CPU seconds <= 25% of baseline.",
    "C3_beyond_standard_counting": "Primary arm mean fresh-test loss < counting_kn fresh-test loss - 0.02 nats. If not met, the speed/quality advantage is not shown to come from the developmental layers beyond standard counting.",
    "C4_layer_zero_helps": "layers_012_v2 fresh loss < layers_12 fresh loss - 0.01 nats (Layer 0 v2 adds value over Layers 1+2). Reported separately for v1.",
    "C5_layer_one_holds": "Every arm using Layer 1 has 0 generated rule violations, and fresh-test coverage stays 100% (no blocked original target). Arms without Layer 1 report their violation rate for comparison.",
    "C6_data_efficiency": "Smallest training fraction at which the primary arm's mean validation loss <= baseline mean validation loss at 100% data; on fresh test that fraction must also be <= baseline 100% fresh loss + 0.02 nats.",
    "honesty": "Criteria are operational predictions about next-character loss, effort and rule compliance on one related public-domain source. Meeting them is not evidence of understanding, general intelligence, or a general speedup.",
}


def mean_range(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values), "min": min(values), "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0, "n": len(values),
    }


def neural_forward_operations(model: NeuralModel, *, prior: bool) -> int:
    inputs = model.context * EMBEDDING
    return inputs + inputs * HIDDEN + HIDDEN + HIDDEN * model.size + (model.size if prior else 0) + 3 * model.size


def neural_update_operations(model: NeuralModel, *, prior: bool) -> int:
    inputs = model.context * EMBEDDING
    parameters = sum(map(len, model.weights.values()))
    backward = model.size + 2 * HIDDEN * model.size + HIDDEN + 2 * inputs * HIDDEN
    return neural_forward_operations(model, prior=prior) + backward + parameters


def build_prior(arm: tuple, train: str):
    _, kind, zero, one, two, mode, _ = arm
    if kind == "kn":
        return KneserNey(train)
    if zero or one or two:
        return DevelopmentalPrior(train, layer_zero=zero, layer_one=one, layer_two=two, layer_zero_mode=mode)
    return None


def priors_for(prior, text: str, model: NeuralModel) -> list[list[float]] | None:
    window = 4 if not isinstance(prior, KneserNey) else prior.order
    return [prior.prior(text[max(0, offset - window):offset], model) for offset in range(len(text))] if prior else None


def generate(model: NeuralModel, prior, neural: bool, seed: int) -> dict:
    """Rule-break rate: count scoped Layer 1 exclusions in generated characters only."""
    rng = random.Random(seed)
    violations = generated = 0
    texts = []
    for prompt in SAMPLE_PROMPTS:
        text = prompt
        for _ in range(SAMPLE_LENGTH):
            context = ([0] * model.context + [model.tokens.get(char, 1) for char in text])[-model.context:]
            weights = probabilities(model, context, prior.prior(text[-4:], model) if prior else None, neural=neural)
            if not any(weights[2:]):
                break
            char = model.characters[rng.choices(range(2, model.size), weights=weights[2:])[0] - 2]
            violations += local_exclusion(text[-3:], char) is not None
            generated += 1
            text += char
        texts.append({"prompt": prompt, "text": text[len(prompt):]})
    return {
        "generated_characters": generated, "rule_violations": violations,
        "violations_per_1000": 1000 * violations / generated if generated else None,
        "samples": texts[:2],
        "note": "Arms using Layer 1 have zero violations by construction (forbidden characters get zero probability); the comparison shows how often unconstrained arms would break the rules.",
    }


def run_task(task: dict) -> dict:
    arm = next(item for item in ARMS if item[0] == task["arm"])
    name, kind, zero, one, two, mode, neural = arm
    train = task["train"][:task["train_characters"]]
    held = task["held"]
    init_seed, draw_seed, sample_seed = task["seeds"]
    cpu_started, started = time.process_time(), time.perf_counter()
    model = NeuralModel(sorted(set(train)), seed=init_seed, context=CONTEXT)
    training = model.examples(train)
    prior = build_prior(arm, train)
    draws = []
    if neural:
        rng = random.Random(draw_seed)
        draws = [rng.randrange(len(train)) for _ in range(STEPS)]
    needed = sorted(set(draws))
    training_priors = {index: prior.prior(train[max(0, index - 4):index], model) for index in needed} if prior else {}
    preparation_cpu = time.process_time() - cpu_started
    preparation_seconds = time.perf_counter() - started
    preparation_operations = sum(prior.operations.values()) if prior else 0
    validation = model.examples(held)
    validation_priors = priors_for(prior, held, model)
    prior_operations_after_validation = sum(prior.operations.values()) if prior else 0
    history = []
    update_cpu = update_seconds = 0.0
    updates = skipped = 0
    update_operations = neural_update_operations(model, prior=prior is not None) if neural else 0

    def measure(step: int) -> None:
        metrics = evaluate(model, validation, validation_priors, neural=neural)
        history.append({
            "step": step, "gradient_updates": updates, "validation_loss": metrics["loss"],
            "coverage": metrics["coverage"], "blocked_targets": metrics["blocked_targets"],
        })

    measure(0)
    for step, index in enumerate(draws, 1):
        context, target = training[index]
        output_prior = training_priors.get(index)
        if output_prior is not None and output_prior[target] == 0:
            skipped += 1
        else:
            wall, cpu = time.perf_counter(), time.process_time()
            model.update(context, target, learning_rate=RATE, output_prior=output_prior)
            update_seconds += time.perf_counter() - wall
            update_cpu += time.process_time() - cpu
            updates += 1
        if step % MEASURE_EVERY == 0:
            if not all(math.isfinite(value) for values in model.weights.values() for value in values):
                raise ArithmeticError("Nonfinite neural weights")
            measure(step)
    training_metrics = evaluate(model, training, priors_for(prior, train, model), neural=neural)
    generation = generate(model, prior, neural, sample_seed)
    result = {
        "arm": name, "fraction": task["fraction"], "seed_index": task["seed_index"],
        "seeds": {"initialization": init_seed, "draws": draw_seed, "sampling": sample_seed},
        "neural": neural, "layers": [i for i, on in enumerate((zero, one, two)) if on] if kind == "layers" else [],
        "layer_zero_mode": mode if zero else None, "kind": kind,
        "training_characters": len(train), "vocabulary": len(model.characters),
        "training_sha256": hashlib.sha256(train.encode()).hexdigest(),
        "draws_sha256": hashlib.sha256(json.dumps(draws).encode()).hexdigest() if neural else None,
        "parameter_count": sum(map(len, model.weights.values())) if neural else 0,
        "candidate_count": len(prior.candidates) if isinstance(prior, DevelopmentalPrior) else 0,
        "gradient_updates": updates, "skipped_rule_conflicts": skipped,
        "training_preparation_seconds": preparation_seconds, "training_preparation_cpu_seconds": preparation_cpu,
        "update_seconds": update_seconds, "update_cpu_seconds": update_cpu,
        "training_cpu_seconds": preparation_cpu + update_cpu,
        "operations": {
            "preparation": preparation_operations,
            "updates": update_operations * updates,
            "training_total": preparation_operations + update_operations * updates,
            "per_update": update_operations,
            "validation_prediction_per_character": (
                (neural_forward_operations(model, prior=prior is not None) if neural else 0)
                + (prior_operations_after_validation - preparation_operations) / len(held)
            ),
            "breakdown": dict(prior.operations) if prior else {},
            "units": "Estimated basic operations: each multiply-add, exp/tanh, parameter update, count-table update/read, or rule lookup counts as 1. Formula-based, not hardware counters.",
        },
        "history": history,
        "final_validation_loss": history[-1]["validation_loss"],
        "final_coverage": history[-1]["coverage"],
        "training_loss": training_metrics["loss"],
        "generation": generation,
        "characters": model.characters,
        "weights": model.weights if neural else None,
    }
    return result


def target_match(history: list[dict], target: float | None) -> int | None:
    if target is None:
        return None
    return next((row["gradient_updates"] for row in history
                 if row["validation_loss"] is not None and row["validation_loss"] <= target), None)


def summarize(results: list[dict]) -> dict:
    baseline_full = [r["final_validation_loss"] for r in results if r["arm"] == "baseline" and r["fraction"] == 1.0]
    full_target = statistics.fmean(baseline_full) if baseline_full else None
    summary = {"baseline_100_mean_validation": full_target, "arms": {}}
    for name, *_ in ARMS:
        rows = {}
        for fraction in FRACTIONS:
            group = [r for r in results if r["arm"] == name and r["fraction"] == fraction]
            if not group:
                continue
            losses = [r["final_validation_loss"] for r in group]
            finite = [value for value in losses if value is not None]
            own = []
            for r in group:
                base = next((b for b in results if b["arm"] == "baseline" and b["fraction"] == fraction
                             and b["seed_index"] == r["seed_index"]), None)
                own.append(target_match(r["history"], base["final_validation_loss"]) if base else None)
            rows[str(fraction)] = {
                "validation": mean_range(finite) if len(finite) == len(losses) else None,
                "blocked_any": len(finite) != len(losses),
                "training_operations": mean_range([r["operations"]["training_total"] for r in group]),
                "training_cpu_seconds": mean_range([r["training_cpu_seconds"] for r in group]),
                "updates_to_match_own_baseline": own,
                "updates_to_match_baseline_100": [target_match(r["history"], full_target) for r in group],
                "rule_violations_per_1000": mean_range([r["generation"]["violations_per_1000"] for r in group]),
                "seed_count": len(group),
                "deterministic_across_seeds": not group[0]["neural"],
            }
        summary["arms"][name] = rows
    layered = [
        (summary["arms"][name]["1.0"]["validation"]["mean"], summary["arms"][name]["1.0"]["training_operations"]["mean"], name)
        for name in LAYERED if summary["arms"].get(name, {}).get("1.0", {}).get("validation")
    ]
    summary["primary_selection"] = min(layered)[2] if layered else None
    return summary


def run_development(root: Path, *, acknowledge_review: bool = False, workers: int = 10) -> dict:
    dataset = load_acts(root, acknowledge_review, include_final_test=False)
    train, held = dataset["training"], dataset["validation"]
    run_id = uuid4().hex
    directory = root / PROOF_DIRECTORY / run_id
    report = {
        "version": 1, "kind": "layer-proof-development", "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "running",
        "fractions": FRACTIONS, "seeds": SEEDS, "steps_per_neural_arm": STEPS,
        "measurement_interval": MEASURE_EVERY, "learning_rate": RATE, "context": CONTEXT,
        "arms": [{"name": a[0], "kind": a[1], "layers": [i for i, on in enumerate(a[2:5]) if on] if a[1] == "layers" else [],
                  "layer_zero_mode": a[5] if a[2] else None, "neural": a[6]} for a in ARMS],
        "training_sha256": hashlib.sha256(train.encode()).hexdigest(),
        "held_out_sha256": hashlib.sha256(held.encode()).hexdigest(),
        "data_provenance": {
            "manifest_sha256": dataset["manifest_sha256"],
            "approval": "User: 'run the whole set' (standard counting comparison, training-size curve, seeds, operation counts, Layer 0 rework, generated rule-break rate, fresh Acts 4-5 test with pre-registered criteria).",
        },
        "policy": {
            "fractions": "Training uses the first 10/25/50/100% of Acts 1 characters; vocabulary comes from that subset only; Acts 2 validation is unchanged and unfiltered.",
            "seeds": "Five (initialization, draw, sampling) seed triples. Seed triple 1 at 100% reproduces the earlier integrated run's baseline and full_012 settings. Counts-only arms are deterministic; their seed repeats only vary generation sampling.",
            "layer_zero_v2": "Layer 0 v2 keeps the same foundation and candidates but treats an unseen 3-4 character block as possible when its overlapping smaller pieces are possible; only single characters and pairs must be observed. Not-possible next characters keep 0.05 exploration (not impossibility).",
            "counting_kn": "Standard interpolated Kneser-Ney (order 3 context, discount 0.75 fixed, not tuned) written from scratch as the comparison a skeptic would ask for.",
            "selection": "No checkpoint or arm is chosen by Acts 2 except the pre-registered primary selection rule; Acts 3 is never read.",
        },
        "results": [], "summary": None,
    }

    def save() -> None:
        atomic_json(directory / "report.json", {**report, "results": [
            {key: value for key, value in result.items() if key not in ("weights", "characters")}
            for result in report["results"]
        ]})
        atomic_json(root / ".data/neural-experiments/latest-proof.json", public_report(root))

    tasks = []
    for fraction in FRACTIONS:
        for seed_index, seeds in enumerate(SEEDS):
            for arm in ARMS:
                tasks.append({
                    "arm": arm[0], "fraction": fraction, "seed_index": seed_index, "seeds": seeds,
                    "train": train, "held": held, "train_characters": round(len(train) * fraction),
                })
    save()
    started = time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(run_task, tasks):
                atomic_json(directory / "models" / f"{result['arm']}-f{int(result['fraction'] * 100):03}-s{result['seed_index']}.json", {
                    "arm": result["arm"], "fraction": result["fraction"], "seed_index": result["seed_index"],
                    "seeds": result["seeds"], "context": CONTEXT, "characters": result["characters"],
                    "weights": result["weights"], "training_sha256": result["training_sha256"],
                })
                report["results"].append(result)
                print(f"{result['arm']} f={result['fraction']} s={result['seed_index']}: "
                      f"validation={result['final_validation_loss']}", file=sys.stderr, flush=True)
                if len(report["results"]) % 20 == 0:
                    save()
        report["summary"] = summarize(report["results"])
        report["wall_seconds"] = time.perf_counter() - started
        report["status"] = "completed"
    except (ValueError, ArithmeticError, OSError) as error:
        report.update(status="failed", error=str(error))
        save()
        raise
    save()
    return report


def select_primary(summary: dict) -> str:
    rows = [(summary["arms"][name]["1.0"]["validation"]["mean"],
             summary["arms"][name]["1.0"]["training_operations"]["mean"], name)
            for name in LAYERED if summary["arms"][name]["1.0"]["validation"]]
    best = min(row[0] for row in rows)
    return min((ops, loss, name) for loss, ops, name in rows if loss <= best + 0.02)[2]


def preregister(root: Path, run_id: str) -> dict:
    proof = root / PROOF_DIRECTORY
    path = proof / "preregistration.json"
    if path.exists():
        raise ValueError("Pre-registration already exists; criteria cannot be rewritten")
    if (root / FRESH_DIRECTORY).exists():
        raise ValueError("Fresh data already exists locally; pre-registration must come first")
    report_bytes = (proof / run_id / "report.json").read_bytes()
    report = json.loads(report_bytes)
    if report.get("status") != "completed":
        raise ValueError("Development run is not completed")
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "development_run_id": run_id,
        "development_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "primary_arm": select_primary(report["summary"]),
        "secondary_arm_first_draft_rule": report["summary"]["primary_selection"],
        "criteria": CRITERIA,
        "fresh_data": "Acts 4 and Acts 5 (World English Bible, eBible engwebp), not yet downloaded at registration time.",
        "evaluations_allowed": 1,
        "approval": "User: 'run the whole set'. Criteria drafted by the assistant and frozen here before fresh data was retrieved.",
    }
    atomic_json(path, record)
    atomic_json(root / ".data/neural-experiments/latest-proof.json", public_report(root))
    return record


class Verses(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.verses: dict[int, list[str]] = {}
        self.current = None
        self.skip = 0
        self.finished = False

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if self.skip:
            self.skip += 1
            return
        if tag == "ul" or attributes.get("class") in ("footnote", "copyright"):
            self.current = None
            if attributes.get("class") in ("footnote", "copyright"):
                self.finished = True
        if tag == "a" or (tag == "span" and attributes.get("class") == "verse"):
            if attributes.get("class") == "verse" and not self.finished:
                verse_id = attributes.get("id", "")
                if not verse_id.startswith("V") or not verse_id[1:].isdigit():
                    raise ValueError("Unexpected verse marker")
                self.current = int(verse_id[1:])
                self.verses[self.current] = []
            self.skip = 1

    def handle_endtag(self, tag):
        if self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if self.current is not None and not self.skip and not self.finished:
            self.verses[self.current].append(data)


def extract(raw: bytes, chapter: int, count: int) -> str:
    parser = Verses()
    parser.feed(raw.decode("utf-8-sig"))
    if sorted(parser.verses) != list(range(1, count + 1)):
        raise ValueError(f"Unexpected verse inventory for Acts {chapter}")
    verses = [" ".join("".join(parts).split()) for _, parts in sorted(parser.verses.items())]
    if not all(verses):
        raise ValueError("Empty verse extracted")
    return "\n".join(verses) + "\n"


def fetch(url: str) -> bytes:
    return subprocess.run(
        ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "30", url],
        check=True, capture_output=True,
    ).stdout


def stage_fresh(root: Path, fetcher=fetch) -> dict:
    if not (root / PROOF_DIRECTORY / "preregistration.json").exists():
        raise ValueError("Pre-register criteria before retrieving fresh data")
    directory = root / FRESH_DIRECTORY
    if directory.exists():
        raise ValueError("Fresh dataset already staged; refusing to overwrite")
    old = json.loads((root / ".data/acts-web-staged/manifest.json").read_text())
    rights = fetcher(old["rights_url"])
    texts, sources = [], []
    for chapter, count in FRESH_CHAPTERS:
        url = f"https://ebible.org/engwebp/ACT{chapter:02}.htm"
        raw = fetcher(url)
        text = extract(raw, chapter, count)
        texts.append((chapter, raw, text))
        sources.append({
            "chapter": chapter, "role": "fresh_test", "url": url, "verses": count,
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "characters": len(text),
        })
    directory.mkdir(parents=True)
    (directory / "rights.html").write_bytes(rights)
    for chapter, raw, text in texts:
        (directory / f"acts-{chapter:02}.html").write_bytes(raw)
        (directory / f"fresh-{chapter:02}.txt").write_text(text, encoding="utf-8")
    manifest = {
        "translation": old["translation"], "rights_url": old["rights_url"],
        "rights_statement": old["rights_statement"],
        "rights_sha256": hashlib.sha256(rights).hexdigest(),
        "rights_matches_original_snapshot": hashlib.sha256(rights).hexdigest() == old["rights_sha256"],
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "status": "fresh test only; never trained, never used for selection; evaluated at most once",
        "preprocessing": old["preprocessing"], "sources": sources,
    }
    atomic_json(directory / "manifest.json", manifest)
    return manifest


def load_fresh(root: Path) -> tuple[str, dict, str]:
    directory = root / FRESH_DIRECTORY
    manifest_bytes = (directory / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if hashlib.sha256((directory / "rights.html").read_bytes()).hexdigest() != manifest["rights_sha256"]:
        raise ValueError("Fresh rights record hash mismatch")
    parts = []
    for (chapter, _), source in zip(FRESH_CHAPTERS, manifest["sources"], strict=True):
        if source["chapter"] != chapter:
            raise ValueError("Unexpected fresh chapter")
        if hashlib.sha256((directory / f"acts-{chapter:02}.html").read_bytes()).hexdigest() != source["raw_sha256"]:
            raise ValueError(f"Acts {chapter} raw hash mismatch")
        text = (directory / f"fresh-{chapter:02}.txt").read_bytes().decode("utf-8")
        if hashlib.sha256(text.encode()).hexdigest() != source["text_sha256"]:
            raise ValueError(f"Acts {chapter} text hash mismatch")
        parts.append(text)
    return "".join(parts), manifest, hashlib.sha256(manifest_bytes).hexdigest()


def fresh_task(task: dict) -> dict:
    arm = next(item for item in ARMS if item[0] == task["arm"])
    train = task["train"][:task["train_characters"]]
    saved = task["saved"]
    model = NeuralModel(saved["characters"], seed=saved["seeds"]["initialization"], context=CONTEXT)
    if arm[6]:
        model.weights = saved["weights"]
    prior = build_prior(arm, train)
    metrics = evaluate(model, model.examples(task["fresh"]), priors_for(prior, task["fresh"], model), neural=arm[6])
    return {"arm": arm[0], "fraction": task["fraction"], "seed_index": task["seed_index"],
            "fresh_loss": metrics["loss"], "coverage": metrics["coverage"],
            "blocked_targets": metrics["blocked_targets"]}


def evaluate_fresh(root: Path, *, acknowledge_review: bool = False, workers: int = 10) -> dict:
    proof = root / PROOF_DIRECTORY
    output = proof / "fresh-evaluation.json"
    if output.exists():
        raise ValueError("Fresh test was already evaluated once; refusing a second evaluation")
    registration = json.loads((proof / "preregistration.json").read_text())
    run_id = registration["development_run_id"]
    report_bytes = (proof / run_id / "report.json").read_bytes()
    if hashlib.sha256(report_bytes).hexdigest() != registration["development_report_sha256"]:
        raise ValueError("Development report changed after pre-registration")
    development = json.loads(report_bytes)
    dataset = load_acts(root, acknowledge_review, include_final_test=False)
    fresh, manifest, manifest_sha = load_fresh(root)
    atomic_json(output, {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()})
    tasks = []
    for result in development["results"]:
        saved = json.loads((proof / run_id / "models" / f"{result['arm']}-f{int(result['fraction'] * 100):03}-s{result['seed_index']}.json").read_text())
        if result["neural"] or result["seed_index"] == 0:
            tasks.append({"arm": result["arm"], "fraction": result["fraction"], "seed_index": result["seed_index"],
                          "saved": saved, "train": dataset["training"], "fresh": fresh,
                          "train_characters": result["training_characters"]})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(fresh_task, tasks))
    record = {
        "status": "completed", "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "development_run_id": run_id, "fresh_manifest_sha256": manifest_sha,
        "fresh_characters": len(fresh), "rows": rows,
        "verdicts": verdicts(registration, development, rows),
        "note": "Single evaluation. Counts-only arms are deterministic, so only seed 1 is evaluated for them.",
    }
    atomic_json(output, record)
    atomic_json(root / ".data/neural-experiments/latest-proof.json", public_report(root))
    return record


def verdicts(registration: dict, development: dict, rows: list[dict]) -> dict:
    out = arm_verdicts(registration["primary_arm"], development, rows)
    secondary = registration.get("secondary_arm_first_draft_rule")
    if secondary and secondary != registration["primary_arm"]:
        out["secondary"] = arm_verdicts(secondary, development, rows)
    return out


def arm_verdicts(primary: str, development: dict, rows: list[dict]) -> dict:
    def fresh(arm: str, fraction: float = 1.0) -> float | None:
        values = [r["fresh_loss"] for r in rows if r["arm"] == arm and r["fraction"] == fraction]
        return None if not values or any(v is None for v in values) else statistics.fmean(values)

    summary = development["summary"]
    arms = summary["arms"]
    base_ops = arms["baseline"]["1.0"]["training_operations"]["mean"]
    base_cpu = arms["baseline"]["1.0"]["training_cpu_seconds"]["mean"]
    primary_fresh, base_fresh, kn_fresh = fresh(primary), fresh("baseline"), fresh("counting_kn")
    out = {"primary_arm": primary, "fresh_loss": {
        name: {str(f): fresh(name, f) for f in FRACTIONS} for name, *_ in ARMS}}
    out["C1_similar_result"] = primary_fresh is not None and primary_fresh <= base_fresh + 0.02
    out["C2_less_effort"] = (arms[primary]["1.0"]["training_operations"]["mean"] <= 0.10 * base_ops
                             and arms[primary]["1.0"]["training_cpu_seconds"]["mean"] <= 0.25 * base_cpu)
    out["C3_beyond_standard_counting"] = primary_fresh is not None and primary_fresh < kn_fresh - 0.02
    l12, v1, v2 = fresh("layers_12"), fresh("layers_012_v1"), fresh("layers_012_v2")
    out["C4_layer_zero_helps"] = {"v2": v2 is not None and v2 < l12 - 0.01, "v1": v1 is not None and v1 < l12 - 0.01}
    layer_one_arms = [a[0] for a in ARMS if a[1] == "layers" and a[3]]
    violations = sum(
        r["generation"]["rule_violations"] for r in development["results"] if r["arm"] in layer_one_arms
    )
    full_coverage = all(r["coverage"] == 1 for r in rows if r["arm"] in layer_one_arms)
    out["C5_layer_one_holds"] = violations == 0 and full_coverage
    target = summary["baseline_100_mean_validation"]
    fraction = next((f for f in FRACTIONS if (v := arms[primary][str(f)]["validation"]) and v["mean"] <= target), None)
    out["C6_data_efficiency"] = {
        "smallest_fraction": fraction,
        "met": fraction is not None and (pf := fresh(primary, fraction)) is not None and pf <= base_fresh + 0.02,
    }
    return out


def public_report(root: Path) -> dict:
    """Small UI summary: never includes weights, fresh text or per-step weights."""
    proof = root / PROOF_DIRECTORY
    runs = sorted(proof.glob("*/report.json"), key=lambda path: path.stat().st_mtime)
    if not runs:
        return {"status": "absent"}
    report = json.loads(runs[-1].read_text())
    for name in ("preregistration.json", "fresh-evaluation.json"):
        path = proof / name
        report[name.removesuffix(".json").replace("-", "_")] = json.loads(path.read_text()) if path.exists() else None
    results = report.pop("results", [])
    report["result_count"] = len(results)
    report["examples"] = [
        {"arm": r["arm"], "fraction": r["fraction"], "samples": r["generation"]["samples"][:1],
         "violations_per_1000": r["generation"]["violations_per_1000"]}
        for r in results if r["seed_index"] == 0 and r["fraction"] == 1.0
    ]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("develop", "preregister", "stage-fresh", "evaluate-fresh"))
    parser.add_argument("--acknowledge-review", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.phase == "develop":
        output = run_development(root, acknowledge_review=args.acknowledge_review, workers=args.workers)
        output = {"run_id": output["run_id"], "summary": output["summary"]}
    elif args.phase == "preregister":
        output = preregister(root, args.run_id)
    elif args.phase == "stage-fresh":
        output = stage_fresh(root)
    else:
        output = evaluate_fresh(root, acknowledge_review=args.acknowledge_review, workers=args.workers)
    print(json.dumps(output, allow_nan=False, indent=1))
