"""Review-gated corpus continuation with approved conversation rehearsal."""

import argparse
import fcntl
import json
import math
from pathlib import Path
import random
import time
from uuid import uuid4

from neural_active_training import APPROVED_SHA256
from neural_base_comparison import WordBase, digest
from neural_base_diagnostic import ambiguity
from neural_chat import CONVERSATION_MODEL, DIRECTORY as CHAT_DIRECTORY, load_active
from neural_continuation import extend_alphabet
from neural_conversation_meal import AUTHORED_FIXTURE, paired_scores, validate_curriculum
from neural_conversation_trial import answer_examples, assess, canonical, feedback_hash, lost_correct_ids, prompt_for, token_loss
from neural_plasticity import PlasticityContext, diagnostics
from neural_training import DIRECTORY, MAX_SATURATION, activate, load_progress
from neural_word_context import WORD
from original_generator import atomic_json
from seed_life import LAYER_ZERO_SEEDS, TERMINUS


CORPUS = "fixtures/neural-user-corpus-2026-11-14.json"
PREFIX = "text: "
CORPUS_RATE = .25


def corpus_records(data: dict) -> list[tuple[str, str]]:
    texts, fragments = data["training"], data.get("additional_fragments", [])
    seeds = data.get("layer_zero_seeds", [])
    if not isinstance(texts, list) or not texts or not isinstance(fragments, list) or not isinstance(seeds, list):
        raise ValueError("Corpus training items, fragments, and seeds must be lists")
    expected_seeds = [
        {"text": seed.removeprefix(TERMINUS),
         "boundary": "source_start" if seed.startswith(TERMINUS) else "interior",
         "source_seed": seed} for seed in LAYER_ZERO_SEEDS
    ]
    if seeds and seeds != expected_seeds:
        raise ValueError("Layer 0 seed records must preserve the exact historical forms and boundaries")
    records = ([(PREFIX, text) for text in texts]
               + [("" if seed["boundary"] == "source_start" else PREFIX.rstrip(), seed["text"])
                  for seed in seeds]
               + [(PREFIX, text) for text in fragments])
    if any(not isinstance(text, str) or not text.strip() or "\n" in text or "\r" in text
           or "\x00" in text or len(text) > 319 for _, text in records):
        raise ValueError("Corpus items must be nonempty single-line text up to 319 characters")
    return records


def corpus_examples(model: PlasticityContext, records: list[tuple[str, str]]) -> list:
    examples = []
    for boundary, text in records:
        prefix = boundary
        for token in model.encode(text + "\n"):
            prior = model.prior(prefix)
            if not prior[token]:
                raise ValueError("Corpus target is excluded by the causal output mask")
            examples.append((model.context(prefix), token, prior))
            prefix += model.payloads[token]
    return examples


def corpus_symbols(records: list[tuple[str, str]]) -> set[str]:
    sources = [prefix + text + "\n" for prefix, text in records]
    text = "".join(sources)
    return ({"char:" + char for char in text}
            | {"identity:" + match.group().lower().replace("’", "'") for match in WORD.finditer(text)}
            | {gap for record in sources for gap in WordBase.gaps(record)})


def corpus_score(model: PlasticityContext, examples: list) -> dict:
    correct = 0
    for context, target, prior in examples:
        probabilities = model.model.forward(context, output_prior=prior)[2]
        allowed = [i for i, value in enumerate(prior) if value]
        correct += max(allowed, key=lambda i: (probabilities[i], -i)) == target
    return {"targets": len(examples), "correct": correct, "loss": token_loss(model, examples)}


def advance(model: PlasticityContext, conversations: list, corpus: list,
            rng: random.Random, updates: int, rate: float) -> None:
    for _ in range(updates):
        examples = corpus if rng.random() < CORPUS_RATE else conversations
        context, target, prior = examples[rng.randrange(len(examples))]
        model.model.update(context, target, output_prior=prior, learning_rate=rate)


def milestone(model: PlasticityContext, teaching: list[dict], conversations: list,
              corpus: list, updates: int) -> dict:
    return {"updates": updates, "training": assess(model, teaching, no_triple_repeat=True),
            "corpus": corpus_score(model, corpus),
            "diagnostics": diagnostics(model, conversations + corpus)}


def run(root: Path, *, acknowledge_review: bool, reviewed_sha256: str,
        cpu_seconds: float = 60.0, rate: float = .02) -> dict:
    raw_corpus = (root / CORPUS).read_bytes()
    if acknowledge_review is not True or digest(raw_corpus) != reviewed_sha256:
        raise ValueError("Explicit approval of the exact corpus file hash is required")
    if (not math.isfinite(cpu_seconds) or not 1 <= cpu_seconds <= 900
            or not math.isfinite(rate) or not 0 < rate <= .1):
        raise ValueError("Invalid training budget or learning rate")
    data = json.loads(raw_corpus)
    records = corpus_records(data)
    raw_fixture = (root / AUTHORED_FIXTURE).read_bytes()
    if digest(raw_fixture) != APPROVED_SHA256:
        raise ValueError("Approved conversation curriculum differs from its pinned hash")
    curriculum = json.loads(raw_fixture)
    validate_curriculum(curriculum)
    directory = root / DIRECTORY
    with (directory / "training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Training is busy; retry when the current round finishes") from error
        active_path = root / CHAT_DIRECTORY / "active.json"
        progress_path = directory / "progress.json"
        active_bytes, progress_bytes = active_path.read_bytes(), progress_path.read_bytes()
        active, parent, index = load_active(root)
        progress, _ = load_progress(root)
        if (active["model"] != CONVERSATION_MODEL or not isinstance(parent, PlasticityContext)
                or index is not None or active["checkpoint_sha256"] != progress["checkpoint_sha256"]
                or progress.get("fixture") != AUTHORED_FIXTURE
                or progress.get("reviewed_file_sha256") != APPROVED_SHA256):
            raise ValueError("Active model and reviewed permanent progress must match")
        corrections = progress.get("approved_corrections", [])
        for item in corrections:
            if item["sha256"] != feedback_hash(item["row"]["turns"], item["row"]["answer"]):
                raise ValueError("Approved feedback hash mismatch")
        corrected = {canonical(item["row"]["turns"]) for item in corrections}
        feedback = [item["row"] for item in corrections]
        teaching = [row for row in curriculum["train"] if canonical(row["turns"]) not in corrected] + feedback
        before_training = assess(parent, teaching, no_triple_repeat=True)
        before_test = assess(parent, curriculum["test"], no_triple_repeat=True)
        before_feedback = assess(parent, feedback, no_triple_repeat=True) if feedback else None
        symbols = corpus_symbols(records)
        for row in feedback:
            text = prompt_for(row) + row["answer"] + "\n"
            symbols.update("char:" + char for char in text)
            symbols.update("identity:" + match.group().lower().replace("’", "'") for match in WORD.finditer(text))
        expanded = extend_alphabet(parent.model, sorted(symbols))
        initial = parent.snapshot()
        initial.update(symbols=expanded.characters, weights=expanded.weights)
        initial = json.loads(json.dumps(initial, allow_nan=False))
        model = PlasticityContext.restore(json.loads(json.dumps(initial)))
        conversations = answer_examples(model, teaching)
        corpus = corpus_examples(model, records)
        before_corpus = corpus_score(model, corpus)
        seed, run_id = 3000000 + progress["round"] + 1, uuid4().hex
        target = directory / "rounds" / run_id
        report = {
            "run_id": run_id, "round": progress["round"] + 1, "status": "running",
            "kind": "reviewed_corpus_continuation", "parent_checkpoint_sha256": progress["checkpoint_sha256"],
            "fixture": AUTHORED_FIXTURE, "reviewed_file_sha256": APPROVED_SHA256,
            "review": {"train_sha256": digest(canonical({"conversation": teaching, "corpus": records})),
                       "corpus_file_sha256": reviewed_sha256, "corpus_path": CORPUS,
                       "approval": "User approved the exact corpus text and boundary-aware historical seeds.",
                       "provenance": data["provenance"], "rights_note": data["rights_note"]},
            "seed": seed, "rate": rate, "cpu_seconds": cpu_seconds,
            "before": {"training": before_training, "test": before_test, "feedback": before_feedback,
                       "corpus_after_alphabet_expansion": before_corpus},
            "representation": ambiguity(corpus),
            "policy": "Each ordinary corpus item has a fixed input-only text: prefix; historical seeds use "
                      "their recorded source-start or interior boundaries without emitting the marker. "
                      "No fabricated conversation feedback. 25% of updates sample corpus positions and "
                      "75% sample approved conversation positions. Text and development are not "
                      "independent held-out evidence. Future ordinary rounds do not automatically replay "
                      "this corpus. Conversation evaluation uses the live three-identical-character "
                      "output filter. Normal retention, paired-development, saturation and replay gates apply.",
        }
        atomic_json(target / "protocol.json", report)
        atomic_json(target / "inputs.json", {"corpus": data, "teaching": teaching, "curriculum": curriculum})
        atomic_json(target / "parent.json", {"state": parent.snapshot(), "selection": active})
        atomic_json(target / "initial.json", {"state": initial})
        for name in ("neural_corpus_training.py", "neural_training.py", "neural_active_training.py", "seed_life.py",
                     "neural_plasticity.py", "neural_experiment.py", "neural_conversation_trial.py",
                     "neural_conversation_context.py", "neural_base_comparison.py", "neural_base_diagnostic.py",
                     "neural_continuation.py", "neural_word_context.py"):
            path = target / "source" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((Path(__file__).parent / name).read_bytes())
        print(f"Corpus continuation round {report['round']}: {run_id}, {cpu_seconds} update-CPU seconds", flush=True)
        rng = random.Random(seed)
        cpu, updates, points = 0.0, 0, []
        for budget in (cpu_seconds / 5, cpu_seconds):
            while cpu < budget:
                start = time.process_time()
                advance(model, conversations, corpus, rng, 100, rate)
                cpu += time.process_time() - start
                updates += 100
            point = milestone(model, teaching, conversations, corpus, updates)
            point["training_cpu_seconds"] = cpu
            points.append(point)
        state = json.loads(json.dumps(model.snapshot(), allow_nan=False))
        atomic_json(target / "trained.json", {"state": state, "checkpoints": points})
        after_test = assess(model, curriculum["test"], no_triple_repeat=True)
        after_feedback = assess(model, feedback, no_triple_repeat=True) if feedback else None
        replayed = PlasticityContext.restore(initial)
        replay_conversations = answer_examples(replayed, teaching)
        replay_corpus = corpus_examples(replayed, records)
        rng, count = random.Random(seed), 0
        for point in points:
            advance(replayed, replay_conversations, replay_corpus, rng, point["updates"] - count, rate)
            count = point["updates"]
            checked = milestone(replayed, teaching, replay_conversations, replay_corpus, count)
            if checked != {key: value for key, value in point.items() if key != "training_cpu_seconds"}:
                raise AssertionError("Corpus continuation milestone did not replay")
        if replayed.snapshot() != state or assess(replayed, curriculum["test"], no_triple_repeat=True) != after_test:
            raise AssertionError("Corpus continuation weights or development outputs did not replay")
        before_pairs, after_pairs = paired_scores(before_test), paired_scores(after_test)
        failures = []
        if any(point["diagnostics"]["hidden_abs_over_099_fraction"] > MAX_SATURATION for point in points):
            failures.append("hidden saturation exceeds limit")
        if points[-1]["training"]["exact"] < before_training["exact"]:
            failures.append("teaching accuracy regressed")
        lost_teaching = lost_correct_ids(before_training, points[-1]["training"])
        if lost_teaching:
            failures.append("previously correct approved teaching regressed")
        if after_pairs["both_exact"] < before_pairs["both_exact"]:
            failures.append("development paired accuracy regressed")
        lost_feedback = lost_correct_ids(before_feedback, after_feedback) if before_feedback else []
        if lost_feedback:
            failures.append("previously correct approved feedback regressed")
        report.update(checkpoints=points, test=after_test, test_pairs=after_pairs, feedback=after_feedback,
                      lost_correct_teaching_ids=lost_teaching, lost_correct_feedback_ids=lost_feedback,
                      validation={"full_weights_and_milestones_replay_exact": True, "test_replay_exact": True})
        if failures:
            report.update(status="rejected", rejection=failures)
            atomic_json(target / "report.json", report)
            raise RuntimeError("Corpus continuation rejected: " + "; ".join(failures))
        if active_path.read_bytes() != active_bytes or progress_path.read_bytes() != progress_bytes:
            raise RuntimeError("Active model or progress changed during training; candidate was not promoted")
        report["status"] = "completed"
        atomic_json(target / "report.json", report)
        evidence = {"run_id": run_id, "round": report["round"],
                    "parent_checkpoint_sha256": progress["checkpoint_sha256"],
                    "reviewed_file_sha256": APPROVED_SHA256, "fixture": AUTHORED_FIXTURE,
                    "report_sha256": digest((target / "report.json").read_bytes()),
                    "approved_corrections": corrections}
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
        print(f"Activated corpus continuation {run_id}: {checksum}", flush=True)
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acknowledge-review", action="store_true")
    parser.add_argument("--reviewed-sha256", required=True)
    parser.add_argument("--cpu-seconds", type=float, default=60.0)
    args = parser.parse_args()
    run(Path(__file__).resolve().parent, acknowledge_review=args.acknowledge_review,
        reviewed_sha256=args.reviewed_sha256, cpu_seconds=args.cpu_seconds)
