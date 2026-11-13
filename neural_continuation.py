"""Continue the paired Acts models on explicitly approved WEB Corinthians text."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from uuid import uuid4

from neural_candidate_layers import CandidateIndex, CandidateCodec
from neural_chat import load_active, restore
from neural_experiment import NeuralModel, EMBEDDING, HIDDEN, RATE
from neural_index_pipeline import evaluate, reinforced_schedule
from neural_proof import Verses, fetch
from neural_unit_benchmark import DIRECTORY, probes, text_hash
from original_generator import atomic_json


CORPUS = ".data/corinthians-web-staged"
BOOKS = {
    "1CO": (31, 16, 23, 21, 13, 20, 40, 13, 27, 33, 34, 31, 13, 40, 58, 24),
    "2CO": (24, 17, 18, 18, 21, 18, 16, 24, 15, 18, 33, 21, 14),
}
APPROVAL = ("User requested feeding 1st and 2nd Corinthians, then chose: Continue both layered and "
            "baseline models for four passes, compare results, then activate the updated layered model.")


def chapter_text(raw: bytes, count: int) -> str:
    parser = Verses()
    parser.feed(raw.decode("utf-8-sig"))
    if sorted(parser.verses) != list(range(1, count + 1)):
        raise ValueError("Corinthians chapter has an incomplete/unexpected verse inventory")
    verses = [" ".join("".join(parts).split()) for _, parts in sorted(parser.verses.items())]
    if not all(verses):
        raise ValueError("Corinthians source contains an empty verse")
    return "\n".join(verses) + "\n"


def stage(root: Path, *, acknowledge_review: bool, fetcher=fetch) -> dict:
    if acknowledge_review is not True:
        raise ValueError("Explicit training/source approval is required")
    directory = root / CORPUS
    if directory.exists():
        raise ValueError("Corinthians staging directory already exists; refusing to overwrite")
    old = json.loads((root / ".data/acts-web-staged/manifest.json").read_text())
    rights = fetcher(old["rights_url"])
    if b"The World English Bible is in the Public Domain." not in rights:
        raise ValueError("Publisher rights statement needs fresh review")
    files, sources = [], []
    for book, counts in BOOKS.items():
        for chapter, count in enumerate(counts, 1):
            name = f"{book}{chapter:02}"
            url = f"https://ebible.org/engwebp/{name}.htm"
            raw = fetcher(url)
            text = chapter_text(raw, count)
            files.append((name, raw, text))
            sources.append({"book": book, "chapter": chapter, "verses": count, "url": url,
                            "raw_sha256": hashlib.sha256(raw).hexdigest(),
                            "text_sha256": text_hash(text), "characters": len(text)})
    directory.mkdir(parents=True)
    (directory / "rights.html").write_bytes(rights)
    for name, raw, text in files:
        (directory / f"{name}.html").write_bytes(raw)
        (directory / f"{name}.txt").write_text(text, encoding="utf-8")
    manifest = {
        "translation": old["translation"], "rights_url": old["rights_url"],
        "rights_statement": old["rights_statement"], "rights_sha256": hashlib.sha256(rights).hexdigest(),
        "preprocessing": old["preprocessing"], "approval": APPROVAL,
        "retrieved_at": datetime.now(timezone.utc).isoformat(), "sources": sources,
    }
    atomic_json(directory / "manifest.json", manifest)
    return manifest


def load_corpus(root: Path) -> tuple[str, dict]:
    directory = root / CORPUS
    raw_manifest = (directory / "manifest.json").read_bytes()
    manifest = json.loads(raw_manifest)
    if hashlib.sha256((directory / "rights.html").read_bytes()).hexdigest() != manifest["rights_sha256"]:
        raise ValueError("Corinthians rights hash mismatch")
    expected = [(book, chapter, count) for book, counts in BOOKS.items()
                for chapter, count in enumerate(counts, 1)]
    if [(row["book"], row["chapter"], row["verses"]) for row in manifest["sources"]] != expected:
        raise ValueError("Corinthians manifest must contain both complete books in order")
    texts = []
    for row in manifest["sources"]:
        name = f"{row['book']}{row['chapter']:02}"
        raw = (directory / f"{name}.html").read_bytes()
        text = (directory / f"{name}.txt").read_text(encoding="utf-8")
        if (hashlib.sha256(raw).hexdigest() != row["raw_sha256"]
                or text_hash(text) != row["text_sha256"]
                or chapter_text(raw, row["verses"]) != text):
            raise ValueError(f"Corinthians source integrity failure: {name}")
        texts.append(text)
    return "\n".join(texts), {**manifest, "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest()}


def extend_alphabet(model: NeuralModel, characters: list[str]) -> NeuralModel:
    """Append tokens without renumbering old tokens or resetting learned parameters."""
    added = sorted(set(characters) - set(model.characters))
    if not added:
        return model
    expanded = NeuralModel(model.characters + added, context=model.context)
    old = model.weights
    expanded.weights = {
        "embedding": old["embedding"][:] + old["embedding"][EMBEDDING:2 * EMBEDDING] * len(added),
        "hidden": old["hidden"][:], "hidden_bias": old["hidden_bias"][:],
        "output": [value for j in range(HIDDEN)
                   for value in (old["output"][j * model.size:(j + 1) * model.size]
                                 + [old["output"][j * model.size + 1]] * len(added))],
        "output_bias": old["output_bias"][:] + [old["output_bias"][1]] * len(added),
    }
    return expanded


def extend_index(parent: CandidateIndex, prior_text: str, training: str,
                 possibilities: set[str]) -> CandidateIndex:
    candidate = CandidateIndex.build({}, prior_text + "\n" + training, 1, possibilities=possibilities)
    eligible = {entry["text"]: entry for entry in candidate.codec.units.values() if entry["layer"] == 1}
    units = deepcopy(parent.codec.units)
    for entry in units.values():
        if entry["layer"] == 1:
            if entry["text"] not in eligible:
                raise ValueError("New training words make an existing L1 identifier ambiguous")
            entry["evidence"] = eligible[entry["text"]]["evidence"]
    known = {entry["text"] for entry in units.values() if entry["layer"] == 1}
    for word in sorted(eligible.keys() - known):
        units[f"\x00layer:1:{len(units)}"] = eligible[word]
    index = CandidateIndex(CandidateCodec(units, {
        **parent.codec.provenance, "continuation": "Revalidate over prior plus new training words; "
        "preserve existing reference IDs and counts; append eligible L1 payloads only.",
    }), parent.context)
    index.counts = deepcopy(parent.counts)
    return index


def run(root: Path, *, acknowledge_review: bool) -> dict:
    if acknowledge_review is not True:
        raise ValueError("Explicit continuation training approval is required")
    active, layered_model, parent_index = load_active(root)
    parent_directory = root / DIRECTORY / active["run_id"]
    parent_report = json.loads((parent_directory / "report.json").read_text())
    inputs = json.loads((parent_directory / "inputs.json").read_text())
    prior = inputs["training"]
    if text_hash(prior) != parent_report["source"]["training_sha256"]:
        raise ValueError("Parent training text hash mismatch")
    seed, parent_passes = active["seed"], active["passes"]
    parent_hashes = {}
    models = {}
    for arm in ("baseline", "layers_01"):
        raw = (parent_directory / f"{seed}-{arm}-passes-{parent_passes}.json").read_bytes()
        saved = json.loads(raw)
        entry = next(row for row in parent_report["runs"] if row["arm"] == arm and row["seed"] == seed)
        expected = next(row for row in entry["checkpoints"] if row["passes"] == parent_passes)
        if saved["evidence"] != expected:
            raise ValueError("Parent checkpoint evidence mismatch")
        parent_hashes[arm] = hashlib.sha256(raw).hexdigest()
        if arm == "layers_01":
            if parent_hashes[arm] != active["checkpoint_sha256"]:
                raise ValueError("Active model differs from saved paired parent")
            models[arm] = layered_model
        else:
            if saved["index"] is not None or saved["characters"] != layered_model.characters:
                raise ValueError("Parent baseline must have the paired character alphabet and no index")
            # Use the same weight validator as layered restore without changing baseline weights.
            models[arm], _ = restore({**saved, "index": parent_index.snapshot()})
    training, manifest = load_corpus(root)
    inventory_dir = root / ".data/possible-words-expanded-scowl-2020.12.07"
    inventory = json.loads((inventory_dir / "manifest.json").read_text())
    words = (inventory_dir / "words.txt").read_bytes()
    if hashlib.sha256(words).hexdigest() != inventory["words_sha256"]:
        raise ValueError("Possibility inventory hash mismatch")
    possibilities = set(words.decode("utf-8").splitlines())
    if len(possibilities) != inventory["entries"]:
        raise ValueError("Possibility inventory count mismatch")
    index = extend_index(parent_index, prior, training, possibilities)
    extracted = index.extract(training)
    new_probes, retention = probes(training, 256), probes(prior, 256)
    run_id = uuid4().hex
    directory = root / DIRECTORY / run_id
    directory.mkdir(parents=True)
    source = {
        "training": "Acts 1-28, then 1 Corinthians 1-16 and 2 Corinthians 1-13",
        "training_sha256": text_hash(prior + "\n" + training),
        "continuation_training_sha256": text_hash(training),
        "evaluation": "Corinthians IN-SAMPLE learning checks and Acts retention checks",
        "corinthians": manifest, "parent_source": active["source"],
        "approval": APPROVAL,
    }
    report = {
        "run_id": run_id, "status": "running", "source": source,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": {"matched_reinforcement": True, "seeds": [seed], "passes": [1, 2, 4],
                     "learning_rate": RATE},
        "continuation": {"parent_run_id": active["run_id"], "parent_passes": parent_passes,
                         "parent_active_checkpoint_sha256": active["checkpoint_sha256"],
                         "parent_checkpoint_sha256": parent_hashes,
                         "new_characters": sorted(set(training) - set(layered_model.characters)),
                         "previous_l1_forms": sum(e["layer"] == 1 for e in parent_index.codec.units.values()),
                         "current_l1_forms": sum(e["layer"] == 1 for e in index.codec.units.values()),
                         "training_characters": len(training), "remainder_characters": len(extracted["remainder"]),
                         "occurrences": len(extracted["occurrences"]), "inventory": inventory,
                         "l0_occurrences": sum(row["layer"] == 0 for row in extracted["occurrences"]),
                         "l1_occurrences": sum(row["layer"] == 1 for row in extracted["occurrences"]),
                         "policy": "Continue trained weights, no reset or repeated initial pretraining. "
                                   "Four source-order passes; identical isolated-literal refresh per indexed "
                                   "occurrence in both arms. New alphabet entries copy each arm's unknown row, "
                                   "preserving old weights/tokens; added outputs change normalization. "
                                   "No Acts replay; retention measured, not guaranteed. One paired seed, "
                                   "in-sample checks only; not a multi-seed generalization claim."},
        "runs": [],
        "code_sha256": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("neural_continuation.py", "neural_experiment.py", "neural_candidate_layers.py",
                         "neural_index_pipeline.py", "neural_skeleton_runtime.py", "neural_unit_benchmark.py")
        },
    }
    atomic_json(directory / "protocol.json", report)
    atomic_json(directory / "report.json", report)
    atomic_json(directory / "inputs.json", {"training": prior + "\n" + training,
                                          "continuation_training": training, "probes": new_probes,
                                          "retention_probes": retention, "index": index.snapshot()})
    print(f"Continuation run {run_id}: {len(training)} characters; "
          f"{len(extracted['occurrences'])} refresh events/pass; "
          f"{report['continuation']['current_l1_forms']} L1 forms", flush=True)
    for arm in ("baseline", "layers_01"):
        model = extend_alphabet(models[arm], list(set(training)))
        selected = index if arm == "layers_01" else None
        schedule = reinforced_schedule(model, training, extracted, remainder=selected is not None)
        entry = {"arm": arm, "seed": seed, "parameters": sum(map(len, model.weights.values())),
                 "starting_weights_sha256": text_hash(json.dumps(model.weights, sort_keys=True)),
                 "before": {"corinthians": evaluate(model, selected, new_probes),
                            "acts_retention": evaluate(model, selected, retention)},
                 "checkpoints": []}
        report["runs"].append(entry)
        ordinary = sum(not refresh for _, _, refresh in schedule)
        replay = len(schedule) - ordinary
        atomic_json(directory / f"{seed}-{arm}-continuation-start.json", {
            "characters": model.characters, "weights": model.weights,
            "index": selected.snapshot() if selected else None, "evidence": entry["before"]})
        cpu = wall = selector_cpu = 0.0
        for passes in range(1, 5):
            started_cpu, started_wall = time.process_time(), time.perf_counter()
            if selected:
                started = time.process_time()
                selected.learn(extracted["pieces"])
                selector_cpu += time.process_time() - started
            for context, target, _ in schedule:
                model.update(context, target, learning_rate=RATE)
            cpu += time.process_time() - started_cpu
            wall += time.perf_counter() - started_wall
            if passes not in (1, 2, 4):
                continue
            row = {"passes": passes, "updates": ordinary * passes, "reinforcement_updates": replay * passes,
                   "total_training_cpu_seconds": cpu, "selector_cpu_seconds": selector_cpu,
                   "training_wall_seconds": wall,
                   **evaluate(model, selected, new_probes),
                   "acts_retention": evaluate(model, selected, retention)}
            entry["checkpoints"].append(row)
            atomic_json(directory / f"{seed}-{arm}-passes-{passes}.json", {
                "characters": model.characters, "weights": model.weights,
                "index": selected.snapshot() if selected else None, "evidence": row})
            atomic_json(directory / "report.json", report)
            print(f"{arm} pass {passes}: new-book accuracy={row['exact_next_word_accuracy']:.4f}; "
                  f"Acts retention={row['acts_retention']['exact_next_word_accuracy']:.4f}; CPU={cpu:.1f}s", flush=True)
    report["status"] = "completed"
    atomic_json(directory / "report.json", report)
    print(f"Completed continuation {run_id}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["stage", "train"])
    parser.add_argument("--acknowledge-review", action="store_true")
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.activate and args.command != "train":
        parser.error("--activate requires train")
    if args.command == "stage":
        manifest = stage(root, acknowledge_review=args.acknowledge_review)
        print(f"Staged {len(manifest['sources'])} chapters, both complete Corinthians books")
    else:
        report = run(root, acknowledge_review=args.acknowledge_review)
        if args.activate:
            from neural_chat import activate
            current, _, _ = load_active(root)
            if current["checkpoint_sha256"] != report["continuation"]["parent_active_checkpoint_sha256"]:
                raise ValueError("Active model changed during training; refusing to replace the new selection")
            active = activate(root, report["run_id"], report["settings"]["seeds"][0], 4)
            print(f"Activated {active['run_id']} SHA256 {active['checkpoint_sha256']}", flush=True)


if __name__ == "__main__":
    main()
