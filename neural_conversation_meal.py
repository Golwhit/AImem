"""Original paired teaching curriculum and isolated, CPU-matched size comparison."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from uuid import uuid4

from neural_base_comparison import WordBase, digest
from neural_base_diagnostic import ambiguity
from neural_chat import BASE_MODEL, load_selection
from neural_conversation_context import build
from neural_conversation_trial import answer_examples, assess, canonical
from neural_plasticity import PlasticityContext, diagnostics, fit, initialize, replay
from original_generator import atomic_json


FIXTURE = "fixtures/neural-conversation-meal.json"
PINNED_BASE_SHA256 = "8887c490caa5631d1b22286edfe3a3f7a0b0908d19cb31ccc6f8cf0d07d08ea7"
ARMS = ("fresh_scaled", "inherited_normalized")
SEEDS = (17, 117, 217)
RATE = .1
CPU_CHECKPOINTS = (8.0, 24.0)
SKILLS = ("explicit_owner", "attribute", "transfer", "negation", "correction",
          "missing_information", "reference", "unknown_location")
SOURCES = ("neural_conversation_meal.py", "neural_plasticity.py", "neural_experiment.py",
           "neural_conversation_context.py", "neural_conversation_trial.py",
           "neural_base_comparison.py", "neural_continuation.py")


def exchanges(owner: str, other: str, obj: str, case: int, split: str,
              *, paraphrase: bool = False) -> list[dict]:
    rows = []
    color_question = f"What color is the {obj}?" if not paraphrase else f"Tell me the color of the {obj}."
    owner_question = f"Who has the {obj}?" if not paraphrase else f"Name the person with the {obj}."
    now_question = f"Who has the {obj} now?" if not paraphrase else f"Name the current holder of the {obj}."
    known_color = ("red", "blue")[case % 2]
    location = ("here", "there")[case % 2]
    for skill in SKILLS:
        pair = f"{split}-{case}-{skill}"
        for variant in (0, 1):
            color = ("red", "blue")[variant]
            opposite = ("blue", "red")[variant]
            turns = []
            if skill == "explicit_owner":
                holder = (owner, other)[variant]
                content = f"{holder} has the {obj}. {owner_question}"
                answer = holder + "."
            elif skill == "attribute":
                content = f"{owner} has a {color} {obj}. {color_question}"
                answer = color.capitalize() + "."
            elif skill == "transfer":
                action = f"keeps the {obj}" if variant == 0 else f"gives the {obj} to {other}"
                content = f"{owner} has a {obj}. {owner} {action}. {now_question}"
                answer = (owner if variant == 0 else other) + "."
            elif skill == "negation":
                query = f"Is the {obj} red?" if not paraphrase else f"Is red the color of the {obj}?"
                content = f"{owner} has a {obj}. The {obj} is not {opposite}. It is {color}. {query}"
                answer = "Yes." if color == "red" else "No."
            elif skill == "correction":
                first = f"{owner}'s {obj} is {opposite}."
                turns = [{"role": "user", "content": first}, {"role": "assistant", "content": first}]
                query = "What color is it?" if not paraphrase else "Tell me its color."
                content = f"Correction: the {obj} is {color}. {query}"
                answer = color.capitalize() + "."
            elif skill == "missing_information":
                attribute = known_color + " " if variant == 0 else ""
                query = "What color is it?" if not paraphrase else "Tell me its color."
                content = f"{owner} has a {attribute}{obj}. {query}"
                answer = known_color.capitalize() + "." if variant == 0 else "I do not know."
            elif skill == "reference":
                first = f"{owner} has a {obj}."
                turns = [{"role": "user", "content": first}, {"role": "assistant", "content": first}]
                content = f"It is {color}. {color_question}"
                answer = color.capitalize() + "."
            else:
                fact = (f"{owner} has a {obj}. The {obj} is {location}."
                        if variant == 0 else f"{owner} has the {obj}.")
                query = f"Where is the {obj}?" if not paraphrase else f"Tell me where the {obj} is."
                content = f"{fact} {query}"
                answer = location.capitalize() + "." if variant == 0 else "I do not know."
            rows.append({"id": f"{pair}-{variant}", "pair": pair, "skill": skill, "case": case,
                         "evaluation_group": split if split != "train" else None,
                         "turns": turns + [{"role": "user", "content": content}], "answer": answer})
    return rows


TEACHING_CASES = (
    ("Mira", "Jon", "cup"), ("Jon", "Lena", "box"), ("Lena", "Omar", "book"), ("Omar", "Mira", "bag"),
    ("Ada", "Ben", "vase"), ("Cora", "Dev", "lamp"), ("Eli", "Faye", "ring"), ("Gus", "Hope", "hat"),
    ("Ivan", "June", "kite"), ("Kai", "Liv", "jar"), ("Milo", "Nia", "map"), ("Otto", "Pearl", "pen"),
    ("Rex", "Sol", "shell"), ("Tara", "Uma", "coin"), ("Vic", "Wren", "plate"), ("Yuri", "Zoe", "rope"),
)
TEACHING_SIZES = (64, 256)
LARGE_FIXTURE = "fixtures/neural-conversation-meal-256.json"


def fixture_for(size: int) -> str:
    return FIXTURE if size == 64 else LARGE_FIXTURE


def curriculum(size: int = 64) -> dict:
    if size not in TEACHING_SIZES:
        raise ValueError(f"Teaching size must be one of {TEACHING_SIZES}")
    training = []
    for case, (owner, other, obj) in enumerate(TEACHING_CASES[:size // 16]):
        training.extend(exchanges(owner, other, obj, case, "train"))
    return {
        "schema": 1, "status": "generated_pending_exact_content_hash_approval",
        "provenance": "Original synthetic examples generated with AI assistance for this project. "
                      "Not copied from an external corpus. Not a guarantee of exclusive rights or legal clearance.",
        "purpose": f"{size} balanced teaching exchanges, not general conversation training.",
        "training_policy": "Only the train split is optimizer input, following explicit approval of exact file SHA256. "
                           "No runtime templates or answer lookup. Test never constructs vocabulary or selects models.",
        "evaluation_policy": "32 newly authored developer-visible diagnostic cases: 16 new entity/object combinations "
                             "with familiar phrasing and 16 changed-wording cases, also with new combinations. "
                             "Not a locked independent benchmark or proof of generalization.",
        "small_control_ids": [row["id"] for row in training if row["case"] == 0],
        "train": training,
        "test": exchanges("Mira", "Jon", "book", 0, "combination") +
                exchanges("Jon", "Mira", "cup", 1, "wording", paraphrase=True),
    }


def validate_curriculum(data: dict) -> None:
    if len(data["train"]) not in TEACHING_SIZES or len(data["test"]) != 32:
        raise ValueError("Meal requires 64 or 256 teaching and exactly 32 evaluation exchanges")
    per_skill = len(data["train"]) // len(SKILLS)
    all_rows = data["train"] + data["test"]
    ids = [row["id"] for row in all_rows]
    prompts = [canonical(row["turns"]) for row in all_rows]
    if len(set(ids)) != len(ids) or len(set(prompts)) != len(prompts):
        raise ValueError("Exchange IDs and complete prompts must be globally unique")
    controls = data["small_control_ids"]
    if len(controls) != 16 or len(set(controls)) != 16 or not set(controls) <= {
            row["id"] for row in data["train"]}:
        raise ValueError("Small control requires 16 distinct teaching IDs")
    for rows, count in ((data["train"], per_skill), (data["test"], 4),
                        ([row for row in data["train"] if row["id"] in controls], 2)):
        if Counter(row["skill"] for row in rows) != Counter({skill: count for skill in SKILLS}):
            raise ValueError("Each split must balance skill counts")
        pairs = defaultdict(list)
        for row in rows:
            if not row["answer"].strip() or "\n" in row["answer"]:
                raise ValueError("Answers must be nonempty single lines")
            pairs[row["pair"]].append(row)
        if any(len(pair) != 2 or pair[0]["answer"] == pair[1]["answer"] for pair in pairs.values()):
            raise ValueError("Every pair must have two distinct expected answers")


def paired_scores(assessment: dict) -> dict:
    groups = defaultdict(list)
    skills = defaultdict(list)
    for row in assessment["rows"]:
        groups[row["pair"]].append(row)
        skills[row["skill"]].append(row)
    if any(len(rows) != 2 or rows[0]["expected"] == rows[1]["expected"] for rows in groups.values()):
        raise ValueError("Paired scoring requires distinct-answer pairs")
    return {
        "pairs": len(groups),
        "both_exact": sum(all(row["exact"] for row in rows) for rows in groups.values()),
        "both_normalized_exact": sum(all(row["normalized_exact"] for row in rows) for rows in groups.values()),
        "identical_pair_answers": sum(rows[0]["prediction"] == rows[1]["prediction"] for rows in groups.values()),
        "by_skill": {skill: {"correct": sum(row["exact"] for row in rows), "examples": len(rows)}
                     for skill, rows in skills.items()},
    }


def protected_files(root: Path) -> dict:
    return {str(p.relative_to(root)): digest(p.read_bytes())
            for p in (root / ".data/neural-chat").glob("*.json")}


def train_trial(job: dict) -> tuple[dict, dict]:
    model = PlasticityContext.restore(json.loads(json.dumps(job["initial"])))
    points = fit(model, job["rows"], seed=job["seed"], rate=RATE,
                 cpu_checkpoints=job["cpu_checkpoints"], decay=job["decay"])
    state = json.loads(json.dumps(model.snapshot(), allow_nan=False))
    trial = {"arm": job["arm"], "seed": job["seed"], "subset": job["subset"], "checkpoints": points,
             "teaching_pairs": paired_scores(points[-1]["training"])}
    return trial, state


def run(root: Path, *, acknowledge_review: bool, reviewed_sha256: str, size: int = 64,
        cpu_checkpoints: tuple[float, ...] = CPU_CHECKPOINTS, decay: float = 1.0) -> dict:
    raw = (root / fixture_for(size)).read_bytes()
    if acknowledge_review is not True or digest(raw) != reviewed_sha256:
        raise ValueError("Explicit approval of exact larger curriculum hash required")
    data = json.loads(raw)
    validate_curriculum(data)
    if len(data["train"]) != size:
        raise ValueError("Fixture teaching size does not match requested size")
    record, parent, index = load_selection(root, {
        "model": BASE_MODEL, "arm": "new", "checkpoint_sha256": PINNED_BASE_SHA256})
    if record["model"] != BASE_MODEL or not isinstance(parent, WordBase) or index is not None:
        raise ValueError("Larger-meal experiment requires pinned layer-free working base")
    protected = protected_files(root)
    base, metadata = build(parent, data["train"], "long_identities")
    small = [row for row in data["train"] if row["id"] in data["small_control_ids"]]
    full = f"meal{size}"
    subsets = {"small16": small, full: data["train"]}
    directory = root / ".data/neural-conversation-meal" / uuid4().hex
    report = {
        "run_id": directory.name, "status": "running", "parent": record, "metadata": metadata,
        "full_subset": full,
        "seeds": SEEDS, "arms": ARMS, "rate": RATE, "decay": decay, "cpu_checkpoints": cpu_checkpoints,
        "review": {"file_sha256": reviewed_sha256,
                   "canonical_file_sha256": digest(canonical(data)),
                   "approval": "User approved exact file SHA256 via approve_exact_curriculum=true",
                   "train_sha256": digest(canonical(data["train"])),
                   "test_sha256": digest(canonical(data["test"])),
                   "scope": f"{size} original teaching exchanges and their 16-example subset; test answers evaluation only.",
                   "recorded_at": datetime.now(timezone.utc).isoformat(),
                   "provenance": data["provenance"], "record_type": "User declaration, not legal clearance"},
        "policy": "Same fixed fresh_scaled and inherited_normalized designs and starting rate .1 from prior teaching-only "
                  f"selection, no new optimizer, auxiliary head, contrastive loss or architecture. Rate for CPU segment k is "
                  f"0.1 * decay**k with decay {decay} fixed before any test. Both arms start from "
                  "new copies, not previous successful trial weights. Freeze lexicon and expanded character/identity "
                  f"alphabet using all {size} TRAINING examples; same initial weights per arm/seed across small16/{full}. "
                  "Trials run as parallel processes, each with its own CPU-time budget. "
                  "Small16 is one complete balanced pair per skill, NOT the historical 8-example dataset. Its input "
                  "vocabulary includes other teaching cases, but their answers never update small16 weights. "
                  "Same parent, parameters/masks and update CPU; faster runs get more target draws. Uniform answer-token "
                  "sampling, not pair-level contrastive optimization. Extra data includes new combinations and cases, "
                  "not simply repeated old examples. Train only answer+newline targets. Check budget per 100 updates "
                  "with explicit overshoot; checkpoints teaching-only. Final fixed endpoint only; save all six "
                  "arm/seed choices per data size before any test, no test-guided selection. Three draw/fresh-init "
                  "seeds from one parent are not independent pretraining seeds. 32 test prompts developer-visible, "
                  "not fresh locked evidence. Compare both pair members correct, not just common-answer guesses. "
                  "No active model changes or corpus-retention claims. Memory capture remains connected. "
                  "Measurement/evaluation/setup/persistence/replay cost is outside update budget and reported separately "
                  "where measured. Generic full-96-character decoder/scoring reused unchanged.",
        "code_sha256": {name: digest((root / name).read_bytes()) for name in SOURCES},
        "protected_before": protected, "trials": [],
        "answer_tokens": {name: len(answer_examples(base, rows)) for name, rows in subsets.items()},
        "representation": {name: ambiguity(answer_examples(base, rows)) for name, rows in subsets.items()},
    }
    atomic_json(directory / "inputs.json", data)
    atomic_json(directory / "parent.json", {"state": parent.snapshot()})
    atomic_json(directory / "review.json", report["review"])
    atomic_json(directory / "protocol.json", report)
    for name in SOURCES:
        path = directory / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((root / name).read_bytes())
    print(f"Meal run {directory.name}", flush=True)
    jobs = []
    for seed in SEEDS:
        for arm in ARMS:
            initial_model = initialize(base, arm, seed)
            initial = json.loads(json.dumps(initial_model.snapshot(), allow_nan=False))
            atomic_json(directory / f"{arm}-{seed}-initial.json", {"state": initial})
            for subset, rows in subsets.items():
                jobs.append({"arm": arm, "seed": seed, "subset": subset, "rows": rows,
                             "initial": initial, "cpu_checkpoints": cpu_checkpoints, "decay": decay})
    with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1)) as pool:
        for job, (trial, state) in zip(jobs, pool.map(train_trial, jobs)):
            arm, seed, subset = job["arm"], job["seed"], job["subset"]
            points = trial["checkpoints"]
            atomic_json(directory / f"{arm}-{seed}-{subset}-final.json", {"state": state, "trial": trial})
            report["trials"].append(trial)
            atomic_json(directory / "report.json", report)
            print(f"{arm}/{seed}/{subset}: loss={points[-1]['teacher_forced_loss']:.4f}; "
                  f"train={points[-1]['training']['exact']:.3f}; "
                  f"CPU={points[-1]['training_cpu_seconds']:.2f}", flush=True)
    atomic_json(directory / "pre-test-endpoints.json", report)
    for trial in report["trials"]:
        name = f"{trial['arm']}-{trial['seed']}-{trial['subset']}-final.json"
        model = PlasticityContext.restore(json.loads((directory / name).read_text())["state"])
        start = time.process_time()
        trial["test"] = assess(model, data["test"])
        trial["test_pairs"] = paired_scores(trial["test"])
        trial["test_groups"] = {
            group: {"correct": sum(row["exact"] for row in trial["test"]["rows"]
                                   if row["id"] in {item["id"] for item in data["test"]
                                                    if item["evaluation_group"] == group}),
                    "examples": sum(item["evaluation_group"] == group for item in data["test"])}
            for group in ("combination", "wording")}
        trial["evaluation_cpu_seconds"] = time.process_time() - start
        print(f"Test {trial['arm']}/{trial['seed']}/{trial['subset']}: "
              f"exact={trial['test']['exact']:.3f}; pairs={trial['test_pairs']['both_exact']}/16", flush=True)
    report["protected_after"] = protected_files(root)
    if protected != report["protected_after"]:
        raise RuntimeError("Pinned working/control files changed during larger-meal comparison")
    report["status"] = "completed"
    atomic_json(directory / "report.json", report)
    return report


def validate(root: Path, run_id: str) -> dict:
    if len(run_id) != 32 or any(char not in "0123456789abcdef" for char in run_id):
        raise ValueError("Invalid meal run ID")
    directory = root / ".data/neural-conversation-meal" / run_id
    report = json.loads((directory / "report.json").read_text())
    if report["status"] != "completed":
        raise ValueError("Completed meal experiment required")
    for name, checksum in report["code_sha256"].items():
        if digest((root / name).read_bytes()) != checksum or digest(
                (directory / "source" / name).read_bytes()) != checksum:
            raise ValueError(f"Experiment source changed: {name}")
    raw = (directory / "inputs.json").read_bytes()
    data = json.loads(raw)
    validate_curriculum(data)
    if digest(canonical(data)) != report["review"]["canonical_file_sha256"]:
        raise ValueError("Saved full curriculum differs from approved content")
    for split in ("train", "test"):
        if digest(canonical(data[split])) != report["review"][f"{split}_sha256"]:
            raise ValueError("Saved teaching/evaluation split changed")
    subset_ids = set(data["small_control_ids"])
    subsets = {"small16": [row for row in data["train"] if row["id"] in subset_ids],
               report.get("full_subset", "meal64"): data["train"]}
    for trial in report["trials"]:
        arm, seed, subset = trial["arm"], trial["seed"], trial["subset"]
        initial = json.loads((directory / f"{arm}-{seed}-initial.json").read_text())["state"]
        model = replay(initial, subsets[subset], rate=report["rate"], seed=seed, checkpoints=trial["checkpoints"],
                       decay=report.get("decay", 1.0))
        saved = json.loads((directory / f"{arm}-{seed}-{subset}-final.json").read_text())["state"]
        if model.snapshot() != saved or assess(model, data["test"]) != trial["test"]:
            raise AssertionError("Full weights or evaluation predictions differ on replay")
        if paired_scores(trial["test"]) != trial["test_pairs"]:
            raise AssertionError("Paired scores changed")
        print(f"Replayed {arm}/{seed}/{subset}", flush=True)
    if protected_files(root) != report["protected_before"]:
        raise AssertionError("Pinned working/control files differ")
    result = {"exact_weight_and_test_replays": len(report["trials"]),
              "all_non_timing_teaching_milestones_exact": True, "protected_checkpoints_unchanged": True}
    atomic_json(directory / "validation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--acknowledge-review", action="store_true")
    parser.add_argument("--reviewed-sha256")
    parser.add_argument("--validate-run")
    parser.add_argument("--size", type=int, default=64, choices=TEACHING_SIZES)
    parser.add_argument("--budget", type=float, help="Total CPU-second training budget per trial; default is CPU_CHECKPOINTS.")
    parser.add_argument("--segments", type=int, default=1, help="Equal CPU segments within --budget.")
    parser.add_argument("--decay", type=float, default=1.0, help="Per-segment learning-rate multiplier.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.prepare:
        path = root / fixture_for(args.size)
        if path.exists():
            raise FileExistsError("Preserve existing generated curriculum; preparation cannot overwrite it")
        data = curriculum(args.size)
        validate_curriculum(data)
        atomic_json(path, data)
        print(f"{path}: SHA256 {digest(path.read_bytes())}")
    elif args.validate_run:
        validate(root, args.validate_run)
    else:
        if not args.reviewed_sha256:
            parser.error("--reviewed-sha256 required before training")
        if args.budget:
            checkpoints = tuple(args.budget * (k + 1) / args.segments for k in range(args.segments))
        else:
            checkpoints = CPU_CHECKPOINTS
        run(root, acknowledge_review=args.acknowledge_review, reviewed_sha256=args.reviewed_sha256,
            size=args.size, cpu_checkpoints=checkpoints, decay=args.decay)
