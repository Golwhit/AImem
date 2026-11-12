"""Isolated CPU-budgeted initialization/normalization diagnostic; never activates chat."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import statistics
import time
from uuid import uuid4

from neural_base_comparison import WordBase, digest
from neural_base_diagnostic import ambiguity
from neural_chat import BASE_MODEL, load_active
from neural_conversation_context import ConversationContext, build
from neural_conversation_trial import answer_examples, assess, canonical, token_loss
from neural_experiment import EMBEDDING, HIDDEN, NeuralModel
from original_generator import atomic_json


ARMS = ("inherited", "fresh_scaled", "inherited_normalized")
SEEDS = (17, 117, 217)
RATES = (.01, .1)
CPU_CHECKPOINTS = (3.0, 6.0, 12.0)
EPSILON = 1e-5


class NormalizedModel(NeuralModel):
    """Fixed unit-gain, zero-offset layer normalization before tanh."""

    def _normalized(self, x: list[float]) -> tuple[list[float], float]:
        z = self._preactivations(x)
        mean = statistics.fmean(z)
        centered = [value - mean for value in z]
        inverse = 1 / math.sqrt(statistics.fmean(value * value for value in centered) + EPSILON)
        return [value * inverse for value in centered], inverse

    def _hidden_activation(self, x: list[float]) -> list[float]:
        normalized, _ = self._normalized(x)
        return [math.tanh(value) for value in normalized]

    def _hidden_gradient(
        self, x: list[float], hidden: list[float], upstream: list[float],
    ) -> list[float]:
        normalized, inverse = self._normalized(x)
        delta = [value * (1 - activation ** 2) for value, activation in zip(upstream, hidden)]
        mean = statistics.fmean(delta)
        projection = statistics.fmean(d * n for d, n in zip(delta, normalized))
        return [inverse * (d - mean - n * projection) for d, n in zip(delta, normalized)]


class PlasticityContext(ConversationContext):
    def snapshot(self) -> dict:
        return {**super().snapshot(), "plasticity_arm": (
            "inherited_normalized" if isinstance(self.model, NormalizedModel) else "plain"),
            "normalization_epsilon": EPSILON}

    @classmethod
    def restore(cls, state: dict) -> "PlasticityContext":
        if state.get("plasticity_arm") not in ("plain", "inherited_normalized"):
            raise ValueError("Missing or unsupported plasticity architecture")
        if state.get("normalization_epsilon") != EPSILON:
            raise ValueError("Unsupported normalization epsilon")
        checked = ConversationContext.restore(state)
        result = cls("", 17, lexicon=state["lexicon"], symbols=state["symbols"])
        if state["plasticity_arm"] == "inherited_normalized":
            result.model = NormalizedModel(state["symbols"], context=checked.model.context)
            result.model.weights = checked.model.weights
        else:
            result.model = checked.model
        return result


def initialize(base: ConversationContext, arm: str, seed: int) -> PlasticityContext:
    if arm not in ARMS:
        raise ValueError("Unknown plasticity arm")
    state = json.loads(json.dumps(base.snapshot(), allow_nan=False))
    state.update(plasticity_arm="inherited_normalized" if arm == "inherited_normalized" else "plain",
                 normalization_epsilon=EPSILON)
    model = PlasticityContext.restore(state)
    if arm == "fresh_scaled":
        rng = random.Random(seed)
        fan_in = model.model.context * EMBEDDING
        bounds = {"embedding": math.sqrt(3 / EMBEDDING),
                  "hidden": math.sqrt(6 / (fan_in + HIDDEN)),
                  "output": math.sqrt(6 / (HIDDEN + model.model.size))}
        for name, values in model.model.weights.items():
            bound = bounds.get(name, 0.0)
            model.model.weights[name] = (
                [rng.uniform(-bound, bound) for _ in values] if bound else [0.0] * len(values))
    return model


def diagnostics(model: WordBase, examples: list) -> dict:
    activations, derivatives, norms = [], [], []
    for context, target, prior in examples:
        hidden = model.model.forward(context, output_prior=prior)[1]
        activations.extend(hidden)
        derivatives.extend(1 - value * value for value in hidden)
        _, gradients = model.model.loss_and_gradients(context, target, output_prior=prior)
        norms.append(math.sqrt(sum(value * value for value in gradients["hidden"])))
    result = {
        "hidden_abs_over_099_fraction": statistics.fmean(abs(value) > .99 for value in activations),
        "mean_tanh_derivative": statistics.fmean(derivatives),
        "mean_hidden_gradient_norm": statistics.fmean(norms),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ArithmeticError("Nonfinite plasticity diagnostics")
    return result


def fit(model: WordBase, rows: list[dict], *, seed: int, rate: float,
        cpu_checkpoints: tuple[float, ...], decay: float = 1.0) -> list[dict]:
    if (not cpu_checkpoints or list(cpu_checkpoints) != sorted(set(cpu_checkpoints))
            or any(not math.isfinite(n) or n <= 0 for n in cpu_checkpoints)
            or not math.isfinite(rate) or rate <= 0
            or not math.isfinite(decay) or not 0 < decay <= 1):
        raise ValueError("Invalid plasticity training budget")
    examples = answer_examples(model, rows)
    rng = random.Random(seed)
    cpu, updates, points = 0.0, 0, []
    for segment, budget in enumerate(cpu_checkpoints):
        segment_rate = rate * decay ** segment
        while cpu < budget:
            start = time.process_time()
            for _ in range(100):
                context, target, prior = examples[rng.randrange(len(examples))]
                model.model.update(context, target, output_prior=prior, learning_rate=segment_rate)
            cpu += time.process_time() - start
            updates += 100
        start = time.process_time()
        point = {"updates": updates, "training_cpu_seconds": cpu,
                 "teacher_forced_loss": token_loss(model, examples),
                 "training": assess(model, rows), "diagnostics": diagnostics(model, examples)}
        point["measurement_cpu_seconds"] = time.process_time() - start
        points.append(point)
    return points


def replay(initial: dict, rows: list[dict], *, rate: float, seed: int,
           checkpoints: list[dict], decay: float = 1.0) -> PlasticityContext:
    model = PlasticityContext.restore(json.loads(json.dumps(initial)))
    examples = answer_examples(model, rows)
    rng = random.Random(seed)
    updates = 0
    for segment, point in enumerate(checkpoints):
        segment_rate = rate * decay ** segment
        while updates < point["updates"]:
            context, target, prior = examples[rng.randrange(len(examples))]
            model.model.update(context, target, output_prior=prior, learning_rate=segment_rate)
            updates += 1
        if (token_loss(model, examples) != point["teacher_forced_loss"]
                or assess(model, rows) != point["training"]
                or diagnostics(model, examples) != point["diagnostics"]):
            raise AssertionError("Replayed plasticity milestone differs")
    return model


def run(root: Path, *, acknowledge_review: bool, reviewed_sha256: str) -> dict:
    raw = (root / "fixtures/neural-conversation-draft.json").read_bytes()
    if acknowledge_review is not True or digest(raw) != reviewed_sha256:
        raise ValueError("Explicit approval of exact teaching curriculum is required")
    data = json.loads(raw)
    ids = [row["id"] for split in ("train", "test") for row in data[split]]
    if (not data["train"] or not data["test"] or len(ids) != len(set(ids))
            or {canonical(row["turns"]) for row in data["train"]} &
            {canonical(row["turns"]) for row in data["test"]}):
        raise ValueError("Nonempty disjoint teaching and test splits required")
    record, parent, index = load_active(root)
    if record["model"] != BASE_MODEL or not isinstance(parent, WordBase) or index is not None:
        raise ValueError("Plasticity comparison requires layer-free working base")
    protected = {str(p.relative_to(root)): digest(p.read_bytes())
                 for p in (root / ".data/neural-chat").glob("*.json")}
    base, metadata = build(parent, data["train"], "long_identities")
    directory = root / ".data/neural-plasticity" / uuid4().hex
    report = {
        "run_id": directory.name, "status": "running", "parent": record,
        "metadata": metadata, "seeds": SEEDS, "rates": RATES, "cpu_checkpoints": CPU_CHECKPOINTS,
        "review": {"file_sha256": reviewed_sha256,
                   "train_sha256": digest(canonical(data["train"])),
                   "test_sha256": digest(canonical(data["test"])),
                   "approval": "User input: \"let's do it\"",
                   "scope": "Initialization/normalization on previously approved eight teaching exchanges. "
                            "Ten developer-visible test answers evaluation only; user declaration, not legal clearance.",
                   "recorded_at": datetime.now(timezone.utc).isoformat(), "provenance": data["provenance"]},
        "policy": "Same frozen long-context representation, training-only symbols, masks, parameter count "
                  "and plain SGD. inherited retains actual weights; fresh_scaled resets all learned weights "
                  "using fan-in/out bounds (embedding variance 1/8, zero biases); inherited_normalized retains "
                  "weights but changes function via fixed unit-gain zero-offset pre-tanh layer normalization "
                  "(epsilon 1e-5), NOT learned affine normalization or a behavior-preserving conversion. "
                  "No pretrained teachers, new data, retired layers, or activation. Three target-draw seeds "
                  "are not three independent parent pretraining seeds. Fresh initialization uses each draw seed. "
                  "Each rate gets equal update CPU budget checked per 100 updates; report overshoot, measurement "
                  "and evaluation overhead separately. Same RNG sequence, but update counts differ with speed. "
                  "Select one rate per arm/seed by final TEACHING loss before ANY test evaluation. "
                  "Existing tests are developer-visible exploratory evidence, not fresh generalization tests. "
                  "No claim retained prior knowledge after reset/normalization; no retention benchmark. "
                  "Replay recorded update counts, full states and non-timing milestones exactly. "
                  "No input can fix remaining representation collisions; token bound is not an answer bound. "
                  "Active working and original control files remain protected; memory stays connected.",
        "code_sha256": {name: digest((root / name).read_bytes()) for name in (
            "neural_plasticity.py", "neural_experiment.py", "neural_conversation_context.py",
            "neural_conversation_trial.py", "neural_base_comparison.py", "neural_continuation.py")},
        "protected_before": protected, "trials": [],
        "answer_tokens": len(answer_examples(base, data["train"])),
        "representation": ambiguity(answer_examples(base, data["train"])),
    }
    atomic_json(directory / "inputs.json", data)
    atomic_json(directory / "parent.json", {"state": parent.snapshot()})
    atomic_json(directory / "review.json", report["review"])
    atomic_json(directory / "protocol.json", report)
    print(f"Plasticity run {directory.name}", flush=True)
    for seed in SEEDS:
        for arm in ARMS:
            initial_model = initialize(base, arm, seed)
            initial = json.loads(json.dumps(initial_model.snapshot(), allow_nan=False))
            initial_examples = answer_examples(initial_model, data["train"])
            initial_diagnostics = diagnostics(initial_model, initial_examples)
            initial_loss = token_loss(initial_model, initial_examples)
            atomic_json(directory / f"{arm}-{seed}-initial.json", {"state": initial})
            for rate in RATES:
                model = PlasticityContext.restore(json.loads(json.dumps(initial)))
                points = fit(model, data["train"], seed=seed, rate=rate, cpu_checkpoints=CPU_CHECKPOINTS)
                state = json.loads(json.dumps(model.snapshot(), allow_nan=False))
                trial = {"arm": arm, "seed": seed, "rate": rate, "checkpoints": points,
                         "initial_diagnostics": initial_diagnostics, "initial_teacher_forced_loss": initial_loss}
                atomic_json(directory / f"{arm}-{seed}-{rate}-final.json", {"state": state, "trial": trial})
                report["trials"].append(trial)
                atomic_json(directory / "report.json", report)
                last = points[-1]
                print(f"{arm} seed={seed} rate={rate}: updates={last['updates']}; "
                      f"loss={last['teacher_forced_loss']:.4f}; train={last['training']['exact']:.3f}; "
                      f"CPU={last['training_cpu_seconds']:.2f}", flush=True)
    selected = []
    for seed in SEEDS:
        for arm in ARMS:
            candidates = [t for t in report["trials"] if t["seed"] == seed and t["arm"] == arm]
            chosen = min(candidates, key=lambda t: t["checkpoints"][-1]["teacher_forced_loss"])
            selected.append({"arm": arm, "seed": seed, "rate": chosen["rate"]})
    report["selection"] = selected
    atomic_json(directory / "selection.json", report)
    for chosen in selected:
        path = directory / f"{chosen['arm']}-{chosen['seed']}-{chosen['rate']}-final.json"
        model = PlasticityContext.restore(json.loads(path.read_text())["state"])
        start = time.process_time()
        chosen["test"] = assess(model, data["test"])
        chosen["evaluation_cpu_seconds"] = time.process_time() - start
        print(f"Selected {chosen['arm']} seed={chosen['seed']} rate={chosen['rate']}: "
              f"test={chosen['test']['exact']:.3f}", flush=True)
    report["protected_after"] = {str(p.relative_to(root)): digest(p.read_bytes())
                                for p in (root / ".data/neural-chat").glob("*.json")}
    if report["protected_after"] != protected:
        raise RuntimeError("Protected chat/control checkpoints changed")
    report["status"] = "completed"
    atomic_json(directory / "report.json", report)
    return report


def validate(root: Path, run_id: str) -> dict:
    if len(run_id) != 32 or any(char not in "0123456789abcdef" for char in run_id):
        raise ValueError("Invalid plasticity run ID")
    directory = root / ".data/neural-plasticity" / run_id
    report = json.loads((directory / "report.json").read_text())
    if report["status"] != "completed":
        raise ValueError("Validation requires completed trial")
    for name, expected in report["code_sha256"].items():
        if digest((root / name).read_bytes()) != expected:
            raise ValueError(f"Replay source changed: {name}")
    data = json.loads((directory / "inputs.json").read_text())
    if any(digest(canonical(data[split])) != report["review"][f"{split}_sha256"]
           for split in ("train", "test")):
        raise ValueError("Saved inputs differ from review")
    for trial in report["trials"]:
        arm, seed, rate = trial["arm"], trial["seed"], trial["rate"]
        initial = json.loads((directory / f"{arm}-{seed}-initial.json").read_text())["state"]
        model = replay(initial, data["train"], rate=rate, seed=seed, checkpoints=trial["checkpoints"])
        saved = json.loads((directory / f"{arm}-{seed}-{rate}-final.json").read_text())["state"]
        if model.snapshot() != saved:
            raise AssertionError("Full trained weights differ on replay")
        chosen = next((s for s in report["selection"]
                       if s["arm"] == arm and s["seed"] == seed and s["rate"] == rate), None)
        if chosen is not None and assess(model, data["test"]) != chosen["test"]:
            raise AssertionError("Selected test predictions differ on replay")
        print(f"Replayed {arm}/{seed}/{rate}", flush=True)
    protected = {str(p.relative_to(root)): digest(p.read_bytes())
                 for p in (root / ".data/neural-chat").glob("*.json")}
    if protected != report["protected_before"]:
        raise AssertionError("Protected checkpoints no longer match protocol")
    result = {"exact_weight_replays": len(report["trials"]), "exact_selected_test_replays":
              len(report["selection"]), "non_timing_milestones_exact": True,
              "protected_checkpoints_unchanged": True}
    atomic_json(directory / "validation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acknowledge-review", action="store_true")
    parser.add_argument("--reviewed-sha256")
    parser.add_argument("--validate-run")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.validate_run:
        validate(root, args.validate_run)
    else:
        if not args.reviewed_sha256:
            parser.error("--reviewed-sha256 is required for training")
        run(root, acknowledge_review=args.acknowledge_review, reviewed_sha256=args.reviewed_sha256)
