import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from neural_candidate_layers import CandidateIndex
from neural_chat import activate, activate_base, generate, status, DIRECTORY, BASE_MODEL
from neural_base_comparison import WordBase, generate as generate_base
from neural_experiment import NeuralModel


def chat_fixture(root: Path) -> str:
    run_id = "a" * 32
    directory = root / ".data/neural-unit-benchmark" / run_id
    directory.mkdir(parents=True)
    text = "A returns unplugged returns unplugged. "
    index = CandidateIndex.build({}, text, 1, possibilities={"returns", "unplugged", "A"})
    index.learn(index.extract(text)["pieces"])
    model = NeuralModel(sorted(set(text)))
    evidence = {"passes": 4}
    (directory / "17-layers_01-passes-4.json").write_text(json.dumps({
        "characters": model.characters, "weights": model.weights, "index": index.snapshot(), "evidence": evidence}))
    (directory / "report.json").write_text(json.dumps({
        "status": "completed", "settings": {"matched_reinforcement": True},
        "source": {"training": "Authored test fixture", "training_sha256": hashlib.sha256(text.encode()).hexdigest()},
        "runs": [{"arm": "layers_01", "seed": 17, "checkpoints": [evidence]}]}))
    return run_id


def base_chat_fixture(root: Path) -> str:
    run_id = "b" * 32
    directory = root / ".data/neural-base-comparison" / run_id
    directory.mkdir(parents=True)
    text = "A cat, a cat. "
    model = WordBase(text, 17)
    original = NeuralModel(sorted(set(text)), seed=17)
    evidence = {"passes": 4, "accuracy": 0.0}
    for arm, state in (("new", model.snapshot()), ("original", {
        "characters": original.characters, "weights": original.weights,
    })):
        (directory / f"17-{arm}-passes-4.json").write_text(json.dumps({
            "state": state, "index": None, "evidence": evidence}))
    checksum = hashlib.sha256(text.encode()).hexdigest()
    (directory / "inputs.json").write_text(json.dumps({"training": text, "evaluation": text}))
    (directory / "report.json").write_text(json.dumps({
        "status": "completed", "training_sha256": checksum, "evaluation_sha256": checksum,
        "runs": [{"arm": arm, "seed": 17, "checkpoints": [evidence]} for arm in ("new", "original")]}))
    return run_id


class NeuralChatTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def test_activation_is_pinned_and_matches_source_without_training(self):
        self.assertFalse(status(self.root)["ready"])
        run_id = chat_fixture(self.root)
        active = activate(self.root, run_id)
        self.assertTrue(active["ready"])
        self.assertEqual(active["seed"], 17)
        self.assertEqual(active["arm"], "layers_01")
        checkpoint = json.loads((self.root / ".data/neural-unit-benchmark" / run_id /
                                 "17-layers_01-passes-4.json").read_text())
        model = NeuralModel(checkpoint["characters"])
        model.weights = checkpoint["weights"]
        index = CandidateIndex.restore(checkpoint["index"])
        expected = index.sample(model, "The ", length=32)
        first = generate(self.root, {"prompt": "The ", "max_chars": 32})
        for key in expected:
            self.assertEqual(first[key], expected[key])
        checkpoint_path = self.root / DIRECTORY / f"{active['checkpoint_sha256']}.json"
        before = checkpoint_path.read_bytes()
        source = self.root / ".data/neural-unit-benchmark" / run_id / "17-layers_01-passes-4.json"
        source.write_text("changed experiment")
        self.assertEqual(generate(self.root, {"prompt": "The ", "max_chars": 32}), first)
        self.assertEqual(checkpoint_path.read_bytes(), before)
        checkpoint_path.write_bytes(before + b" ")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            status(self.root)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            generate(self.root, {"prompt": "test"})
        checkpoint_path.unlink()
        with self.assertRaises(FileNotFoundError):
            generate(self.root, {"prompt": "test"})

    def test_bad_selection_and_inputs_fail_explicitly(self):
        run_id = chat_fixture(self.root)
        for invalid in ("../bad", "z" * 32):
            with self.assertRaises(ValueError):
                activate(self.root, invalid)
        with self.assertRaises(ValueError):
            activate(self.root, run_id, seed=999)
        activate(self.root, run_id)
        for payload in ({}, {"prompt": ""}, {"prompt": "\x00"}, {"prompt": "a" * 2001},
                        {"prompt": "x", "max_chars": True}, {"prompt": "x", "max_chars": 501}):
            with self.assertRaises(ValueError):
                generate(self.root, payload)

    def test_new_base_retires_layers_and_pins_original_without_training(self):
        activate(self.root, chat_fixture(self.root))
        old_selection = (self.root / DIRECTORY / "active.json").read_bytes()
        run_id = base_chat_fixture(self.root)
        active = activate_base(self.root, run_id)
        self.assertEqual(active["model"], BASE_MODEL)
        self.assertEqual(active["retired_layers"], [0, 1, 2])
        self.assertEqual(active["comparison_baseline"]["arm"], "original")
        self.assertEqual((self.root / DIRECTORY / (
            "retired-selection-" + hashlib.sha256(old_selection).hexdigest() + ".json")).read_bytes(), old_selection)
        source = self.root / ".data/neural-base-comparison" / run_id
        checkpoint = json.loads((source / "17-new-passes-4.json").read_text())
        expected = generate_base(WordBase.restore(checkpoint["state"]), "A ", length=32)
        result = generate(self.root, {"prompt": "A ", "max_chars": 32})
        for key in expected:
            self.assertEqual(result[key], expected[key])
        self.assertEqual(result["index_lookups"], 0)
        control = json.loads((source / "17-original-passes-4.json").read_text())["state"]
        original_model = NeuralModel(control["characters"])
        original_model.weights = control["weights"]
        expected_control = generate_base(original_model, "A ", length=32)
        control_result = generate(self.root, {"prompt": "A ", "max_chars": 32}, original=True)
        for key in expected_control:
            self.assertEqual(control_result[key], expected_control[key])
        self.assertEqual(control_result["chat_model"]["arm"], "original")
        before = {p.name: p.read_bytes() for p in (self.root / DIRECTORY).glob("*.json")}
        (source / "17-new-passes-4.json").write_text("source changed")
        self.assertEqual(generate(self.root, {"prompt": "A ", "max_chars": 32}), result)
        self.assertEqual({p.name: p.read_bytes() for p in (self.root / DIRECTORY).glob("*.json")}, before)
        baseline_path = self.root / DIRECTORY / f"{active['comparison_baseline']['checkpoint_sha256']}.json"
        baseline_path.write_bytes(baseline_path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            generate(self.root, {"prompt": "A "}, original=True)
        pinned = self.root / DIRECTORY / f"{active['checkpoint_sha256']}.json"
        pinned.write_bytes(pinned.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            status(self.root)

    def test_new_base_rejects_bad_evidence_weights_and_layer_state(self):
        run_id = base_chat_fixture(self.root)
        checkpoint = self.root / ".data/neural-base-comparison" / run_id / "17-new-passes-4.json"
        original = json.loads(checkpoint.read_text())
        for field in ("weights", "evidence", "index", "vocabulary"):
            value = json.loads(json.dumps(original))
            if field == "weights":
                value["state"]["weights"]["embedding"][0] = float("nan")
            elif field == "evidence":
                value["evidence"]["accuracy"] = 1.0
            elif field == "index":
                value["index"] = {"layers": True}
            else:
                value["state"]["symbols"].append(value["state"]["symbols"][0])
            checkpoint.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                activate_base(self.root, run_id)
            self.assertFalse((self.root / DIRECTORY / "active.json").exists())


if __name__ == "__main__":
    unittest.main()
