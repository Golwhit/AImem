import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import neural_skeletons
from neural_experiment import NeuralModel, generate_checkpoint, train_experiment
from neural_skeleton_runtime import BANK_PATH, END, UnitCodec, build_bank, digest, load_bank


def fixture_bank(root: Path) -> dict:
    record = {
        "version": 1, "source": "Original test fixture", "learning_sha256": "fixture",
        "dictionary_sha256": "fixture", "experiment_code_sha256": "fixture",
        "policy": "Exact word confirmation", "patterns": {
            "hidden1": {"th*t": "that", "w*th": "with"}, "hidden2": {},
        },
    }
    record["sha256"] = digest(record)
    path = root / BANK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return record


class SkeletonRuntimeTests(unittest.TestCase):
    def test_build_uses_only_learning_and_development(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / neural_skeletons.SKELETON_DIRECTORY
            path.mkdir(parents=True)
            (path / "preregistration.json").write_text(json.dumps({
                "code_sha256": hashlib.sha256(Path(neural_skeletons.__file__).read_bytes()).hexdigest(),
            }))
            with patch.object(neural_skeletons, "load", return_value=("and that with them. " * 6, {})) as load:
                with patch.object(neural_skeletons, "dictionary", return_value=({"that", "with", "and", "them"}, "fixture")):
                    summary = build_bank(root)
            load.assert_called_once_with(root, ("learn", "develop"))
            self.assertGreater(summary["patterns"]["hidden1"], 0)
            self.assertNotIn("dropped1", load_bank(root)["patterns"])

    def test_lossless_reading_preserves_unknowns_names_quotes_and_spacing(self):
        with tempfile.TemporaryDirectory() as directory:
            codec = UnitCodec.from_bank(fixture_bank(Path(directory)), "that with that")
            text = "that\twith  That thatcher father’ don’t.\n"
            pieces = codec.encode(text)
            self.assertEqual("".join(pieces), text)
            self.assertEqual(pieces.count("that"), 1)
            self.assertEqual(pieces.count("with"), 1)
            self.assertNotIn("That", pieces)
            for unknown in ("thaté", "éthat", "that2", "2that", "_that", "that_"):
                self.assertEqual(codec.encode(unknown), list(unknown))
            self.assertEqual(codec.encode("that", open_tail=True), list("that"))
            reading = codec.reading(text)
            self.assertEqual(reading["decisions_saved"], 6)
            model = NeuralModel(sorted(set(text) | set(codec.units) | {END}))
            targets = [target for _, target in codec.examples(model, text)]
            self.assertEqual(targets.count(model.tokens["that"]), 1)
            self.assertEqual(targets.count(model.tokens["with"]), 1)
            self.assertEqual(targets[-1], model.tokens[END])
            self.assertEqual(len(targets), reading["decisions"] + 1)
            with self.assertRaisesRegex(ValueError, "reserved"):
                codec.encode(END)

    def test_units_are_selected_from_training_only(self):
        with tempfile.TemporaryDirectory() as directory:
            codec = UnitCodec.from_bank(fixture_bank(Path(directory)), "that that that")
            self.assertEqual(set(codec.units), {"that"})
            self.assertEqual(codec.encode("with"), list("with"))
            with self.assertRaisesRegex(ValueError, "Invalid runtime"):
                UnitCodec({"that": {"pattern": "*hat", "kind": "hidden1"}}, {})

    def test_generation_emits_complete_units_and_enforces_both_boundaries(self):
        codec = UnitCodec({"that": {"pattern": "th*t", "kind": "hidden1"}}, {})
        model = NeuralModel(sorted(set("that x") | {"that", END}))

        def forward(context):
            probabilities = [0.0] * model.size
            for piece, weight in {"that": 1e9, "x": 1e6, " ": 1000, END: 1}.items():
                probabilities[model.tokens[piece]] = weight
            return [], [], probabilities

        with patch.object(model, "forward", side_effect=forward):
            result = codec.sample(model, "", length=10)
            self.assertEqual(result["text"], "that that ")
            self.assertEqual(result["unit_occurrences"], 2)
            self.assertEqual(result["decisions"], 4)
            self.assertEqual(codec.sample(model, "", length=3)["unit_occurrences"], 0)
            self.assertTrue(codec.sample(model, "prefix", length=4)["text"].startswith("x"))
            self.assertTrue(codec.sample(model, "that", length=4)["text"].startswith("x"))

    def test_training_changes_weights_and_reloads_without_external_bank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture_bank(root)
            payload = {"training_text": "that with that. " * 6, "held_out_text": "with that unknown.",
                       "source": "Original test text", "acknowledge_review": True,
                       "steps": 20, "skeleton_units": True}
            report = train_experiment(payload, root)
            second = train_experiment({**payload, "held_out_text": "that novel words."}, root)
            path = root / ".data/neural-experiments"
            first_checkpoint = json.loads((path / f"{report['run_id']}.json").read_text())
            second_checkpoint = json.loads((path / f"{second['run_id']}.json").read_text())
            self.assertEqual(first_checkpoint["weights"], second_checkpoint["weights"])
            self.assertGreater(report["changed_parameters"], 0)
            self.assertGreater(report["unit_runtime"]["training"]["decisions_saved"], 0)
            self.assertIn("that", first_checkpoint["characters"])
            unit_index = first_checkpoint["characters"].index("that") + 2
            self.assertNotEqual(first_checkpoint["weights"]["output_bias"][unit_index], 0.0)
            (root / BANK_PATH).unlink()
            generated = generate_checkpoint({"run_id": report["run_id"], "prompt": report["sample_prompt"]}, root)
            self.assertEqual(generated["text"], report["sample"])
            self.assertEqual(generated["emitted_units"], report["unit_runtime"]["generation"]["emitted_units"])
            self.assertEqual(report["loss_unit"], "mixed unit/character decision (including end marker)")

    def test_bad_requests_and_missing_or_corrupt_bank_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "missing"):
                load_bank(root)
            fixture_bank(root)
            path = root / BANK_PATH
            path.write_text(path.read_text().replace("th*t", "t*at"))
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_bank(root)
            for payload in [
                {"synthetic": True, "skeleton_units": True},
                {"synthetic": True, "skeleton_units": "yes"},
            ]:
                with self.assertRaises(ValueError):
                    train_experiment(payload, root)
            for payload in [{"run_id": "../bad"}, {"run_id": "a" * 32, "length": True}]:
                with self.assertRaises(ValueError):
                    generate_checkpoint(payload, root)


if __name__ == "__main__":
    unittest.main()
