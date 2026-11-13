import json
from pathlib import Path
import tempfile
import unittest

from neural_experiment import NeuralModel, status, train_experiment


class NeuralExperimentTests(unittest.TestCase):
    def test_backprop_matches_finite_differences_in_every_weight_group(self):
        model = NeuralModel(list("ab "))
        context, target = model.examples("aba")[2]
        _, gradients = model.loss_and_gradients(context, target)
        epsilon = 1e-5
        for name, values in model.weights.items():
            # Include all embeddings: repeated boundary tokens must accumulate gradients.
            indices = range(len(values)) if name == "embedding" else {0, len(values) // 2, len(values) - 1}
            for index in indices:
                original = values[index]
                values[index] = original + epsilon
                plus = model.loss([(context, target)])
                values[index] = original - epsilon
                minus = model.loss([(context, target)])
                values[index] = original
                self.assertAlmostEqual(gradients[name][index], (plus - minus) / (2 * epsilon), places=7)

    def test_learning_and_persistence_without_other_model_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(status(root)["ready"])
            report = train_experiment({"synthetic": True}, root)
            self.assertGreater(report["changed_parameters"], 0)
            self.assertEqual(report["parameter_count"], 25 * report["vocabulary"] + 528)
            self.assertLess(report["history"][-1]["training_loss"], report["history"][0]["training_loss"] * 0.3)
            self.assertLess(report["history"][-1]["held_out_loss"], report["history"][0]["held_out_loss"] * 0.3)
            checkpoint = json.loads((root / ".data/neural-experiments" / f"{report['run_id']}.json").read_text())
            restored = NeuralModel(checkpoint["characters"])
            restored.weights = checkpoint["weights"]
            self.assertEqual(restored.sample(report["sample_prompt"]), report["sample"])
            self.assertEqual(report["parameter_count"], sum(map(len, restored.weights.values())))
            self.assertEqual(status(root)["report"], report)
            self.assertEqual(sorted(path.name for path in (root / ".data").iterdir()), ["neural-experiments"])

    def test_protected_condition_changes_forward_and_keeps_backprop_correct(self):
        model = NeuralModel(list("ab "))
        context, target = model.examples("aba")[2]
        original = model.forward(context)[2]
        features = tuple((i - 8) / 10 for i in range(16))
        model.set_reference_condition(features)
        self.assertNotEqual(model.forward(context)[2], original)
        _, gradients = model.loss_and_gradients(context, target)
        epsilon = 1e-5
        for name, values in model.weights.items():
            for index in {0, len(values) // 2, len(values) - 1}:
                before = values[index]
                values[index] = before + epsilon
                plus = model.loss([(context, target)])
                values[index] = before - epsilon
                minus = model.loss([(context, target)])
                values[index] = before
                self.assertAlmostEqual(gradients[name][index], (plus - minus) / (2 * epsilon), places=7)
        model.update(context, target)
        self.assertEqual(model.reference_condition, features)
        fresh = NeuralModel(list("ab "))
        fresh.set_reference_condition(features)
        fresh.set_reference_condition(None)
        self.assertEqual(fresh.forward(context)[2], original)
        with self.assertRaises(ValueError):
            model.set_reference_condition((float("nan"),) * 16)
        with self.assertRaises(ValueError):
            model.set_reference_condition((0.0,))

    def test_approval_and_input_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for payload in [
                {}, {"synthetic": True, "steps": True}, {"synthetic": True, "steps": 1001},
                {"training_text": "abcdefgh", "held_out_text": "abcdefgh", "source": "Mine", "acknowledge_review": True},
            ]:
                with self.assertRaises(ValueError):
                    train_experiment(payload, root)
            self.assertFalse((root / ".data").exists())

    def test_vocabulary_uses_training_only_and_held_out_is_not_trained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {"training_text": "ab ab ab ab", "held_out_text": "bc bc bc bc",
                       "source": "Original test fixture", "acknowledge_review": True, "steps": 5}
            first = train_experiment(payload, root)
            second = train_experiment({**payload, "held_out_text": "zz zz zz zz"}, root)
            def checkpoint(report):
                return json.loads((root / ".data/neural-experiments" / f"{report['run_id']}.json").read_text())
            self.assertNotIn("c", checkpoint(first)["characters"])
            self.assertGreater(first["held_out_unknown_characters"], 0)
            self.assertEqual(checkpoint(first)["weights"], checkpoint(second)["weights"])
