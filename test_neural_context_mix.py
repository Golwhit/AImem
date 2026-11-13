from copy import deepcopy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from neural_context_mix import ContextMix, ContextTriple, compare


class ContextMixTests(unittest.TestCase):
    def test_triple_gradients_and_requested_protocol(self):
        model = ContextTriple(list("ab "), 17, contexts=(3, 5, 7))
        context, target = [0] * 5 + [2, 3], 4
        before = deepcopy(model)
        epsilon, rate = 1e-5, 0.03
        checks = []
        for branch_id, branch in enumerate(before.branches):
            for name in branch.weights:
                _, gradients = branch.loss_and_gradients(context[-branch.context:], target)
                index = max(range(len(gradients[name])), key=lambda i: abs(gradients[name][i]))
                old = branch.weights[name][index]
                branch.weights[name][index] = old + epsilon
                plus = -math.log(before.probabilities(context)[target])
                branch.weights[name][index] = old - epsilon
                minus = -math.log(before.probabilities(context)[target])
                branch.weights[name][index] = old
                checks.append((branch_id, name, index, (plus - minus) / (2 * epsilon)))
        gate_gradients = []
        for i in range(3):
            before.gates[i] = epsilon
            plus = -math.log(before.probabilities(context)[target])
            before.gates[i] = -epsilon
            minus = -math.log(before.probabilities(context)[target])
            before.gates[i] = 0.0
            gate_gradients.append((plus - minus) / (2 * epsilon))
        model.update(context, target, rate)
        for i, name, index, gradient in checks:
            self.assertAlmostEqual(model.branches[i].weights[name][index],
                                   before.branches[i].weights[name][index] - rate * gradient, places=9)
        for gate, gradient in zip(model.gates, gate_gradients):
            self.assertAlmostEqual(gate, -rate * gradient, places=9)
        self.assertAlmostEqual(sum(model.branch_weights()), 1)
        with tempfile.TemporaryDirectory() as folder:
            dataset = {"training": "ab ba " * 8, "validation": "ba ab " * 4, "manifest_sha256": "fixture"}
            with patch("neural_context_mix.load_acts", return_value=dataset):
                report = compare(Path(folder), acknowledge_review=True, steps=4,
                                 contexts=(3, 5, 7), pair_only=True)
            self.assertEqual(set(report["summary"]), {"context4", "mixed3_5_7"})
            self.assertEqual(len(report["runs"]), 6)
            for entry in report["runs"]:
                if entry["arm"] == "mixed3_5_7":
                    self.assertEqual(entry["parameters"], 3 * (25 * 5 + 16) + 128 * 15 + 3)
                    self.assertEqual(len(entry["branch_weights"]), 3)

    def test_exact_mixture_gradients(self):
        model = ContextMix(list("ab "), 17)
        context = [0] * 14 + [2, 3]
        target = 4
        before = deepcopy(model)
        epsilon, rate = 1e-5, 0.03
        original_gate = before.gate
        before.gate = original_gate + epsilon
        plus = -math.log(before.probabilities(context)[target])
        before.gate = original_gate - epsilon
        minus = -math.log(before.probabilities(context)[target])
        gate_gradient = (plus - minus) / (2 * epsilon)
        before.gate = original_gate
        derivatives = []
        for branch in (before.short, before.long):
            for name in branch.weights:
                # Choose a nonzero coordinate for each group where possible.
                _, gradients = branch.loss_and_gradients(context[-branch.context:], target)
                index = max(range(len(gradients[name])), key=lambda i: abs(gradients[name][i]))
                old = branch.weights[name][index]
                branch.weights[name][index] = old + epsilon
                plus = -math.log(before.probabilities(context)[target])
                branch.weights[name][index] = old - epsilon
                minus = -math.log(before.probabilities(context)[target])
                branch.weights[name][index] = old
                derivatives.append((branch is before.long, name, index, (plus - minus) / (2 * epsilon)))
        model.update(context, target, rate)
        self.assertAlmostEqual(model.gate, original_gate - rate * gate_gradient, places=9)
        for long, name, index, gradient in derivatives:
            branch, prior = (model.long, before.long) if long else (model.short, before.short)
            self.assertAlmostEqual(branch.weights[name][index],
                                   prior.weights[name][index] - rate * gradient, places=9)
        self.assertAlmostEqual(sum(model.probabilities(context)), 1.0)

    def test_small_comparison_and_approval(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(ValueError):
                compare(root, acknowledge_review=False)
            dataset = {"training": "ab ba " * 8, "validation": "ba ab " * 4, "manifest_sha256": "fixture"}
            with patch("neural_context_mix.load_acts", return_value=dataset) as loader:
                report = compare(root, acknowledge_review=True, steps=4)
            loader.assert_called_once_with(root, True, include_final_test=False)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(len(report["runs"]), 12)
            for seed in report["seeds"]:
                rows = [r for r in report["runs"] if r["seed"] == seed]
                self.assertEqual(len({r["draws_sha256"] for r in rows}), 1)
                mix = next(r for r in rows if r["arm"] == "mixed4_16")
                control = next(r for r in rows if r["arm"] == "single_capacity_control")
                self.assertLessEqual(abs(mix["parameters"] - control["parameters"]), 64)
            self.assertFalse((root / ".data/neural-chat").exists())

    def test_five_twelve_pair_uses_requested_windows(self):
        model = ContextMix(list("ab "), 17, contexts=(5, 12))
        self.assertEqual((model.short.context, model.long.context), (5, 12))
        context = [0] * 10 + [2, 3]
        short = model.short.forward(context[-5:])[2]
        long = model.long.forward(context)[2]
        self.assertEqual(model.probabilities(context), [(a + b) / 2 for a, b in zip(short, long)])
        before = model.probabilities(context)[4]
        model.update(context, 4, 0.03)
        self.assertGreater(model.probabilities(context)[4], before)
        for contexts in ((0, 12), (12, 5), (5, 129), (True, 12)):
            with self.assertRaises(ValueError):
                ContextMix(list("ab "), 17, contexts=contexts)
        with tempfile.TemporaryDirectory() as folder:
            dataset = {"training": "ab ba " * 8, "validation": "ba ab " * 4, "manifest_sha256": "fixture"}
            with patch("neural_context_mix.load_acts", return_value=dataset):
                report = compare(Path(folder), acknowledge_review=True, steps=4,
                                 contexts=(5, 12), pair_only=True)
            self.assertEqual(report["mixture_contexts"], [5, 12])
            self.assertEqual(len(report["runs"]), 6)
            self.assertEqual(set(report["summary"]), {"context4", "mixed5_12"})
            for seed in report["seeds"]:
                self.assertEqual(len({r["draws_sha256"] for r in report["runs"] if r["seed"] == seed}), 1)


if __name__ == "__main__":
    unittest.main()
