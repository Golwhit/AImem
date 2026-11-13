from copy import deepcopy
import inspect
import math
from pathlib import Path
import unittest

from neural_base_lab import trace
from neural_experiment import NeuralModel


class BaseLabTests(unittest.TestCase):
    def test_trace_is_exact_core_training_with_no_persistence(self):
        payload = {"text": "ab ab ab ", "updates": 12, "rate": 0.1}
        result = trace(payload)
        model = NeuralModel(sorted(set(payload["text"])), seed=17)
        self.assertEqual(result["initial_weights"], model.weights)
        examples = model.examples(payload["text"])
        self.assertEqual(len(result["curve"]), 13)
        self.assertEqual(result["curve"][0]["corpus_loss"], model.loss(examples))
        for step in range(12):
            context, target = examples[step % len(examples)]
            before = deepcopy(model.weights)
            loss, gradients = model.loss_and_gradients(context, target)
            model.update(context, target, learning_rate=0.1)
            self.assertEqual(result["curve"][step + 1]["corpus_loss"], model.loss(examples))
        self.assertEqual(context, result["context"])
        self.assertEqual(target, result["target"])
        self.assertEqual(loss, result["loss_before"])
        for group in result["weights"]:
            name = group["name"]
            self.assertEqual(before[name], group["before"])
            self.assertEqual(gradients[name], group["gradient"])
            self.assertEqual(model.weights[name], group["after"])
            for old, gradient, new in zip(group["before"], group["gradient"], group["after"]):
                self.assertEqual(old - 0.1 * gradient, new)
        x, hidden, probabilities = model.forward(context)
        self.assertEqual(x, result["after"]["embedding"])
        self.assertEqual(hidden, result["after"]["hidden"])
        self.assertEqual(probabilities, result["after"]["probabilities"])
        self.assertEqual(-math.log(probabilities[target]), result["loss_after"])
        self.assertEqual(result, trace(payload))
        self.assertTrue(result["isolated"])

    def test_intermediate_values_and_source_are_real(self):
        result = trace({})
        self.assertEqual(len(result["lessons"]), 11)
        source = Path(inspect.getfile(NeuralModel)).read_text().splitlines()
        for lesson in result["lessons"]:
            self.assertTrue(any(row["highlight"] for row in lesson["source"]["lines"]))
            self.assertTrue(any(row["note"] for row in lesson["source"]["lines"]))
            for row in lesson["source"]["lines"]:
                self.assertEqual(row["code"], source[row["number"] - 1])
        self.assertEqual(result["lessons"][0]["source"]["lines"][0]["number"], 1)
        self.assertEqual(len(result["before"]["embedding"]), 32)
        self.assertEqual(len(result["before"]["hidden"]), 16)
        self.assertAlmostEqual(sum(result["before"]["probabilities"]), 1.0)
        logits = result["before"]["logits"]
        denominator = sum(math.exp(value - max(logits)) for value in logits)
        for logit, probability in zip(logits, result["before"]["probabilities"]):
            self.assertEqual(math.exp(logit - max(logits)) / denominator, probability)
        self.assertLess(result["loss_after"], result["loss_before"])
        self.assertEqual(result["context"], [0] * 4)
        self.assertEqual(result["examples_per_pass"], len(result["text"]))

    def test_all_parameter_gradients_match_numerical_derivatives_for_selected_entries(self):
        result = trace({"text": "ab ", "updates": 4})
        model = NeuralModel(sorted(set(result["text"])), seed=17)
        for group in result["weights"]:
            model.weights[group["name"]] = group["before"][:]
        context, target = result["context"], result["target"]
        epsilon = 1e-5
        for group in result["weights"]:
            index = max(range(group["count"]), key=lambda i: abs(group["gradient"][i]))
            original = model.weights[group["name"]][index]
            model.weights[group["name"]][index] = original + epsilon
            plus = -math.log(model.forward(context)[2][target])
            model.weights[group["name"]][index] = original - epsilon
            minus = -math.log(model.forward(context)[2][target])
            model.weights[group["name"]][index] = original
            self.assertAlmostEqual((plus - minus) / (2 * epsilon), group["gradient"][index], places=7)

    def test_invalid_inputs_are_explicit_and_limits_work(self):
        for payload in (None, [], {"text": ""}, {"text": " "}, {"text": "\x00"},
                        {"text": "é"}, {"text": "x" * 129}, {"text": "abcdefghijklmnopqrstuvwxyz"},
                        {"updates": True}, {"updates": 0}, {"updates": 33},
                        {"rate": True}, {"rate": 0}, {"rate": 2}, {"rate": float("nan")}):
            with self.assertRaises(ValueError):
                trace(payload)
        result = trace({"text": "a", "updates": 32, "rate": 1})
        self.assertEqual(result["updates"], 32)
        self.assertTrue(math.isfinite(result["loss_after"]))


if __name__ == "__main__":
    unittest.main()
