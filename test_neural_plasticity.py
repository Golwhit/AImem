import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from neural_base_comparison import WordBase, digest
from neural_conversation_context import build
from neural_conversation_trial import answer_examples, assess, token_loss
from neural_experiment import HIDDEN, NeuralModel
from neural_plasticity import (
    ARMS, EPSILON, NormalizedModel, PlasticityContext, diagnostics, fit, initialize, replay, run,
)


class PlasticityTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"id": "red", "skill": "color", "turns": [
            {"role": "user", "content": "Mira has a red box. What color?"}], "answer": "red."}]
        parent = WordBase("red blue cup box has the. user assistant: \n?", 17)
        self.base, _ = build(parent, self.rows, "long_identities")

    def test_normalization_gradients_with_masks_repeats_and_condition(self):
        model = NormalizedModel(list("ab "), context=4)
        model.set_reference_condition(tuple((i - 8) / 10 for i in range(HIDDEN)))
        context, target = [0, 0, 2, 2], 3
        prior = [0.0, 0.0, 1.0, 1.0, 0.0]
        _, gradients = model.loss_and_gradients(context, target, output_prior=prior)
        epsilon = 1e-5
        for name, values in model.weights.items():
            indices = range(len(values)) if name == "embedding" else {0, min(7, len(values) - 1),
                                                                      len(values) // 2, len(values) - 1}
            for index in indices:
                before = values[index]
                values[index] = before + epsilon
                plus = -math.log(model.forward(context, output_prior=prior)[2][target])
                values[index] = before - epsilon
                minus = -math.log(model.forward(context, output_prior=prior)[2][target])
                values[index] = before
                self.assertAlmostEqual(gradients[name][index], (plus - minus) / (2 * epsilon), places=7)
        for name in ("embedding", "hidden", "hidden_bias"):
            model.weights[name] = [0.0] * len(model.weights[name])
        model.set_reference_condition(None)
        self.assertTrue(all(math.isfinite(v) for values in
                            model.loss_and_gradients(context, target, output_prior=prior)[1].values()
                            for v in values))

    def test_plain_forward_and_update_match_original_equations_exactly(self):
        model = NeuralModel(list("ab "), context=4)
        context = [0, 0, 2, 2]
        x, hidden, probabilities = model.forward(context)
        w = model.weights
        expected = [math.tanh(w["hidden_bias"][j] + sum(
            value * w["hidden"][i * HIDDEN + j] for i, value in enumerate(x)))
            for j in range(HIDDEN)]
        self.assertEqual(hidden, expected)
        _, gradients = model.loss_and_gradients(context, 3)
        delta = probabilities[:]
        delta[3] -= 1
        expected_delta = [sum(w["output"][j * model.size + k] * delta[k]
                              for k in range(model.size)) * (1 - hidden[j] ** 2)
                          for j in range(HIDDEN)]
        self.assertEqual(gradients["hidden_bias"], expected_delta)

    def test_initialization_isolated_and_architecture_survives_roundtrip(self):
        before = json.loads(json.dumps(self.base.snapshot()))
        inherited = initialize(self.base, "inherited", 17)
        for arm in ARMS:
            model = initialize(self.base, arm, 17)
            self.assertEqual(model.model.characters, self.base.model.characters)
            self.assertEqual(model.model.context, self.base.model.context)
            self.assertEqual(model.model.weights.keys(), self.base.model.weights.keys())
            self.assertEqual(sum(map(len, model.model.weights.values())),
                             sum(map(len, self.base.model.weights.values())))
            if arm != "fresh_scaled":
                self.assertEqual(model.model.weights, inherited.model.weights)
            else:
                self.assertNotEqual(model.model.weights, inherited.model.weights)
                self.assertEqual(model.snapshot(), initialize(self.base, arm, 17).snapshot())
                self.assertNotEqual(model.snapshot(), initialize(self.base, arm, 117).snapshot())
            restored = PlasticityContext.restore(json.loads(json.dumps(model.snapshot())))
            self.assertEqual(assess(model, self.rows), assess(restored, self.rows))
            context, _, prior = answer_examples(model, self.rows)[0]
            self.assertEqual(model.model.forward(context, output_prior=prior),
                             restored.model.forward(context, output_prior=prior))
        self.assertEqual(before, self.base.snapshot())
        state = inherited.snapshot()
        state["normalization_epsilon"] = EPSILON * 10
        with self.assertRaises(ValueError):
            PlasticityContext.restore(state)
        with self.assertRaises(ValueError):
            initialize(self.base, "other", 17)

    def test_cpu_budgeted_training_replays_all_non_timing_metrics(self):
        model = initialize(self.base, "inherited_normalized", 17)
        initial = json.loads(json.dumps(model.snapshot()))
        points = fit(model, self.rows, seed=17, rate=.01, cpu_checkpoints=(.001,))
        self.assertGreaterEqual(points[-1]["training_cpu_seconds"], .001)
        self.assertEqual(points[-1]["updates"] % 100, 0)
        restored = replay(initial, self.rows, seed=17, rate=.01, checkpoints=points)
        self.assertEqual(model.snapshot(), restored.snapshot())
        self.assertGreater(diagnostics(model, answer_examples(model, self.rows))["mean_tanh_derivative"], 0)
        with self.assertRaises(ValueError):
            fit(model, self.rows, seed=17, rate=.01, cpu_checkpoints=(0.0,))

    def test_normalized_model_fits_a_conflict_free_target(self):
        model = NormalizedModel(list("ab "), context=4)
        context, target = [0, 0, 2, 2], 3
        prior = [0.0, 0.0, 1.0, 1.0, 0.0]
        initial = model.loss_and_gradients(context, target, output_prior=prior)[0]
        for _ in range(200):
            model.update(context, target, output_prior=prior, learning_rate=.03)
        self.assertLess(model.loss_and_gradients(context, target, output_prior=prior)[0], .02)
        self.assertGreater(initial, .1)

    def test_review_gate_and_rate_selection_before_any_test(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "fixtures").mkdir()
            parent = WordBase("red blue cup box has the. user assistant: \n?", 17)
            data = {"train": self.rows, "test": [{"id": "test", "skill": "color",
                    "turns": [{"role": "user", "content": "Mira has a blue box. What color?"}],
                    "answer": "blue."}], "provenance": "Original test fixture"}
            raw = json.dumps(data).encode()
            (root / "fixtures/neural-conversation-draft.json").write_bytes(raw)
            with self.assertRaises(ValueError):
                run(root, acknowledge_review=False, reviewed_sha256=digest(raw))
            with self.assertRaises(ValueError):
                run(root, acknowledge_review=True, reviewed_sha256="0" * 64)
            self.assertFalse((root / ".data").exists())
            for name in ("neural_plasticity.py", "neural_experiment.py", "neural_conversation_context.py",
                         "neural_conversation_trial.py", "neural_base_comparison.py", "neural_continuation.py"):
                (root / name).write_bytes((Path(__file__).parent / name).read_bytes())
            calls = []

            def watched_assess(model, rows):
                if rows == data["test"]:
                    selections = list((root / ".data/neural-plasticity").glob("*/selection.json"))
                    self.assertEqual(len(selections), 1)
                    selection = json.loads(selections[0].read_text())
                    self.assertEqual(len(selection["selection"]), len(ARMS))
                    self.assertTrue(all("test" not in row for row in selection["selection"]))
                    calls.append(rows)
                return assess(model, rows)

            def brief_fit(model, rows, *, seed, rate, cpu_checkpoints):
                context, target, prior = answer_examples(model, rows)[0]
                model.model.update(context, target, output_prior=prior, learning_rate=rate)
                return [{"updates": 1, "teacher_forced_loss": token_loss(model, answer_examples(model, rows)),
                         "training": {"exact": 0.0}, "training_cpu_seconds": 0.0}]

            with patch("neural_plasticity.load_active", return_value=(
                    {"model": "original-word-punctuation-mlp-chat-v1"}, parent, None)), \
                    patch("neural_plasticity.SEEDS", (17,)), \
                    patch("neural_plasticity.fit", side_effect=brief_fit), \
                    patch("neural_plasticity.assess", side_effect=watched_assess):
                result = run(root, acknowledge_review=True, reviewed_sha256=digest(raw))
            self.assertEqual(len(calls), len(ARMS))
            self.assertEqual(result["status"], "completed")
            for chosen in result["selection"]:
                candidates = [t for t in result["trials"] if t["arm"] == chosen["arm"]]
                best = min(candidates, key=lambda t: t["checkpoints"][-1]["teacher_forced_loss"])
                self.assertEqual(chosen["rate"], best["rate"])


if __name__ == "__main__":
    unittest.main()
