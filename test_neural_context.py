import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from neural_context import compare_contexts, compare_learning_rates
from neural_experiment import NeuralModel, status


class NeuralContextTests(unittest.TestCase):
    def stage(self, root, held="ab ba z"):
        directory = root / ".data/acts-web-staged"
        directory.mkdir(parents=True, exist_ok=True)
        rights = b"Original fixture rights record"
        (directory / "rights.html").write_bytes(rights)
        sources = []
        for role, chapter, text in [("training", 1, "ab ab ba ba"), ("validation", 2, held)]:
            raw = f"Original local fixture {chapter}".encode()
            (directory / f"acts-{chapter:02}.html").write_bytes(raw)
            (directory / f"{role}.txt").write_text(text)
            sources.append({
                "role": role, "chapter": chapter, "characters": len(text),
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
            })
        (directory / "manifest.json").write_text(json.dumps({
            "rights_sha256": hashlib.sha256(rights).hexdigest(), "sources": sources,
        }))

    def test_controlled_draws_saved_models_and_no_final_test_or_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.stage(root, "ab ba zz")
            base = root / ".data/neural-experiments"
            base.mkdir()
            old_report = b'{"sentinel": "existing Acts run"}'
            (base / "latest.json").write_bytes(old_report)
            with patch("neural_context.STEPS", 6), patch("neural_context.MEASURE_EVERY", 3):
                report = compare_contexts(root, acknowledge_review=True)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["final_test_evaluations"], 0)
            self.assertEqual((base / "latest.json").read_bytes(), old_report)
            self.assertEqual(status(root)["context_comparison"], report)
            self.assertEqual([m["context"] for m in report["models"]], [4, 16, 32])
            for result in report["models"]:
                context = result["context"]
                model = NeuralModel(sorted(set("ab ab ba ba")), context=context)
                expected = 25 * model.size + context * 128 + 16
                self.assertEqual(result["parameter_count"], expected)
                self.assertEqual(result["held_out_unknown_characters"], 2)
                self.assertEqual([row["step"] for row in result["history"]], [0, 3, 6])
                self.assertEqual(result["status"], "completed")
                examples = model.examples("ab ab ba ba")
                rng = random.Random(19)
                for _ in range(6):
                    model.update(*rng.choice(examples))
                checkpoint = json.loads((base / report["run_id"] / f"context-{context}" / "step-00006.json").read_text())
                self.assertEqual(checkpoint["weights"], model.weights)
                self.assertNotIn("test_text", checkpoint)
                self.assertEqual(checkpoint["context"], context)
                self.assertEqual(result["samples"][0]["text"], model.sample("The "))
                self.assertEqual(result["history"][-1]["training_loss"], model.loss(examples))
            self.stage(root, "xxxxxxxx")
            with patch("neural_context.STEPS", 6), patch("neural_context.MEASURE_EVERY", 3):
                changed = compare_contexts(root, acknowledge_review=True)
            for context in report["contexts"]:
                def weights(run):
                    path = base / run["run_id"] / f"context-{context}" / "step-00006.json"
                    return json.loads(path.read_text())["weights"]
                self.assertEqual(weights(report), weights(changed))

    def test_approval_hashes_and_failure_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.stage(root, "ab ba zz")
            with self.assertRaises(ValueError):
                compare_contexts(root)
            with patch("neural_context.STEPS", 1), patch.object(NeuralModel, "update", side_effect=ArithmeticError("fixture failure")):
                with self.assertRaisesRegex(ArithmeticError, "fixture failure"):
                    compare_contexts(root, acknowledge_review=True)
            report = status(root)["context_comparison"]
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["error"], "fixture failure")
            (root / ".data/acts-web-staged/training.txt").write_text("tampered text")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                compare_contexts(root, acknowledge_review=True)

    def test_context_comparison_uses_selected_rate_and_preserves_archives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.stage(root, "ab ba zz")
            with patch("neural_context.STEPS", 6), patch("neural_context.MEASURE_EVERY", 3):
                original = compare_contexts(root, acknowledge_review=True)
            base = root / ".data/neural-experiments"
            original_files = {
                path: path.read_bytes() for path in (base / original["run_id"]).rglob("*.json")
            }
            for invalid in (0, -0.1, True, float("inf"), float("nan")):
                with self.assertRaises(ValueError):
                    compare_contexts(root, acknowledge_review=True, learning_rate=invalid)
            with self.assertRaises(ValueError):
                compare_contexts(root, learning_rate=0.03)
            with patch("neural_context.STEPS", 6), patch("neural_context.MEASURE_EVERY", 3):
                report = compare_contexts(root, acknowledge_review=True, learning_rate=0.03)
            self.assertEqual(report["learning_rate"], 0.03)
            self.assertEqual(report["training_draws_sha256"], original["training_draws_sha256"])
            self.assertEqual(report["final_test_evaluations"], 0)
            self.assertEqual(status(root)["context_comparison"], report)
            for path, content in original_files.items():
                self.assertEqual(path.read_bytes(), content)
            for result in report["models"]:
                context = result["context"]
                model = NeuralModel(sorted(set("ab ab ba ba")), context=context)
                examples = model.examples("ab ab ba ba")
                rng = random.Random(19)
                for _ in range(6):
                    model.update(*rng.choice(examples), learning_rate=0.03)
                saved = json.loads((
                    base / report["run_id"] / f"context-{context}" / "step-00006.json"
                ).read_text())
                self.assertEqual(result["learning_rate"], 0.03)
                self.assertEqual(saved["learning_rate"], 0.03)
                self.assertEqual(saved["weights"], model.weights)
                self.assertEqual(result["history"][-1]["training_loss"], model.loss(examples))
                self.assertEqual(result["samples"][0]["text"], model.sample("The "))
                self.assertNotIn("test_text", saved)

    def test_learning_rate_changes_only_update_size_and_preserves_default(self):
        baseline = NeuralModel(list("ab "))
        context, target = baseline.examples("ab ba")[-1]
        _, gradients = baseline.loss_and_gradients(context, target)
        default = NeuralModel(list("ab "))
        default.update(context, target)
        for rate in (0.1, 0.03, 0.01):
            model = NeuralModel(list("ab "))
            self.assertEqual(model.weights, baseline.weights)
            model.update(context, target, learning_rate=rate)
            for name, values in baseline.weights.items():
                self.assertEqual(model.weights[name], [
                    value - rate * gradient for value, gradient in zip(values, gradients[name])
                ])
            if rate == 0.1:
                self.assertEqual(model.weights, default.weights)
        for invalid in (0, -0.1, True, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                baseline.update(context, target, learning_rate=invalid)

    def test_rate_comparison_is_matched_isolated_and_persistent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.stage(root, "ab ba zz")
            base = root / ".data/neural-experiments"
            base.mkdir()
            sentinels = {
                "latest.json": b'{"sentinel": "ordinary run"}',
                "latest-context.json": b'{"sentinel": "context run"}',
            }
            for name, content in sentinels.items():
                (base / name).write_bytes(content)
            with self.assertRaises(ValueError):
                compare_learning_rates(root)
            with patch("neural_context.STEPS", 6), patch("neural_context.MEASURE_EVERY", 3):
                report = compare_learning_rates(root, acknowledge_review=True)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["kind"], "rate-comparison")
            self.assertEqual(report["contexts"], [4])
            self.assertEqual(report["learning_rates"], [0.1, 0.03, 0.01])
            self.assertEqual(report["final_test_evaluations"], 0)
            self.assertEqual(status(root)["rate_comparison"], report)
            for name, content in sentinels.items():
                self.assertEqual((base / name).read_bytes(), content)
            initial = []
            def checkpoint(run, rate, step=6):
                path = base / run["run_id"] / f"rate-{rate:g}" / f"step-{step:05}.json"
                return json.loads(path.read_text())
            for result in report["models"]:
                rate = result["learning_rate"]
                model = NeuralModel(sorted(set("ab ab ba ba")))
                initial.append(checkpoint(report, rate, 0)["weights"])
                self.assertEqual(initial[-1], model.weights)
                self.assertEqual(result["context"], 4)
                self.assertEqual(result["parameter_count"], 25 * model.size + 528)
                self.assertEqual([row["step"] for row in result["history"]], [0, 3, 6])
                examples = model.examples("ab ab ba ba")
                rng = random.Random(19)
                for _ in range(6):
                    model.update(*rng.choice(examples), learning_rate=rate)
                saved = checkpoint(report, rate)
                self.assertEqual(saved["weights"], model.weights)
                self.assertEqual(saved["learning_rate"], rate)
                self.assertNotIn("test_text", saved)
                self.assertEqual(result["samples"][0]["text"], model.sample("The "))
                self.assertEqual(result["history"][-1]["training_loss"], model.loss(examples))
            self.assertEqual(initial[0], initial[1])
            self.assertEqual(initial[0], initial[2])
            self.stage(root, "xxxxxxxx")
            with patch("neural_context.STEPS", 6), patch("neural_context.MEASURE_EVERY", 3):
                changed = compare_learning_rates(root, acknowledge_review=True)
            for rate in report["learning_rates"]:
                self.assertEqual(checkpoint(report, rate)["weights"], checkpoint(changed, rate)["weights"])
            with patch("neural_context.STEPS", 1), patch.object(NeuralModel, "update", side_effect=ArithmeticError("rate failure")):
                with self.assertRaisesRegex(ArithmeticError, "rate failure"):
                    compare_learning_rates(root, acknowledge_review=True)
            failed = status(root)["rate_comparison"]
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["error"], "rate failure")

    def test_context_windows_gradients_and_legacy_default(self):
        for context_size in (4, 16, 32):
            model = NeuralModel(list("ab "), context=context_size)
            examples = model.examples("ab ba")
            self.assertEqual(len(examples), 5)
            self.assertEqual(examples[0][0], [0] * context_size)
            self.assertEqual(examples[1][0], [0] * (context_size - 1) + [2])
            context, target = examples[-1]
            _, gradients = model.loss_and_gradients(context, target)
            for name, values in model.weights.items():
                for index in {0, len(values) // 2, len(values) - 1}:
                    original = values[index]
                    values[index] = original + 1e-5
                    plus = model.loss([(context, target)])
                    values[index] = original - 1e-5
                    minus = model.loss([(context, target)])
                    values[index] = original
                    self.assertAlmostEqual(gradients[name][index], (plus - minus) / 2e-5, places=7)
        self.assertEqual(NeuralModel(list("ab ")).weights, NeuralModel(list("ab "), context=4).weights)
        for invalid in (0, 129, True, 4.5):
            with self.assertRaises(ValueError):
                NeuralModel(list("ab "), context=invalid)
