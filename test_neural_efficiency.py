import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from neural_efficiency import first_match, run_efficiency
from neural_experiment import NeuralModel, status
from neural_layers import DevelopmentalPrior, evaluate, local_exclusion, sample
from seed_life import ALIVE_SEEDS, LAYER_ZERO_SEEDS, boundary_text, exclusion_reason


class NeuralEfficiencyTests(unittest.TestCase):
    def stage(self, root, held="I am a plant. A seed grows.", train="I am a seed. A seed can grow."):
        data = root / ".data/acts-web-staged"
        data.mkdir(parents=True, exist_ok=True)
        rights = b"Original fixture rights"
        (data / "rights.html").write_bytes(rights)
        sources = []
        for role, chapter, text in [("training", 1, train), ("validation", 2, held)]:
            raw = f"Original fixture chapter {chapter}".encode()
            (data / f"acts-{chapter:02}.html").write_bytes(raw)
            (data / f"{role}.txt").write_text(text)
            sources.append({
                "role": role, "chapter": chapter, "characters": len(text),
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            })
        (data / "manifest.json").write_text(json.dumps({
            "rights_sha256": hashlib.sha256(rights).hexdigest(), "sources": sources,
        }))

    def test_layered_gradients_match_finite_differences_and_change_updates(self):
        source = "I am a seed. A seed can grow."
        model = NeuralModel(sorted(set(source)))
        prior = DevelopmentalPrior(source).prior("a s", model)
        context, target = model.examples(source)[8]
        _, gradients = model.loss_and_gradients(context, target, output_prior=prior)
        for name, values in model.weights.items():
            for index in {0, len(values) // 2, len(values) - 1}:
                original = values[index]
                values[index] = original + 1e-5
                plus = -math.log(model.forward(context, output_prior=prior)[2][target])
                values[index] = original - 1e-5
                minus = -math.log(model.forward(context, output_prior=prior)[2][target])
                values[index] = original
                self.assertAlmostEqual(gradients[name][index], (plus - minus) / 2e-5, places=7)
        original = NeuralModel(sorted(set(source)))
        model.update(context, target, learning_rate=0.03, output_prior=prior)
        original.update(context, target, learning_rate=0.03)
        self.assertNotEqual(model.weights, original.weights)
        for invalid in ([0.0] * model.size, [1.0], [float("nan")] * model.size, [-1.0] * model.size):
            with self.assertRaises(ValueError):
                model.forward(context, output_prior=invalid)
        mask = [1.0] * model.size
        mask[target] = 0
        with self.assertRaisesRegex(ValueError, "excluded"):
            model.update(context, target, output_prior=mask)

    def test_foundation_candidate_growth_and_each_ablation_affect_probabilities(self):
        source = "I am a seed. A seed can grow."
        model = NeuralModel(sorted(set(source)))
        full = DevelopmentalPrior(source)
        evidence = full.evidence()
        self.assertEqual(evidence["foundation"], list(LAYER_ZERO_SEEDS))
        self.assertEqual(evidence["confirmed_forms"], sorted(ALIVE_SEEDS))
        self.assertTrue(full.available("I", True))
        self.assertFalse(full.confirmed("I", True))
        self.assertTrue(full.available("see", False))
        self.assertFalse(full.confirmed("see", False))
        self.assertFalse(full.available("zzz", False))
        for candidate in evidence["candidates"]:
            self.assertEqual(candidate["state"], "candidate")
            self.assertFalse(candidate["confirmed"])
            start = candidate["at_source_start"]
            if candidate["components"]:
                self.assertTrue(full.available(candidate["components"][0], start))
                self.assertTrue(full.available(candidate["components"][1], False))
        self.assertFalse(evidence["permanent_admissions_changed"])
        for flags, prefix in [
            ({"layer_zero": False}, "zz"),
            ({"layer_one": False}, " i"),
            ({"layer_two": False}, "a s"),
        ]:
            self.assertNotEqual(full.prior(prefix, model), DevelopmentalPrior(source, **flags).prior(prefix, model))
        self.assertGreater(full.prior("zzz", model)[model.tokens["a"]], 0)
        self.assertGreater(full.prior("a s", model)[1], 0)
        generated = sample(model, full, "I ", neural=True)
        self.assertIsNone(exclusion_reason(boundary_text("I " + generated, True)))

    def test_exclusion_boundaries_and_raw_conflicting_targets_are_not_hidden(self):
        for text in ("I.", "t ", "the ", "a i b", "ab aaa", "I am", "a I."):
            for offset, char in enumerate(text):
                prefix = text[:offset]
                local = local_exclusion(prefix[-4:], char)
                previous = exclusion_reason(boundary_text(prefix, True))
                expected = exclusion_reason(boundary_text(prefix + char, True))
                if previous is None:
                    self.assertEqual(local is not None, expected is not None)
        source = "I am a seed. A seed can grow."
        prior = DevelopmentalPrior(source)
        model = NeuralModel(sorted(set(source)))
        text = "a i "
        priors = [prior.prior(text[:i], model) for i in range(len(text))]
        metrics = evaluate(model, model.examples(text), priors, neural=True)
        self.assertIsNone(metrics["loss"])
        self.assertEqual(metrics["blocked_offsets"], [3])
        self.assertEqual(metrics["scored_targets"], 4)
        self.assertEqual(metrics["coverage"], 0.75)
        self.assertIsNone(first_match([{"validation": metrics}], 100))
        conflicts = DevelopmentalPrior("a i aaa").evidence()["source_rule_conflicts"]
        self.assertTrue(conflicts)

    def test_persistence_effort_and_held_out_isolation_without_final_test_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.stage(root)
            base = root / ".data/neural-experiments"
            base.mkdir()
            sentinels = {
                name: json.dumps({"existing": name}).encode()
                for name in ("latest.json", "latest-context.json", "latest-rate.json")
            }
            for name, content in sentinels.items():
                (base / name).write_bytes(content)
            with self.assertRaises(ValueError):
                run_efficiency(root)
            with patch("neural_efficiency.STEPS", 2), patch("neural_efficiency.MEASURE_EVERY", 1):
                report = run_efficiency(root, acknowledge_review=True)
            self.assertEqual(status(root)["layered_comparison"], report)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["final_test_evaluations"], 0)
            self.assertEqual(len(report["models"]), 6)
            self.assertEqual(
                report["target_validation_loss"], report["models"][0]["history"][-1]["validation"]["loss"],
            )
            for name, content in sentinels.items():
                self.assertEqual((base / name).read_bytes(), content)
            def checkpoint(run, arm):
                steps = 0 if arm == "layers_only" else 2
                return json.loads((base / run["run_id"] / arm / f"step-{steps:05}.json").read_text())
            for result in report["models"]:
                arm = result["name"]
                saved = checkpoint(report, arm)
                self.assertEqual(result["status"], "completed")
                self.assertEqual(saved["result"]["status"], "completed")
                self.assertEqual(result["gradient_updates"], 2 if result["neural"] else 0)
                self.assertGreater(result["training_preparation_seconds"], 0)
                self.assertGreater(result["history"][-1]["validation_prediction_seconds"], 0)
                self.assertEqual(result["source_characters_read"], len("I am a seed. A seed can grow."))
                if arm != "baseline":
                    self.assertGreater(result["layer_state_json_bytes"], 0)
                evidence = json.loads((base / report["run_id"] / arm / "evidence.json").read_text())
                if saved["weights"] is not None:
                    model = NeuralModel(saved["characters"])
                    model.weights = saved["weights"]
                    flags = result["layers"]
                    prior = DevelopmentalPrior(
                        evidence["training_text"], layer_zero=0 in flags, layer_one=1 in flags, layer_two=2 in flags,
                    ) if flags else None
                    text = evidence["held_out_text"]
                    priors = [prior.prior(text[:i][-4:], model) for i in range(len(text))] if prior else None
                    self.assertEqual(
                        evaluate(model, model.examples(text), priors, neural=True),
                        result["history"][-1]["validation"],
                    )
            self.stage(root, "zzzzzzzz")
            with patch("neural_efficiency.STEPS", 2), patch("neural_efficiency.MEASURE_EVERY", 1):
                changed = run_efficiency(root, acknowledge_review=True)
            for arm in [result["name"] for result in report["models"]]:
                self.assertEqual(checkpoint(report, arm)["weights"], checkpoint(changed, arm)["weights"])
                def evidence(run):
                    return json.loads((base / run["run_id"] / arm / "evidence.json").read_text())["evidence"]
                self.assertEqual(evidence(report), evidence(changed))
            self.assertFalse((root / ".data/seed-development.json").exists())

    def test_numeric_failure_is_explicit_and_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.stage(root)
            with patch("neural_efficiency.STEPS", 1), patch.object(NeuralModel, "update", side_effect=ArithmeticError("fixture failure")):
                with self.assertRaisesRegex(ArithmeticError, "fixture failure"):
                    run_efficiency(root, acknowledge_review=True)
            report = status(root)["layered_comparison"]
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["error"], "fixture failure")

    def test_conflicting_training_is_counted_and_cannot_win_with_filtered_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = "aaa a i aaa a i aaa a i aaaa"
            self.stage(root, held="a i aaaa", train=source)
            with patch("neural_efficiency.STEPS", 30), patch("neural_efficiency.MEASURE_EVERY", 30):
                report = run_efficiency(root, acknowledge_review=True)
            result = next(model for model in report["models"] if model["name"] == "full_012")
            self.assertGreater(result["skipped_rule_conflicts"], 0)
            self.assertEqual(result["gradient_updates"] + result["skipped_rule_conflicts"], 30)
            self.assertIsNone(result["history"][-1]["training"]["loss"])
            self.assertIsNone(result["history"][-1]["validation"]["loss"])
            self.assertIsNone(result["first_match"])
            path = root / ".data/neural-experiments" / report["run_id"] / "full_012/evidence.json"
            evidence = json.loads(path.read_text())
            self.assertEqual(evidence["training_text"], source)
            self.assertTrue(evidence["evidence"]["source_rule_conflicts"])
