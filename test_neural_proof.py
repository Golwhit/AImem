import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import neural_proof as proof
from neural_counting import KneserNey
from neural_experiment import EMBEDDING, HIDDEN, NeuralModel
from neural_layers import DevelopmentalPrior
import test_neural_efficiency


def chapter_html(verses: list[str]) -> bytes:
    body = "".join(f'<span class="verse" id="V{i}">{i}&#160;</span>{text} ' for i, text in enumerate(verses, 1))
    return f'<html><body><a href="x">nav</a><p>{body}</p><div class="footnote">note</div></body></html>'.encode()


class NeuralProofTests(unittest.TestCase):
    stage = test_neural_efficiency.NeuralEfficiencyTests.stage

    def small(self):
        return patch.multiple(
            proof, STEPS=40, MEASURE_EVERY=20, FRACTIONS=(0.5, 1.0),
            SEEDS=((17, 19, 23), (117, 119, 123)), SAMPLE_PROMPTS=("I a",), SAMPLE_LENGTH=20,
            FRESH_CHAPTERS=((4, 2), (5, 1)),
        )

    def test_kneser_ney_normalizes_and_uses_continuation_counts(self):
        source = "ab ab cb"
        model = NeuralModel(sorted(set(source)))
        kn = KneserNey(source, order=1)
        for prefix in ("", "a", "c", " ", "z"):
            weights = kn.prior(prefix, model)
            self.assertAlmostEqual(sum(weights), 1.0, places=12)
            self.assertTrue(all(weight > 0 for weight in weights))
        # "b" follows two distinct characters, so its continuation count is 2.
        self.assertEqual(kn.continuation[""]["b"], 2)
        self.assertEqual(kn.raw["a"]["b"], 2)
        # Continuation unigram: b=2, space=1, a=1, c=1 (total 5, 4 types), then uniform over 6 tokens.
        continuation_b = (2 - 0.75) / 5 + 0.75 * 4 / 5 / model.size
        self.assertAlmostEqual(kn.prior("a", model)[model.tokens["b"]], (2 - 0.75) / 2 + 0.75 / 2 * continuation_b)
        # Without context the top level uses raw counts instead: b occurs 3 times in 8 characters.
        self.assertAlmostEqual(kn.prior("", model)[model.tokens["b"]], (3 - 0.75) / 8 + 0.75 * 4 / 8 / model.size)
        with self.assertRaises(ValueError):
            KneserNey("")

    def test_layer_zero_v2_composes_unseen_blocks_from_known_pieces(self):
        source = "xabc bcd"
        model = NeuralModel(sorted(set(source)))
        v1 = DevelopmentalPrior(source, layer_one=False, layer_two=False, layer_zero_mode="v1")
        v2 = DevelopmentalPrior(source, layer_one=False, layer_two=False, layer_zero_mode="v2")
        self.assertEqual(v1.candidates, v2.candidates)
        self.assertFalse(v2.available("abcd", False))
        self.assertTrue(v2.possible("abcd", False))
        self.assertFalse(v2.possible("ac", False))
        self.assertFalse(v2.possible("acd", False))
        token = model.tokens["d"]
        self.assertEqual(v2.prior("xabc", model)[token], 1.0)
        self.assertLess(v1.prior("xabc", model)[token], 1.0)
        self.assertEqual(v2.prior("xaba", model)[model.tokens["c"]], 0.05)
        self.assertEqual(v2.evidence()["layer_zero_mode"], "v2")
        self.assertFalse(any(item["confirmed"] for item in v2.evidence()["candidates"]))
        with self.assertRaises(ValueError):
            DevelopmentalPrior(source, layer_zero_mode="v3")

    def test_operation_formulas_and_generation_violation_counting(self):
        model = NeuralModel(["a", "b"])
        inputs = model.context * EMBEDDING
        forward = inputs + inputs * HIDDEN + HIDDEN + HIDDEN * model.size + 3 * model.size
        self.assertEqual(proof.neural_forward_operations(model, prior=False), forward)
        parameters = sum(map(len, model.weights.values()))
        self.assertEqual(
            proof.neural_update_operations(model, prior=True),
            forward + model.size + model.size + 2 * HIDDEN * model.size + HIDDEN + 2 * inputs * HIDDEN + parameters,
        )
        only = NeuralModel(["a"])
        with patch.multiple(proof, SAMPLE_PROMPTS=("xyz ",), SAMPLE_LENGTH=10):
            unconstrained = proof.generate(only, None, True, 1)
            self.assertGreater(unconstrained["rule_violations"], 0)
            ruled = proof.generate(NeuralModel(["a", " "]), DevelopmentalPrior("a a", layer_zero=False), True, 1)
            self.assertEqual(ruled["rule_violations"], 0)

    def test_development_preregistration_and_single_fresh_evaluation(self):
        with tempfile.TemporaryDirectory() as directory, self.small():
            root = Path(directory)
            self.stage(root, held="A seed can grow. I am a plant.", train="I am a seed. A seed can grow. A plant can grow.")
            with self.assertRaises(ValueError):
                proof.run_development(root, workers=1)
            report = proof.run_development(root, acknowledge_review=True, workers=1)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(len(report["results"]), 2 * 2 * len(proof.ARMS))
            self.assertIn(report["summary"]["primary_selection"], proof.LAYERED)
            half = [r for r in report["results"] if r["fraction"] == 0.5]
            length = len("I am a seed. A seed can grow. A plant can grow.")
            self.assertTrue(all(r["training_characters"] == round(length * 0.5) for r in half))
            baseline = [r for r in report["results"] if r["arm"] == "baseline" and r["fraction"] == 1.0]
            self.assertNotEqual(baseline[0]["draws_sha256"], baseline[1]["draws_sha256"])
            counts = [r for r in report["results"] if r["arm"] == "layers_12" and r["fraction"] == 1.0]
            self.assertEqual(counts[0]["final_validation_loss"], counts[1]["final_validation_loss"])
            self.assertTrue(all(r["generation"]["rule_violations"] == 0 for r in report["results"] if 1 in r["layers"]))
            saved = json.loads((root / proof.PROOF_DIRECTORY / report["run_id"] / "report.json").read_text())
            self.assertNotIn("weights", saved["results"][0])
            public = json.loads((root / ".data/neural-experiments/latest-proof.json").read_text())
            self.assertNotIn("results", public)
            self.assertNotIn("final_test", json.dumps(saved["results"]))

            with self.assertRaisesRegex(ValueError, "Pre-register"):
                proof.stage_fresh(root, fetcher=lambda url: b"")
            registration = proof.preregister(root, report["run_id"])
            self.assertEqual(registration["primary_arm"], proof.select_primary(report["summary"]))
            self.assertIn(registration["primary_arm"], proof.LAYERED)
            with self.assertRaisesRegex(ValueError, "already exists"):
                proof.preregister(root, report["run_id"])

            (root / ".data/acts-web-staged/manifest.json").write_text(json.dumps({
                **json.loads((root / ".data/acts-web-staged/manifest.json").read_text()),
                "translation": "fixture", "rights_url": "fixture-rights", "rights_statement": "fixture",
                "preprocessing": "fixture",
            }))
            pages = {
                "fixture-rights": b"Original fixture rights",
                "https://ebible.org/engwebp/ACT04.htm": chapter_html(["A seed can be.", "I can grow."]),
                "https://ebible.org/engwebp/ACT05.htm": chapter_html(["A plant is a seed."]),
            }
            manifest = proof.stage_fresh(root, fetcher=pages.__getitem__)
            self.assertTrue(manifest["rights_matches_original_snapshot"])
            fresh, _, _ = proof.load_fresh(root)
            self.assertEqual(fresh, "A seed can be.\nI can grow.\nA plant is a seed.\n")
            with self.assertRaisesRegex(ValueError, "already staged"):
                proof.stage_fresh(root, fetcher=pages.__getitem__)

            result = proof.evaluate_fresh(root, acknowledge_review=True, workers=1)
            self.assertEqual(result["status"], "completed")
            self.assertIn("C1_similar_result", result["verdicts"])
            self.assertTrue(all(r["coverage"] <= 1 for r in result["rows"]))
            with self.assertRaisesRegex(ValueError, "already evaluated"):
                proof.evaluate_fresh(root, acknowledge_review=True, workers=1)

    def test_fresh_evaluation_refuses_changed_development_report(self):
        with tempfile.TemporaryDirectory() as directory, self.small():
            root = Path(directory)
            self.stage(root)
            report = proof.run_development(root, acknowledge_review=True, workers=1)
            proof.preregister(root, report["run_id"])
            path = root / proof.PROOF_DIRECTORY / report["run_id"] / "report.json"
            path.write_text(path.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "changed"):
                proof.evaluate_fresh(root, acknowledge_review=True, workers=1)

    def test_primary_selection_prefers_cheapest_within_tolerance(self):
        def row(loss, ops):
            return {"1.0": {"validation": {"mean": loss}, "training_operations": {"mean": ops}}}
        arms = {name: row(9.0, 1) for name in proof.LAYERED}
        arms.update(neural_12=row(2.00, 1000), layers_12=row(2.015, 10), layers_2=row(2.03, 1))
        arms["layers_02_v2"] = {"1.0": {"validation": None, "training_operations": {"mean": 0}}}
        self.assertEqual(proof.select_primary({"arms": arms}), "layers_12")

    def test_extract_rejects_wrong_verse_inventory(self):
        self.assertEqual(proof.extract(chapter_html(["One  two.", "Three"]), 4, 2), "One two.\nThree\n")
        with self.assertRaises(ValueError):
            proof.extract(chapter_html(["One"]), 4, 2)


if __name__ == "__main__":
    unittest.main()
