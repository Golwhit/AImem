import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from neural_experiment import NeuralModel, train_experiment
from neural_rounds import TEST, TRAIN, VALIDATION, load_acts, run_rounds


class NeuralRoundsTests(unittest.TestCase):
    def test_real_text_requires_explicit_approval_and_verified_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / ".data/acts-web-staged"
            data.mkdir(parents=True)
            rights = b"Public-domain fixture rights record"
            (data / "rights.html").write_bytes(rights)
            sources = []
            for role, chapter, text in [
                ("training", 1, TRAIN), ("validation", 2, VALIDATION), ("final_test", 3, TEST),
            ]:
                raw = f"Original local fixture {chapter}".encode()
                (data / f"acts-{chapter:02}.html").write_bytes(raw)
                (data / f"{role}.txt").write_text(text)
                sources.append({
                    "role": role, "chapter": chapter, "characters": len(text),
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "raw_sha256": hashlib.sha256(raw).hexdigest(),
                })
            (data / "manifest.json").write_text(json.dumps({
                "rights_sha256": hashlib.sha256(rights).hexdigest(), "sources": sources,
            }))
            with self.assertRaises(ValueError):
                load_acts(root, False)
            with patch("neural_rounds.ROUNDS", 1), patch("neural_rounds.UPDATES", 2):
                report = run_rounds(root, acts=True, acknowledge_review=True)
            self.assertFalse(report["synthetic"])
            self.assertTrue(report["automated_rounds"])
            self.assertEqual(report["training_sha256"], sources[0]["text_sha256"])
            self.assertEqual(report["final_test_evaluations"], 1)
            self.assertIn("not scripture", report["evaluation_note"])
            parent_path = root / ".data/neural-experiments" / f"{report['run_id']}.json"
            original_parent = parent_path.read_bytes()
            with patch("neural_rounds.UPDATES", 2):
                continued = run_rounds(
                    root, acts=True, acknowledge_review=True, resume_run=report["run_id"],
                    additional_rounds=2,
                )
            self.assertEqual(continued["rounds_completed"], 3)
            self.assertEqual(continued["steps"], 6)
            self.assertEqual(continued["final_test_evaluations"], 0)
            self.assertNotIn("final_test_loss", continued)
            self.assertEqual(continued["history"][0]["training_loss"], report["history"][-1]["training_loss"])
            self.assertEqual(parent_path.read_bytes(), original_parent)
            with patch("neural_rounds.ROUNDS", 3), patch("neural_rounds.UPDATES", 2):
                uninterrupted = run_rounds(root, acts=True, acknowledge_review=True)
            base = root / ".data/neural-experiments"
            a = json.loads((base / f"{continued['run_id']}.json").read_text())
            b = json.loads((base / f"{uninterrupted['run_id']}.json").read_text())
            self.assertEqual(a["weights"], b["weights"])
            (data / "training.txt").write_text("tampered source")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_acts(root, True)

    def test_fifty_rounds_save_finite_learning_and_only_final_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = run_rounds(root)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["steps"], 5000)
            self.assertEqual(report["rounds_completed"], 50)
            self.assertEqual(len(report["history"]), 51)
            self.assertEqual([row["round"] for row in report["fixed_prompt_samples"]], [0, 10, 20, 30, 40, 50])
            self.assertEqual(report["final_test_evaluations"], 1)
            self.assertEqual(report["layer_two"]["count_builds"], 1)
            self.assertLess(report["history"][-1]["training_loss"], report["history"][0]["training_loss"])
            self.assertGreater(report["changed_parameters"], 0)
            base = root / ".data/neural-experiments"
            for number in range(50):
                checkpoint = json.loads((base / report["run_id"] / f"round-{number:02}.json").read_text())
                self.assertNotIn("final_test_loss", checkpoint["report"])
                self.assertEqual(checkpoint["report"]["steps"], number * 100)
            final = json.loads((base / f"{report['run_id']}.json").read_text())
            model = NeuralModel(final["characters"])
            model.weights = final["weights"]
            self.assertEqual(model.sample(report["sample_prompt"]), report["sample"])
            self.assertEqual(model.loss(model.examples(TEST)), report["final_test_loss"])
            self.assertEqual(final["training_text"], TRAIN)
            self.assertEqual(final["held_out_text"], VALIDATION)
            self.assertEqual(sorted(path.name for path in (root / ".data").iterdir()), ["neural-experiments"])
            self.assertEqual(json.loads((base / "latest.json").read_text()), report)

    def test_evaluation_text_does_not_change_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("neural_rounds.ROUNDS", 1), patch("neural_rounds.UPDATES", 2):
                a = run_rounds(root)
                with patch("neural_rounds.TEST", "zzzzzzzz"), patch("neural_rounds.VALIDATION", "yyyyyyyy"):
                    b = run_rounds(root)
            def weights(report):
                return json.loads((root / ".data/neural-experiments" / f"{report['run_id']}.json").read_text())["weights"]
            self.assertEqual(weights(a), weights(b))
            self.assertEqual(b["test_unknown_characters"], 8)

    def test_no_real_text_or_self_training_in_automated_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                train_experiment({"automated_synthetic": True}, Path(directory))
            self.assertFalse((Path(directory) / ".data").exists())
