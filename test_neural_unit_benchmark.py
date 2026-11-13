import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from neural_experiment import NeuralModel, status
from neural_character_onlies import CharacterOnlies
from neural_skeleton_runtime import END, UnitCodec
from neural_unit_benchmark import (add_combined, character_probe_examples, edit_distance,
                                   next_word, probes, quality, summarize, text_hash)
from neural_unit_benchmark import run


class UnitBenchmarkTests(unittest.TestCase):
    def test_comparison_status_is_separate_from_latest_training_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertIsNone(status(root)["unit_benchmark"])
            path = root / ".data/neural-experiments/latest-unit-benchmark.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"status": "completed", "runs": []}))
            self.assertEqual(status(root)["unit_benchmark"]["status"], "completed")
            self.assertFalse(status(root)["ready"])

    def test_common_metric_and_probe_selection(self):
        self.assertEqual(edit_distance("that", "than"), 1)
        self.assertEqual(edit_distance("", "that"), 4)
        text = "a prior context is long enough to begin. that with them. that with them."
        items = probes(text, 4)
        self.assertEqual(len(items), 4)
        self.assertEqual(items, probes(text, 4))
        for item in items:
            self.assertEqual(text[item["offset"]:][:len(item["target"])], item["target"])
            self.assertEqual(item["prefix"], text[:item["offset"]])
        with patch("neural_unit_benchmark.next_word", side_effect=["that", "then"]):
            result = quality(NeuralModel(list("that en")), None,
                             [{"prefix": "", "target": "that", "offset": 0},
                              {"prefix": "", "target": "them", "offset": 10}])
        self.assertEqual(result["exact_next_word_accuracy"], 0.5)
        self.assertEqual(result["normalized_edit_distance"], 0.125)

    def test_character_and_unit_predictions_share_whole_word_metric(self):
        codec = UnitCodec({"that": {"pattern": "th*t", "kind": "hidden1"}}, {})
        model = NeuralModel(sorted(set("that ") | {"that", END}))

        def forward(context):
            row = [0.0] * model.size
            row[model.tokens["that"]] = 100
            row[model.tokens[" "]] = 10
            return [], [], row

        with patch.object(model, "forward", side_effect=forward):
            self.assertEqual(next_word(model, codec, "and "), "that")
        chars = NeuralModel(list("that "))
        sequence = iter("that ")

        def character_forward(context):
            row = [0.0] * chars.size
            row[chars.tokens[next(sequence)]] = 1
            return [], [], row

        with patch.object(chars, "forward", side_effect=character_forward):
            self.assertEqual(next_word(chars, None, "and "), "that")

    def test_unterminated_prediction_is_not_counted_as_a_word(self):
        model = NeuralModel(list("ab "))
        row = [0.0] * model.size
        row[model.tokens["a"]] = 1
        with patch.object(model, "forward", return_value=([], [], row)):
            self.assertEqual(next_word(model, None, "and "), "")

    def test_verdicts_use_equal_passes_not_equal_updates(self):
        runs = []
        for arm, cpu in (("characters", 10), ("skeleton_units", 7)):
            runs.append({"arm": arm, "checkpoints": [{
                "budgets": ["passes-4", "updates-12000"], "updates": 100,
                "training_cpu_seconds": cpu, "training_wall_seconds": cpu,
                "estimated_update_operations": cpu * 100,
                "exact_next_word_accuracy": 0.25, "normalized_edit_distance": 0.5,
            }]})
        summary = summarize(runs)
        self.assertTrue(summary["verdicts"]["similar_quality"])
        self.assertTrue(summary["verdicts"]["less_effort"])
        self.assertAlmostEqual(summary["passes-4"]["cpu_ratio"], 0.7)
        for run in runs:
            run["checkpoints"][0]["budgets"] = ["passes-4", "updates-120000"]
            run["checkpoints"][0]["all_character_probe_loss"] = 2.5
        large_summary = summarize(runs, matched_updates=120000)
        self.assertIn("updates-120000", large_summary)
        self.assertNotIn("updates-12000", large_summary)
        self.assertEqual(large_summary["passes-4"]["arms"]["characters"]["all_character_probe_loss"], 2.5)

    def test_character_probes_include_skipped_spelling_positions(self):
        model = NeuralModel(list("that "))
        self.assertEqual(character_probe_examples(model, "that ", count=5), model.examples("that "))
        self.assertEqual(len(character_probe_examples(model, "that " * 10, count=7)), 7)
        with self.assertRaises(ValueError):
            character_probe_examples(model, "", count=7)

    def test_combined_preserves_baseline_weights_and_every_training_update(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("neural_unit_benchmark.py", "neural_character_onlies.py", "neural_experiment.py"):
                (root / name).write_text("test implementation")
            source_id = "a" * 32
            directory = root / ".data/neural-unit-benchmark" / source_id
            directory.mkdir(parents=True)
            text = "that "
            rules = CharacterOnlies({"that"}, {"th": "that"}, {})
            model = NeuralModel(sorted(set(text)))
            model.update(*model.examples(text)[0], learning_rate=0.1)
            loss = model.loss(character_probe_examples(model, text))
            checkpoint = {
                "budgets": ["passes-4", "updates-12000"], "updates": 12000,
                "training_cpu_seconds": 10, "training_wall_seconds": 11,
                "estimated_update_operations": 1000, "exact_next_word_accuracy": 0,
                "normalized_edit_distance": 1, "all_character_probe_loss": loss,
            }
            original = {
                "run_id": source_id, "status": "completed", "method": "character_recognition_onlies",
                "settings": {"matched_updates": 12000},
                "source": {"training_sha256": text_hash(text), "evaluation_sha256": text_hash(text)},
                "runs": [{"seed": 17, "arm": arm, "checkpoints": [checkpoint], "samples": []}
                         for arm in ("characters", "skeleton_units")],
            }
            original_text = json.dumps(original)
            (directory / "report.json").write_text(original_text)
            (directory / "inputs.json").write_text(json.dumps({
                "training": text, "evaluation": text,
                "probes": [{"prefix": "", "target": "that", "offset": 0}],
                "unit_codec": rules.snapshot(),
            }))
            for arm in ("characters", "skeleton_units"):
                (directory / f"17-{arm}-step-12000.json").write_text(json.dumps({
                    "characters": model.characters, "weights": model.weights,
                }))
            with patch.object(NeuralModel, "update", side_effect=AssertionError("Must not retrain")):
                report = add_combined(root, source_id, acknowledge_review=True)
            self.assertEqual((directory / "report.json").read_text(), original_text)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(len(report["runs"]), 3)
            combined = report["runs"][-1]
            row = combined["checkpoints"][0]
            for key in ("updates", "training_cpu_seconds", "estimated_update_operations",
                        "all_character_probe_loss"):
                self.assertEqual(row[key], checkpoint[key])
            saved = json.loads((root / ".data/neural-unit-benchmark" / report["run_id"] /
                                "17-baseline_plus_rules-step-12000.json").read_text())
            self.assertEqual(saved["weights"], model.weights)
            self.assertEqual(saved["unit_codec"], rules.snapshot())
            self.assertEqual(len(report["matched_generation"]["runs"]), 3)
            self.assertIn("baseline_plus_rules", report["summary"]["passes-4"]["arms"])
            with self.assertRaises(ValueError):
                add_combined(root, source_id, acknowledge_review=False)
            with self.assertRaises(ValueError):
                add_combined(root, "../escape", acknowledge_review=True)

    def test_quality_records_rule_usage(self):
        model = NeuralModel(list("that "))
        rules = CharacterOnlies({"that"}, {"th": "that"}, {})
        result = quality(model, rules, [{"prefix": "th", "target": "at", "offset": 2}])
        self.assertGreaterEqual(result["generation_rule_matches"], 1)
        self.assertGreaterEqual(result["generation_forced_characters"], 2)

    def test_indexed_run_persists_atomic_index_and_equal_update_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("neural_unit_benchmark.py", "neural_character_onlies.py",
                         "neural_experiment.py", "neural_skeleton_runtime.py"):
                (root / name).write_text("test implementation")
            bank = {"version": 1, "sha256": "fixture", "source": "test fixture",
                    "learning_sha256": "fixture", "dictionary_sha256": "fixture",
                    "experiment_code_sha256": "fixture", "policy": "test fixture",
                    "patterns": {"hidden1": {"th*t": "that"}, "hidden2": {}}}
            text = "that x " * 8
            with patch("neural_unit_benchmark.load_chapters", return_value=({6: text, 7: text}, "fixture")), \
                    patch("neural_unit_benchmark.load_bank", return_value=bank), \
                    patch("neural_unit_benchmark.SEEDS", (17,)), \
                    patch("neural_unit_benchmark.MATCHED_UPDATES", 20):
                report = run(root, acknowledge_review=True, indexed_onlies=True)
            self.assertEqual(report["method"], "indexed_onlies")
            base, indexed = report["runs"]
            self.assertGreater(indexed["vocabulary"], base["vocabulary"])
            self.assertEqual(indexed["targets_per_pass"], len(UnitCodec(
                {"that": {"pattern": "th*t", "kind": "hidden1"}}, {}).encode(text)) + 1)
            index = indexed["index"]["that"]
            self.assertEqual(index["pattern"], "th*t")
            directory = root / ".data/neural-unit-benchmark" / report["run_id"]
            final = json.loads((directory / "17-skeleton_units.json").read_text())
            self.assertEqual(final["characters"][index["token_id"] - 2], "that")
            last = max(row["updates"] for row in indexed["checkpoints"])
            saved = json.loads((directory / f"17-skeleton_units-step-{last}.json").read_text())
            self.assertEqual(final["weights"], saved["weights"])
            self.assertEqual(report["matched_generation"]["budget"], "updates-20")
            self.assertEqual(len(report["matched_generation"]["runs"]), 2)
            self.assertTrue((directory / "matched-generation.json").exists())


if __name__ == "__main__":
    unittest.main()
