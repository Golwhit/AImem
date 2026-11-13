import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from neural_experiment import NeuralModel
from neural_index_pipeline import PlacementIndex, evaluate, run
from neural_skeleton_runtime import UnitCodec
from neural_unit_benchmark import text_hash


class IndexPipelineTests(unittest.TestCase):
    def setUp(self):
        self.codec = UnitCodec({"that": {"pattern": "th*t", "kind": "hidden1"}}, {})
        self.index = PlacementIndex(self.codec)

    def test_extract_removes_entire_occurrences_and_preserves_positions(self):
        text = "that x that thatcher That."
        extracted = self.index.extract(text)
        self.assertEqual(extracted["remainder"], " x  thatcher That.")
        self.assertEqual("".join(extracted["pieces"]), text)
        self.assertEqual([(row["start"], row["end"]) for row in extracted["occurrences"]],
                         [(0, 4), (7, 11)])
        self.assertEqual(self.index.extract("that", open_tail=True)["remainder"], "that")
        for row in extracted["occurrences"]:
            self.assertEqual(text[row["start"]:row["end"]], row["pattern"])

    def test_base_examples_are_only_remainder_with_no_index_output(self):
        model = NeuralModel(sorted(set("that x ")))
        remainder = self.index.extract("that x that ")["remainder"]
        self.assertEqual(remainder, " x  ")
        examples = model.examples(remainder)
        self.assertEqual(len(examples), 4)
        self.assertEqual([model.characters[target - 2] for _, target in examples], list(remainder))
        self.assertNotIn("that", model.tokens)
        self.assertEqual(examples[1][0], [0, 0, 0, model.tokens[" "]])

    def test_selector_learns_placement_separately_and_roundtrips(self):
        pieces = self.index.extract("that x that x ")["pieces"]
        self.assertGreater(self.index.learn(pieces), 0)
        self.assertEqual(dict(self.index.choices(["x", " "], 10)), {"that": 1})
        self.assertEqual(dict(self.index.choices(["that", " "], 10)), {"": 2})
        restored = PlacementIndex.restore(self.index.snapshot())
        self.assertEqual(restored.snapshot(), self.index.snapshot())
        self.assertEqual(restored.choices(["x", " "], 10), self.index.choices(["x", " "], 10))
        self.assertEqual(PlacementIndex(self.codec).choices([], 10), {"": 1})
        self.assertEqual(dict(self.index.choices(["x", " "], 3)), {"": 2})

    def test_index_emits_without_neural_call_or_base_context_advance(self):
        self.index.learn(self.index.extract("x that x that ")["pieces"])
        model = NeuralModel(sorted(set("that x ")))
        contexts = []

        def forward(context):
            contexts.append(context[:])
            row = [0.0] * model.size
            row[model.tokens["x"]] = 100
            row[model.tokens[" "]] = 1
            return [], [], row

        with patch.object(model, "forward", side_effect=forward):
            output = self.index.sample(model, "x ", length=5, greedy=True)
        self.assertEqual(output["text"], "that ")
        self.assertEqual(output["indexed_emissions"], 1)
        self.assertEqual(output["neural_calls"], 1)
        self.assertEqual(contexts, [[0, 0, model.tokens["x"], model.tokens[" "]]])
        before = self.index.snapshot()
        evaluate(model, self.index, [{"prefix": "x ", "target": "that", "offset": 2}])
        self.assertEqual(self.index.snapshot(), before)
        for length in (0, True, 501):
            with self.assertRaises(ValueError):
                self.index.sample(model, "", length=length)

    def test_run_preserves_identical_base_size_and_remainder_only_targets(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("neural_index_pipeline.py", "neural_experiment.py",
                         "neural_skeleton_runtime.py", "neural_unit_benchmark.py"):
                (root / name).write_text("fixture")
            text = "x that " * 8
            with patch("neural_index_pipeline.load_chapters", return_value=({6: text, 7: text}, "fixture")), \
                    patch("neural_index_pipeline.load_bank", return_value={}), \
                    patch.object(UnitCodec, "from_bank", return_value=self.codec):
                report = run(root, acknowledge_review=True)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(len(report["runs"]), 6)
            directory = root / ".data/neural-unit-benchmark" / report["run_id"]
            inputs = json.loads((directory / "inputs.json").read_text())
            for seed in report["settings"]["seeds"]:
                base = next(e for e in report["runs"] if e["seed"] == seed and e["arm"] == "baseline")
                split = next(e for e in report["runs"] if e["seed"] == seed and e["arm"] == "separated")
                for key in ("vocabulary", "parameters", "initial_weights_sha256"):
                    self.assertEqual(base[key], split[key])
                self.assertEqual(split["targets_per_pass"], len(inputs["extracted"]["remainder"]))
                model = NeuralModel(sorted(set(text)), seed=seed)
                self.assertEqual(split["examples_sha256"], text_hash(
                    json.dumps(model.examples(inputs["extracted"]["remainder"]))))
                row = split["checkpoints"][-1]
                self.assertEqual(row["updates"], 4 * len(inputs["extracted"]["remainder"]))
                self.assertGreater(row["selector_count_updates"], 0)
                self.assertEqual(row["total_training_cpu_seconds"],
                                 row["neural_cpu_seconds"] + row["selector_cpu_seconds"])
                saved = json.loads((directory / f"{seed}-separated-passes-4.json").read_text())
                self.assertTrue(all(len(c) == 1 for c in saved["characters"]))
                restored = NeuralModel(saved["characters"], seed=seed)
                restored.weights = saved["weights"]
                restored_index = PlacementIndex.restore(saved["index"])
                sample = split["samples"][0]
                self.assertEqual(restored_index.sample(restored, sample["prompt"])["text"], sample["text"])
            with self.assertRaises(ValueError):
                run(root, acknowledge_review=False)

    def test_large_profile_uses_complete_chapters_five_seeds_and_common_probes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("neural_index_pipeline.py", "neural_experiment.py",
                         "neural_skeleton_runtime.py", "neural_unit_benchmark.py"):
                (root / name).write_text("fixture")
            chapters = {chapter: f"x that {chapter} " * 4 for chapter in range(6, 15)}
            with patch("neural_index_pipeline.load_chapters", return_value=(chapters, "fixture")), \
                    patch("neural_index_pipeline.load_bank", return_value={}), \
                    patch.object(UnitCodec, "from_bank", return_value=self.codec), \
                    patch("neural_index_pipeline.probes", return_value=[
                        {"prefix": "x ", "target": "that", "offset": 2}]) as select_probes:
                report = run(root, acknowledge_review=True, large=True)
            self.assertEqual(report["profile"], "large")
            self.assertEqual(report["settings"]["seeds"], [17, 117, 217, 317, 417])
            self.assertEqual(len(report["runs"]), 10)
            directory = root / ".data/neural-unit-benchmark" / report["run_id"]
            inputs = json.loads((directory / "inputs.json").read_text())
            self.assertEqual(inputs["training"], "\n".join(chapters[c] for c in range(6, 13)))
            self.assertEqual(inputs["evaluation"], "\n".join(chapters[c] for c in (13, 14)))
            select_probes.assert_called_once_with(inputs["evaluation"], 256)
            self.assertEqual(len(list(directory.glob("*-passes-*.json"))), 30)

    def test_custom_corpus_preserves_provenance_and_evaluation_cannot_change_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("neural_index_pipeline.py", "neural_experiment.py",
                         "neural_skeleton_runtime.py", "neural_unit_benchmark.py"):
                (root / name).write_text("fixture")
            record = {"title": "Original fixture", "provenance": "Authored test text",
                      "rights_note": "Test use approved", "limitations": "Artificial",
                      "training": "x that " * 12, "evaluation": "y that x " * 10}
            corpus = root / "corpus.json"
            corpus.write_text(json.dumps(record))
            with patch("neural_index_pipeline.load_chapters", side_effect=AssertionError("No Acts loading")), \
                    patch("neural_index_pipeline.load_bank", return_value={}), \
                    patch.object(UnitCodec, "from_bank", return_value=self.codec) as build:
                first = run(root, acknowledge_review=True, corpus=corpus)
                self.assertEqual(first["profile"], "custom")
                build.assert_called_once_with({}, record["training"])
                self.assertEqual(first["source"]["corpus"]["provenance"], record["provenance"])
                self.assertEqual(first["source"]["training_sha256"], text_hash(record["training"]))
                inputs = json.loads((root / ".data/neural-unit-benchmark" / first["run_id"] /
                                     "inputs.json").read_text())
                self.assertEqual(inputs["corpus_record"], record)
                record["evaluation"] = "different than that x " * 5
                corpus.write_text(json.dumps(record))
                second = run(root, acknowledge_review=True, corpus=corpus)
            for a, b in zip(first["runs"], second["runs"], strict=True):
                self.assertEqual(a["examples_sha256"], b["examples_sha256"])
                for passes in (1, 2, 4):
                    name = f"{a['seed']}-{a['arm']}-passes-{passes}.json"
                    one = json.loads((root / ".data/neural-unit-benchmark" / first["run_id"] / name).read_text())
                    two = json.loads((root / ".data/neural-unit-benchmark" / second["run_id"] / name).read_text())
                    self.assertEqual(one["weights"], two["weights"])
                    self.assertEqual(one["index"], two["index"])
            with self.assertRaises(ValueError):
                run(root, acknowledge_review=True, corpus=corpus, large=True)
            for bad in ([], {**record, "provenance": ""},
                        {**record, "training": "short"}, {**record, "evaluation": record["training"]}):
                corpus.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    run(root, acknowledge_review=True, corpus=corpus)


if __name__ == "__main__":
    unittest.main()
