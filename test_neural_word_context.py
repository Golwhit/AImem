from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from neural_experiment import NeuralModel
from neural_word_context import ARMS, compare, examples, parse, vocabulary


class WordContextTests(unittest.TestCase):
    def test_larger_passages_match_archived_baseline_and_reject_mismatch(self):
        import json
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parent = root / ".data/neural-unit-benchmark/6045eb140d8e47b1a5c6890199251039"
            parent.mkdir(parents=True)
            chapters = {chapter: "Hello she said. " * 2 for chapter in range(6, 15)}
            training = "\n".join(chapters[c] for c in range(6, 13))
            evaluation = "\n".join(chapters[c] for c in (13, 14))
            (parent / "inputs.json").write_text(json.dumps({"training": training, "evaluation": evaluation}))
            (parent / "report.json").write_text(json.dumps({"source": {
                "training_sha256": hashlib.sha256(training.encode()).hexdigest(),
                "evaluation_sha256": hashlib.sha256(evaluation.encode()).hexdigest()}}))
            with patch("neural_word_context.load_chapters", return_value=(chapters, "fixture")):
                result = compare(root, acknowledge_review=True, steps=2, known_targets_only=True, larger=True)
                self.assertTrue(result["larger"])
                self.assertEqual(result["training_characters"], len(training))
                (parent / "inputs.json").write_text(json.dumps({"training": "tampered", "evaluation": evaluation}))
                with self.assertRaisesRegex(ValueError, "exactly match"):
                    compare(root, acknowledge_review=True, steps=2, larger=True)

    def test_lossless_words_apostrophes_hyphens_and_punctuation(self):
        text = '“Hello,” she said. Can\'t can’t cant well-known!  \n'
        parsed = parse(text)
        self.assertEqual("".join(row["gap"] + row["word"] for row in parsed["words"]) + parsed["tail"], text)
        self.assertEqual([row["identity"] for row in parsed["words"]],
                         ["hello", "she", "said", "can't", "can't", "cant", "well-known"])
        model = NeuralModel(vocabulary(text))
        normal = examples(text, model.tokens, "words_punctuation")
        altered = examples(text.replace('“Hello,”', 'Hello'), model.tokens, "words_punctuation")
        plain = examples(text, model.tokens, "words_only")
        self.assertEqual([r["target"] for r in normal], [r["target"] for r in altered])
        self.assertNotEqual(normal[1]["context"], altered[1]["context"])
        self.assertEqual(normal[1]["context"][2], plain[1]["context"][2])
        self.assertEqual(normal[0]["context"], [0] * 4)

    def test_context_does_not_read_target_or_future_words(self):
        text = "alpha, beta; gamma delta."
        model = NeuralModel(vocabulary(text))
        for arm in ARMS:
            original = examples(text, model.tokens, arm)
            changed = examples("alpha, beta; other future.", model.tokens, arm)
            self.assertEqual(original[2]["context"], changed[2]["context"])
            self.assertNotEqual(original[2]["target"], changed[2]["target"])
        with self.assertRaises(ValueError):
            examples(text, model.tokens, "bad")

    def test_protocol_and_exact_small_checkpoint_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(ValueError):
                compare(root, acknowledge_review=False)
            data = {"training": "Hello, she said. Hello she said! " * 3,
                    "validation": "Hello, she said. mystery mystery!", "manifest_sha256": "fixture"}
            with patch("neural_word_context.load_acts", return_value=data) as loader:
                report = compare(root, acknowledge_review=True, steps=4)
            loader.assert_called_once_with(root, True, include_final_test=False)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(len(report["runs"]), 9)
            for seed in report["seeds"]:
                rows = [r for r in report["runs"] if r["seed"] == seed]
                self.assertEqual(len({r["initial_weights_sha256"] for r in rows}), 1)
                self.assertEqual(len({r["draws_sha256"] for r in rows}), 1)
            self.assertEqual(report["summary"]["raw4"]["known_words"]["targets"], 3)
            self.assertNotIn("word:mystery", report["symbols"])
            import json
            import random
            from neural_word_context import measure
            directory = root / ".data/neural-word-context" / report["run_id"]
            inputs = json.loads((directory / "inputs.json").read_text())
            for row in report["runs"]:
                model = NeuralModel(report["symbols"], seed=row["seed"])
                rng = random.Random(row["seed"] + 2)
                training = inputs["training_examples"][row["arm"]]
                for _ in range(4):
                    example = training[rng.randrange(len(training))]
                    model.update(example["context"], example["target"], learning_rate=0.03,
                                 output_prior=inputs["output_prior"])
                saved = json.loads((directory / f"{row['seed']}-{row['arm']}.json").read_text())
                self.assertEqual(saved["weights"], model.weights)
                before = deepcopy(model.weights)
                self.assertEqual(row["history"][-1]["validation"],
                                 measure(model, inputs["validation_examples"][row["arm"]], inputs["output_prior"]))
                self.assertEqual(model.weights, before)
            self.assertFalse((root / ".data/neural-chat").exists())

    def test_closed_set_excludes_other_targets_not_unknown_context(self):
        data = {"training": "Hello, she said. " * 3,
                "validation": "Unknown hello, she said. mystery",
                "manifest_sha256": "fixture"}
        with tempfile.TemporaryDirectory() as folder:
            with patch("neural_word_context.load_acts", return_value=data):
                result = compare(Path(folder), acknowledge_review=True, steps=4, known_targets_only=True)
            for entry in result["runs"]:
                measured = entry["history"][-1]["validation"]
                self.assertEqual(measured["excluded_targets"], 2)
                self.assertEqual(measured["all_scored_targets"]["targets"], 3)
                self.assertTrue(all(row["known"] for row in measured["rows"]))


if __name__ == "__main__":
    unittest.main()
