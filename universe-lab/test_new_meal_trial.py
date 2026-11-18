import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from character_context_trial import character_examples, character_model
from neural_plasticity import PlasticityContext
from new_meal_trial import run, train_new_meal, word_cases


class NewMealTrialTests(unittest.TestCase):
    def test_new_meal_training_replays_and_requires_examples(self):
        parent = PlasticityContext("text: cat dog.\n", 17)
        first = character_model(PlasticityContext.restore(copy.deepcopy(parent.snapshot())))
        replay = character_model(PlasticityContext.restore(copy.deepcopy(parent.snapshot())))
        examples = character_examples(first, [("text: ", "cat dog.")])
        train_new_meal(first, examples, updates=30, seed=9, rate=0.01)
        train_new_meal(replay, examples, updates=30, seed=9, rate=0.01)
        self.assertEqual(first.snapshot(), replay.snapshot())
        self.assertNotEqual(first.snapshot()["weights"], parent.snapshot()["weights"])
        with self.assertRaises(ValueError):
            train_new_meal(first, [], updates=30, seed=9, rate=0.01)

    def test_word_cases_use_prompt_and_up_to_two_expected_words(self):
        row = {"turns": [{"role": "user", "content": "Where?"}], "answer": "On table."}
        self.assertEqual(word_cases([row], "meal"),
                         [("meal", "user: Where?\nassistant: ", ["On", "table"])])
        with self.assertRaises(ValueError):
            word_cases([{**row, "answer": "?"}], "meal")

    def test_split_training_continues_the_same_sampling_stream(self):
        parent = PlasticityContext("text: cat dog.\n", 17)
        whole = character_model(PlasticityContext.restore(copy.deepcopy(parent.snapshot())))
        split = character_model(PlasticityContext.restore(copy.deepcopy(parent.snapshot())))
        examples = character_examples(whole, [("text: ", "cat dog.")])
        train_new_meal(whole, examples, updates=30, seed=9, rate=0.01)
        train_new_meal(split, examples, updates=12, seed=9, rate=0.01)
        train_new_meal(split, examples, updates=18, seed=9, rate=0.01, offset=12)
        self.assertEqual(whole.snapshot(), split.snapshot())
        with self.assertRaises(ValueError):
            train_new_meal(split, examples, updates=1, seed=9, rate=0.01, offset=-1)

    def test_continuation_requires_origin_and_same_reviewed_meal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meal = root / "meal.json"
            meal.write_text("{}")
            meal_hash = hashlib.sha256(meal.read_bytes()).hexdigest()
            origin = root / "origin"
            origin.mkdir()
            (origin / "report.json").write_text("{}")
            origin_hash = hashlib.sha256((origin / "report.json").read_bytes()).hexdigest()
            previous = root / "previous"
            previous.mkdir()
            (previous / "trained.json").write_text('{"output_mode": "character_only"}')
            report = {"kind": "private_new_conversation_meal_trial",
                      "validation": {"exact_replay": True},
                      "trained_sha256": hashlib.sha256((previous / "trained.json").read_bytes()).hexdigest(),
                      "parent_report_sha256": origin_hash,
                      "training": {"meal_sha256": "different", "new_meal_only": True}}
            (previous / "report.json").write_text(json.dumps(report))
            args = SimpleNamespace(acknowledge_review=True, meal=meal, reviewed_sha256=meal_hash,
                                   previous_run=previous, origin_run=None,
                                   previous_report_sha256=hashlib.sha256((previous / "report.json").read_bytes()).hexdigest())
            with patch("new_meal_trial.validate_curriculum"):
                with self.assertRaisesRegex(ValueError, "requires the original"):
                    run(args)
                args.origin_run = origin
                with self.assertRaisesRegex(ValueError, "same reviewed meal"):
                    run(args)


if __name__ == "__main__":
    unittest.main()
