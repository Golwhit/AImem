from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from neural_base_comparison import WordBase, digest
from neural_conversation_context import build
from neural_conversation_meal import (
    ARMS, FIXTURE, SKILLS, SOURCES, curriculum, paired_scores, run, validate, validate_curriculum,
)
from neural_conversation_trial import answer_examples, assess, prompt_for, token_loss
from neural_plasticity import PlasticityContext, diagnostics, initialize


class ConversationMealTests(unittest.TestCase):
    def test_exact_counts_balance_disjointness_and_pairs(self):
        data = curriculum()
        validate_curriculum(data)
        self.assertEqual(len(data["train"]), 64)
        self.assertEqual(len(data["test"]), 32)
        self.assertEqual(Counter(row["evaluation_group"] for row in data["test"]),
                         {"combination": 16, "wording": 16})
        for skill in SKILLS:
            rows = [row for row in data["train"] if row["skill"] == skill]
            self.assertEqual(len(rows), 8)
            self.assertEqual(len({row["pair"] for row in rows}), 4)
        for split in ("train", "test"):
            for row in data[split]:
                self.assertNotEqual(row["answer"], "")
                self.assertTrue(prompt_for(row).endswith("assistant: "))
                if row["skill"] in ("reference", "correction"):
                    self.assertEqual(len(row["turns"]), 3)
                if row["skill"] == "transfer":
                    prompt = prompt_for(row)
                    if "keeps" in prompt:
                        self.assertEqual(row["answer"].split(".")[0], prompt.split()[1])
                    else:
                        recipient = prompt.split(" to ")[1].split(".")[0]
                        self.assertEqual(row["answer"], recipient + ".")
        data["test"][0]["turns"] = data["train"][0]["turns"]
        with self.assertRaises(ValueError):
            validate_curriculum(data)

    def test_larger_teaching_set_keeps_original_rows_and_evaluation_disjoint(self):
        small = curriculum()
        large = curriculum(256)
        validate_curriculum(large)
        self.assertEqual(len(large["train"]), 256)
        self.assertEqual(large["train"][:64], small["train"])
        self.assertEqual(large["test"], small["test"])
        self.assertEqual(large["small_control_ids"], small["small_control_ids"])
        for skill in SKILLS:
            rows = [row for row in large["train"] if row["skill"] == skill]
            self.assertEqual(len(rows), 32)
            self.assertEqual(len({row["pair"] for row in rows}), 16)
        train_text = [turn["content"] for row in large["train"] for turn in row["turns"]]
        self.assertFalse(any("Mira" in text and "book" in text for text in train_text))
        with self.assertRaises(ValueError):
            curriculum(100)

    def test_training_only_symbols_and_equal_initials_across_sizes(self):
        data = curriculum()
        teaching = "".join(prompt_for(row) + row["answer"] for row in data["train"])
        parent = WordBase(teaching, 17)
        base, _ = build(parent, data["train"], "long_identities")
        self.assertIn("identity:omar", base.model.tokens)
        self.assertNotIn("identity:current", base.model.tokens)
        for arm in ARMS:
            small = initialize(base, arm, 17)
            large = initialize(base, arm, 17)
            self.assertEqual(small.snapshot(), large.snapshot())
            self.assertEqual(len(answer_examples(large, data["train"])),
                             len(answer_examples(small, data["train"])))

    def test_pair_score_rejects_constant_answer_success(self):
        data = curriculum()
        rows = [dict(id=row["id"], pair=row["pair"], skill=row["skill"], expected=row["answer"],
                     prediction="Red.", exact=row["answer"] == "Red.",
                     normalized_exact=row["answer"] == "Red.") for row in data["test"]]
        result = paired_scores({"rows": rows})
        self.assertEqual(result["pairs"], 16)
        self.assertEqual(result["both_exact"], 0)
        self.assertEqual(result["identical_pair_answers"], 16)

    def test_gate_pretest_boundary_and_full_saved_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "fixtures").mkdir()
            data = curriculum()
            raw = json.dumps(data).encode()
            (root / FIXTURE).write_bytes(raw)
            with self.assertRaises(ValueError):
                run(root, acknowledge_review=False, reviewed_sha256=digest(raw))
            with self.assertRaises(ValueError):
                run(root, acknowledge_review=True, reviewed_sha256="0" * 64)
            self.assertFalse((root / ".data").exists())
            for name in SOURCES:
                (root / name).write_bytes((Path(__file__).parent / name).read_bytes())
            teaching = "".join(prompt_for(row) + row["answer"] for row in data["train"])
            parent = WordBase(teaching, 17)
            calls = []

            def watched_assess(model, rows):
                endpoints = list((root / ".data/neural-conversation-meal").glob("*/pre-test-endpoints.json"))
                self.assertEqual(len(endpoints), 1)
                recorded = json.loads(endpoints[0].read_text())
                self.assertEqual(len(recorded["trials"]), 4)
                self.assertTrue(all("test" not in row for row in recorded["trials"]))
                calls.append(rows)
                return assess(model, rows, max_chars=1)

            def brief_fit(model, rows, *, seed, rate, cpu_checkpoints, decay=1.0):
                import random
                examples = answer_examples(model, rows)
                context, target, prior = examples[random.Random(seed).randrange(len(examples))]
                model.model.update(context, target, output_prior=prior, learning_rate=rate)
                return [{"updates": 1, "teacher_forced_loss": token_loss(model, examples),
                         "training": assess(model, rows), "diagnostics": diagnostics(model, examples),
                         "training_cpu_seconds": 0.0}]

            with patch("neural_conversation_meal.load_selection", return_value=(
                    {"model": "original-word-punctuation-mlp-chat-v1"}, parent, None)), \
                    patch("neural_conversation_meal.SEEDS", (17,)), \
                    patch("neural_conversation_meal.fit", side_effect=brief_fit), \
                    patch("neural_conversation_meal.assess", side_effect=watched_assess):
                result = run(root, acknowledge_review=True, reviewed_sha256=digest(raw))
            self.assertEqual(len(calls), 4)
            self.assertEqual(result["status"], "completed")
            with patch("neural_conversation_meal.assess",
                       side_effect=lambda model, rows: assess(model, rows, max_chars=1)):
                verified = validate(root, result["run_id"])
            self.assertEqual(verified["exact_weight_and_test_replays"], 4)


if __name__ == "__main__":
    unittest.main()
