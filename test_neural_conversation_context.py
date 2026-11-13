import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from neural_base_comparison import WordBase, generate
from neural_conversation_context import ARMS, CONTEXT, ConversationContext, build, lower_rate_screen, run
from neural_base_comparison import digest
from neural_conversation_trial import answer_examples, assess, prompt_for, train
from neural_experiment import EMBEDDING, HIDDEN


class ConversationContextTests(unittest.TestCase):
    def setUp(self):
        self.parent = WordBase("red blue cup box has the. user assistant: \n?", 17)
        self.rows = [
            {"id": "red", "skill": "color", "turns": [
                {"role": "user", "content": "Mira has a red box. What color is the box?"}], "answer": "red."},
            {"id": "blue", "skill": "color", "turns": [
                {"role": "user", "content": "Mira has a blue box. What color is the box?"}], "answer": "blue."},
        ]

    def test_long_features_preserve_fact_and_train_only_word_identity(self):
        short, _ = build(self.parent, self.rows, "short_identities")
        long, metadata = build(self.parent, self.rows, "long_identities")
        red, blue = [prompt_for(row) for row in self.rows]
        self.assertEqual(short.context(red), short.context(blue))
        self.assertNotEqual(long.context(red), long.context(blue))
        self.assertEqual(long.context(red)[-4:], short.context(red))
        self.assertEqual(len(long.context(red)), CONTEXT)
        self.assertIn("identity:mira", long.model.tokens)
        self.assertNotIn("identity:tavi", long.model.tokens)
        self.assertNotIn("emit:Mira", long.model.tokens)
        self.assertEqual(short.model.characters, long.model.characters)
        self.assertEqual(short.prior(red), long.prior(red))
        self.assertEqual(long.model.weights["hidden"][:(CONTEXT - 4) * EMBEDDING * HIDDEN],
                         [0.0] * ((CONTEXT - 4) * EMBEDDING * HIDDEN))
        for prompt in (red, blue, red + "r", red + "red", red + "red. "):
            p_short = short.model.forward(short.context(prompt), output_prior=short.prior(prompt))[2]
            p_long = long.model.forward(long.context(prompt), output_prior=long.prior(prompt))[2]
            self.assertEqual(p_short, p_long)
        self.assertEqual(short.model.weights["embedding"], long.model.weights["embedding"])
        self.assertGreater(metadata["parameters"], sum(map(len, short.model.weights.values())))

    def test_roundtrip_replay_and_new_context_weights_learn(self):
        for arm in ARMS:
            model, _ = build(self.parent, self.rows, arm)
            state = json.loads(json.dumps(model.snapshot()))
            points = train(model, self.rows, rate=.1, seed=17, checkpoints=(3, 10))
            restored = ConversationContext.restore(json.loads(json.dumps(model.snapshot())))
            self.assertEqual(assess(model, self.rows), assess(restored, self.rows))
            replay = ConversationContext.restore(state)
            rng = random.Random(17)
            examples = answer_examples(replay, self.rows)
            for _ in range(10):
                context, target, prior = examples[rng.randrange(len(examples))]
                replay.model.update(context, target, output_prior=prior, learning_rate=.1)
            self.assertEqual(replay.snapshot(), model.snapshot())
            self.assertEqual(points[-1]["training"], assess(replay, self.rows))
            if arm == "long_identities":
                self.assertTrue(any(model.model.weights["hidden"][:(CONTEXT - 4) * EMBEDDING * HIDDEN]))
            self.assertEqual(generate(model, prompt_for(self.rows[0]), length=32),
                             generate(restored, prompt_for(self.rows[0]), length=32))

    def test_local_features_padding_and_review_gate(self):
        long, _ = build(self.parent, self.rows, "long_identities")
        for prefix in ("", "red", "red ", "red blue ", "red blue cup", "red blue cup "):
            context = long.context(prefix)
            self.assertEqual(len(context), CONTEXT)
            self.assertEqual(context[-4:], WordBase.context(long, prefix))
        state = long.snapshot()
        state["context"] = 8
        with self.assertRaises(ValueError):
            ConversationContext.restore(state)
        state = json.loads(json.dumps(long.snapshot()))
        state["weights"]["hidden"][0] = float("nan")
        with self.assertRaises(ValueError):
            ConversationContext.restore(state)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "fixtures").mkdir()
            (root / "fixtures/neural-conversation-draft.json").write_text("{}")
            with self.assertRaises(ValueError):
                run(root, acknowledge_review=True, reviewed_sha256="0" * 64)

    def test_lower_rate_selection_uses_teaching_loss_before_one_test(self):
        long, _ = build(self.parent, self.rows, "long_identities")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "fixtures").mkdir()
            data = {"train": self.rows, "test": [{"id": "test", "skill": "color",
                    "turns": [{"role": "user", "content": "red?"}], "answer": "red."}]}
            raw = json.dumps(data).encode()
            checksum = digest(raw)
            (root / "fixtures/neural-conversation-draft.json").write_bytes(raw)
            directory = root / ".data/neural-conversation-context" / ("a" * 32)
            directory.mkdir(parents=True)
            (directory / "report.json").write_text(json.dumps({
                "status": "completed", "review": {"file_sha256": checksum},
                "shuffle_seed": 17, "updates": [3, 5]}))
            (directory / "inputs.json").write_bytes(raw)
            (directory / "long_identities-initial.json").write_text(json.dumps({"state": long.snapshot()}))
            real_assess = assess
            test_calls = []
            def watched_assess(model, rows):
                if rows == data["test"]:
                    self.assertTrue((directory / "lower-rate-selection.json").exists())
                    test_calls.append(rows)
                return real_assess(model, rows)
            with patch("neural_conversation_context.assess", side_effect=watched_assess):
                result = lower_rate_screen(root, "a" * 32, acknowledge_review=True,
                                           reviewed_sha256=checksum)
            self.assertEqual(len(test_calls), 1)
            chosen = min(result["trials"], key=lambda t: t["checkpoints"][-1]["teacher_forced_loss"])
            self.assertEqual(result["selected_rate_by_training_loss"], chosen["rate"])
            with self.assertRaises(ValueError):
                lower_rate_screen(root, "a" * 32, acknowledge_review=True, reviewed_sha256=checksum)


if __name__ == "__main__":
    unittest.main()
