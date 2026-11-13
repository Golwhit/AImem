import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from neural_base_comparison import WordBase
from neural_conversation_trial import answer_examples, assess, prepare, prompt_for, run, train


class ConversationTrialTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"id": "a", "skill": "fact", "turns": [
            {"role": "user", "content": "cat?"}], "answer": "cat."}]
        self.model = WordBase("user: cat?\nassistant: cat.\n", 17)

    def test_only_answers_are_targets_with_causal_context(self):
        examples = answer_examples(self.model, self.rows)
        tokens = self.model.encode("cat.\n")
        self.assertEqual([target for _, target, _ in examples], tokens)
        prefix = prompt_for(self.rows[0])
        for context, target, prior in examples:
            self.assertEqual(context, self.model.context(prefix))
            self.assertEqual(prior, self.model.prior(prefix))
            prefix += self.model.payloads[target]
        self.assertTrue(prefix.endswith("cat.\n"))

    def test_exact_replay_and_newline_scoring(self):
        state = json.loads(json.dumps(self.model.snapshot()))
        points = train(self.model, self.rows, rate=.03, seed=17, checkpoints=(3, 7))
        replay = WordBase.restore(state)
        examples = answer_examples(replay, self.rows)
        rng = random.Random(17)
        for _ in range(7):
            context, target, prior = examples[rng.randrange(len(examples))]
            replay.model.update(context, target, learning_rate=.03, output_prior=prior)
        self.assertEqual(replay.snapshot(), self.model.snapshot())
        self.assertEqual(points[-1]["training"], assess(replay, self.rows))
        for text, exact, normalized in (("cat.\nignored", 1, 1), ("cat.", 0, 0),
                                       ("CAT. \n", 0, 1)):
            with patch("neural_conversation_trial.generate", return_value={"text": text}):
                result = assess(replay, self.rows)
            self.assertEqual(result["exact"], exact)
            self.assertEqual(result["normalized_exact"], normalized)

    def test_content_change_rejects_review_before_model_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "fixtures").mkdir()
            (root / "fixtures/neural-conversation-draft.json").write_text("{}")
            with patch("neural_conversation_trial.load_active", side_effect=AssertionError("loaded")):
                with self.assertRaises(ValueError):
                    run(root, acknowledge_review=True, reviewed_sha256="0" * 64)
                with self.assertRaises(ValueError):
                    run(root, acknowledge_review=False, reviewed_sha256="0" * 64)

    def test_training_only_character_extension_preserves_existing_parameters(self):
        state = json.loads(json.dumps(self.model.snapshot()))
        rows = json.loads(json.dumps(self.rows))
        rows[0]["answer"] = "cat's."
        prepared, added = prepare(self.model, rows)
        self.assertEqual(added, ["char:'"])
        self.assertEqual(self.model.snapshot(), state)
        self.assertEqual(prepared.lexicon, self.model.lexicon)
        for symbol, token in self.model.model.tokens.items():
            self.assertEqual(prepared.model.tokens[symbol], token)
        self.assertEqual(prepared.model.weights["hidden"], self.model.model.weights["hidden"])
        self.assertTrue(answer_examples(prepared, rows))
        self.assertNotIn("char:Z", prepared.model.tokens)

    def test_role_format_does_not_restore_discarded_facts(self):
        red = {"turns": [{"role": "user", "content": "Mira has a red box. What color is the box?"}]}
        blue = {"turns": [{"role": "user", "content": "Mira has a blue box. What color is the box?"}]}
        self.assertNotEqual(prompt_for(red), prompt_for(blue))
        self.assertEqual(self.model.context(prompt_for(red)), self.model.context(prompt_for(blue)))
        self.assertEqual(self.model.prior(prompt_for(red)), self.model.prior(prompt_for(blue)))


if __name__ == "__main__":
    unittest.main()
