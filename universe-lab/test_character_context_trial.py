import copy
import unittest
from unittest.mock import patch

from character_context_trial import (assessment_summary, character_examples, character_model, score_cases,
                                     text_cases, train_characters, train_meal)
from fresh_pair_comparison import training_rng
from neural_corpus_training import CORPUS_RATE
from neural_plasticity import PlasticityContext


class CharacterContextTrialTests(unittest.TestCase):
    def test_single_target_is_excluded_but_missing_second_is_wrong(self):
        cases = [
            ("conversation", "one", ["cat"]),
            ("conversation", "two", ["cat", "dog"]),
            ("conversation", "three", ["cat", "dog"]),
        ]
        outputs = iter(["cat\n", "cat\n", "cat dog\n"])
        with patch("character_context_trial.generate", side_effect=lambda *a, **k: {"text": next(outputs)}):
            scores = score_cases(None, cases)
        self.assertEqual(scores["word_1_targets"], 3)
        self.assertEqual(scores["word_1_exact"], 3)
        self.assertEqual(scores["no_second_target"], 1)
        self.assertEqual(scores["word_2_targets"], 2)
        self.assertEqual(scores["word_2_missing"], 1)
        self.assertEqual(scores["word_2_exact"], 1)
        self.assertEqual(scores["letters"]["word_2_letter_1"], {"targets": 2, "correct": 1})
        self.assertEqual(scores["letters"]["word_2_letter_4"], {"targets": 0, "correct": 0})

    def test_unfinished_word_is_not_exact(self):
        with patch("character_context_trial.generate", return_value={"text": "cat"}):
            scores = score_cases(None, [("conversation", "prompt", ["cat"])])
        self.assertEqual(scores["word_1_exact"], 0)
        self.assertEqual(scores["word_1_unfinished"], 1)

    def test_character_examples_mask_word_tokens_without_mutating_parent(self):
        parent = PlasticityContext("text: cat cat.\n", 17)
        model = character_model(parent)
        examples = character_examples(model, [("text: ", "cat")])
        self.assertEqual(len(examples), 4)
        self.assertTrue(parent.word_tokens)
        self.assertFalse(model.word_tokens)
        for _, target, prior in examples:
            self.assertIn(target, model.character_tokens)
            self.assertTrue(all(not prior[token] for token in parent.word_tokens))

    def test_text_context_does_not_include_target_and_stops_at_boundary(self):
        docs = [("bible", "test", "one two three. next should not count."),
                ("oanc", "test", "red blue green! extra words here.")]
        cases = text_cases(docs, limit=1)
        self.assertEqual(cases[0], ("bible", "text: one two ", ["three"]))
        self.assertEqual(cases[1], ("bible", "text: one ", ["two", "three"]))
        self.assertEqual(cases[2], ("oanc", "text: red blue ", ["green"]))
        with self.assertRaises(ValueError):
            text_cases(docs[:1], limit=1)

    def test_conversation_only_samples_no_corpus_and_replays(self):
        class Backend:
            def __init__(self):
                self.selected = []

            def update(self, context, target, *, output_prior, learning_rate):
                self.selected.append(target)

        class Model:
            def __init__(self):
                self.model = Backend()

        conversation = [([0], 1, [1])]
        corpus = [([0], 2, [1])]
        first, replay, mixed = Model(), Model(), Model()
        args = {"updates": 100, "seed": 3, "rate": 0.01}
        draws = train_characters(first, conversation, corpus, conversation_only=True, **args)s)
        replay_draws = train_characters(replay, conversation, corpus, conversation_only=True, **args)
        mixed_draws = train_characters(mixed, conversation, corpus, conversation_only=False, **args)
        self.assertEqual(draws, {"conversation": 100, "corpus": 0})
        self.assertEqual(draws, replay_draws)
        self.assertEqual(first.model.selected, replay.model.selected)
        self.assertEqual(set(first.model.selected), {1})
        self.assertGreater(mixed_draws["corpus"], 0)
        self.assertGreater(mixed_draws["conversation"], 0)
        rng = training_rng(3, 0, len(conversation), len(corpus))
        original_stream = []
        for _ in range(100):
            pool = corpus if rng.random() < CORPUS_RATE else conversation
            original_stream.append(pool[rng.randrange(len(pool))][1])
        self.assertEqual(mixed.model.selected, original_stream)

    def test_split_continuation_matches_uninterrupted_training(self):
        source = "text: cat dog.\nuser: Hi\nassistant: Yes.\n"
        parent = PlasticityContext(source, 17)
        for conversation_only in (True, False):
            with self.subTest(conversation_only=conversation_only):
                whole = character_model(PlasticityContext.restore(copy.deepcopy(parent.snapshot())))
                split = character_model(PlasticityContext.restore(copy.deepcopy(parent.snapshot())))
                conversations = character_examples(whole, [("assistant: ", "Yes.")])
                corpus = character_examples(whole, [("text: ", "cat dog.")])
                args = {"seed": 3, "rate": 0.01, "conversation_only": conversation_only}
                train_characters(whole, conversations, corpus, updates=40, **args)
                train_characters(split, conversations, corpus, updates=20, **args)
                resumed = character_model(PlasticityContext.restore(copy.deepcopy(split.snapshot())))
                train_characters(resumed, conversations, corpus, updates=20, offset=20, **args)
                self.assertEqual(resumed.snapshot(), whole.snapshot())

    def test_meal_training_replays_both_reviewed_pools(self):
        source = "text: cat dog.\nuser: Hi\nassistant: Yes.\n"
        parent = PlasticityContext(source, 17)
        first = character_model(PlasticityContext.restore(copy.deepcopy(parent.snapshot())))
        replay = character_model(PlasticityContext.restore(copy.deepcopy(parent.snapshot())))
        rehearsal = character_examples(first, [("assistant: ", "Yes.")])
        meal = character_examples(first, [("text: ", "cat dog.")])
        args = {"updates": 100, "seed": 3, "rate": 0.01}
        draws = train_meal(first, rehearsal, meal, **args)
        self.assertEqual(draws, train_meal(replay, rehearsal, meal, **args))
        self.assertEqual(first.snapshot(), replay.snapshot())
        self.assertGreater(draws["meal"], 0)
        self.assertGreater(draws["rehearsal"], 0)
        with self.assertRaises(ValueError):
            train_meal(first, rehearsal, [], **args)

    def test_split_meal_training_preserves_sampling_stream(self):
        parent = PlasticityContext("text: cat dog.\n", 17)
        whole = character_model(PlasticityContext.restore(copy.deepcopy(parent.snapshot())))
        split = character_model(PlasticityContext.restore(copy.deepcopy(parent.snapshot())))
        rehearsal = character_examples(whole, [("text: ", "cat")])
        meal = character_examples(whole, [("text: ", "dog")])
        args = {"seed": 3, "rate": 0.01}
        train_meal(whole, rehearsal, meal, updates=40, **args)
        train_meal(split, rehearsal, meal, updates=20, **args)
        resumed = character_model(PlasticityContext.restore(copy.deepcopy(split.snapshot())))
        train_meal(resumed, rehearsal, meal, updates=20, offset=20, **args)
        self.assertEqual(resumed.snapshot(), whole.snapshot())

    def test_retention_reports_replaced_correct_answers(self):
        before = {"examples": 2, "rows": [
            {"id": "a", "expected": "A", "exact": True},
            {"id": "b", "expected": "B", "exact": False},
        ]}
        after = {"examples": 2, "rows": [
            {"id": "a", "expected": "A", "exact": False},
            {"id": "b", "expected": "B", "exact": True},
        ]}
        self.assertEqual(assessment_summary(before, after),
                         {"examples": 2, "before_exact": 1, "after_exact": 1,
                          "previously_correct_lost": 1})


if __name__ == "__main__":
    unittest.main()
