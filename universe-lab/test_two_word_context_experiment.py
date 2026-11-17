from collections import Counter
import string
import unittest
from unittest.mock import patch

from contextual_word_experiment import ContextualGenerator
from next_word_after_generation import NextInitialRanker
from two_word_context_experiment import (
    TwoWordContext, blend, build_index, evaluate, generated_predictions, greedy_third, sampled, training_triples,
)
from word_beam_experiment import next_initial_distribution
from word_prefix_experiment import PrefixRanker


def mapping():
    return {letter: [(letter, ())] for letter in string.ascii_letters}


class TwoWordContextTests(unittest.TestCase):
    def test_training_samples_are_nested_and_do_not_remove_test_documents(self):
        docs = [("x", "train", f"document {number}") for number in range(100)]
        docs.append(("x", "test", "held-out"))
        small = set(sampled(docs, 25))
        medium = set(sampled(docs, 50))
        full = set(sampled(docs, 100))
        self.assertTrue(small < medium < full)
        self.assertIn(("x", "test", "held-out"), small)
        self.assertEqual(full, set(docs))

    def test_triples_reset_at_punctuation_excluded_words_and_documents(self):
        docs = [("x", "train", "a cat sun. a dog ran can't a fox ran"),
                ("x", "train", "a cat"), ("x", "test", "a cat sun")]
        self.assertEqual(list(training_triples(docs)),
                         [(("a", "cat"), "sun"), (("a", "dog"), "ran"),
                          (("a", "fox"), "ran")])

    def test_two_words_change_prediction_and_unseen_pairs_fall_back(self):
        base = PrefixRanker(mapping(), Counter({"a": 10, "cat": 10, "sun": 10, "bat": 10}))
        contextual = ContextualGenerator(base, {})
        ranker = NextInitialRanker(mapping(), Counter({("cat", "b"): 20}))
        two = TwoWordContext(contextual, {("a", "cat"): Counter({"s": 100})},
                             {(("a", "cat"), "s"): Counter({"u": 100})})
        self.assertEqual(greedy_third("a", "cat", ranker, contextual), ("b", "bat"))
        self.assertEqual(greedy_third("a", "cat", ranker, contextual, two), ("s", "sun"))
        scores, used = two.weights("a", "cat", "s")
        self.assertTrue(used)
        self.assertGreater(scores["u"], contextual.weights("cat", "s")[0]["u"])
        prior = next_initial_distribution("cat", ranker)
        self.assertEqual(two.initial_distribution("x", "cat", ranker), prior)
        self.assertEqual(two.weights("x", "cat", "b")[0], contextual.weights("cat", "b")[0])
        self.assertEqual(blend(Counter({"a": 1}), None), Counter({"a": 1}))

    def test_index_keeps_endings_and_long_word_continuations_distinct(self):
        docs = [("x", "train", "a cat sun. a cat sunny")]
        initials, prefixes = build_index(docs, {("a", "cat")})
        self.assertEqual(initials["a", "cat"], Counter({"s": 2}))
        self.assertEqual(prefixes[(("a", "cat"), "sun")], Counter({"<END>": 1, "n": 1}))
        self.assertEqual(prefixes[(("a", "cat"), "sunn")], Counter({"y": 1}))

    def test_generated_chain_uses_both_generated_words_for_third(self):
        base = PrefixRanker(mapping(), Counter({"a": 10, "cat": 10, "sun": 10, "bat": 10}))
        contextual = ContextualGenerator(base, {})
        ranker = NextInitialRanker(mapping(), Counter({("a", "c"): 100, ("cat", "b"): 100}))
        two = TwoWordContext(contextual, {("a", "cat"): Counter({"s": 100})}, {})
        self.assertEqual(generated_predictions(base, contextual, ranker, two)["a"],
                         ("a", "cat", ("b", "bat"), ("s", "sun")))

    def test_generated_arms_use_same_prior_words_without_target_leakage(self):
        base = PrefixRanker(mapping(), Counter({"a": 10, "cat": 10, "sun": 10, "bat": 10}))
        contextual = ContextualGenerator(base, {})
        ranker = NextInitialRanker(mapping(), Counter({("cat", "b"): 20}))
        two = TwoWordContext(contextual, {}, {})
        predictions = {"a": ("a", "cat", ("s", "sun"), ("b", "bat"))}
        triples = {"x": Counter({("a", "cat", "sun"): 2, ("a", "cat", "bat"): 3})}
        with patch("two_word_context_experiment.generated_predictions", return_value=predictions):
            scores = evaluate(base, contextual, ranker, two, triples)["combined"]
        self.assertEqual(scores["fragments"], 5)
        self.assertEqual(scores["generated_first_two_correct"], 5)
        self.assertEqual(scores["one_word_phrase_correct"], 2)
        self.assertEqual(scores["two_word_phrase_correct"], 3)


if __name__ == "__main__":
    unittest.main()
