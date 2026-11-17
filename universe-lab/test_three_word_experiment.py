from collections import Counter
import string
import unittest
from unittest.mock import Mock, patch

from contextual_word_experiment import ContextualGenerator
from joint_word_beam_experiment import JointWordBeam
from next_word_after_generation import NextInitialRanker
from three_word_experiment import PhrasePredictor, collect, evaluate
from word_prefix_experiment import PrefixRanker


def mapping():
    return {letter: [(letter, ())] for letter in string.ascii_letters}


class ThreeWordTests(unittest.TestCase):
    def test_fragment_collection_respects_documents_and_punctuation(self):
        docs = [("x", "train", "a cat sun"),
                ("x", "test", "a cat sun. a dog ran can't a fox ran"),
                ("x", "test", "a cat")]
        _, _, tests, _ = collect(docs)
        self.assertEqual(tests["x"], {("a", "cat", "sun"): 1, ("a", "dog", "ran"): 1,
                                     ("a", "fox", "ran"): 1})

    def test_three_words_are_generated_without_future_initials(self):
        base = PrefixRanker(mapping(), Counter({"a": 10, "cat": 5, "sun": 5}))
        ranker = NextInitialRanker(mapping(), Counter({("a", "c"): 100, ("cat", "s"): 100}))
        contextual = ContextualGenerator(base, {})
        predictor = PhrasePredictor(base, contextual, ranker, JointWordBeam(base, contextual, ranker))
        self.assertEqual(predictor.greedy("a"), ("a", "c", "cat", "s", "sun"))
        self.assertEqual(predictor.beam("a"), ("a", "c", "cat", "s", "sun"))
        result = evaluate(predictor, Counter({("a", "cat", "sun"): 2, ("a", "cat", "sad"): 3}))
        self.assertEqual(result["beam_first_two_correct"], 5)
        self.assertEqual(result["beam_third_initial_correct"], 5)
        self.assertEqual(result["beam_phrase_correct"], 2)
        self.assertEqual(result["known_second_word_correct"], 2)
        self.assertEqual(result["beam_phrase_correct_when_greedy_available"], 2)

    def test_unfinished_first_word_is_not_a_complete_phrase(self):
        base = PrefixRanker(mapping(), Counter({"ababa": 10, "cat": 2}))
        ranker = NextInitialRanker(mapping(), Counter({("abab", "c"): 10}))
        contextual = ContextualGenerator(base, {})
        predictor = PhrasePredictor(base, contextual, ranker, JointWordBeam(base, contextual, ranker))
        result = evaluate(predictor, Counter({("abab", "cat", "sun"): 1}))
        self.assertEqual(result["greedy_third_available"], 0)
        self.assertEqual(result["beam_third_available"], 0)
        self.assertEqual(result["beam_phrase_correct"], 0)
        self.assertEqual(result["beam_phrase_correct_when_greedy_available"], 0)

    def test_continuing_paths_are_not_completed_words(self):
        joint = Mock()
        predictor = PhrasePredictor(None, None, None, joint)
        with patch("three_word_experiment.beam_words", return_value=[("a", 1.0)]):
            joint.search_candidates.return_value = [("catsx", 0.9, "continues")]
            self.assertEqual(predictor.beam("a"), ("a", "c", None, None, None))
            joint.search_candidates.assert_called_with([("a", 1.0)], 20)
            joint.search_candidates.side_effect = [
                [("cat", 0.9, "ended")], [("sunny", 0.9, "continues")],
            ]
            self.assertEqual(predictor.beam("a"), ("a", "c", "cat", "s", None))
            joint.search_candidates.assert_called_with([("cat", 1.0)], 20)


if __name__ == "__main__":
    unittest.main()
