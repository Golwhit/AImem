from collections import Counter
import string
import unittest

from contextual_word_experiment import ContextualGenerator
from next_word_after_generation import NextInitialRanker
from unsupplied_next_word_experiment import collect, evaluate, mixed_initial_ranking, proposals
from word_prefix_experiment import PrefixRanker


def mapping():
    return {letter: [(letter, ())] for letter in string.ascii_letters}


class UnsuppliedNextWordTests(unittest.TestCase):
    def test_collection_does_not_skip_excluded_words_or_cross_documents(self):
        docs = [("a", "train", "cat bird"), ("a", "test", "cat bird can't dog sun"),
                ("a", "test", "cat tree")]
        _, _, tests, _ = collect(docs)
        self.assertEqual(tests["a"], {("cat", "bird"): 1, ("dog", "sun"): 1, ("cat", "tree"): 1})

    def test_mixture_uses_candidate_probabilities_without_the_actual_initial(self):
        ranker = NextInitialRanker(mapping(), Counter({
            ("cat", "b"): 9, ("cat", "s"): 1, ("car", "b"): 1, ("car", "s"): 9,
        }))
        self.assertEqual(mixed_initial_ranking([("cat", 0.9), ("car", 0.1)], ranker)[0], "b")
        self.assertEqual(mixed_initial_ranking([("cat", 0.1), ("car", 0.9)], ranker)[0], "s")
        self.assertEqual(mixed_initial_ranking([], ranker), ())
        with self.assertRaises(ValueError):
            mixed_initial_ranking([("cat", 0)], ranker)

    def test_predicted_initial_is_reused_for_different_targets(self):
        ranker = NextInitialRanker(mapping(), Counter({
            ("cat", "b"): 9, ("cat", "s"): 1, ("car", "s"): 8, ("car", "b"): 2,
        }))
        base = PrefixRanker(mapping(), Counter({"cat": 8, "car": 2, "book": 5, "sack": 4}))
        candidate = ContextualGenerator(base, {})
        selected = proposals(base, ranker, "c")["greedy"]
        self.assertEqual(selected[1][0], "b")
        result = evaluate(base, candidate, ranker, Counter({("cat", "book"): 2, ("cat", "sack"): 3}))
        self.assertEqual(result["pairs"], 5)
        self.assertEqual(result["greedy_initial_correct"], 2)
        self.assertEqual(result["greedy_word_correct"], 2)
        self.assertEqual(result["beam20_mixture_initial_correct"], 2)

    def test_unavailable_greedy_path_is_not_counted_as_a_prediction(self):
        ranker = NextInitialRanker(mapping(), Counter({("abab", "b"): 9}))
        base = PrefixRanker(mapping(), Counter({"ababa": 9, "abab": 1, "book": 2}))
        result = evaluate(base, ContextualGenerator(base, {}), ranker, Counter({("abab", "book"): 1}))
        self.assertEqual(result["greedy_available"], 0)
        self.assertEqual(result["greedy_initial_correct"], 0)
        self.assertEqual(result["beam20_mixture_available"], 1)
        self.assertEqual(result["beam20_mixture_word_correct"], 1)


if __name__ == "__main__":
    unittest.main()
