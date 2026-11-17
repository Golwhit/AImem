from collections import Counter
import string
import unittest

from next_word_after_generation import NextInitialRanker
from word_beam_experiment import beam_words, evaluate, select_with_next_initial
from word_prefix_experiment import PrefixRanker


def mapping():
    return {letter: [(letter, ())] for letter in string.ascii_letters}


class WordBeamTests(unittest.TestCase):
    def test_multiple_candidates_can_be_reranked_by_next_initial(self):
        base = PrefixRanker(mapping(), Counter({"cat": 4, "car": 3, "can": 2}))
        candidates = beam_words(base, "c", 5)
        self.assertEqual([word for word, _ in candidates], ["cat", "car", "can"])
        transitions = NextInitialRanker(mapping(), Counter({
            ("cat", "b"): 1, ("cat", "s"): 9,
            ("car", "b"): 9, ("car", "s"): 1,
            ("can", "b"): 1, ("can", "s"): 9,
        }))
        self.assertEqual(select_with_next_initial(candidates, "b", transitions), "car")
        self.assertIsNone(select_with_next_initial([], "b", transitions))

    def test_beam_counts_continuing_paths_against_the_width(self):
        base = PrefixRanker(mapping(), Counter({"ababa": 9, "abab": 1}))
        self.assertEqual(base.generate_up_to_four("a"), ("abab", False))
        self.assertEqual(beam_words(base, "a", 1), [])
        self.assertEqual([word for word, _ in beam_words(base, "a", 2)], ["abab"])
        with self.assertRaises(ValueError):
            beam_words(base, "a", 0)

    def test_scoring_propagates_the_selected_word_on_the_same_subset(self):
        base = PrefixRanker(mapping(), Counter({"cat": 4, "car": 3, "can": 2}))
        transitions = NextInitialRanker(mapping(), Counter({
            ("cat", "b"): 1, ("cat", "s"): 9,
            ("car", "b"): 9, ("car", "s"): 1,
            ("can", "b"): 1, ("can", "s"): 9,
            ("be", "s"): 4, ("by", "z"): 4,
        }))

        class Candidate:
            def generate_up_to_four(self, previous, initial):
                return ("be" if previous == "car" else "by", True, True)

        scores = evaluate(base, Candidate(), transitions, Counter({("car", "be", "s"): 2}))
        self.assertEqual(scores["greedy_available_windows"], 2)
        self.assertEqual(scores["greedy_previous_correct"], 0)
        self.assertEqual(scores["beam5_next_initial_previous_correct"], 2)
        self.assertEqual(scores["beam5_next_initial_current_correct_when_greedy_available"], 2)
        self.assertEqual(scores["beam5_next_initial_following_correct"], 2)


if __name__ == "__main__":
    unittest.main()
