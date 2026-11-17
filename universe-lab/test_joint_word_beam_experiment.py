from collections import Counter
import string
import unittest

from contextual_word_experiment import ContextualGenerator
from joint_word_beam_experiment import JointWordBeam, evaluate
from next_word_after_generation import NextInitialRanker
from word_prefix_experiment import PrefixRanker
from word_beam_experiment import beam_words


def mapping():
    return {letter: [(letter, ())] for letter in string.ascii_letters}


class JointWordBeamTests(unittest.TestCase):
    def test_search_does_not_convert_a_longer_path_into_a_short_word(self):
        base = PrefixRanker(mapping(), Counter({"cat": 5, "ababa": 9, "abab": 1}))
        ranker = NextInitialRanker(mapping(), Counter({("cat", "a"): 100}))
        joint = JointWordBeam(base, ContextualGenerator(base, {}), ranker)
        paths = joint.search("c", 20)
        self.assertEqual(paths[0][0], "ababa")
        self.assertEqual(paths[0][2], "continues")
        self.assertTrue(any(word == "abab" and state == "ended" for word, _, state in paths))

    def test_evaluation_predicts_without_using_actual_target_initial(self):
        base = PrefixRanker(mapping(), Counter({"cat": 8, "car": 2, "book": 5, "sack": 4}))
        ranker = NextInitialRanker(mapping(), Counter({
            ("cat", "b"): 9, ("cat", "s"): 1, ("car", "s"): 8, ("car", "b"): 2,
        }))
        contextual = ContextualGenerator(base, {})
        joint = JointWordBeam(base, contextual, ranker)
        results = evaluate(base, contextual, ranker, joint, Counter({("cat", "book"): 2, ("cat", "sack"): 3}))
        self.assertEqual(results["pairs"], 5)
        self.assertEqual(results["joint20_initial_correct"], 2)
        self.assertEqual(results["joint20_word_correct"], 2)
        self.assertEqual(results["baseline_word_correct"], 2)

    def test_word_paths_marginalize_preceding_candidates(self):
        base = PrefixRanker(mapping(), Counter({"cat": 6, "car": 4, "star": 3, "stop": 7}))
        ranker = NextInitialRanker(mapping(), Counter({("cat", "s"): 10, ("car", "s"): 10}))
        contexts = {
            ("cat", "s"): Counter({"t": 100}), ("car", "s"): Counter({"t": 100}),
            ("cat", "st"): Counter({"a": 100}), ("car", "st"): Counter({"o": 100}),
            ("cat", "sta"): Counter({"r": 100}), ("car", "sto"): Counter({"p": 100}),
        }
        paths = JointWordBeam(base, ContextualGenerator(base, contexts), ranker).search("c", 20)
        self.assertEqual((paths[0][0], paths[0][2]), ("star", "ended"))
        self.assertTrue(any(word == "stop" and state == "ended" for word, _, state in paths))
        joint = JointWordBeam(base, ContextualGenerator(base, contexts), ranker)
        self.assertEqual(joint.search("c", 20), joint.search_candidates(beam_words(base, "c", 20), 20))
        with self.assertRaises(ValueError):
            joint.search_candidates([("cat", float("nan"))], 20)

    def test_no_finished_previous_candidate_is_unavailable(self):
        base = PrefixRanker(mapping(), Counter({"ababa": 10, "book": 2}))
        ranker = NextInitialRanker(mapping(), Counter({("abab", "b"): 9}))
        contextual = ContextualGenerator(base, {})
        results = evaluate(base, contextual, ranker, JointWordBeam(base, contextual, ranker),
                           Counter({("abab", "book"): 1}))
        self.assertEqual(results["baseline_available"], 0)
        self.assertEqual(results["joint20_available"], 0)
        self.assertEqual(results["joint20_word_correct"], 0)


if __name__ == "__main__":
    unittest.main()
