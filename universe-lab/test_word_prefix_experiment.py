from collections import Counter
import string
import unittest

from word_prefix_experiment import END, PrefixRanker, collect, evaluate, lookup


def mapping():
    return {letter: [(letter, ())] for letter in string.ascii_letters}


class WordPrefixTests(unittest.TestCase):
    def test_collection_includes_short_words_and_excludes_partial_non_ascii_words(self):
        groups, totals = collect([("x", "train", "a star can't Élan Story")])
        self.assertEqual(groups["x", "train"], {"a": 1, "star": 1, "Story": 1})
        self.assertEqual(totals["x", "train"]["excluded_non_ascii_or_nonletter"], 2)

    def test_end_prediction_generation_and_fallback(self):
        ranker = PrefixRanker(mapping(), Counter({"star": 5, "stop": 3, "story": 1, "a": 2}))
        self.assertEqual(ranker.rank("str")[2], "position")
        self.assertEqual(ranker.rank("sxo")[2], "last_letter")
        self.assertEqual(ranker.rank("sta")[0][0], "r")
        self.assertEqual(ranker.rank("star")[0][0], END)
        self.assertEqual(ranker.generate_up_to_four("s"), ("star", True))
        self.assertEqual(ranker.generate_up_to_four("a"), ("a", True))
        saved = ranker.saved_counts()
        self.assertEqual(lookup(saved, "sta")["ranking"][0]["next"], "r")
        self.assertEqual(lookup(saved, "str")["fallback"], "position")
        self.assertEqual(lookup(saved, "sxo")["fallback"], "last_letter")
        self.assertEqual(lookup(saved, "st")["ranking"][0]["next"], ranker.rank("st")[0][0])
        with self.assertRaises(ValueError):
            ranker.rank("s!")

    def test_four_letter_score_requires_the_correct_end_decision(self):
        ranker = PrefixRanker(mapping(), Counter({"star": 5, "stop": 3, "story": 1, "a": 2}))
        results = evaluate(ranker, Counter({"star": 2, "stop": 1, "story": 1, "a": 1}))
        self.assertEqual(results["generation"]["prefix_four_letter_targets"], 3)
        self.assertEqual(results["generation"]["prefix_correct_complete_four"], 2)
        self.assertEqual(results["generation"]["prefix_generated_four"], 4)
        self.assertEqual(results["generation"]["prefix_generated_four_correct"], 2)
        self.assertEqual(results["steps"]["1"]["actual_end"], 1)

        continues = PrefixRanker(mapping(), Counter({"starfish": 5, "star": 1}))
        unfinished = evaluate(continues, Counter({"star": 1}))["generation"]
        self.assertEqual(unfinished["prefix_correct_first_four"], 1)
        self.assertEqual(unfinished["prefix_correct_complete_four"], 0)


if __name__ == "__main__":
    unittest.main()
