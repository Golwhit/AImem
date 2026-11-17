from collections import Counter
import string
import unittest

from next_word_after_generation import NextInitialRanker, collect, evaluate


def mapping():
    return {letter: [(letter, ())] for letter in string.ascii_letters}


class Generator:
    def generate_up_to_four(self, initial):
        return {"c": ("cat", True), "d": ("dog", True), "t": ("tall", False)}[initial]


class NextWordTests(unittest.TestCase):
    def test_collection_never_skips_invalid_words_or_crosses_documents(self):
        docs = [("x", "train", "cat sun can't dog tree tall sun"),
                ("x", "train", "dog time")]
        _, pairs, totals = collect(docs)
        self.assertEqual(pairs["x", "train"], {
            ("cat", "s"): 1, ("dog", "t"): 2, ("tree", "t"): 1, ("tall", "s"): 1,
        })
        self.assertEqual(totals["x", "train"]["excluded_words"], 1)

    def test_compares_correct_word_with_generated_word_and_no_stop(self):
        ranker = NextInitialRanker(mapping(), Counter({
            ("cat", "s"): 4, ("dog", "t"): 3, ("dot", "s"): 2,
            ("tall", "t"): 4, ("thin", "s"): 1,
        }))
        pairs = Counter({("cat", "s"): 2, ("dog", "t"): 1, ("dot", "s"): 1,
                         ("tall", "t"): 2, ("thin", "s"): 1})
        result = evaluate(ranker, Generator(), pairs)
        self.assertEqual(result["pairs"], 7)
        self.assertEqual(result["known_word_top1_correct"], 7)
        self.assertEqual(result["generated_end_available"], 4)
        self.assertEqual(result["generated_previous_exact"], 3)
        self.assertEqual(result["generated_top1_correct"], 3)
        self.assertEqual(result["baseline_top1_correct"], 4)
        self.assertTrue(ranker.rank("X")[1])


if __name__ == "__main__":
    unittest.main()
