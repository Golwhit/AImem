from collections import Counter
import string
import unittest

from contextual_word_experiment import ContextualGenerator, collect, evaluate, train_contexts
from next_word_after_generation import NextInitialRanker
from word_prefix_experiment import PrefixRanker


def mapping():
    return {letter: [(letter, ())] for letter in string.ascii_letters}


class ContextualWordTests(unittest.TestCase):
    def test_collection_does_not_skip_excluded_words_or_cross_documents(self):
        docs = [("a", "train", "in star"), ("a", "test", "in star tree can't a star sun"),
                ("a", "test", "in star")]
        _, _, windows, _ = collect(docs)
        self.assertEqual(windows["a"], {("in", "star", "t"): 1, ("a", "star", "s"): 1})

    def test_context_can_change_generation_and_unseen_context_falls_back(self):
        base = PrefixRanker(mapping(), Counter({"star": 20, "stop": 4}))
        docs = [("x", "train", "in stop " * 20), ("x", "test", "in star " * 100)]
        candidate = ContextualGenerator(base, train_contexts(docs, {"in"}))
        self.assertEqual(base.generate_up_to_four("s"), ("star", True))
        self.assertEqual(candidate.generate_up_to_four("in", "s"), ("stop", True, True))
        self.assertEqual(candidate.generate_up_to_four("zz", "s"), ("star", True, False))

    def test_truncating_training_words_preserves_all_four_prefix_decisions(self):
        complete = PrefixRanker(mapping(), Counter({"longer": 2, "long": 1}))
        truncated = PrefixRanker(mapping(), Counter({"longe": 2, "long": 1}))
        for prefix in ("l", "lo", "lon", "long"):
            self.assertEqual(complete.rank(prefix)[0], truncated.rank(prefix)[0])
            self.assertEqual(complete.rank(prefix)[3], truncated.rank(prefix)[3])

    def test_evaluation_scores_generated_word_and_next_initial_on_same_windows(self):
        base = PrefixRanker(mapping(), Counter({"star": 20, "stop": 4, "in": 30}))
        contexts = train_contexts([("x", "train", "in stop " * 20)], {"in"})
        candidate = ContextualGenerator(base, contexts)
        next_ranker = NextInitialRanker(mapping(), Counter({("star", "x"): 2, ("stop", "t"): 4}))
        result = evaluate(base, candidate, next_ranker, Counter({("in", "stop", "t"): 3}))
        self.assertEqual(result["control_word_correct"], 0)
        self.assertEqual(result["known_previous_word_correct"], 3)
        self.assertEqual(result["known_previous_next_initial_correct"], 3)
        self.assertEqual(result["generated_previous_available"], 3)


if __name__ == "__main__":
    unittest.main()
