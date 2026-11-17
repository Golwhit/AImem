import unittest

from word_start_experiment import evaluate, observations, split, train


class WordStartTests(unittest.TestCase):
    def test_context_prediction_and_fallback_are_scored_without_crossing_documents(self):
        docs = [("one", "train", "before Apple before Apple before zebra before"),
                ("one", "test", "before Apple unseen zebra"),
                ("two", "test", "before zebra")]
        predictions, baseline, counts = train(iter(docs))
        patterns = {letter: [(letter, ())] for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"}
        patterns["A"].append(("different", ()))
        results = evaluate(iter(docs), predictions, baseline, patterns)
        self.assertEqual(predictions["before"], "A")
        self.assertEqual(baseline, "b")
        self.assertEqual(counts["one_observations"], 6)
        self.assertEqual(results["one"]["observations"], 3)
        self.assertEqual(results["one"]["context_correct"], 1)
        self.assertEqual(results["one"]["decoded_pattern_correct"], 1)
        self.assertEqual(results["two"]["context_correct"], 0)

    def test_tokenization_and_split_are_stable(self):
        self.assertEqual(list(observations("Start Élan Don't STOP")), [("élan", "D"), ("don't", "S")])
        self.assertEqual(split(b"same document"), split(b"same document"))


if __name__ == "__main__":
    unittest.main()
