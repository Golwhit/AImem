from collections import Counter
import unittest
from unittest.mock import patch

from active_three_word_evaluation import evaluate, words_from_output


class ActiveThreeWordEvaluationTests(unittest.TestCase):
    def test_requires_completed_words_and_whitespace(self):
        self.assertEqual(words_from_output("a", " cat sun\nmore"),
                         [("a", "a"), ("c", "cat"), ("s", "sun")])
        self.assertEqual(words_from_output("a", " cat sun"),
                         [("a", "a"), ("c", "cat"), ("s", None)])
        self.assertEqual(words_from_output("a", " cat. sun\n"),
                         [("a", "a"), ("c", "cat")])
        self.assertEqual(words_from_output("a", " cat\n sun"),
                         [("a", "a"), ("c", "cat")])

    def test_evaluation_predicts_without_using_second_or_third_targets(self):
        triples = {"x": Counter({("a", "cat", "sun"): 2, ("a", "cat", "sad"): 3})}
        with patch("active_three_word_evaluation.generate", return_value={"text": " cat sun\n"}):
            result = evaluate(object(), triples)["combined"]
        self.assertEqual(result["fragments"], 5)
        self.assertEqual(result["third_initial_correct"], 5)
        self.assertEqual(result["third_word_completed"], 5)
        self.assertEqual(result["third_word_correct"], 2)
        self.assertEqual(result["phrase_correct"], 2)


if __name__ == "__main__":
    unittest.main()
