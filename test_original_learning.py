import math
import unittest

from original_learning import features, fit, loss, probability, tokens


class OriginalLearningTests(unittest.TestCase):
    def test_text_and_overlap(self):
        self.assertEqual(tokens("Apple, PIE!"), ["apple", "pie"])
        self.assertEqual(features("apple pie", "apple pie", ["apple pie"]), [1.0] * 4)
        self.assertEqual(features("apple", "bonds", []), [0.0] * 4)
        self.assertEqual(features("", "apple", []), [0.0] * 4)

    def test_training_reduces_error_and_changes_predictions(self):
        rows = [[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]] * 4
        labels = [1, 0] * 4
        weights, intercept = fit(rows, labels)
        self.assertLess(loss(rows, labels, weights, intercept), loss(rows, labels, [0.0] * 4, 0.0))
        self.assertGreater(probability(rows[0], weights, intercept), 0.8)
        self.assertLess(probability(rows[1], weights, intercept), 0.2)

    def test_extreme_scores_are_finite(self):
        self.assertEqual(probability([1.0] * 4, [1000.0] * 4, 0.0), 1.0)
        self.assertEqual(probability([1.0] * 4, [-1000.0] * 4, 0.0), 0.0)
        self.assertTrue(math.isfinite(loss([[1.0] * 4], [1], [-1000.0] * 4, 0.0)))

    def test_context_target_learns_between_direct_and_unrelated(self):
        rows = [[1.0] * 4, [0.5] * 4, [0.0] * 4] * 5
        labels = [1.0, 0.5, 0.0] * 5
        weights, intercept = fit(rows, labels)
        direct, context, unrelated = [probability(row, weights, intercept) for row in rows[:3]]
        self.assertGreater(direct, context)
        self.assertGreater(context, unrelated)
        self.assertAlmostEqual(context, 0.5, delta=0.06)
        self.assertLess(loss(rows, labels, weights, intercept), loss(rows, labels, [0.0] * 4, 0.0))
        with self.assertRaises(ValueError):
            fit(rows, [0.2] * len(rows))


if __name__ == "__main__":
    unittest.main()