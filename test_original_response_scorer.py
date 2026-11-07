import tempfile
import unittest
from pathlib import Path

from original_response_scorer import ResponseScorer, response_features


class ResponseScorerTests(unittest.TestCase):
    def test_empty_candidates_have_explicit_unscored_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            scorer = ResponseScorer(Path(directory))
            result = scorer.select("How can I tell if you're actually trainiing?", [{"text": ""}, {"text": " "}])
            self.assertEqual(result["text"], "")
            self.assertEqual(result["selection"]["candidate_count"], 0)
            self.assertIsNone(result["selection"]["score"])

    def test_local_feedback_changes_selection_before_global_training(self):
        with tempfile.TemporaryDirectory() as directory:
            scorer = ResponseScorer(Path(directory))
            candidates = [{"text": "hello world"}, {"text": "hello friend"}]
            self.assertEqual(scorer.select("greeting", candidates)["text"], "hello world")
            scorer.feedback("greeting", "hello world", "needs_correction", "user test", True)
            self.assertEqual(scorer.select("greeting", candidates)["text"], "hello friend")
            self.assertEqual(scorer.status()["generation"], 0)
            restored = ResponseScorer(Path(directory))
            self.assertEqual(restored.select("greeting", candidates)["text"], "hello friend")

    def test_review_required_and_revisions_replace_examples(self):
        with tempfile.TemporaryDirectory() as directory:
            scorer = ResponseScorer(Path(directory))
            with self.assertRaises(ValueError):
                scorer.feedback("apple", "apple", "useful", "test")
            self.assertFalse(scorer.path.exists())
            scorer.feedback("apple", "apple", "useful", "test", True)
            scorer.feedback("APPLE", "apple", "needs_correction", "test", True)
            restored = ResponseScorer(Path(directory))
            self.assertEqual(restored.status()["examples"], 1)
            self.assertEqual(restored.state["examples"][0]["rating"], "needs_correction")
            self.assertEqual(restored.status()["generation"], 0)

    def test_training_improves_unseen_queries_and_preserves_previous_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            scorer = ResponseScorer(Path(directory))
            for subject in ("apple", "garden", "water", "moon", "sun", "book"):
                prompt = f"describe {subject}"
                scorer.feedback(prompt, f"describe {subject} clearly", "useful", "original test fixture", True)
                status = scorer.feedback(prompt, "noise noise noise noise", "needs_correction", "original test fixture", True)
            self.assertEqual(status["training"]["status"], "accepted")
            self.assertLess(status["training"]["candidate_loss"], status["training"]["baseline_loss"])
            self.assertGreater(scorer.score("describe star", "describe star clearly"), scorer.score("describe star", "noise noise noise noise"))
            self.assertTrue(scorer.path.with_name("response-scorer.previous.json").exists())
            self.assertEqual(status["training"]["train_examples"], 8)
            self.assertEqual(status["training"]["validation_examples"], 4)
            self.assertLess(response_features("noise", "noise noise noise")[3], 1.0)


if __name__ == "__main__":
    unittest.main()