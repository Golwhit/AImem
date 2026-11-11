import json
import tempfile
import unittest
from pathlib import Path

import neural_patterns
from neural_patterns import FixedPatterns


class FixedPatternTests(unittest.TestCase):
    def test_pattern_crosses_spaces_and_needs_support(self):
        text = "the Holy Spirit came. " * 6 + "the Holy One. "
        patterns = FixedPatterns(text, support=5, max_trigger=4)
        self.assertEqual(patterns.match("Spi")[1], "rit came. the Holy ")
        self.assertIsNone(patterns.match("Holy "[-4:]))  # "oly " was followed by "S" and "O"
        self.assertIsNone(FixedPatterns(text, support=50, max_trigger=4).match("Spi"))

    def test_longer_trigger_kept_only_when_it_adds(self):
        patterns = FixedPatterns("abcd abce " * 10, support=5, max_trigger=3)
        self.assertIn("d", patterns.patterns)
        self.assertNotIn("cd", patterns.patterns)  # "d" already forces the same continuation
        self.assertNotIn("abc", patterns.patterns)  # never forced: d or e follows

    def test_reading_counts_coverage_accuracy_and_decisions(self):
        patterns = FixedPatterns("xyz " * 20, support=5, max_trigger=2, max_span=3)
        result = patterns.read("xyz xyQ")
        self.assertEqual(result["characters"], 7)
        self.assertGreater(result["patterns_fired"], 0)
        self.assertEqual(result["correct_characters"] + result["wrong_characters"], result["claimed_characters"])
        self.assertGreaterEqual(result["wrong_characters"], 1)
        self.assertLess(result["decisions"], result["characters"])
        self.assertAlmostEqual(result["decisions_saved_fraction"], 1 - result["decisions"] / 7)

    def test_locked_evaluation_runs_only_once_and_needs_registration(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(FileNotFoundError):
                neural_patterns.evaluate_locked(root, acknowledge_review=True)
            (root / neural_patterns.PATTERN_DIRECTORY).mkdir(parents=True)
            (root / neural_patterns.PATTERN_DIRECTORY / "locked-evaluation.json").write_text(json.dumps({}))
            with self.assertRaisesRegex(ValueError, "already evaluated"):
                neural_patterns.evaluate_locked(root, acknowledge_review=True)

    def test_staging_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / neural_patterns.LOCKED_DIRECTORY).mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                neural_patterns.stage_chapters(root, neural_patterns.LOCKED_DIRECTORY, (28,), "locked",
                                               fetcher=lambda url: b"")


if __name__ == "__main__":
    unittest.main()
