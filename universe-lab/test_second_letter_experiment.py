import hashlib
import json
from pathlib import Path
import string
import subprocess
import sys
import tempfile
import unittest

from second_letter_experiment import PatternRanker, collect, evaluate, require_hash


def mapping():
    result = {letter: [(letter, ())] for letter in string.ascii_letters}
    result["s"].append(("another-s-pattern", ()))
    return result


class SecondLetterTests(unittest.TestCase):
    def test_hash_gate_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input"
            path.write_bytes(b"approved")
            approved = hashlib.sha256(b"approved").hexdigest()
            self.assertEqual(require_hash(path, approved), approved)
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                require_hash(path, approved)

    def test_saved_ranking_can_be_queried_without_training_inputs(self):
        ranker = PatternRanker(mapping(), {("s", "t"): 2})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps({"kind": "private_second_letter_pattern_ranking",
                                        "first_letter_rankings": {"s": ranker.ranking(("s", ()))}}))
            script = Path(__file__).with_name("second_letter_experiment.py")
            result = subprocess.run([sys.executable, str(script), "--from-report", str(path),
                                     "--first", "s"], capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout)["ranking"][0]["letter"], "t")
            invalid = subprocess.run([sys.executable, str(script), "--from-report", str(path)],
                                     capture_output=True, text=True)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn("requires a single", invalid.stderr)

    def test_collection_counts_only_real_case_sensitive_second_letters(self):
        groups, totals = collect([("a", "train", "a an An cat a's Élan"),
                                  ("a", "test", "at Éclair")])
        self.assertEqual(groups["a", "train"], {("a", "n"): 1, ("A", "n"): 1, ("c", "a"): 1})
        self.assertEqual(totals["a", "train"]["one_character_words"], 1)
        self.assertEqual(totals["a", "train"]["non_ascii_or_nonletter_pairs"], 2)
        self.assertEqual(groups["a", "test"], {("a", "t"): 1})

    def test_ranking_normalizes_pattern_family_and_falls_back_without_test_leakage(self):
        ranker = PatternRanker(mapping(), {("s", "t"): 3, ("s", "h"): 1, ("T", "h"): 2})
        first = ranker.ranking(("another-s-pattern", ()))
        self.assertEqual([item["letter"] for item in first["ranking"][:2]], ["t", "h"])
        self.assertEqual(first["observations"], 4)
        unseen = ranker.ranking(("z", ()))
        self.assertTrue(unseen["uses_global_fallback"])
        self.assertEqual(unseen["observations"], 0)
        self.assertEqual(unseen["ranking"][0]["letter"], "h")
        result = evaluate(ranker, {("s", "t"): 2, ("s", "h"): 1, ("s", "A"): 1, ("z", "h"): 1})
        self.assertEqual((result["top1_correct"], result["top3_correct"]), (3, 4))
        self.assertEqual((result["baseline_top1_correct"], result["unknown_first"]), (2, 1))
        with self.assertRaises(ValueError):
            ranker.ranking(("not-assigned", ()))


if __name__ == "__main__":
    unittest.main()
