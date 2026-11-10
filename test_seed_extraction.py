import json
import tempfile
import unittest
from pathlib import Path

from seed_extraction import MAX_SOURCE_CHARACTERS, RULESET, extract, extract_file


class SeedExtractionTests(unittest.TestCase):
    def test_dead_opening_is_separate_from_identical_interior_windows(self):
        groups = extract("t next t next")
        self.assertTrue(any(item["text"] == "t " and item["offsets"] == [0] for item in groups["dead"]))
        self.assertTrue(any(item["text"] == "t " and item["offsets"] == [5, 7] for item in groups["maybe"]))
        self.assertFalse(any(0 in item["offsets"] and item["text"] == "t " for item in groups["maybe"]))
        self.assertFalse(extract("t. next")["dead"])

    def test_source_start_is_a_terminus_without_admitting_interior_fragments(self):
        for letter in ("A", "I"):
            with self.subTest(letter=letter):
                text = f"{letter} test. {letter} again."
                groups = extract(text)
                alive = next(item for item in groups["alive"] if item["text"] == letter + " ")
                self.assertEqual(alive["offsets"], [0])
                self.assertTrue(alive["layer_zero_eligible"])
                interior = next(item for item in groups["maybe"] if item["text"] == letter + " ")
                self.assertEqual(interior["offsets"], [8])
                self.assertFalse(interior["layer_zero_eligible"])
                bare = [item for item in groups["maybe"] if item["text"] == letter]
                self.assertEqual(len(bare), 2)
                self.assertEqual(next(item["offsets"] for item in bare if item["layer_zero_eligible"]), [0])
                self.assertEqual(next(item["offsets"] for item in bare if not item["layer_zero_eligible"]), [8])
                self.assertEqual(
                    sum(len(item["offsets"]) for entries in groups.values() for item in entries),
                    3 * len(text) - 3,
                )

    def test_newline_and_lowercase_do_not_imply_a_terminus_seed(self):
        for text in ("a test", "\nA test", "\nI test"):
            with self.subTest(text=text):
                groups = extract(text)
                self.assertFalse(any(item["layer_zero_eligible"] for items in groups.values() for item in items))

    def test_development_metadata_does_not_change_foundation_admission(self):
        groups = extract("t am I ")
        unknown = next(item for item in groups["maybe"] if item["text"] == "am")
        self.assertFalse(unknown["layer_zero_eligible"])
        self.assertEqual(unknown["development"][0]["state"], "developing")
        self.assertFalse(unknown["development"][0]["complete"])
        self.assertEqual(unknown["development"][0]["offsets"], [2])
        dead = next(item for item in groups["dead"] if item["text"] == "t ")
        self.assertEqual(dead["development"][0]["state"], "excluded")

    def test_windows_keep_exact_text_offsets_and_separate_dead(self):
        groups = extract(" I  i iii")
        by_status = {
            status: {item["text"]: item["offsets"] for item in items}
            for status, items in groups.items()
        }
        self.assertEqual(by_status["alive"][" I "], [0])
        self.assertEqual(by_status["dead"][" i "], [3])
        self.assertEqual(by_status["dead"]["iii"], [6])
        self.assertNotIn(" i ", by_status["maybe"])
        self.assertNotIn("iii", by_status["maybe"])
        self.assertEqual(by_status["maybe"]["i"], [4, 6, 7, 8])
        self.assertEqual(sum(len(offsets) for group in by_status.values() for offsets in group.values()), 24)
        self.assertTrue(all(item["layer_zero_eligible"] for item in groups["alive"]))
        self.assertTrue(all(not item["layer_zero_eligible"] for item in groups["maybe"]))
        self.assertTrue(all(not item["layer_zero_eligible"] for item in groups["dead"]))

    def test_extraction_preserves_unicode_and_newlines(self):
        groups = extract("é\nI")
        unknown = {item["text"]: item["offsets"] for item in groups["maybe"]}
        self.assertEqual(unknown["é\n"], [0])
        self.assertEqual(unknown["\nI"], [1])
        self.assertEqual(unknown["é\nI"], [0])

    def test_persistent_queue_tracks_sources_and_repeat_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            queue = root / "review.json"
            source.write_bytes(b" I  i iii\r\n")
            first = extract_file(source, queue)
            self.assertEqual(source.read_bytes(), b" I  i iii\r\n")
            self.assertEqual(extract_file(source, queue), first)
            state = json.loads(queue.read_text(encoding="utf-8"))
            self.assertEqual(len(state["sources"]), 1)
            record = state["sources"][0]
            self.assertEqual(record["sha256"], first["sha256"])
            self.assertEqual(record["characters"], 11)
            self.assertEqual(first["layer_zero_eligible_occurrences"], 1)
            self.assertTrue(all(not entry["layer_zero_eligible"] for entry in record["review"]))
            self.assertIn("iii", [entry["text"] for entry in record["dead"]])
            self.assertNotIn("iii", [entry["text"] for entry in record["review"]])
            self.assertIn("\r\n", [entry["text"] for entry in record["review"]])
            source.write_text("changed", encoding="utf-8")
            extract_file(source, queue)
            state = json.loads(queue.read_text(encoding="utf-8"))
            self.assertEqual(len(state["sources"]), 2)

    def test_reextraction_preserves_old_ruleset_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            queue = root / "review.json"
            source.write_text(" I ", encoding="utf-8")
            extract_file(source, queue)
            state = json.loads(queue.read_text(encoding="utf-8"))
            state["sources"][0]["ruleset"] = "seed-life-v1"
            del state["sources"][0]["alive"][0]["layer_zero_eligible"]
            queue.write_text(json.dumps(state), encoding="utf-8")
            extract_file(source, queue)
            extract_file(source, queue)
            records = json.loads(queue.read_text(encoding="utf-8"))["sources"]
            self.assertEqual([record["ruleset"] for record in records], ["seed-life-v1", RULESET])
            self.assertNotIn("layer_zero_eligible", records[0]["alive"][0])
            self.assertTrue(records[1]["alive"][0]["layer_zero_eligible"])

    def test_errors_do_not_overwrite_queue_or_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            queue = root / "review.json"
            source.write_text("abc", encoding="utf-8")
            queue.write_text('{"version": 99}', encoding="utf-8")
            with self.assertRaises(ValueError):
                extract_file(source, queue)
            self.assertEqual(queue.read_text(encoding="utf-8"), '{"version": 99}')
            with self.assertRaises(ValueError):
                extract_file(source, source)
            self.assertEqual(source.read_text(encoding="utf-8"), "abc")
            for value in ("", "a" * (MAX_SOURCE_CHARACTERS + 1)):
                with self.assertRaises(ValueError):
                    extract(value)
            with self.assertRaises(TypeError):
                extract(None)


if __name__ == "__main__":
    unittest.main()
