import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model_lab import compare, overlay_captures
from original_generator import CharacterGenerator


class ModelLabTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.generator = CharacterGenerator(self.root)

    def test_comparison_preserves_checkpoint_and_uses_same_actual_display_length(self):
        self.generator.add_document(text="I keep a note. I keep a note.", acknowledge_review=True)
        checkpoint = self.generator.path.read_bytes()
        result = compare(self.generator, "I keep a note.", 40)
        self.assertEqual(self.generator.path.read_bytes(), checkpoint)
        self.assertEqual(result["generation"], 1)
        self.assertEqual(result["timing"], "before_prompt_training")
        outputs = result["generation_comparison"]
        common = result["common_display_characters"]
        self.assertGreater(common, 0)
        self.assertTrue(all(len(item["display_text"]) == common for item in outputs.values()))
        self.assertTrue(all(len(item["text"]) <= 40 for item in outputs.values()))
        self.assertTrue(all(item["characters"] + item["shortfall"] == 40 for item in outputs.values()))
        self.assertEqual(outputs["combined"]["protected_prefix"], "I ")
        self.assertTrue(outputs["combined"]["text"].startswith("I "))
        self.assertEqual(outputs["combined"]["context_mode"], "source_start")
        self.assertEqual(outputs["old"]["text"], result["old_response"]["text"])
        self.assertEqual(compare(self.generator, "I keep a note.", 40), result)
        self.assertLess(result["seed"], 2**53)
        self.assertFalse((self.root / ".data" / "response-scorer.json").exists())

    def test_extraction_modes_keep_unknowns_and_do_not_confuse_exclusion_with_eligibility(self):
        result = compare(self.generator, "I keep a note. i iii", 40)
        extraction = result["extraction"]
        for key in ("baseline_windows", "layer_zero", "layer_one", "combined"):
            self.assertEqual(
                sum(len(item["offsets"]) for item in extraction[key]),
                3 * len(result["prompt"]) - 3,
            )
        self.assertTrue(any(item["text"] == "iii" for item in extraction["baseline_windows"]))
        self.assertTrue(any(item["text"] == "iii" and item["status"] == "excluded" for item in extraction["layer_one"]))
        self.assertTrue(any(item["text"] == " i " and item["status"] == "dead" for item in extraction["combined"]))
        self.assertTrue(any(item["text"] == "I " and item["offsets"] == [0] and item["status"] == "eligible" for item in extraction["layer_zero"]))
        self.assertTrue(any(item["text"] == "kee" and item["status"] == "not_excluded" for item in extraction["layer_one"]))
        self.assertTrue(any(item["text"] == "kee" and not item["layer_zero_eligible"] for item in extraction["combined"]))
        self.assertTrue(extraction["old_model_transitions"])

    def test_no_capture_is_reported_instead_of_falling_back(self):
        result = compare(self.generator, "hello there", 40)
        self.assertEqual(result["common_display_characters"], 0)
        for key in ("layers", "combined"):
            self.assertEqual(result["generation_comparison"][key]["text"], "")
            self.assertIn("No living seed", result["generation_comparison"][key]["reason"])

    def test_layer_two_is_read_only_and_keeps_baseline_generation(self):
        self.generator.add_document(text="I am a note. I am a note.", acknowledge_review=True)
        checkpoint = self.generator.path.read_bytes()
        result = compare(self.generator, " am amount ", 40)
        with patch("layer_two.Attraction.generate", return_value={"text": "", "reason": "test disabled"}):
            without_evidence = compare(self.generator, " am amount ", 40)
        for key in ("old", "layers", "combined", "old_first"):
            self.assertEqual(result["generation_comparison"][key]["text"], without_evidence["generation_comparison"][key]["text"])
        self.assertEqual(result["layer_two"]["source_sha256"], result["prompt_sha256"])
        self.assertEqual(result["version"], 7)
        self.assertEqual(checkpoint, self.generator.path.read_bytes())
        self.assertFalse((self.root / ".data" / "layer-two.json").exists())

    def test_admission_and_revocation_affect_future_lab_only(self):
        from seed_development import DevelopmentRegistry

        result = compare(self.generator, "am amount", 20)
        record_id = "a" * 64
        directory = self.root / ".data" / "model-lab"
        directory.mkdir(parents=True)
        record = directory / f"{record_id}.json"
        record.write_text(json.dumps(result))
        payload = {"record_id": record_id, "offset": 0, "size": 2, "action": "admit",
                   "rationale": "Fixture review", "acknowledge_lab_only": True}
        registry = DevelopmentRegistry(self.root)
        registry.decide(payload)
        admitted = compare(self.generator, "unrelated query", 20)
        self.assertEqual(admitted["generation_comparison"]["layers"]["text"], "am")
        self.assertFalse(admitted["generation_comparison"]["layers"]["attraction_metrics"]["layer_one_violation"])
        self.assertFalse(result["development"]["registry"]["events"])
        self.assertFalse(json.loads(record.read_text())["development"]["registry"]["events"])
        registry.decide({**payload, "action": "revoke"})
        self.assertEqual(compare(self.generator, "unrelated query", 20)["generation_comparison"]["layers"]["text"], "")
        self.assertFalse(self.generator.path.exists())

    def test_missing_model_context_retains_prefix_without_unrelated_continuation(self):
        self.generator.add_document(text="Hello from training.", acknowledge_review=True)
        result = compare(self.generator, "I keep a note.", 40)
        self.assertEqual(result["generation_comparison"]["combined"]["text"], "I ")
        self.assertIn("no context", result["generation_comparison"]["combined"]["reason"])
        self.assertEqual(result["common_display_characters"], 2)

    def test_interior_capture_uses_explicit_short_context_handoff(self):
        self.generator.add_document(text="the I follow is a marker.", acknowledge_review=True)
        result = compare(self.generator, "test I follow", 40)
        output = result["generation_comparison"]["combined"]
        self.assertEqual(output["context_mode"], "suffix_backoff")
        self.assertEqual(output["protected_prefix"], " I ")
        self.assertTrue(output["text"].startswith(" I "))
        self.assertGreater(len(output["text"]), 3)

    def test_baseline_contexts_match_existing_normalization_and_training_counts(self):
        prompt = "I  keep\r\na note."
        result = compare(self.generator, prompt, 40)
        expected = self.generator._rebuild([{"text": prompt}], 0)["transitions"]
        actual = {}
        for row in result["extraction"]["old_model_transitions"]:
            actual.setdefault(row["context"], {})[row["next"]] = row["count"]
        self.assertEqual(actual, expected)
        self.assertEqual(result["prompt"], prompt)
        json.dumps(result, allow_nan=False)

    def test_budget_and_prompt_validation(self):
        for budget in (True, 0, 501, 3.5):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                compare(self.generator, "I keep a note.", budget)
        for prompt in ("", " ", "a" * 2001):
            with self.assertRaises(ValueError):
                compare(self.generator, prompt)

    def test_old_first_changes_only_aligned_letters_with_capture_provenance(self):
        with patch.object(self.generator, "generate", return_value={"text": "i test and i go.", "ready": True}):
            result = compare(self.generator, "I test and I go.", 40)
        outputs = result["generation_comparison"]
        output = outputs["old_first"]
        self.assertEqual(len(outputs), 6)
        self.assertEqual(output["base_text"], outputs["old"]["text"])
        self.assertEqual(output["text"], "I test and I go.")
        self.assertEqual(len(output["text"]), len(outputs["old"]["text"]))
        self.assertEqual([item["offset"] for item in output["replacements"]], [0, 11])
        self.assertEqual([item["source_offset"] for item in output["replacements"]], [0, 10])
        self.assertEqual(len(output["display_text"]), result["common_display_characters"])

    def test_old_first_does_not_append_or_force_unmatched_captures(self):
        for old_text, prompt, expected_reason in (
            ("x" * 20, "I test", "No matching"),
            ("old.", "hello", "No living seed"),
            ("", "I test", "no output"),
            ("x" * 19, "I test", "No matching"),
        ):
            with self.subTest(old_text=old_text, prompt=prompt):
                with patch.object(self.generator, "generate", return_value={"text": old_text}):
                    output = compare(self.generator, prompt, 20)["generation_comparison"]["old_first"]
                self.assertEqual(output["text"], old_text)
                self.assertEqual(output["replacements"], [])
                self.assertIn(expected_reason, output["reason"])

    def test_overlay_requires_the_same_letter_and_exact_boundary(self):
        text = "a i aim i. i\t i "
        output, changes = overlay_captures(text, [(0, "I "), (4, " I ")])
        self.assertEqual(output, "a I aim i. i\t I ")
        self.assertEqual([change["offset"] for change in changes], [2, 14])
        self.assertEqual(overlay_captures("a test", [(0, "A ")])[0], "A test")
        self.assertEqual(overlay_captures("a test", [(4, " A ")])[0], "a test")
        self.assertEqual(overlay_captures("word i x", [(0, "I ")])[0], "word i x")
        self.assertEqual(overlay_captures("I test", [(0, "I ")])[1], [])

    def test_dead_initial_consonant_is_repaired_only_with_start_capture(self):
        for letter in ("t", "T", "b", "Y"):
            for block in ("A ", "I "):
                with self.subTest(letter=letter, block=block):
                    output, changes = overlay_captures(letter + " next", [(0, block)])
                    self.assertEqual(output, block + "next")
                    self.assertEqual(len(output), len(letter + " next"))
                    self.assertEqual(changes[0]["before"], letter)
                    self.assertEqual(changes[0]["capture"], block)
                    self.assertIn("consonant", changes[0]["reason"])
        for text in ("t. next", "the next", " t next"):
            self.assertEqual(overlay_captures(text, [(0, "I ")])[0], text)
        self.assertEqual(overlay_captures("t next", [(4, " I ")])[0], "t next")
        self.assertEqual(overlay_captures("t next", [])[0], "t next")

    def test_lab_reports_unresolved_dead_opening_without_start_capture(self):
        with patch.object(self.generator, "generate", return_value={"text": "t next"}):
            result = compare(self.generator, "test I follow", 40)
        output = result["generation_comparison"]["old_first"]
        self.assertEqual(output["text"], "t next")
        self.assertIn("Unresolved", output["reason"])
        self.assertEqual(output["replacements"], [])
        with patch.object(self.generator, "generate", return_value={"text": "t next"}):
            repaired = compare(self.generator, "I follow", 40)["generation_comparison"]["old_first"]
        self.assertEqual(repaired["text"], "I next")
        self.assertEqual(repaired["reason"], "")

    def test_extraction_layer_one_and_combined_agree_on_dead_opening(self):
        result = compare(self.generator, "t next t next", 40)
        extraction = result["extraction"]
        for key, status in (("layer_one", "excluded"), ("combined", "dead")):
            self.assertTrue(any(
                item["text"] == "t " and item["offsets"] == [0] and item["status"] == status
                for item in extraction[key]
            ))

if __name__ == "__main__":
    unittest.main()
