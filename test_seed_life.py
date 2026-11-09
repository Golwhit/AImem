import unittest

from seed_life import ALIVE_SEEDS, CONSONANTS, LAYER_ZERO_SEEDS, TERMINUS, assess, boundary_text, exclusion_reason


class SeedLifeTests(unittest.TestCase):
    def test_terminus_single_consonant_space_is_dead_in_either_case(self):
        for consonant in CONSONANTS:
            with self.subTest(consonant=consonant):
                result = assess(TERMINUS + consonant + " remaining text")
                self.assertEqual(result.status, "dead")
                self.assertIn("single consonant", result.reason)
        for text in ("</>t.", "</>t", "</>the ", "</>a ", "</>I ", " t ", "t "):
            with self.subTest(text=text):
                self.assertIsNone(exclusion_reason(text))
        self.assertEqual(boundary_text("t ", True), "</>t ")
        self.assertEqual(boundary_text("</>t ", True), "</>t ")
        self.assertEqual(boundary_text("t ", False), "t ")
        self.assertEqual(boundary_text(" I ", True), " I ")

    def test_exact_initial_possibilities(self):
        self.assertEqual(LAYER_ZERO_SEEDS, (
            "</>A", "</>I", "</>I ", "</>A ", " a ", " A ", " I ",
        ))

    def test_confirmed_living_forms(self):
        for text in (TERMINUS + "A ", TERMINUS + "I ", " a ", " A ", " I "):
            with self.subTest(text=text):
                self.assertIn(text, ALIVE_SEEDS)
                self.assertEqual(assess(text).status, "alive")

    def test_bare_terminus_forms_need_more_context(self):
        for text in (TERMINUS + "A", TERMINUS + "I", "", "A ", " i"):
            with self.subTest(text=text):
                self.assertEqual(assess(text).status, "maybe")

    def test_lowercase_i_surrounded_by_spaces_is_dead(self):
        for text in (" i ", " a i A "):
            with self.subTest(text=text):
                result = assess(text)
                self.assertEqual(result.status, "dead")
                self.assertIn("Lowercase i", result.reason)
        self.assertEqual(assess(" I ").status, "alive")

    def test_three_identical_characters_are_dead(self):
        for text in ("iii", "AAA", "   ", "a!!!b", "111", "aaaa"):
            with self.subTest(text=text):
                result = assess(text)
                self.assertEqual(result.status, "dead")
                self.assertIn("Three identical", result.reason)
        self.assertEqual(assess("iIi").status, "maybe")
        self.assertEqual(assess("ii").status, "maybe")

    def test_terminus_i_period_cannot_be_rescued_by_more_text(self):
        for text in ("</>I.", "</>I. am here", "</>I. "):
            with self.subTest(text=text):
                result = assess(text)
                self.assertEqual(result.status, "dead")
                self.assertIn("period", result.reason)
        for text in (" I.", "or I.", " A, ", "area.", "area. "):
            with self.subTest(text=text):
                self.assertEqual(assess(text).status, "maybe")

    def test_unknown_text_is_not_automatically_dead_or_alive(self):
        for text in ("hello", " A  ", "</>A word", " i", "i "):
            with self.subTest(text=text):
                self.assertEqual(assess(text).status, "maybe")

    def test_non_text_input_is_rejected(self):
        with self.assertRaises(TypeError):
            assess(None)
        with self.assertRaises(TypeError):
            exclusion_reason(None)

    def test_only_exact_frozen_layer_zero_forms_are_eligible(self):
        for text in LAYER_ZERO_SEEDS:
            with self.subTest(text=text):
                result = assess(text)
                self.assertTrue(result.layer_zero_eligible)
                self.assertIsNone(exclusion_reason(text))
                self.assertEqual(result.status, "alive" if text in ALIVE_SEEDS else "maybe")
        for text in ("", "I", " A", " A  ", " x ", "hello", "</>A word", " i ", "iii", "</>I."):
            with self.subTest(text=text):
                result = assess(text)
                self.assertFalse(result.layer_zero_eligible)
                self.assertNotEqual(result.status, "alive")

    def test_layer_one_only_excludes_and_does_not_confer_life(self):
        for text in ("hello", " x ", " I ", "</>I"):
            with self.subTest(text=text):
                self.assertIsNone(exclusion_reason(text))
        for text in (" i ", "iii", "</>I."):
            with self.subTest(text=text):
                self.assertIsNotNone(exclusion_reason(text))
        self.assertEqual(assess("hello").status, "maybe")
        self.assertFalse(assess("hello").layer_zero_eligible)


if __name__ == "__main__":
    unittest.main()
