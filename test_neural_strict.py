import random
import unittest

from neural_counting import KneserNey
from neural_experiment import NeuralModel
from neural_patterns import FixedPatterns
from neural_strict import StrictUnits, UnitChooser, generate, light_rows


TEXT = "the Holy Spirit came. " * 6 + "the office was open. "


class StrictLayerTests(unittest.TestCase):
    def test_reading_claims_only_whole_confirmed_units(self):
        units = StrictUnits(TEXT, phrase_support=5)
        self.assertIn("the Holy Spirit came", units.phrases)
        result = units.read("the difficulty of the Holy Spirit came.")
        self.assertEqual(result["strict_wrong_claims"], 0)
        self.assertIn("difficulty", result["unknown_word_examples"])
        self.assertIsNone(units.recognize("difficulty", 0))
        self.assertIsNone(units.recognize("offices", 0))  # "office" is known but this word is longer
        self.assertEqual(units.recognize("xoffice", 1), None)  # not a word start
        self.assertEqual(units.recognize("the Holy Spirit came.", 0), "the Holy Spirit came")

    def test_decisions_count_units_once_and_unknown_letters_each(self):
        units = StrictUnits(TEXT, phrase_support=5)
        result = units.read("office zz")
        self.assertEqual(result["decisions"], 1 + 1 + 2)
        self.assertEqual(result["recognized_characters"], 6)

    def test_generation_only_spells_learned_words(self):
        units, chooser = StrictUnits(TEXT), UnitChooser(TEXT)
        sample = generate(units, chooser, "the", 3)
        self.assertEqual(sample["misspelled_words"], 0)
        self.assertNotIn("  ", sample["text"])
        self.assertLess(sample["decisions"], sample["characters"])
        self.assertTrue(set(chooser.choose("Holy", random.Random(1)) for _ in range(5)) <= set(chooser.vocabulary))

    def test_light_layer_adds_weight_but_never_forbids(self):
        patterns = FixedPatterns(TEXT, support=5, max_trigger=4)
        counts = KneserNey(TEXT)
        model = NeuralModel(sorted(set(TEXT)))
        rows, fired = light_rows(patterns, counts, model, "the office", None)
        self.assertGreater(sum(fired.values()), 0)
        for row in rows:
            self.assertAlmostEqual(sum(row), 1.0, places=9)
            self.assertTrue(all(value > 0 for value in row))


if __name__ == "__main__":
    unittest.main()
