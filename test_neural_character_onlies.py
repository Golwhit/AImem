import unittest
from unittest.mock import patch
from pathlib import Path

from neural_character_onlies import CharacterOnlies
from neural_experiment import NeuralModel


class CharacterOnlyTests(unittest.TestCase):
    def setUp(self):
        self.rules = CharacterOnlies({"that", "with", "a", "I"}, {"tha": "that"}, {})
        self.model = NeuralModel(sorted(set("that with I a.")))

    def test_absorption_skips_only_interior_and_keeps_original_context(self):
        text = "that with unknown I a."
        original = self.model.examples(text)
        skip = self.rules.offsets(text)
        self.assertEqual(skip, {1, 2, 3, 6, 7, 8})
        self.assertEqual(self.rules.examples(self.model, text),
                         [example for i, example in enumerate(original) if i not in skip])
        self.assertEqual(self.rules.offsets("That thatcher that2 éthat"), set())
        self.assertEqual(self.rules.offsets("a I"), set())

    def test_generation_supplies_suffix_without_forward_and_advances_context(self):
        contexts = []

        def forward(context):
            contexts.append(context[:])
            row = [0.0] * self.model.size
            row[self.model.tokens["a"]] = 100
            row[self.model.tokens[" "]] = 10
            return [], [], row

        with patch.object(self.model, "forward", side_effect=forward):
            result = self.rules.sample(self.model, "tha", length=2, greedy=True)
        self.assertEqual(result["text"], "t ")
        self.assertEqual(result["forced_characters"], 1)
        self.assertEqual(result["decisions"], 1)
        self.assertEqual(contexts, [[self.model.tokens[c] for c in "that"]])

    def test_ambiguous_prefix_and_unavailable_suffix_do_not_force(self):
        result = self.rules.sample(self.model, "th", length=1)
        self.assertEqual(result["forced_characters"], 0)
        other = CharacterOnlies({"that"}, {"tha": "thatz"}, {})
        self.assertEqual(other.sample(self.model, "tha", length=1)["forced_characters"], 0)

    def test_internal_apostrophe_identifies_suffix(self):
        rules = CharacterOnlies({"couldn't"}, {"couldn'": "couldn't"}, {})
        model = NeuralModel(sorted(set("couldn't ")))
        result = rules.sample(model, "couldn'", length=1, greedy=True)
        self.assertEqual(result["text"], "t")
        self.assertEqual(result["forced_characters"], 1)
        self.assertEqual(result["decisions"], 0)

    def test_vocabulary_parameters_and_context_are_unchanged(self):
        baseline = NeuralModel(self.model.characters)
        self.assertEqual(self.model.weights, baseline.weights)
        self.rules.examples(self.model, "that with")
        self.assertEqual(self.model.characters, baseline.characters)
        self.assertEqual(self.model.weights, baseline.weights)
        self.assertEqual(CharacterOnlies.restore(self.rules.snapshot()).prefixes, self.rules.prefixes)

    def test_build_never_treats_masked_uniqueness_as_prefix_uniqueness(self):
        bank = {"sha256": "fixture", "patterns": {
            "hidden1": {"th*t": "that", "couldn'*": "couldn't"},
        }}
        with patch("neural_character_onlies.load_bank", return_value=bank), \
                patch("neural_character_onlies.dictionary", return_value=({"that", "than", "could", "couldn't"}, "fixture")), \
                patch("neural_character_onlies.load", return_value=("that than couldn't", {})):
            rules = CharacterOnlies.build(Path("."))
        self.assertIn("that", rules.words)
        self.assertNotIn("th", rules.prefixes)
        self.assertNotIn("tha", rules.prefixes)
        self.assertEqual(rules.prefixes["couldn'"], "couldn't")


if __name__ == "__main__":
    unittest.main()
