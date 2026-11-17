import unittest
from types import SimpleNamespace

from pattern_meal import factors, first_mass, meal_counts, pairs


class PatternMealTests(unittest.TestCase):
    def test_pairs_only_use_previous_initial_and_word_start_prefix(self):
        self.assertEqual(list(pairs("One blue bird.")),
                         [("O", "b", "text: One "), ("b", "b", "text: One blue ")])

    def test_small_meal_shrinks_seen_families_and_leaves_unseen_unchanged(self):
        counts, overall = meal_counts(["A bird sings.", "A bird waits."])
        seen = factors("A", counts, overall)
        self.assertGreater(seen["b"], 1.0)
        self.assertLess(seen["s"], 1.0)
        self.assertTrue(all(value == 1.0 for value in factors("Z", counts, overall).values()))

    def test_scoring_aggregates_word_and_character_outputs_and_keeps_nonletters(self):
        model = SimpleNamespace(payloads={0: "b", 1: "bird", 2: "."})
        self.assertEqual(first_mass(model, [0.2, 0.3, 0.5]), {"b": 0.5, ".": 0.5})


if __name__ == "__main__":
    unittest.main()
