import unittest

from layer_two import analyze, Attraction
from seed_life import boundary_text, exclusion_reason


class LayerTwoTests(unittest.TestCase):
    def test_generation_without_living_seed_is_deterministic_and_constrained(self):
        model = Attraction("How are pairs being captured looking now?")
        result = model.generate(80, 5)
        self.assertTrue(result["text"])
        self.assertLessEqual(len(result["text"]), 80)
        self.assertEqual(result, model.generate(80, 5))
        self.assertIsNone(exclusion_reason(boundary_text(result["text"], True)))
        self.assertIn("outside Layer 0", result["policy"])

    def test_blending_retains_shorter_context_and_terminal_probability(self):
        model = Attraction("ab ac a")
        probabilities = model.distribution("a")
        self.assertAlmostEqual(sum(probabilities.values()), 1)
        self.assertGreater(probabilities[None], 0)
        self.assertGreater(probabilities["b"], 0)
        self.assertGreater(probabilities["c"], 0)
        self.assertGreater(probabilities[" "], 0)

    def test_old_transitions_are_reweighted_and_all_layer_one_rules_blocked(self):
        from original_generator import START, ORDER
        begin = START * ORDER
        for source, prefix, forbidden in [
            ("t t t ", "t", " "), ("I.I.I.", "I", "."), ("xxx", "xx", "x"),
            (" i i ", " i", " "),
        ]:
            transitions = {}
            for offset, char in enumerate(prefix):
                transitions[(begin + prefix[:offset])[-ORDER:]] = {char: 1}
            transitions[(begin + prefix)[-ORDER:]] = {forbidden: 1}
            output = Attraction(source).generate(20, 1, transitions)
            self.assertEqual(output["text"], prefix)
            self.assertIn("excluded", output["reason"])
            self.assertTrue(output["trace"][-1]["rejected"])
        model = Attraction("ab ab ab")
        transitions = {begin: {"a": 1, "z": 1}}
        result = model.generate(1, 1, transitions)
        weights = {item["next"]: item["weight"] for item in result["trace"][0]["candidates"]}
        self.assertGreater(weights["a"], weights["z"])
        self.assertEqual(model.metrics("")["prompt_attraction_nll"], None)

    def test_equal_output_is_allowed_and_stop_is_explicit(self):
        from original_generator import START, ORDER, END
        begin = START * ORDER
        output = Attraction("ab ab").generate(
            20, 7, {begin: {"a": 1}, (begin + "a")[-ORDER:]: {"b": 1},
                    (begin + "ab")[-ORDER:]: {END: 1}},
        )
        self.assertEqual(output["text"], "ab")
        self.assertIn("end selected", output["reason"])
        self.assertEqual(output["trace"][-1]["selected"], None)
        missing = Attraction("ab").generate(20, 7, {})
        self.assertEqual(missing["text"], "")
        self.assertIn("no further transition", missing["reason"])

    def test_exact_windows_and_boundaries(self):
        text = " am amount "
        report = analyze(text)
        rows = {row["text"]: row for row in report["patterns"]}
        self.assertEqual(rows[" am "]["offsets"], [0])
        self.assertEqual(rows["am"]["count"], 2)
        self.assertEqual(rows["am "]["likelihood"], 0.5)
        self.assertEqual(rows["amo"]["likelihood"], 0.5)
        self.assertEqual(rows[" am "]["source_start_offsets"], [0])
        self.assertEqual(sum(row["count"] for row in rows.values()), sum(len(text) - n + 1 for n in (2, 3, 4)))

    def test_always_is_directional_and_not_fooled_by_source_end(self):
        pairs = {row["text"]: row for row in analyze("ab ab a")["pairs"]}
        self.assertFalse(pairs["ab"]["always_followed_in_source"])
        self.assertTrue(pairs["ab"]["always_preceded_in_source"])
        self.assertAlmostEqual(pairs["ab"]["forward_likelihood"], 2 / 3)
        rows = {row["text"]: row for row in analyze("ab ab a")["patterns"]}
        self.assertEqual(rows["ab"]["source_end_opportunities"], 1)
        pair = next(row for row in analyze("ab ab")["pairs"] if row["text"] == "ab")
        self.assertTrue(pair["mutually_always_in_source"])

    def test_support_bounds_and_literal_characters(self):
        single = next(row for row in analyze("ab")["patterns"] if row["text"] == "ab")
        repeated = next(row for row in analyze("ab ab ab")["patterns"] if row["text"] == "ab")
        self.assertTrue(single["observed_always"])
        self.assertLess(single["wilson_lower_95"], repeated["wilson_lower_95"])
        report = analyze("A a\r\n")
        self.assertIn("A ", [row["text"] for row in report["patterns"]])
        self.assertIn("\r\n", [row["text"] for row in report["patterns"]])
        self.assertEqual(analyze("x")["patterns"], [])
        for bad in ("", "x" * 2001, None):
            with self.assertRaises(ValueError):
                analyze(bad)
