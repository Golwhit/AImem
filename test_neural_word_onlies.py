import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import neural_word_onlies as words
from neural_experiment import NeuralModel
import test_neural_efficiency


class WordOnlyTests(unittest.TestCase):
    stage = test_neural_efficiency.NeuralEfficiencyTests.stage

    def test_forward_and_backward_states_mutate_without_losing_paths(self):
        source = "Holy Spirit. Holy Spirit. the Spirit. Holy Ghost."
        states = words.WordOnlies(source, min_support=2)
        self.assertEqual(states.forward_states["Holy"]["path"], "Spirit")
        self.assertEqual(set(states.forward_states["Holy"]["variants"]), {"Ghost"})
        self.assertEqual(states.backward_states["Spirit"]["path"], "Holy")
        self.assertEqual(set(states.backward_states["Spirit"]["variants"]), {"the"})
        with self.assertRaises(ValueError):
            words.WordOnlies("one")

    def test_forward_guides_the_partial_word_and_its_end(self):
        source = "Holy Spirit. Holy Spirit. go home."
        model = NeuralModel(sorted(set(source + "x")))
        states = words.WordOnlies(source, min_support=2, backward=False)
        self.assertEqual(states.split("the Holy Sp"), ("Holy", "Sp"))
        after = states.multipliers("Holy Sp", model, 0.2)
        self.assertEqual(after[model.tokens["i"]], 1.0)
        self.assertEqual(after[model.tokens["x"]], 0.2)
        complete = states.multipliers("Holy Spirit", model, 0.2)
        self.assertEqual(complete[model.tokens["."]], 1.0)
        self.assertEqual(complete[model.tokens["x"]], 0.2)
        self.assertEqual(states.multipliers("Holy Sx", model, 0.2), [1.0] * model.size)  # off path: no guidance
        self.assertTrue(all(value > 0 for value in after))

    def test_backward_penalizes_words_that_need_a_different_predecessor(self):
        source = "Holy Spirit. Holy Spirit. the sun. the sun."
        model = NeuralModel(sorted(set(source)))
        states = words.WordOnlies(source, min_support=2, forward=False)
        weights = states.multipliers("the ", model, 0.2)
        self.assertEqual(weights[model.tokens["S"]], 0.2)  # Spirit needs "Holy"
        self.assertEqual(weights[model.tokens["s"]], 1.0)  # sun follows "the"

    def test_hold_rates_and_run_persistence(self):
        states = words.WordOnlies("Holy Spirit. Holy Spirit.", min_support=2)
        rates = words.hold_rates(states, "Holy Spirit. Holy Ghost.")["forward"]
        self.assertEqual((rates["path"], rates["new"]), (1, 1))
        with tempfile.TemporaryDirectory() as directory, patch.multiple(words, FRACTIONS=(1.0,)):
            root = Path(directory)
            self.stage(root, held="A seed can grow. I am a seed.", train="I am a seed. A seed can grow. I am a seed.")
            with self.assertRaises(ValueError):
                words.run(root)
            report = words.run(root, acknowledge_review=True)
            self.assertEqual(report["status"], "completed")
            scored = [row for row in report["results"] if row.get("validation_loss") is not None]
            self.assertTrue(scored and all(row["coverage"] == 1 for row in scored))
            self.assertTrue((root / ".data/neural-onlies" / report["run_id"] / "word-ledger.json").exists())
            self.assertTrue((root / ".data/neural-experiments/latest-word-onlies.json").exists())
            self.assertFalse((root / ".data/seed-development.json").exists())


if __name__ == "__main__":
    unittest.main()
