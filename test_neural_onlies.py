import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import neural_onlies as onlies
from neural_experiment import NeuralModel
import test_neural_efficiency


class OnlyStateTests(unittest.TestCase):
    stage = test_neural_efficiency.NeuralEfficiencyTests.stage

    def test_only_states_are_earned_then_mutate_without_losing_the_path(self):
        states = onlies.OnlyStates("qu qu qu qa", max_context=1, backward=False)
        state = states.forward_states["q"]
        self.assertEqual(state["path"], "u")
        self.assertEqual(set(state["variants"]), {"a"})
        self.assertEqual(state["support"], 4)
        mixed = onlies.OnlyStates("ab ac ab ab", max_context=1, backward=False)
        self.assertNotIn("a", mixed.forward_states)  # two outcomes before support 3: never an only

    def test_multipliers_guide_but_never_forbid(self):
        source = "qu qu qu "
        model = NeuralModel(sorted(set(source + "a")))
        states = onlies.OnlyStates(source, max_context=2)
        weights = states.multipliers("q", model, 0.2)
        self.assertEqual(weights[model.tokens["u"]], 1.0)
        self.assertEqual(weights[model.tokens["a"]], 0.2)
        self.assertTrue(all(weight > 0 for weight in weights))
        # Backward: "u" has only ever been entered from "q".
        self.assertEqual(states.backward_states["u"]["path"], "q")
        self.assertLess(states.multipliers(" ", model, 0.2)[model.tokens["u"]], 1.0)
        with self.assertRaises(ValueError):
            onlies.OnlyStates("")

    def test_hold_rates_distinguish_path_variant_and_new(self):
        states = onlies.OnlyStates("qu qu qu qa", max_context=1, backward=False)
        rates = onlies.hold_rates(states, "qu qa qi")["forward"]["2"]
        # q->u, u->space twice, space->q: four path holds; q->a is a known variant; q->i is new.
        self.assertEqual((rates["path"], rates["variant"], rates["new"]), (4, 1, 1))

    def test_run_persists_ledger_and_never_blocks_targets(self):
        with tempfile.TemporaryDirectory() as directory, patch.multiple(onlies, FRACTIONS=(0.5, 1.0)):
            root = Path(directory)
            self.stage(root, held="A seed can grow. I am a plant.", train="I am a seed. A seed can grow. A plant can grow.")
            with self.assertRaises(ValueError):
                onlies.run_onlies(root)
            report = onlies.run_onlies(root, acknowledge_review=True)
            self.assertEqual(report["status"], "completed")
            expected = 2 * (3 + 3 * len(onlies.PENALTIES))
            self.assertEqual(len(report["results"]), expected)
            self.assertTrue(all(row["coverage"] == 1 for row in report["results"]))
            ledger = json.loads((root / ".data/neural-onlies" / report["run_id"] / "ledger.json").read_text())
            self.assertIn("unconfirmed", ledger["only_states"]["state"])
            self.assertTrue((root / ".data/neural-experiments/latest-onlies.json").exists())
            self.assertFalse((root / ".data/seed-development.json").exists())


if __name__ == "__main__":
    unittest.main()
