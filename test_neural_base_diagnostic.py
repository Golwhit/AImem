import json
from pathlib import Path
import random
import unittest

from neural_base_comparison import WordBase
from neural_base_diagnostic import ambiguity, measure, train_trial, word_probes


class BaseDiagnosticTests(unittest.TestCase):
    def test_exact_input_bound_includes_masks_and_target_frequencies(self):
        examples = [([0, 0, 0, 0], 2, [0, 0, 1, 1])] * 3
        examples += [([0, 0, 0, 0], 3, [0, 0, 1, 1])]
        examples += [([0, 0, 0, 0], 3, [0, 0, 0, 1])]
        result = ambiguity(examples)
        self.assertEqual(result["distinct_observable_inputs"], 2)
        self.assertEqual(result["unavoidable_token_errors"], 1)
        self.assertEqual(result["deterministic_token_accuracy_ceiling"], .8)
        self.assertGreater(result["minimum_mean_token_cross_entropy"], 0)
        with self.assertRaises(ValueError):
            ambiguity([])

    def test_short_trial_replays_exactly_and_keeps_input_states_unchanged(self):
        first, second = "cat sat. cat sat. ", "cat ran. cat ran. "
        model = WordBase(first + second, 17)
        initial = json.loads(json.dumps(model.snapshot()))
        trial, trained = train_trial(model, first, second, seed=17, rate=.03,
                                     checkpoints=(5, 10), second_updates=4)
        self.assertTrue(trial["restoration_verified"])
        replay = WordBase.restore(json.loads(json.dumps(initial)))
        rng = random.Random(19)
        for text, count in ((first, 10), (second, 4)):
            examples = replay.examples(text)
            for _ in range(count):
                context, target, prior = examples[rng.randrange(len(examples))]
                replay.model.update(context, target, learning_rate=.03, output_prior=prior)
        self.assertEqual(replay.snapshot(), trained.snapshot())
        self.assertEqual(measure(replay, replay.examples(first), word_probes(first)),
                         trial["after_second_first"])
        self.assertEqual(initial["weights"], WordBase.restore(initial).snapshot()["weights"])
        self.assertNotEqual(initial["weights"], trained.snapshot()["weights"])

    def test_context_alias_for_unseen_names_is_observable(self):
        model = WordBase("red cup blue cup. ", 17)
        self.assertEqual(model.context("Mira has the cup. Who has it? "),
                         model.context("Jon has the cup. Who has it? "))
        self.assertEqual(model.context("Mira "), model.context("Jon "))
        self.assertNotIn("identity:mira", model.model.tokens)
        with self.assertRaises(ValueError):
            train_trial(model, "red ", "blue ", seed=17, rate=.03, checkpoints=(10, 5))

    def test_curriculum_remains_a_disjoint_review_only_draft(self):
        draft = json.loads((Path(__file__).parent / "fixtures/neural-conversation-draft.json").read_text())
        self.assertEqual(draft["status"], "draft_pending_user_review_not_approved_for_training")
        identifiers = [row["id"] for split in ("train", "test") for row in draft[split]]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        training_prompts = {json.dumps(row["turns"], sort_keys=True) for row in draft["train"]}
        self.assertTrue(all(json.dumps(row["turns"], sort_keys=True) not in training_prompts
                            for row in draft["test"]))
        pair = [row for row in draft["test"] if row.get("pair") == "changed-color"]
        model = WordBase("red blue box. What color is the box? ", 17)
        prompts = ["".join(turn["content"] + "\n" for turn in row["turns"]) for row in pair]
        self.assertNotEqual(pair[0]["answer"], pair[1]["answer"])
        self.assertEqual(model.context(prompts[0]), model.context(prompts[1]))
        self.assertEqual(model.prior(prompts[0]), model.prior(prompts[1]))


if __name__ == "__main__":
    unittest.main()
