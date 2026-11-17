import copy
import random
import unittest

from fresh_pair_comparison import train_and_replay
from neural_conversation_trial import answer_examples
from neural_corpus_training import advance, corpus_examples
from neural_plasticity import PlasticityContext, initialize


class FreshPairComparisonTests(unittest.TestCase):
    def test_fresh_training_replays_and_keeps_parent_unchanged(self):
        source = "user: Hi\nassistant: Yes.\ntext: cat.\n"
        parent = PlasticityContext(source, 17)
        parent_before = copy.deepcopy(parent.snapshot())
        fresh = initialize(parent, "fresh_scaled", seed=17)
        self.assertNotEqual(fresh.snapshot()["weights"], parent_before["weights"])
        teaching = [{"turns": [{"role": "user", "content": "Hi"}], "answer": "Yes."}]
        initial, state, conversations, corpus, cpu = train_and_replay(
            fresh, teaching, [("text: ", "cat.")], updates=20, seed=3, rate=0.01)
        self.assertNotEqual(initial["weights"], state["weights"])
        self.assertEqual(fresh.snapshot(), state)
        self.assertEqual(parent.snapshot(), parent_before)
        self.assertTrue(conversations)
        self.assertTrue(corpus)
        self.assertGreaterEqual(cpu, 0)

    def test_invalid_budget_is_rejected(self):
        parent = PlasticityContext("abc\n", 17)
        fresh = initialize(parent, "fresh_scaled", seed=17)
        with self.assertRaises(ValueError):
            train_and_replay(fresh, [], [], updates=0, seed=3, rate=0.01)

    def test_continuation_reuses_checkpoint_and_exact_random_stream(self):
        source = "user: Hi\nassistant: Yes.\ntext: cat.\n"
        parent = PlasticityContext(source, 17)
        first = initialize(parent, "fresh_scaled", seed=17)
        whole = PlasticityContext.restore(copy.deepcopy(first.snapshot()))
        teaching = [{"turns": [{"role": "user", "content": "Hi"}], "answer": "Yes."}]
        records = [("text: ", "cat.")]
        advance(whole, answer_examples(whole, teaching), corpus_examples(whole, records),
                random.Random(3), 40, 0.01)
        _, checkpoint, _, _, _ = train_and_replay(first, teaching, records, updates=20, seed=3, rate=0.01)
        continued = PlasticityContext.restore(copy.deepcopy(checkpoint))
        resumed_initial, resumed, _, _, _ = train_and_replay(
            continued, teaching, records, updates=20, seed=3, rate=0.01, offset=20)
        self.assertEqual(resumed_initial, checkpoint)
        self.assertEqual(resumed, whole.snapshot())


if __name__ == "__main__":
    unittest.main()
