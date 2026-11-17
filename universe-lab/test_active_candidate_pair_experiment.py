from collections import Counter
import copy
import unittest
from unittest.mock import patch

from active_candidate_pair_experiment import evaluate_pairs, mixture
from neural_base_comparison import WordBase, generate


class ActiveCandidatePairTests(unittest.TestCase):
    def test_equal_models_preserve_live_decoding(self):
        first = WordBase("a cat sun.\n", 17)
        second = WordBase.restore(copy.deepcopy(first.snapshot()))
        combined = mixture(first, second)
        for prompt in ("text: a", "text: ", "user: a\nassistant: "):
            self.assertEqual(generate(first, prompt, length=20, no_triple_repeat=True),
                             generate(combined, prompt, length=20, no_triple_repeat=True))

    def test_mixture_averages_masked_probabilities_without_changing_weights(self):
        first = WordBase("a cat sun.\n", 17)
        second = WordBase.restore(copy.deepcopy(first.snapshot()))
        second.model.weights["output_bias"][second.model.tokens["char:a"]] += 2
        before = copy.deepcopy(first.snapshot())
        combined = mixture(first, second)
        prefix = "text: "
        context = first.context(prefix)
        prior = first.prior(prefix, 20)
        left = first.model.forward(context, output_prior=prior)[2]
        right = second.model.forward(context, output_prior=prior)[2]
        actual = combined.model.forward(context, output_prior=prior)[2]
        for probability, a, b in zip(actual, left, right):
            self.assertAlmostEqual(probability, (a + b) / 2)
        self.assertEqual(first.snapshot(), before)

    def test_incompatible_models_are_rejected(self):
        first = WordBase("a cat sun.\n", 17)
        second = WordBase("different words!\n", 17)
        with self.assertRaises(ValueError):
            mixture(first, second)

    def test_pair_scoring_does_not_use_next_word_as_input(self):
        pairs = {"x": Counter({("a", "cat"): 2, ("a", "car"): 3})}
        with patch("active_candidate_pair_experiment.generate", return_value={"text": " cat\n"}):
            result = evaluate_pairs(object(), pairs)["combined"]
        self.assertEqual(result["pairs"], 5)
        self.assertEqual(result["second_initial_correct"], 5)
        self.assertEqual(result["both_words_correct"], 2)


if __name__ == "__main__":
    unittest.main()
