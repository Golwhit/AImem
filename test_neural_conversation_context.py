import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from neural_base_comparison import WordBase, generate
from neural_conversation_context import ARMS, CONTEXT, ConversationContext, build, lower_rate_screen, run
from neural_base_comparison import digest
from neural_conversation_trial import answer_examples, assess, prompt_for, train
from neural_experiment import EMBEDDING, HIDDEN
from neural_model_migration import CHARACTERS, WORDS, migrate, migrate_word_position, pair_ambiguity
from neural_plasticity import PlasticityContext, initialize


class ConversationContextTests(unittest.TestCase):
    def setUp(self):
        self.parent = WordBase("red blue cup box has the. user assistant: \n?", 17)
        self.rows = [
            {"id": "red", "skill": "color", "turns": [
                {"role": "user", "content": "Mira has a red box. What color is the box?"}], "answer": "red."},
            {"id": "blue", "skill": "color", "turns": [
                {"role": "user", "content": "Mira has a blue box. What color is the box?"}], "answer": "blue."},
        ]

    def test_long_features_preserve_fact_and_train_only_word_identity(self):
        short, _ = build(self.parent, self.rows, "short_identities")
        long, metadata = build(self.parent, self.rows, "long_identities")
        red, blue = [prompt_for(row) for row in self.rows]
        self.assertEqual(short.context(red), short.context(blue))
        self.assertNotEqual(long.context(red), long.context(blue))
        self.assertEqual(long.context(red)[-4:], short.context(red))
        self.assertEqual(len(long.context(red)), CONTEXT)
        self.assertIn("identity:mira", long.model.tokens)
        self.assertNotIn("identity:tavi", long.model.tokens)
        self.assertNotIn("emit:Mira", long.model.tokens)
        self.assertEqual(short.model.characters, long.model.characters)
        self.assertEqual(short.prior(red), long.prior(red))
        self.assertEqual(long.model.weights["hidden"][:(CONTEXT - 4) * EMBEDDING * HIDDEN],
                         [0.0] * ((CONTEXT - 4) * EMBEDDING * HIDDEN))
        for prompt in (red, blue, red + "r", red + "red", red + "red. "):
            p_short = short.model.forward(short.context(prompt), output_prior=short.prior(prompt))[2]
            p_long = long.model.forward(long.context(prompt), output_prior=long.prior(prompt))[2]
            self.assertEqual(p_short, p_long)
        self.assertEqual(short.model.weights["embedding"], long.model.weights["embedding"])
        self.assertGreater(metadata["parameters"], sum(map(len, short.model.weights.values())))

    def test_roundtrip_replay_and_new_context_weights_learn(self):
        for arm in ARMS:
            model, _ = build(self.parent, self.rows, arm)
            state = json.loads(json.dumps(model.snapshot()))
            points = train(model, self.rows, rate=.1, seed=17, checkpoints=(3, 10))
            restored = ConversationContext.restore(json.loads(json.dumps(model.snapshot())))
            self.assertEqual(assess(model, self.rows), assess(restored, self.rows))
            replay = ConversationContext.restore(state)
            rng = random.Random(17)
            examples = answer_examples(replay, self.rows)
            for _ in range(10):
                context, target, prior = examples[rng.randrange(len(examples))]
                replay.model.update(context, target, output_prior=prior, learning_rate=.1)
            self.assertEqual(replay.snapshot(), model.snapshot())
            self.assertEqual(points[-1]["training"], assess(replay, self.rows))
            if arm == "long_identities":
                self.assertTrue(any(model.model.weights["hidden"][:(CONTEXT - 4) * EMBEDDING * HIDDEN]))
            self.assertEqual(generate(model, prompt_for(self.rows[0]), length=32),
                             generate(restored, prompt_for(self.rows[0]), length=32))

    def test_reviewed_structural_migration_retains_weights_and_learns_spacing(self):
        long, _ = build(self.parent, self.rows, "long_identities")
        parent = initialize(long, "fresh_scaled", 17)
        old = json.loads(json.dumps(parent.snapshot()))
        migrated = migrate(parent)
        self.assertEqual(parent.snapshot(), old)
        self.assertEqual(migrated.model.characters[:len(old["symbols"])], old["symbols"])
        self.assertEqual(migrated.model.weights["hidden"][:len(old["weights"]["hidden"])],
                         old["weights"]["hidden"])
        self.assertEqual(migrated.model.weights["hidden"][len(old["weights"]["hidden"]):],
                         [0.0] * (64 * EMBEDDING * HIDDEN))
        for word in WORDS:
            token = migrated.model.tokens["identity:" + word]
            if "identity:" + word not in parent.model.tokens:
                other = parent.model.tokens["identity:<other>"]
                self.assertEqual(migrated.model.weights["embedding"][token * EMBEDDING:(token + 1) * EMBEDDING],
                                 parent.model.weights["embedding"][other * EMBEDDING:(other + 1) * EMBEDDING])
        for char in CHARACTERS:
            token = migrated.model.tokens["char:" + char]
            self.assertGreater(migrated.prior("test ")[token], 0)
            self.assertEqual(migrated.encode(char), [token])
            probe = ConversationContext.restore(json.loads(json.dumps(migrated.snapshot())))
            probe.model.weights["output_bias"][token] += 1000
            self.assertEqual(generate(probe, "red ", length=1)["text"], char)
        for mark in ".?!,":
            for suffix in ("", "b", "b c d "):
                contexts = [migrated.context("red" + mark + " " * n + suffix)
                            for n in range(3)]
                self.assertEqual(len({tuple(context) for context in contexts}), 3)
        self.assertNotEqual(migrated.context("red?" + " " * 318),
                            migrated.context("red?" + " " * 319))
        self.assertNotEqual(migrated.context("red?" + " " * 4094),
                            migrated.context("red?" + " " * 4095))
        with self.assertRaisesRegex(ValueError, "4095"):
            migrated.context("red?" + " " * 4096)
        self.assertEqual(migrated.snapshot(), migrate(parent).snapshot())
        self.assertEqual(ConversationContext.restore(migrated.snapshot()).snapshot()["weights"],
                         migrated.snapshot()["weights"])
        bad = json.loads(json.dumps(migrated.snapshot()))
        bad["symbols"].remove("gap:space-high:63")
        with self.assertRaisesRegex(ValueError, "Invalid saved conversation context"):
            ConversationContext.restore(bad)
        one, two = (migrated.context("red? "), migrated.context("red?  "))
        prior = migrated.prior("red? ")
        target = migrated.model.tokens["char:b"]
        for _ in range(5):
            migrated.model.update(one, target, output_prior=prior, learning_rate=.1)
        self.assertNotEqual(migrated.model.forward(one, output_prior=prior)[2],
                            migrated.model.forward(two, output_prior=prior)[2])

    def test_local_features_padding_and_review_gate(self):
        long, _ = build(self.parent, self.rows, "long_identities")
        for prefix in ("", "red", "red ", "red blue ", "red blue cup", "red blue cup "):
            context = long.context(prefix)
            self.assertEqual(len(context), CONTEXT)
            self.assertEqual(context[-4:], WordBase.context(long, prefix))
        state = long.snapshot()
        state["context"] = 8
        with self.assertRaises(ValueError):
            ConversationContext.restore(state)
        state = json.loads(json.dumps(long.snapshot()))
        state["weights"]["hidden"][0] = float("nan")
        with self.assertRaises(ValueError):
            ConversationContext.restore(state)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "fixtures").mkdir()
            (root / "fixtures/neural-conversation-draft.json").write_text("{}")
            with self.assertRaises(ValueError):
                run(root, acknowledge_review=True, reviewed_sha256="0" * 64)

    def test_word_position_migration_resolves_reviewed_collision_without_changing_predictions(self):
        question = "Can you quickly grab the brown fox that's jumping over the lazy dog by the fence?"
        answer = ("Sure, but before I do, I was just reading about five boxing wizards who jump quickly. "
                  "Weird coincidence, right?")
        row = {"id": "focus", "skill": "conversation",
               "turns": [{"role": "user", "content": question}], "answer": answer}
        text = prompt_for(row) + answer + "\n"
        base, _ = build(WordBase(text, 17, lexicon=["user", "assistant"]), [row], "long_identities")
        parent = migrate(initialize(base, "fresh_scaled", 17))
        old_state = json.loads(json.dumps(parent.snapshot()))
        upgraded = migrate_word_position(parent)
        self.assertEqual(parent.snapshot(), old_state)
        self.assertEqual(pair_ambiguity(parent, row)["conflicting_inputs"], 1)
        self.assertEqual(pair_ambiguity(upgraded, row)["conflicting_inputs"], 0)
        for prefix in (prompt_for(row), prompt_for(row) + "Sure, but before I do",
                       "Weird coinc", "Weird coincidenc", "red?  "):
            self.assertEqual(upgraded.context(prefix)[:128], parent.context(prefix))
            self.assertEqual(upgraded.model.forward(upgraded.context(prefix), output_prior=upgraded.prior(prefix))[2],
                             parent.model.forward(parent.context(prefix), output_prior=parent.prior(prefix))[2])
        self.assertNotEqual(upgraded.context("Weird coinc"), upgraded.context("Weird coincidenc"))
        self.assertNotEqual(upgraded.context("a" * 4094), upgraded.context("a" * 4095))
        with self.assertRaisesRegex(ValueError, "4095"):
            upgraded.context("a" * 4096)
        restored = PlasticityContext.restore(json.loads(json.dumps(upgraded.snapshot())))
        self.assertEqual(restored.snapshot(), upgraded.snapshot())
        self.assertEqual(restored.context("Weird coinc"), upgraded.context("Weird coinc"))
        malformed = json.loads(json.dumps(upgraded.snapshot()))
        del malformed["partial_word_features"]
        with self.assertRaisesRegex(ValueError, "Invalid saved conversation context"):
            PlasticityContext.restore(malformed)

    def test_lower_rate_selection_uses_teaching_loss_before_one_test(self):
        long, _ = build(self.parent, self.rows, "long_identities")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "fixtures").mkdir()
            data = {"train": self.rows, "test": [{"id": "test", "skill": "color",
                    "turns": [{"role": "user", "content": "red?"}], "answer": "red."}]}
            raw = json.dumps(data).encode()
            checksum = digest(raw)
            (root / "fixtures/neural-conversation-draft.json").write_bytes(raw)
            directory = root / ".data/neural-conversation-context" / ("a" * 32)
            directory.mkdir(parents=True)
            (directory / "report.json").write_text(json.dumps({
                "status": "completed", "review": {"file_sha256": checksum},
                "shuffle_seed": 17, "updates": [3, 5]}))
            (directory / "inputs.json").write_bytes(raw)
            (directory / "long_identities-initial.json").write_text(json.dumps({"state": long.snapshot()}))
            real_assess = assess
            test_calls = []
            def watched_assess(model, rows):
                if rows == data["test"]:
                    self.assertTrue((directory / "lower-rate-selection.json").exists())
                    test_calls.append(rows)
                return real_assess(model, rows)
            with patch("neural_conversation_context.assess", side_effect=watched_assess):
                result = lower_rate_screen(root, "a" * 32, acknowledge_review=True,
                                           reviewed_sha256=checksum)
            self.assertEqual(len(test_calls), 1)
            chosen = min(result["trials"], key=lambda t: t["checkpoints"][-1]["teacher_forced_loss"])
            self.assertEqual(result["selected_rate_by_training_loss"], chosen["rate"])
            with self.assertRaises(ValueError):
                lower_rate_screen(root, "a" * 32, acknowledge_review=True, reviewed_sha256=checksum)


if __name__ == "__main__":
    unittest.main()
