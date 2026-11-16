import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from neural_base_comparison import WordBase, digest
from neural_conversation_context import build
from neural_conversation_meal import curriculum, FIXTURE
from neural_conversation_trial import answer_examples, assess, lost_correct_ids, prompt_for, token_loss
from neural_plasticity import diagnostics, initialize
from neural_training import DIRECTORY, INITIAL_RUN, activate, health_problem, load_progress, rollback, run
from neural_chat import CONVERSATION_MODEL, activate_base, generate, load_active
from test_neural_chat import base_chat_fixture


def checkpoint(updates, saturation, exact):
    return {"updates": updates, "training": {"exact": exact},
            "diagnostics": {"hidden_abs_over_099_fraction": saturation}}


class HealthGuardTests(unittest.TestCase):
    def test_retention_rejects_answer_swaps_and_mismatched_assessments(self):
        before = {"rows": [{"id": "a", "expected": "A", "exact": True},
                           {"id": "b", "expected": "B", "exact": False}]}
        after = {"rows": [{"id": "a", "expected": "A", "exact": False},
                          {"id": "b", "expected": "B", "exact": True}]}
        self.assertEqual(lost_correct_ids(before, after), ["a"])
        with self.assertRaisesRegex(ValueError, "different numbers"):
            lost_correct_ids(before, {"rows": after["rows"][:1]})
        with self.assertRaisesRegex(ValueError, "different examples"):
            lost_correct_ids(before, {"rows": list(reversed(after["rows"]))})

    def test_accepts_healthy_round_and_rejects_saturated_or_regressed_rounds(self):
        self.assertIsNone(health_problem([checkpoint(1, .1, 1.0), checkpoint(2, .43, 1.0)]))
        self.assertIn("saturation", health_problem([checkpoint(1, .2, 1.0), checkpoint(2, 1.0, 1.0)]))
        self.assertIn("teaching exact", health_problem([checkpoint(1, .2, 1.0), checkpoint(2, .2, 0.0)]))
        self.assertIn("no checkpoints", health_problem([]))


class PermanentTrainingTests(unittest.TestCase):
    def test_rounds_resume_saved_weights_and_reject_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "fixtures").mkdir()
            data = curriculum()
            raw = json.dumps(data).encode()
            (root / FIXTURE).write_bytes(raw)
            checksum = digest(raw)
            original_chat = activate_base(root, base_chat_fixture(root))
            original_reply = generate(root, {"prompt": "A ", "max_chars": 8}, original=True)
            text = "".join(prompt_for(row) + row["answer"] for row in data["train"])
            base, _ = build(WordBase(text, 17), data["train"], "long_identities")
            model = initialize(base, "fresh_scaled", 17)
            source = root / ".data/neural-conversation-meal" / INITIAL_RUN
            source.mkdir(parents=True)
            (source / "fresh_scaled-17-meal64-final.json").write_text(json.dumps({
                "state": model.snapshot(), "trial": {"arm": "fresh_scaled", "seed": 17, "subset": "meal64"}}))
            (source / "report.json").write_text(json.dumps({
                "status": "completed", "review": {"file_sha256": checksum}}))
            (source / "validation.json").write_text(json.dumps({"exact_weight_and_test_replays": 12}))
            for name in ("neural_training.py", "neural_plasticity.py", "neural_experiment.py",
                         "neural_conversation_trial.py", "neural_conversation_context.py",
                         "neural_base_comparison.py"):
                (root / name).write_bytes((Path(__file__).parent / name).read_bytes())
            with self.assertRaises(ValueError):
                run(root, cpu_seconds=float("nan"))

            def brief_fit(model, rows, *, seed, rate, cpu_checkpoints, no_triple_repeat=False):
                examples = answer_examples(model, rows)
                context, target, prior = examples[random.Random(seed).randrange(len(examples))]
                model.model.update(context, target, output_prior=prior, learning_rate=rate)
                return [{"updates": 1, "teacher_forced_loss": token_loss(model, examples),
                         "training": assess(model, rows, no_triple_repeat=no_triple_repeat),
                         "diagnostics": diagnostics(model, examples),
                         "training_cpu_seconds": 0.0}]

            with patch("neural_training.fit", side_effect=brief_fit), \
                    patch("neural_training.health_problem", return_value=None):
                first = run(root, cpu_seconds=1)
                self.assertEqual(first["reviewed_file_sha256"], checksum)
                pointer, trained = load_progress(root)
                first_state = json.loads(json.dumps(trained.snapshot()))
                first_hash = pointer["checkpoint_sha256"]
                active = activate(root)
                self.assertEqual(active["model"], CONVERSATION_MODEL)
                self.assertEqual(active["checkpoint_sha256"], first_hash)
                self.assertEqual(load_active(root)[1].snapshot(), trained.snapshot())
                self.assertEqual(activate(root), active)
                archive = root / ".data/neural-chat" / (
                    "retired-selection-" + active["rollback_selection_sha256"] + ".json")
                archived = archive.read_bytes()
                archive.write_bytes(archived + b" ")
                with self.assertRaisesRegex(ValueError, "Rollback selection hash mismatch"):
                    rollback(root)
                self.assertEqual(load_active(root)[0]["checkpoint_sha256"], first_hash)
                archive.write_bytes(archived)
                with patch("neural_chat.generate_base", return_value={"text": "Red.\nnoise"}) as generated:
                    reply = generate(root, {"prompt": "What color?", "max_chars": 320})
                self.assertEqual(reply["text"], "Red.")
                self.assertTrue(reply["terminated"])
                self.assertEqual(generated.call_args.args[1], "user: What color?\nassistant: ")
                self.assertEqual(generated.call_args.kwargs["length"], 320)
                with patch("neural_chat.generate_base", return_value={"text": "\nnoise"}):
                    with self.assertRaisesRegex(ArithmeticError, "empty first answer"):
                        generate(root, {"prompt": "What color?"})
                self.assertEqual(generate(root, {"prompt": "A ", "max_chars": 8}, original=True), original_reply)
                second = run(root, cpu_seconds=1)
            self.assertEqual(first["round"], 1)
            self.assertEqual(second["round"], 2)
            self.assertEqual(second["parent_checkpoint_sha256"], first_hash)
            initial = root / DIRECTORY / "rounds" / second["run_id"] / "initial.json"
            self.assertEqual(json.loads(initial.read_text())["state"], first_state)
            pointer, _ = load_progress(root)
            activate(root)
            self.assertEqual(rollback(root)["checkpoint_sha256"], first_hash)
            self.assertEqual(rollback(root)["checkpoint_sha256"], original_chat["checkpoint_sha256"])
            self.assertNotEqual(pointer["checkpoint_sha256"], first_hash)
            self.assertTrue((root / DIRECTORY / f"{first_hash}.json").exists())
            progress_before = (root / DIRECTORY / "progress.json").read_bytes()

            def baseline(_model, rows, **kwargs):
                measured = [{"id": row["id"], "skill": row["skill"], "pair": row.get("pair"),
                             "expected": row["answer"], "prediction": row["answer"],
                             "exact": True, "normalized_exact": True, "terminated": True} for row in rows]
                if len(rows) == len(data["train"]):
                    measured[1]["exact"] = False
                return {"rows": measured, "exact": sum(row["exact"] for row in measured) / len(measured)}

            def swapped_fit(model, rows, **kwargs):
                after = baseline(model, rows)
                after["rows"][0]["exact"] = False
                after["rows"][1]["exact"] = True
                return [{"updates": 0, "training": after,
                         "diagnostics": {"hidden_abs_over_099_fraction": .1}}]

            with patch("neural_training.assess", side_effect=baseline), \
                    patch("neural_training.fit", side_effect=swapped_fit), \
                    patch("neural_training.replay", side_effect=lambda state, *a, **k:
                          type(trained).restore(state)), \
                    patch("neural_training.health_problem", return_value=None):
                rejected = run(root, cpu_seconds=1)
            self.assertEqual(rejected["status"], "rejected")
            self.assertIn("previously correct approved teaching regressed", rejected["rejection"])
            self.assertEqual((root / DIRECTORY / "progress.json").read_bytes(), progress_before)
            checkpoint = root / DIRECTORY / f"{pointer['checkpoint_sha256']}.json"
            checkpoint.write_bytes(checkpoint.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_progress(root)


if __name__ == "__main__":
    unittest.main()
