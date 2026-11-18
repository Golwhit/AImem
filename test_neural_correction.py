import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from neural_base_comparison import WordBase, digest
from neural_chat import CONVERSATION_MODEL, generate, load_active
from neural_conversation_context import build
from neural_conversation_meal import AUTHORED_FIXTURE, curriculum, FIXTURE
from neural_conversation_trial import canonical, prompt_for
from neural_plasticity import initialize
from neural_correction import add_spacing_features, train_correction
from neural_model_migration import migrate, migrate_word_position
from neural_training import load_progress, run
from original_generator import atomic_json


class ImmediateCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        data = curriculum()
        path = self.root / FIXTURE
        path.parent.mkdir()
        atomic_json(path, data)
        text = "".join(prompt_for(row) + row["answer"] for row in data["train"])
        base, _ = build(WordBase(text, 17), data["train"], "long_identities")
        model = initialize(base, "fresh_scaled", 17)
        evidence = {"run_id": "a" * 32, "round": 1, "reviewed_file_sha256": digest(path.read_bytes())}
        raw = canonical({"state": model.snapshot(), "evidence": evidence})
        checksum = digest(raw)
        training = self.root / ".data/neural-training"
        chat = self.root / ".data/neural-chat"
        training.mkdir(parents=True)
        chat.mkdir(parents=True)
        (training / f"{checksum}.json").write_bytes(raw)
        (chat / f"{checksum}.json").write_bytes(raw)
        atomic_json(training / "progress.json", {**evidence, "checkpoint_sha256": checksum})
        atomic_json(chat / "active.json", {"model": CONVERSATION_MODEL, "arm": "conversation",
                    "checkpoint_sha256": checksum, "passes": 1})
        atomic_json(self.root / "chat_memory.json", {"messages": [
            {"id": "You 1", "role": "You", "text": "What is the code?", "created_cycle": 1},
            {"id": "AI 1", "role": "Assistant", "text": "No.", "created_cycle": 1,
             "source": "local-generated", "model_provenance": {"model": CONVERSATION_MODEL}}]})
        self.payload = {"response_id": "AI 1", "correction": "Z.", "acknowledge_review": True}

    def test_explicit_correction_changes_active_weights_and_is_idempotent(self):
        before, _ = load_progress(self.root)
        result = train_correction(self.root, self.payload)
        self.assertTrue(result["trained"])
        self.assertEqual(result["prediction"], "Z.")
        pointer, trained = load_progress(self.root)
        self.assertNotEqual(pointer["checkpoint_sha256"], before["checkpoint_sha256"])
        self.assertIn("char:Z", trained.model.tokens)
        self.assertEqual(load_active(self.root)[1].snapshot(), trained.snapshot())
        self.assertEqual(generate(self.root, {"prompt": "What is the code?"})["text"], "Z.")
        repeated = train_correction(self.root, self.payload)
        self.assertTrue(repeated["already_learned"])
        self.assertEqual(repeated["checkpoint_sha256"], pointer["checkpoint_sha256"])
        self.assertEqual(pointer["approved_corrections"][0]["row"]["answer"], "Z.")
        archive = self.root / ".data/neural-chat" / (
            "retired-selection-" + load_active(self.root)[0]["rollback_selection_sha256"] + ".json")
        self.assertTrue(archive.exists())

    def test_character_only_correction_replays_and_keeps_output_mode(self):
        pointer, model = load_progress(self.root)
        model.set_output_mode("character_only")
        evidence = {key: value for key, value in pointer.items() if key != "checkpoint_sha256"}
        raw = canonical({"state": model.snapshot(), "evidence": evidence})
        checksum = digest(raw)
        training = self.root / ".data/neural-training"
        chat = self.root / ".data/neural-chat"
        (training / f"{checksum}.json").write_bytes(raw)
        (chat / f"{checksum}.json").write_bytes(raw)
        atomic_json(training / "progress.json", {**evidence, "checkpoint_sha256": checksum})
        selection = json.loads((chat / "active.json").read_bytes())
        atomic_json(chat / "active.json", {**selection, "checkpoint_sha256": checksum})
        result = train_correction(self.root, self.payload)
        self.assertTrue(result["trained"])
        _, trained = load_progress(self.root)
        self.assertEqual(trained.output_mode, "character_only")
        self.assertFalse(trained.word_tokens)
        self.assertEqual(load_active(self.root)[1].snapshot(), trained.snapshot())
        self.assertEqual(generate(self.root, {"prompt": "What is the code?"})["text"], "Z.")

    def test_failed_training_and_invalid_submission_do_not_promote(self):
        before = (self.root / ".data/neural-training/progress.json").read_bytes()
        for update in ({"acknowledge_review": False}, {"response_id": "missing"},
                       {"correction": "No."}, {"correction": "x\nx"}, {"correction": "a" * 320},
                       {"correction": "No..."}):
            with self.assertRaises(ValueError):
                train_correction(self.root, {**self.payload, **update})
        with patch("neural_correction.MAX_UPDATES", 0):
            with self.assertRaisesRegex(RuntimeError, "did not fit"):
                train_correction(self.root, self.payload)
        self.assertEqual((self.root / ".data/neural-training/progress.json").read_bytes(), before)

    def test_saturated_correction_candidate_is_not_promoted(self):
        before = (self.root / ".data/neural-training/progress.json").read_bytes()
        saturated = {"hidden_abs_over_099_fraction": 0.650001, "mean_tanh_derivative": 0.0,
                     "mean_hidden_gradient_norm": 0.0}
        with patch("neural_correction.diagnostics", return_value=saturated):
            with self.assertRaisesRegex(RuntimeError, "saturated"):
                train_correction(self.root, self.payload)
        self.assertEqual((self.root / ".data/neural-training/progress.json").read_bytes(), before)

    def test_correction_rejects_lost_approved_answer_despite_equal_total(self):
        from neural_conversation_trial import assess

        active = (self.root / ".data/neural-chat/active.json").read_bytes()
        progress = (self.root / ".data/neural-training/progress.json").read_bytes()
        calls = 0

        def assessment(model, rows, **kwargs):
            nonlocal calls
            if len(rows) == 1:
                return assess(model, rows, **kwargs)
            calls += 1
            result = assess(model, rows, **kwargs)
            for entry in result["rows"]:
                entry["exact"] = False
            result["rows"][0 if calls == 1 else 1]["exact"] = True
            result["exact"] = 1 / len(rows)
            return result

        with patch("neural_correction.assess", side_effect=assessment):
            with self.assertRaisesRegex(RuntimeError, "previously correct approved answer"):
                train_correction(self.root, self.payload)
        self.assertEqual((self.root / ".data/neural-chat/active.json").read_bytes(), active)
        self.assertEqual((self.root / ".data/neural-training/progress.json").read_bytes(), progress)
        reports = list((self.root / ".data/neural-training/rounds").glob("*/report.json"))
        report = json.loads(reports[0].read_text())
        self.assertEqual(report["status"], "rejected")
        self.assertEqual(len(report["retention"]["lost_correct_ids"]), 1)

    def test_correction_accepts_saturation_at_limit_and_records_it(self):
        diagnostics = {"hidden_abs_over_099_fraction": 0.65, "mean_tanh_derivative": 0.0,
                       "mean_hidden_gradient_norm": 0.0}
        with patch("neural_correction.diagnostics", return_value=diagnostics):
            result = train_correction(self.root, self.payload)
        report = json.loads((self.root / ".data/neural-training/rounds" / result["run_id"] /
                             "report.json").read_text())
        self.assertEqual(result["prediction"], "Z.")
        self.assertEqual(report["saturation"],
                         {"hidden_abs_over_099_fraction": 0.65, "limit": 0.65})

    def test_relevant_uses_saved_answer_and_trains_immediately(self):
        result = train_correction(self.root, {**self.payload, "rating": "useful",
                                             "correction": "Ignore this supplied override."})
        self.assertTrue(result["trained"])
        self.assertEqual(result["prediction"], "No.")
        pointer, _ = load_progress(self.root)
        approval = pointer["approved_corrections"][-1]
        self.assertEqual(approval["row"]["answer"], "No.")
        self.assertEqual(approval["rating"], "useful")
        self.assertGreater(result["updates"], 0)
        self.assertEqual(generate(self.root, {"prompt": "What is the code?"})["text"], "No.")

    def test_context_feedback_replays_the_exact_conversation_input(self):
        turns = [{"role": "user", "content": "The code is Z."},
                 {"role": "assistant", "content": "Okay."},
                 {"role": "user", "content": " What is the code? "}]
        memory = json.loads((self.root / "chat_memory.json").read_text())
        memory["messages"][1]["model_provenance"]["input_turns"] = turns
        atomic_json(self.root / "chat_memory.json", memory)

        result = train_correction(self.root, self.payload)
        pointer, _ = load_progress(self.root)
        approved = pointer["approved_corrections"][-1]
        self.assertEqual(approved["row"]["turns"], turns)
        self.assertEqual(result["reviewed_pair_sha256"], digest(canonical({"turns": turns, "answer": "Z."})))
        self.assertEqual(generate(self.root, {"prompt": turns[-1]["content"], "history": turns[:-1]})["text"], "Z.")
        self.assertTrue(train_correction(self.root, self.payload)["already_learned"])

    def test_context_feedback_rejects_mismatched_or_invalid_saved_input(self):
        memory = json.loads((self.root / "chat_memory.json").read_text())
        for turns in (
            [{"role": "user", "content": "Earlier"}, {"role": "assistant", "content": "Okay"},
             {"role": "user", "content": "A different question"}],
            [{"role": "assistant", "content": "Earlier"}, {"role": "assistant", "content": "Okay"},
             {"role": "user", "content": "What is the code?"}],
        ):
            memory["messages"][1]["model_provenance"]["input_turns"] = turns
            atomic_json(self.root / "chat_memory.json", memory)
            with self.assertRaises(ValueError):
                train_correction(self.root, self.payload)

    def test_spacing_features_preserve_initial_predictions_and_round_trip(self):
        _, model = load_progress(self.root)
        prefix = "user: Test?\nassistant: No. "
        self.assertEqual(model.context(prefix), model.context(prefix + " "))
        before = model.model.forward(model.context(prefix), output_prior=model.prior(prefix))[2]
        upgraded = add_spacing_features(model, "user: Test?\nassistant: ", "No.  No.")
        after = upgraded.model.forward(upgraded.context(prefix), output_prior=upgraded.prior(prefix))[2]
        self.assertEqual(before, after[:len(before)])
        self.assertTrue(all(value == 0 for value in after[len(before):]))
        self.assertNotEqual(upgraded.context(prefix), upgraded.context(prefix + " "))
        restored = type(upgraded).restore(upgraded.snapshot())
        self.assertEqual(restored.context(prefix), upgraded.context(prefix))
        self.assertEqual(restored.context(prefix + " "), upgraded.context(prefix + " "))

    def migrate_fixture(self):
        pointer, model = load_progress(self.root)
        evidence = {key: value for key, value in pointer.items() if key != "checkpoint_sha256"}
        encoded = canonical({"state": migrate(model).snapshot(), "evidence": evidence})
        checksum = digest(encoded)
        directory = self.root / ".data/neural-training"
        chat = self.root / ".data/neural-chat"
        (directory / f"{checksum}.json").write_bytes(encoded)
        (chat / f"{checksum}.json").write_bytes(encoded)
        atomic_json(directory / "progress.json", {**evidence, "checkpoint_sha256": checksum})
        active = json.loads((chat / "active.json").read_text())
        active["checkpoint_sha256"] = checksum
        atomic_json(chat / "active.json", active)

    def test_correction_trains_after_structural_migration(self):
        self.migrate_fixture()
        result = train_correction(self.root, {**self.payload, "correction": "A?  B."})
        self.assertEqual(result["prediction"], "A?  B.")
        self.assertEqual(load_progress(self.root)[1].model.context, 128)

    def test_correction_trains_after_word_position_migration(self):
        self.migrate_fixture()
        pointer, model = load_progress(self.root)
        evidence = {key: value for key, value in pointer.items() if key != "checkpoint_sha256"}
        encoded = canonical({"state": migrate_word_position(model).snapshot(), "evidence": evidence})
        checksum = digest(encoded)
        training = self.root / ".data/neural-training"
        chat = self.root / ".data/neural-chat"
        (training / f"{checksum}.json").write_bytes(encoded)
        (chat / f"{checksum}.json").write_bytes(encoded)
        atomic_json(training / "progress.json", {**evidence, "checkpoint_sha256": checksum})
        active = json.loads((chat / "active.json").read_text())
        active["checkpoint_sha256"] = checksum
        atomic_json(chat / "active.json", active)
        result = train_correction(self.root, self.payload)
        self.assertEqual(result["prediction"], "Z.")
        self.assertEqual(load_progress(self.root)[1].model.context, 130)

    def test_long_correction_can_fit_with_expanded_migration_budget(self):
        self.migrate_fixture()
        self.test_long_correction_trains_replays_and_generates_without_truncation()

    def test_long_correction_trains_replays_and_generates_without_truncation(self):
        answer = ("One blue bird rests on the red box. The next small bird waits nearby. "
                  "They are safe until the morning comes.")
        self.assertGreaterEqual(len(answer), 105)
        result = train_correction(self.root, {**self.payload, "correction": answer})
        self.assertTrue(result["trained"])
        self.assertEqual(result["prediction"], answer)
        self.assertEqual(generate(self.root, {"prompt": "What is the code?", "max_chars": 320})["text"], answer)
        self.assertTrue(train_correction(self.root, {**self.payload, "correction": answer})["already_learned"])

    def test_future_round_rehearses_approved_pair_instead_of_conflicting_base_answer(self):
        data = json.loads((self.root / FIXTURE).read_text())
        prompt = data["train"][0]["turns"][0]["content"]
        memory = json.loads((self.root / "chat_memory.json").read_text())
        memory["messages"][0]["text"] = prompt
        atomic_json(self.root / "chat_memory.json", memory)
        train_correction(self.root, self.payload)
        pointer, _ = load_progress(self.root)
        for name in ("neural_training.py", "neural_plasticity.py", "neural_experiment.py",
                     "neural_conversation_trial.py", "neural_conversation_context.py",
                     "neural_base_comparison.py"):
            (self.root / name).write_bytes((Path(__file__).parent / name).read_bytes())
        with patch("neural_training.fit", side_effect=RuntimeError("captured optimizer input")) as fit:
            with self.assertRaisesRegex(RuntimeError, "captured optimizer input"):
                run(self.root, cpu_seconds=1)
        rows = fit.call_args.args[1]
        self.assertEqual(len(rows), len(data["train"]))
        targets = [row["answer"] for row in rows if row["turns"] == data["train"][0]["turns"]]
        self.assertEqual(targets, ["Z."])
        self.assertEqual(load_progress(self.root)[0], pointer)

    def test_correction_preserves_authored_fixture_lineage(self):
        pointer, model = load_progress(self.root)
        fixture = self.root / AUTHORED_FIXTURE
        fixture.write_bytes((self.root / FIXTURE).read_bytes())
        evidence = {key: value for key, value in pointer.items() if key != "checkpoint_sha256"}
        evidence["fixture"] = AUTHORED_FIXTURE
        encoded = canonical({"state": model.snapshot(), "evidence": evidence})
        checksum = digest(encoded)
        directory = self.root / ".data/neural-training"
        chat = self.root / ".data/neural-chat"
        (directory / f"{checksum}.json").write_bytes(encoded)
        (chat / f"{checksum}.json").write_bytes(encoded)
        atomic_json(directory / "progress.json", {**evidence, "checkpoint_sha256": checksum})
        active = json.loads((chat / "active.json").read_text())
        active["checkpoint_sha256"] = checksum
        atomic_json(chat / "active.json", active)
        train_correction(self.root, self.payload)
        pointer, _ = load_progress(self.root)
        self.assertEqual(pointer["fixture"], AUTHORED_FIXTURE)
        self.assertEqual(len(pointer["approved_corrections"]), 1)


if __name__ == "__main__":
    unittest.main()
