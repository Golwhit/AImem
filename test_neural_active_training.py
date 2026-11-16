import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from neural_active_training import APPROVED_SHA256, round_once
from neural_base_comparison import WordBase, digest
from neural_chat import CONVERSATION_MODEL, load_active
from neural_conversation_context import build
from neural_conversation_meal import AUTHORED_FIXTURE
from neural_conversation_trial import canonical, feedback_hash, prompt_for
from neural_base_diagnostic import ambiguity
from neural_continuation import extend_alphabet
from neural_corpus_training import CORPUS, advance, corpus_examples, corpus_records, corpus_symbols, run as run_corpus
from neural_model_migration import (
    CHARACTERS, WORDS, WORD_POSITION_REVIEW, migrate, migrate_word_position, run as migrate_active,
    run_word_position,
)
from neural_plasticity import PlasticityContext, initialize
from neural_training import DIRECTORY, activate, load_progress, rollback
from original_generator import atomic_json


class ActiveContinuationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        raw = (Path(__file__).parent / AUTHORED_FIXTURE).read_bytes()
        self.assertEqual(digest(raw), APPROVED_SHA256)
        fixture = self.root / AUTHORED_FIXTURE
        fixture.parent.mkdir()
        fixture.write_bytes(raw)
        data = json.loads(raw)
        text = "".join(prompt_for(row) + row["answer"] for row in data["train"])
        base, _ = build(WordBase(text, 17), data["train"], "long_identities")
        model = initialize(base, "fresh_scaled", 17)
        self.parent_state = model.snapshot()
        chat = self.root / ".data/neural-chat"
        chat.mkdir(parents=True)
        raw_checkpoint = canonical({"state": self.parent_state, "index": None})
        active_hash = digest(raw_checkpoint)
        (chat / f"{active_hash}.json").write_bytes(raw_checkpoint)
        run_id = "a" * 32
        train_hash = digest(canonical(data["train"]))
        active = {"model": CONVERSATION_MODEL, "arm": "conversation", "run_id": run_id,
                  "seed": 17, "passes": 0, "checkpoint_sha256": active_hash,
                  "source": {"run_id": run_id, "arm": "fresh_scaled", "subset": "meal256",
                             "training": "approved", "training_sha256": train_hash}}
        atomic_json(chat / "active.json", active)
        meal = self.root / ".data/neural-conversation-meal" / run_id
        atomic_json(meal / "report.json", {
            "status": "completed", "trials": [{}] * 12,
            "review": {"file_sha256": APPROVED_SHA256,
                       "canonical_file_sha256": digest(canonical(data)), "train_sha256": train_hash}})
        atomic_json(meal / "validation.json", {"exact_weight_and_test_replays": 12,
                                                "protected_checkpoints_unchanged": True})
        atomic_json(meal / "fresh_scaled-17-meal256-final.json", {"state": self.parent_state})
        row = {"id": "correction-1", "skill": "user_correction",
               "turns": [{"role": "user", "content": "What is the code?"}], "answer": "Z."}
        correction = {"row": row, "sha256": digest(canonical(
            {"prompt": row["turns"][0]["content"], "answer": row["answer"]}))}
        evidence = {"run_id": "b" * 32, "round": 1, "reviewed_file_sha256": "c" * 64,
                    "approved_corrections": [correction]}
        raw_old = canonical({"state": self.parent_state, "evidence": evidence})
        old_hash = digest(raw_old)
        directory = self.root / DIRECTORY
        directory.mkdir(parents=True)
        (directory / f"{old_hash}.json").write_bytes(raw_old)
        atomic_json(directory / "progress.json", {**evidence, "checkpoint_sha256": old_hash})
        self.old_progress = (directory / "progress.json").read_bytes()
        self.active_bytes = (chat / "active.json").read_bytes()

    @staticmethod
    def assess_correct(model, rows, **kwargs):
        measured = [{"id": row["id"], "skill": row["skill"], "pair": row.get("pair"),
                     "expected": row["answer"], "prediction": row["answer"],
                     "exact": True, "normalized_exact": True, "terminated": True} for row in rows]
        return {"exact": 1.0, "rows": measured}

    def migration_parent(self):
        evidence = {"run_id": "c" * 32, "round": 1, "reviewed_file_sha256": APPROVED_SHA256,
                    "fixture": AUTHORED_FIXTURE, "approved_corrections": []}
        raw = canonical({"state": self.parent_state, "evidence": evidence})
        checksum = digest(raw)
        training = self.root / DIRECTORY
        chat = self.root / ".data/neural-chat"
        (training / f"{checksum}.json").write_bytes(raw)
        (chat / f"{checksum}.json").write_bytes(raw)
        atomic_json(training / "progress.json", {**evidence, "checkpoint_sha256": checksum})
        atomic_json(chat / "active.json", {**json.loads(self.active_bytes), "checkpoint_sha256": checksum})
        return checksum, digest(canonical({"words": list(WORDS), "characters": list(CHARACTERS)}))

    def test_migration_review_gate_activation_and_rollback(self):
        parent_hash, reviewed = self.migration_parent()
        with self.assertRaisesRegex(ValueError, "exact vocabulary hash"):
            migrate_active(self.root, acknowledge_review=True, reviewed_sha256="0" * 64)
        with patch("neural_model_migration.assess", side_effect=self.assess_correct):
            report = migrate_active(self.root, acknowledge_review=True, reviewed_sha256=reviewed)
        pointer, model = load_progress(self.root)
        self.assertEqual(model.model.context, 128)
        self.assertEqual(load_active(self.root)[0]["checkpoint_sha256"], pointer["checkpoint_sha256"])
        self.assertEqual(report["kind"], "reviewed_structural_migration")
        self.assertNotIn("full_weights_and_milestones_replay_exact", report["validation"])
        report_path = self.root / DIRECTORY / "rounds" / report["run_id"] / "report.json"
        original_report = report_path.read_bytes()
        bad_report = {**report, "validation": {"full_weights_and_milestones_replay_exact": True,
                                                "test_replay_exact": True}}
        bad_bytes = canonical(bad_report)
        report_path.write_bytes(bad_bytes)
        with patch("neural_training.load_progress", return_value=(
                {**pointer, "report_sha256": digest(bad_bytes)}, model)):
            with self.assertRaisesRegex(ValueError, "replay-verified"):
                activate(self.root)
        report_path.write_bytes(original_report)
        snapshots = []
        def fake_fit(model, rows, **kwargs):
            model.model.weights["hidden_bias"][0] += .001
            snapshots.append(model.snapshot())
            return [{"updates": 1, "training": self.assess_correct(model, rows),
                     "diagnostics": {"hidden_abs_over_099_fraction": .1}}]
        with patch("neural_active_training.assess", side_effect=self.assess_correct), \
                patch("neural_active_training.fit", side_effect=fake_fit), \
                patch("neural_active_training.replay", side_effect=lambda state, *a, **k:
                      PlasticityContext.restore(snapshots[-1])):
            round_once(self.root, cpu_seconds=1)
        self.assertEqual(load_progress(self.root)[1].model.context, 128)
        self.assertEqual(rollback(self.root)["checkpoint_sha256"], pointer["checkpoint_sha256"])
        self.assertEqual(rollback(self.root)["checkpoint_sha256"], parent_hash)

    def test_migration_rejects_lost_correct_answer(self):
        _, reviewed = self.migration_parent()
        active_path = self.root / ".data/neural-chat/active.json"
        progress_path = self.root / DIRECTORY / "progress.json"
        active_before, progress_before = active_path.read_bytes(), progress_path.read_bytes()

        def assessment(model, rows):
            result = self.assess_correct(model, rows)
            if model.model.context == 128:
                result["rows"][0]["exact"] = False
            return result

        with patch("neural_model_migration.assess", side_effect=assessment):
            with self.assertRaisesRegex(RuntimeError, "previously correct answers regressed"):
                migrate_active(self.root, acknowledge_review=True, reviewed_sha256=reviewed)
        self.assertEqual(active_path.read_bytes(), active_before)
        self.assertEqual(progress_path.read_bytes(), progress_before)

    def test_user_corpus_representation_replay_and_guarded_activation(self):
        self.migration_parent()
        progress, parent = load_progress(self.root)
        evidence = {key: value for key, value in progress.items() if key != "checkpoint_sha256"}
        upgraded = migrate_word_position(migrate(parent))
        checkpoint = canonical({"state": upgraded.snapshot(), "evidence": evidence})
        parent_hash = digest(checkpoint)
        training = self.root / DIRECTORY
        chat = self.root / ".data/neural-chat"
        (training / f"{parent_hash}.json").write_bytes(checkpoint)
        (chat / f"{parent_hash}.json").write_bytes(checkpoint)
        atomic_json(training / "progress.json", {**evidence, "checkpoint_sha256": parent_hash})
        atomic_json(chat / "active.json", {**json.loads(self.active_bytes), "checkpoint_sha256": parent_hash})
        corpus_path = self.root / CORPUS
        corpus_path.write_bytes((Path(__file__).parent / CORPUS).read_bytes())
        raw = corpus_path.read_bytes()
        data = json.loads(raw)
        records = corpus_records(data)
        self.assertEqual(len(records), 30)
        self.assertEqual(records[20:24], [("", "A"), ("", "I"), ("", "I "), ("", "A ")])
        self.assertEqual(records[24:27], [("text:", " a "), ("text:", " A "), ("text:", " I ")])
        self.assertEqual([text for _, text in records[-3:]], ["abs", "sch", "bl"])
        _, parent = load_progress(self.root)
        expanded = extend_alphabet(parent.model, sorted(corpus_symbols(records)))
        state = parent.snapshot()
        state.update(symbols=expanded.characters, weights=expanded.weights)
        model = PlasticityContext.restore(state)
        for record in records:
            self.assertEqual(ambiguity(corpus_examples(model, [record]))["conflicting_inputs"], 0)
        numeric = corpus_examples(model, [records[5]])
        self.assertEqual(len({tuple(context) for context, _, _ in numeric}), len(numeric))
        replayed = PlasticityContext.restore(json.loads(json.dumps(state)))
        conversations = corpus_examples(model, [("text: ", "test")])
        corpus = corpus_examples(model, records)
        import random
        advance(model, conversations, corpus, random.Random(7), 12, .02)
        advance(replayed, corpus_examples(replayed, [("text: ", "test")]), corpus_examples(replayed, records),
                random.Random(7), 12, .02)
        self.assertEqual(model.snapshot(), replayed.snapshot())
        with self.assertRaisesRegex(ValueError, "exact corpus file hash"):
            run_corpus(self.root, acknowledge_review=True, reviewed_sha256="0" * 64, cpu_seconds=1)

        def cheap_milestone(model, teaching, _conversations, corpus, updates):
            return {"updates": updates, "training": self.assess_correct(model, teaching),
                    "corpus": {"targets": len(corpus), "correct": 0, "loss": 1.0},
                    "diagnostics": {"hidden_abs_over_099_fraction": .1}}

        with patch("neural_corpus_training.assess", side_effect=self.assess_correct), \
                patch("neural_corpus_training.milestone", side_effect=cheap_milestone):
            report = run_corpus(self.root, acknowledge_review=True, reviewed_sha256=digest(raw), cpu_seconds=1)
        pointer, _ = load_progress(self.root)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(load_active(self.root)[0]["checkpoint_sha256"], pointer["checkpoint_sha256"])
        self.assertEqual(report["review"]["corpus_file_sha256"], digest(raw))
        self.assertTrue(report["validation"]["full_weights_and_milestones_replay_exact"])
        self.assertEqual(rollback(self.root)["checkpoint_sha256"], parent_hash)
        atomic_json(training / "progress.json", {**evidence, "checkpoint_sha256": parent_hash})
        active_before = (chat / "active.json").read_bytes()
        progress_before = (training / "progress.json").read_bytes()

        def regressed_milestone(model, teaching, conversations, corpus, updates):
            point = cheap_milestone(model, teaching, conversations, corpus, updates)
            point["training"]["exact"] = 0.0
            return point

        with patch("neural_corpus_training.assess", side_effect=self.assess_correct), \
                patch("neural_corpus_training.milestone", side_effect=regressed_milestone):
            with self.assertRaisesRegex(RuntimeError, "teaching accuracy regressed"):
                run_corpus(self.root, acknowledge_review=True, reviewed_sha256=digest(raw), cpu_seconds=1)
        self.assertEqual((chat / "active.json").read_bytes(), active_before)
        self.assertEqual((training / "progress.json").read_bytes(), progress_before)

        def baseline(model, rows, **kwargs):
            result = self.assess_correct(model, rows)
            if len(rows) > 200:
                result["rows"][1]["exact"] = False
                result["exact"] = (len(rows) - 1) / len(rows)
            return result

        def swapped_milestone(model, teaching, conversations, corpus, updates):
            point = cheap_milestone(model, teaching, conversations, corpus, updates)
            point["training"] = baseline(model, teaching)
            point["training"]["rows"][0]["exact"] = False
            point["training"]["rows"][1]["exact"] = True
            return point

        with patch("neural_corpus_training.assess", side_effect=baseline), \
                patch("neural_corpus_training.milestone", side_effect=swapped_milestone):
            with self.assertRaisesRegex(RuntimeError, "previously correct approved teaching regressed"):
                run_corpus(self.root, acknowledge_review=True, reviewed_sha256=digest(raw), cpu_seconds=1)
        self.assertEqual((chat / "active.json").read_bytes(), active_before)
        self.assertEqual((training / "progress.json").read_bytes(), progress_before)

    def test_word_position_migration_requires_collision_free_replay_and_can_roll_back(self):
        self.migration_parent()
        progress, parent = load_progress(self.root)
        row = {"id": "reviewed", "skill": "conversation",
               "turns": [{"role": "user", "content": "Spell it?"}], "answer": "coincidence."}
        pair_hash = feedback_hash(row["turns"], row["answer"])
        evidence = {key: value for key, value in progress.items() if key != "checkpoint_sha256"}
        evidence["approved_corrections"] = [{"row": row, "sha256": pair_hash}]
        raw = canonical({"state": migrate(parent).snapshot(), "evidence": evidence})
        parent_hash = digest(raw)
        training = self.root / DIRECTORY
        chat = self.root / ".data/neural-chat"
        (training / f"{parent_hash}.json").write_bytes(raw)
        (chat / f"{parent_hash}.json").write_bytes(raw)
        atomic_json(training / "progress.json", {**evidence, "checkpoint_sha256": parent_hash})
        atomic_json(chat / "active.json", {**json.loads(self.active_bytes), "checkpoint_sha256": parent_hash})
        reviewed = digest(canonical(WORD_POSITION_REVIEW))
        with self.assertRaisesRegex(ValueError, "exact word-position feature hash"):
            run_word_position(self.root, acknowledge_review=True, reviewed_sha256="0" * 64,
                              target_pair_sha256=pair_hash)
        with patch("neural_model_migration.assess", side_effect=self.assess_correct):
            report = run_word_position(self.root, acknowledge_review=True, reviewed_sha256=reviewed,
                                       target_pair_sha256=pair_hash)
        pointer, model = load_progress(self.root)
        self.assertEqual(model.model.context, 130)
        self.assertEqual(report["representation"]["before"]["conflicting_inputs"], 1)
        self.assertEqual(report["representation"]["after"]["conflicting_inputs"], 0)
        self.assertEqual(load_active(self.root)[0]["checkpoint_sha256"], pointer["checkpoint_sha256"])
        report_path = training / "rounds" / report["run_id"] / "report.json"
        original = report_path.read_bytes()
        invalid = {**report, "validation": {**report["validation"], "target_pair_collision_free": False}}
        bad_bytes = canonical(invalid)
        report_path.write_bytes(bad_bytes)
        with patch("neural_training.load_progress", return_value=(
                {**pointer, "report_sha256": digest(bad_bytes)}, model)):
            with self.assertRaisesRegex(ValueError, "replay-verified"):
                activate(self.root)
        report_path.write_bytes(original)
        self.assertEqual(rollback(self.root)["checkpoint_sha256"], parent_hash)

    def test_failed_migration_activation_restores_progress(self):
        _, reviewed = self.migration_parent()
        active_path = self.root / ".data/neural-chat/active.json"
        progress_path = self.root / DIRECTORY / "progress.json"
        active_before, progress_before = active_path.read_bytes(), progress_path.read_bytes()
        with patch("neural_model_migration.assess", side_effect=self.assess_correct), \
                patch("neural_model_migration.activate", side_effect=OSError("activation failed")):
            with self.assertRaisesRegex(OSError, "activation failed"):
                migrate_active(self.root, acknowledge_review=True, reviewed_sha256=reviewed)
        self.assertEqual(active_path.read_bytes(), active_before)
        self.assertEqual(json.loads(progress_path.read_bytes()), json.loads(progress_before))

    def test_continues_active_weights_keeps_feedback_and_rolls_back(self):
        snapshots = []

        def fake_fit(model, rows, **kwargs):
            self.assertTrue(any(row["id"] == "correction-1" for row in rows))
            model.model.weights["hidden_bias"][0] += .001
            snapshots.append(model.snapshot())
            return [{"updates": 1, "training": self.assess_correct(model, rows),
                     "diagnostics": {"hidden_abs_over_099_fraction": .1}}]

        def fake_replay(*args, **kwargs):
            return PlasticityContext.restore(snapshots[-1])

        with patch("neural_active_training.assess", side_effect=self.assess_correct), \
                patch("neural_active_training.fit", side_effect=fake_fit), \
                patch("neural_active_training.replay", side_effect=fake_replay):
            first = round_once(self.root, cpu_seconds=1)
            pointer, model = load_progress(self.root)
            self.assertEqual(pointer["fixture"], AUTHORED_FIXTURE)
            self.assertEqual(len(pointer["approved_corrections"]), 1)
            self.assertEqual(model.snapshot(), load_active(self.root)[1].snapshot())
            self.assertIn("char:Z", model.snapshot()["symbols"])
            parent = json.loads((self.root / DIRECTORY / "rounds" / first["run_id"] / "parent.json").read_text())
            self.assertEqual(parent["state"], self.parent_state)
            first_state = model.snapshot()
            second = round_once(self.root, cpu_seconds=1)
            parent = json.loads((self.root / DIRECTORY / "rounds" / second["run_id"] / "parent.json").read_text())
            self.assertEqual(parent["state"], first_state)
            self.assertEqual(rollback(self.root)["checkpoint_sha256"], pointer["checkpoint_sha256"])
            self.assertEqual(rollback(self.root)["checkpoint_sha256"], json.loads(self.active_bytes)["checkpoint_sha256"])

    def test_rejection_does_not_change_active_or_progress(self):
        def fake_fit(model, rows, **kwargs):
            return [{"updates": 0, "training": {**self.assess_correct(model, rows), "exact": 0.0},
                     "diagnostics": {"hidden_abs_over_099_fraction": .1}}]

        with patch("neural_active_training.assess", side_effect=self.assess_correct), \
                patch("neural_active_training.fit", side_effect=fake_fit), \
                patch("neural_active_training.replay", side_effect=lambda state, *a, **k:
                      PlasticityContext.restore(state)):
            with self.assertRaisesRegex(RuntimeError, "teaching accuracy regressed"):
                round_once(self.root, cpu_seconds=1)
        self.assertEqual((self.root / DIRECTORY / "progress.json").read_bytes(), self.old_progress)
        self.assertEqual((self.root / ".data/neural-chat/active.json").read_bytes(), self.active_bytes)

    def test_equal_teaching_total_cannot_hide_a_lost_answer(self):
        def baseline(model, rows, **kwargs):
            result = self.assess_correct(model, rows)
            if len(rows) > 200:
                result["rows"][1]["exact"] = False
                result["exact"] = (len(rows) - 1) / len(rows)
            return result

        def fake_fit(model, rows, **kwargs):
            after = baseline(model, rows)
            after["rows"][0]["exact"] = False
            after["rows"][1]["exact"] = True
            return [{"updates": 0, "training": after,
                     "diagnostics": {"hidden_abs_over_099_fraction": .1}}]

        with patch("neural_active_training.assess", side_effect=baseline), \
                patch("neural_active_training.fit", side_effect=fake_fit), \
                patch("neural_active_training.replay", side_effect=lambda state, *a, **k:
                      PlasticityContext.restore(state)):
            with self.assertRaisesRegex(RuntimeError, "previously correct approved teaching regressed"):
                round_once(self.root, cpu_seconds=1)
        self.assertEqual((self.root / DIRECTORY / "progress.json").read_bytes(), self.old_progress)
        self.assertEqual((self.root / ".data/neural-chat/active.json").read_bytes(), self.active_bytes)

    def test_contextual_feedback_is_accepted_for_future_rehearsal(self):
        turns = [{"role": "user", "content": "The box is blue."},
                 {"role": "assistant", "content": "Okay."},
                 {"role": "user", "content": "What color?"}]
        row = {"id": "correction-context", "skill": "user_correction", "turns": turns, "answer": "Blue."}
        evidence = {"run_id": "c" * 32, "round": 1, "reviewed_file_sha256": APPROVED_SHA256,
                    "fixture": AUTHORED_FIXTURE,
                    "approved_corrections": [{"row": row, "sha256": feedback_hash(turns, row["answer"])}]}
        raw = canonical({"state": self.parent_state, "evidence": evidence})
        checksum = digest(raw)
        training = self.root / DIRECTORY
        chat = self.root / ".data/neural-chat"
        (training / f"{checksum}.json").write_bytes(raw)
        (chat / f"{checksum}.json").write_bytes(raw)
        atomic_json(training / "progress.json", {**evidence, "checkpoint_sha256": checksum})
        atomic_json(chat / "active.json", {**json.loads(self.active_bytes), "checkpoint_sha256": checksum})
        with patch("neural_active_training.assess", side_effect=self.assess_correct), \
                patch("neural_active_training.fit", side_effect=RuntimeError("captured")) as fit:
            with self.assertRaisesRegex(RuntimeError, "captured"):
                round_once(self.root, cpu_seconds=1)
        self.assertIn(row, fit.call_args.args[1])

    def test_rejects_changed_curriculum(self):
        path = self.root / AUTHORED_FIXTURE
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "differs from the approved"):
            round_once(self.root, cpu_seconds=1)
        self.assertEqual((self.root / DIRECTORY / "progress.json").read_bytes(), self.old_progress)

    def test_active_selection_change_is_not_overwritten(self):
        active_path = self.root / ".data/neural-chat/active.json"

        def fake_fit(model, rows, **kwargs):
            selection = json.loads(active_path.read_text())
            selection["note"] = "changed concurrently"
            atomic_json(active_path, selection)
            return [{"updates": 0, "training": self.assess_correct(model, rows),
                     "diagnostics": {"hidden_abs_over_099_fraction": .1}}]

        with patch("neural_active_training.assess", side_effect=self.assess_correct), \
                patch("neural_active_training.fit", side_effect=fake_fit), \
                patch("neural_active_training.replay", side_effect=lambda state, *a, **k:
                      PlasticityContext.restore(state)):
            with self.assertRaisesRegex(RuntimeError, "Active model changed"):
                round_once(self.root, cpu_seconds=1)
        self.assertEqual(json.loads(active_path.read_text())["note"], "changed concurrently")
        self.assertEqual((self.root / DIRECTORY / "progress.json").read_bytes(), self.old_progress)


if __name__ == "__main__":
    unittest.main()
