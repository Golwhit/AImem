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
from neural_conversation_trial import canonical, prompt_for
from neural_plasticity import PlasticityContext, initialize
from neural_training import DIRECTORY, load_progress, rollback
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
    def assess_correct(model, rows):
        measured = [{"id": row["id"], "skill": row["skill"], "pair": row.get("pair"),
                     "expected": row["answer"], "prediction": row["answer"],
                     "exact": True, "normalized_exact": True} for row in rows]
        return {"exact": 1.0, "rows": measured}

    def test_continues_active_weights_keeps_feedback_and_rolls_back(self):
        snapshots = []

        def fake_fit(model, rows, **kwargs):
            self.assertTrue(any(row["id"] == "correction-1" for row in rows))
            model.model.weights["hidden_bias"][0] += .001
            snapshots.append(model.snapshot())
            return [{"updates": 1, "training": {"exact": 1.0},
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
            return [{"updates": 0, "training": {"exact": 0.0},
                     "diagnostics": {"hidden_abs_over_099_fraction": .1}}]

        with patch("neural_active_training.assess", side_effect=self.assess_correct), \
                patch("neural_active_training.fit", side_effect=fake_fit), \
                patch("neural_active_training.replay", side_effect=lambda state, *a, **k:
                      PlasticityContext.restore(state)):
            with self.assertRaisesRegex(RuntimeError, "teaching accuracy regressed"):
                round_once(self.root, cpu_seconds=1)
        self.assertEqual((self.root / DIRECTORY / "progress.json").read_bytes(), self.old_progress)
        self.assertEqual((self.root / ".data/neural-chat/active.json").read_bytes(), self.active_bytes)

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
            return [{"updates": 0, "training": {"exact": 1.0},
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
