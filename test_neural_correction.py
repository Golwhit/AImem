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
from neural_correction import train_correction
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

    def test_failed_training_and_invalid_submission_do_not_promote(self):
        before = (self.root / ".data/neural-training/progress.json").read_bytes()
        for update in ({"acknowledge_review": False}, {"response_id": "missing"},
                       {"correction": "No."}, {"correction": "x\nx"}, {"correction": "a" * 96}):
            with self.assertRaises(ValueError):
                train_correction(self.root, {**self.payload, **update})
        with patch("neural_correction.MAX_UPDATES", 0):
            with self.assertRaisesRegex(RuntimeError, "did not fit"):
                train_correction(self.root, self.payload)
        self.assertEqual((self.root / ".data/neural-training/progress.json").read_bytes(), before)

    def test_saturated_correction_candidate_is_not_promoted(self):
        before = (self.root / ".data/neural-training/progress.json").read_bytes()
        saturated = {"hidden_abs_over_099_fraction": 1.0, "mean_tanh_derivative": 0.0,
                     "mean_hidden_gradient_norm": 0.0}
        with patch("neural_correction.diagnostics", return_value=saturated):
            with self.assertRaisesRegex(RuntimeError, "saturated"):
                train_correction(self.root, self.payload)
        self.assertEqual((self.root / ".data/neural-training/progress.json").read_bytes(), before)

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
