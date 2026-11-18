import json
from pathlib import Path
import tempfile
import unittest

from neural_base_comparison import digest
from neural_character_promotion import select
from neural_chat import CONVERSATION_MODEL, load_active
from neural_conversation_trial import answer_examples, canonical
from neural_plasticity import PlasticityContext
from neural_training import activate, load_progress, rollback
from original_generator import atomic_json


class CharacterPromotionTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        chat = self.root / ".data/neural-chat"
        training = self.root / ".data/neural-training"
        chat.mkdir(parents=True)
        training.mkdir(parents=True)
        (training / "rounds").mkdir()
        (self.root / "universe-lab/experiments").mkdir(parents=True)
        fixture = self.root / "fixture.json"
        fixture.write_text('{"train":[]}')
        self.model = PlasticityContext("user: cat\nassistant: cat.\n", 17)
        evidence = {"run_id": "a" * 32, "round": 1, "fixture": "fixture.json",
                    "reviewed_file_sha256": digest(fixture.read_bytes()), "approved_corrections": []}
        checkpoint = canonical({"state": self.model.snapshot(), "evidence": evidence})
        checkpoint_hash = digest(checkpoint)
        (training / f"{checkpoint_hash}.json").write_bytes(checkpoint)
        (chat / f"{checkpoint_hash}.json").write_bytes(checkpoint)
        atomic_json(training / "progress.json", {**evidence, "checkpoint_sha256": checkpoint_hash})
        atomic_json(chat / "active.json", {"model": CONVERSATION_MODEL, "arm": "conversation",
                                          "checkpoint_sha256": checkpoint_hash, "passes": 1})
        self.old_active = (chat / "active.json").read_bytes()
        self.old_progress = (training / "progress.json").read_bytes()
        self.candidate = self.root / "candidate"
        self.candidate.mkdir()
        (self.candidate / "source").mkdir()
        source = b"example source"
        (self.candidate / "source/example.py").write_bytes(source)
        atomic_json(self.candidate / "trained.json", {"state": self.model.snapshot(), "output_mode": "character_only"})
        candidate_hash = digest((self.candidate / "trained.json").read_bytes())
        self.fresh = self.root / "fresh.json"
        self.previous = self.root / "previous.json"
        self.fresh.write_text('{"fresh":true}')
        self.previous.write_text('{"previous":true}')
        atomic_json(self.candidate / "report.json", {
            "kind": "private_fresh_meal_rehearsal_continuation", "trained_sha256": candidate_hash,
            "validation": {"exact_replay": True},
            "training": {"fresh_meal_sha256": digest(self.fresh.read_bytes()),
                         "previous_meal_sha256": digest(self.previous.read_bytes()),
                         "reviewed_training_sha256": "b" * 64},
            "source_sha256": {"example.py": digest(source)},
        })
        self.report_hash = digest((self.candidate / "report.json").read_bytes())
        atomic_json(self.candidate / "activation-assessment.json", {
            "candidate_report_sha256": self.report_hash, "candidate_checkpoint_sha256": candidate_hash,
            "live_selection_sha256": digest(self.old_active), "live_checkpoint_sha256": checkpoint_hash,
            "progress_sha256": digest(self.old_progress), "reviewed_fixture_sha256": digest(fixture.read_bytes()),
            "validation": {"milestone_replay_exact": True, "protected_unchanged": True},
            "hidden_saturation": {"midpoint": .2, "final": .2, "limit": .6},
            "activation_check_failures": ["previously correct live teaching answers were lost"],
        })
        self.assessment_hash = digest((self.candidate / "activation-assessment.json").read_bytes())

    def select(self, accept=True):
        return select(self.root, candidate_dir=self.candidate, report_sha256=self.report_hash,
                      assessment_sha256=self.assessment_hash, fresh_meal=self.fresh,
                      previous_meal=self.previous, accept_regression=accept)

    def test_explicit_exception_selects_character_mode_and_preserves_rollback(self):
        with self.assertRaisesRegex(ValueError, "acceptance"):
            self.select(False)
        result = self.select()
        active, model, _ = load_active(self.root)
        progress, trained = load_progress(self.root)
        self.assertEqual(active["checkpoint_sha256"], progress["checkpoint_sha256"])
        self.assertEqual(active["output_mode"], "character_only")
        self.assertFalse(model.word_tokens)
        self.assertEqual(model.snapshot(), trained.snapshot())
        row = {"turns": [{"role": "user", "content": "cat"}], "answer": "cat."}
        self.assertTrue(all(target in model.character_tokens for _, target, _ in answer_examples(model, [row])))
        copy = json.loads(Path(result["experimental_copy"]).read_bytes())
        self.assertEqual(copy["state"], model.snapshot())
        self.assertEqual(copy["base_checkpoint_sha256"], active["checkpoint_sha256"])
        self.assertEqual(rollback(self.root)["checkpoint_sha256"], json.loads(self.old_active)["checkpoint_sha256"])

    def test_changed_evidence_is_rejected_without_selecting(self):
        (self.candidate / "activation-assessment.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Hash mismatch"):
            self.select()
        self.assertEqual((self.root / ".data/neural-chat/active.json").read_bytes(), self.old_active)
        self.assertEqual((self.root / ".data/neural-training/progress.json").read_bytes(), self.old_progress)

    def test_selection_revalidates_saved_assessment(self):
        result = self.select()
        Path(result["report"]).with_name("assessment.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "replay-verified"):
            activate(self.root)


if __name__ == "__main__":
    unittest.main()
