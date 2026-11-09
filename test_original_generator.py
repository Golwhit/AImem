import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from original_generator import CharacterGenerator


class CharacterGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_only_reviewed_documents_train_and_generation_persists(self):
        (self.root / "chat_memory.json").write_text(json.dumps({"messages": [
            {"id": "old", "role": "You", "text": "unapproved historical text"}
        ]}))
        generator = CharacterGenerator(self.root)
        self.assertEqual(generator.status()["document_count"], 0)
        with self.assertRaises(ValueError):
            generator.add_document(
                title="Sample", source="user text", text="A small seed corpus for generation.",
                acknowledge_review=False,
            )

        added = generator.add_document(
            title="Sample", source="user text", text="the slow moon moves over the quiet water. " * 4,
            acknowledge_review=True,
        )
        self.assertEqual(added["document_count"], 1)
        self.assertTrue(added["documents"][0]["sha256"])
        self.assertTrue(added["documents"][0]["rights_reviewed"])

        generated = generator.generate("the ", max_chars=80, seed=7)
        self.assertTrue(generated["text"])
        self.assertEqual(generated["model"], "original-character-ngram-v1")
        self.assertEqual(generated["generation"], 1)
        self.assertEqual(generated["training_sources"][0]["title"], "Sample")

        restored = CharacterGenerator(self.root)
        self.assertEqual(restored.status()["document_count"], 1)
        self.assertEqual(restored.generate("the ", max_chars=80, seed=7)["text"], generated["text"])

    def test_removing_document_rebuilds_empty_model(self):
        generator = CharacterGenerator(self.root)
        added = generator.add_document(
            text="A short temporary corpus for the generator.",
            acknowledge_review=True,
        )
        self.assertEqual(added["documents"][0]["title"], "Untitled source 1")
        self.assertEqual(added["documents"][0]["source"], "Me")
        removed = generator.remove_document(added["documents"][0]["id"])
        self.assertEqual(removed["document_count"], 0)
        self.assertFalse(generator.generate("hello", max_chars=20, seed=1)["ready"])

    def test_seed_extraction_runs_alongside_without_filtering_training(self):
        text = "I keep a note.  i iii\r\n"
        generator = CharacterGenerator(self.root)
        generator.add_document(text=text, acknowledge_review=True)
        document = generator.state["documents"][0]
        self.assertEqual(document["text"], "I keep a note. i iii")
        self.assertEqual(generator.state["transitions"], generator._rebuild(
            [{key: value for key, value in document.items() if key != "seed_extraction"}], 1,
        )["transitions"])
        queue = json.loads((self.root / ".data" / "seed-training-review.json").read_text())
        record = queue["sources"][0]
        self.assertEqual(record["text"], text)
        self.assertEqual(record["provenance"]["document_id"], document["id"])
        self.assertEqual(record["provenance"]["training_sha256"], document["sha256"])
        self.assertNotEqual(record["sha256"], document["sha256"])
        self.assertIn("iii", [item["text"] for item in record["dead"]])
        self.assertNotIn("iii", [item["text"] for item in record["review"]])
        self.assertTrue(any(item["text"] == "I " and item["offsets"] == [0] for item in record["alive"]))
        status = CharacterGenerator(self.root).status()["seed_extraction"]
        self.assertFalse(status["restricts_training"])
        self.assertEqual(status["document_count"], 1)
        self.assertGreater(status["counts"]["maybe"], 0)
        generator.remove_document(document["id"])
        self.assertEqual(generator.status()["seed_extraction"]["document_count"], 0)
        self.assertTrue((self.root / ".data" / "seed-training-review.json").exists())

    def test_unapproved_or_duplicate_text_does_not_extract(self):
        generator = CharacterGenerator(self.root)
        queue = self.root / ".data" / "seed-training-review.json"
        with self.assertRaises(ValueError):
            generator.add_document(text="I keep a note.", acknowledge_review=False)
        self.assertFalse(queue.exists())
        generator.add_document(text="I keep a note.", acknowledge_review=True)
        before = queue.read_bytes()
        with self.assertRaises(ValueError):
            generator.add_document(text="I keep a note.", acknowledge_review=True)
        self.assertEqual(before, queue.read_bytes())

    def test_extraction_failure_does_not_change_model(self):
        generator = CharacterGenerator(self.root)
        with patch("original_generator.extract_text", side_effect=OSError("queue unwritable")):
            with self.assertRaisesRegex(OSError, "queue unwritable"):
                generator.add_document(text="I keep a note.", acknowledge_review=True)
        self.assertFalse(generator.path.exists())
        self.assertEqual(generator.state["generation"], 0)

    def test_old_checkpoints_remain_unprocessed_until_new_submission(self):
        generator = CharacterGenerator(self.root)
        generator.add_document(text="An older approved document.", acknowledge_review=True)
        del generator.state["documents"][0]["seed_extraction"]
        generator._save(generator.state)
        queue = self.root / ".data" / "seed-training-review.json"
        before = queue.read_bytes()
        restored = CharacterGenerator(self.root)
        self.assertEqual(restored.status()["seed_extraction"]["unprocessed_document_count"], 1)
        self.assertEqual(queue.read_bytes(), before)
        restored.ingest_chat_prompt(" I keep a new note.  ")
        records = json.loads(queue.read_text())["sources"]
        self.assertEqual(records[-1]["text"], " I keep a new note.  ")
        self.assertEqual(restored.status()["seed_extraction"]["document_count"], 1)

    def test_chat_prompt_ingestion_is_opt_in_idempotent_and_provenanced(self):
        generator = CharacterGenerator(self.root)
        too_short = generator.ingest_chat_prompt("hey")
        self.assertFalse(too_short["added"])
        self.assertEqual(generator.status()["document_count"], 0)

        added = generator.ingest_chat_prompt("A user's question for the local memory chat.")
        self.assertTrue(added["added"])
        self.assertEqual(added["generation"], 1)
        document = generator.status()["documents"][0]
        self.assertEqual(document["source"], "Memory chat")
        self.assertIn("opted in", document["rights_note"])

        duplicate = generator.ingest_chat_prompt(" A user's question for the local memory chat. ")
        self.assertFalse(duplicate["added"])
        self.assertEqual(duplicate["generation"], 1)
        self.assertEqual(generator.status()["document_count"], 1)

    def test_terminal_only_prompt_suffix_falls_back_to_corpus_start(self):
        generator = CharacterGenerator(self.root)
        generator.add_document(
            title="Training text", source="test", text="Hello from training?",
            acknowledge_review=True,
        )
        result = generator.generate("How can I tell if you are actually training?", max_chars=32, seed=4)
        self.assertGreater(len(result["text"]), 4)
        self.assertFalse(result["conditioned_on_prompt"])

    def test_fixed_prompts_capture_generations_and_feedback_stays_out_of_training(self):
        generator = CharacterGenerator(self.root)
        tracked = generator.track_prompt("the moon")
        self.assertEqual(len(tracked["prompts"]), 1)
        self.assertEqual(tracked["prompts"][0]["runs"], [])

        added = generator.add_document(
            title="Seed", text="the moon moves over the quiet water. " * 4,
            acknowledge_review=True,
        )
        prompt = added["documents"][0]
        evaluation = generator.evaluation_status()["prompts"][0]
        self.assertEqual(len(evaluation["runs"]), 1)
        first_run = evaluation["runs"][0]
        self.assertEqual(first_run["generation"], 1)
        self.assertTrue(first_run["text"])
        self.assertEqual(first_run["training_sources"][0]["id"], prompt["id"])

        rated = generator.rate_evaluation(evaluation["id"], 1, "edit", "The moon moves quietly.")
        self.assertEqual(rated["prompts"][0]["runs"][0]["rating"], "edit")
        self.assertEqual(generator.status()["document_count"], 1)

        generator.add_document(
            title="Second seed", text="bright stars move over the quiet hills. " * 4,
            acknowledge_review=True,
        )
        runs = generator.evaluation_status()["prompts"][0]["runs"]
        self.assertEqual([run["generation"] for run in runs], [1, 2])
        self.assertEqual(runs[0]["correction"], "The moon moves quietly.")
        restored = CharacterGenerator(self.root)
        self.assertEqual(len(restored.evaluation_status()["prompts"][0]["runs"]), 2)

        with self.assertRaises(ValueError):
            generator.rate_evaluation(evaluation["id"], 2, "edit")


if __name__ == "__main__":
    unittest.main()