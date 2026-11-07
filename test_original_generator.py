import json
import tempfile
import unittest
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