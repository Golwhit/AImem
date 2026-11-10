import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from layer_two import Attraction
from seed_development import DevelopmentRegistry, compose_blocks
from seed_life import DOMAIN, RULES, development_assessment, exclusion_reason


class DevelopmentTests(unittest.TestCase):
    def test_incomplete_completed_and_foundation_are_distinct(self):
        for text in ("t", "am", "I"):
            result = development_assessment(text, at_source_start=True, complete=False)
            self.assertEqual(result["state"], "developing")
        for text in ("t", "am"):
            result = development_assessment(text, at_source_start=True, complete=True)
            self.assertEqual(result["state"], "unreviewed")
            self.assertFalse(result["foundation_eligible"])
        self.assertEqual(development_assessment("I ", at_source_start=True, complete=False)["state"], "confirmed")
        self.assertEqual(development_assessment("t ", at_source_start=True, complete=False)["state"], "excluded")
        self.assertEqual(development_assessment("t ", at_source_start=False, complete=True)["state"], "unreviewed")
        self.assertEqual(development_assessment("I.", at_source_start=False, complete=True)["state"], "unreviewed")
        for invalid in ("code", "science"):
            with self.assertRaises(ValueError):
                development_assessment("aaa", at_source_start=True, complete=True, domain=invalid)
        for rule in RULES:
            self.assertEqual(rule["domain"], DOMAIN)
            self.assertIsNotNone(exclusion_reason(rule["excluded_example"]))
            self.assertIsNone(exclusion_reason(rule["surviving_example"]))

    def test_composition_checks_joins_and_start_placement_before_attraction(self):
        blocks = [
            {"text": "I ", "placement": "start"},
            {"text": " i", "placement": "interior"},
            {"text": " ", "placement": "interior"},
        ]
        result = compose_blocks(blocks, 20, 1, Attraction("I i "))
        self.assertNotEqual(result["text"], "")
        self.assertNotIn(" i ", result["text"])
        self.assertNotIn("   ", result["text"])
        self.assertTrue(any(step["rejected"] for step in result["trace"]))
        self.assertEqual(result, compose_blocks(blocks, 20, 1, Attraction("I i ")))
        self.assertTrue(all(step.get("selected") is None or step["offset"] == 0
                            or step["selected"]["placement"] != "start" for step in result["trace"]))

    def test_review_persists_exact_evidence_and_revocation_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt = "am amount"
            record_id = "a" * 64
            records = root / ".data" / "model-lab"
            records.mkdir(parents=True)
            record = {"prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
            path = records / f"{record_id}.json"
            path.write_text(json.dumps(record))
            payload = {"record_id": record_id, "offset": 0, "size": 2, "action": "admit",
                       "rationale": "Original fixture reviewed for lab composition only", "acknowledge_lab_only": True}
            registry = DevelopmentRegistry(root)
            original_hash = registry.status()["sha256"]
            status = registry.decide(payload)
            event = status["candidates"][0]
            self.assertFalse(event["training_approved"])
            self.assertEqual(event["source_text"], prompt)
            self.assertTrue(all(test["passed"] for test in event["tests"]))
            self.assertEqual(event["text"], "am")
            self.assertEqual(event["placement"], "start")
            self.assertNotEqual(original_hash, status["sha256"])
            restored = DevelopmentRegistry(root)
            self.assertEqual(restored.status(), status)
            revoked = restored.decide({**payload, "action": "revoke", "rationale": "Need more tests"})
            self.assertEqual(revoked["candidates"][0]["action"], "revoke")
            self.assertEqual(len(revoked["events"]), 2)
            self.assertEqual(revoked["events"][0]["action"], "admit")
            self.assertFalse((root / ".data" / "original-generator.json").exists())
            self.assertFalse((root / "chat_memory.json").exists())
            self.assertEqual(json.loads(path.read_text()), record)

    def test_admission_fails_closed_for_bad_evidence_and_exclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = root / ".data" / "model-lab"
            records.mkdir(parents=True)
            record_id = "a" * 64
            path = records / f"{record_id}.json"
            text = "t next"
            path.write_text(json.dumps({"prompt": text, "prompt_sha256": hashlib.sha256(text.encode()).hexdigest()}))
            payload = {"record_id": record_id, "offset": 0, "size": 1, "action": "admit",
                       "rationale": "Fixture", "acknowledge_lab_only": True}
            for override in ({}, {"offset": -1}, {"offset": True}, {"size": 5},
                             {"record_id": "../bad"}, {"acknowledge_lab_only": False}, {"rationale": ""},
                             {"action": "revoke"}, {"offset": 6}):
                with self.subTest(override=override), self.assertRaises(ValueError):
                    DevelopmentRegistry(root).decide({**payload, **override})
            self.assertFalse((root / ".data/seed-development.json").exists())
            deferred = DevelopmentRegistry(root).decide({**payload, "action": "defer"})
            self.assertTrue(any(not test["passed"] for test in deferred["events"][0]["tests"]))
            path.write_text(json.dumps({"prompt": "changed", "prompt_sha256": "bad"}))
            before = (root / ".data/seed-development.json").read_bytes()
            with self.assertRaises(ValueError):
                DevelopmentRegistry(root).decide(payload)
            self.assertEqual(before, (root / ".data/seed-development.json").read_bytes())
