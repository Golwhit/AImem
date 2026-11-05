import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from memory_learner import MemoryLearner, memory_hash


class LearnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.transcript = {"messages": [
            {"id": "food", "role": "You", "text": "apples fruit recipes dinner", "topics": ["food"]},
            {"id": "finance", "role": "You", "text": "bonds savings debt budget", "topics": ["finance"]},
            {"id": "garden", "role": "You", "text": "orchids humidity watering garden", "topics": []},
        ]}
        (self.root / "chat_memory.json").write_text(json.dumps(self.transcript))

    def test_offline_search_and_checkpoint(self):
        with patch("socket.socket", side_effect=AssertionError("Network access forbidden")):
            learner = MemoryLearner(self.root)
            self.assertEqual(learner.search("apples fruit")["results"][0]["id"], "food")
        self.assertEqual(learner.search("xyzunseenword")["results"], [])
        restored = MemoryLearner(self.root)
        self.assertEqual(restored.state["weights"], learner.state["weights"])
        self.assertFalse(restored.status()["pretrained_weights"])

    def test_training_search_returns_only_user_authored_memories(self):
        self.transcript["messages"].append({"id": "assistant", "role": "Assistant", "text": "apples fruit", "topics": []})
        (self.root / "chat_memory.json").write_text(json.dumps(self.transcript))
        learner = MemoryLearner(self.root)
        self.assertTrue(any(item["id"] == "assistant" for item in learner.search("apples")["results"]))
        self.assertTrue(all(item["role"] == "You" for item in learner.search("apples", user_only=True)["results"]))
        self.assertFalse(any(item["id"] == "food" for item in learner.search("apples", exclude_ids=["food"])["results"]))

    def test_copilot_replies_are_searchable_but_not_automatically_trainable(self):
        self.transcript["messages"].append({"id": "copilot", "role": "Assistant", "text": "apples fruit explanation", "source": "copilot", "topics": []})
        (self.root / "chat_memory.json").write_text(json.dumps(self.transcript))
        learner = MemoryLearner(self.root)
        self.assertTrue(any(item["id"] == "copilot" for item in learner.search("apples fruit")["results"]))
        learner.feedback("apples fruit", "copilot", "direct_answer")
        self.assertEqual(learner.status()["eligible_examples"], 0)
        with self.assertRaises(ValueError):
            learner.review(["copilot"], ["apples fruit"], True)

    def test_explicit_feedback_deduplicates_and_does_not_change_transcript(self):
        learner = MemoryLearner(self.root)
        before = (self.root / "chat_memory.json").read_bytes()
        learner.feedback("apples", "food", "direct_answer")
        learner.feedback("APPLES", "food", "unrelated")
        self.assertEqual(len(learner.state["feedback"]), 1)
        self.assertEqual(learner.state["feedback"][0]["label"], "unrelated")
        self.assertEqual(before, (self.root / "chat_memory.json").read_bytes())
        with self.assertRaises(ValueError):
            learner.feedback("apples", "nonexistent", "direct_answer")

    def test_status_exposes_parameters_and_review_progress_without_training(self):
        learner = MemoryLearner(self.root)
        learner.feedback("apples", "food", "direct_answer")
        before = learner.path.read_bytes()
        status = learner.status()
        self.assertEqual(status["generation"], 0)
        self.assertEqual(len(status["weights"]), 4)
        self.assertEqual(status["intercept"], -2.0)
        self.assertEqual(status["pending_review"], 1)
        self.assertEqual(status["eligible_examples"], 0)
        self.assertEqual(status["learning_state"], "awaiting-review")
        self.assertEqual(before, learner.path.read_bytes())

    def test_context_feedback_is_explicit_and_replaces_previous_judgment(self):
        learner = MemoryLearner(self.root)
        learner.feedback("apples", "food", "direct_answer")
        self.transcript["messages"].extend([
            {"id": "answer", "role": "Assistant", "text": "fabricated retrievalword", "source": "local-retrieval"},
            {"id": "generated", "role": "Assistant", "text": "fabricated generatedword", "source": "local-generated"},
        ])
        self.assertEqual(result["target"], 0.5)
        self.assertEqual(learner.state["feedback"][0]["label"], "related_context")
        self.assertNotIn("retrievalword", learner.vocabulary)
        self.assertNotIn("generatedword", learner.vocabulary)
        self.assertEqual(learner.status()["judgment_counts"]["related_context"], 1)
        self.assertNotIn("generated", learner.by_id)
        self.assertEqual(learner.status()["eligible_examples"], 0)
        restored = MemoryLearner(self.root)
        result = restored.search("APPLES", user_only=True)["results"][0]
        self.assertEqual(result["judgment"]["label"], "related_context")
        self.assertFalse(result["judgment"]["reviewed"])
        for invalid in (True, False, "relevant", None, 0.5):
            with self.assertRaises(ValueError):
                learner.feedback("apples", "food", invalid)

    def test_binary_checkpoint_and_ambiguous_judgments_are_preserved(self):
        legacy = self.root / ".data" / "learner-original.json"
        legacy.parent.mkdir()
        content = json.dumps({"version": 2, "generation": 3, "weights": [99] * 4,
                              "feedback": [{"query": "apples", "memory_id": "food", "relevant": True, "memory_sha256": memory_hash(self.transcript["messages"][0])}]})
        legacy.write_text(content)
        learner = MemoryLearner(self.root)
        self.assertEqual(learner.status()["needs_relabel"], 1)
        self.assertEqual(learner.status()["generation"], 0)
        self.assertEqual(learner.status()["eligible_examples"], 0)
        self.assertTrue(learner.state["feedback"][0]["relevant"])
        self.assertIsNone(learner.state["feedback"][0]["label"])
        learner.feedback("apples", "food", "related_context")
        self.assertEqual(legacy.read_text(), content)
        self.assertEqual(len(learner.state["feedback"]), 1)
        self.assertEqual(MemoryLearner(self.root).status()["judgment_counts"]["related_context"], 1)

    def test_three_targets_reach_training_without_context_becoming_zero(self):
        learner = MemoryLearner(self.root)
        queries = [f"teaching question {index}" for index in range(6)]
        learner.review(["food", "finance", "garden"], queries, True)
        learner.state["weights"] = [0.0] * 4
        learner.state["intercept"] = 0.0
        rows = {"food": [1.0] * 4, "finance": [0.5] * 4, "garden": [0.0] * 4}
        with patch.object(learner, "features", side_effect=lambda query, memory_id: rows[memory_id]):
            for query in queries:
                for memory_id, label in [("food", "direct_answer"), ("finance", "related_context"), ("garden", "unrelated")]:
                    learner.feedback(query, memory_id, label)
        self.assertGreater(learner.state["generation"], 0)
        self.assertEqual(learner.status()["eligible_examples"], 18)
        self.assertEqual(learner.status()["judgment_counts"], {"direct_answer": 6, "related_context": 6, "unrelated": 6})
        direct, context, unrelated = [learner.probability(rows[memory_id]) for memory_id in ("food", "finance", "garden")]
        self.assertGreater(direct, context)
        self.assertGreater(context, unrelated)
        self.assertAlmostEqual(context, 0.5, delta=0.08)

    def test_learning_is_query_held_out_and_keeps_rollback(self):
        learner = MemoryLearner(self.root)
        starting_weights = learner.state["weights"].copy()
        learner.review(["food", "finance"], [f"apples fruit context{index}" for index in range(6)], True)
        for index in range(6):
            query = f"apples fruit context{index}"
            learner.feedback(query, "food", "unrelated")
            outcome = learner.feedback(query, "finance", "direct_answer")
        self.assertEqual(outcome["training"]["status"], "accepted")
        self.assertLess(outcome["training"]["candidate_loss"], outcome["training"]["baseline_loss"])
        self.assertTrue((self.root / ".data" / "learner-graded.previous.json").exists())
        validation_queries = set(outcome["training"]["validation_queries"])
        train_queries = {item["query"] for item in learner.state["feedback"] if item["query"] not in validation_queries}
        self.assertFalse(validation_queries & train_queries)
        self.assertNotEqual(learner.state["weights"], starting_weights)

    def test_generated_answers_do_not_teach_the_representation(self):
        self.transcript["messages"].extend([
            {"id": "answer", "role": "Assistant", "text": "fabricated retrievalword", "source": "local-retrieval"},
            {"id": "generated", "role": "Assistant", "text": "fabricated generatedword", "source": "local-generated"},
        ])
        (self.root / "chat_memory.json").write_text(json.dumps(self.transcript))
        learner = MemoryLearner(self.root)
        self.assertNotIn("retrievalword", learner.vocabulary)
        self.assertNotIn("generatedword", learner.vocabulary)
        self.assertNotIn("answer", learner.by_id)
        self.assertNotIn("generated", learner.by_id)

    def test_local_user_prompts_remain_training_input(self):
        self.transcript["messages"].append({"id": "prompt", "role": "You", "text": "telescopes astronomy starlight", "source": "local-retrieval"})
        (self.root / "chat_memory.json").write_text(json.dumps(self.transcript))
        learner = MemoryLearner(self.root)
        learner.review(["prompt"], [], True)
        restored = MemoryLearner(self.root)
        self.assertIn("telescopes", restored.vocabulary)

    def test_unreviewed_examples_cannot_train(self):
        learner = MemoryLearner(self.root)
        for index in range(6):
            learner.feedback(f"apples fruit context{index}", "food", "unrelated")
            result = learner.feedback(f"apples fruit context{index}", "finance", "direct_answer")
        self.assertEqual(result["training"]["examples"], 0)
        self.assertEqual(result["training"]["pending_review"], 12)
        self.assertEqual(learner.state["generation"], 0)

    def test_review_requires_acknowledgment_and_invalidates_changed_content(self):
        learner = MemoryLearner(self.root)
        with self.assertRaises(ValueError):
            learner.review(["food"], ["apples"], False)
        learner.review(["food"], ["apples"], True)
        self.assertTrue(learner.is_reviewed("food"))
        self.transcript["messages"][0]["text"] = "changed memory"
        (self.root / "chat_memory.json").write_text(json.dumps(self.transcript))
        self.assertFalse(MemoryLearner(self.root).is_reviewed("food"))

    def test_legacy_checkpoint_is_preserved_not_reused(self):
        legacy = self.root / ".data" / "learner.json"
        legacy.parent.mkdir()
        original = '{"version":1,"weights":[99,99,99,99]}'
        legacy.write_text(original)
        learner = MemoryLearner(self.root)
        learner.search("apples")
        self.assertEqual(legacy.read_text(), original)
        self.assertNotEqual(learner.state["weights"], [99] * 4)

    def test_rejected_candidate_keeps_current_weights(self):
        learner = MemoryLearner(self.root)
        learner.review(["food", "finance"], [f"apples fruit context{index}" for index in range(6)], True)
        learner.state["feedback"] = [
            {"query": f"apples fruit context{index}", "memory_id": memory_id, "label": label, "memory_sha256": memory_hash(learner.by_id[memory_id])}
            for index in range(6)
            for memory_id, label in [("food", "direct_answer"), ("finance", "unrelated")]
        ]
        original = learner.state["weights"].copy()
        with patch("memory_learner.fit", return_value=([0.0] * 4, 0.0)):
            result = learner.train()
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(learner.state["weights"], original)
        self.assertEqual(learner.state["generation"], 0)

    def test_http_routes_use_local_learner_and_save_excerpts(self):
        project = Path(__file__).resolve().parent
        for script in project.glob("*.py"):
            shutil.copy(script, self.root)
        shutil.copy(project / "package.json", self.root)
        (self.root / "server").mkdir()
        shutil.copy(project / "server" / "index.js", self.root / "server")
        (self.root / "node_modules").symlink_to(project / "node_modules", target_is_directory=True)
        self.transcript.update({"cycles": 0, "seed_id": "food", "links": []})
        self.transcript["messages"][0]["text"] = "apples fruit recipes dinner " * 30
        self.transcript["messages"].append({"id": "assistant", "role": "Assistant", "text": "apples onions pasta cooking vegetables recipes", "topics": [], "source": "copilot"})
        (self.root / "chat_memory.json").write_text(json.dumps(self.transcript))
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        environment = {**os.environ, "PORT": str(port), "MEMORY_PYTHON": sys.executable}
        process = subprocess.Popen(["node", "server/index.js"], cwd=self.root, env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            started = process.stdout.readline()
            self.assertIn("server running", started)
            base = f"http://127.0.0.1:{port}"

            def request(path, payload=None):
                data = json.dumps(payload).encode() if payload is not None else None
                with urlopen(Request(base + path, data=data, headers={"Content-Type": "application/json"}), timeout=30) as response:
                    return json.load(response)

            self.assertFalse(request("/api/learner-status")["pretrained_weights"])
            generator_status = request("/api/generator")
            self.assertFalse(generator_status["pretrained_weights"])
            self.assertEqual(generator_status["document_count"], 0)
            result = request("/api/semantic-search?query=apples")
            self.assertEqual(result["results"][0]["id"], "food")
            self.assertEqual(request("/api/learner-feedback", {"query": "apples", "memory_id": "food", "label": "direct_answer"})["training"]["status"], "collecting")
            context = request("/api/learner-feedback", {"query": "apples", "memory_id": "food", "label": "related_context"})
            self.assertEqual(context["target"], 0.5)
            self.assertEqual(request("/api/learner-status")["judgment_counts"]["related_context"], 1)
            approved = request("/api/learner-feedback", {"query": "apples", "memory_id": "food", "label": "related_context", "approve_for_training": True})
            self.assertEqual(approved["training"]["examples"], 1)
            self.assertEqual(request("/api/learner-status")["approved_memories"], 1)
            with self.assertRaises(HTTPError) as rejected:
                request("/api/learner-feedback", {"query": "apples", "memory_id": "food", "relevant": True})
            self.assertEqual(rejected.exception.code, 400)
            rejected.exception.close()
            chat = request("/api/chat", {"message": "apples", "sessionId": "http-test", "training_mode": True})
            self.assertEqual(chat["mode"], "memory-excerpts")
            self.assertEqual(chat["generator_learning"]["reason"], "opted_out")
            self.assertEqual(request('/api/generator')['document_count'], 0)
            self.assertIn("[assistant]", chat["reply"])
            self.assertEqual(chat["response_source_id"], "assistant")
            self.assertTrue(any(item["role"] == "Assistant" for item in chat["memories"]))
            self.assertLessEqual(len(chat["reply"]), 650)
            self.assertEqual(chat["memories"][0]["judgment"]["label"], "related_context")
            self.assertTrue(chat["prompt_memory_id"])
            self.assertTrue(chat["response_memory_id"])
            self.assertFalse(any(item["id"] == chat["prompt_memory_id"] for item in chat["memories"]))
            refresh = request("/api/semantic-search?query=apples&user_only=true&exclude_id=" + chat["prompt_memory_id"].replace(" ", "%20"))
            self.assertFalse(any(item["id"] == chat["prompt_memory_id"] for item in refresh["results"]))
            response_feedback = request('/api/response-feedback', {"response_id": chat["response_memory_id"], "rating": "needs_correction", "correction": "This should describe the standalone x16 requirement."})
            self.assertFalse(response_feedback["used_for_training"])
            stored_feedback = json.loads((self.root / '.data' / 'response-feedback.json').read_text())
            self.assertFalse(stored_feedback["items"][0]["training_approved"])
            trained = request('/api/generator/documents', {
                'title': 'Approved seed', 'source': 'User-authored sample',
                'rights_note': 'Original text written for this test.',
                'text': 'the moon moves over the water. the moon moves over the water. ' * 6,
                'acknowledge_review': True,
            })
            self.assertEqual(trained['document_count'], 1)
            tracked = request('/api/generator/evaluations', {'prompt': 'the moon'})
            evaluation = tracked['prompts'][0]
            self.assertEqual([run['generation'] for run in evaluation['runs']], [1])
            rated = request(f"/api/generator/evaluations/{evaluation['id']}/1", {
                'rating': 'edit', 'correction': 'The moon travels over water.',
            })
            self.assertEqual(rated['prompts'][0]['runs'][0]['rating'], 'edit')
            generated_chat = request('/api/chat', {
                'message': 'the moon ', 'sessionId': 'http-test', 'include_in_generator': True,
            })
            self.assertEqual(generated_chat['mode'], 'local-character-ngram')
            self.assertTrue(generated_chat['reply'])
            self.assertEqual(generated_chat['training_sources'][0]['title'], 'Approved seed')
            self.assertTrue(generated_chat['generator_learning']['added'])
            generator_status = request('/api/generator')
            self.assertEqual(generator_status['document_count'], 2)
            self.assertEqual(generator_status['documents'][1]['source'], 'Memory chat')
            self.assertEqual(generator_status['documents'][1]['title'], 'Memory chat prompt')
            generated_feedback = request('/api/response-feedback', {
                'response_id': generated_chat['response_memory_id'],
                'rating': 'needs_correction', 'correction': 'An experimental generation, not a fact.',
            })
            self.assertFalse(generated_feedback['used_for_training'])
            defaulted = request('/api/generator/documents', {
                'text': 'A second locally reviewed source for the generator.' * 2,
                'acknowledge_review': True,
            })
            self.assertEqual(defaulted['documents'][2]['title'], 'Untitled source 3')
            self.assertEqual(defaulted['documents'][2]['source'], 'Me')
            evaluations = request('/api/generator/evaluations')
            runs = evaluations['prompts'][0]['runs']
            self.assertEqual([run['generation'] for run in runs], [1, 2, 3])
            self.assertEqual(runs[0]['correction'], 'The moon travels over water.')
            self.assertEqual(request('/api/generator')['evaluation_sample_count'], 3)
            saved = json.loads((self.root / "chat_memory.json").read_text())
            self.assertEqual(len(saved["messages"]), len(self.transcript["messages"]) + 4)
            self.assertEqual(saved["messages"][-1]["source"], "local-generated")
            snapshot = {"nodes": [], "links": [], "padding": "x" * 120000}
            self.assertTrue(request("/api/graph-state", snapshot)["ok"])
            canonical = request("/api/graph-state")["snapshot"]
            self.assertEqual(canonical["cycles"], 2)
            self.assertTrue(any(node["id"] == "food" for node in canonical["nodes"]))
            newest = canonical["nodes"][-1]["id"]
            minimal = {"nodes": [{"id": "food", "feedback": {"helpful": 2}}], "links": [], "cycles": 0}
            self.assertTrue(request("/api/graph-state", minimal)["ok"])
            preserved = request("/api/graph-state")["snapshot"]
            self.assertEqual(preserved["cycles"], 2)
            self.assertTrue(any(node["id"] == newest for node in preserved["nodes"]))
            self.assertTrue(all("domain" in node for node in preserved["nodes"]))
            self.assertEqual(next(node for node in preserved["nodes"] if node["id"] == "food")["feedback"]["helpful"], 2)
        finally:
            process.terminate()
            process.communicate(timeout=10)


if __name__ == "__main__":
    unittest.main()