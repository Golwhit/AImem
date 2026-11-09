import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import generate_html_graph
from memory_clusters import apply_graph_domains, subject_for_message, update_clusters
from update_chat_memory import add_cycle


class ClusterTests(unittest.TestCase):
    def test_updater_keeps_origin_separate_from_link_source(self):
        transcript = {"cycles": 0, "seed_id": "user", "messages": [
            {"id": "user", "role": "You", "text": "hello", "topics": []},
            {"id": "assistant", "role": "Assistant", "text": "hello", "topics": []},
        ], "links": [{"source": "user", "target": "assistant", "strength": 0.35, "kind": "sequence"}]}
        with tempfile.TemporaryDirectory() as directory:
            memory_path = Path(directory) / "chat_memory.json"
            memory_path.write_text(json.dumps(transcript), encoding="utf-8")
            cycle = add_cycle(memory_path, "memory graph", "memory reply", [], [], "session-test", "copilot")
            saved = json.loads(memory_path.read_text(encoding="utf-8"))
            for message in saved["messages"][-2:]:
                self.assertEqual(message["source"], "copilot")
                self.assertEqual(message["session_id"], "session-test")
                self.assertEqual(message["created_cycle"], cycle)

    def test_memory_discussion_mentions_food_without_food_membership(self):
        self.assertEqual(subject_for_message({"text": "Why is food pulsing in the memory graph?"}), "memory")
        self.assertEqual(subject_for_message({"text": "I like pizza for dinner"}), "food")
        self.assertEqual(subject_for_message({"text": "is there enough to break out ai creation and dev?"}), "memory")

    def test_recent_ai_followups_stay_in_memory_without_assistant_votes(self):
        transcript = {"messages": [
            {"id": "u1", "role": "You", "text": "AI creation and dev", "session_id": "one"},
            {"id": "a1", "role": "Assistant", "text": "Food is an example"},
            {"id": "u2", "role": "You", "text": "sounds good let's do it", "session_id": "one"},
            {"id": "a2", "role": "Assistant", "text": "We are implementing it"},
            {"id": "u3", "role": "You", "text": "so where are we at now?", "session_id": "one"},
        ]}
        update_clusters(transcript)
        self.assertTrue(all(message["parent_domain"] == "memory" for message in transcript["messages"]))
        self.assertTrue(all(message.get("candidate_cluster") is None for message in transcript["messages"]))

    def test_followups_do_not_borrow_another_sessions_subject_or_override_food(self):
        transcript = {"messages": [
            {"id": "u1", "role": "You", "text": "AI development", "session_id": "one"},
            {"id": "u2", "role": "You", "text": "alright go", "session_id": "two"},
            {"id": "u3", "role": "You", "text": "pizza for dinner", "session_id": "one"},
            {"id": "u4", "role": "You", "text": "yes", "session_id": "one"},
        ]}
        update_clusters(transcript)
        self.assertEqual([message["parent_domain"] for message in transcript["messages"]], ["memory", "evaluation", "food", "food"])

    def test_assistant_inherits_user_subject_and_transition_is_not_membership(self):
        transcript = {"messages": [
            {"id": "u1", "role": "You", "text": "pizza for dinner"},
            {"id": "a1", "role": "Assistant", "text": "food is good"},
            {"id": "u2", "role": "You", "text": "How does the memory model grow?"},
            {"id": "a2", "role": "Assistant", "text": "Food is one example"},
        ]}
        update_clusters(transcript)
        self.assertEqual([message["domain"] for message in transcript["messages"]], ["food", "food", "memory", "memory"])

    def test_promotion_requires_independent_sessions_and_keeps_identity(self):
        transcript = {"messages": [
            {"id": f"u{index}", "role": "You", "text": "orchids greenhouse humidity watering", "session_id": "one"}
            for index in range(5)
        ]}
        clusters = update_clusters(transcript)
        cluster_id = next(iter(clusters))
        self.assertFalse(clusters[cluster_id]["promoted"])
        for index, message in enumerate(transcript["messages"]):
            message["session_id"] = str(index % 3)
        clusters = update_clusters(transcript)
        self.assertTrue(clusters[cluster_id]["promoted"])
        self.assertTrue(all(message["domain"] == cluster_id for message in transcript["messages"]))

    def test_assistant_repetition_cannot_promote(self):
        transcript = {"messages": [{"id": "u1", "role": "You", "text": "orchids humidity"}] + [
            {"id": f"a{index}", "role": "Assistant", "text": "orchids humidity", "session_id": str(index)}
            for index in range(10)
        ]}
        self.assertEqual(update_clusters(transcript), {})

    def test_subtopic_can_grow_inside_memory_in_one_sustained_session(self):
        aspects = ["temperature", "humidity", "lighting", "watering", "fertilizer", "repotting", "roots", "flowers"]
        transcript = {"messages": [
            {"id": f"u{index}", "role": "You", "text": f"memory model orchids greenhouse care {aspect}", "session_id": "one"}
            for index, aspect in enumerate(aspects)
        ]}
        clusters = update_clusters(transcript)
        cluster_id = next(iter(clusters))
        self.assertTrue(clusters[cluster_id]["promoted"])
        self.assertEqual(clusters[cluster_id]["parent_domain"], "memory")
        self.assertEqual(clusters[cluster_id]["promotion_path"], "sustained-discussion")
        self.assertTrue(all(message["domain"] == cluster_id and message["parent_domain"] == "memory" for message in transcript["messages"]))

    def test_parent_words_and_acknowledgments_do_not_make_subtopics(self):
        transcript = {"messages": [
            {"id": f"u{index}", "role": "You", "text": "yes good memory graph model test", "session_id": "one"}
            for index in range(12)
        ]}
        self.assertEqual(update_clusters(transcript), {})

    def test_one_recorded_message_cannot_promote_unprovenanced_history(self):
        aspects = ["temperature", "humidity", "lighting", "watering", "fertilizer", "repotting", "roots", "flowers"]
        transcript = {"messages": [
            {"id": f"u{index}", "role": "You", "text": f"memory model orchids greenhouse care {aspect}"}
            for index, aspect in enumerate(aspects)
        ]}
        transcript["messages"][-1]["session_id"] = "one"
        clusters = update_clusters(transcript)
        self.assertFalse(next(iter(clusters.values()))["promoted"])

    def test_transition_link_does_not_change_subject_membership(self):
        transcript = {"links": [{"source": "food", "target": "memory", "kind": "sequence"}], "messages": []}
        graph = {"nodes": [
            {"id": "food", "kind": "message", "domain": "food"},
            {"id": "memory", "kind": "message", "domain": "memory"},
        ], "links": [{"source": "food", "target": "memory"}]}
        apply_graph_domains(graph, transcript)
        self.assertEqual(graph["links"][0]["kind"], "transition")
        self.assertEqual(graph["nodes"][1]["domain"], "memory")

    def test_growing_cluster_keeps_promotion_and_label(self):
        transcript = {"messages": [
            {"id": f"u{index}", "role": "You", "text": "orchids greenhouse humidity watering", "session_id": str(index % 3)}
            for index in range(5)
        ]}
        clusters = update_clusters(transcript)
        cluster_id = next(iter(clusters))
        label = clusters[cluster_id]["label"]
        transcript["messages"].extend([
            {"id": f"u{index}", "role": "You", "text": "orchids greenhouse humidity watering", "session_id": str(index % 3)}
            for index in range(5, 20)
        ])
        clusters = update_clusters(transcript)
        self.assertTrue(clusters[cluster_id]["promoted"])
        self.assertEqual(clusters[cluster_id]["label"], label)

    def test_generator_persists_promotion_and_memory_pulse_metadata(self):
        messages = [
            {"id": f"u{index}", "role": "You", "text": "orchids greenhouse humidity watering", "topics": [], "session_id": str(index % 3)}
            for index in range(5)
        ]
        messages.extend([
            {"id": "food", "role": "You", "text": "pizza for dinner", "topics": []},
            {"id": "memory", "role": "You", "text": "How does the memory graph grow?", "topics": [], "created_cycle": 1},
            {"id": "reply", "role": "Assistant", "text": "Food promotion is an example", "topics": [], "created_cycle": 1},
        ])
        transcript = {"messages": messages, "cycles": 1, "seed_id": "u0", "links": [
            {"source": "food", "target": "memory", "kind": "sequence", "strength": 0.35},
        ]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            memory_path = root / "chat_memory.json"
            memory_path.write_text(json.dumps(transcript), encoding="utf-8")
            with patch.object(generate_html_graph, "__file__", str(root / "generate_html_graph.py")):
                with contextlib.redirect_stdout(io.StringIO()):
                    generate_html_graph.main()
            snapshot = json.loads((root / "memory_graph_state.json").read_text(encoding="utf-8"))
            cluster_id = next(iter(snapshot["clusters"]))
            self.assertTrue(snapshot["clusters"][cluster_id]["promoted"])
            self.assertTrue(any(node.get("domain") == cluster_id for node in snapshot["nodes"]))
            latest = [node for node in snapshot["nodes"] if node.get("created_cycle") == 1]
            self.assertEqual({node["domain"] for node in latest}, {"memory"})
            self.assertTrue(any(link["kind"] == "transition" for link in snapshot["links"]))
            self.assertIn(cluster_id, (root / "memory_graph.html").read_text(encoding="utf-8"))
            persisted = json.loads(memory_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["cycles"], 1)
            self.assertEqual(len(persisted["messages"]), len(messages))
            self.assertIn(cluster_id, persisted["clusters"])
            self.assertFalse(any(node["id"] == "parent-domain-evaluation" for node in snapshot["nodes"]))
            hubs = [node for node in snapshot["nodes"] if node["id"] == cluster_id + "-hub"]
            self.assertTrue(all(isinstance(node["feedback"], dict) for node in hubs))

    def test_promoted_evaluation_galaxy_keeps_memory_pairs_without_parent_tether(self):
        messages = []
        for index in range(5):
            messages.extend([
                {"id": f"u{index}", "role": "You", "text": "orchids greenhouse humidity watering", "topics": [], "session_id": str(index % 3)},
                {"id": f"a{index}", "role": "Assistant", "text": "Related reply", "topics": []},
            ])
        messages.extend([
            {"id": "evaluation", "role": "You", "text": "unclassified"},
            {"id": "food", "role": "You", "text": "pizza dinner"},
        ])
        original_links = [
            {"source": "u0", "target": "evaluation", "kind": "similarity"},
            {"source": "evaluation", "target": "a0", "kind": "sequence"},
            {"source": "u0", "target": "food", "kind": "similarity"},
        ]
        transcript = {"messages": messages, "links": original_links.copy()}
        for _ in range(2):
            clusters = update_clusters(transcript)
            cluster_id = next(iter(clusters))
            self.assertTrue(clusters[cluster_id]["promoted"])
            self.assertTrue(all(message["domain"] == cluster_id and message["parent_domain"] == cluster_id for message in messages[:10]))
            graph = {"nodes": [
                {"id": message["id"], "kind": "message", "domain": message["domain"]}
                for message in messages
            ], "links": [dict(link) for link in original_links]}
            apply_graph_domains(graph, transcript)
            self.assertTrue(any(link["source"] == "u0" and link["target"] == "food" for link in graph["links"]))
            for original in original_links:
                self.assertTrue(any(
                    link["source"] == original["source"] and link["target"] == original["target"]
                    for link in graph["links"]
                ))
            self.assertFalse(any(node["id"] == "parent-domain-evaluation" for node in graph["nodes"]))
            self.assertEqual(sum(link["kind"] == "hub" for link in graph["links"]), 10)
            self.assertFalse(any(link["kind"] == "parent" for link in graph["links"]))
            self.assertEqual(transcript["links"], original_links)

    def test_promoted_subtopic_keeps_non_evaluation_parent(self):
        transcript = {"messages": [
            {"id": f"u{index}", "role": "You", "text": "memory orchids greenhouse watering", "session_id": str(index % 3)}
            for index in range(5)
        ]}
        clusters = update_clusters(transcript)
        cluster_id = next(iter(clusters))
        graph = {"nodes": [
            {"id": message["id"], "kind": "message", "domain": message["domain"]}
            for message in transcript["messages"]
        ], "links": []}
        apply_graph_domains(graph, transcript)
        self.assertIn(
            {"source": cluster_id + "-hub", "target": "parent-domain-memory", "kind": "parent", "strength": 0.5},
            graph["links"],
        )


if __name__ == "__main__":
    unittest.main()