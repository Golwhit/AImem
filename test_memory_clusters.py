import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import generate_html_graph
from memory_clusters import apply_graph_domains, subject_for_message, update_clusters
from update_chat_memory import add_cycle


class ClusterTests(unittest.TestCase):
    def test_border_neighbors_follow_strongest_candidate_without_other_galaxy_links(self):
        messages = [
            {"id": "a1", "role": "You", "text": "memory alpha target one"},
            {"id": "a2", "role": "You", "text": "memory alpha target two"},
            {"id": "b1", "role": "You", "text": "memory beta target one"},
            {"id": "b2", "role": "You", "text": "memory beta target two"},
            {"id": "shared", "role": "You", "text": "memory unrelated shared"},
            {"id": "cross", "role": "You", "text": "memory unrelated cross"},
            {"id": "tied", "role": "You", "text": "memory unrelated tied"},
            {"id": "zero", "role": "You", "text": "memory unrelated zero"},
            {"id": "food", "role": "You", "text": "pizza dinner"},
        ]
        edges = [
            ("a1", "shared", 0.3), ("b1", "shared", 0.8),
            ("a1", "cross", 0.9), ("cross", "food", 0.1),
            ("a1", "tied", 0.5), ("b1", "tied", 0.5),
            ("a1", "zero", 0.0),
        ]
        transcript = {"messages": messages, "links": [
            {"source": first, "target": second, "strength": strength, "kind": "similarity"}
            for first, second, strength in edges
        ]}
        similarity = [
            {"source": "a1", "target": "a2", "similarity": 0.8},
            {"source": "b1", "target": "b2", "similarity": 0.8},
        ]
        original_links = [link.copy() for link in transcript["links"]]
        with patch("memory_clusters.infer_similarity_links", return_value=similarity):
            clusters = update_clusters(transcript)
        nodes = {message["id"]: message for message in messages}
        target = nodes["b1"]["candidate_cluster"]
        self.assertEqual(nodes["shared"]["border_cluster"], target)
        self.assertEqual(nodes["shared"]["domain"], "memory")
        self.assertEqual(clusters[target]["border_members"], ["shared"])
        self.assertEqual(clusters[target]["interactions"], 2)
        self.assertTrue(all("border_cluster" not in nodes[key] for key in ("cross", "tied", "zero")))
        self.assertEqual(transcript["links"], original_links)
        graph_links = [
            {"source": "shared", "target": "food", "strength": 0.4, "kind": "similarity"},
        ]
        with patch("memory_clusters.infer_similarity_links", return_value=similarity):
            update_clusters(transcript, graph_links)
        self.assertNotIn("border_cluster", nodes["shared"])

    def test_promoted_border_members_keep_parent_and_survive_recomputation(self):
        messages = [
            {"id": f"u{i}", "role": "You", "text": f"memory orchids greenhouse watering aspect{i}"}
            for i in range(5)
        ]
        messages.append({"id": "border", "role": "You", "text": "memory unrelated supporting note"})
        links = [{"source": "u0", "target": "border", "strength": 0.7, "kind": "similarity"}]
        similarities = [
            {"source": f"u{i}", "target": f"u{j}", "similarity": 0.9}
            for i in range(5) for j in range(i + 1, 5)
        ]
        transcript = {"messages": messages, "links": links}
        with patch("memory_clusters.infer_similarity_links", return_value=similarities):
            first = update_clusters(transcript)
            cluster_id = next(iter(first))
            self.assertTrue(first[cluster_id]["promoted"])
            self.assertEqual(first[cluster_id]["interactions"], 5)
            self.assertEqual(messages[-1]["domain"], cluster_id)
            self.assertEqual(messages[-1]["parent_domain"], "memory")
            self.assertEqual(messages[-1]["border_cluster"], cluster_id)
            links[0]["strength"] = 0
            second = update_clusters(transcript)
        self.assertEqual(second[cluster_id]["border_members"], ["border"])
        self.assertEqual(messages[-1]["domain"], cluster_id)
        self.assertEqual(messages[-1]["parent_domain"], "memory")

    def test_existing_promoted_galaxy_link_excludes_new_border_memory(self):
        messages = [
            {"id": f"old{i}", "role": "You", "text": f"memory orchids greenhouse watering aspect{i}"}
            for i in range(5)
        ] + [
            {"id": "new1", "role": "You", "text": "memory engine training alpha"},
            {"id": "new2", "role": "You", "text": "memory engine training beta"},
            {"id": "neighbor", "role": "You", "text": "memory unrelated note"},
        ]
        pairs = [
            {"source": f"old{i}", "target": f"old{j}", "similarity": 0.9}
            for i in range(5) for j in range(i + 1, 5)
        ] + [{"source": "new1", "target": "new2", "similarity": 0.9}]
        transcript = {"messages": messages, "links": []}
        with patch("memory_clusters.infer_similarity_links", return_value=pairs):
            update_clusters(transcript)
            transcript["links"] = [
                {"source": "new1", "target": "neighbor", "strength": 0.9, "kind": "similarity"},
                {"source": "old0", "target": "neighbor", "strength": 0.5, "kind": "similarity"},
            ]
            update_clusters(transcript)
        self.assertNotIn("border_cluster", messages[-1])
        self.assertEqual(messages[-1]["domain"], "memory")

    def test_preview_border_moves_when_candidate_promotes(self):
        messages = [
            {"id": f"u{i}", "role": "You", "text": f"memory orchids greenhouse watering aspect{i}"}
            for i in range(5)
        ]
        messages.append({"id": "border", "role": "You", "text": "memory unrelated note"})
        links = [{"source": "u0", "target": "border", "strength": 0.7, "kind": "similarity"}]
        transcript = {"messages": messages, "links": links}
        pairs = [
            {"source": f"u{i}", "target": f"u{j}", "similarity": 0.9}
            for i in range(5) for j in range(i + 1, 5)
        ]
        with patch("memory_clusters.infer_similarity_links", return_value=pairs[:3]):
            pending = update_clusters(transcript)
        cluster_id = next(iter(pending))
        self.assertFalse(pending[cluster_id]["promoted"])
        self.assertEqual(messages[-1]["border_cluster"], cluster_id)
        self.assertEqual(messages[-1]["domain"], "memory")
        with patch("memory_clusters.infer_similarity_links", return_value=pairs):
            promoted = update_clusters(transcript)
        self.assertTrue(promoted[cluster_id]["promoted"])
        self.assertEqual(messages[-1]["domain"], cluster_id)
        self.assertEqual(messages[-1]["parent_domain"], "memory")

    def test_candidate_previews_avoid_existing_and_other_galaxies(self):
        body = generate_html_graph.HTML.split("function placeCandidatePreviews(", 1)[1].split("\n}", 1)[0]
        script = "function placeCandidatePreviews(" + body + "\n}\n" + """
const assert = require("node:assert/strict");
const names = Array.from({length: 20}, (_, i) => `candidate-${i}`);
const layouts = new Map([["memory", {radius: 170}], ["other", {radius: 100}],
  ...names.map(id => [id, {radius: 34}])]);
const offsets = new Map([["memory", {x: 0, y: 0}], ["other", {x: 0, y: -320}]]);
placeCandidatePreviews(names, layouts, offsets, 170);
assert.equal(offsets.size, 22);
for (const name of names) {
  for (const [other, point] of offsets) {
    if (name === other) continue;
    const here = offsets.get(name);
    const gap = other === "memory" ? 5 : 8;
    assert.ok(Math.hypot(here.x - point.x, here.y - point.y) >= layouts.get(name).radius + layouts.get(other).radius + gap - 1e-9);
  }
}
const soloLayouts = new Map([["memory", {radius: 170}], ["candidate", {radius: 34}]]);
const soloOffsets = new Map([["memory", {x: 0, y: 0}]]);
placeCandidatePreviews(["candidate"], soloLayouts, soloOffsets, 170);
assert.ok(Math.abs(Math.hypot(soloOffsets.get("candidate").x, soloOffsets.get("candidate").y) - 209) < 1e-9);
"""
        subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)

    def test_candidate_preview_keeps_parent_membership_and_pairs_replies(self):
        body = generate_html_graph.HTML.split("function candidatePreviewLayouts(", 1)[1].split("\n}", 1)[0]
        script = "function candidatePreviewLayouts(" + body + "\n}\n" + """
const assert = require("node:assert/strict");
const nodes = [
  {id: "a", kind: "message", role: "You", domain: "memory", candidate_cluster: "pending"},
  {id: "b", kind: "message", role: "Assistant", domain: "memory"},
  {id: "topic", kind: "topic", domain: "memory"},
  {id: "c", kind: "message", role: "You", domain: "memory", candidate_cluster: "promoted"},
  {id: "d", kind: "message", role: "Assistant", domain: "memory"},
  {id: "edge", kind: "message", role: "You", domain: "memory", border_cluster: "pending"}
];
const clusters = {pending: {promoted: false, parent_domain: "memory"}, promoted: {promoted: true, parent_domain: "memory"}};
const result = candidatePreviewLayouts(nodes, clusters);
assert.deepEqual([...result.entries()], [["a", "pending"], ["b", "pending"], ["edge", "pending"]]);
assert.equal(nodes[0].domain, "memory");
assert.equal(nodes[1].domain, "memory");
"""
        subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)

    def test_candidate_evidence_combines_similarity_with_current_links(self):
        messages = [
            {"id": f"u{i}", "role": "You", "text": f"memory orchids greenhouse watering aspect{i}"}
            for i in range(5)
        ]
        links = [
            {"source": f"u{i}", "target": f"u{j}", "kind": "similarity", "strength": 1.0}
            for i in range(5) for j in range(i + 1, 5)
        ]
        transcript = {"messages": messages, "links": links}
        strong = update_clusters(transcript, links)
        self.assertTrue(next(iter(strong.values()))["promoted"])
        fresh = {"messages": [dict(m) for m in messages], "links": links}
        weak = update_clusters(fresh, [{**link, "strength": 0.0} for link in links])
        self.assertLess(next(iter(weak.values()))["cohesion"], next(iter(strong.values()))["cohesion"])

    def test_memory_sizes_compare_strength_and_connections_per_galaxy(self):
        names = ("relativeRanks", "memoryRadii")
        script = "function learnedScore(node) { return node.score; }\n"
        for name in names:
            body = generate_html_graph.HTML.split(f"function {name}(", 1)[1].split("\n}", 1)[0]
            script += f"function {name}(" + body + "\n}\n"
        script += """
const assert = require("node:assert/strict");
const nodes = [
  {id: "a", layoutKey: "memory", score: 0.1},
  {id: "b", layoutKey: "memory", score: 0.5},
  {id: "c", layoutKey: "memory", score: 2},
  {id: "x", layoutKey: "food", score: 100},
  {id: "t1", layoutKey: "other", score: 0.4},
  {id: "t2", layoutKey: "other", score: 0.4}
];
const links = [
  {source: "a", target: "x", strength: 0.2, kind: "sequence"},
  {source: "b", target: "x", strength: 0.5, kind: "related"},
  {source: "c", target: "x", strength: 1, kind: "related"},
  {source: "a", target: "c", strength: 100, kind: "parent"},
  {source: "a", target: "a", strength: 100, kind: "related"}
];
const radii = memoryRadii(nodes, links);
assert.ok(Math.abs(radii.get("a") - 2.8) < 1e-12);
assert.ok(Math.abs(radii.get("b") - 5.15) < 1e-12);
assert.ok(Math.abs(radii.get("c") - 7.5) < 1e-12);
assert.ok(Math.abs(radii.get("x") - 5.15) < 1e-12);
assert.equal(radii.get("t1"), radii.get("t2"));
const sameScores = nodes.slice(0, 3).map(node => ({...node, score: 0.5}));
const connectionRadii = memoryRadii(sameScores, links);
assert.ok(connectionRadii.get("a") < connectionRadii.get("b"));
assert.ok(connectionRadii.get("b") < connectionRadii.get("c"));
const noLinks = memoryRadii(nodes.slice(0, 3), []);
assert.ok(noLinks.get("a") < noLinks.get("b") && noLinks.get("b") < noLinks.get("c"));
const consolidated = memoryRadii([{...nodes[0], consolidated: true}], []);
assert.ok(Math.abs(consolidated.get("a") - radii.get("x") - 0.75) < 1e-12);
assert.equal(memoryRadii([], []).size, 0);
"""
        subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)

    def test_ai_memory_brightness_is_subtle_and_preserves_highlights(self):
        body = generate_html_graph.HTML.split("function memoryBrightness(", 1)[1].split("\n}", 1)[0]
        script = "function memoryBrightness(" + body + "\n}\n" + """
const assert = require("node:assert/strict");
assert.equal(memoryBrightness("memory", 1, false), 1);
assert.equal(memoryBrightness("memory", 0, false), 0.8);
assert.ok(memoryBrightness("memory", 0.25, false) < memoryBrightness("memory", 0.75, false));
assert.ok(Math.abs(memoryBrightness("memory", 0.5, false) - 0.9) < 1e-12);
assert.equal(memoryBrightness("food", 0, false), 1);
assert.equal(memoryBrightness("memory", 0, true), 1);
assert.equal(memoryBrightness("memory", -1, false), 0.8);
assert.equal(memoryBrightness("memory", 2, false), 1);
"""
        subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)

    def test_strength_brightness_does_not_use_memory_age(self):
        names = ("defaultFeedback", "importance", "learnedScore", "memoryBrightness")
        script = ""
        for name in names:
            body = generate_html_graph.HTML.split(f"function {name}(", 1)[1].split("\n}", 1)[0]
            script += f"function {name}(" + body + "\n}\n"
        script += """
const assert = require("node:assert/strict");
const old = {rawImportance: 0.3, history: [], feedback: defaultFeedback(), kind: "message", timestamp: 1, created_cycle: 1};
const recent = {...old, timestamp: 999999, created_cycle: 999};
assert.equal(learnedScore(old), learnedScore(recent));
assert.equal(memoryBrightness("memory", learnedScore(old), false), memoryBrightness("memory", learnedScore(recent), false));
"""
        subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
        highlighted = generate_html_graph.HTML.split("const highlighted = ", 1)[1].split(";", 1)[0]
        self.assertNotIn("isCreated", highlighted)

    def test_connection_presentation_preserves_cross_galaxy_layering(self):
        body = generate_html_graph.HTML.split("function connectionPresentation(", 1)[1].split("\n}", 1)[0]
        script = "function connectionPresentation(" + body + "\n}\n" + """
const assert = require("node:assert/strict");
const within = connectionPresentation("food", "food", "sequence", 0.5, false, false);
assert.equal(within.withinGalaxy, true);
assert.equal(within.drawAbove, true);
assert.ok(Math.abs(within.width - 0.95) < 1e-12);
assert.ok(Math.abs(within.opacity - 0.33) < 1e-12);
assert.equal(connectionPresentation("memory", "memory", "sequence", 0, false, false).opacity, 0);
assert.equal(connectionPresentation("memory", "memory", "sequence", -1, false, false).opacity, 0);
assert.equal(connectionPresentation("memory", "memory", "sequence", 2, false, false).opacity, 0.66);
assert.ok(connectionPresentation("food", "food", "topic", 0.2, false, false).opacity
  < connectionPresentation("food", "food", "topic", 0.8, false, false).opacity);
assert.equal(connectionPresentation("food", "food", "sequence", 0, true, false).opacity, null);
assert.equal(connectionPresentation("food", "food", "sequence", 0, false, true).opacity, null);
const across = connectionPresentation("food", "memory", "sequence", 0.5, false, false);
assert.equal(across.withinGalaxy, false);
assert.equal(across.drawAbove, false);
assert.equal(across.width, 0.75);
assert.equal(across.opacity, null);
assert.equal(connectionPresentation("food", "memory", "sequence", 0.5, true, false).drawAbove, true);
assert.equal(connectionPresentation("food", "memory", "sequence", 0.5, false, true).drawAbove, true);
const parent = connectionPresentation("food", "food", "parent", 0.5, false, false);
assert.equal(parent.withinGalaxy, false);
assert.equal(parent.drawAbove, false);
assert.equal(parent.width, 2);
assert.equal(parent.opacity, null);
assert.equal(connectionPresentation(undefined, undefined, "sequence", 0.5, false, false).withinGalaxy, false);
"""
        subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)

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

    def test_promotion_uses_distinct_contributions_not_sessions(self):
        transcript = {"messages": [
            {"id": f"u{index}", "role": "You", "text": "orchids greenhouse humidity watering", "session_id": "one"}
            for index in range(5)
        ]}
        clusters = update_clusters(transcript)
        cluster_id = next(iter(clusters))
        self.assertFalse(clusters[cluster_id]["promoted"])
        for index, message in enumerate(transcript["messages"]):
            message["text"] += f" aspect{index}"
            message.pop("session_id")
        clusters = update_clusters(transcript)
        self.assertTrue(clusters[cluster_id]["promoted"])
        self.assertTrue(all(message["domain"] == cluster_id for message in transcript["messages"]))

    def test_promoted_galaxy_never_returns_to_evaluation(self):
        transcript = {"messages": [
            {"id": f"u{index}", "role": "You", "text": f"orchids greenhouse humidity watering aspect{index}", "session_id": str(index % 3)}
            for index in range(5)
        ]}
        cluster_id = next(iter(update_clusters(transcript)))
        self.assertTrue(transcript["clusters"][cluster_id]["promoted"])
        unrelated = ["granite", "ferry", "saxophone", "walrus", "lantern"]
        for message, word in zip(transcript["messages"], unrelated):
            message["text"] = f"{word} {word}x quartz vessel"
        clusters = update_clusters(transcript)
        self.assertTrue(clusters[cluster_id]["promoted"])
        self.assertTrue(all(message.get("candidate_cluster") == cluster_id for message in transcript["messages"]))

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
        self.assertEqual(clusters[cluster_id]["promotion_path"], "distinct-discussion")
        self.assertTrue(all(message["domain"] == cluster_id and message["parent_domain"] == "memory" for message in transcript["messages"]))

    def test_parent_words_and_acknowledgments_do_not_make_subtopics(self):
        transcript = {"messages": [
            {"id": f"u{index}", "role": "You", "text": "yes good memory graph model test", "session_id": "one"}
            for index in range(12)
        ]}
        self.assertEqual(update_clusters(transcript), {})

    def test_promotion_does_not_depend_on_session_provenance(self):
        aspects = ["temperature", "humidity", "lighting", "watering", "fertilizer", "repotting", "roots", "flowers"]
        transcript = {"messages": [
            {"id": f"u{index}", "role": "You", "text": f"memory model orchids greenhouse care {aspect}"}
            for index, aspect in enumerate(aspects)
        ]}
        transcript["messages"][-1]["session_id"] = "one"
        clusters = update_clusters(transcript)
        self.assertTrue(next(iter(clusters.values()))["promoted"])

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
            {"id": f"u{index}", "role": "You", "text": f"orchids greenhouse humidity watering aspect{index}", "session_id": str(index % 3)}
            for index in range(5)
        ]}
        clusters = update_clusters(transcript)
        cluster_id = next(iter(clusters))
        label = clusters[cluster_id]["label"]
        transcript["messages"].extend([
            {"id": f"u{index}", "role": "You", "text": f"orchids greenhouse humidity watering aspect{index}", "session_id": str(index % 3)}
            for index in range(5, 20)
        ])
        clusters = update_clusters(transcript)
        self.assertTrue(clusters[cluster_id]["promoted"])
        self.assertEqual(clusters[cluster_id]["label"], label)

    def test_generator_persists_promotion_and_memory_pulse_metadata(self):
        messages = [
            {"id": f"u{index}", "role": "You", "text": f"orchids greenhouse humidity watering aspect{index}", "topics": [], "session_id": str(index % 3)}
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
            (root / "package.json").write_text(json.dumps({"version": "0.2.0"}), encoding="utf-8")
            memory_path = root / "chat_memory.json"
            memory_path.write_text(json.dumps(transcript), encoding="utf-8")
            (root / "memory_graph_state.json").write_text(json.dumps({
                "nodes": [{"id": "u0", "rawImportance": 0.4, "userImportant": True, "timestamp": 123.0}],
                "links": [{"source": "food", "target": "memory", "kind": "sequence", "strength": 0.91}],
                "query": "saved query", "sidebarCollapsed": True,
            }), encoding="utf-8")
            with patch.object(generate_html_graph, "__file__", str(root / "generate_html_graph.py")):
                with contextlib.redirect_stdout(io.StringIO()):
                    generate_html_graph.main()
            snapshot = json.loads((root / "memory_graph_state.json").read_text(encoding="utf-8"))
            preserved = next(node for node in snapshot["nodes"] if node["id"] == "u0")
            self.assertEqual(preserved["rawImportance"], 0.4)
            self.assertTrue(preserved["userImportant"])
            self.assertEqual(preserved["timestamp"], 123.0)
            self.assertEqual(snapshot["query"], "saved query")
            self.assertTrue(snapshot["sidebarCollapsed"])
            cluster_id = next(iter(snapshot["clusters"]))
            self.assertTrue(snapshot["clusters"][cluster_id]["promoted"])
            self.assertTrue(any(node.get("domain") == cluster_id for node in snapshot["nodes"]))
            latest = [node for node in snapshot["nodes"] if node.get("created_cycle") == 1]
            self.assertEqual({node["domain"] for node in latest}, {"memory"})
            self.assertTrue(any(link["kind"] == "transition" for link in snapshot["links"]))
            transition = next(link for link in snapshot["links"] if link["source"] == "food" and link["target"] == "memory")
            self.assertEqual(transition["strength"], 0.91)
            self.assertIn(cluster_id, (root / "memory_graph.html").read_text(encoding="utf-8"))
            self.assertIn('aria-label="Application version 0.2.0">v0.2.0</span>',
                          (root / "memory_graph.html").read_text(encoding="utf-8"))
            html = (root / "memory_graph.html").read_text(encoding="utf-8")
            canvas = html.split('<div class="galaxy-canvas">', 1)[1].split('</div>', 1)[0]
            self.assertIn('id="graph"', canvas)
            self.assertIn('class="version-badge"', canvas)
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
                {"id": f"u{index}", "role": "You", "text": f"orchids greenhouse humidity watering aspect{index}", "topics": [], "session_id": str(index % 3)},
                {"id": f"a{index}", "role": "Assistant", "text": "Related reply", "topics": []},
            ])
        messages.extend([
            {"id": "evaluation", "role": "You", "text": "unclassified"},
            {"id": "food", "role": "You", "text": "pizza dinner"},
        ])
        original_links = [
            {"source": "u0", "target": "evaluation", "kind": "similarity", "strength": 0.5},
            {"source": "evaluation", "target": "a0", "kind": "sequence", "strength": 0.5},
            {"source": "u0", "target": "food", "kind": "similarity", "strength": 0.5},
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
            {"id": f"u{index}", "role": "You", "text": f"memory orchids greenhouse watering aspect{index}", "session_id": str(index % 3)}
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

    def test_related_galaxy_pairs_and_aimem_mentions_get_related_links(self):
        transcript = {"messages": [
            {"id": "git1", "role": "You", "text": "git push", "domain": "git"},
            {"id": "commit1", "role": "You", "text": "commit push", "domain": "commit"},
            {"id": "aimem1", "role": "You", "text": "aimem note", "domain": "aimem"},
            {"id": "mention", "role": "AI", "text": "AImem reply", "domain": "memory"},
            {"id": "plain", "role": "AI", "text": "memory reply", "domain": "memory"},
        ], "clusters": {
            "cluster-git": {"label": "Git / Aimem", "promoted": True, "parent_domain": "evaluation"},
            "cluster-commit": {"label": "Commit / Push", "promoted": True, "parent_domain": "evaluation"},
        }}
        transcript["messages"][0]["domain"] = "cluster-git"
        transcript["messages"][1]["domain"] = "cluster-commit"
        transcript["messages"][2]["domain"] = "cluster-git"
        graph = {"nodes": [
            {"id": message["id"], "kind": "message", "domain": message["domain"], "role": message["role"], "content": message["text"]}
            for message in transcript["messages"]
        ], "links": []}
        apply_graph_domains(graph, transcript)
        related = {(link["source"], link["target"]) for link in graph["links"] if link["kind"] == "related"}
        self.assertIn(("git1", "commit1"), related)
        self.assertIn(("aimem1", "commit1"), related)
        self.assertIn(("git1", "mention"), related)
        self.assertIn(("aimem1", "mention"), related)
        self.assertNotIn(("git1", "plain"), related)


if __name__ == "__main__":
    unittest.main()