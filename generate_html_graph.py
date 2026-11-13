"""Generate a self-contained interactive visualization of the chat memory graph."""

import json
import re
from html import escape
from itertools import combinations
from pathlib import Path

from memory_system import MemoryGraph
from memory_links import infer_similarity_links
from memory_clusters import apply_graph_domains, update_clusters


def build_chat_graph(memory_path: Path) -> tuple[MemoryGraph, str]:
    transcript = json.loads(memory_path.read_text(encoding="utf-8"))
    graph = MemoryGraph()
    messages = transcript["messages"]
    for message in messages:
        graph.add_node(
            message["id"],
            f"{message['role']}: {message['text']}",
            importance=message.get("importance", 0.9 if message["role"] == "You" else 0.8),
            kind="message",
        )
        graph.nodes[message["id"]]._consolidated = message.get("consolidated", False)

    links = transcript.get("links")
    if links is None:
        links = []
        weights: dict[tuple[str, str], float] = {}
        for previous, following in zip(messages, messages[1:]):
            weights[(previous["id"], following["id"])] = 0.35
        for first, second in combinations(messages, 2):
            shared_topics = set(first["topics"]) & set(second["topics"])
            if shared_topics:
                key = (first["id"], second["id"])
                weights[key] = max(weights.get(key, 0), min(0.95, 0.35 + 0.15 * len(shared_topics)))
        links = [
            {"source": source, "target": target, "strength": strength}
            for (source, target), strength in weights.items()
        ]
    links = list(links)
    message_by_id = {message["id"]: message for message in messages}
    message_positions = {message["id"]: index for index, message in enumerate(messages)}
    for link in links:
        if "kind" not in link:
            source, target = link["source"], link["target"]
            is_sequence = abs(message_positions[source] - message_positions[target]) == 1
            shares_topic = bool(
                set(message_by_id[source]["topics"]) & set(message_by_id[target]["topics"])
            )
            link["kind"] = (
                "sequence" if is_sequence else "topic" if shares_topic else "similarity"
            )
    existing_pairs = {
        frozenset((link["source"], link["target"])) for link in links
    }
    for inferred in infer_similarity_links(messages):
        pair = frozenset((inferred["source"], inferred["target"]))
        if pair not in existing_pairs:
            links.append(inferred)
            existing_pairs.add(pair)

    topic_counts: dict[str, int] = {}
    topic_messages: dict[str, list[str]] = {}
    for message in messages:
        for topic in message.get("topics", []):
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
            topic_messages.setdefault(topic, []).append(message["id"])

    def topic_id(topic: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")
        return slug or "topic"

    for topic, count in sorted(topic_counts.items(), key=lambda item: (-item[1], item[0])):
        node_id = topic_id(topic)
        if node_id not in graph.nodes:
            graph.add_node(
                node_id,
                f"Topic: {topic}",
                importance=min(0.95, 0.20 + 0.10 * count),
                kind="topic",
            )
        for message_id in topic_messages[topic]:
            links.append(
                {
                    "source": message_id,
                    "target": node_id,
                    "strength": round(min(0.85, 0.24 + 0.06 * count), 4),
                    "kind": "hub",
                }
            )

    for link in links:
        graph.connect(link["source"], link["target"], strength=link["strength"])
    return graph, transcript["seed_id"]


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#0b1020">
  <title>Chat Memory in Motion</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #080d19;
      color: #e8eefc;
      font-synthesis: none;
      --muted: #98a6c4;
      --line: rgba(158, 177, 218, .13);
      --panel: rgba(17, 25, 46, .88);
      --cyan: #67e8f9;
      --violet: #a78bfa;
      --green: #6ee7b7;
      --amber: #fbbf24;
    }
    * { box-sizing: border-box; }
    body {
      min-height: 100vh;
      margin: 0;
      padding: 36px 20px;
      background:
        radial-gradient(ellipse at 50% -20%, rgba(55, 75, 143, .28), transparent 55%),
        radial-gradient(ellipse at 90% 80%, rgba(38, 111, 119, .10), transparent 42%),
        #080d19;
    }
    main { max-width: 1700px; margin: 0 auto; }
    body.sidebar-collapsed main { max-width: 2016px; }
    header { display: flex; justify-content: flex-end; align-items: center; gap: 12px; margin-bottom: 10px; }
    .eyebrow { color: var(--cyan); text-transform: uppercase; letter-spacing: .16em; font-size: 11px; font-weight: 750; }
    .live-pill { white-space: nowrap; border: 1px solid rgba(110,231,183,.25); color: var(--green); border-radius: 30px; padding: 3px 8px; font-size: 9px; background: rgba(110,231,183,.07); }
    .stats { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 8px; margin-bottom: 10px; }
    .stat, .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 16px; box-shadow: 0 18px 60px rgba(0,0,0,.18); }
    .stat { height: 22px; padding: 0 12px; display: flex; align-items: center; justify-content: space-between; gap: 8px; }
    .stat-label { color: var(--muted); font-size: 9px; text-transform: uppercase; letter-spacing: .06em; white-space: nowrap; }
    .stat-value { font-size: 19px; line-height: 1; font-weight: 700; font-variant-numeric: tabular-nums; }
    .workspace { display: grid; grid-template-columns: minmax(0, 1fr); gap: 16px; align-items: stretch; }
    .graph-panel { position: relative; min-width: 0; padding: 15px; overflow: hidden; }
    .graph-head { display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 4px 7px 0; }
    .graph-actions { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .zoom-button { padding: 4px 10px; font-size: 16px; line-height: 1; }
    h2 { font-size: 14px; margin: 0; letter-spacing: .01em; }
    .hint { font-size: 11px; color: var(--muted); }
    svg { display: block; width: 100%; height: auto; min-height: 330px; overflow: visible; }
    .edge { stroke: #64749b; stroke-linecap: round; opacity: .5; transition: stroke .25s, opacity .25s, stroke-width .25s; }
    .edge.topic { stroke: #2dd4bf; }
    .edge.hub { stroke: #f59e0b; stroke-dasharray: 2 4; }
    .edge.similarity { stroke: #c084fc; stroke-dasharray: 5 5; }
    .edge.created { stroke: #22d3ee; opacity: 1; filter: drop-shadow(0 0 4px rgba(34,211,238,.6)); }
    .edge.cross-domain { stroke: #fda4af; stroke-dasharray: 2 8; opacity: .07; }
    .edge.transition { stroke: #fda4af; stroke-dasharray: 3 6; opacity: .35; }
    .edge.transition.created { opacity: .9; }
    .edge.parent { stroke: #6ee7b7; stroke-dasharray: none; opacity: .8; }
    .learner-values { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 6px 12px; margin: 12px 0; font-size: 12px; }
    .learner-values dt { color: var(--muted); overflow-wrap: anywhere; }
    .learner-values dd { margin: 0; font-variant-numeric: tabular-nums; }
    .learner-mode { color: var(--green); font-size: 12px; }
    .edge.recalled { stroke: #67e8f9; opacity: .98; filter: drop-shadow(0 0 4px rgba(103,232,249,.7)); }
    .edge-label { fill: #9ba9c6; font-size: 11px; text-anchor: middle; paint-order: stroke; stroke: #11192e; stroke-width: 5px; stroke-linejoin: round; }
    .node { cursor: pointer; outline: none; }
    .node circle.core { fill: #3b4f7a; stroke: #cfe0ff; stroke-width: .6; transition: fill .2s, stroke .2s, r .2s; }
    .node:hover circle.core, .node:focus circle.core { stroke: white; }
    .node.selected circle.core { fill: #0f5e78; stroke: #67e8f9; stroke-width: .7; }
    .node.consolidated circle.core { stroke: var(--green); }
    .node.created circle.core { fill: #0f5e78 !important; stroke: #ecfeff !important; stroke-width: .8 !important; filter: drop-shadow(0 0 6px rgba(34,211,238,.28)); }
    .node.topic circle.core { fill: #26324f; stroke: #b0c4ff; stroke-width: .35; opacity: .98; }
    .node.topic.selected circle.core { fill: #0f5e78; stroke: #67e8f9; stroke-width: .7; }
    .node.dimmed { opacity: .18; }
    .node.match circle.core { stroke: var(--violet); filter: drop-shadow(0 0 7px rgba(167,139,250,.4)); }
    .node.related circle.core { stroke: #7be0c4; }
    .node.memory circle.core { fill: #3f5f9a; stroke: #bfdbfe; }
    .node.scripture circle.core { fill: #5b2143; stroke: #f472b6; }
    .node.topic.scripture circle.core { fill: #7a2e59; stroke: #f9a8d4; }
    .node.food circle.core { fill: #23421f; stroke: #4ade80; }
    .node.topic.food circle.core { fill: #2f5b2a; stroke: #86efac; }
    .node.finance circle.core { fill: #3f2f12; stroke: #f59e0b; }
    .node.topic.finance circle.core { fill: #63471d; stroke: #fbbf24; }
    .node.evaluation circle.core { fill: #e879f9; stroke: #fdf4ff; stroke-width: .9; filter: drop-shadow(0 0 3px rgba(232,121,249,.85)); }
    .node.topic.evaluation circle.core { fill: #f0abfc; stroke: #ffffff; stroke-width: .9; filter: drop-shadow(0 0 3px rgba(240,171,252,.85)); }
    .node.entertainment circle.core { fill: #1f3d62; stroke: #60a5fa; }
    .node.topic.entertainment circle.core { fill: #26507f; stroke: #bfdbfe; }
    .node.bridge circle.core { stroke-dasharray: 2 3; }
    .node.user-important circle.core { stroke: var(--amber); stroke-width: 1.6; filter: drop-shadow(0 0 5px rgba(251,191,36,.75)); }
    .cluster-bg { fill: rgba(14, 22, 39, .2); stroke: rgba(165,180,220,.16); stroke-width: 1.2; rx: 18; ry: 18; }
    .cluster-bg.memory { fill: rgba(11, 32, 52, .26); stroke: rgba(103,232,249,.32); }
    .cluster-bg.scripture { fill: rgba(56, 18, 47, .24); stroke: rgba(244,114,182,.44); stroke-dasharray: 7 6; }
    .cluster-bg.food { fill: rgba(20, 42, 22, .24); stroke: rgba(134,239,172,.40); stroke-dasharray: 8 7; }
    .cluster-bg.finance { fill: rgba(58, 37, 13, .24); stroke: rgba(251,191,36,.42); stroke-dasharray: 6 7; }
    .cluster-bg.evaluation { fill: rgba(41, 27, 64, .24); stroke: rgba(196,181,253,.42); stroke-dasharray: 5 8; }
    .cluster-bg.entertainment { fill: rgba(17, 34, 58, .24); stroke: rgba(147,197,253,.42); stroke-dasharray: 6 8; }
    .cluster-bg.other { fill: rgba(52, 34, 11, .2); stroke: rgba(251,191,36,.30); }
    .cluster-bg { stroke-dasharray: none !important; stroke-width: 1.6; }
    .region-shadow { fill: #000; opacity: .28; }
    .region-depth { pointer-events: none; }
    .node circle.core { filter: drop-shadow(1px 2px 1px rgba(0,0,0,.65)); }
    .node-shine { fill: #fff; opacity: .48; pointer-events: none; }
    .cluster-subtitle { fill: var(--muted); font-size: 10px; text-anchor: middle; }
    .cluster-title { fill: #eff6ff; font-size: 12px; font-weight: 800; letter-spacing: .05em; text-transform: uppercase; }
    .cluster-title.scripture { fill: #f9a8d4; }
    .cluster-title.food { fill: #86efac; }
    .cluster-title.finance { fill: #fcd34d; }
    .cluster-title.evaluation { fill: #c4b5fd; }
    .cluster-title.entertainment { fill: #93c5fd; }
    .cluster-title.memory { fill: #67e8f9; }
    .pulse-ring { fill: none; stroke: var(--cyan); stroke-width: 3; opacity: 0; pointer-events: none; filter: drop-shadow(0 0 4px rgba(103,232,249,.8)); }
    .pulse-ring.animate { animation: ripple 1.5s ease-out infinite; }
    @keyframes ripple { 0% { r: 22px; opacity: 1; } 45% { opacity: .85; } 100% { r: 58px; opacity: 0; } }
    aside { padding: 20px; display: flex; flex-direction: column; gap: 20px; }
    .section-label { color: var(--muted); text-transform: uppercase; letter-spacing: .12em; font-size: 10px; font-weight: 750; margin-bottom: 11px; }
    .selected-memory-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 11px; }
    .selected-memory-head .section-label { margin-bottom: 0; }
    .selected-memory-actions { display: flex; gap: 8px; flex-wrap: wrap; }
    .selected-memory-empty { padding: 16px; border: 1px dashed rgba(165,180,220,.18); border-radius: 14px; background: rgba(18, 27, 48, .72); }
    .selected-memory-empty p { margin: 0 0 12px; color: var(--muted); font-size: 13px; line-height: 1.5; }
    .selected-name { font-size: 22px; font-weight: 750; margin: 0 0 5px; }
    .selected-content { color: var(--muted); font-size: 13px; line-height: 1.5; min-height: 38px; }
    .meter-row { display: flex; justify-content: space-between; color: #bdc8dd; font-size: 11px; margin: 13px 0 7px; }
    .meter { height: 7px; background: #202b45; border-radius: 10px; overflow: hidden; }
    .meter > span { display: block; height: 100%; width: 0; border-radius: inherit; background: linear-gradient(90deg, #7c72f6, #67e8f9); transition: width .35s; }
    .actions { display: grid; gap: 8px; }
    .query-input { width: 100%; border: 1px solid rgba(165,180,220,.2); color: #e8eefc; border-radius: 10px; padding: 10px 12px; background: #17223a; font: inherit; font-size: 12px; }
    .query-input:focus { outline: 2px solid rgba(103,232,249,.35); border-color: rgba(103,232,249,.35); }
    .search-results { display: grid; gap: 8px; margin-top: 10px; }
    .result { border: 1px solid rgba(165,180,220,.15); background: #121b30; border-radius: 12px; padding: 10px 12px; cursor: pointer; transition: transform .16s, border-color .16s, background .16s; }
    .result:hover { transform: translateY(-1px); border-color: rgba(165,180,220,.35); background: #17233d; }
    .result strong { display: block; color: #f2f6ff; margin-bottom: 4px; }
    .result .meta { color: var(--muted); font-size: 11px; line-height: 1.4; }
    .signal-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
    .signal { border: 1px solid rgba(165,180,220,.14); background: #17233d; color: #d9e4fb; border-radius: 999px; padding: 6px 9px; font-size: 10px; line-height: 1; }
    .signal.good { border-color: rgba(110,231,183,.28); color: #9ff0d0; }
    .signal.bad { border-color: rgba(248,113,113,.3); color: #ffb7b7; }
    .signal.neutral { border-color: rgba(251,191,36,.25); color: #ffe39a; }
    button { border: 1px solid rgba(165,180,220,.18); color: #e8eefc; border-radius: 10px; padding: 10px 12px; background: #202b46; font: inherit; font-size: 12px; font-weight: 650; cursor: pointer; transition: background .16s, border-color .16s, transform .16s; }
    button:hover { background: #29385b; border-color: rgba(165,180,220,.4); transform: translateY(-1px); }
    button.primary { background: #15546a; border-color: rgba(103,232,249,.32); }
    button.primary:hover { background: #17677f; }
    button:focus-visible { outline: 2px solid var(--cyan); outline-offset: 2px; }
    .event { border-top: 1px solid var(--line); padding-top: 15px; min-height: 70px; color: var(--muted); font-size: 11px; line-height: 1.55; }
    .event strong { color: #e8eefc; }
    body.sidebar-collapsed aside { display: none; }
    footer { color: #687896; font-size: 10px; padding: 14px 4px 0; line-height: 1.5; }
    .galaxy-canvas { position: relative; }
    .version-badge { position: absolute; right: 10px; bottom: 8px; z-index: 20; padding: 3px 6px; border-radius: 4px; background: #10182b; color: #8795af; font-size: 10px; line-height: 1.3; pointer-events: none; }
    @media (max-width: 850px) { .workspace { grid-template-columns: 1fr; } aside { display: grid; grid-template-columns: 1fr 1fr; } aside > :last-child { grid-column: 1 / -1; } .graph-head { flex-wrap: wrap; } .graph-actions { margin-left: auto; } }
    @media (max-width: 560px) { body { padding: 24px 12px; } .stats { grid-template-columns: repeat(3, minmax(0, 1fr)); } .stat { padding: 8px 10px; } .stat-value { font-size: 18px; } aside { display: flex; } svg { min-height: 265px; } }
  </style>
</head>
<body>
<main>
  <header>
    <div class="live-pill" id="learner-state-pill">LEARNER: CONNECTING</div>
  </header>
  <section class="stats" aria-label="Graph statistics">
    <div class="stat"><div class="stat-label">Memories</div><div class="stat-value" id="node-count">0</div></div>
    <div class="stat"><div class="stat-label">Connections</div><div class="stat-value" id="link-count">0</div></div>
    <div class="stat"><div class="stat-label">Avg. importance</div><div class="stat-value" id="avg-importance">0.00</div></div>
    <div class="stat"><div class="stat-label">Decay cycles</div><div class="stat-value" id="cycle-count">0</div></div>
    <div class="stat"><div class="stat-label">Topics</div><div class="stat-value" id="topic-count">0</div></div>
  </section>
  <section class="workspace">
    <div class="panel graph-panel">
      <div class="graph-head">
        <h2>AI Memory</h2>
        <div class="graph-actions">
          <button type="button" class="zoom-button" id="zoom-out" aria-label="Zoom out">&minus;</button>
          <button type="button" class="zoom-button" id="zoom-in" aria-label="Zoom in">+</button>
        </div>
      </div>
      <div class="galaxy-canvas">
        <svg id="graph" viewBox="0 0 1120 760" role="img" aria-label="Interactive graph of connected memories"></svg>
        <span class="version-badge" aria-label="Application version __APP_VERSION__">v__APP_VERSION__</span>
      </div>
    </div>
    <section class="panel" id="prompt-training-panel">
        <div class="section-label">Prompt training</div>
        <form id="training-prompt-form">
          <textarea id="memory-query" class="query-input" rows="2" placeholder="Enter a training prompt" aria-label="Training prompt" required></textarea>
        </form>
        <div class="section-label" style="margin-top:16px">Latest evaluation</div>
        <div id="evaluation-question" class="selected-content"></div>
        <div class="event" id="event" aria-live="polite">Ready</div>
        <section id="model-response-panel" hidden>
          <div class="section-label" id="model-response-label">Retrieved memory excerpt</div>
          <div id="model-response" class="selected-content" style="white-space:pre-wrap"></div>
          <div id="model-response-meta" class="selected-content"></div>
          <div id="response-evaluation-status" class="learner-mode" hidden></div>
          <div id="response-feedback-controls">
          <textarea id="response-correction" class="query-input" rows="2" placeholder="Your correction (optional)" aria-label="Response correction" maxlength="2000"></textarea>
          <div class="graph-actions"><button type="button" id="response-useful">Response useful</button><button type="button" id="response-needs-correction">Needs correction</button></div>
          </div>
        </section>
        <div id="search-results" class="search-results" aria-live="polite"></div>
      </section>
    <aside class="panel">
      <section id="learner-panel">
        <div class="section-label">Learner component</div>
        <div class="learner-mode" id="learner-mode" aria-live="polite">Connecting</div>
        <dl class="learner-values" id="learner-values"></dl>
      </section>
      <section id="selected-memory-panel">
        <div class="selected-memory-head">
          <div class="section-label">Selected memory</div>
          <div class="selected-memory-actions">
            <button type="button" id="importance-toggle-button" hidden aria-pressed="false">Mark important</button>
          </div>
        </div>
        <h2 id="selected-name" class="selected-name">Pick a memory</h2>
        <div id="selected-content" class="selected-content">Select a message in the graph to inspect and activate it.</div>
        <div id="cluster-importance-evidence" class="selected-content" hidden></div>
        <div class="signal-row" id="selected-signals"></div>
        <div class="meter-row"><span>Importance</span><span id="importance-value">—</span></div>
        <div class="meter"><span id="importance-meter"></span></div>
        <div class="meter-row"><span>Connected ideas</span><span id="connection-value">—</span></div>
        <div class="meter"><span id="connection-meter"></span></div>
      </section>
    </aside>
  </section>
  <footer>Source: chat_memory.json. Local models: memory_learner.py and original_generator.py.</footer>
</main>
<script>
const initialGraph = __GRAPH_DATA__;
const original = JSON.parse(JSON.stringify(initialGraph));
let savedEvaluation = null;
try { savedEvaluation = JSON.parse(sessionStorage.getItem("latest-memory-evaluation") || "null"); } catch {}
const state = {
  nodes: initialGraph.nodes.map(n => ({
    ...n,
    rawImportance: typeof n.rawImportance === "number" ? n.rawImportance : (typeof n.importance === "number" ? n.importance : 0.0),
    history: [],
    feedback: n.feedback || {helpful: 0, corrections: 0, weak: 0, recalls: 0},
    userImportant: !!n.userImportant,
    created_cycle: n.created_cycle
  })),
  links: initialGraph.links.map(l => ({...l})),
  cycles: initialGraph.cycles || 0,
  selected: null,
  sidebarCollapsed: false,
  graphStateStamp: null,
  active: {},
  lastActivated: null,
  query: savedEvaluation?.query || "",
  currentEvaluation: savedEvaluation,
  submitting: false,
  results: [],
  searchStatus: "idle",
  searchRequestId: 0,
  searchGeneration: null,
  latestJudgment: null,
  learner: null,
  learnerOffline: false,
  generator: null,
  generatorOffline: false,
};
const revisingJudgments = new Set();
const graphStateApi = "http://localhost:3001/api/graph-state";
const svg = document.getElementById("graph");
let viewExtent = 380;
let zoomLevel = 1;
function applyViewBox() {
  const size = 2 * viewExtent / zoomLevel;
  const offset = viewExtent - size / 2;
  svg.setAttribute("viewBox", `${offset} ${offset} ${size} ${size}`);
}
function changeZoom(factor) {
  zoomLevel = Math.min(4, Math.max(0.5, zoomLevel * factor));
  applyViewBox();
}
const ns = "http://www.w3.org/2000/svg";
const positions = {};
const queryInput = document.getElementById("memory-query");
const searchResults = document.getElementById("search-results");
const importanceToggleButton = document.getElementById("importance-toggle-button");
const selectedMemoryPanel = document.getElementById("selected-memory-panel");
async function refreshLearnerStatus() {
  try {
    const response = await fetch("http://localhost:3001/api/learner-status");
    if (!response.ok) throw new Error("Learner unavailable");
    const status = await response.json();
    const previousGeneration = state.learner?.generation;
    const changed = state.learnerOffline || JSON.stringify(status) !== JSON.stringify(state.learner);
    state.learner = status;
    state.learnerOffline = false;
    if (changed) render();
    if (previousGeneration != null && previousGeneration !== status.generation && state.query) void refreshSearchResults();
  } catch {
    if (!state.learnerOffline) {
      state.learnerOffline = true;
      render();
    }
  }
}
async function refreshGeneratorStatus() {
  try {
    const response = await fetch("http://localhost:3001/api/generator");
    if (!response.ok) throw new Error("Generator unavailable");
    const status = await response.json();
    const changed = state.generatorOffline || JSON.stringify(status) !== JSON.stringify(state.generator);
    state.generator = status;
    state.generatorOffline = false;
    if (changed) render();
  } catch {
    if (!state.generatorOffline) {
      state.generatorOffline = true;
      render();
    }
  }
}
function renderLearnerStatus() {
  const status = state.learner;
  const mode = state.learnerOffline ? "Unavailable" : !status ? "Connecting" : status.generation > 0 ? "Trained checkpoint" : status.pending_review ? "Awaiting review" : "Initial scoring rule";
  document.getElementById("learner-mode").textContent = mode;
  document.getElementById("learner-state-pill").textContent = `LEARNER: ${mode.toUpperCase()}`;
  const list = document.getElementById("learner-values");
  list.replaceChildren();
  if (!status) return;
  const values = [
    ["Generation", status.generation], ["User memories", status.user_documents],
    ["Reviewed memories", status.approved_memories], ["Recorded judgments", status.feedback_examples],
    ["Eligible judgments", `${status.eligible_examples} / ${status.minimum_examples}`],
    ["Eligible questions", `${status.eligible_queries} / ${status.minimum_queries}`],
    ["Pending review", status.pending_review],
    ["Needs relabeling", status.needs_relabel || 0],
    ...Object.entries(status.judgment_counts || {}).map(([name, value]) => [name.replaceAll("_", " ") + " judgments", value]),
    ...Object.entries(status.weights || {}).map(([name, value]) => [name.replaceAll("_", " "), Number(value).toFixed(4)]),
    ["Intercept", Number(status.intercept).toFixed(4)],
    ["Last evaluation", status.training?.status || "Not run"]
  ];
  if (status.training?.candidate_loss != null) {
    values.push(["Previous loss", status.training.baseline_loss.toFixed(4)], ["Candidate loss", status.training.candidate_loss.toFixed(4)]);
  }
  for (const [name, value] of values) {
    const term = document.createElement("dt"), description = document.createElement("dd");
    term.textContent = name;
    description.textContent = String(value);
    list.append(term, description);
  }
}
function renderModelResponse() {
  const evaluation = state.currentEvaluation;
  document.getElementById("model-response-panel").hidden = !evaluation;
  if (evaluation) {
    const reply = evaluation.reply;
    const conversation = evaluation.mode === "local-conversation-neural";
    const neural = ["local-layered-neural", "local-base-neural", "local-conversation-neural"].includes(evaluation.mode);
    const generated = neural || evaluation.mode === "local-character-ngram";
    document.getElementById("model-response-label").textContent = neural ? (evaluation.mode === "local-conversation-neural" ? "Persistent narrow-task conversation learner" : evaluation.mode === "local-base-neural" ? "Experimental layer-free New Base continuation" : "Historical L0/L1 neural continuation") : generated ? "Experimental local character output" : "Retrieved memory excerpt";
    document.getElementById("model-response").textContent = reply.length > 650 ? reply.slice(0, 647).trimEnd() + "..." : reply;
    const sources = (evaluation.training_sources || []).map(source => source.title).filter(Boolean);
    document.getElementById("model-response-meta").textContent = generated
      ? neural ? `Not a grounded answer or scripture. Neural run ${evaluation.chat_model.run_id}, seed ${evaluation.chat_model.seed}, passes ${evaluation.chat_model.passes}. ${conversation ? "Unmarked replies do not train. Relevant trains the saved answer; entering and submitting a correction trains the edited answer." : "No online neural training."}`
        : `Experimental, not a grounded answer. Generator ${evaluation.generation}. Training sources: ${sources.join(", ") || "unknown"}.`
      : "Selected from stored conversation memory.";
    const correction = document.getElementById("response-correction");
    correction.maxLength = conversation ? 95 : 2000;
    correction.placeholder = conversation ? "Correction: one line, up to 95 characters. Submit with Needs correction to train now." : "Your correction (optional)";
    document.getElementById("response-useful").textContent = conversation ? "Relevant" : "Response useful";
    document.getElementById("response-needs-correction").title = conversation
      ? "An entered correction approves immediate weight training; without text, only record the judgment." : "Record response feedback";
  }
  const rated = evaluation?.response_rating;
  const trained = evaluation?.neural_training?.trained;
  const awaitingCorrection = evaluation?.mode === "local-conversation-neural" && rated === "needs_correction" && !trained;
  document.getElementById("response-feedback-controls").hidden = !!rated && !awaitingCorrection;
  document.getElementById("response-evaluation-status").hidden = !rated;
  if (rated) document.getElementById("response-evaluation-status").textContent = trained
    ? "Approved answer trained, verified and active in chat."
    : awaitingCorrection ? "No weight update. Enter a correction above and submit Needs correction to train it."
      : `Evaluated: ${rated === "useful" ? "Useful" : "Needs correction"}`;
}
async function recordResponseFeedback(rating) {
  if (!state.currentEvaluation || state.responseFeedbackPending) return;
  state.responseFeedbackPending = true;
  const responseButtons = document.querySelectorAll("#response-feedback-controls button");
  responseButtons.forEach(button => { button.disabled = true; });
  announce("Submitting feedback; explicitly approved conversation answers train before confirmation.");
  try {
    const response = await fetch("http://localhost:3001/api/response-feedback", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({response_id: state.currentEvaluation.response_memory_id, rating, correction: document.getElementById("response-correction").value})
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Response feedback was not saved");
    state.currentEvaluation.response_rating = rating;
    state.currentEvaluation.neural_training = result.neural_training;
    sessionStorage.setItem("latest-memory-evaluation", JSON.stringify(state.currentEvaluation));
    renderModelResponse();
    state.latestJudgment = {query: state.query, responseFeedback: true};
    announce(result.neural_training?.trained ? "Approved answer trained, verified and active in chat."
      : "Feedback recorded without neural weight training.");
  } catch (error) { document.getElementById("event").textContent = error.message; }
  finally { state.responseFeedbackPending = false; responseButtons.forEach(button => { button.disabled = false; }); }
}
function domainForNode(node) {
  if (node.domain) return node.domain;
  const topics = nodeTopics(node).join(" ").toLowerCase();
  const text = nodeSearchText(node);
  if (
    node.kind === "topic" &&
    ["scripture", "john-3-16", "christianity"].some(item => topics.includes(item))
  ) {
    return "scripture";
  }
  if (
    ["scripture", "john-3-16", "christianity", "bible", "verse", "verses", "gospel"].some(item => text.includes(item))
  ) {
    return "scripture";
  }
  if (
    ["food", "pizza", "meal", "meals", "recipe", "recipes", "cooking", "cook", "cuisine", "dinner", "lunch", "breakfast", "snack", "favorite-food", "favorite foods"].some(item => text.includes(item) || topics.includes(item))
  ) {
    return "food";
  }
  if (
    ["finance", "invest", "investing", "investment", "investments", "portfolio", "stock", "stocks", "bond", "bonds", "saving", "savings", "budget", "budgeting", "debt", "loan", "loans", "interest", "retirement", "401k", "ira", "crypto", "mining", "coin", "coins", "market", "markets", "profit", "profitability"].some(item => text.includes(item) || topics.includes(item))
  ) {
    return "finance";
  }
  if (
    ["evaluation", "meta", "domain-separation", "classification", "policy", "under evaluation", "under-evaluation", "general area", "outside the core", "not related to ai memory"].some(item => text.includes(item) || topics.includes(item))
  ) {
    return "evaluation";
  }
  if (
    ["entertainment", "tv", "tv show", "television", "series", "show", "shows", "elementary", "star trek", "next generation", "mcgyver", "macgyver", "cast"].some(item => text.includes(item) || topics.includes(item))
  ) {
    return "entertainment";
  }
  return "memory";
}
state.nodes.forEach(node => {
  node.domain = domainForNode(node);
});
const SPLIT_RADIUS = 180;
const SUN_RADIUS = 100;
const BLAST_DISTANCE = 200;
const SUN_SPREAD = 0.5;
const BABY_RADIUS = 9;
const BABY_INTERACTIONS = 5;
const SUN_BALANCE = 0.4;
const CROSS_SUN_MIN = 0.3;
const EVAL_SUNS = ["evaluation", "evaluation-2"];
function isBaby(domain) {
  const cluster = initialGraph.clusters?.[domain];
  return Boolean(cluster?.promoted) && (cluster.interactions || 0) < BABY_INTERACTIONS;
}
function layoutStats(nodes) {
  const messages = nodes.filter(node => node.kind === "message");
  const salience = messages.reduce((sum, node) => sum + Math.min(1, learnedScore(node)), 0) / Math.max(1, messages.length);
  return {messages, salience, raw: 48 + 6 * Math.sqrt(messages.length) + 20 * salience};
}
function relaxSun(nodeIds, radius, weights) {
  const limit = Math.max(1, radius - 28);
  const count = nodeIds.length;
  const k = limit / Math.sqrt(Math.max(1, count));
  const index = new Map(nodeIds.map((id, i) => [id, i]));
  const pos = nodeIds.map((id, i) => {
    const angle = i * Math.PI * (3 - Math.sqrt(5));
    const distance = limit * Math.sqrt((i + 0.5) / Math.max(1, count));
    return {x: distance * Math.cos(angle), y: distance * Math.sin(angle)};
  });
  const edges = [];
  nodeIds.forEach(id => (weights.get(id) || new Map()).forEach((w, other) => {
    if (index.has(other) && index.get(id) < index.get(other)) edges.push([index.get(id), index.get(other), w]);
  }));
  let temperature = radius * 0.1;
  for (let iteration = 0; iteration < 80; iteration++) {
    const disp = pos.map(() => ({x: 0, y: 0}));
    for (let i = 0; i < count; i++) {
      for (let j = i + 1; j < count; j++) {
        const dx = pos[i].x - pos[j].x, dy = pos[i].y - pos[j].y;
        const d = Math.max(0.01, Math.hypot(dx, dy));
        const f = k * k / d;
        disp[i].x += dx / d * f; disp[i].y += dy / d * f;
        disp[j].x -= dx / d * f; disp[j].y -= dy / d * f;
      }
    }
    edges.forEach(([i, j, w]) => {
      const dx = pos[i].x - pos[j].x, dy = pos[i].y - pos[j].y;
      const d = Math.max(0.01, Math.hypot(dx, dy));
      const f = w * d * d / k;
      disp[i].x -= dx / d * f; disp[i].y -= dy / d * f;
      disp[j].x += dx / d * f; disp[j].y += dy / d * f;
    });
    for (let i = 0; i < count; i++) {
      const len = Math.hypot(disp[i].x, disp[i].y);
      if (len > 0) {
        const step = Math.min(len, temperature) / len;
        pos[i].x += disp[i].x * step; pos[i].y += disp[i].y * step;
      }
      const r = Math.hypot(pos[i].x, pos[i].y);
      if (r > limit) { pos[i].x *= limit / r; pos[i].y *= limit / r; }
    }
    temperature *= 0.96;
  }
  return new Map(nodeIds.map((id, i) => [id, pos[i]]));
}
function refineSunMembership(sunOf, weights, ordered) {
  const total = ordered.length;
  const minSize = Math.floor(total * SUN_BALANCE);
  const maxSize = Math.ceil(total * (1 - SUN_BALANCE));
  const sizes = [0, 0];
  ordered.forEach(id => sizes[sunOf.get(id)]++);
  for (let round = 0; round < 20; round++) {
    let moved = 0;
    ordered.forEach(id => {
      const current = sunOf.get(id), other = 1 - current;
      if (sizes[current] - 1 < minSize || sizes[other] + 1 > maxSize) return;
      let pullCurrent = 0, pullOther = 0;
      weights.get(id).forEach((w, neighbor) => {
        if (sunOf.get(neighbor) === current) pullCurrent += w;
        else pullOther += w;
      });
      const gain = pullOther - pullCurrent;
      const tieToSmaller = Math.abs(gain) < 1e-9 && sizes[other] < sizes[current];
      if (gain > 1e-9 || tieToSmaller) {
        sunOf.set(id, other);
        sizes[current]--; sizes[other]++;
        moved++;
      }
    });
    if (!moved) break;
  }
}
function planEvaluationSplit() {
  const keyFor = new Map();
  const relative = new Map();
  const evaluation = state.nodes.filter(node => node.domain === "evaluation");
  if (layoutStats(evaluation).raw < SPLIT_RADIUS) return {keyFor, relative};
  const members = new Set(evaluation.map(node => node.id));
  const weights = new Map();
  members.forEach(id => weights.set(id, new Map()));
  state.links.forEach(link => {
    if (link.kind === "parent" || link.source === link.target || !members.has(link.source) || !members.has(link.target)) return;
    const w = Number(link.strength) || 0.3;
    weights.get(link.source).set(link.target, (weights.get(link.source).get(link.target) || 0) + w);
    weights.get(link.target).set(link.source, (weights.get(link.target).get(link.source) || 0) + w);
  });
  const ordered = evaluation.map(node => node.id);
  const degree = new Map(ordered.map(id => [id, [...weights.get(id).values()].reduce((sum, w) => sum + w, 0)]));
  let vector = new Map(ordered.map((id, i) => [id, (i % 2 ? 1 : -1) * (1 + (i % 7) / 7)]));
  for (let round = 0; round < 200; round++) {
    const next = new Map();
    ordered.forEach(id => {
      let sum = 0;
      weights.get(id).forEach((w, other) => { sum += w * vector.get(other); });
      const walk = degree.get(id) > 0 ? sum / degree.get(id) : 0;
      next.set(id, 0.5 * vector.get(id) + 0.5 * walk);
    });
    let total = 0, mass = 0;
    ordered.forEach(id => { total += degree.get(id) * next.get(id); mass += degree.get(id); });
    const mean = mass > 0 ? total / mass : 0;
    let norm = 0;
    ordered.forEach(id => { next.set(id, next.get(id) - mean); norm += next.get(id) ** 2; });
    norm = Math.sqrt(norm) || 1;
    ordered.forEach(id => next.set(id, next.get(id) / norm));
    vector = next;
  }
  const ranked = [...ordered].sort((x, y) => vector.get(y) - vector.get(x) || (x < y ? -1 : 1));
  const sunOf = new Map(ranked.map((id, i) => [id, i < Math.ceil(ranked.length / 2) ? 0 : 1]));
  refineSunMembership(sunOf, weights, ordered);
  [0, 1].forEach(sun => {
    const ids = ordered.filter(id => sunOf.get(id) === sun);
    const key = sun === 0 ? "evaluation" : "evaluation-2";
    ids.forEach(id => keyFor.set(id, key));
    const nodes = evaluation.filter(node => sunOf.get(node.id) === sun);
    const radius = domainLayoutFor(key, nodes).radius;
    relaxSun(ids, radius, weights).forEach((point, id) => relative.set(id, point));
  });
  return {keyFor, relative};
}
const evaluationSplit = planEvaluationSplit();
state.nodes.forEach(node => {
  node.layoutKey = evaluationSplit.keyFor.get(node.id) || node.domain || "memory";
});
const preferredSun = new Map();
(() => {
  const pull = new Map();
  const keyById = new Map(state.nodes.map(n => [n.id, n.layoutKey]));
  state.links.forEach(link => {
    if (link.kind === "parent") return;
    const keyA = keyById.get(link.source), keyB = keyById.get(link.target);
    const w = Number(link.strength) || 0.3;
    [[link.source, keyB, keyA], [link.target, keyA, keyB]].forEach(([id, other, own]) => {
      if (!EVAL_SUNS.includes(other) || EVAL_SUNS.includes(own)) return;
      const m = pull.get(id) || new Map();
      m.set(other, (m.get(other) || 0) + w);
      pull.set(id, m);
    });
  });
  pull.forEach((m, id) => {
    const [best] = [...m.entries()].sort((x, y) => (y[1] - x[1]) || (x[0] < y[0] ? -1 : 1));
    preferredSun.set(id, best[0]);
  });
})();

const domainGroups = Object.entries(
  state.nodes.reduce((acc, node) => {
    const domain = node.layoutKey || "memory";
    (acc[domain] ||= []).push(node);
    return acc;
  }, {})
);
function domainLayoutFor(domain, nodes) {
  const cluster = initialGraph.clusters?.[domain];
  const labels = {memory: "AI memory", scripture: "Scripture", food: "Food", finance: "Finance", evaluation: "Under evaluation", "evaluation-2": "Under evaluation II", entertainment: "Entertainment", other: "Other"};
  const {messages, salience, raw} = layoutStats(nodes);
  const overflow = layoutStats(state.nodes.filter(node => node.domain === "evaluation")).raw >= SPLIT_RADIUS;
  const radius = domain.startsWith("evaluation") && overflow ? SUN_RADIUS : isBaby(domain) ? BABY_RADIUS : Math.min(SPLIT_RADIUS, raw);
  return {center: {x: 0, y: 0}, radius, label: cluster?.label || labels[domain] || domain, subtitle: `${messages.length} memories`, messages: messages.length, salience, styleDomain: domain === "evaluation-2" ? "evaluation" : domain};
}
const domainLayouts = new Map();
function memoryRadius(node) {
  return 2.5 + 3.5 * Math.sqrt(Math.min(1, learnedScore(node))) + (node.consolidated ? 0.75 : 0) + (node.domain === "evaluation" ? 1 : 0);
}
function linkWeightsFor(domain) {
  const keyById = new Map(state.nodes.map(node => [node.id, node.layoutKey || node.domain || "memory"]));
  const totals = new Map();
  state.links.forEach(link => {
    if (link.kind === "parent") return;
    const a = keyById.get(link.source), b = keyById.get(link.target);
    const w = Number(link.strength) || 0.3;
    if (a === domain && b !== domain) totals.set(b, (totals.get(b) || 0) + w);
    if (b === domain && a !== domain) totals.set(a, (totals.get(a) || 0) + w);
  });
  return totals;
}
function nearestSunFor(domain) {
  const suns = ["evaluation", "evaluation-2"];
  const totals = linkWeightsFor(domain);
  return [...totals.entries()].filter(([key]) => suns.includes(key)).sort((x, y) => y[1] - x[1])[0]?.[0] || null;
}
function updateLayout() {
  domainGroups.forEach(([domain, nodes]) => domainLayouts.set(domain, domainLayoutFor(domain, nodes)));
  const babySuns = new Map();
  domainLayouts.forEach((_, domain) => {
    if (!isBaby(domain)) return;
    const sun = nearestSunFor(domain);
    if (sun && domainLayouts.has(sun)) babySuns.set(domain, sun);
  });
  const satellites = [...domainLayouts.keys()].filter(domain => domain !== "memory" && !babySuns.has(domain)).sort();
  const split = domainLayouts.has("evaluation-2");
  const centerRadius = domainLayouts.get("memory")?.radius || 48;
  const satelliteRadius = Math.max(48, ...satellites.map(domain => domainLayouts.get(domain).radius));
  const orbit = Math.max(centerRadius + satelliteRadius + 65, satellites.length > 1 ? (2 * satelliteRadius + 45) / (2 * Math.sin(Math.PI / satellites.length)) : 0);
  const offsets = new Map([["memory", {x: 0, y: 0}]]);
  satellites.forEach((domain, index) => {
    const angle = -Math.PI / 2 + index * 2 * Math.PI / satellites.length;
    const isSun = split && domain.startsWith("evaluation");
    const spread = domain === "evaluation" ? -SUN_SPREAD : domain === "evaluation-2" ? SUN_SPREAD : 0;
    const reach = isSun ? orbit + BLAST_DISTANCE : orbit;
    offsets.set(domain, {x: reach * Math.cos(angle + (isSun ? spread : 0)), y: reach * Math.sin(angle + (isSun ? spread : 0))});
  });
  babySuns.forEach((sun, domain) => {
    const sunOffset = offsets.get(sun);
    const toward = {x: 0, y: 0};
    linkWeightsFor(domain).forEach((weight, key) => {
      const target = offsets.get(key);
      if (!target || key === sun || babySuns.has(key)) return;
      const dx = target.x - sunOffset.x, dy = target.y - sunOffset.y;
      const distance = Math.hypot(dx, dy);
      if (distance === 0) return;
      toward.x += weight * dx / distance;
      toward.y += weight * dy / distance;
    });
    const outward = Math.hypot(sunOffset.x, sunOffset.y) || 1;
    const pull = Math.hypot(toward.x, toward.y) || 1;
    const dir = Math.hypot(toward.x, toward.y) ? {x: toward.x / pull, y: toward.y / pull} : {x: sunOffset.x / outward, y: sunOffset.y / outward};
    const gap = domainLayouts.get(sun).radius + domainLayouts.get(domain).radius + 12;
    offsets.set(domain, {x: sunOffset.x + dir.x * gap, y: sunOffset.y + dir.y * gap});
  });
  let reach = 0;
  offsets.forEach((offset, domain) => { reach = Math.max(reach, Math.hypot(offset.x, offset.y) + domainLayouts.get(domain).radius); });
  const extent = Math.max(380, reach + 55);
  const center = {x: extent, y: extent};
  viewExtent = extent;
  applyViewBox();
  domainLayouts.forEach((layout, domain) => {
    const offset = offsets.get(domain) || {x: 0, y: 0};
    layout.center = {x: center.x + offset.x, y: center.y + offset.y};
  });
  domainGroups.forEach(([domain, nodes]) => {
    const layout = domainLayouts.get(domain);
    nodes.forEach((node, index) => {
      const relaxed = evaluationSplit.relative.get(node.id);
      if (relaxed) {
        positions[node.id] = {x: layout.center.x + relaxed.x, y: layout.center.y + relaxed.y, depth: (relaxed.y / Math.max(1, layout.radius) + 1) / 2};
        return;
      }
      const angle = index * Math.PI * (3 - Math.sqrt(5));
      const distance = (layout.radius - 28) * Math.sqrt((index + 0.5) / Math.max(1, nodes.length));
      positions[node.id] = {x: layout.center.x + distance * Math.cos(angle), y: layout.center.y + distance * Math.sin(angle), depth: (Math.sin(angle) + 1) / 2};
    });
  });
}
function defaultFeedback() {
  return {helpful: 0, corrections: 0, weak: 0, recalls: 0};
}
function tokenize(text) {
  return String(text || "").toLowerCase().match(/[a-z0-9]+/g) || [];
}
function nodeTopics(node) {
  return Array.isArray(node.topics) ? node.topics : [];
}
function nodeSearchText(node) {
  return [node.id, node.content, node.kind, node.role || "", nodeTopics(node).join(" ")].join(" ").toLowerCase();
}
async function loadSnapshot() {
  try {
    const response = await fetch(graphStateApi, { headers: { Accept: "application/json" } });
    if (!response.ok) return null;
    const payload = await response.json();
    if (typeof payload.updatedAt === "number") {
      state.graphStateStamp = payload.updatedAt;
    }
    return payload.snapshot || null;
  } catch {
    return null;
  }
}
async function saveSnapshot() {
  try {
    const response = await fetch(graphStateApi, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({
        nodes: state.nodes.map(node => ({
          id: node.id,
          rawImportance: node.rawImportance,
          history: node.history,
          consolidated: !!node.consolidated,
          created_cycle: node.created_cycle ?? null,
          feedback: node.feedback || defaultFeedback(),
          userImportant: !!node.userImportant
        })),
        links: state.links.map(link => ({
          source: link.source,
          target: link.target,
          strength: link.strength,
          kind: link.kind
        })),
        cycles: state.cycles,
        sidebarCollapsed: state.sidebarCollapsed,
        query: state.query
      })
    });
    if (!response.ok) {
      throw new Error(`Failed to save graph state (${response.status})`);
    }
    const payload = await response.json();
    if (typeof payload.updatedAt === "number") {
      state.graphStateStamp = payload.updatedAt;
    }
  } catch {
    // Ignore storage failures when the backend is unavailable.
  }
}
function applySnapshot(snapshot) {
  if (!snapshot) return;
  const nodesById = new Map(state.nodes.map(node => [node.id, node]));
  (snapshot.nodes || []).forEach(saved => {
    const node = nodesById.get(saved.id);
    if (!node) return;
    if (typeof saved.rawImportance === "number") node.rawImportance = saved.rawImportance;
    if (Array.isArray(saved.history)) node.history = saved.history.slice(-50);
    node.consolidated = !!saved.consolidated;
    node.feedback = saved.feedback ? {...defaultFeedback(), ...saved.feedback} : defaultFeedback();
    node.userImportant = !!saved.userImportant;
  });
  const linksByKey = new Map(state.links.map(link => [`${link.source}|${link.target}|${link.kind || ""}`, link]));
  (snapshot.links || []).forEach(saved => {
    const link = linksByKey.get(`${saved.source}|${saved.target}|${saved.kind || ""}`);
    if (link && typeof saved.strength === "number") link.strength = saved.strength;
  });
  if (typeof snapshot.cycles === "number") state.cycles = snapshot.cycles;
  if (!state.currentEvaluation) state.query = typeof snapshot.query === "string" ? snapshot.query : "";
  if (queryInput) queryInput.value = state.query;
}
async function refreshRemoteSnapshot() {
  if (state.submitting) return;
  try {
    const response = await fetch(graphStateApi, { headers: { Accept: "application/json" } });
    if (!response.ok) return;
    const payload = await response.json();
    if (state.submitting) return;
    if (!payload.snapshot) return;
    const knownIds = new Set(state.nodes.map(node => node.id));
    const knownNodes = new Map(state.nodes.map(node => [node.id, node]));
    const changedMembership = (payload.snapshot.nodes || []).some(node => {
      const current = knownNodes.get(node.id);
      return current && typeof node.domain === "string" && (current.domain !== node.domain || current.parent_domain !== node.parent_domain || current.candidate_cluster !== node.candidate_cluster);
    });
    if ((payload.snapshot.nodes || []).some(node => !knownIds.has(node.id)) || changedMembership) {
      location.reload();
      return;
    }
    if (typeof payload.updatedAt === "number" && payload.updatedAt === state.graphStateStamp) return;
    if (typeof payload.updatedAt === "number") {
      state.graphStateStamp = payload.updatedAt;
    }
    applySnapshot(payload.snapshot);
    render();
    if (state.query) {
      void refreshSearchResults();
    }
  } catch {
    // Best-effort sync; keep the local graph interactive if the backend blips.
  }
}
function el(tag, attrs = {}, text = "") {
  const item = document.createElementNS(ns, tag);
  for (const [key, value] of Object.entries(attrs)) item.setAttribute(key, value);
  if (text) item.textContent = text;
  return item;
}
function importance(node) {
  if (!node.history.length) return node.rawImportance;
  const weights = node.history.map((_, i) => 2 ** (i / Math.max(node.history.length, 1)));
  const total = weights.reduce((a, b) => a + b, 0);
  const average = node.history.reduce((sum, value, i) => sum + value * weights[i] / total, 0);
  return node.rawImportance * average;
}
function learnedScore(node) {
  const fb = node.feedback && typeof node.feedback === "object" ? node.feedback : defaultFeedback();
  const base = importance(node);
  const recency = Math.min(0.25, node.history.length * 0.02);
  const helpful = fb.helpful * 0.09;
  const correction = fb.corrections * 0.14;
  const weak = fb.weak * 0.07;
  const recall = fb.recalls * 0.03;
  const topicBoost = node.kind === "topic" ? 0.08 : 0;
  return Math.max(0.02, base * (1 + recency + helpful + recall + topicBoost - correction - weak));
}
function neighborsFor(nodeId) {
  return state.links.filter(link => link.source === nodeId || link.target === nodeId);
}
function relatedIds(nodeId) {
  const ids = new Set([nodeId]);
  neighborsFor(nodeId).forEach(link => {
    ids.add(link.source);
    ids.add(link.target);
  });
  return ids;
}
function queryMatches(node) {
  const terms = tokenize(state.query);
  if (!terms.length) return false;
  const haystack = nodeSearchText(node);
  return terms.some(term => haystack.includes(term));
}
function queryScore(node) {
  const terms = tokenize(state.query);
  if (!terms.length) return learnedScore(node);
  const haystack = nodeSearchText(node);
  const overlap = terms.reduce((sum, term) => sum + (haystack.includes(term) ? 1 : 0), 0);
  const topicHits = nodeTopics(node).reduce((sum, topic) => sum + (state.query.toLowerCase().includes(topic.toLowerCase()) ? 1 : 0), 0);
  return learnedScore(node) + overlap * 0.55 + topicHits * 0.3;
}
function makeFocusSet() {
  const focus = new Set();
  if (state.query) {
    state.results.forEach(result => {
      focus.add(result.node.id);
      relatedIds(result.node.id).forEach(id => focus.add(id));
    });
  }
  return focus;
}
function newestMessageIds() {
  const created = state.nodes.filter(node => node.kind === "message" && node.created_cycle === state.cycles).map(node => node.id);
  if (created.length) return new Set(created);
  const assistantNodes = state.nodes.filter(node => node.role === "Assistant" && typeof node.timestamp === "number");
  if (assistantNodes.length) {
    const newestTs = Math.max(...assistantNodes.map(node => node.timestamp));
    return new Set(assistantNodes.filter(node => node.timestamp === newestTs).map(node => node.id));
  }
  const newestNodes = state.nodes.filter(node => typeof node.timestamp === "number");
  if (newestNodes.length) {
    const newestTs = Math.max(...newestNodes.map(node => node.timestamp));
    return new Set(newestNodes.filter(node => node.timestamp === newestTs).map(node => node.id));
  }
  return new Set();
}
function saveAndRender() {
  void saveSnapshot();
  render();
}
function syncSidebarVisibility() {
  document.body.classList.toggle("sidebar-collapsed", state.sidebarCollapsed);
}
function syncSelectedMemoryPanel() {
  const node = state.nodes.find(item => item.id === state.selected);
  const hasSelection = Boolean(node);
  if (selectedMemoryPanel) {
    selectedMemoryPanel.hidden = !hasSelection;
  }
  if (importanceToggleButton) {
    const canMark = node?.role === "You";
    importanceToggleButton.hidden = !canMark;
    importanceToggleButton.textContent = node?.userImportant ? "Marked important" : "Mark important";
    importanceToggleButton.setAttribute("aria-pressed", node?.userImportant ? "true" : "false");
  }
}
function render() {
  syncSidebarVisibility();
  updateLayout();
  svg.replaceChildren();
  const definitions = el("defs");
  const shading = el("radialGradient", {id: "region-shading", cx: "32%", cy: "25%", r: "75%"});
  shading.append(el("stop", {offset: "0%", "stop-color": "#ffffff", "stop-opacity": ".10"}), el("stop", {offset: "50%", "stop-color": "#ffffff", "stop-opacity": ".01"}), el("stop", {offset: "100%", "stop-color": "#000000", "stop-opacity": ".18"}));
  definitions.append(shading);
  svg.append(definitions);
  const createdIds = newestMessageIds();
  const edgesBehind = el("g");
  svg.append(edgesBehind);
  const backdropLayer = el("g");
  domainLayouts.forEach((layout, domain) => {
    const nodes = state.nodes.filter(node => (node.layoutKey || "memory") === domain);
    if (!nodes.length) return;
    const style = layout.styleDomain || domain;
    const y = layout.center.y - layout.radius;
    backdropLayer.append(el("ellipse", {cx: layout.center.x + 5, cy: layout.center.y + 12, rx: layout.radius, ry: layout.radius, class: "region-shadow"}));
    const boundary = el("circle", {
      cx: layout.center.x, cy: layout.center.y, r: layout.radius,
      class: `cluster-bg ${style}`
    });
    boundary.append(el("title", {}, `${layout.label}: radius ${layout.radius.toFixed(1)}; ${layout.messages} messages; mean capped salience ${layout.salience.toFixed(2)}. Size is visual salience, not intelligence or training approval.`));
    backdropLayer.append(boundary);
    backdropLayer.append(el("circle", {cx: layout.center.x, cy: layout.center.y, r: layout.radius, fill: "url(#region-shading)", class: "region-depth"}));
    if (isBaby(domain)) {
      const hover = el("circle", {cx: layout.center.x, cy: layout.center.y, r: layout.radius + 4, fill: "rgba(0,0,0,0)"});
      hover.append(el("title", {}, `${layout.label} (${layout.messages} messages)`));
      backdropLayer.append(hover);
      return;
    }
    backdropLayer.append(el("text", {x: layout.center.x, y: y + 22, class: `cluster-title ${style}`, "text-anchor": "middle"}, layout.label));
    if (layout.subtitle) {
      backdropLayer.append(el("text", {x: layout.center.x, y: y + 38, class: `cluster-subtitle ${style}`, "text-anchor": "middle"}, layout.subtitle));
    }
  });
  svg.append(backdropLayer);
  const edges = edgesBehind;
  const layoutKeyById = new Map(state.nodes.map(node => [node.id, node.layoutKey || node.domain || "memory"]));
  const focus = makeFocusSet();
  state.links.forEach(link => {
    let a = positions[link.source], b = positions[link.target];
    const active = (state.active[link.source] || 0) > 0.05 && (state.active[link.target] || 0) > 0.05;
    const dimmed = focus.size > 0 && !(focus.has(link.source) && focus.has(link.target));
    const sourceDomain = state.nodes.find(node => node.id === link.source)?.domain || "memory";
    const targetDomain = state.nodes.find(node => node.id === link.target)?.domain || "memory";
    if (link.kind === "parent") {
      if (sourceDomain === "evaluation" || targetDomain === "evaluation") return;
      const sourceLayout = domainLayouts.get(sourceDomain);
      const targetLayout = domainLayouts.get(targetDomain);
      const dx = targetLayout.center.x - sourceLayout.center.x;
      const dy = targetLayout.center.y - sourceLayout.center.y;
      const distance = Math.hypot(dx, dy);
      if (distance === 0) return;
      a = {x: sourceLayout.center.x + dx / distance * sourceLayout.radius, y: sourceLayout.center.y + dy / distance * sourceLayout.radius};
      b = {x: targetLayout.center.x - dx / distance * targetLayout.radius, y: targetLayout.center.y - dy / distance * targetLayout.radius};
    }
    const sourceKey = layoutKeyById.get(link.source), targetKey = layoutKeyById.get(link.target);
    const sourceSun = EVAL_SUNS.includes(sourceKey), targetSun = EVAL_SUNS.includes(targetKey);
    let sunFade = null;
    if (sourceSun && targetSun && sourceKey !== targetKey) {
      if (link.strength < CROSS_SUN_MIN) return;
      sunFade = Math.min(0.4, Math.max(0.03, (link.strength - 0.2) * 0.6));
    }
    if (sourceSun !== targetSun) {
      const outside = sourceSun ? link.target : link.source;
      const sunKey = sourceSun ? sourceKey : targetKey;
      if (preferredSun.has(outside) && preferredSun.get(outside) !== sunKey) return;
    }
    const crossDomain = sourceDomain !== targetDomain;
    const created = createdIds.has(link.source) || createdIds.has(link.target) || link.created_cycle === state.cycles;
    edges.append(el("line", {
      x1: a.x, y1: a.y, x2: b.x, y2: b.y,
      class: `edge ${link.kind || "sequence"}${created ? " created" : ""}${active ? " recalled" : ""}${dimmed ? " dimmed" : ""}${crossDomain ? " cross-domain" : ""}`,
      "stroke-width": link.kind === "parent" ? 2 : 0.25 + link.strength * 1,
      "data-link": `${link.source}-${link.target}`
    }));
    const relation = link.kind === "similarity" ? "Related wording"
      : link.kind === "parent" ? `${domainLayouts.get(sourceDomain).label} grew from ${domainLayouts.get(targetDomain).label} (separate membership; visual relationship only)`
      : link.kind === "transition" ? "Subject transition (not core membership)"
      : link.kind === "topic" ? "Shared topic" : "Adjacent turn";
    edges.lastChild.append(el("title", {}, `${relation}: ${link.strength.toFixed(2)}`));
    if (sunFade !== null) edges.lastChild.style.opacity = sunFade;
  });

  [...state.nodes].sort((first, second) => positions[first.id].depth - positions[second.id].depth).forEach(node => {
    const p = positions[node.id], value = learnedScore(node);
    const isMatch = queryMatches(node);
    const dimmed = focus.size > 0 && !focus.has(node.id);
    const fb = node.feedback || defaultFeedback();
    const isCreated = createdIds.has(node.id);
    const group = el("g", {
      class: `node${state.selected === node.id ? " selected" : ""}${node.consolidated ? " consolidated" : ""}${node.userImportant ? " user-important" : ""}${node.kind === "topic" ? " topic" : ""}${node.domain ? ` ${node.domain}` : ""}${isCreated ? " created" : ""}${isMatch ? " match" : ""}${dimmed ? " dimmed" : ""}`,
      transform: `translate(${p.x} ${p.y})`,
      tabindex: "0",
      role: "button",
      "aria-label": `${node.id}, score ${value.toFixed(2)}. Click to activate.`
    });
    group.append(el("circle", {class: `pulse-ring${(state.active[node.id] || 0) > 0.05 || isCreated ? " animate" : ""}`, r: 2.75}));
    const radius = memoryRadius(node);
    const circle = el("circle", {class: "core", r: radius});
    const candidate = initialGraph.clusters?.[node.candidate_cluster];
    const evidence = candidate ? `\n${candidate.label}: ${candidate.promoted ? "Promoted" : "Under evaluation"}\n${candidate.reason}` : "";
    const title = el("title", {}, `${node.content}\nScore: ${value.toFixed(2)}\nHelpful: ${fb.helpful} · Corrections: ${fb.corrections} · Recalls: ${fb.recalls}${node.consolidated ? "\nConsolidated" : ""}${evidence}`);
    circle.append(title);
    group.append(circle);
    group.append(el("circle", {class: "node-shine", cx: -radius * .3, cy: -radius * .35, r: radius * .24}));
    group.addEventListener("click", () => activate(node.id));
    group.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); activate(node.id); }
    });
    svg.append(group);
  });
  updateStats();
  updateDetails();
  renderSearchResults();
  renderLearnerStatus();
  renderModelResponse();
}
function updateStats() {
  document.getElementById("node-count").textContent = state.nodes.length;
  document.getElementById("link-count").textContent = state.links.length;
  const average = state.nodes.reduce((sum, node) => sum + learnedScore(node), 0) / Math.max(state.nodes.length, 1);
  document.getElementById("avg-importance").textContent = average.toFixed(2);
  document.getElementById("cycle-count").textContent = state.cycles;
  document.getElementById("topic-count").textContent = state.nodes.filter(node => node.kind === "topic").length;
}
function updateDetails() {
  const node = state.nodes.find(item => item.id === state.selected);
  const clusterEvidence = document.getElementById("cluster-importance-evidence");
  if (!node) {
    document.getElementById("selected-name").textContent = "Pick a memory";
    document.getElementById("selected-content").textContent = "Select a message in the graph to inspect and activate it.";
    document.getElementById("importance-value").textContent = "—";
    document.getElementById("importance-meter").style.width = "0";
    document.getElementById("connection-value").textContent = "—";
    document.getElementById("connection-meter").style.width = "0";
    document.getElementById("selected-signals").innerHTML = "";
    clusterEvidence.hidden = true;
    clusterEvidence.textContent = "";
    syncSelectedMemoryPanel();
    return;
  }
  const value = learnedScore(node);
  const neighbors = state.links.filter(link => link.source === node.id || link.target === node.id);
  const fb = node.feedback || defaultFeedback();
  const clusterId = node.candidate_cluster;
  const cluster = clusterId ? initialGraph.clusters?.[clusterId] : null;
  if (cluster) {
    const members = state.nodes.filter(item => item.kind === "message" && item.role === "You" && item.candidate_cluster === clusterId);
    const marked = members.filter(item => item.userImportant).length;
    const helpful = members.reduce((sum, item) => sum + (item.feedback?.helpful || 0), 0);
    const repetition = Math.min(1, (cluster.distinct_interactions || members.length) / 8);
    const sessionEvidence = Math.min(1, (cluster.sessions || 0) / 3);
    const cohesion = Math.max(0, Math.min(1, cluster.cohesion || 0));
    const helpfulEvidence = Math.min(1, helpful / 3);
    const explicitEvidence = Math.min(1, marked / 3);
    const estimate = 0.25 * repetition + 0.20 * sessionEvidence + 0.25 * cohesion + 0.15 * helpfulEvidence + 0.15 * explicitEvidence;
    clusterEvidence.hidden = false;
    clusterEvidence.textContent = `Candidate cluster: ${cluster.label} · Importance estimate ${estimate.toFixed(2)}/1 (heuristic) · ${cluster.interactions} interactions · ${cluster.sessions} sessions · cohesion ${cohesion.toFixed(2)} · ${helpful} helpful marks · ${marked} important marks. This estimate does not promote the cluster.`;
  } else {
    clusterEvidence.hidden = true;
    clusterEvidence.textContent = "";
  }
  document.getElementById("selected-name").textContent = node.id;
  document.getElementById("selected-content").textContent = node.content;
  document.getElementById("importance-value").textContent = value.toFixed(2);
  document.getElementById("importance-meter").style.width = `${Math.max(0, Math.min(100, value * 100))}%`;
  document.getElementById("connection-value").textContent = `${neighbors.length} · ${neighbors.length ? (neighbors.reduce((s, l) => s + l.strength, 0) / neighbors.length).toFixed(2) + " avg strength" : "none"}`;
  document.getElementById("connection-meter").style.width = `${neighbors.length ? neighbors.reduce((s, l) => s + l.strength, 0) / neighbors.length * 100 : 0}%`;
  document.getElementById("selected-signals").innerHTML = [
    `<span class="signal good">Helpful ${fb.helpful}</span>`,
    `<span class="signal bad">Corrections ${fb.corrections}</span>`,
    `<span class="signal neutral">Recalls ${fb.recalls}</span>`,
    `<span class="signal">${node.consolidated ? "Consolidated" : "Not consolidated"}</span>`
  ].join("");
  syncSelectedMemoryPanel();
}
function announce(html) { document.getElementById("event").innerHTML = html; }
function localRankedNodes() {
  return state.nodes
    .map(node => ({node, score: queryScore(node)}))
    .sort((a, b) => b.score - a.score);
}
function renderSearchResults() {
  if (!searchResults) return;
  if (!state.query) {
    const ranked = localRankedNodes().slice(0, 6);
    searchResults.replaceChildren();
    ranked.forEach(result => {
      const node = result.node;
      const item = document.createElement("button");
      item.type = "button";
      item.className = "result";
      item.innerHTML = `<strong>${node.id}</strong><div class="meta">${node.content}</div><div class="meta">Score ${result.score.toFixed(2)} · ${node.kind}${nodeTopics(node).length ? ` · ${nodeTopics(node).join(", ")}` : ""}</div>`;
      item.addEventListener("click", () => activate(node.id));
      searchResults.append(item);
    });
    return;
  }
  if (state.searchStatus === "loading") {
    searchResults.innerHTML = `<div class="result"><strong>Searching local memory...</strong></div>`;
    return;
  }
  if (!state.results.length) {
    searchResults.innerHTML = `<div class="result"><strong>No direct match</strong><div class="meta">Try a different phrasing or a nearby topic.</div></div>`;
    return;
  }
  searchResults.replaceChildren();
  state.results.forEach(result => {
    const node = result.node;
    const item = document.createElement("button");
    item.type = "button";
    item.className = "result";
    const sourceLabel = state.searchStatus === "semantic" ? "Local model" : "Fallback";
    item.innerHTML = `<strong>${node.id}</strong><div class="meta">${node.role === "Assistant" ? "Assistant reply (unverified)" : "Your message"}</div><div class="meta">${node.content}</div><div class="meta">${sourceLabel} score ${result.score.toFixed(2)} · ${node.kind}${nodeTopics(node).length ? ` · ${nodeTopics(node).join(", ")}` : ""}</div>`;
    item.addEventListener("click", () => activate(node.id));
    searchResults.append(item);
    const judgmentKey = JSON.stringify([state.query.toLowerCase().trim(), node.id]);
    if (result.judgment && !revisingJudgments.has(judgmentKey)) {
      const status = document.createElement("div");
      status.className = "learner-mode";
      status.textContent = `Evaluated: ${result.judgment.label.replaceAll("_", " ")}${result.judgment.reviewed ? "" : " (pending review)"}`;
      const revise = document.createElement("button");
      revise.type = "button";
      revise.textContent = "Revise judgment";
      revise.setAttribute("aria-label", `Revise judgment: ${node.id}`);
      revise.addEventListener("click", () => { revisingJudgments.add(judgmentKey); renderSearchResults(); });
      searchResults.append(status, revise);
    } else if (state.searchStatus === "semantic") {
      const controls = document.createElement("div");
      controls.style.cssText = "display:flex;gap:8px;margin:4px 0 12px;flex-wrap:wrap";
      const query = state.query;
      [["direct_answer", "Direct answer"], ["related_context", "Related context"], ["unrelated", "Unrelated"]].forEach(([label, caption]) => {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = caption;
        button.title = label === "direct_answer" ? "Helps answer this question" : label === "related_context" ? "Connected context, but not a direct answer" : "Does not help with this question or its context";
        button.setAttribute("aria-label", `${button.textContent}: ${node.id}`);
        button.addEventListener("click", async () => {
          const approveForTraining = node.role === "You";
          if (approveForTraining && sessionStorage.getItem("training-grading-consent") !== "true") {
            if (!confirm("Use your judgments for training? Each grading click confirms review of your displayed memory/current prompt. Assistant replies remain unverified and pending review. This applies to later clicks in this tab, not all stored memories. Cancel skips this judgment.")) return;
            sessionStorage.setItem("training-grading-consent", "true");
          }
          controls.querySelectorAll("button").forEach(control => { control.disabled = true; });
          try {
            const response = await fetch("http://localhost:3001/api/learner-feedback", {
              method: "POST", headers: {"Content-Type": "application/json"},
              body: JSON.stringify({query, memory_id: node.id, label, approve_for_training: approveForTraining})
            });
            const outcome = await response.json();
            if (!response.ok) throw new Error(outcome.error || "Feedback was not saved");
            result.judgment = {label, reviewed: approveForTraining};
            revisingJudgments.delete(judgmentKey);
            renderSearchResults();
            state.latestJudgment = {query, label, outcome};
            const reviewStatus = outcome.training.pending_review ? ` ${outcome.training.pending_review} judgments pending review.` : "";
            const progress = outcome.training.examples != null ? ` ${outcome.training.examples}/12 eligible judgments; ${outcome.training.queries}/6 questions.` : "";
            announce(`${caption} saved. Generation ${outcome.generation}: ${outcome.training.status}.${progress}${reviewStatus}`);
            void refreshLearnerStatus();
            if (state.query === query) void refreshSearchResults();
          } catch (error) {
            document.getElementById("event").textContent = error.message;
          } finally { controls.querySelectorAll("button").forEach(control => { control.disabled = false; }); }
        });
        controls.append(button);
      });
      searchResults.append(controls);
    }
  });
}
async function refreshSearchResults() {
  const query = state.query.trim();
  const requestId = state.searchRequestId + 1;
  state.searchRequestId = requestId;

  if (!query) {
    state.searchStatus = "idle";
    state.results = [];
    render();
    return;
  }

  state.searchStatus = "loading";
  state.results = [];
  render();

  try {
    const response = await fetch(`http://localhost:3001/api/semantic-search?query=${encodeURIComponent(query)}&limit=3&exclude_id=${encodeURIComponent(state.currentEvaluation?.prompt_memory_id || "")}`, {
      headers: { Accept: "application/json" }
    });
    if (!response.ok) {
      throw new Error(`Semantic search failed (${response.status})`);
    }
    const payload = await response.json();
    if (state.searchRequestId !== requestId) return;
    state.searchGeneration = payload.generation ?? null;
    state.results = (payload.results || [])
      .map(result => {
        const node = state.nodes.find(item => item.id === result.id);
        return node ? {node, score: result.score, judgment: result.judgment || null} : null;
      })
      .filter(Boolean);
    state.searchStatus = "semantic";
    document.getElementById("evaluation-question").textContent = query;
    if (state.latestJudgment?.query !== query) announce(`Generation ${state.searchGeneration}: ${state.results.length} conversation memories. Choose a judgment below.`);
    render();
  } catch (error) {
    if (state.searchRequestId !== requestId) return;
    const visible = localRankedNodes()
      .filter(result => result.score > 0.15 || queryMatches(result.node))
      .slice(0, 6);
    state.results = visible;
    state.searchStatus = "fallback";
    render();
  }
}
function recallLevels(seedId, depth = 2, threshold = 0.05) {
  const activated = {[seedId]: 1};
  let frontier = [[seedId, 1]], visited = new Set([seedId]);
  while (frontier.length && depth > 0) {
    const next = [];
    for (const [id, level] of frontier) {
      const node = state.nodes.find(item => item.id === id);
      state.links.forEach(link => {
        const neighborId = link.source === id ? link.target : link.target === id ? link.source : null;
        if (!neighborId) return;
        const neighbor = state.nodes.find(item => item.id === neighborId);
        const nextLevel = level * link.strength * (importance(node) / Math.max(node.rawImportance, 0.01));
        if (!visited.has(neighborId) && nextLevel > threshold) {
          visited.add(neighborId);
          activated[neighborId] = nextLevel;
          next.push([neighborId, nextLevel]);
        }
      });
    }
    frontier = next;
    depth--;
  }
  return activated;
}
function activate(nodeId) {
  const node = state.nodes.find(item => item.id === nodeId);
  if (!node) return;
  node.history.push(1);
  if (node.history.length > 50) node.history = node.history.slice(-50);
  node.feedback = node.feedback || defaultFeedback();
  node.feedback.recalls += 1;
  state.links.forEach(link => {
    if (link.source === nodeId || link.target === nodeId) link.strength = Math.min(1, link.strength + 0.18);
  });
  state.selected = nodeId;
  state.lastActivated = nodeId;
  state.active = recallLevels(nodeId);
  saveAndRender();
  const spread = Object.entries(state.active).filter(([id]) => id !== nodeId);
  announce(`<strong>${nodeId}</strong> recalled. ${spread.length} related ${spread.length === 1 ? "memory lit up" : "memories lit up"}; adjacent links strengthened through co-activation.`);
}
function decay() {
  state.nodes.forEach(node => {
    const current = learnedScore(node);
    const factor = 0.15 / Math.max(current, 0.1);
    node.rawImportance *= Math.max(0, 1 - factor);
  });
  state.links.forEach(link => { if (link.strength > 0.01) link.strength *= 0.95; else link.strength = 0; });
  state.cycles++;
  state.active = {};
  saveAndRender();
  announce(`<strong>Cycle ${state.cycles}</strong> completed. Memories lost some importance; unused links weakened by 5%.`);
}
function consolidate() {
  let count = 0;
  state.nodes.forEach(node => {
    if (!node.consolidated && learnedScore(node) >= 0.7) {
      node.consolidated = true;
      node.rawImportance *= 1.2;
      count++;
    }
  });
  saveAndRender();
  announce(count
    ? `<strong>${count} ${count === 1 ? "memory" : "memories"} consolidated.</strong> Strong memories are marked green and get a 20% importance boost.`
    : "No unconsolidated memories meet the 0.70 importance threshold yet. Recall a memory to raise its activation.");
}
function recordFeedback(kind) {
  const node = state.nodes.find(item => item.id === state.selected);
  if (!node) {
    announce("Pick a memory first, then mark how it should change.");
    return;
  }
  node.feedback = node.feedback || defaultFeedback();
  const related = neighborsFor(node.id);
  if (kind === "helpful") {
    node.feedback.helpful += 1;
    node.rawImportance = Math.min(3, node.rawImportance * 1.05 + 0.02);
    related.forEach(link => { link.strength = Math.min(1, link.strength + 0.04); });
    announce(`<strong>${node.id}</strong> marked helpful. It will rank higher next time.`);
  } else if (kind === "correction") {
    node.feedback.corrections += 1;
    node.rawImportance = Math.max(0.03, node.rawImportance * 0.92);
    related.forEach(link => { link.strength = Math.max(0, link.strength - 0.05); });
    announce(`<strong>${node.id}</strong> marked as needing correction. It will be down-weighted.`);
  } else if (kind === "weak") {
    node.feedback.weak += 1;
    related.forEach(link => { link.strength = Math.max(0, link.strength - 0.08); });
    announce(`<strong>${node.id}</strong> link strength reduced. Nearby associations will fade faster.`);
  }
  saveAndRender();
}
function reset() {
  state.nodes = original.nodes.map(n => ({...n, rawImportance: n.importance, history: [], feedback: n.feedback || defaultFeedback()}));
  state.links = original.links.map(l => ({...l}));
  state.cycles = 0; state.selected = null; state.active = {}; state.lastActivated = null; state.query = "";
  state.results = [];
  if (queryInput) queryInput.value = "";
  document.getElementById("selected-name").textContent = "Pick a memory";
  document.getElementById("selected-content").textContent = "Select a message in the graph to inspect and activate it.";
  document.getElementById("importance-value").textContent = "—";
  document.getElementById("importance-meter").style.width = "0";
  document.getElementById("connection-value").textContent = "—";
  document.getElementById("connection-meter").style.width = "0";
  void saveSnapshot();
  render();
  announce("Simulation reset to the original memory graph.");
}
importanceToggleButton.addEventListener("click", () => {
  const node = state.nodes.find(item => item.id === state.selected);
  if (!node || node.role !== "You") return;
  node.userImportant = !node.userImportant;
  saveAndRender();
  announce(node.userImportant ? `<strong>${node.id}</strong> marked important.` : `<strong>${node.id}</strong> importance mark removed.`);
});
document.getElementById("zoom-in").addEventListener("click", () => changeZoom(1.25));
document.getElementById("zoom-out").addEventListener("click", () => changeZoom(0.8));
document.getElementById("response-useful").addEventListener("click", () => recordResponseFeedback("useful"));
document.getElementById("response-needs-correction").addEventListener("click", () => recordResponseFeedback("needs_correction"));
document.getElementById("training-prompt-form").addEventListener("submit", async event => {
  event.preventDefault();
  if (state.submitting) return;
  const prompt = queryInput.value.trim();
  state.query = prompt;
  queryInput.value = "";
  state.latestJudgment = null;
  state.results = [];
  document.getElementById("evaluation-question").textContent = state.query;
  state.submitting = true;
  queryInput.readOnly = true;
  announce("Evaluating and saving this question...");
  try {
    let sessionId = sessionStorage.getItem("memory-trainer-session");
    if (!sessionId) { sessionId = crypto.randomUUID(); sessionStorage.setItem("memory-trainer-session", sessionId); }
    const response = await fetch("http://localhost:3001/api/chat", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({message: prompt, training_mode: true, sessionId, include_in_generator: true})
    });
    const result = await response.json();
    if (!response.ok || !result.prompt_memory_id || !result.response_memory_id) throw new Error(result.error || "Question capture failed");
    state.currentEvaluation = {...result, query: prompt};
    sessionStorage.setItem("latest-memory-evaluation", JSON.stringify(state.currentEvaluation));
    queryInput.value = "";
    sessionStorage.setItem("clear-training-prompt", "true");
    await saveSnapshot();
    location.reload();
  } catch (error) {
    state.query = prompt;
    queryInput.value = prompt;
    sessionStorage.removeItem("clear-training-prompt");
    document.getElementById("event").textContent = error.message;
  }
  finally { state.submitting = false; queryInput.readOnly = false; }
});
queryInput.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    document.getElementById("training-prompt-form").requestSubmit();
  }
});
async function initialize() {
  const snapshot = await loadSnapshot();
  if (snapshot) {
    applySnapshot(snapshot);
  } else {
    void saveSnapshot();
  }
  if (sessionStorage.getItem("clear-training-prompt") === "true") {
    queryInput.value = "";
    sessionStorage.removeItem("clear-training-prompt");
  }
  render();
  if (state.query) {
    void refreshSearchResults();
  }
  window.setInterval(() => {
    void refreshRemoteSnapshot();
  }, 2000);
  void refreshLearnerStatus();
  window.setInterval(() => { void refreshLearnerStatus(); }, 5000);
  void refreshGeneratorStatus();
  window.setInterval(() => { void refreshGeneratorStatus(); }, 5000);
}
void initialize();
</script>
</body>
</html>
"""


def main() -> None:
    memory_path = Path(__file__).with_name("chat_memory.json")
    state_path = Path(__file__).with_name("memory_graph_state.json")
    transcript = json.loads(memory_path.read_text(encoding="utf-8"))
    clusters = update_clusters(transcript)
    memory_path.write_text(json.dumps(transcript, indent=2) + "\n", encoding="utf-8")
    graph, seed_id = build_chat_graph(memory_path)
    data = graph.to_dict()
    latest_cycle = transcript.get("cycles", 0)
    message_meta = {message["id"]: message for message in transcript["messages"]}
    message_created_cycles = {}
    for index, message in enumerate(transcript["messages"]):
        created_cycle = message.get("created_cycle")
        if created_cycle is None and latest_cycle and index == len(transcript["messages"]) - 1:
            created_cycle = latest_cycle
        message_created_cycles[message["id"]] = created_cycle
    for node in data["nodes"]:
        if node["kind"] == "message":
            source = message_meta.get(node["id"])
            if source:
                node["role"] = source.get("role")
                node["text"] = source.get("text")
                node["topics"] = source.get("topics", [])
                node["domain"] = source["domain"]
                node["parent_domain"] = source["parent_domain"]
                node["candidate_cluster"] = source.get("candidate_cluster")
                if message_created_cycles.get(node["id"]) is not None:
                    node["created_cycle"] = message_created_cycles[node["id"]]
    for link in data["links"]:
        if (
            message_created_cycles.get(link["source"]) == latest_cycle
            or message_created_cycles.get(link["target"]) == latest_cycle
        ):
            link["created_cycle"] = latest_cycle
    apply_graph_domains(data, transcript)
    data["seed_id"] = seed_id
    data["clusters"] = clusters
    data["cycles"] = transcript.get("cycles", 0)
    current_state = None
    if state_path.exists():
        try:
            current_state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current_state = None

    if current_state:
        nodes_by_id = {node["id"]: node for node in current_state.get("nodes", []) if "id" in node}
        links_by_key = {
            (link.get("source"), link.get("target"), link.get("kind")): link
            for link in current_state.get("links", [])
            if link.get("source") is not None and link.get("target") is not None
        }
        for node in data["nodes"]:
            preserved = nodes_by_id.get(node["id"])
            if not preserved:
                continue
            for key in ("rawImportance", "history", "consolidated", "feedback"):
                if key in preserved:
                    node[key] = preserved[key]
        for link in data["links"]:
            preserved = links_by_key.get((link.get("source"), link.get("target"), link.get("kind")))
            if not preserved:
                continue
            for key in ("strength", "created_cycle"):
                if key in preserved:
                    link[key] = preserved[key]
        data["sidebarCollapsed"] = current_state.get("sidebarCollapsed", False)
        data["query"] = current_state.get("query", "")
    else:
        data["sidebarCollapsed"] = False
        data["query"] = ""

    for node in data["nodes"]:
        node.setdefault("rawImportance", node.get("importance", 0.0))
        node.setdefault("history", [])
        node.setdefault("consolidated", False)
        node.setdefault("feedback", "neutral")
        if node.get("kind") == "message" and node.get("created_cycle") is None:
            node["created_cycle"] = message_created_cycles.get(node["id"])

    state_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    graph_json = json.dumps(data, separators=(",", ":"))
    graph_json = graph_json.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    version = json.loads(Path(__file__).with_name("package.json").read_text(encoding="utf-8"))["version"]
    html = HTML.replace("__APP_VERSION__", escape(version, quote=True)).replace("__GRAPH_DATA__", graph_json)
    output = Path(__file__).with_name("memory_graph.html")
    output.write_text(html, encoding="utf-8")
    print(
        f"Created {output} with {len(data['nodes'])} chat messages "
        f"and {len(data['links'])} connections from {memory_path.name}."
    )


if __name__ == "__main__":
    main()
