"""Generate a self-contained interactive visualization of the chat memory graph."""

import json
import re
from itertools import combinations
from pathlib import Path

from memory_system import MemoryGraph
from memory_links import infer_similarity_links


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
    main { max-width: 1180px; margin: 0 auto; }
    body.sidebar-collapsed main { max-width: 1400px; }
    header { display: flex; justify-content: space-between; align-items: end; gap: 20px; margin-bottom: 24px; }
    .eyebrow { color: var(--cyan); text-transform: uppercase; letter-spacing: .16em; font-size: 11px; font-weight: 750; }
    h1 { font-size: clamp(30px, 5vw, 46px); letter-spacing: -.045em; line-height: 1.05; margin: 8px 0 8px; }
    header p { color: var(--muted); margin: 0; max-width: 650px; line-height: 1.55; }
    .live-pill { white-space: nowrap; border: 1px solid rgba(110,231,183,.25); color: var(--green); border-radius: 30px; padding: 9px 13px; font-size: 12px; background: rgba(110,231,183,.07); }
    .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px; }
    .stat, .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 16px; box-shadow: 0 18px 60px rgba(0,0,0,.18); }
    .stat { padding: 15px 18px; }
    .stat-label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .1em; }
    .stat-value { margin-top: 5px; font-size: 25px; font-weight: 700; font-variant-numeric: tabular-nums; }
    .workspace { display: grid; grid-template-columns: minmax(0, 1fr) 332px; gap: 16px; align-items: stretch; }
    body.sidebar-collapsed .workspace { grid-template-columns: minmax(0, 1fr); }
    .graph-panel { position: relative; min-width: 0; padding: 15px; overflow: hidden; }
    .graph-head { display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 4px 7px 0; }
    .graph-actions { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    h2 { font-size: 14px; margin: 0; letter-spacing: .01em; }
    .hint { font-size: 11px; color: var(--muted); }
    svg { display: block; width: 100%; height: auto; min-height: 330px; overflow: visible; }
    .edge { stroke: #64749b; stroke-linecap: round; opacity: .5; transition: stroke .25s, opacity .25s, stroke-width .25s; }
    .edge.topic { stroke: #2dd4bf; }
    .edge.hub { stroke: #f59e0b; stroke-dasharray: 2 4; }
    .edge.similarity { stroke: #c084fc; stroke-dasharray: 5 5; }
    .edge.created { stroke: #22d3ee; opacity: 1; filter: drop-shadow(0 0 4px rgba(34,211,238,.6)); }
    .edge.cross-domain { stroke: #fda4af; stroke-dasharray: 2 8; opacity: .07; }
    .edge.recalled { stroke: #67e8f9; opacity: .98; filter: drop-shadow(0 0 4px rgba(103,232,249,.7)); }
    .edge-label { fill: #9ba9c6; font-size: 11px; text-anchor: middle; paint-order: stroke; stroke: #11192e; stroke-width: 5px; stroke-linejoin: round; }
    .node { cursor: pointer; outline: none; }
    .node circle.core { fill: #24304c; stroke: #9fb4e6; stroke-width: .45; transition: fill .2s, stroke .2s, r .2s; }
    .node:hover circle.core, .node:focus circle.core { stroke: white; }
    .node.selected circle.core { fill: #0f5e78; stroke: #67e8f9; stroke-width: .7; }
    .node.consolidated circle.core { stroke: var(--green); }
    .node.created circle.core { fill: #0f5e78 !important; stroke: #ecfeff !important; stroke-width: .8 !important; filter: drop-shadow(0 0 6px rgba(34,211,238,.28)); }
    .node.topic circle.core { fill: #26324f; stroke: #b0c4ff; stroke-width: .35; opacity: .98; }
    .node.topic.selected circle.core { fill: #0f5e78; stroke: #67e8f9; stroke-width: .7; }
    .node.dimmed { opacity: .18; }
    .node.match circle.core { stroke: var(--violet); filter: drop-shadow(0 0 7px rgba(167,139,250,.4)); }
    .node.related circle.core { stroke: #7be0c4; }
    .node.memory circle.core { fill: #24304c; stroke: #93c5fd; }
    .node.scripture circle.core { fill: #5b2143; stroke: #f472b6; }
    .node.topic.scripture circle.core { fill: #7a2e59; stroke: #f9a8d4; }
    .node.food circle.core { fill: #23421f; stroke: #4ade80; }
    .node.topic.food circle.core { fill: #2f5b2a; stroke: #86efac; }
    .node.finance circle.core { fill: #3f2f12; stroke: #f59e0b; }
    .node.topic.finance circle.core { fill: #63471d; stroke: #fbbf24; }
    .node.evaluation circle.core { fill: #46316d; stroke: #c084fc; }
    .node.topic.evaluation circle.core { fill: #5b3d8c; stroke: #ddd6fe; }
    .node.entertainment circle.core { fill: #1f3d62; stroke: #60a5fa; }
    .node.topic.entertainment circle.core { fill: #26507f; stroke: #bfdbfe; }
    .node.bridge circle.core { stroke-dasharray: 2 3; }
    .cluster-bg { fill: rgba(14, 22, 39, .68); stroke: rgba(165,180,220,.16); stroke-width: 1.2; rx: 18; ry: 18; }
    .cluster-bg.memory { fill: rgba(11, 32, 52, .84); stroke: rgba(103,232,249,.32); }
    .cluster-bg.scripture { fill: rgba(56, 18, 47, .80); stroke: rgba(244,114,182,.44); stroke-dasharray: 7 6; }
    .cluster-bg.food { fill: rgba(20, 42, 22, .80); stroke: rgba(134,239,172,.40); stroke-dasharray: 8 7; }
    .cluster-bg.finance { fill: rgba(58, 37, 13, .80); stroke: rgba(251,191,36,.42); stroke-dasharray: 6 7; }
    .cluster-bg.evaluation { fill: rgba(41, 27, 64, .80); stroke: rgba(196,181,253,.42); stroke-dasharray: 5 8; }
    .cluster-bg.entertainment { fill: rgba(17, 34, 58, .80); stroke: rgba(147,197,253,.42); stroke-dasharray: 6 8; }
    .cluster-bg.other { fill: rgba(52, 34, 11, .66); stroke: rgba(251,191,36,.30); }
    .cluster-title { fill: #eff6ff; font-size: 12px; font-weight: 800; letter-spacing: .05em; text-transform: uppercase; }
    .cluster-title.scripture { fill: #f9a8d4; }
    .cluster-title.food { fill: #86efac; }
    .cluster-title.finance { fill: #fcd34d; }
    .cluster-title.evaluation { fill: #c4b5fd; }
    .cluster-title.entertainment { fill: #93c5fd; }
    .cluster-title.memory { fill: #67e8f9; }
    .pulse-ring { fill: none; stroke: var(--cyan); stroke-width: 2; opacity: 0; pointer-events: none; }
    .pulse-ring.animate { animation: ripple 1.15s ease-out infinite; }
    @keyframes ripple { 0% { r: 22px; opacity: .85; } 100% { r: 58px; opacity: 0; } }
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
    @media (max-width: 850px) { .workspace { grid-template-columns: 1fr; } aside { display: grid; grid-template-columns: 1fr 1fr; } aside > :last-child { grid-column: 1 / -1; } .graph-head { flex-wrap: wrap; } .graph-actions { margin-left: auto; } }
    @media (max-width: 560px) { body { padding: 24px 12px; } header { display: block; } .live-pill { display: inline-block; margin-top: 15px; } .stats { grid-template-columns: repeat(2, 1fr); } .stat { padding: 12px 14px; } .stat-value { font-size: 21px; } aside { display: flex; } svg { min-height: 265px; } }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <div class="eyebrow">Interactive memory model</div>
      <h1>Our chat, in memory</h1>
      <p>Each message is a memory. Nearby turns and shared topics form connections; recall spreads through them, while unused memories fade.</p>
    </div>
    <div class="live-pill">● LEARNING STATE ON</div>
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
        <h2>Conversation memory graph</h2>
        <div class="graph-actions">
          <button type="button" id="sidebar-toggle-button">Hide right menu</button>
        </div>
      </div>
      <svg id="graph" viewBox="0 0 1120 760" role="img" aria-label="Interactive graph of connected memories"></svg>
    </div>
    <aside class="panel">
      <section id="selected-memory-panel">
        <div class="selected-memory-head">
          <div class="section-label">Selected memory</div>
          <div class="selected-memory-actions">
            <button type="button" id="selected-sidebar-toggle-button">Hide right menu</button>
          </div>
        </div>
        <h2 id="selected-name" class="selected-name">Pick a memory</h2>
        <div id="selected-content" class="selected-content">Select a message in the graph to inspect and activate it.</div>
        <div class="signal-row" id="selected-signals"></div>
        <div class="meter-row"><span>Importance</span><span id="importance-value">—</span></div>
        <div class="meter"><span id="importance-meter"></span></div>
        <div class="meter-row"><span>Connected ideas</span><span id="connection-value">—</span></div>
        <div class="meter"><span id="connection-meter"></span></div>
      </section>
      <section>
        <div class="section-label">Search and learn</div>
        <input id="memory-query" class="query-input" type="search" placeholder="Search memories or topics" aria-label="Search memories">
        <div id="search-results" class="search-results" aria-live="polite"></div>
      </section>
      <section>
        <div class="section-label">Run the memory</div>
        <div class="actions">
          <button class="primary" id="recall-button">Recall this chat</button>
          <button id="helpful-button">Mark helpful</button>
          <button id="correction-button">Mark correction</button>
          <button id="weaken-button">Weaken links</button>
          <button id="decay-button">Advance 1 decay cycle</button>
          <button id="consolidate-button">Consolidate strong memories</button>
          <button id="reset-button">Reset simulation</button>
        </div>
      </section>
      <section>
        <div class="section-label">What's happening</div>
        <div class="event" id="event" aria-live="polite">Ready. Click any message, or start with <strong>Recall this chat</strong>.</div>
      </section>
    </aside>
  </section>
  <footer>Source: this conversation is stored locally in chat_memory.json. Links combine turn order, shared-topic tags, learned feedback stored in a backend graph-state file, and semantic search served from the local embeddings endpoint when available. Recall, reinforcement, correction, decay, and consolidation persist across refreshes through backend storage.</footer>
</main>
<script>
const initialGraph = __GRAPH_DATA__;
const original = JSON.parse(JSON.stringify(initialGraph));
const state = {
  nodes: initialGraph.nodes.map(n => ({
    ...n,
    rawImportance: typeof n.rawImportance === "number" ? n.rawImportance : (typeof n.importance === "number" ? n.importance : 0.0),
    history: [],
    feedback: n.feedback || {helpful: 0, corrections: 0, weak: 0, recalls: 0},
    created_cycle: n.created_cycle
  })),
  links: initialGraph.links.map(l => ({...l})),
  cycles: initialGraph.cycles || 0,
  selected: null,
  sidebarCollapsed: false,
  graphStateStamp: null,
  active: {},
  lastActivated: null,
  query: "",
  results: [],
  searchStatus: "idle",
  searchRequestId: 0
};
const graphStateApi = "http://localhost:3001/api/graph-state";
const svg = document.getElementById("graph");
const ns = "http://www.w3.org/2000/svg";
const positions = {};
const queryInput = document.getElementById("memory-query");
const searchResults = document.getElementById("search-results");
const sidebarToggleButton = document.getElementById("sidebar-toggle-button");
const selectedSidebarToggleButton = document.getElementById("selected-sidebar-toggle-button");
const selectedMemoryPanel = document.getElementById("selected-memory-panel");
function domainForNode(node) {
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
const domainGroups = Object.entries(
  state.nodes.reduce((acc, node) => {
    const domain = node.domain || "memory";
    (acc[domain] ||= []).push(node);
    return acc;
  }, {})
);
function domainLayoutFor(domain, nodes) {
  const totalImportance = nodes.reduce((sum, node) => sum + (node.importance || 0), 0);
  if (domain === "scripture") {
  return { center: {x: 684, y: 214}, radius: 42, label: "Scripture", subtitle: "" };
  }
  if (domain === "food") {
    const promoted = totalImportance >= 1.8 || nodes.length >= 3;
    return promoted
      ? {
          center: {x: 802, y: 486},
          radius: 28,
          label: "Food",
          subtitle: "",
        }
      : {
          center: {x: 802, y: 486},
          radius: 28,
          label: "Food",
          subtitle: "",
        };
  }
  if (domain === "finance") {
    return {
        center: {x: 968, y: 548},
        radius: 46,
        label: "Finance",
        subtitle: "",
    };
  }
  if (domain === "evaluation") {
    return {
          center: {x: 742, y: 56},
        radius: 32,
        label: "Under evaluation",
        subtitle: "",
    };
  }
  if (domain === "entertainment") {
    return {
      center: {x: 586, y: 494},
      radius: 32,
      label: "Entertainment",
      subtitle: "",
    };
  }
  if (domain === "other") {
    return { center: {x: 872, y: 304}, radius: 44, label: "Other", subtitle: "" };
  }
  return { center: {x: 272, y: 286}, radius: 122, label: "AI memory", subtitle: "" };
}
const domainLayouts = new Map();
domainGroups.forEach(([domain, nodes]) => {
  const layout = domainLayoutFor(domain, nodes);
  domainLayouts.set(domain, layout);
  const center = layout.center;
  const radius = layout.radius;
  nodes.forEach((node, index) => {
    const angle = -Math.PI / 2 + (2 * Math.PI * index / Math.max(nodes.length, 1));
    positions[node.id] = {
      x: center.x + radius * Math.cos(angle),
      y: center.y + radius * Math.sin(angle)
    };
  });
});
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
          feedback: node.feedback || defaultFeedback()
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
  });
  const linksByKey = new Map(state.links.map(link => [`${link.source}|${link.target}|${link.kind || ""}`, link]));
  (snapshot.links || []).forEach(saved => {
    const link = linksByKey.get(`${saved.source}|${saved.target}|${saved.kind || ""}`);
    if (link && typeof saved.strength === "number") link.strength = saved.strength;
  });
  if (typeof snapshot.cycles === "number") state.cycles = snapshot.cycles;
  if (typeof snapshot.sidebarCollapsed === "boolean") state.sidebarCollapsed = snapshot.sidebarCollapsed;
  state.query = typeof snapshot.query === "string" ? snapshot.query : "";
  if (queryInput) queryInput.value = state.query;
}
async function refreshRemoteSnapshot() {
  try {
    const response = await fetch(graphStateApi, { headers: { Accept: "application/json" } });
    if (!response.ok) return;
    const payload = await response.json();
    if (!payload.snapshot) return;
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
  const fb = node.feedback || defaultFeedback();
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
  const created = state.nodes.filter(node => node.created_cycle === state.cycles).map(node => node.id);
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
  if (sidebarToggleButton) {
    sidebarToggleButton.textContent = state.sidebarCollapsed ? "Show right menu" : "Hide right menu";
    sidebarToggleButton.setAttribute(
      "aria-pressed",
      state.sidebarCollapsed ? "true" : "false"
    );
  }
  if (selectedSidebarToggleButton) {
    selectedSidebarToggleButton.textContent = state.sidebarCollapsed ? "Show right menu" : "Hide right menu";
    selectedSidebarToggleButton.setAttribute("aria-pressed", state.sidebarCollapsed ? "true" : "false");
  }
}
function syncSelectedMemoryPanel() {
  const node = state.nodes.find(item => item.id === state.selected);
  const hasSelection = Boolean(node);
  if (selectedMemoryPanel) {
    selectedMemoryPanel.hidden = !hasSelection;
  }
  if (selectedSidebarToggleButton) {
    selectedSidebarToggleButton.hidden = !hasSelection;
    selectedSidebarToggleButton.textContent = state.sidebarCollapsed ? "Show right menu" : "Hide right menu";
    selectedSidebarToggleButton.setAttribute("aria-pressed", state.sidebarCollapsed ? "true" : "false");
  }
}
function render() {
  syncSidebarVisibility();
  svg.replaceChildren();
  const createdIds = newestMessageIds();
  const backdropLayer = el("g");
  ["memory", "scripture", "food", "evaluation", "entertainment", "other"].forEach(domain => {
    const nodes = state.nodes.filter(node => (node.domain || "memory") === domain);
    if (!nodes.length) return;
    const layout = domainLayouts.get(domain) || domainLayoutFor(domain, nodes);
    const padX = 38;
    const padY = 30;
    const width = layout.radius * 2 + padX * 2;
    const height = layout.radius * 2 + padY * 2;
    const x = layout.center.x - width / 2;
    const y = layout.center.y - height / 2;
    backdropLayer.append(el("rect", {
      x, y, width, height,
      class: `cluster-bg ${domain}`
    }));
    backdropLayer.append(el("text", {x: layout.center.x, y: y + 22, class: `cluster-title ${domain}`, "text-anchor": "middle"}, layout.label));
    if (layout.subtitle) {
      backdropLayer.append(el("text", {x: layout.center.x, y: y + 38, class: `cluster-subtitle ${domain}`, "text-anchor": "middle"}, layout.subtitle));
    }
  });
  svg.append(backdropLayer);
  const edges = el("g");
  const focus = makeFocusSet();
  state.links.forEach(link => {
    const a = positions[link.source], b = positions[link.target];
    const active = (state.active[link.source] || 0) > 0.05 && (state.active[link.target] || 0) > 0.05;
    const dimmed = focus.size > 0 && !(focus.has(link.source) && focus.has(link.target));
    const sourceDomain = state.nodes.find(node => node.id === link.source)?.domain || "memory";
    const targetDomain = state.nodes.find(node => node.id === link.target)?.domain || "memory";
    const crossDomain = sourceDomain !== targetDomain;
    const created = createdIds.has(link.source) || createdIds.has(link.target) || link.created_cycle === state.cycles;
    edges.append(el("line", {
      x1: a.x, y1: a.y, x2: b.x, y2: b.y,
      class: `edge ${link.kind || "sequence"}${created ? " created" : ""}${active ? " recalled" : ""}${dimmed ? " dimmed" : ""}${crossDomain ? " cross-domain" : ""}`,
      "stroke-width": 0.25 + link.strength * 1,
      "data-link": `${link.source}-${link.target}`
    }));
    const relation = link.kind === "similarity" ? "Related wording"
      : link.kind === "topic" ? "Shared topic" : "Adjacent turn";
    edges.lastChild.append(el("title", {}, `${relation}: ${link.strength.toFixed(2)}`));
  });
  svg.append(edges);

  state.nodes.forEach(node => {
    const p = positions[node.id], value = learnedScore(node);
    const isMatch = queryMatches(node);
    const dimmed = focus.size > 0 && !focus.has(node.id);
    const fb = node.feedback || defaultFeedback();
    const isCreated = createdIds.has(node.id);
    const group = el("g", {
      class: `node${state.selected === node.id ? " selected" : ""}${node.consolidated ? " consolidated" : ""}${node.kind === "topic" ? " topic" : ""}${node.domain ? ` ${node.domain}` : ""}${isCreated ? " created" : ""}${isMatch ? " match" : ""}${dimmed ? " dimmed" : ""}`,
      transform: `translate(${p.x} ${p.y})`,
      tabindex: "0",
      role: "button",
      "aria-label": `${node.id}, score ${value.toFixed(2)}. Click to activate.`
    });
    group.append(el("circle", {class: `pulse-ring${(state.active[node.id] || 0) > 0.05 || isCreated ? " animate" : ""}`, r: 2.75}));
    const circle = el("circle", {class: "core", r: 2.25 + value * 1.75});
    const title = el("title", {}, `${node.content}\nScore: ${value.toFixed(2)}\nHelpful: ${fb.helpful} · Corrections: ${fb.corrections} · Recalls: ${fb.recalls}${node.consolidated ? "\nConsolidated" : ""}`);
    circle.append(title);
    group.append(circle);
    group.addEventListener("click", () => activate(node.id));
    group.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); activate(node.id); }
    });
    svg.append(group);
  });
  updateStats();
  updateDetails();
  renderSearchResults();
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
  if (!node) {
    document.getElementById("selected-name").textContent = "Pick a memory";
    document.getElementById("selected-content").textContent = "Select a message in the graph to inspect and activate it.";
    document.getElementById("importance-value").textContent = "—";
    document.getElementById("importance-meter").style.width = "0";
    document.getElementById("connection-value").textContent = "—";
    document.getElementById("connection-meter").style.width = "0";
    document.getElementById("selected-signals").innerHTML = "";
    syncSelectedMemoryPanel();
    return;
  }
  const value = learnedScore(node);
  const neighbors = state.links.filter(link => link.source === node.id || link.target === node.id);
  const fb = node.feedback || defaultFeedback();
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
    searchResults.innerHTML = `<div class="result"><strong>Searching semantic memory…</strong><div class="meta">Embedding the query and ranking nearby meanings.</div></div>`;
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
    const sourceLabel = state.searchStatus === "semantic" ? "Embedding" : "Fallback";
    item.innerHTML = `<strong>${node.id}</strong><div class="meta">${node.content}</div><div class="meta">${sourceLabel} score ${result.score.toFixed(2)} · ${node.kind}${nodeTopics(node).length ? ` · ${nodeTopics(node).join(", ")}` : ""}</div>`;
    item.addEventListener("click", () => activate(node.id));
    searchResults.append(item);
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
    const response = await fetch(`http://localhost:3001/api/semantic-search?query=${encodeURIComponent(query)}&limit=6`, {
      headers: { Accept: "application/json" }
    });
    if (!response.ok) {
      throw new Error(`Semantic search failed (${response.status})`);
    }
    const payload = await response.json();
    if (state.searchRequestId !== requestId) return;
    state.results = (payload.results || [])
      .map(result => {
        const node = state.nodes.find(item => item.id === result.id);
        return node ? {node, score: result.score} : null;
      })
      .filter(Boolean);
    state.searchStatus = "semantic";
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
document.getElementById("recall-button").addEventListener("click", () => activate(initialGraph.seed_id));
document.getElementById("helpful-button").addEventListener("click", () => recordFeedback("helpful"));
document.getElementById("correction-button").addEventListener("click", () => recordFeedback("correction"));
document.getElementById("weaken-button").addEventListener("click", () => recordFeedback("weak"));
document.getElementById("decay-button").addEventListener("click", decay);
document.getElementById("consolidate-button").addEventListener("click", consolidate);
document.getElementById("reset-button").addEventListener("click", reset);
sidebarToggleButton.addEventListener("click", () => {
  state.sidebarCollapsed = !state.sidebarCollapsed;
  saveAndRender();
});
selectedSidebarToggleButton.addEventListener("click", () => {
  if (!state.selected) return;
  state.sidebarCollapsed = !state.sidebarCollapsed;
  saveAndRender();
});
queryInput.addEventListener("input", event => {
  state.query = event.target.value.trim();
  state.results = [];
  void saveSnapshot();
  void refreshSearchResults();
});
async function initialize() {
  const snapshot = await loadSnapshot();
  if (snapshot) {
    applySnapshot(snapshot);
  } else {
    void saveSnapshot();
  }
  render();
  if (state.query) {
    void refreshSearchResults();
  }
  window.setInterval(() => {
    void refreshRemoteSnapshot();
  }, 2000);
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
                if message_created_cycles.get(node["id"]) is not None:
                    node["created_cycle"] = message_created_cycles[node["id"]]
        elif node["kind"] == "topic":
            node["topic"] = node["content"].removeprefix("Topic: ")
    for link in data["links"]:
        if (
            message_created_cycles.get(link["source"]) == latest_cycle
            or message_created_cycles.get(link["target"]) == latest_cycle
        ):
            link["created_cycle"] = latest_cycle
    data["seed_id"] = seed_id
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
            for key in ("rawImportance", "history", "consolidated", "feedback", "created_cycle"):
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
    html = HTML.replace("__GRAPH_DATA__", json.dumps(data, separators=(",", ":")))
    output = Path(__file__).with_name("memory_graph.html")
    output.write_text(html, encoding="utf-8")
    print(
        f"Created {output} with {len(data['nodes'])} chat messages "
        f"and {len(data['links'])} connections from {memory_path.name}."
    )


if __name__ == "__main__":
    main()
