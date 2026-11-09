"""Subject membership and evidence-based promotion for conversation memories."""

import re
from collections import Counter
from hashlib import sha256

from memory_links import STOP_WORDS, infer_similarity_links


DOMAIN_TERMS = {
    "memory": {"memory", "memories", "graph", "model", "copilot", "cluster", "clusters", "visualization", "prompt", "prompts"},
    "food": {"food", "pizza", "meal", "meals", "recipe", "recipes", "cooking", "dinner", "lunch", "breakfast", "snack"},
    "scripture": {"scripture", "bible", "verse", "verses", "gospel", "christianity"},
    "finance": {"finance", "investing", "investment", "portfolio", "stocks", "bonds", "budget", "debt", "loan", "retirement", "crypto"},
    "entertainment": {"television", "series", "entertainment", "mcgyver", "macgyver", "trek"},
}
MIN_INTERACTIONS = 5
MIN_SESSIONS = 3
MIN_COHESION = 0.30
MIN_SUSTAINED_INTERACTIONS = 8
MIN_SUSTAINED_SPAN = 7
DISCOVERY_NOISE = STOP_WORDS | {
    "alright", "hello", "test", "yes", "okay", "sounds", "good", "need",
    "want", "let", "step", "next", "correct", "still", "enough",
}


def discovery_text(message: dict) -> str:
    parent_words = DOMAIN_TERMS.get(message["parent_domain"], set())
    terms = re.findall(r"[a-z0-9]+", message["text"].lower())
    return " ".join(term for term in terms if len(term) > 2 and not term.isdigit() and term not in DISCOVERY_NOISE and term not in parent_words)


def apply_graph_domains(data: dict, transcript: dict) -> None:
    nodes_by_id = {node["id"]: node for node in data["nodes"]}
    for node in data["nodes"]:
        if node["kind"] != "topic":
            continue
        topic = node["content"].removeprefix("Topic: ")
        node["topic"] = topic
        owners = [
            message["domain"]
            for message in transcript["messages"]
            if topic in message.get("topics", [])
        ]
        node["domain"] = max(set(owners), key=owners.count) if owners else "evaluation"
    originals = {
        (link["source"], link["target"]): link
        for link in transcript.get("links", [])
    }
    for link in data["links"]:
        original = originals.get((link["source"], link["target"]))
        if original:
            link["kind"] = original.get("kind", "sequence")
        source = nodes_by_id[link["source"]]
        target = nodes_by_id[link["target"]]
        if link.get("kind") == "sequence" and source.get("domain") != target.get("domain"):
            link["kind"] = "transition"
    for cluster_id, cluster in transcript.get("clusters", {}).items():
        if not cluster["promoted"]:
            continue
        parent = cluster["parent_domain"]
        parent_id = "parent-domain-" + parent
        if parent != "evaluation" and parent_id not in nodes_by_id:
            parent_node = {"id": parent_id, "kind": "topic", "content": "Domain: " + parent.title(), "domain": parent, "importance": 0.6, "feedback": {"helpful": 0, "corrections": 0, "weak": 0, "recalls": 0}}
            data["nodes"].append(parent_node)
            nodes_by_id[parent_id] = parent_node
        hub_id = cluster_id + "-hub"
        hub = {"id": hub_id, "kind": "topic", "content": "Topic: " + cluster["label"], "domain": cluster_id, "parent_domain": cluster_id if parent == "evaluation" else parent, "importance": 0.7, "feedback": {"helpful": 0, "corrections": 0, "weak": 0, "recalls": 0}}
        data["nodes"].append(hub)
        nodes_by_id[hub_id] = hub
        if parent != "evaluation":
            data["links"].append({"source": hub_id, "target": parent_id, "kind": "parent", "strength": 0.5})
        for node in list(data["nodes"]):
            if node.get("kind") == "message" and node.get("domain") == cluster_id:
                data["links"].append({"source": node["id"], "target": hub_id, "kind": "hub", "strength": 0.5})
def subject_for_message(message: dict) -> str:
    tokens = set(re.findall(r"[a-z0-9]+", message.get("text", "").lower()))
    topics = set(message.get("topics", []))
    scores = {
        domain: len(tokens & terms) + 2 * len(topics & (terms | {domain}))
        for domain, terms in DOMAIN_TERMS.items()
    }
    ai_development = "ai" in tokens and bool(tokens & {"creation", "development", "dev", "build", "building", "training", "train", "learner"})
    if scores["memory"] or ai_development:
        return "memory"
    best = max(scores, key=scores.get)
    return best if scores[best] else "evaluation"


def is_contextual_followup(text: str) -> bool:
    normalized = " ".join(re.findall(r"[a-z0-9]+", text.lower()))
    return normalized in {
        "yes", "okay", "ok", "alright go", "alright then", "go ahead",
        "sounds good let s do it", "sounds good lets do it", "let s do it",
        "so where are we at now", "where are we at now", "next step",
        "what s next", "i don t know what that means",
    }


def update_clusters(transcript: dict) -> dict[str, dict]:
    messages = transcript["messages"]
    users = []
    current_subject = "evaluation"
    session_subjects = {}
    for position, message in enumerate(messages):
        message.pop("candidate_cluster", None)
        if message["role"] == "You":
            current_subject = subject_for_message(message)
            session_id = message.get("session_id")
            if current_subject == "evaluation" and session_id and is_contextual_followup(message["text"]):
                current_subject = session_subjects.get(session_id, "evaluation")
            if session_id:
                session_subjects[session_id] = current_subject
            message["parent_domain"] = current_subject
            text = discovery_text(message)
            if len(set(text.split())) >= 2:
                users.append({**message, "text": text, "position": position})
        message["domain"] = current_subject
        message["parent_domain"] = current_subject

    links = infer_similarity_links(users, max_neighbors=len(users), min_similarity=MIN_COHESION)
    adjacency = {message["id"]: set() for message in users}
    scores = {}
    for link in links:
        adjacency[link["source"]].add(link["target"])
        adjacency[link["target"]].add(link["source"])
        scores[frozenset((link["source"], link["target"]))] = link["similarity"]

    previous = transcript.get("clusters", {})
    clusters = {}
    remaining = set(adjacency)
    while remaining:
        seed = min(remaining)
        component = {seed}
        pending = [seed]
        while pending:
            for neighbor in adjacency[pending.pop()] - component:
                component.add(neighbor)
                pending.append(neighbor)
        remaining -= component
        if len(component) < 2:
            continue
        members = [message for message in users if message["id"] in component]
        matching = next(
            (key for key, cluster in previous.items() if cluster["anchor"] in component),
            None,
        )
        anchor = previous[matching]["anchor"] if matching else members[0]["id"]
        cluster_id = matching or "cluster-" + sha256(anchor.encode()).hexdigest()[:12]
        terms = Counter(
            token
            for message in members
            for token in set(re.findall(r"[a-z0-9]+", message["text"].lower()))
            if len(token) > 2 and token not in STOP_WORDS
        )
        label = previous[matching]["label"] if matching and previous[matching]["promoted"] else " / ".join(
            term for term, _ in sorted(terms.items(), key=lambda item: (-item[1], item[0]))[:2]
        ).title()
        sessions = {message["session_id"] for message in members if message.get("session_id")}
        distinct = len({message["text"] for message in members})
        span = max(message["position"] for message in members) - min(message["position"] for message in members)
        sustained = False
        for session_id in sessions:
            session_members = [message for message in members if message.get("session_id") == session_id]
            session_distinct = len({message["text"] for message in session_members})
            session_span = max(message["position"] for message in session_members) - min(message["position"] for message in session_members)
            if session_distinct >= MIN_SUSTAINED_INTERACTIONS and session_span >= MIN_SUSTAINED_SPAN:
                sustained = True
        parent_counts = Counter(message["parent_domain"] for message in members)
        parent = sorted(parent_counts, key=lambda domain: (-parent_counts[domain], domain == "evaluation", domain))[0]
        pairs = len(members) * (len(members) - 1) / 2
        cohesion = sum(value for pair, value in scores.items() if pair <= component) / pairs
        promoted = (
            len(members) >= MIN_INTERACTIONS
            and (len(sessions) >= MIN_SESSIONS or sustained)
            and cohesion >= MIN_COHESION
        )
        clusters[cluster_id] = {
            "anchor": anchor,
            "label": label,
            "members": [message["id"] for message in members],
            "interactions": len(members),
            "sessions": len(sessions),
            "parent_domain": parent,
            "distinct_interactions": distinct,
            "interaction_span": span,
            "promotion_path": "sustained-discussion" if sustained else "multiple-sessions",
            "cohesion": round(cohesion, 4),
            "promoted": promoted,
            "reason": f"{len(members)} user interactions ({distinct} distinct), {len(sessions)} sessions, span {span}, cohesion {cohesion:.2f}",
        }
        for message in messages:
            if message["id"] in component:
                message["candidate_cluster"] = cluster_id

    for message in messages:
        message.pop("cluster_id", None)
    active_cluster = None
    for message in messages:
        if message["role"] == "You":
            candidate = message.get("candidate_cluster")
            active_cluster = candidate if candidate in clusters and clusters[candidate]["promoted"] else None
        if active_cluster:
            message["domain"] = active_cluster
            message["cluster_id"] = active_cluster
            if message["parent_domain"] == "evaluation" and clusters[active_cluster]["parent_domain"] == "evaluation":
                message["parent_domain"] = active_cluster
    transcript["clusters"] = clusters
    return clusters