"""Append one conversation exchange to the local chat memory and age prior memories."""

import argparse
import json
from itertools import combinations
from pathlib import Path

from generate_html_graph import build_chat_graph, main as generate_visualization
from memory_links import infer_similarity_links
from memory_system import DEFAULT_DECAY_RATE


def add_cycle(
    memory_path: Path,
    user_text: str,
    assistant_text: str,
    user_topics: list[str],
    assistant_topics: list[str],
    session_id: str | None = None,
    origin: str = "app",
) -> int:
    transcript = json.loads(memory_path.read_text(encoding="utf-8"))
    messages = transcript["messages"]
    links = transcript.get("links")
    cycle = transcript.get("cycles", 0) + 1

    if links is None:
        graph, _ = build_chat_graph(memory_path)
        links = [
            {"source": link.source_id, "target": link.target_id, "strength": link.strength}
            for link in graph.links.values()
        ]
    link_weights = {
        (link["source"], link["target"]): link["strength"]
        for link in links
    }
    link_kinds = {}
    link_created_cycle = {
        (link["source"], link["target"]): link.get("created_cycle")
        for link in links
        if link.get("created_cycle") is not None
    }
    message_by_id = {message["id"]: message for message in messages}
    message_positions = {message["id"]: index for index, message in enumerate(messages)}
    for link in links:
        key = (link["source"], link["target"])
        source, target = key
        is_sequence = abs(message_positions[source] - message_positions[target]) == 1
        shares_topic = bool(
            set(message_by_id[source]["topics"]) & set(message_by_id[target]["topics"])
        )
        link_kinds[key] = link.get(
            "kind",
            "sequence" if is_sequence else "topic" if shares_topic else "similarity",
        )

    for message in messages:
        importance = message.get("importance", 0.9 if message["role"] == "You" else 0.8)
        decay_factor = DEFAULT_DECAY_RATE / max(importance, 0.1)
        message["importance"] = importance * max(0.0, 1.0 - decay_factor)
    for key in link_weights:
        link_weights[key] *= 0.95

    user_number = sum(message["role"] == "You" for message in messages) + 1
    assistant_number = sum(message["role"] == "Assistant" for message in messages) + 1
    new_messages = [
        {
            "id": f"You {user_number}",
            "role": "You",
            "text": user_text,
            "topics": user_topics,
            "importance": 0.9,
            "consolidated": False,
        },
        {
            "id": f"AI {assistant_number}",
            "role": "Assistant",
            "text": assistant_text,
            "topics": assistant_topics,
            "importance": 0.8,
            "consolidated": False,
            "created_cycle": cycle,
        },
    ]
    previous = messages[-1]
    for message in new_messages:
        message["created_cycle"] = cycle
        message["source"] = origin
        if session_id:
            message["session_id"] = session_id
        key = (previous["id"], message["id"])
        link_weights[key] = max(link_weights.get(key, 0), 0.35)
        link_kinds.setdefault(key, "sequence")
        link_created_cycle.setdefault(key, cycle)
        messages.append(message)
        previous = message

    new_message_ids = {message["id"] for message in new_messages}
    message_positions = {message["id"]: index for index, message in enumerate(messages)}
    for new_message in new_messages:
        for existing in messages:
            if existing["id"] == new_message["id"] or (
                existing["id"] in new_message_ids
                and message_positions[existing["id"]] > message_positions[new_message["id"]]
            ):
                continue
            shared_topics = set(existing["topics"]) & set(new_message["topics"])
            if shared_topics:
                key = (existing["id"], new_message["id"])
                weight = min(0.95, 0.35 + 0.15 * len(shared_topics))
                link_weights[key] = max(link_weights.get(key, 0), weight)
                link_kinds[key] = "topic"
                link_created_cycle.setdefault(key, cycle)

    for inferred in infer_similarity_links(messages):
        key = (inferred["source"], inferred["target"])
        undirected_key = frozenset(key)
        existing_key = next(
            (stored_key for stored_key in link_weights if frozenset(stored_key) == undirected_key),
            None,
        )
        if existing_key is None:
            link_weights[key] = inferred["strength"]
            link_kinds[key] = "similarity"
            link_created_cycle[key] = cycle

    transcript["cycles"] = cycle
    transcript["links"] = [
        {
            "source": source,
            "target": target,
            "strength": strength,
            "kind": link_kinds[(source, target)],
            **(
                {"created_cycle": link_created_cycle[(source, target)]}
                if (source, target) in link_created_cycle
                else {}
            ),
        }
        for (source, target), strength in link_weights.items()
    ]
    memory_path.write_text(json.dumps(transcript, indent=2) + "\n", encoding="utf-8")
    return cycle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reindex-links",
        action="store_true",
        help="Add missing text-similarity links without adding a message or advancing decay.",
    )
    parser.add_argument("--user", help="Text of the user message to remember.")
    parser.add_argument("--assistant", help="Text of the assistant reply to remember.")
    parser.add_argument("--session-id", help="Conversation identifier for cluster evidence.")
    parser.add_argument("--source", default="app", help="Origin of this exchange.")
    parser.add_argument(
        "--user-topic",
        action="append",
        default=[],
        help="Topic tag for the user message; repeat for additional topics.",
    )
    parser.add_argument(
        "--assistant-topic",
        action="append",
        default=[],
        help="Topic tag for the assistant reply; repeat for additional topics.",
    )
    args = parser.parse_args()
    memory_path = Path(__file__).with_name("chat_memory.json")
    if args.reindex_links:
        transcript = json.loads(memory_path.read_text(encoding="utf-8"))
        links = transcript.get("links", [])
        messages = transcript["messages"]
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
        known_pairs = {frozenset((link["source"], link["target"])) for link in links}
        inferred = infer_similarity_links(transcript["messages"])
        additions = [
            link for link in inferred
            if frozenset((link["source"], link["target"])) not in known_pairs
        ]
        links.extend(additions)
        transcript["links"] = links
        memory_path.write_text(json.dumps(transcript, indent=2) + "\n", encoding="utf-8")
        generate_visualization()
        print(f"Added {len(additions)} inferred text-similarity links without advancing the cycle.")
        return

    if not args.user or not args.assistant:
        parser.error("--user and --assistant are required unless --reindex-links is used.")
    cycle = add_cycle(
        memory_path,
        args.user,
        args.assistant,
        args.user_topic,
        args.assistant_topic,
        args.session_id,
        args.source,
    )
    generate_visualization()
    print(f"Saved conversation cycle {cycle} to {memory_path.name} and refreshed memory_graph.html.")


if __name__ == "__main__":
    main()
