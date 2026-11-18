"""Lightweight text-similarity links for conversation memory nodes."""

import math
import re
from collections import Counter

STOP_WORDS = {
    "about", "after", "again", "also", "and", "are", "been", "being", "but", "can",
    "could", "did", "does", "doing", "for", "from", "get", "got", "had", "has",
    "have", "her", "here", "him", "his", "how", "into", "its", "just", "like",
    "make", "made", "more", "most", "not", "now", "our", "out", "over", "same",
    "see", "should", "some", "such", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "through", "too", "under",
    "use", "used", "using", "very", "was", "were", "what", "when", "where",
    "which", "while", "who", "will", "with", "would", "you", "your",
}

SHORT_WORD = re.compile(r"[a-z0-9]+(?:['’-][a-z0-9]+)*")
CONTEXT_NEGATION = {"no", "not", "never", "neither", "nor", "without", "cannot", "can't", "dont", "don't"}


def short_words(text: str) -> list[str]:
    return SHORT_WORD.findall(text.lower().replace("’", "'"))


def context_terms(text: str) -> set[str]:
    return {word for word in short_words(text) if len(word) > 2 and word not in STOP_WORDS}


def evaluation_exchange_pairs(messages: list[dict]) -> list[tuple[dict, dict]]:
    """Find unambiguous user/assistant exchanges in the evaluation domain."""
    cycles = Counter(message.get("created_cycle") for message in messages
                     if message.get("created_cycle") is not None)
    pairs = []
    for user, assistant in zip(messages, messages[1:]):
        cycle = user.get("created_cycle")
        if (user.get("role") != "You" or assistant.get("role") != "Assistant"
                or type(cycle) is not int or cycle <= 0 or assistant.get("created_cycle") != cycle
                or cycles[cycle] != 2 or user.get("session_id") != assistant.get("session_id")
                or user.get("source") != assistant.get("source")
                or user.get("domain") != "evaluation" or assistant.get("domain") != "evaluation"):
            continue
        pairs.append((user, assistant))
    return pairs


def short_context_pairs(messages: list[dict]) -> list[dict]:
    """Find short turns with longer exchange context without modifying originals."""
    pairs = []
    for user, assistant in evaluation_exchange_pairs(messages):
        for short, detail in ((user, assistant), (assistant, user)):
            short_tokens = short_words(short["text"])
            detail_text = detail["text"]
            if (1 <= len(short_tokens) <= 2 and len(short_words(detail_text)) >= 8
                    and len(context_terms(detail_text)) >= 3
                    and "no stored assistant reply matched" not in detail_text.lower()):
                pairs.append({"id": short["id"], "detail_id": detail["id"], "role": short["role"],
                              "short_tokens": short_tokens, "text": detail_text})
    return pairs


def contextual_associations(pairs: list[dict], existing_links: list[dict]) -> list[dict]:
    """Propose new provisional edges only when no original cross-link exists."""
    by_id = {pair["id"]: pair for pair in pairs}
    existing = {frozenset((link["source"], link["target"])) for link in existing_links}
    result = []
    for link in infer_similarity_links(pairs, max_neighbors=max(1, len(pairs)), min_similarity=0.30):
        first, second = by_id[link["source"]], by_id[link["target"]]
        if (first["role"] != second["role"] or first["short_tokens"] != second["short_tokens"]
                or (first["role"] == "You" and first["short_tokens"] == ["next"])
                or len(context_terms(first["text"]) & context_terms(second["text"])) < 3
                or (set(short_words(first["text"])) & CONTEXT_NEGATION)
                != (set(short_words(second["text"])) & CONTEXT_NEGATION)):
            continue
        cross = [frozenset((a, b)) for a in (first["id"], first["detail_id"])
                 for b in (second["id"], second["detail_id"])]
        if any(pair in existing for pair in cross):
            continue
        result.append({"source": first["detail_id"], "target": second["detail_id"],
                       "kind": "contextual", "strength": link["strength"]})
        existing.add(frozenset((first["detail_id"], second["detail_id"])))
    return result


def infer_similarity_links(
    messages: list[dict],
    max_neighbors: int = 2,
    min_similarity: float = 0.025,
) -> list[dict]:
    """Return sparse TF-IDF links between messages with overlapping content."""
    if len(messages) < 2:
        return []

    tokens = [
        [
            token
            for token in re.findall(r"[a-z0-9]+", message["text"].lower())
            if len(token) > 2 and token not in STOP_WORDS
        ]
        for message in messages
    ]
    term_counts = [Counter(document) for document in tokens]
    document_frequency = Counter(
        term for counts in term_counts for term in counts.keys()
    )
    document_count = len(messages)
    idf = {
        term: math.log((document_count + 1) / (frequency + 1)) + 1
        for term, frequency in document_frequency.items()
    }
    vectors = [
        {term: count * idf[term] for term, count in counts.items()}
        for counts in term_counts
    ]
    norms = [
        math.sqrt(sum(weight * weight for weight in vector.values()))
        for vector in vectors
    ]

    candidates: list[tuple[int, int, float]] = []
    for first in range(document_count):
        if not norms[first]:
            continue
        for second in range(first + 1, document_count):
            if not norms[second]:
                continue
            shared = vectors[first].keys() & vectors[second].keys()
            similarity = sum(
                vectors[first][term] * vectors[second][term] for term in shared
            ) / (norms[first] * norms[second])
            if similarity >= min_similarity:
                candidates.append((first, second, similarity))

    neighbors: dict[int, list[tuple[float, int]]] = {
        index: [] for index in range(document_count)
    }
    for first, second, similarity in candidates:
        neighbors[first].append((similarity, second))
        neighbors[second].append((similarity, first))

    selected: set[tuple[int, int]] = set()
    for index, related in neighbors.items():
        related.sort(reverse=True)
        for _, other in related[:max_neighbors]:
            selected.add((min(index, other), max(index, other)))

    scores = {(first, second): score for first, second, score in candidates}
    return [
        {
            "source": messages[first]["id"],
            "target": messages[second]["id"],
            "strength": round(min(0.65, 0.25 + scores[(first, second)] * 0.8), 4),
            "similarity": round(scores[(first, second)], 4),
            "kind": "similarity",
        }
        for first, second in sorted(selected)
    ]
