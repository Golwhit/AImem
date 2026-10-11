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
