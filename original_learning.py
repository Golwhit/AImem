"""Small relevance calculations implemented with Python's standard library."""

import math
import re
import json
from collections import Counter


FEATURE_NAMES = ["word_overlap", "phrase_overlap", "query_coverage", "topic_coverage"]
INITIAL_WEIGHTS = [3.0, 0.5, 1.0, 0.5]
LABEL_TARGETS = {"direct_answer": 1.0, "related_context": 0.5, "unrelated": 0.0}


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def dot(first: list[float], second: list[float]) -> float:
    if len(first) != len(second):
        raise ValueError("Vector dimensions must match")
    return sum(left * right for left, right in zip(first, second))


def similarity(first: Counter, second: Counter) -> float:
    denominator = math.sqrt(sum(value * value for value in first.values()) * sum(value * value for value in second.values()))
    if not denominator:
        return 0.0
    return sum(value * second.get(term, 0) for term, value in first.items()) / denominator


def features(query: str, text: str, topics: list[str]) -> list[float]:
    query_tokens = tokens(query)
    text_tokens = tokens(text)
    query_terms = set(query_tokens)
    text_terms = set(text_tokens)
    topic_terms = set(tokens(" ".join(topics)))
    denominator = max(1, len(query_terms))
    query_pairs = Counter(zip(query_tokens, query_tokens[1:]))
    text_pairs = Counter(zip(text_tokens, text_tokens[1:]))
    return [
        similarity(Counter(query_tokens), Counter(text_tokens)),
        similarity(query_pairs, text_pairs),
        len(query_terms & text_terms) / denominator,
        len(query_terms & topic_terms) / denominator,
    ]


def probability(row: list[float], weights: list[float], intercept: float) -> float:
    score = dot(row, weights) + intercept
    if score >= 0:
        return 1.0 / (1.0 + math.exp(-score))
    exponential = math.exp(score)
    return exponential / (1.0 + exponential)


def loss(rows: list[list[float]], labels: list[float], weights: list[float], intercept: float) -> float:
    if not rows or len(rows) != len(labels) or any(label not in LABEL_TARGETS.values() for label in labels):
        raise ValueError("Loss requires matching nonempty examples and labels")
    total = 0.0
    for row, label in zip(rows, labels):
        predicted = max(1e-12, min(1 - 1e-12, probability(row, weights, intercept)))
        total -= label * math.log(predicted) + (1 - label) * math.log(1 - predicted)
    return total / len(rows)


def fit(rows: list[list[float]], labels: list[float], epochs: int = 600, rate: float = 0.3) -> tuple[list[float], float]:
    if not rows or len(rows) != len(labels) or any(label not in LABEL_TARGETS.values() for label in labels):
        raise ValueError("Training requires matching examples and targets 0, 0.5, or 1")
    weights = [0.0] * len(FEATURE_NAMES)
    intercept = 0.0
    for _ in range(epochs):
        gradients = [0.0] * len(weights)
        intercept_gradient = 0.0
        for row, label in zip(rows, labels):
            error = probability(row, weights, intercept) - label
            for index, value in enumerate(row):
                gradients[index] += error * value
            intercept_gradient += error
        weights = [weight - rate * (gradient / len(rows) + 0.01 * weight) for weight, gradient in zip(weights, gradients)]
        intercept -= rate * intercept_gradient / len(rows)
    return weights, intercept


def demo() -> dict:
    relevant = features("apple pie", "apple pie recipe", ["food"])
    context = features("apple pie", "apple orchard growing trees", ["fruit"])
    irrelevant = features("apple pie", "saving money budget", ["finance"])
    rows = [relevant, context, irrelevant] * 4
    labels = [1.0, 0.5, 0.0] * 4
    weights, intercept = fit(rows, labels)
    return {
        "before_loss": loss(rows, labels, [0.0] * 4, 0.0),
        "after_loss": loss(rows, labels, weights, intercept),
        "relevant_score": probability(relevant, weights, intercept),
        "context_score": probability(context, weights, intercept),
        "irrelevant_score": probability(irrelevant, weights, intercept),
        "label_targets": LABEL_TARGETS,
        "weights": weights,
        "persistent_changes": False,
    }


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2))