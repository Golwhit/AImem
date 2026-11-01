"""Original relevance learner with reviewed training data and standard-library arithmetic."""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

from original_learning import FEATURE_NAMES, INITIAL_WEIGHTS, LABEL_TARGETS, features, fit, loss, probability, tokens


ROOT = Path(__file__).resolve().parent
FEATURES = FEATURE_NAMES
MODEL_NAME = "original-memory-learner-v3"


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def memory_hash(message: dict) -> str:
    return text_hash(json.dumps({key: message.get(key) for key in ("role", "text", "topics")}, sort_keys=True))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, allow_nan=False)
        temporary = stream.name
    os.replace(temporary, path)


class MemoryLearner:
    def __init__(self, root: Path = ROOT):
        self.root = root
        self.path = root / ".data" / "learner-graded.json"
        self.review_path = root / ".data" / "learning-review.json"
        transcript = json.loads((root / "chat_memory.json").read_text(encoding="utf-8"))
        self.records = {message["id"]: message for message in transcript["messages"]}
        self.messages = [message for message in transcript["messages"] if not (message["role"] == "Assistant" and message.get("source") == "local-retrieval")]
        self.by_id = {message["id"]: message for message in self.messages}
        self.state = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {
            "version": 3, "generation": 0, "features": FEATURES,
            "weights": INITIAL_WEIGHTS.copy(), "intercept": -2.0, "feedback": [], "training": None,
        }
        legacy_path = self.path.with_name("learner-original.json")
        if not self.path.exists() and legacy_path.exists():
            legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
            self.state["feedback"] = [{**item, "label": None, "needs_relabel": True} for item in legacy.get("feedback", [])]
            self.state["legacy_checkpoint"] = legacy_path.name
            self.state["legacy_generation"] = legacy.get("generation", 0)
        self.review_state = json.loads(self.review_path.read_text(encoding="utf-8")) if self.review_path.exists() else {"memories": {}, "queries": {}}
        approved = [message for message in self.messages if self.is_reviewed(message["id"])]
        self.fingerprint = text_hash(json.dumps([memory_hash(message) for message in approved]))
        counts = Counter(token for message in approved for token in tokens(message["text"]))
        self.vocabulary = {term: index for index, (term, _) in enumerate(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:512])}
        self.state["representation"] = {
            "corpus_sha256": self.fingerprint, "approved_documents": len(approved),
            "vocabulary": self.vocabulary, "method": "original-word-and-phrase-overlap",
        }

    def is_reviewed(self, memory_id: str) -> bool:
        message = self.by_id.get(memory_id)
        return bool(message and message["role"] == "You" and self.review_state["memories"].get(memory_id) == memory_hash(message))

    def review(self, memory_ids: list[str], queries: list[str], acknowledge: bool) -> dict:
        if acknowledge is not True:
            raise ValueError("Explicit acknowledgment of training-data review is required")
        if any(memory_id not in self.by_id or self.by_id[memory_id]["role"] != "You" for memory_id in memory_ids):
            raise ValueError("Only existing user-authored memories can be reviewed in this first version")
        normalized = [" ".join(query.lower().split()) for query in queries]
        if any(not query or len(query) > 2000 for query in normalized):
            raise ValueError("Reviewed queries must be nonempty and at most 2000 characters")
        for memory_id in memory_ids:
            self.review_state["memories"][memory_id] = memory_hash(self.by_id[memory_id])
        for query in normalized:
            self.review_state["queries"][text_hash(query)] = "user-reviewed"
        atomic_json(self.review_path, self.review_state)
        return {"ok": True, "reviewed_memories": len(memory_ids), "reviewed_queries": len(normalized), "legal_clearance": False}

    def features(self, query: str, memory_id: str) -> list[float]:
        if memory_id not in self.by_id:
            raise ValueError("Unknown memory ID")
        message = self.by_id[memory_id]
        return features(query, message["text"], message.get("topics", []))

    def probability(self, features: list[float], weights=None, intercept=None) -> float:
        weights = self.state["weights"] if weights is None else weights
        intercept = self.state["intercept"] if intercept is None else intercept
        return probability(features, weights, intercept)

    def search(self, query: str, limit: int = 6, user_only: bool = False, exclude_ids: list[str] | None = None) -> dict:
        if not query.strip():
            raise ValueError("Query must not be empty")
        results = []
        for message in self.messages:
            if message["id"] in (exclude_ids or []):
                continue
            if user_only and message["role"] != "You":
                continue
            features = self.features(query, message["id"])
            if max(features) < 0.05:
                continue
            judgment = next((item for item in self.state["feedback"]
                             if item["query"] == " ".join(query.lower().split()) and item["memory_id"] == message["id"]
                             and item.get("memory_sha256") == memory_hash(message)
                             and item.get("label") in LABEL_TARGETS), None)
            results.append({**message, "score": round(self.probability(features), 4),
                            "judgment": {"label": judgment["label"], "reviewed": self.is_reviewed(message["id"]) and text_hash(judgment["query"]) in self.review_state["queries"]} if judgment else None})
        results.sort(key=lambda result: (-result["score"], result["id"]))
        atomic_json(self.path, self.state)
        return {"query": query, "model": MODEL_NAME, "generation": self.state["generation"], "results": results[:max(1, min(20, limit))]}

    def feedback(self, query: str, memory_id: str, label: str) -> dict:
        if not isinstance(query, str) or not isinstance(memory_id, str) or not isinstance(label, str) or label not in LABEL_TARGETS:
            raise ValueError("Feedback requires a query, memory ID, and label: direct_answer, related_context, or unrelated")
        query = " ".join(query.lower().split())
        if not query or len(query) > 2000:
            raise ValueError("Feedback query must be nonempty and at most 2000 characters")
        self.features(query, memory_id)
        example = {"query": query, "memory_id": memory_id, "label": label, "memory_sha256": memory_hash(self.by_id[memory_id])}
        self.state["feedback"] = [item for item in self.state["feedback"] if (item["query"], item["memory_id"]) != (query, memory_id)] + [example]
        outcome = self.train()
        atomic_json(self.path, self.state)
        return {"ok": True, "label": label, "target": LABEL_TARGETS[label], "training": outcome, "generation": self.state["generation"]}

    def eligible_feedback(self) -> list[dict]:
        return [item for item in self.state["feedback"]
            if isinstance(item.get("label"), str) and item["label"] in LABEL_TARGETS
            and self.is_reviewed(item["memory_id"])
                and item.get("memory_sha256") == memory_hash(self.by_id[item["memory_id"]])
                and text_hash(item["query"]) in self.review_state["queries"]]

    def response_feedback(self, response_id: str, rating: str, correction: str = "") -> dict:
        response = self.records.get(response_id)
        if not response or response["role"] != "Assistant" or response.get("source") != "local-retrieval":
            raise ValueError("Unknown local model response")
        if rating not in ("useful", "needs_correction") or not isinstance(correction, str) or len(correction) > 2000:
            raise ValueError("Response feedback requires a valid rating and correction up to 2000 characters")
        path = self.path.with_name("response-feedback.json")
        saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"version": 1, "items": []}
        saved["items"] = [item for item in saved["items"] if item["response_id"] != response_id] + [{
            "response_id": response_id, "response_sha256": memory_hash(response), "rating": rating,
            "correction": correction.strip(), "source": "user-response-feedback", "training_approved": False,
        }]
        atomic_json(path, saved)
        return {"ok": True, "response_id": response_id, "rating": rating, "used_for_training": False}

    def train(self) -> dict:
        examples = self.eligible_feedback()
        queries = sorted({item["query"] for item in examples}, key=lambda query: hashlib.sha256(query.encode()).hexdigest())
        if len(examples) < 12 or len(queries) < 6:
            return {"status": "collecting", "examples": len(examples), "queries": len(queries), "pending_review": len(self.state["feedback"]) - len(examples), "minimum_examples": 12, "minimum_queries": 6}
        held_out = set(queries[:max(2, len(queries) // 4)])
        train = [item for item in examples if item["query"] not in held_out]
        validation = [item for item in examples if item["query"] in held_out]
        if any(not {"direct_answer", "unrelated"}.issubset({item["label"] for item in partition}) for partition in (train, validation)):
            return {"status": "collecting", "reason": "Need direct-answer and unrelated anchors in both query-separated partitions"}
        signature = hashlib.sha256(json.dumps(examples, sort_keys=True).encode() + self.fingerprint.encode()).hexdigest()
        if (self.state.get("training") or {}).get("signature") == signature:
            return self.state["training"]
        candidate_weights, candidate_intercept = fit([self.features(item["query"], item["memory_id"]) for item in train], [LABEL_TARGETS[item["label"]] for item in train])
        features = [self.features(item["query"], item["memory_id"]) for item in validation]
        labels = [LABEL_TARGETS[item["label"]] for item in validation]
        baseline_loss = loss(features, labels, self.state["weights"], self.state["intercept"])
        candidate_loss = loss(features, labels, candidate_weights, candidate_intercept)
        accepted = candidate_loss + 0.001 < baseline_loss
        outcome = {"status": "accepted" if accepted else "rejected", "signature": signature,
                   "baseline_loss": baseline_loss, "candidate_loss": candidate_loss,
                   "train_examples": len(train), "validation_examples": len(validation),
                   "validation_queries": sorted(held_out)}
        if accepted:
            atomic_json(self.path.with_name("learner-graded.previous.json"), self.state)
            self.state["weights"] = candidate_weights
            self.state["intercept"] = candidate_intercept
            self.state["generation"] += 1
        self.state["training"] = outcome
        return outcome

    def status(self) -> dict:
        eligible = self.eligible_feedback()
        pending = len(self.state["feedback"]) - len(eligible)
        return {"model": MODEL_NAME, "generation": self.state["generation"],
                "ranking_parameters": 5, "features": FEATURES, "feedback_examples": len(self.state["feedback"]),
                "user_documents": sum(message["role"] == "You" for message in self.messages),
                "vocabulary_size": len(self.vocabulary), "approved_memories": sum(self.is_reviewed(message["id"]) for message in self.messages),
                "latent_dimensions": 0, "third_party_ml_libraries": False,
                "weights": dict(zip(FEATURES, self.state["weights"])), "intercept": self.state["intercept"],
                "eligible_examples": len(eligible), "eligible_queries": len({item["query"] for item in eligible}),
                "pending_review": pending, "minimum_examples": 12, "minimum_queries": 6,
                "label_targets": LABEL_TARGETS.copy(),
                "judgment_counts": {label: sum(item.get("label") == label for item in self.state["feedback"]) for label in LABEL_TARGETS},
                "needs_relabel": sum(item.get("label") not in LABEL_TARGETS for item in self.state["feedback"]),
                "learning_state": "trained" if self.state["generation"] else "awaiting-review" if pending else "awaiting-feedback",
                "training": self.state["training"], "pretrained_weights": False, "network_required": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("command", choices=["search", "feedback", "status", "review", "response-feedback"])
    args = parser.parse_args()
    payload = json.load(sys.stdin)
    learner = MemoryLearner(args.root)
    if args.command == "search":
        result = learner.search(payload.get("query", ""), payload.get("limit", 6), payload.get("user_only", False), payload.get("exclude_ids"))
    elif args.command == "feedback":
        result = learner.feedback(payload.get("query", ""), payload.get("memory_id", ""), payload.get("label"))
    elif args.command == "review":
        result = learner.review(payload.get("memory_ids", []), payload.get("queries", []), payload.get("acknowledge", False))
    elif args.command == "response-feedback":
        result = learner.response_feedback(payload.get("response_id", ""), payload.get("rating", ""), payload.get("correction", ""))
    else:
        result = learner.status()
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        raise SystemExit(1)