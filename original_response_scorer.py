"""Reviewed response selection experiments, separate from language generation."""

import hashlib
import json
from pathlib import Path

from original_generator import atomic_json
from original_learning import features, fit, loss, probability, tokens


FEATURES = ["word_overlap", "phrase_overlap", "query_coverage", "word_diversity"]
TARGETS = {"useful": 1.0, "needs_correction": 0.0}


def response_features(prompt: str, reply: str) -> list[float]:
    words = tokens(reply)
    return features(prompt, reply, [])[:3] + [len(set(words)) / max(1, len(words))]


class ResponseScorer:
    def __init__(self, root: Path):
        self.path = root / ".data" / "response-scorer.json"
        self.state = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {
            "version": 1, "generation": 0, "weights": [0.0] * 4,
            "intercept": 0.0, "examples": [], "training": None,
        }

    def score(self, prompt: str, reply: str) -> float:
        row = response_features(prompt, reply)
        baseline = probability(row, self.state["weights"], self.state["intercept"])
        evidence = []
        for item in self.state["examples"]:
            prompt_match = features(prompt, item["prompt"], [])[0]
            reply_match = features(reply, item["reply"], [])[0]
            if prompt_match >= 0.6 and reply_match >= 0.6:
                evidence.append((2 * TARGETS[item["rating"]] - 1) * prompt_match * reply_match)
        adjustment = 0.25 * sum(evidence) / max(1, len(evidence))
        return max(0.0, min(1.0, baseline + adjustment - 0.1 * (1 - row[3])))

    def select(self, prompt: str, candidates: list[dict]) -> dict:
        available = [candidate for candidate in candidates if candidate.get("text", "").strip()]
        if not available:
            return {"text": "", "selection": {**self.status(), "candidate_count": 0, "score": None}}
        chosen = max(available, key=lambda candidate: self.score(prompt, candidate["text"]))
        return {**chosen, "selection": {**self.status(), "candidate_count": len(available), "score": self.score(prompt, chosen["text"])}}

    def feedback(self, prompt: str, reply: str, rating: str, source: str, acknowledge_review: bool = False) -> dict:
        if acknowledge_review is not True:
            raise ValueError("Explicit review of this prompt-response pair is required")
        if any(not isinstance(value, str) or not value.strip() for value in (prompt, reply, source)):
            raise ValueError("Prompt, reply, and provenance are required")
        if len(prompt) > 2000 or len(reply) > 2000 or len(source) > 500 or rating not in TARGETS:
            raise ValueError("Invalid response feedback")
        prompt = " ".join(prompt.lower().split())
        digest = hashlib.sha256(json.dumps([prompt, reply], ensure_ascii=False).encode()).hexdigest()
        example = {
            "id": digest, "prompt": prompt, "reply": reply, "rating": rating,
            "source": source, "reviewed": True, "legal_clearance": False,
        }
        self.state["examples"] = [item for item in self.state["examples"] if item["id"] != digest] + [example]
        self.state["training"] = self._train()
        atomic_json(self.path, self.state)
        return self.status()

    def _train(self) -> dict:
        examples = self.state["examples"]
        queries = sorted({item["prompt"] for item in examples}, key=lambda query: hashlib.sha256(query.encode()).hexdigest())
        if len(examples) < 12 or len(queries) < 6:
            return {"status": "collecting", "examples": len(examples), "queries": len(queries)}
        held_out = set(queries[:max(2, len(queries) // 4)])
        training = [item for item in examples if item["prompt"] not in held_out]
        validation = [item for item in examples if item["prompt"] in held_out]
        if any(set(TARGETS) != {item["rating"] for item in group} for group in (training, validation)):
            return {"status": "collecting", "reason": "Both ratings required in training and validation"}
        rows = [response_features(item["prompt"], item["reply"]) for item in training]
        weights, intercept = fit(rows, [TARGETS[item["rating"]] for item in training])
        validation_rows = [response_features(item["prompt"], item["reply"]) for item in validation]
        labels = [TARGETS[item["rating"]] for item in validation]
        baseline = loss(validation_rows, labels, self.state["weights"], self.state["intercept"])
        candidate = loss(validation_rows, labels, weights, intercept)
        accepted = candidate + 0.001 < baseline
        if accepted:
            atomic_json(self.path.with_name("response-scorer.previous.json"), self.state)
            self.state["weights"] = weights
            self.state["intercept"] = intercept
            self.state["generation"] += 1
        return {
            "status": "accepted" if accepted else "rejected", "baseline_loss": baseline,
            "candidate_loss": candidate, "validation_queries": sorted(held_out),
            "train_examples": len(training), "validation_examples": len(validation),
        }

    def status(self) -> dict:
        return {
            "model": "original-response-scorer-v1", "generation": self.state["generation"],
            "examples": len(self.state["examples"]), "training": self.state["training"],
            "queries": len({item["prompt"] for item in self.state["examples"]}),
            "minimum_examples": 12, "minimum_queries": 6,
            "weights": dict(zip(FEATURES, self.state["weights"])),
            "local_feedback_active": bool(self.state["examples"]),
            "pretrained_weights": False,
        }