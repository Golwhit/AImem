"""A small character-level generator trained only on explicitly reviewed text."""

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path


MODEL_NAME = "original-character-ngram-v1"
ORDER = 4
MAX_DOCUMENT_CHARACTERS = 50_000
MAX_CORPUS_CHARACTERS = 500_000
MAX_GENERATED_CHARACTERS = 500
MAX_EVALUATION_PROMPTS = 30
MAX_EVALUATION_PROMPT_CHARACTERS = 1000
MAX_EVALUATION_CORRECTION_CHARACTERS = 2000
START = "\u0002"
END = "\u0003"


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        temporary = stream.name
    os.replace(temporary, path)


class CharacterGenerator:
    def __init__(self, root: Path | None = None):
        self.root = root or Path(__file__).resolve().parent
        self.path = self.root / ".data" / "original-generator.json"
        self.evaluation_path = self.root / ".data" / "generator-evaluations.json"
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.state = self._empty_state()
        self.evaluations = (
            json.loads(self.evaluation_path.read_text(encoding="utf-8"))
            if self.evaluation_path.exists()
            else {"version": 1, "prompts": []}
        )

    @staticmethod
    def _empty_state(generation: int = 0) -> dict:
        return {
            "version": 1,
            "model": MODEL_NAME,
            "generation": generation,
            "order": ORDER,
            "documents": [],
            "transitions": {},
        }

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.split())

    def _rebuild(self, documents: list[dict], generation: int) -> dict:
        transitions: dict[str, Counter[str]] = {}
        for document in documents:
            text = self._normalize(document["text"])
            sequence = START * ORDER + text + END
            for index in range(len(sequence) - ORDER):
                context = sequence[index:index + ORDER]
                next_character = sequence[index + ORDER]
                transitions.setdefault(context, Counter())[next_character] += 1
        return {
            "version": 1,
            "model": MODEL_NAME,
            "generation": generation,
            "order": ORDER,
            "documents": documents,
            "transitions": {context: dict(counts) for context, counts in transitions.items()},
        }

    def _save(self, state: dict) -> None:
        atomic_json(self.path, state)
        self.state = state

    def _save_evaluations(self) -> None:
        atomic_json(self.evaluation_path, self.evaluations)

    @staticmethod
    def _evaluation_seed(prompt: str, generation: int) -> int:
        digest = hashlib.sha256(f"{generation}\0{prompt}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big")

    def _capture_current_generation(self) -> None:
        if not self.state["documents"]:
            return
        changed = False
        for item in self.evaluations["prompts"]:
            if any(run["generation"] == self.state["generation"] for run in item["runs"]):
                continue
            sample = self.generate(
                item["prompt"],
                max_chars=MAX_GENERATED_CHARACTERS,
                seed=self._evaluation_seed(item["prompt"], self.state["generation"]),
            )
            item["runs"].append({
                "generation": self.state["generation"],
                "text": sample["text"],
                "conditioned_on_prompt": sample.get("conditioned_on_prompt", False),
                "training_sources": sample["training_sources"],
                "rating": None,
                "correction": "",
            })
            changed = True
        if changed:
            self._save_evaluations()

    def track_prompt(self, prompt: str) -> dict:
        if not isinstance(prompt, str):
            raise ValueError("Evaluation prompt must be text")
        prompt = " ".join(prompt.split())
        if not prompt or len(prompt) > MAX_EVALUATION_PROMPT_CHARACTERS:
            raise ValueError("Evaluation prompt must be nonempty and at most 1000 characters")
        prompt_id = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        item = next((entry for entry in self.evaluations["prompts"] if entry["id"] == prompt_id), None)
        if item is None:
            if len(self.evaluations["prompts"]) >= MAX_EVALUATION_PROMPTS:
                raise ValueError(f"At most {MAX_EVALUATION_PROMPTS} fixed prompts can be tracked")
            item = {"id": prompt_id, "prompt": prompt, "runs": []}
            self.evaluations["prompts"].append(item)
            self._save_evaluations()
        self._capture_current_generation()
        return self.evaluation_status()

    def rate_evaluation(self, prompt_id: str, generation: int, rating: str, correction: str = "") -> dict:
        if rating not in {"keep", "edit", "reject"}:
            raise ValueError("Rating must be keep, edit, or reject")
        if not isinstance(correction, str) or len(correction) > MAX_EVALUATION_CORRECTION_CHARACTERS:
            raise ValueError("Correction must be text up to 2000 characters")
        if rating == "edit" and not correction.strip():
            raise ValueError("An edited rating requires corrected text")
        item = next((entry for entry in self.evaluations["prompts"] if entry["id"] == prompt_id), None)
        run = next((entry for entry in item["runs"] if entry["generation"] == generation), None) if item else None
        if run is None:
            raise ValueError("Unknown prompt evaluation")
        run["rating"] = rating
        run["correction"] = correction.strip() if rating == "edit" else ""
        self._save_evaluations()
        return self.evaluation_status()

    def evaluation_status(self) -> dict:
        return {
            "current_generation": self.state["generation"],
            "prompts": self.evaluations["prompts"],
        }

    def add_document(
        self,
        title: str = "",
        source: str = "Me",
        text: str = "",
        acknowledge_review: bool = False,
        rights_note: str = "",
    ) -> dict:
        if acknowledge_review is not True:
            raise ValueError("Explicit review and approval of this text is required")
        if not isinstance(title, str) or not isinstance(source, str):
            raise ValueError("Document title and source must be text")
        if not isinstance(text, str):
            raise ValueError("Document text must be a string")
        title = title.strip() or f"Untitled source {self.state['generation'] + 1}"
        source = source.strip() or "Me"
        normalized = self._normalize(text)
        if len(normalized) < ORDER:
            raise ValueError(f"Document text must contain at least {ORDER} characters")
        if len(normalized) > MAX_DOCUMENT_CHARACTERS:
            raise ValueError(f"Document text must not exceed {MAX_DOCUMENT_CHARACTERS} characters")
        if not isinstance(rights_note, str) or len(rights_note) > 1000:
            raise ValueError("Rights and provenance note must be text up to 1000 characters")
        current_size = sum(document["characters"] for document in self.state["documents"])
        if current_size + len(normalized) > MAX_CORPUS_CHARACTERS:
            raise ValueError(f"Approved generator text must not exceed {MAX_CORPUS_CHARACTERS} characters total")

        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if any(document["sha256"] == digest for document in self.state["documents"]):
            raise ValueError("This exact text is already in the generator corpus")
        document = {
            "id": digest[:16],
            "title": title.strip()[:160],
            "source": source.strip()[:500],
            "rights_note": rights_note.strip(),
            "text": normalized,
            "characters": len(normalized),
            "sha256": digest,
            "rights_reviewed": True,
            "legal_clearance": False,
        }
        documents = [*self.state["documents"], document]
        self._save(self._rebuild(documents, self.state["generation"] + 1))
        self._capture_current_generation()
        return self.status()

    def ingest_chat_prompt(self, text: str) -> dict:
        if not isinstance(text, str):
            raise ValueError("Chat prompt must be text")
        normalized = self._normalize(text)
        if len(normalized) < ORDER:
            return {"added": False, "reason": "too_short", "generation": self.state["generation"]}
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        existing = next(
            (document for document in self.state["documents"] if document["sha256"] == digest),
            None,
        )
        if existing:
            return {
                "added": False, "reason": "already_present",
                "document_id": existing["id"], "generation": self.state["generation"],
            }
        status = self.add_document(
            title="Memory chat prompt",
            source="Memory chat",
            text=normalized,
            acknowledge_review=True,
            rights_note="User opted in to local generator training for this memory-chat prompt.",
        )
        return {
            "added": True, "document_id": digest[:16],
            "generation": status["generation"],
        }

    def remove_document(self, document_id: str) -> dict:
        documents = [document for document in self.state["documents"] if document["id"] != document_id]
        if len(documents) == len(self.state["documents"]):
            raise ValueError("Unknown generator document ID")
        self._save(self._rebuild(documents, self.state["generation"] + 1))
        self._capture_current_generation()
        return self.status()

    def status(self) -> dict:
        documents = [
            {key: document[key] for key in (
                "id", "title", "source", "rights_note", "characters", "sha256",
                "rights_reviewed", "legal_clearance",
            )}
            for document in self.state["documents"]
        ]
        character_count = sum(document["characters"] for document in self.state["documents"])
        evaluation_runs = sum(len(item["runs"]) for item in self.evaluations["prompts"])
        return {
            "model": MODEL_NAME,
            "generation": self.state["generation"],
            "order": ORDER,
            "document_count": len(documents),
            "training_characters": character_count,
            "evaluation_prompt_count": len(self.evaluations["prompts"]),
            "evaluation_sample_count": evaluation_runs,
            "ready": bool(documents),
            "documents": documents,
            "trained_from_scratch": True,
            "pretrained_weights": False,
            "external_calls": False,
        }

    def generate(self, prompt: str, max_chars: int = 320, seed: int | None = None) -> dict:
        if not isinstance(prompt, str) or len(prompt) > 2000:
            raise ValueError("Prompt must be text up to 2000 characters")
        if not isinstance(max_chars, int) or isinstance(max_chars, bool):
            raise ValueError("max_chars must be an integer")
        max_chars = max(1, min(MAX_GENERATED_CHARACTERS, max_chars))
        transitions = self.state["transitions"]
        if not self.state["documents"] or not transitions:
            return {
                "model": MODEL_NAME,
                "ready": False,
                "text": "",
                "generation": self.state["generation"],
                "training_sources": [],
            }

        context = START * ORDER
        conditioned = False
        for end in range(len(prompt), ORDER - 1, -1):
            candidate = prompt[end - ORDER:end]
            if candidate in transitions:
                context = candidate
                conditioned = True
                break

        generator = random.Random(seed)
        output = []
        for _ in range(max_chars):
            counts = transitions.get(context)
            if not counts:
                break
            choices = list(counts)
            next_character = generator.choices(choices, weights=[counts[char] for char in choices], k=1)[0]
            if next_character == END:
                break
            output.append(next_character)
            context = (context + next_character)[-ORDER:]

        return {
            "model": MODEL_NAME,
            "ready": True,
            "text": "".join(output),
            "generation": self.state["generation"],
            "conditioned_on_prompt": conditioned,
            "training_sources": [
                {key: document[key] for key in ("id", "title", "source", "sha256")}
                for document in self.state["documents"]
            ],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["status", "add", "remove", "generate", "generate-selected", "scorer-feedback", "scorer-status", "ingest-chat-prompt", "evaluations", "track-prompt", "rate-evaluation"])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    payload = json.load(sys.stdin)
    generator = CharacterGenerator(args.root)
    if args.command in {"generate-selected", "scorer-feedback", "scorer-status"}:
        from original_response_scorer import ResponseScorer

        scorer = ResponseScorer(args.root)
        if args.command == "generate-selected":
            prompt = payload.get("prompt", "")
            result = scorer.select(prompt, [generator.generate(prompt, payload.get("max_chars", 320)) for _ in range(4)])
        elif args.command == "scorer-feedback":
            if payload.get("acknowledge_review") is not True:
                raise ValueError("Explicit prompt-response review is required")
            transcript = json.loads((args.root / "chat_memory.json").read_text(encoding="utf-8"))
            messages = transcript["messages"]
            index = next((position for position, message in enumerate(messages) if message["id"] == payload.get("response_id")), -1)
            if index < 1:
                raise ValueError("Unknown response")
            response, prompt = messages[index], messages[index - 1]
            if response["role"] != "Assistant" or response.get("source") != "local-generated" or prompt["role"] != "You" or prompt.get("created_cycle") != response.get("created_cycle"):
                raise ValueError("Only a saved local-generated prompt-response pair can train the scorer")
            correction = payload.get("correction", "")
            approve_correction = payload.get("approve_correction") is True
            if approve_correction and (payload.get("rating") != "needs_correction" or not isinstance(correction, str) or len(correction.strip()) < ORDER or len(correction) > 2000 or correction.strip() == response["text"].strip()):
                raise ValueError("An approved correction must be edited text of 4 to 2000 characters")
            result = scorer.feedback(prompt["text"], response["text"], payload.get("rating"), f"user-rated:{prompt['id']}:{response['id']}", True)
            if approve_correction:
                correction = correction.strip()
                digest = hashlib.sha256(generator._normalize(correction).encode("utf-8")).hexdigest()
                added = not any(document["sha256"] == digest for document in generator.state["documents"])
                if added:
                    generator.add_document(
                        title=f"Correction for {response['id']}", source=f"Me:correction:{response['id']}",
                        text=correction, acknowledge_review=True,
                        rights_note=f"User explicitly approved edited text for prompt {prompt['id']} and response {response['id']}.",
                    )
                result = scorer.feedback(prompt["text"], correction, "useful", f"user-correction:{prompt['id']}:{response['id']}", True)
                result["correction_training"] = {"trained": True, "document_added": added, "generator_generation": generator.state["generation"]}
        else:
            result = scorer.status()
    elif args.command == "add":
        result = generator.add_document(**payload)
    elif args.command == "remove":
        result = generator.remove_document(payload.get("document_id", ""))
    elif args.command == "generate":
        result = generator.generate(payload.get("prompt", ""), payload.get("max_chars", 320), payload.get("seed"))
    elif args.command == "ingest-chat-prompt":
        result = generator.ingest_chat_prompt(payload.get("text", ""))
    elif args.command == "track-prompt":
        result = generator.track_prompt(payload.get("prompt", ""))
    elif args.command == "rate-evaluation":
        result = generator.rate_evaluation(
            payload.get("prompt_id", ""), payload.get("generation", -1),
            payload.get("rating", ""), payload.get("correction", ""),
        )
    elif args.command == "evaluations":
        result = generator.evaluation_status()
    else:
        result = generator.status()
        from original_response_scorer import ResponseScorer

        result["scorer"] = ResponseScorer(args.root).status()
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)