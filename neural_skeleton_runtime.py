"""Reusable skeleton bank and lossless mixed word/character neural tokens.

The frozen experiment is imported, never edited or rerun on locked books.
Runtime recognition confirms the entire word; skeletons are not prefix guesses.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import neural_skeletons as experiment
from neural_experiment import NeuralModel
from original_generator import atomic_json


BANK_PATH = ".data/neural-skeletons/runtime-bank.json"
END = "\x03"
MAX_UNITS = 128


def word_character(character: str) -> bool:
    return bool(character) and (character.isalnum() or character in "_’'")


def word_boundaries(text: str, start: int, end: int) -> bool:
    return not word_character(text[start - 1:start] if start else "") and not word_character(text[end:end + 1])


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def build_bank(root: Path) -> dict:
    directory = root / experiment.SKELETON_DIRECTORY
    registration = json.loads((directory / "preregistration.json").read_text())
    code_hash = hashlib.sha256(Path(experiment.__file__).read_bytes()).hexdigest()
    if code_hash != registration["code_sha256"]:
        raise ValueError("Frozen skeleton code no longer matches its pre-registration")
    learning, provenance = experiment.load(root, ("learn", "develop"))
    possible, dictionary_hash = experiment.dictionary(root)
    states = experiment.StrictSkeletons(learning, possible, support=experiment.SUPPORT)
    patterns = {kind: states.strict[kind] for kind in ("hidden1", "hidden2")}
    record = {
        "version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "Approved public-domain WEB learning/development books; SCOWL possible-word list",
        "learning_groups": ["learn", "develop"], "learning": provenance,
        "learning_sha256": hashlib.sha256(learning.encode()).hexdigest(),
        "dictionary_sha256": dictionary_hash, "experiment_code_sha256": code_hash,
        "support": experiment.SUPPORT, "patterns": patterns,
        "policy": "No locked text. No dropped-letter claims. Confirm complete lowercase words before unit encoding.",
    }
    record["sha256"] = digest(record)
    atomic_json(root / BANK_PATH, record)
    return bank_summary(record)


def bank_summary(record: dict) -> dict:
    return {key: record[key] for key in ("version", "sha256", "source", "learning_sha256",
                                        "dictionary_sha256", "experiment_code_sha256", "policy")} | {
        "patterns": {kind: len(table) for kind, table in record["patterns"].items()},
        "words": len({word for table in record["patterns"].values() for word in table.values()}),
    }


def load_bank(root: Path) -> dict:
    path = root / BANK_PATH
    if not path.exists():
        raise ValueError("Skeleton runtime bank is missing; run python3 neural_skeleton_runtime.py build first")
    record = json.loads(path.read_text())
    checksum = record.pop("sha256")
    if record.get("version") != 1 or digest(record) != checksum:
        raise ValueError("Skeleton runtime bank version or checksum mismatch")
    record["sha256"] = checksum
    return record


class UnitCodec:
    def __init__(self, units: dict[str, dict], provenance: dict):
        for word, evidence in units.items():
            pattern, kind = evidence["pattern"], evidence["kind"]
            if (kind not in ("hidden1", "hidden2") or pattern.startswith("*")
                    or not experiment.claimable(word) or not experiment.WORD.fullmatch(word)
                    or pattern not in experiment.skeletons(word, kind)):
                raise ValueError("Invalid runtime unit evidence")
        self.units = units
        self.provenance = provenance

    @classmethod
    def from_bank(cls, bank: dict, training: str) -> "UnitCodec":
        evidence = {}
        for kind, table in sorted(bank["patterns"].items()):
            for pattern, word in sorted(table.items()):
                evidence.setdefault(word, {"pattern": pattern, "kind": kind})
        counts = Counter(match.group() for match in experiment.WORD.finditer(training)
                         if word_boundaries(training, match.start(), match.end()))
        chosen = sorted(counts.keys() & evidence.keys(), key=lambda w: (-counts[w] * (len(w) - 1), w))[:MAX_UNITS]
        return cls({word: evidence[word] for word in chosen}, bank_summary(bank))

    def encode(self, text: str, *, open_tail: bool = False) -> list[str]:
        if END in text:
            raise ValueError("Text contains the reserved neural end marker")
        pieces = []
        cursor = 0
        for match in experiment.WORD.finditer(text):
            pieces.extend(text[cursor:match.start()])
            word = match.group()
            if (word in self.units and word_boundaries(text, match.start(), match.end())
                    and not (open_tail and match.end() == len(text))):
                pieces.append(word)
            else:
                pieces.extend(word)
            cursor = match.end()
        pieces.extend(text[cursor:])
        if "".join(pieces) != text:
            raise AssertionError("Unit encoding changed the original text")
        return pieces

    def examples(self, model: NeuralModel, text: str) -> list[tuple[list[int], int]]:
        sequence = self.encode(text) + [END]
        tokens = [0] * model.context + [model.tokens.get(piece, 1) for piece in sequence]
        return [(tokens[i - model.context:i], tokens[i]) for i in range(model.context, len(tokens))]

    def reading(self, text: str) -> dict:
        pieces = self.encode(text)
        used = Counter(piece for piece in pieces if piece in self.units)
        return {
            "characters": len(text), "decisions": len(pieces), "unit_occurrences": sum(used.values()),
            "unit_characters": sum(len(word) * count for word, count in used.items()),
            "decisions_saved": len(text) - len(pieces),
            "decisions_saved_fraction": 1 - len(pieces) / len(text) if text else 0.0,
            "confirmation": "Whole-word exact match; no guessed letters replace input.",
        }

    def snapshot(self) -> dict:
        return {"units": self.units, "provenance": self.provenance}

    def sample(self, model: NeuralModel, prompt: str, *, length: int = 120, seed: int = 23) -> dict:
        if type(length) is not int or not 1 <= length <= 500:
            raise ValueError("Generation length must be an integer from 1 to 500 characters")
        rng = random.Random(seed)
        context = ([0] * model.context + [model.tokens.get(piece, 1)
                   for piece in self.encode(prompt, open_tail=True)])[-model.context:]
        output = ""
        emitted = []
        decisions = character_steps = 0
        after_unit = False
        ended = False
        while len(output) < length:
            probabilities = model.forward(context)[2]
            tail = (prompt + output)[-1:]
            in_word = word_character(tail)
            remaining = length - len(output)
            for piece, token in model.tokens.items():
                is_unit = piece in self.units
                if (piece != END and (len(piece) > remaining
                        or (is_unit and (in_word or after_unit))
                        or (after_unit and (is_unit or word_character(piece))))):
                    probabilities[token] = 0.0
            if not any(probabilities[2:]):
                raise ArithmeticError("Unit generation has no allowed output")
            token = rng.choices(range(2, model.size), weights=probabilities[2:])[0]
            piece = model.characters[token - 2]
            decisions += 1
            if piece == END:
                ended = True
                break
            output += piece
            after_unit = piece in self.units
            if after_unit:
                emitted.append({"word": piece, "offset": len(output) - len(piece)})
            else:
                character_steps += 1
            context = context[1:] + [token]
        return {"text": output, "decisions": decisions, "character_steps": character_steps,
                "unit_occurrences": len(emitted), "unit_characters": sum(len(item["word"]) for item in emitted),
                "emitted_units": emitted, "ended": ended, "characters": len(output),
                "note": "The neural model chooses units or characters; strict units only guarantee literal spelling, not a correct response."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "status"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    print(json.dumps(build_bank(root) if args.command == "build" else bank_summary(load_bank(root)), indent=2))
