"""Character-only recognition shortcuts; no word tokens or new neural outputs."""

from bisect import bisect_left
import random
import json
import re

from original_generator import atomic_json

from neural_skeleton_runtime import load_bank, word_boundaries
from neural_skeletons import WORD, dictionary, load, regular_forms

OPEN_WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*['’]?$")


class CharacterOnlies:
    def __init__(self, words: set[str], prefixes: dict[str, str], provenance: dict):
        self.words = words
        self.prefixes = prefixes
        self.provenance = provenance

    @classmethod
    def build(cls, root):
        bank = load_bank(root)
        dictionary_words, _ = dictionary(root)
        learning, _ = load(root, ("learn", "develop"))
        possible = dictionary_words | {m.group().lower().replace("’", "'") for m in WORD.finditer(learning)}
        possible |= regular_forms(possible)
        ordered = sorted(possible)
        words = {w for table in bank["patterns"].values() for w in table.values()}
        prefixes = {}
        for word in sorted(words):
            for size in range(2, len(word)):
                prefix = word[:size]
                index = bisect_left(ordered, prefix)
                if (index < len(ordered) and ordered[index] == word
                        and (index + 1 == len(ordered) or not ordered[index + 1].startswith(prefix))):
                    prefixes[prefix] = word
                    break
        return cls(words | {"a", "A", "I"}, prefixes, {
            "bank_sha256": bank["sha256"],
            "policy": "Exact bounded recognition; unique prefixes against dictionary, learning words and regular forms. "
                      "Closed-list evidence, not a universal language rule. No locked text.",
        })

    @classmethod
    def cached(cls, root):
        path = root / ".data/neural-skeletons/character-onlies.json"
        bank = load_bank(root)
        if path.exists():
            state = json.loads(path.read_text())
            if state["provenance"]["bank_sha256"] == bank["sha256"]:
                return cls.restore(state)
        rules = cls.build(root)
        atomic_json(path, rules.snapshot())
        return rules

    def snapshot(self):
        return {"words": sorted(self.words), "prefixes": self.prefixes, "provenance": self.provenance}

    @classmethod
    def restore(cls, state):
        return cls(set(state["words"]), state["prefixes"], state["provenance"])

    def offsets(self, text: str) -> set[int]:
        skip = set()
        for match in WORD.finditer(text):
            if match.group() in self.words and word_boundaries(text, match.start(), match.end()):
                skip.update(range(match.start() + 1, match.end()))
        return skip

    def examples(self, model, text):
        skip = self.offsets(text)
        return [example for offset, example in enumerate(model.examples(text)) if offset not in skip]

    def reading(self, text):
        skipped = self.offsets(text)
        return {"characters": len(text), "decisions": len(text) - len(skipped),
                "decisions_saved": len(skipped),
                "decisions_saved_fraction": len(skipped) / len(text) if text else 0.0,
                "confirmation": "Exact bounded words; original character contexts preserved."}

    def sample(self, model, prompt, *, length=120, seed=23, greedy=False):
        if type(length) is not int or not 1 <= length <= 500:
            raise ValueError("Generation length must be an integer from 1 to 500 characters")
        context = ([0] * model.context + [model.tokens.get(c, 1) for c in prompt])[-model.context:]
        rng = random.Random(seed)
        text = prompt
        output = ""
        decisions = forced = matches = 0
        pending = ""
        boundary = False
        while len(output) < length:
            if pending:
                char, pending = pending[0], pending[1:]
                forced += 1
                if not pending:
                    boundary = True
            else:
                # Only a final, boundary-aligned word can identify a continuation.
                match = OPEN_WORD.search(text)
                prefix = match.group() if match and match.end() == len(text) else ""
                word = self.prefixes.get(prefix)
                if (word and match is not None and word_boundaries(text, match.start(), match.end())
                        and all(c in model.tokens for c in word[len(prefix):])
                        and len(word) - len(prefix) <= length - len(output)):
                    pending = word[len(prefix):]
                    matches += 1
                    continue
                probabilities = model.forward(context)[2]
                if boundary:
                    for c, token in model.tokens.items():
                        if c.isalnum() or c in "_’'":
                            probabilities[token] = 0.0
                if not any(probabilities[2:]):
                    raise ArithmeticError("Only-state boundary has no available delimiter")
                token = (max(range(2, model.size), key=lambda k: (probabilities[k], -k)) if greedy else
                         rng.choices(range(2, model.size), weights=probabilities[2:])[0])
                char = model.characters[token - 2]
                decisions += 1
                boundary = False
            text += char
            output += char
            context = context[1:] + [model.tokens.get(char, 1)]
        return {"text": output, "decisions": decisions, "character_steps": decisions,
                "unit_occurrences": matches, "forced_characters": forced, "characters": len(output),
                "note": "Character-only neural choices plus closed-list unique-prefix spelling rules."}
