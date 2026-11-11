"""Strict recognition/spelling layer plus a light pattern-weighting layer.

Strict layer ("if it sees this, it is this"): a unit is a word, or a phrase of 2-4 words joined by
single spaces seen at least PHRASE_SUPPORT times, learned from approved text. In reading, a unit is
claimed only after the text in front of the reader is confirmed to be that whole unit, ending at a
word boundary, so it is skipped with one decision instead of per-character prediction. In generation,
a unit is chosen first by soft word counts, then spelled out whole. The strict layer never guesses the
rest of a word from its start; it can only say "unknown" and hand characters to other layers.

Light layer: the earned 1-9 character patterns from neural_patterns.py guess a continuation from a
partial start. They were 94-95% right, so here they only add weight: (support+1)/(support+2) on the
guessed character, mixed with Kneser-Ney counts. Never certain, never forbidding.

Development only: learn Acts 1, 2, 4-26 and measure on Acts 27. Acts 28 is spent; Acts 3 never read.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time
from uuid import uuid4

from neural_counting import KneserNey
from neural_experiment import NeuralModel
from neural_layers import evaluate
from neural_patterns import DEVELOPMENT_CHAPTER, MAX_SPAN, STAGED_DIRECTORY, FixedPatterns, learning_text, load_chapters
from original_generator import atomic_json


WORD = re.compile(r"[A-Za-z’']+")
TOKEN = re.compile(r"[A-Za-z’']+|[^A-Za-z’']")
PHRASE_SUPPORT = 5
MAX_PHRASE_WORDS = 4
LIGHT_SUPPORT = 10
FIXED_WEIGHTS = (0.5, 0.9)
DISCOUNT = 0.75
GENERATION_PROMPTS = ("The", "Paul", "And", "He", "When")
GENERATION_UNITS = 120
APPROVAL = "User: 'Yes - strict recognition/spelling layer + move guessing patterns to a light layer'; no commit."


def is_letter(char: str) -> bool:
    return bool(WORD.fullmatch(char))


class StrictUnits:
    def __init__(self, text: str, *, phrase_support: int = PHRASE_SUPPORT, max_phrase_words: int = MAX_PHRASE_WORDS):
        if not text or phrase_support < 2 or not 1 <= max_phrase_words <= 8:
            raise ValueError("Strict units need text, phrase support >= 2 and 1-8 phrase words")
        found = list(WORD.finditer(text))
        self.words = Counter(match.group() for match in found)
        phrases = Counter()
        for start in range(len(found)):
            for size in range(2, max_phrase_words + 1):
                end = start + size - 1
                if end >= len(found):
                    break
                if any(text[found[k].end():found[k + 1].start()] != " " for k in range(start, end)):
                    break
                phrases[text[found[start].start():found[end].end()]] += 1
        self.phrases = {phrase: count for phrase, count in phrases.items() if count >= phrase_support}
        self.by_first: dict[str, list[str]] = {}
        for unit in [*self.words, *self.phrases]:
            self.by_first.setdefault(WORD.match(unit).group(), []).append(unit)
        for units in self.by_first.values():
            units.sort(key=len, reverse=True)
        self.operations = Counter()

    def recognize(self, text: str, position: int) -> str | None:
        """The longest known unit that the text at a word start is, in full, ending at a word boundary."""
        if position > 0 and is_letter(text[position - 1]):
            return None
        first = WORD.match(text, position)
        if not first:
            return None
        self.operations["lookup_probes"] += 1
        for unit in self.by_first.get(first.group(), ()):
            self.operations["confirm_comparisons"] += 1
            end = position + len(unit)
            if text.startswith(unit, position) and (end == len(text) or not is_letter(text[end])):
                return unit
        return None

    def read(self, text: str) -> dict:
        i = decisions = recognized = phrase_characters = 0
        handed: list[str] = []
        used = Counter()
        while i < len(text):
            unit = self.recognize(text, i)
            decisions += 1
            if unit:
                if text[i:i + len(unit)] != unit:
                    raise AssertionError("Strict layer claimed text that does not match")
                recognized += len(unit)
                phrase_characters += len(unit) if unit in self.phrases else 0
                used[unit] += 1
                i += len(unit)
                continue
            if is_letter(text[i]):
                end = WORD.match(text, i).end()
                handed.append(text[i:end])
                decisions += end - i - 1
                i = end
            else:
                i += 1
        letters = sum(1 for char in text if is_letter(char))
        return {
            "characters": len(text), "letters": letters, "strict_claims": sum(used.values()),
            "strict_wrong_claims": 0, "recognized_characters": recognized,
            "phrase_characters": phrase_characters,
            "coverage": recognized / len(text) if text else 0.0,
            "letter_coverage": recognized / letters if letters else 0.0,
            "decisions": decisions, "decisions_saved_fraction": 1 - decisions / len(text) if text else 0.0,
            "unknown_words": len(handed), "unknown_word_examples": sorted(set(handed))[:40],
            "top_phrases": [{"phrase": u, "count": c} for u, c in used.most_common() if u in self.phrases][:12],
            "note": "Claims are made only after confirming the whole unit in the text, so 0 wrong claims is by construction; the measured result is work saved.",
        }

    def summary(self) -> dict:
        strongest = sorted(self.phrases.items(), key=lambda kv: -kv[1] * len(kv[0]))[:20]
        return {"words": len(self.words), "phrases": len(self.phrases), "phrase_support": PHRASE_SUPPORT,
                "strongest_phrases": [{"phrase": p, "support": c} for p, c in strongest]}


def kind(token: str) -> str:
    return "word" if WORD.fullmatch(token) else token


class UnitChooser:
    """Soft layer for generation: absolute-discount bigram counts over tokens (words, spaces, punctuation).

    Layer-1-style exclusion: a token kind may not follow a kind it never followed in training
    (e.g. a word directly after a word, or two spaces), so spelled words keep their boundaries.
    """

    def __init__(self, text: str, *, discount: float = DISCOUNT):
        tokens = TOKEN.findall(text)
        self.unigram = Counter(tokens)
        self.bigram: dict[str, Counter] = {}
        for before, after in zip(tokens, tokens[1:]):
            self.bigram.setdefault(before, Counter())[after] += 1
        self.discount = discount
        self.vocabulary = sorted(self.unigram)
        self.allowed = {(kind(before), kind(after)) for before, after in zip(tokens, tokens[1:])}

    def choose(self, previous: str, rng: random.Random) -> str:
        total = sum(self.unigram.values())
        following = self.bigram.get(previous, Counter())
        seen = sum(following.values())
        backoff = self.discount * len(following) / seen if seen else 1.0
        weights = [(max(following[t] - self.discount, 0) / seen if seen else 0.0) + backoff * self.unigram[t] / total
                   if (kind(previous), kind(t)) in self.allowed else 0.0 for t in self.vocabulary]
        return rng.choices(self.vocabulary, weights=weights)[0]


def generate(units: StrictUnits, chooser: UnitChooser, prompt: str, seed: int) -> dict:
    rng = random.Random(seed)
    tokens = TOKEN.findall(prompt)
    for _ in range(GENERATION_UNITS):
        tokens.append(chooser.choose(tokens[-1], rng))
    text = "".join(tokens)
    produced = tokens[len(TOKEN.findall(prompt)):]
    words = [t for t in produced if WORD.fullmatch(t)]
    if any(word not in units.words for word in words):
        raise AssertionError("Strict spelling produced a word that was never learned")
    if any(WORD.fullmatch(a) and WORD.fullmatch(b) for a, b in zip(tokens, tokens[1:])):
        raise AssertionError("Two words were joined without a boundary")
    characters = sum(len(t) for t in produced)
    return {"prompt": prompt, "seed": seed, "text": text, "characters": characters, "decisions": len(produced),
            "spelled_letters": sum(len(w) for w in words), "words": len(words), "misspelled_words": 0,
            "decisions_saved_fraction": 1 - len(produced) / characters if characters else 0.0,
            "label": "Generated sample from word counts with strict spelling; not scripture and not the World English Bible."}


def light_rows(patterns: FixedPatterns, counts: KneserNey, model: NeuralModel, text: str,
               weight: float | None) -> tuple[list[list[float]], Counter]:
    rows, fired = [], Counter()
    for i in range(len(text)):
        row = counts.prior(text[max(0, i - counts.order):i], model)
        guess = None
        for start in range(i, max(0, i - MAX_SPAN) - 1, -1):
            found = patterns.match(text[max(0, start - patterns.max_trigger):start])
            if found and len(found[1]) > i - start and text[start:i] == found[1][:i - start]:
                guess = (found[1][i - start], patterns.patterns[found[0]]["support"])
                break
        if guess and guess[0] in model.tokens:
            w = weight if weight is not None else (guess[1] + 1) / (guess[1] + 2)
            token = model.tokens[guess[0]]
            row = [(1 - w) * value for value in row]
            row[token] += w
            fired["right" if text[i] == guess[0] else "wrong"] += 1
        rows.append(row)
    return rows, fired


def run(root: Path, *, acknowledge_review: bool = False) -> dict:
    learn, chapters = learning_text(root, acknowledge_review, DEVELOPMENT_CHAPTER - 1)
    held = load_chapters(root, STAGED_DIRECTORY)[0][DEVELOPMENT_CHAPTER]
    started = time.perf_counter()
    units = StrictUnits(learn)
    chooser = UnitChooser(learn)
    model = NeuralModel(sorted(set(learn)))
    counts = KneserNey(learn)
    patterns = FixedPatterns(learn, support=LIGHT_SUPPORT)
    examples = model.examples(held)
    base = evaluate(model, examples, [counts.prior(held[max(0, i - 3):i], model) for i in range(len(held))], neural=False)
    light = [{"weight": "counts only", "validation_loss": base["loss"], "fired": None}]
    for weight in (None, *FIXED_WEIGHTS):
        rows, fired = light_rows(patterns, counts, model, held, weight)
        metrics = evaluate(model, examples, rows, neural=False)
        light.append({"weight": "rule of succession (support+1)/(support+2)" if weight is None else weight,
                      "validation_loss": metrics["loss"], "fired": dict(fired)})
    run_id = uuid4().hex
    report = {
        "version": 1, "kind": "strict-and-light-layers", "run_id": run_id, "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(), "approval": APPROVAL,
        "learning_chapters": chapters, "learning_characters": len(learn),
        "learning_sha256": hashlib.sha256(learn.encode()).hexdigest(),
        "held_chapter": DEVELOPMENT_CHAPTER, "held_characters": len(held),
        "held_sha256": hashlib.sha256(held.encode()).hexdigest(),
        "strict": {"discovery": units.summary(), "reading": units.read(held),
                   "generation": [generate(units, chooser, p, 17 + i) for i, p in enumerate(GENERATION_PROMPTS)],
                   "operations": dict(units.operations)},
        "light": {"support": LIGHT_SUPPORT, "patterns": len(patterns.patterns), "results": light,
                  "note": "Loss is natural-log cross-entropy per character on Acts 27; lower is better. Fixed weights are a sensitivity check, not tuned choices."},
        "policy": "Development measurement on Acts 27 (already used to develop fixed patterns); no fresh claim. Acts 28 spent, Acts 3 never read. Strict layer: whole confirmed units only. Light layer: weights only, never zero.",
    }
    report["wall_seconds"] = time.perf_counter() - started
    atomic_json(root / ".data/neural-patterns/strict" / run_id / "report.json", report)
    atomic_json(root / ".data/neural-experiments/latest-strict.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acknowledge-review", action="store_true")
    args = parser.parse_args()
    output = run(Path(__file__).resolve().parent, acknowledge_review=args.acknowledge_review)
    reading = {k: v for k, v in output["strict"]["reading"].items() if k not in ("unknown_word_examples", "top_phrases")}
    print(json.dumps({"run_id": output["run_id"], "strict_reading": reading, "light": output["light"]["results"]}, indent=1))
