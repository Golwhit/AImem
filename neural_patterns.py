"""Fixed patterns ("only chains"): once a trigger is seen, the next characters were always the same.

A trigger is any block of 1-9 characters seen at least SUPPORT times in approved learning text
where every occurrence was followed by the exact same continuation. The continuation is the
longest shared run (capped at MAX_SPAN) and may cross letters, spaces, words and punctuation, so
this is not a word dictionary. Reading recognizes a known pattern and skips per-character
processing for it; generation emits the whole continuation as one decision.

Phases keep chapter 28 untouched: stage (Acts 6-27) -> develop (learn 1,2,4-26, check on 27) ->
preregister -> stage-locked (fetch Acts 28) -> evaluate-locked (learn 1,2,4-27, test 28 once).
Acts 3 is never read. Standard library only.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time
from uuid import uuid4

from neural_counting import KneserNey
from neural_experiment import NeuralModel
from neural_proof import Verses, fetch, load_fresh
from neural_rounds import load_acts
from original_generator import atomic_json


SUPPORT = 10
CONTEXT_SUPPORTS = (10, 25, 50, 100)
MAX_TRIGGER = 9
MAX_SPAN = 24
LEARN_CHAPTERS = tuple(range(6, 28))
DEVELOPMENT_CHAPTER = 27
LOCKED_CHAPTER = 28
STAGED_DIRECTORY = ".data/acts-patterns-staged"
LOCKED_DIRECTORY = ".data/acts-patterns-locked"
PATTERN_DIRECTORY = ".data/neural-patterns"
CRITERIA = {
    "P1_coverage": "Patterns correctly supply at least 10% of the characters of the locked chapter.",
    "P2_accuracy": "At least 95% of the characters the patterns claim on the locked chapter are correct.",
}
COVERAGE_TARGET = 0.10
ACCURACY_TARGET = 0.95
APPROVAL = ("User asked to expand only states into fixed patterns for fast reading and generation; chose "
            "learn Acts 6-27, lock Acts 28 only, support 50, success = >=25% coverage at >=95% correct; no push. "
            "After the development check, user chose: 'Use support 10 and reset target to >=10% coverage at >=95% correct'.")
REVISION = ("Revised after development run 4f84e153b69c4d38ac0d85486184f8d5 (support 50: 3.4% coverage, 98.4% correct on "
            "Acts 27) and before Acts 28 was fetched. Original plan: support 50, >=25% coverage at >=95% correct.")
GENERATION_PROMPTS = ("The ", "Paul", "And ", "He s", "of t")
GENERATION_LENGTH = 300


class FixedPatterns:
    def __init__(self, text: str, *, support: int = SUPPORT, max_trigger: int = MAX_TRIGGER, max_span: int = MAX_SPAN):
        if not text or support < 2 or not 1 <= max_trigger <= 16 or max_span < 1:
            raise ValueError("Fixed patterns need text, support >= 2, trigger 1-16 and a positive span cap")
        self.support = support
        self.max_trigger = max_trigger
        self.max_span = max_span
        self.patterns: dict[str, dict] = {}
        self.operations = Counter()
        followers: dict[str, Counter] = {}
        for size in range(1, max_trigger + 1):
            for end in range(size, len(text)):
                followers.setdefault(text[end - size:end], Counter())[text[end]] += 1
                self.operations["learn_count_updates"] += 1
        forced = {t for t, after in followers.items() if len(after) == 1 and sum(after.values()) >= support}
        positions: dict[str, list[int]] = {t: [] for t in forced}
        for size in range(1, max_trigger + 1):
            for end in range(size, len(text)):
                trigger = text[end - size:end]
                if trigger in positions:
                    positions[trigger].append(end)
        for trigger, ends in positions.items():
            span = ""
            while len(span) < max_span:
                nexts = {text[e + len(span)] if e + len(span) < len(text) else None for e in ends}
                self.operations["learn_span_checks"] += len(ends)
                if len(nexts) != 1 or None in nexts:
                    break
                span += nexts.pop()
            self.patterns[trigger] = {"continuation": span, "support": len(ends)}
        # A longer trigger is only kept when it extends the continuation of its own suffix.
        for trigger in sorted(self.patterns, key=len, reverse=True):
            shorter = next((trigger[i:] for i in range(1, len(trigger)) if trigger[i:] in self.patterns), None)
            if shorter and len(self.patterns[shorter]["continuation"]) >= len(self.patterns[trigger]["continuation"]):
                del self.patterns[trigger]

    def match(self, prefix: str) -> tuple[str, str] | None:
        for size in range(min(self.max_trigger, len(prefix)), 0, -1):
            self.operations["lookup_probes"] += 1
            found = self.patterns.get(prefix[-size:])
            if found:
                return prefix[-size:], found["continuation"]
        return None

    def read(self, text: str) -> dict:
        """Teacher-forced reading: a firing pattern claims its continuation; a mismatch costs one normal step."""
        i = claimed = correct = fired = whole = normal = 0
        used = Counter()
        while i < len(text):
            found = self.match(text[max(0, i - self.max_trigger):i])
            if not found:
                normal += 1
                i += 1
                continue
            trigger, span = found
            span = span[:len(text) - i]
            agree = next((k for k, (a, b) in enumerate(zip(span, text[i:i + len(span)])) if a != b), len(span))
            fired += 1
            claimed += len(span)
            correct += agree
            if agree == len(span):
                whole += 1
                used[trigger + "|" + span] += len(span)
                i += len(span)
            else:
                normal += 1
                i += agree + 1
        decisions = normal + fired
        return {
            "characters": len(text), "patterns_fired": fired, "patterns_whole": whole,
            "claimed_characters": claimed, "correct_characters": correct, "wrong_characters": claimed - correct,
            "coverage": correct / len(text) if text else 0.0,
            "character_accuracy": correct / claimed if claimed else None,
            "span_accuracy": whole / fired if fired else None,
            "normal_predictions": normal, "decisions": decisions,
            "decisions_saved_fraction": 1 - decisions / len(text) if text else 0.0,
            "top_patterns": [{"trigger": k.split("|", 1)[0], "continuation": k.split("|", 1)[1], "characters": v}
                             for k, v in used.most_common(25)],
        }

    def summary(self) -> dict:
        spans = Counter(len(p["continuation"]) for p in self.patterns.values())
        triggers = Counter(len(t) for t in self.patterns)
        longest = sorted(self.patterns.items(), key=lambda kv: (-len(kv[1]["continuation"]), -kv[1]["support"]))[:25]
        strongest = sorted(self.patterns.items(), key=lambda kv: -kv[1]["support"] * len(kv[1]["continuation"]))[:25]
        view = lambda items: [{"trigger": t, "continuation": p["continuation"], "support": p["support"]} for t, p in items]
        return {
            "support": self.support, "patterns": len(self.patterns),
            "trigger_lengths": dict(sorted(triggers.items())), "continuation_lengths": dict(sorted(spans.items())),
            "longest": view(longest), "strongest": view(strongest),
        }


def generate(patterns: FixedPatterns, counts: KneserNey, model: NeuralModel, prompt: str, seed: int) -> dict:
    rng = random.Random(seed)
    text, decisions, macro_characters, macros = prompt, 0, 0, []
    while len(text) - len(prompt) < GENERATION_LENGTH:
        found = patterns.match(text[-patterns.max_trigger:])
        decisions += 1
        if found:
            text += found[1]
            macro_characters += len(found[1])
            macros.append(found[0] + "|" + found[1])
            continue
        weights = counts.prior(text[-counts.order:], model)[2:]
        text += rng.choices(model.characters, weights=weights)[0]
    produced = len(text) - len(prompt)
    return {"prompt": prompt, "seed": seed, "text": text, "characters": produced, "decisions": decisions,
            "macro_characters": macro_characters, "macros": len(macros),
            "decisions_saved_fraction": 1 - decisions / produced,
            "label": "Generated sample from counts plus fixed patterns; not scripture and not the World English Bible."}


def chapter_source(raw: bytes, chapter: int) -> tuple[str, int, list[int]]:
    """Like neural_proof.extract, but verses the translation leaves empty (e.g. 8:37) are skipped and recorded."""
    parser = Verses()
    parser.feed(raw.decode("utf-8-sig"))
    count = max(parser.verses)
    if sorted(parser.verses) != list(range(1, count + 1)):
        raise ValueError(f"Unexpected verse inventory for Acts {chapter}")
    verses = {number: " ".join("".join(parts).split()) for number, parts in sorted(parser.verses.items())}
    omitted = [number for number, verse in verses.items() if not verse]
    if len(omitted) > 2:
        raise ValueError(f"Too many empty verses in Acts {chapter}")
    return "\n".join(verse for verse in verses.values() if verse) + "\n", count, omitted


def stage_chapters(root: Path, directory: str, chapters: tuple[int, ...], role: str, fetcher=fetch) -> dict:
    target = root / directory
    if target.exists():
        raise ValueError(f"{directory} already staged; refusing to overwrite")
    old = json.loads((root / ".data/acts-web-staged/manifest.json").read_text())
    rights = fetcher(old["rights_url"])
    files, sources = [], []
    for chapter in chapters:
        url = f"https://ebible.org/engwebp/ACT{chapter:02}.htm"
        raw = fetcher(url)
        text, count, omitted = chapter_source(raw, chapter)
        files.append((chapter, raw, text))
        sources.append({"chapter": chapter, "role": role, "url": url, "verses": count, "omitted_empty_verses": omitted,
                        "raw_sha256": hashlib.sha256(raw).hexdigest(),
                        "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "characters": len(text)})
    target.mkdir(parents=True)
    (target / "rights.html").write_bytes(rights)
    for chapter, raw, text in files:
        (target / f"acts-{chapter:02}.html").write_bytes(raw)
        (target / f"acts-{chapter:02}.txt").write_text(text, encoding="utf-8")
    manifest = {
        "translation": old["translation"], "rights_url": old["rights_url"], "rights_statement": old["rights_statement"],
        "rights_sha256": hashlib.sha256(rights).hexdigest(),
        "rights_matches_original_snapshot": hashlib.sha256(rights).hexdigest() == old["rights_sha256"],
        "retrieved_at": datetime.now(timezone.utc).isoformat(), "preprocessing": old["preprocessing"],
        "status": role, "approval": APPROVAL, "sources": sources,
    }
    atomic_json(target / "manifest.json", manifest)
    return manifest


def load_chapters(root: Path, directory: str) -> tuple[dict[int, str], str]:
    target = root / directory
    manifest_bytes = (target / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if hashlib.sha256((target / "rights.html").read_bytes()).hexdigest() != manifest["rights_sha256"]:
        raise ValueError("Rights record hash mismatch")
    texts = {}
    for source in manifest["sources"]:
        chapter = int(source["chapter"])
        if hashlib.sha256((target / f"acts-{chapter:02}.html").read_bytes()).hexdigest() != source["raw_sha256"]:
            raise ValueError(f"Acts {chapter} raw hash mismatch")
        text = (target / f"acts-{chapter:02}.txt").read_bytes().decode("utf-8")
        if hashlib.sha256(text.encode()).hexdigest() != source["text_sha256"]:
            raise ValueError(f"Acts {chapter} text hash mismatch")
        texts[chapter] = text
    return texts, hashlib.sha256(manifest_bytes).hexdigest()


def learning_text(root: Path, acknowledge_review: bool, through: int) -> tuple[str, list[int]]:
    early = load_acts(root, acknowledge_review, include_final_test=False)
    fresh, _, _ = load_fresh(root)
    staged, _ = load_chapters(root, STAGED_DIRECTORY)
    later = [chapter for chapter in sorted(staged) if chapter <= through]
    return early["training"] + early["validation"] + fresh + "".join(staged[c] for c in later), [1, 2, 4, 5, *later]


def measure(root: Path, learn: str, chapters: list[int], held: str, held_chapter: int, *, contexts: bool) -> dict:
    started = time.perf_counter()
    primary = FixedPatterns(learn)
    model = NeuralModel(sorted(set(learn)))
    counts = KneserNey(learn)
    report = {
        "learning_chapters": chapters, "learning_characters": len(learn),
        "learning_sha256": hashlib.sha256(learn.encode()).hexdigest(),
        "held_chapter": held_chapter, "held_characters": len(held),
        "held_sha256": hashlib.sha256(held.encode()).hexdigest(),
        "discovery": primary.summary(), "reading": primary.read(held),
        "generation": [generate(primary, counts, model, prompt, 17 + i) for i, prompt in enumerate(GENERATION_PROMPTS)],
        "operations": dict(primary.operations),
    }
    if contexts:
        report["other_supports"] = {}
        for support in CONTEXT_SUPPORTS:
            if support != SUPPORT:
                other = FixedPatterns(learn, support=support)
                reading = other.read(held)
                reading.pop("top_patterns")
                report["other_supports"][str(support)] = {"patterns": len(other.patterns), "reading": reading}
    report["wall_seconds"] = time.perf_counter() - started
    return report


def develop(root: Path, *, acknowledge_review: bool = False) -> dict:
    learn, chapters = learning_text(root, acknowledge_review, DEVELOPMENT_CHAPTER - 1)
    staged, manifest_sha = load_chapters(root, STAGED_DIRECTORY)
    run_id = uuid4().hex
    report = {
        "version": 1, "kind": "fixed-patterns-development", "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(), "support": SUPPORT, "max_trigger": MAX_TRIGGER,
        "max_span": MAX_SPAN, "staged_manifest_sha256": manifest_sha, "criteria": CRITERIA, "approval": APPROVAL,
        **measure(root, learn, chapters, staged[DEVELOPMENT_CHAPTER], DEVELOPMENT_CHAPTER, contexts=True),
        "note": "Development check only: Acts 27 was not used for learning here but is used for the final learning set.",
    }
    directory = root / PATTERN_DIRECTORY
    atomic_json(directory / run_id / "development.json", report)
    atomic_json(root / ".data/neural-experiments/latest-patterns.json", public_report(root, development=report))
    return report


def preregister(root: Path, run_id: str) -> dict:
    directory = root / PATTERN_DIRECTORY
    path = directory / "preregistration.json"
    if path.exists():
        raise ValueError("Fixed-pattern criteria already pre-registered")
    development = (directory / run_id / "development.json").read_bytes()
    record = {
        "registered_at": datetime.now(timezone.utc).isoformat(), "development_run_id": run_id,
        "development_sha256": hashlib.sha256(development).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "support": SUPPORT, "max_trigger": MAX_TRIGGER, "max_span": MAX_SPAN,
        "learning_chapters": [1, 2, 4, 5, *LEARN_CHAPTERS], "locked_chapter": LOCKED_CHAPTER,
        "criteria": CRITERIA, "coverage_target": COVERAGE_TARGET, "accuracy_target": ACCURACY_TARGET,
        "revision": REVISION,
        "rule": "Criteria were chosen by the user; the revision is recorded above. Acts 28 is fetched only after this record and evaluated once.",
    }
    atomic_json(path, record)
    return record


def evaluate_locked(root: Path, *, acknowledge_review: bool = False) -> dict:
    directory = root / PATTERN_DIRECTORY
    output = directory / "locked-evaluation.json"
    if output.exists():
        raise ValueError("Locked chapter was already evaluated once; refusing a second evaluation")
    registration = json.loads((directory / "preregistration.json").read_text())
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != registration["code_sha256"]:
        raise ValueError("Pattern code changed after pre-registration")
    development = (directory / registration["development_run_id"] / "development.json").read_bytes()
    if hashlib.sha256(development).hexdigest() != registration["development_sha256"]:
        raise ValueError("Development report changed after pre-registration")
    learn, chapters = learning_text(root, acknowledge_review, LEARN_CHAPTERS[-1])
    locked, manifest_sha = load_chapters(root, LOCKED_DIRECTORY)
    atomic_json(output, {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()})
    report = measure(root, learn, chapters, locked[LOCKED_CHAPTER], LOCKED_CHAPTER, contexts=False)
    reading = report["reading"]
    record = {
        "status": "completed", "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "locked_manifest_sha256": manifest_sha, **report,
        "verdicts": {
            "P1_coverage": {"value": reading["coverage"], "target": COVERAGE_TARGET,
                            "met": reading["coverage"] >= COVERAGE_TARGET},
            "P2_accuracy": {"value": reading["character_accuracy"], "target": ACCURACY_TARGET,
                            "met": (reading["character_accuracy"] or 0) >= ACCURACY_TARGET},
        },
        "note": "Single evaluation on untouched Acts 28. Related chapters are one book, so this is not general-language evidence.",
    }
    atomic_json(output, record)
    atomic_json(root / ".data/neural-experiments/latest-patterns.json", public_report(root))
    return record


def public_report(root: Path, *, development: dict | None = None) -> dict:
    directory = root / PATTERN_DIRECTORY
    registration = directory / "preregistration.json"
    locked = directory / "locked-evaluation.json"
    if development is None and registration.exists():
        run_id = json.loads(registration.read_text())["development_run_id"]
        development = json.loads((directory / run_id / "development.json").read_text())
    return {
        "kind": "fixed-patterns", "criteria": CRITERIA, "support": SUPPORT,
        "development": development,
        "preregistration": json.loads(registration.read_text()) if registration.exists() else None,
        "locked": json.loads(locked.read_text()) if locked.exists() else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["stage", "develop", "preregister", "stage-locked", "evaluate-locked"])
    parser.add_argument("--acknowledge-review", action="store_true")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.phase == "stage":
        result = stage_chapters(root, STAGED_DIRECTORY, LEARN_CHAPTERS, "learning and development only")
    elif args.phase == "develop":
        result = develop(root, acknowledge_review=args.acknowledge_review)
    elif args.phase == "preregister":
        result = preregister(root, args.run_id)
    elif args.phase == "stage-locked":
        if not (root / PATTERN_DIRECTORY / "preregistration.json").exists():
            raise SystemExit("Pre-register criteria before fetching the locked chapter")
        result = stage_chapters(root, LOCKED_DIRECTORY, (LOCKED_CHAPTER,), "locked test; evaluated at most once")
    else:
        result = evaluate_locked(root, acknowledge_review=args.acknowledge_review)
    shown = {k: v for k, v in result.items() if k not in ("generation", "discovery")}
    print(json.dumps(shown, indent=1, ensure_ascii=False)[:6000])
