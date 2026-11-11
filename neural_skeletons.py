"""Strict skeletons: patterns with hidden or missing letters that can resolve to only one possible word.

Layer 0 here is the list of possible words: the SCOWL word list (permissive notice license, see
THIRD_PARTY_NOTICES.md) plus every word in the learning text. A skeleton is a word with 1 or 2 letters
hidden ("th*t") or with one letter missing ("tht"). It is strict when exactly one possible word fits it and
that word was seen at least SUPPORT times in learning text: "if the model sees this, it is this".
Anything weaker is light: a skeleton that only the learning text resolves uniquely (e.g. "sa*d" -> said,
although "sand" and "sad" exist), and phrase skeletons such as "a*d th*t" whose words are not each forced.
A phrase is strict only when every hidden letter is forced by the word list, so it adds no strict claims of
its own beyond its words.

Development revisions (made on learn/develop books only, before pre-registration):
- Names are an open class no word list can complete, so a word capitalized in most of its learning uses is
  never a strict target, and a seen skeleton containing a capital letter is handed on, not claimed.
- Apostrophes count only inside words; a closing quote mark is not part of "father".
- Layer 0 also lists regular forms (-s, -ed, -er, -ing, un-, re-, -ies, ...) of every possible word. These
  forms over-generate on purpose: a made-up form can only veto a strict claim, never create one.
- A skeleton whose first letter is hidden is never strict, because hiding it hides whether it is a name.

Books of the public-domain World English Bible are split by canonical position before analysis:
position % 4 in (0, 1) learn, 2 develop, 3 locked. Acts is always learning text (chapters 1-28 were already
used) and Acts 3 is excluded everywhere. Locked books are read only by the single locked evaluation.
Standard library only.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import time
import zipfile

from neural_proof import fetch
from original_generator import atomic_json


ZIP_URL = "https://ebible.org/Scriptures/engwebp_readaloud.zip"
RIGHTS_URL = "https://ebible.org/engwebp/copyright.htm"
BIBLE_DIRECTORY = ".data/bible-web-staged"
WORDS_DIRECTORY = ".data/possible-words"
SKELETON_DIRECTORY = ".data/neural-skeletons"
DICTIONARY = Path("/usr/share/dict/american-english")
DICTIONARY_NOTICE = Path("/usr/share/doc/wamerican/copyright")
WORD = re.compile(r"[A-Za-z]+(?:[’'][A-Za-z]+)*")  # apostrophes only inside words; a closing ’ is a quote mark
FILL = "abcdefghijklmnopqrstuvwxyz'"
MIN_LENGTH = 3
SUPPORT = 5
SUPPORTS = (1, 5, 20)
KINDS = ("hidden1", "hidden2", "dropped1")
TARGET = 1e-5
CRITERIA = {
    "K1_observed": "On the locked books, strict skeletons (support 5, all kinds) are wrong at most 1 in 100,000 checks.",
    "K2_bound": "The one-sided 95% upper bound on that error rate is at most 1 in 100,000 checks.",
}
REVISIONS = ("After development (learn -> develop): names excluded from strict and capitalized skeletons handed on; "
             "apostrophes only inside words; regular forms of possible words added to Layer 0 as vetoes; "
             "no strict claim when the first letter is hidden. User chose these rules (A+B) before the locked run.")
APPROVAL = ("User: use the SCOWL word list as the Layer 0 list of possible words (keep its notice); fetch the "
            "whole public-domain WEB Bible, learn on most books and lock some away for the 1-in-100,000 test; "
            "phrases are strict only if every hidden letter is forced by the word list, otherwise light.")


def normal(word: str) -> str:
    return word.replace("’", "'").lower()


def skeletons(word: str, kind: str) -> set[str]:
    if kind == "hidden1":
        return {word[:i] + "*" + word[i + 1:] for i in range(len(word))}
    if kind == "hidden2":
        return {word[:i] + "*" + word[i + 1:j] + "*" + word[j + 1:]
                for i in range(len(word)) for j in range(i + 1, len(word)) if len(word) - 2 >= 2}
    if kind == "dropped1":
        return {word[:i] + word[i + 1:] for i in range(len(word))}
    raise ValueError(f"Unknown skeleton kind {kind}")


class Possible:
    """Which possible words fit a skeleton; only 0, 1 or 'more than one' matters."""

    def __init__(self, words: set[str]):
        self.words = words
        self.hidden: dict[str, str | None] = {}
        self.dropped: dict[str, str | None] = {}
        for word in words:
            for index, table in ((skeletons(word, "hidden1"), self.hidden), (skeletons(word, "dropped1"), self.dropped)):
                for skeleton in index:
                    table[skeleton] = word if table.get(skeleton, word) == word else None

    def fits(self, skeleton: str, kind: str) -> set[str] | None:
        """The single fitting word as a one-item set, an empty set, or None when more than one fits."""
        if kind == "hidden1":
            found = self.hidden.get(skeleton, "")
            return None if found is None else ({found} if found else set())
        if kind == "hidden2":
            first = skeleton.index("*")
            seen: set[str] = set()
            for letter in FILL:
                found = self.hidden.get(skeleton[:first] + letter + skeleton[first + 1:], "")
                if found is None:
                    return None
                if found:
                    seen.add(found)
                    if len(seen) > 1:
                        return None
            return seen
        found = self.dropped.get(skeleton, "")
        if found is None:
            return None
        candidates = ({found} if found else set()) | ({skeleton} if skeleton in self.words else set())
        return None if len(candidates) > 1 else candidates


def regular_forms(words: set[str]) -> set[str]:
    """Regular English forms of possible words, deliberately over-generated; used only to veto strict claims."""
    derived = set()
    for word in words:
        if "'" in word:
            continue
        derived |= {word + end for end in ("s", "es", "ed", "d", "er", "r", "ers", "rs", "ing", "ly", "ness")}
        derived |= {"un" + word, "re" + word}
        if word.endswith("y"):
            derived |= {word[:-1] + end for end in ("ies", "ied", "ier", "iers", "ily", "iness")}
        if word.endswith("e"):
            derived |= {word[:-1] + "ing", word[:-1] + "er"}
    return derived - words


def words_of(text: str) -> list[str]:
    return [normal(match.group()) for match in WORD.finditer(text)]


def claimable(skeleton: str) -> bool:
    """A visibly capitalized skeleton may be a name, an open class, so the strict layer hands it on."""
    return skeleton == skeleton.lower()


class StrictSkeletons:
    def __init__(self, learning: str, dictionary: set[str], *, support: int = SUPPORT,
                 indexes: tuple[Possible, Possible] | None = None, derived: set[str] | None = None):
        tokens = words_of(learning)
        self.counts = Counter(tokens)
        capitalized = Counter(normal(m.group()) for m in WORD.finditer(learning) if m.group()[0].isupper())
        self.names = {word for word, count in self.counts.items() if capitalized[word] * 2 > count}
        self.possible, self.corpus = indexes or (Possible(dictionary | set(self.counts)), Possible(set(self.counts)))
        self.support = support
        self._fits_cache: dict[str, set[str]] = {}
        self.strict: dict[str, dict[str, str]] = {kind: {} for kind in KINDS}
        self.light: dict[str, dict[str, str]] = {kind: {} for kind in KINDS}
        for word, count in self.counts.items():
            if len(word) < MIN_LENGTH or count < support:
                continue
            for kind in KINDS:
                for skeleton in skeletons(word, kind):
                    if (word not in self.names and not skeleton.startswith("*")
                            and self.possible.fits(skeleton, kind) == {word}):
                        self.strict[kind][skeleton] = word
                    elif self.corpus.fits(skeleton, kind) == {word}:
                        self.light[kind][skeleton] = word
        if derived is None:
            derived = regular_forms(self.possible.words)
        self.vetoed = 0
        for kind in KINDS:
            table = self.strict[kind]
            for form in derived:
                for skeleton in skeletons(form, kind) | ({form} if kind == "dropped1" else set()):
                    if table.pop(skeleton, None) is not None:
                        self.vetoed += 1
        self.bigrams = Counter(zip(tokens, tokens[1:]))
        self.light_phrases: dict[str, tuple[str, str]] = {}
        for (first, second), count in self.bigrams.items():
            if count < support or len(first) < MIN_LENGTH or len(second) < MIN_LENGTH:
                continue
            for left in skeletons(first, "hidden1"):
                for right in skeletons(second, "hidden1"):
                    if left in self.strict["hidden1"] and right in self.strict["hidden1"]:
                        continue  # already strict through its words
                    pairs = {(x, y) for x in self._corpus_fits(left) for y in self._corpus_fits(right) if (x, y) in self.bigrams}
                    if pairs == {(first, second)}:
                        self.light_phrases[left + " " + right] = (first, second)

    def _corpus_fits(self, skeleton: str) -> set[str]:
        if skeleton not in self._fits_cache:
            index = skeleton.index("*")
            self._fits_cache[skeleton] = {skeleton[:index] + letter + skeleton[index + 1:] for letter in FILL} & self.corpus.words
        return self._fits_cache[skeleton]

    def summary(self) -> dict:
        examples = {}
        for kind in KINDS:
            ranked = sorted(self.strict[kind].items(), key=lambda kv: (-self.counts[kv[1]], kv[0]))
            examples[kind] = [{"skeleton": s, "word": w, "support": self.counts[w]} for s, w in ranked[:20]]
        light_ranked = sorted(self.light["hidden1"].items(), key=lambda kv: -self.counts[kv[1]])[:12]
        phrase_ranked = sorted(self.light_phrases.items(), key=lambda kv: -self.bigrams[kv[1]])[:12]
        return {
            "support": self.support, "possible_words": len(self.possible.words), "learning_words": len(self.counts),
            "names_excluded_from_strict": len(self.names), "vetoed_by_regular_forms": self.vetoed,
            "strict": {kind: len(table) for kind, table in self.strict.items()},
            "strict_words": len({w for table in self.strict.values() for w in table.values()}),
            "light": {kind: len(table) for kind, table in self.light.items()},
            "light_phrases": len(self.light_phrases),
            "strict_examples": examples,
            "light_examples": [{"skeleton": s, "word": w, "support": self.counts[w]} for s, w in light_ranked],
            "light_phrase_examples": [{"skeleton": s, "phrase": " ".join(p), "support": self.bigrams[p]} for s, p in phrase_ranked],
        }

    def check(self, text: str) -> dict:
        originals = [m.group().replace("’", "'") for m in WORD.finditer(text)]
        tokens = [word.lower() for word in originals]
        result = {}
        for layer, tables in (("strict", self.strict), ("light", self.light)):
            per_kind = {}
            for kind in KINDS:
                table = tables[kind]
                checks = wrong = total = handed = 0
                wrongs = Counter()
                for original in originals:
                    word = original.lower()
                    if len(word) < MIN_LENGTH:
                        continue
                    for shown in skeletons(original, kind):
                        total += 1
                        skeleton = shown.lower()
                        if layer == "strict" and not claimable(shown):
                            handed += skeleton in table
                            continue
                        found = table.get(skeleton)
                        if found is not None:
                            checks += 1
                            if found != word:
                                wrong += 1
                                wrongs[(skeleton, found, word)] += 1
                per_kind[kind] = {
                    "skeletons_seen": total, "checks": checks, "wrong": wrong, "handed_on_capitalized": handed,
                    "share_resolvable": checks / total if total else 0.0,
                    "error_rate": wrong / checks if checks else None,
                    "upper_95": upper_bound(wrong, checks) if checks else None,
                    "wrong_examples": [{"skeleton": s, "said": f, "was": w, "times": c} for (s, f, w), c in wrongs.most_common(12)],
                }
            checks = sum(item["checks"] for item in per_kind.values())
            wrong = sum(item["wrong"] for item in per_kind.values())
            result[layer] = {"kinds": per_kind, "checks": checks, "wrong": wrong,
                             "error_rate": wrong / checks if checks else None,
                             "upper_95": upper_bound(wrong, checks) if checks else None}
        occurrences = [w for w in originals if len(w) >= MIN_LENGTH]
        claims = lambda w, k: [(s.lower(), w.lower()) for s in skeletons(w, k) if claimable(s) and s.lower() in self.strict[k]]
        with_strict = sum(1 for w in occurrences if claims(w, "hidden1"))
        words_wrong = sum(1 for w in occurrences if any(self.strict[k][s] != low for k in KINDS for s, low in claims(w, k)))
        result["strict"]["word_occurrences"] = len(occurrences)
        result["strict"]["word_occurrences_with_a_strict_hidden1_skeleton"] = with_strict
        result["strict"]["word_occurrences_with_any_wrong_claim"] = words_wrong
        result["strict"]["word_upper_95"] = upper_bound(words_wrong, len(occurrences)) if occurrences else None
        phrase_checks = phrase_wrong = 0
        phrase_wrongs = Counter()
        for first, second in zip(tokens, tokens[1:]):
            if len(first) < MIN_LENGTH or len(second) < MIN_LENGTH:
                continue
            for left in skeletons(first, "hidden1"):
                for right in skeletons(second, "hidden1"):
                    found = self.light_phrases.get(left + " " + right)
                    if found is not None:
                        phrase_checks += 1
                        if found != (first, second):
                            phrase_wrong += 1
                            phrase_wrongs[(left + " " + right, " ".join(found), first + " " + second)] += 1
        result["light_phrases"] = {
            "checks": phrase_checks, "wrong": phrase_wrong,
            "error_rate": phrase_wrong / phrase_checks if phrase_checks else None,
            "wrong_examples": [{"skeleton": s, "said": f, "was": w, "times": c} for (s, f, w), c in phrase_wrongs.most_common(12)],
        }
        return result


def upper_bound(errors: int, trials: int, confidence: float = 0.95) -> float:
    """Exact one-sided Clopper-Pearson upper bound for an error rate."""
    if trials <= 0:
        raise ValueError("Upper bound needs at least one trial")
    if errors >= trials:
        return 1.0
    alpha = 1 - confidence

    def cdf(p: float) -> float:
        total = 0.0
        for k in range(errors + 1):
            total += math.exp(math.lgamma(trials + 1) - math.lgamma(k + 1) - math.lgamma(trials - k + 1)
                              + k * math.log(p) + (trials - k) * math.log1p(-p))
        return total

    low, high = errors / trials, 1.0
    for _ in range(200):
        middle = (low + high) / 2
        if cdf(middle) > alpha:
            low = middle
        else:
            high = middle
    return high


def chapter_text(raw: bytes) -> str:
    lines = raw.decode("utf-8-sig").replace("\r", "").split("\n")
    if len(lines) < 3 or not lines[1].startswith("Chapter "):
        raise ValueError("Unexpected read-aloud chapter layout")
    body = [" ".join(line.split()) for line in lines[2:]]
    return "\n".join(line for line in body if line) + "\n"


def chapters(archive: zipfile.ZipFile) -> list[tuple[str, str, int]]:
    found = []
    for name in archive.namelist():
        parts = name.split("_")
        if name.endswith("_read.txt") and len(parts) == 5 and parts[1] != "000":
            found.append((name, parts[2], int(parts[3])))
    return sorted(found, key=lambda item: (item[0].split("_")[1], item[2]))


def book_groups(names: list[tuple[str, str, int]]) -> dict[str, str]:
    books = list(dict.fromkeys(book for _, book, _ in names))
    if len(books) != 66:
        raise ValueError("Expected the 66-book canon")
    groups = {}
    for position, book in enumerate(books):
        group = position % 4
        groups[book] = "learn" if book == "ACT" or group in (0, 1) else "develop" if group == 2 else "locked"
    return groups


def stage(root: Path, fetcher=fetch) -> dict:
    target = root / BIBLE_DIRECTORY
    if target.exists():
        raise ValueError("Bible already staged; refusing to overwrite")
    raw = fetcher(ZIP_URL)
    rights = fetcher(RIGHTS_URL)
    archive = zipfile.ZipFile(io.BytesIO(raw))
    names = chapters(archive)
    groups = book_groups(names)
    sources = []
    for name, book, chapter in names:
        if book == "ACT" and chapter == 3:
            continue
        text = chapter_text(archive.read(name))
        sources.append({"file": name, "book": book, "chapter": chapter, "group": groups[book],
                        "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "characters": len(text)})
    words = root / WORDS_DIRECTORY
    if not words.exists():
        words.mkdir(parents=True)
        shutil.copyfile(DICTIONARY, words / "american-english")
        shutil.copyfile(DICTIONARY_NOTICE, words / "copyright")
        atomic_json(words / "manifest.json", {
            "source": str(DICTIONARY), "package": "wamerican (SCOWL)", "notice": "copyright (kept as required)",
            "sha256": hashlib.sha256(DICTIONARY.read_bytes()).hexdigest(),
            "notice_sha256": hashlib.sha256(DICTIONARY_NOTICE.read_bytes()).hexdigest(),
            "role": "Layer 0 list of possible words; not training text.", "approval": APPROVAL,
        })
    target.mkdir(parents=True)
    (target / "engwebp_readaloud.zip").write_bytes(raw)
    (target / "rights.html").write_bytes(rights)
    manifest = {
        "translation": "World English Bible, updated 66-book edition (engwebp), read-aloud text", "url": ZIP_URL,
        "zip_sha256": hashlib.sha256(raw).hexdigest(), "rights_url": RIGHTS_URL,
        "rights_sha256": hashlib.sha256(rights).hexdigest(),
        "rights_statement": "Public domain; 'World English Bible' is a trademark, so changed or generated text must not be called the World English Bible.",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "split": "Canonical book position % 4: 0-1 learn, 2 develop, 3 locked. Acts always learn; Acts 3 excluded.",
        "preprocessing": "Drop book-title and chapter-heading lines; collapse whitespace; keep one verse per line.",
        "approval": APPROVAL, "groups": groups, "sources": sources,
    }
    atomic_json(target / "manifest.json", manifest)
    return {key: value for key, value in manifest.items() if key != "sources"}


def load(root: Path, groups: tuple[str, ...]) -> tuple[str, dict]:
    target = root / BIBLE_DIRECTORY
    manifest_bytes = (target / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    raw = (target / "engwebp_readaloud.zip").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["zip_sha256"]:
        raise ValueError("Bible zip hash mismatch")
    if "locked" in groups and not (root / SKELETON_DIRECTORY / "preregistration.json").exists():
        raise ValueError("Locked books are read only after pre-registration")
    archive = zipfile.ZipFile(io.BytesIO(raw))
    parts = []
    for source in manifest["sources"]:
        if source["group"] in groups:
            text = chapter_text(archive.read(source["file"]))
            if hashlib.sha256(text.encode()).hexdigest() != source["text_sha256"]:
                raise ValueError(f"{source['file']} text hash mismatch")
            parts.append(text)
    text = "".join(parts)
    return text, {"groups": list(groups), "chapters": sum(1 for s in manifest["sources"] if s["group"] in groups),
                  "books": sorted({s["book"] for s in manifest["sources"] if s["group"] in groups}),
                  "characters": len(text), "words": len(WORD.findall(text)),
                  "sha256": hashlib.sha256(text.encode()).hexdigest(),
                  "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}


def dictionary(root: Path) -> tuple[set[str], str]:
    words = root / WORDS_DIRECTORY
    manifest = json.loads((words / "manifest.json").read_text())
    raw = (words / "american-english").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["sha256"]:
        raise ValueError("Possible-word list hash mismatch")
    return {normal(line) for line in raw.decode("utf-8").split("\n") if WORD.fullmatch(line.strip() or "-")}, manifest["sha256"]


def measure(root: Path, learn_groups: tuple[str, ...], held_groups: tuple[str, ...], supports=SUPPORTS) -> dict:
    started = time.perf_counter()
    learning, learning_info = load(root, learn_groups)
    held, held_info = load(root, held_groups)
    possible, dictionary_sha = dictionary(root)
    report = {"learning": learning_info, "held": held_info, "dictionary_sha256": dictionary_sha,
              "dictionary_words": len(possible), "by_support": {}}
    learned = set(words_of(learning))
    indexes = (Possible(possible | learned), Possible(learned))
    derived = regular_forms(possible | learned)
    report["regular_forms"] = len(derived)
    for support in supports:
        states = StrictSkeletons(learning, possible, support=support, indexes=indexes, derived=derived)
        report["by_support"][str(support)] = {"discovery": states.summary(), "held_check": states.check(held)}
    report["wall_seconds"] = time.perf_counter() - started
    return report


def develop(root: Path) -> dict:
    report = {"kind": "skeleton-development", "created_at": datetime.now(timezone.utc).isoformat(),
              "criteria": CRITERIA, "approval": APPROVAL, "revisions": REVISIONS, **measure(root, ("learn",), ("develop",))}
    atomic_json(root / SKELETON_DIRECTORY / "development.json", report)
    atomic_json(root / ".data/neural-experiments/latest-skeletons.json", public_report(root))
    return report


def preregister(root: Path) -> dict:
    directory = root / SKELETON_DIRECTORY
    path = directory / "preregistration.json"
    if path.exists():
        raise ValueError("Skeleton criteria already pre-registered")
    development = (directory / "development.json").read_bytes()
    record = {
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "development_sha256": hashlib.sha256(development).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "primary": {"support": SUPPORT, "kinds": list(KINDS), "layer": "strict"},
        "learning_groups": ["learn", "develop"], "locked_group": "locked",
        "criteria": CRITERIA, "target": TARGET, "revisions": REVISIONS,
        "unit": "A check is one strict skeleton of one word occurrence in the locked books. Checks from the same word occurrence are related, so word-level counts are reported too.",
    }
    atomic_json(path, record)
    return record


def evaluate_locked(root: Path) -> dict:
    directory = root / SKELETON_DIRECTORY
    output = directory / "locked-evaluation.json"
    if output.exists():
        raise ValueError("Locked books were already evaluated once; refusing a second evaluation")
    registration = json.loads((directory / "preregistration.json").read_text())
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != registration["code_sha256"]:
        raise ValueError("Skeleton code changed after pre-registration")
    if hashlib.sha256((directory / "development.json").read_bytes()).hexdigest() != registration["development_sha256"]:
        raise ValueError("Development report changed after pre-registration")
    atomic_json(output, {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()})
    report = measure(root, ("learn", "develop"), ("locked",), supports=(SUPPORT,))
    strict = report["by_support"][str(SUPPORT)]["held_check"]["strict"]
    record = {
        "status": "completed", "evaluated_at": datetime.now(timezone.utc).isoformat(), **report,
        "verdicts": {
            "K1_observed": {"value": strict["error_rate"], "target": TARGET, "met": strict["error_rate"] <= TARGET},
            "K2_bound": {"value": strict["upper_95"], "target": TARGET, "met": strict["upper_95"] <= TARGET},
        },
        "note": "Single evaluation. All text is one translation of one book collection; not general-English evidence.",
    }
    atomic_json(output, record)
    atomic_json(root / ".data/neural-experiments/latest-skeletons.json", public_report(root))
    return record


def public_report(root: Path) -> dict:
    directory = root / SKELETON_DIRECTORY
    read = lambda name: json.loads((directory / name).read_text()) if (directory / name).exists() else None
    return {"kind": "strict-skeletons", "criteria": CRITERIA, "support": SUPPORT, "target": TARGET,
            "development": read("development.json"), "preregistration": read("preregistration.json"),
            "locked": read("locked-evaluation.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["stage", "develop", "preregister", "evaluate-locked"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    result = {"stage": stage, "develop": develop, "preregister": preregister, "evaluate-locked": evaluate_locked}[args.phase](root)
    print(json.dumps(result, indent=1, ensure_ascii=False)[:5000])
