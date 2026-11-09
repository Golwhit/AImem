"""Manually extract character windows into a separate local review queue."""

import argparse
import hashlib
import json
from pathlib import Path

from seed_life import assess, boundary_text


MAX_SOURCE_CHARACTERS = 50_000
RULESET = "seed-life-v4"
DEFAULT_QUEUE = Path(__file__).resolve().parent / ".data" / "seed-extraction-review.json"


def extract(text: str) -> dict:
    if not isinstance(text, str):
        raise TypeError("Extraction input must be text")
    if not text:
        raise ValueError("Extraction input must not be empty")
    if len(text) > MAX_SOURCE_CHARACTERS:
        raise ValueError(f"Extraction input must not exceed {MAX_SOURCE_CHARACTERS} characters")

    groups: dict[str, dict[tuple[str, bool], list[int]]] = {"alive": {}, "dead": {}, "maybe": {}}
    for size in (1, 2, 3):
        for offset in range(len(text) - size + 1):
            window = text[offset:offset + size]
            assessed_text = boundary_text(window, offset == 0)
            result = assess(assessed_text)
            groups[result.status].setdefault((window, result.layer_zero_eligible), []).append(offset)

    return {
        status: [
            {
                "text": window,
                "offsets": offsets,
                "layer_zero_eligible": eligible,
            }
            for (window, eligible), offsets in sorted(patterns.items())
        ]
        for status, patterns in groups.items()
    }


def extract_file(source: Path, queue: Path = DEFAULT_QUEUE) -> dict:
    source = source.resolve()
    queue = queue.resolve()
    if source == queue:
        raise ValueError("Source and review queue must be different files")
    text = source.read_bytes().decode("utf-8")
    return extract_text(text, str(source), queue)


def extract_text(
    text: str,
    source: str,
    queue: Path,
    provenance: dict | None = None,
) -> dict:
    """Persist exact source text and extraction without approving any patterns."""
    from original_generator import atomic_json

    if not isinstance(source, str) or not source:
        raise ValueError("Extraction requires a nonempty source identifier")
    groups = extract(text)
    if queue.exists():
        state = json.loads(queue.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or state.get("version") != 1 or not isinstance(state.get("sources"), list):
            raise ValueError("Unsupported or invalid extraction review queue")
    else:
        state = {"version": 1, "sources": []}

    source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    record = {
        "path": source,
        "sha256": source_hash,
        "text": text,
        "characters": len(text),
        "ruleset": RULESET,
        "provenance": provenance or {},
        "review": groups["maybe"],
        "alive": groups["alive"],
        "dead": groups["dead"],
    }
    state["sources"] = [
        item for item in state["sources"]
        if not (item["path"] == source and item["sha256"] == source_hash and item["ruleset"] == RULESET)
    ]
    state["sources"].append(record)
    atomic_json(queue, state)
    return {
        "source": source,
        "sha256": source_hash,
        "queue": str(queue),
        "counts": {
            status: sum(len(item["offsets"]) for item in entries)
            for status, entries in groups.items()
        },
        "unique_unknown_patterns": len({item["text"] for item in groups["maybe"]}),
        "layer_zero_eligible_occurrences": sum(
            len(item["offsets"])
            for entries in groups.values()
            for item in entries
            if item["layer_zero_eligible"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="UTF-8 text file to extract")
    args = parser.parse_args()
    print(json.dumps(extract_file(args.source), indent=2))


if __name__ == "__main__":
    main()
