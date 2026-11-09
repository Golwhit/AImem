"""An isolated experiment with the initial English life-and-death rules."""

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Literal


# The marker comes from the earlier project; its exact meaning remains open.
TERMINUS = "</>"
CONSONANTS = frozenset("bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ")
LAYER_ZERO_SEEDS = (
    TERMINUS + "A",
    TERMINUS + "I",
    TERMINUS + "I ",
    TERMINUS + "A ",
    " a ",
    " A ",
    " I ",
)
ALIVE_SEEDS = frozenset((
    TERMINUS + "I ",
    TERMINUS + "A ",
    " a ",
    " A ",
    " I ",
))


@dataclass(frozen=True)
class Assessment:
    status: Literal["alive", "dead", "maybe"]
    reason: str
    layer_zero_eligible: bool


def exclusion_reason(text: str) -> str | None:
    """Apply only the agreed Layer 1 exclusions, not a positive grammar."""
    if not isinstance(text, str):
        raise TypeError("Seed input must be text")
    for index in range(len(text) - 2):
        if text[index] == text[index + 1] == text[index + 2]:
            return f"Three identical characters at position {index}"
    if " i " in text:
        return "Lowercase i surrounded by spaces cannot live"
    if text.startswith(TERMINUS + "I."):
        return "A response starting with I followed by a period cannot live"
    opening = text.removeprefix(TERMINUS)
    if text.startswith(TERMINUS) and len(opening) >= 2 and opening[0] in CONSONANTS and opening[1] == " ":
        return "A single consonant followed by a space at the terminus cannot live"
    return None


def boundary_text(text: str, at_source_start: bool) -> str:
    """Supply source-start context without duplicating an explicit marker."""
    return TERMINUS + text if at_source_start and not text.startswith((TERMINUS, " ")) else text


def assess(text: str) -> Assessment:
    """Assess exact seed forms and confirmed death rules, preserving all text."""
    if not isinstance(text, str):
        raise TypeError("Seed input must be text")
    eligible = text in LAYER_ZERO_SEEDS
    excluded = exclusion_reason(text)
    if excluded is not None:
        return Assessment("dead", excluded, eligible)
    if not eligible:
        return Assessment("maybe", "Outside frozen Layer 0; retained for review, not eligible for life", False)
    if text in ALIVE_SEEDS:
        return Assessment("alive", "Confirmed living seed form", True)
    return Assessment("maybe", "Eligible seed; more context is needed before it is alive", True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Exact text to assess, including spaces")
    args = parser.parse_args()
    print(json.dumps(asdict(assess(args.text))))


if __name__ == "__main__":
    main()
