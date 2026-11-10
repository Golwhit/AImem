"""An isolated experiment with the initial English life-and-death rules."""

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Literal


# Legacy assessment notation for an explicit source-start boundary.
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
DOMAIN = "english-seed-experiment"
DEVELOPMENT_VERSION = "seed-development-v1"
RULES = (
    {"id": "triple-identical", "domain": DOMAIN, "boundary": "anywhere",
     "rationale": "Agreed seed-experiment exclusion, not a universal language law",
     "excluded_example": "aaa", "surviving_example": "aa"},
    {"id": "lowercase-i-word", "domain": DOMAIN, "boundary": "literal space on both sides",
     "rationale": "Agreed single-letter lowercase i exclusion",
     "excluded_example": " i ", "surviving_example": " I "},
    {"id": "initial-i-period", "domain": DOMAIN, "boundary": "source start only",
     "rationale": "Agreed initial I followed by period exclusion",
     "excluded_example": "</>I.", "surviving_example": "</>I "},
    {"id": "initial-consonant-space", "domain": DOMAIN, "boundary": "source start only",
     "rationale": "Agreed initial single English consonant followed by literal space exclusion",
     "excluded_example": "</>t ", "surviving_example": "</>the "},
)


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
        return Assessment("maybe", "Outside current foundational seed forms; retained for developmental review, not admitted", False)
    if text in ALIVE_SEEDS:
        return Assessment("alive", "Confirmed living seed form", True)
    return Assessment("maybe", "Eligible seed; more context is needed before it is alive", True)


def development_assessment(
    text: str, *, at_source_start: bool, complete: bool, domain: str = DOMAIN,
) -> dict:
    """Keep maturity separate from exact foundation membership."""
    if domain != DOMAIN:
        raise ValueError(f"No Layer 1 rules are defined for domain {domain!r}")
    if not isinstance(text, str):
        raise TypeError("Development input must be exact text")
    if type(at_source_start) is not bool or type(complete) is not bool:
        raise ValueError("Source-start and completion boundaries must be explicit booleans")
    contextual = boundary_text(text, at_source_start)
    assessment = assess(contextual)
    if assessment.status == "dead":
        state = "excluded"
    elif assessment.status == "alive":
        state = "confirmed"
    elif not complete:
        state = "developing"
    else:
        state = "unreviewed"
    return {
        "state": state, "foundation_eligible": assessment.layer_zero_eligible,
        "reason": assessment.reason if state in ("excluded", "confirmed") else
        ("No exclusion detected; observation is incomplete, not confirmed or admitted" if not complete else
         "No exclusion detected; completed observation needs explicit review, not automatic admission"),
        "at_source_start": at_source_start, "complete": complete,
        "domain": domain, "version": DEVELOPMENT_VERSION,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Exact text to assess, including spaces")
    args = parser.parse_args()
    print(json.dumps(asdict(assess(args.text))))


if __name__ == "__main__":
    main()
