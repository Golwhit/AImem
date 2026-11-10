"""Reversible lab-only building-block admission with exact saved evidence."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
from uuid import uuid4

from seed_life import DEVELOPMENT_VERSION, DOMAIN, RULES, development_assessment


class DevelopmentRegistry:
    def __init__(self, root: Path):
        self.root = root
        self.path = root / ".data" / "seed-development.json"
        self.state = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {
            "version": 1, "events": [],
        }
        if self.state.get("version") != 1 or not isinstance(self.state.get("events"), list):
            raise ValueError("Unsupported development registry; no changes made")

    def status(self) -> dict:
        latest = {}
        for event in self.state["events"]:
            latest[event["candidate_id"]] = event
        return {
            "version": DEVELOPMENT_VERSION, "domain": DOMAIN, "rules": RULES,
            "candidates": list(latest.values()), "events": self.state["events"],
            "sha256": hashlib.sha256(json.dumps(self.state, sort_keys=True).encode()).hexdigest(),
            "policy": "Explicit reversible lab-generation admission only; no training approval or legal clearance. Layer 0 is unchanged.",
        }

    def decide(self, payload: dict) -> dict:
        from original_generator import atomic_json

        if not isinstance(payload, dict) or payload.get("acknowledge_lab_only") is not True:
            raise ValueError("Explicit lab-only review acknowledgment required; this does not approve training")
        action = payload.get("action")
        if action not in ("admit", "defer", "revoke"):
            raise ValueError("Choose admit, defer, or revoke")
        rationale = payload.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 1000:
            raise ValueError("A review rationale of 1–1000 characters is required")
        record_id = payload.get("record_id")
        if not isinstance(record_id, str) or not re.fullmatch(r"[a-f0-9]{64}", record_id):
            raise ValueError("Choose a saved lab record ID")
        record_path = self.root / ".data" / "model-lab" / f"{record_id}.json"
        if not record_path.exists():
            raise ValueError("Saved lab evidence not found")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        text = record["prompt"]
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest != record["prompt_sha256"]:
            raise ValueError("Saved source hash mismatch; admission blocked")
        offset, size = payload.get("offset"), payload.get("size")
        if type(offset) is not int or type(size) is not int or not 1 <= size <= 4 or not 0 <= offset <= len(text) - size:
            raise ValueError("Select an exact source offset and a 1–4-character pattern")
        block = text[offset:offset + size]
        # Admission preserves observed start/interior placement, never invents a boundary.
        at_start = offset == 0
        assessment = development_assessment(block, at_source_start=at_start, complete=True)
        surrounding = text[max(0, offset - 2):min(len(text), offset + size + 2)]
        surrounding_assessment = development_assessment(
            surrounding, at_source_start=offset <= 2, complete=False,
        )
        tests = [
            {"name": "exact_source_and_hash", "passed": True},
            {"name": "standalone_boundary_exclusions", "passed": assessment["state"] != "excluded",
             "assessment": assessment},
            {"name": "observed_join_exclusions", "passed": surrounding_assessment["state"] != "excluded",
             "text": surrounding, "assessment": surrounding_assessment},
        ]
        if action == "admit" and not all(test["passed"] for test in tests):
            raise ValueError("Candidate failed boundary/composition checks; defer it instead of admitting")
        candidate_id = hashlib.sha256(f"{DOMAIN}\0{at_start}\0{block}".encode()).hexdigest()
        if action == "revoke" and not any(
            event["candidate_id"] == candidate_id for event in self.state["events"]
        ):
            raise ValueError("Cannot revoke a candidate without an existing review")
        self.state["events"].append({
            "event_id": uuid4().hex, "candidate_id": candidate_id,
            "action": action, "text": block, "placement": "start" if at_start else "interior",
            "domain": DOMAIN, "rules_version": DEVELOPMENT_VERSION,
            "record_id": record_id, "source_sha256": digest, "offset": offset,
            "source_text": text, "provenance": "Explicit user review of saved lab prompt",
            "rationale": rationale.strip(), "tests": tests,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "training_approved": False, "legal_clearance": False,
        })
        atomic_json(self.path, self.state)
        return self.status()


def compose_blocks(blocks: list[dict], budget: int, seed: int, attraction) -> dict:
    """Foundation/explicitly admitted blocks -> whole-join exclusions -> attraction."""
    rng = random.Random(seed)
    text = ""
    trace = []
    reason = "Character budget reached"
    while len(text) < budget:
        candidates, rejected = [], []
        for block in blocks:
            content = block["text"]
            if not content or len(text) + len(content) > budget:
                continue
            if (block["placement"] == "start" and text) or (block["placement"] == "interior" and not text):
                continue
            assessment = development_assessment(text + content, at_source_start=True, complete=False)
            if assessment["state"] == "excluded":
                rejected.append({"text": content, "reason": assessment["reason"]})
                continue
            loss = 0.0
            for index, character in enumerate(content):
                loss -= math.log(max(attraction.distribution(text + content[:index]).get(character, 0), 1e-12))
            weight = math.exp(-loss / len(content))
            candidates.append((block, weight))
        if not candidates:
            reason = "No living seed was captured or admitted block fits this boundary/budget without an exclusion"
            trace.append({"offset": len(text), "rejected": rejected, "selected": None})
            break
        block = rng.choices([item[0] for item in candidates], weights=[item[1] for item in candidates])[0]
        trace.append({
            "offset": len(text), "selected": block, "rejected": rejected,
            "candidates": [{"text": item["text"], "weight": weight} for item, weight in candidates],
        })
        text += block["text"]
    return {
        "text": text, "reason": reason, "trace": trace, "context_mode": "developmental_blocks",
        "assessment": development_assessment(text, at_source_start=True, complete=True),
        "policy": "Foundation captures or explicitly admitted developmental blocks only; whole-join Layer 1 checks then Layer 2 ranking. No training approval.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["status", "decide"])
    args = parser.parse_args()
    try:
        registry = DevelopmentRegistry(Path(__file__).resolve().parent)
        result = registry.status() if args.command == "status" else registry.decide(json.load(sys.stdin))
    except (ValueError, TypeError, OSError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
