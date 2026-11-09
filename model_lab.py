"""Read-only, pre-training comparisons of seed rules and the existing model."""

import hashlib
import json
import random
from typing import TYPE_CHECKING

from seed_extraction import RULESET, extract
from seed_life import boundary_text, exclusion_reason

if TYPE_CHECKING:
    from original_generator import CharacterGenerator


def sample_blocks(captures: list[tuple[int, str]], budget: int, seed: int) -> str:
    rng = random.Random(seed)
    text = ""
    while True:
        fitting = [block for _, block in captures if len(text) + len(block) <= budget]
        if not fitting:
            return text
        text += rng.choice(fitting)


def overlay_captures(text: str, captures: list[tuple[int, str]]) -> tuple[str, list[dict]]:
    characters = list(text)
    replacements = []
    opening_reason = exclusion_reason(boundary_text(text[:2], True))
    if opening_reason:
        candidates = sorted(
            (offset, block) for offset, block in captures
            if offset == 0 and block in ("A ", "I ")
        )
        if candidates:
            source_offset, block = candidates[0]
            characters[0] = block[0]
            replacements.append({
                "offset": 0, "before": text[0], "after": block[0],
                "matched_text": text[:2], "capture": block,
                "source_offset": source_offset, "reason": opening_reason,
            })
    for offset, character in enumerate(text):
        if character not in ("a", "i"):
            continue
        for source_offset, block in captures:
            if source_offset == 0 and block in ("A ", "I "):
                start = 0
                if offset != 0:
                    continue
            elif block in (" A ", " I "):
                start = offset - 1
                if start < 0:
                    continue
            else:
                continue
            before = text[start:start + len(block)]
            if before != block.lower():
                continue
            characters[offset] = character.upper()
            replacements.append({
                "offset": offset, "before": character, "after": character.upper(),
                "matched_text": before, "capture": block,
                "source_offset": source_offset,
            })
            break
    return "".join(characters), replacements


def compare(generator: "CharacterGenerator", prompt: str, budget: int = 320) -> dict:
    from original_generator import END, MAX_GENERATED_CHARACTERS, ORDER, START
    from original_response_scorer import ResponseScorer

    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 2000:
        raise ValueError("Lab prompt must be nonempty text up to 2000 characters")
    if not isinstance(budget, int) or isinstance(budget, bool) or not 3 <= budget <= MAX_GENERATED_CHARACTERS:
        raise ValueError(f"Lab character budget must be between 3 and {MAX_GENERATED_CHARACTERS}")
    groups = extract(prompt)
    baseline = []
    layer_zero = []
    layer_one = []
    combined = []
    for status, entries in groups.items():
        for entry in entries:
            baseline.append({"text": entry["text"], "offsets": entry["offsets"]})
            layer_zero.append({
                **entry,
                "status": "eligible" if entry["layer_zero_eligible"] else "outside",
            })
            for excluded in (False, True):
                offsets = [
                    offset for offset in entry["offsets"]
                    if (exclusion_reason(boundary_text(entry["text"], offset == 0)) is not None) == excluded
                ]
                if offsets:
                    layer_one.append({
                        "text": entry["text"], "offsets": offsets,
                        "status": "excluded" if excluded else "not_excluded",
                    })
            combined.append({**entry, "status": status})
    normalized = generator._normalize(prompt)
    transitions = generator._rebuild([{"text": normalized}], 0)["transitions"]
    old_extraction = [
        {"context": context, "next": char, "count": count}
        for context, counts in sorted(transitions.items())
        for char, count in sorted(counts.items())
    ]
    checkpoint_hash = hashlib.sha256(
        json.dumps(generator.state, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    seed = int.from_bytes(hashlib.sha256(
        f"{checkpoint_hash}\0{prompt}\0{budget}\0{RULESET}".encode("utf-8")
    ).digest()[:6], "big")
    scorer = ResponseScorer(generator.root)
    old = scorer.select(prompt.strip(), [
        generator.generate(prompt.strip(), budget, seed + index) for index in range(4)
    ])
    captures = [
        (offset, item["text"])
        for item in groups["alive"]
        for offset in item["offsets"]
        if item["layer_zero_eligible"]
    ]
    captures.sort()
    layers_text = sample_blocks(captures, budget, seed)

    protected = captures[0][1] if captures else ""
    combined_text = protected
    reason = "No living seed was captured" if not captures else ""
    context_mode = None
    if protected:
        if captures[0][0] == 0:
            context = (START * ORDER + protected)[-ORDER:]
            contexts = [context] if context in generator.state["transitions"] else []
            context_mode = "source_start"
        else:
            contexts = [
                context for context in generator.state["transitions"]
                if context.endswith(protected) and END not in context
            ]
            context_mode = "suffix_backoff"
        if not contexts:
            reason = "Existing model has no context matching the captured seed"
        else:
            rng = random.Random(seed)
            contexts.sort()
            context = rng.choices(contexts, weights=[
                sum(generator.state["transitions"][candidate].values()) for candidate in contexts
            ], k=1)[0]
            while len(combined_text) < budget:
                counts = generator.state["transitions"].get(context)
                if not counts:
                    reason = "Existing model has no further transition"
                    break
                char = rng.choices(list(counts), weights=list(counts.values()), k=1)[0]
                if char == END:
                    reason = "Existing model reached its end marker"
                    break
                combined_text += char
                context = (context + char)[-ORDER:]

    overlaid, replacements = overlay_captures(old["text"], captures)
    opening_reason = exclusion_reason(boundary_text(overlaid[:2], True))
    if opening_reason:
        old_first_reason = f"Unresolved: {opening_reason}; no living response-start A or I capture supports a replacement"
    elif not old["text"]:
        old_first_reason = "Existing generator produced no output; layers did not replace it"
    elif not captures:
        old_first_reason = "No living seed was captured for an aligned overlay"
    elif not replacements:
        old_first_reason = "No matching lowercase letter and boundary; old output is unchanged"
    else:
        old_first_reason = ""
    outputs = {
        "old": {"text": old["text"], "reason": "" if old["text"] else "Existing generator is not ready", "selection": old.get("selection")},
        "layers": {"text": layers_text, "reason": "" if layers_text else "No living seed was captured"},
        "combined": {
            "text": combined_text, "reason": reason, "protected_prefix": protected,
            "context_mode": context_mode,
        },
        "old_first": {
            "text": overlaid,
            "reason": old_first_reason,
            "base_text": old["text"],
            "replacements": replacements,
            "context_mode": "aligned_capture_overlay",
        },
    }
    common_length = min(len(item["text"]) for item in outputs.values())
    for item in outputs.values():
        item["characters"] = len(item["text"])
        item["display_text"] = item["text"][:common_length]
        item["shortfall"] = budget - len(item["text"])
    return {
        "version": 4, "ruleset": RULESET, "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "checkpoint_sha256": checkpoint_hash, "generation": generator.state["generation"],
        "scorer_generation": scorer.state["generation"], "seed": seed,
        "scorer_sha256": hashlib.sha256(
            json.dumps(scorer.state, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "timing": "before_prompt_training", "character_budget": budget,
        "common_display_characters": common_length,
        "extraction": {
            "baseline_windows": baseline, "old_model_transitions": old_extraction,
            "layer_zero": layer_zero, "layer_one": layer_one, "combined": combined,
        },
        "generation_comparison": outputs,
        "old_response": old,
    }
