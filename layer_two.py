"""Read-only directional adjacency evidence; likelihood is not permission."""

from collections import Counter
import hashlib
import math
import random

from seed_life import boundary_text, exclusion_reason


VERSION = "character-attraction-evidence-v1"


def lower_bound(successes: int, opportunities: int) -> float:
    """Wilson 95% lower bound under an independent-trials assumption."""
    z = 1.959963984540054
    proportion = successes / opportunities
    return (
        proportion + z * z / (2 * opportunities)
        - z * math.sqrt(proportion * (1 - proportion) / opportunities + z * z / (4 * opportunities**2))
    ) / (1 + z * z / opportunities)


def analyze(text: str, *, max_characters: int = 2000) -> dict:
    if not isinstance(text, str) or not text or len(text) > max_characters:
        raise ValueError(f"Layer 2 evidence requires 1–{max_characters} exact characters")
    patterns: dict[str, list[int]] = {}
    context_counts: Counter[str] = Counter()
    endings: Counter[str] = Counter()
    for size in (1, 2, 3):
        for offset in range(len(text) - size + 1):
            context = text[offset:offset + size]
            context_counts[context] += 1
            if offset + size == len(text):
                endings[context] += 1
            else:
                pattern = text[offset:offset + size + 1]
                patterns.setdefault(pattern, []).append(offset)
    rows = []
    for pattern, offsets in sorted(patterns.items(), key=lambda item: (len(item[0]), item[0])):
        context = pattern[:-1]
        count = len(offsets)
        opportunities = context_counts[context]
        rows.append({
            "text": pattern, "size": len(pattern), "context": context, "next": pattern[-1],
            "offsets": offsets, "source_start_offsets": [offset for offset in offsets if offset == 0],
            "count": count, "opportunities": opportunities, "source_end_opportunities": endings[context],
            "likelihood": count / opportunities,
            "wilson_lower_95": lower_bound(count, opportunities),
            "observed_always": count == opportunities,
        })
    characters = Counter(text)
    pairs = []
    for row in rows:
        if row["size"] != 2:
            continue
        left, right = row["text"]
        count = row["count"]
        pairs.append({
            "text": row["text"], "offsets": row["offsets"], "count": count,
            "left_occurrences": characters[left], "right_occurrences": characters[right],
            "forward_likelihood": count / characters[left],
            "backward_likelihood": count / characters[right],
            "always_followed_in_source": count == characters[left],
            "always_preceded_in_source": count == characters[right],
            "mutually_always_in_source": count == characters[left] == characters[right],
        })
    return {
        "version": VERSION, "source": "current exact lab prompt",
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "characters": len(text), "patterns": rows, "pairs": pairs,
        "policy": "Evidence does not grant Layer 0 admission or override Layer 1. Experimental generation policy is reported separately; no persistent training.",
        "boundary_policy": "Literal spaces/case are preserved. Offsets mark source start. Source ends count as non-character outcomes; no cross-source joins.",
        "uncertainty": "Observed always means only this source, even at support 1. Wilson bounds assume independent trials; overlapping/repeated text is not independent evidence.",
    }


class Attraction:
    """Prompt-local estimates; no persisted learned state or seed admission."""

    def __init__(self, source: str, *, max_characters: int = 2000):
        self.report = analyze(source, max_characters=max_characters)
        self.unigrams = Counter(source)
        self.contexts: dict[str, Counter[str | None]] = {}
        for size in (1, 2, 3):
            for offset in range(len(source) - size + 1):
                context = source[offset:offset + size]
                following = source[offset + size] if offset + size < len(source) else None
                self.contexts.setdefault(context, Counter())[following] += 1

    def distribution(self, prefix: str) -> dict[str | None, float]:
        total = sum(self.unigrams.values())
        probabilities: dict[str | None, float] = {
            char: count / total for char, count in self.unigrams.items()
        }
        for size in (1, 2, 3):
            if len(prefix) < size:
                continue
            counts = self.contexts.get(prefix[-size:])
            if not counts:
                continue
            support = sum(counts.values())
            weight = support / (support + 2)
            probabilities = {
                char: (1 - weight) * probabilities.get(char, 0) + weight * counts[char] / support
                for char in probabilities.keys() | counts.keys()
            }
        return probabilities

    def metrics(self, text: str) -> dict:
        losses = []
        supported = 0
        for offset, char in enumerate(text):
            prefix = text[:offset]
            probabilities = self.distribution(prefix)
            losses.append(-math.log(max(probabilities.get(char, 0), 1e-12)))
            supported += any(
                self.contexts.get(prefix[-size:], {}).get(char, 0) > 0
                for size in (1, 2, 3) if len(prefix) >= size
            )
        return {
            "prompt_attraction_nll": sum(losses) / len(losses) if losses else None,
            "supported_next_characters": supported,
            "scored_characters": len(text),
            "layer_one_violation": exclusion_reason(boundary_text(text, True)),
            "interpretation": "Descriptive fit to current prompt, not held-out accuracy or semantic quality; empty output has no score.",
        }

    def generate(self, budget: int, seed: int, old_transitions: dict | None = None) -> dict:
        rng = random.Random(seed)
        text = ""
        trace = []
        reason = "Character budget reached"
        for _ in range(budget):
            probabilities = self.distribution(text)
            if old_transitions is None:
                weights = probabilities
            else:
                from original_generator import END, ORDER, START

                context = (START * ORDER + text)[-ORDER:]
                counts = old_transitions.get(context)
                if not counts:
                    reason = "Existing model has no further transition; no unrelated fallback"
                    break
                # Positive floor retains old candidates unseen in this prompt.
                weights = {
                    None if char == END else char: count * (0.05 + probabilities.get(None if char == END else char, 0))
                    for char, count in counts.items()
                }
            candidates = []
            rejected = []
            for char, weight in sorted(weights.items(), key=lambda item: (item[0] is not None, item[0] or "")):
                violation = None if char is None else exclusion_reason(boundary_text(text + char, True))
                if violation:
                    rejected.append({"next": char, "reason": violation})
                elif weight > 0:
                    candidates.append((char, weight))
            if not candidates:
                reason = "All available next characters excluded by Layer 1"
                trace.append({"offset": len(text), "rejected": rejected, "selected": None})
                break
            char = rng.choices([item[0] for item in candidates], weights=[item[1] for item in candidates])[0]
            trace.append({
                "offset": len(text), "context": text[-3:], "selected": char,
                "rejected": rejected,
                "candidates": [{"next": value, "weight": weight} for value, weight in candidates],
            })
            if char is None:
                reason = "Observed source/model end selected"
                break
            text += char
        return {
            "text": text, "reason": reason, "trace": trace,
            "context_mode": "prompt_attraction" if old_transitions is None else "old_transitions_times_attraction",
            "policy": "Experimental outside Layer 0 admission; known Layer 1 exclusions enforced. No persistent training.",
        }
