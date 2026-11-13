"""Reversible candidate extraction; no changes to foundational admissions."""

from collections import Counter
import math
import random

from neural_index_pipeline import PlacementIndex
from neural_skeleton_runtime import UnitCodec, word_boundaries
from neural_skeletons import WORD
from neural_experiment import HIDDEN
from seed_life import LAYER_ZERO_SEEDS, TERMINUS


REFERENCE = "\x00layer:"


def discriminating_forms(training: str, possibilities: set[str]) -> dict:
    words = {m.group() for m in WORD.finditer(training)
             if word_boundaries(training, m.start(), m.end())}
    universe = {word.lower().replace("’", "'") for word in possibilities | words}
    candidates = {}
    for word in words:
        normalized = word.lower().replace("’", "'")
        for size in range(1, len(normalized)):
            for offset in range(len(normalized) - size + 1):
                candidates.setdefault(normalized[offset:offset + size], set()).add(word)
    by_length = {}
    for fragment in candidates:
        by_length.setdefault(len(fragment), set()).add(fragment)
    owners = {}
    ambiguous = set()
    for word in universe:
        for size, fragments in by_length.items():
            for offset in range(len(word) - size + 1):
                fragment = word[offset:offset + size]
                if fragment not in fragments or fragment in ambiguous:
                    continue
                previous = owners.get(fragment)
                if previous is not None and previous != word:
                    ambiguous.add(fragment)
                else:
                    owners[fragment] = word
    result = {}
    for fragment, targets in candidates.items():
        if fragment in ambiguous:
            continue
        for word in targets:
            normalized = word.lower().replace("’", "'")
            if owners.get(fragment) != normalized:
                continue
            previous = result.get(word)
            if previous is None or (len(fragment), fragment) < (len(previous["key"]), previous["key"]):
                result[word] = {"key": fragment, "normalized_word": normalized,
                                "matching_entries": 1, "key_length": len(fragment)}
    return dict(sorted(result.items()))


def selective_forms(training: str, words: set[str], *, savings_fraction: float | None = None) -> dict:
    if savings_fraction is not None and savings_fraction not in (0.25, 0.5):
        raise ValueError("Savings fraction must be 0.25 or 0.5")
    contexts = {}
    occurrences = {}
    for match in WORD.finditer(training):
        if not word_boundaries(training, match.start(), match.end()):
            continue
        prefix = training[max(0, match.start() - 4):match.start()]
        contexts.setdefault(prefix, Counter())[match.group()] += 1
        if match.group() in words:
            occurrences.setdefault(match.group(), []).append(prefix)
    evidence = {}
    for word in sorted(words):
        observed = occurrences.get(word, [])
        reliable = sum(contexts[prefix][word] >= 2 and
                       contexts[prefix][word] / sum(contexts[prefix].values()) >= 0.9
                       for prefix in observed)
        fraction = reliable / len(observed) if observed else 0
        saving = len(observed) * (len(word) - 1)
        evidence[word] = {
            "occurrences": len(observed), "reliable_occurrences": reliable,
            "reliable_fraction": fraction, "potential_saved_characters": saving,
            "selected": len(observed) >= 2 and saving >= 6 and fraction >= 0.8,
            "contexts": {prefix: dict(contexts[prefix]) for prefix in sorted(set(observed))},
        }
    if savings_fraction is not None:
        ranked = sorted((word for word in words if evidence[word]["occurrences"] >= 2),
                        key=lambda word: (-evidence[word]["potential_saved_characters"], word))
        chosen = set(ranked[:math.ceil(len(ranked) * savings_fraction)])
        for word, row in evidence.items():
            row["selected"] = word in chosen
    return evidence


class CandidateCodec(UnitCodec):
    def __init__(self, units: dict, provenance: dict):
        self.units = units
        self.provenance = provenance

    def encode(self, text: str, *, open_tail: bool = False) -> list[str]:
        if "\x00" in text:
            raise ValueError("Input contains the reserved reference namespace")
        pieces = list(text)
        for layer in (0, 1, 2):
            by_first = {}
            for reference, entry in self.units.items():
                if entry["layer"] == layer:
                    by_first.setdefault(entry["text"][0], []).append((reference, entry))
            output = []
            offset = 0
            cursor = 0
            while cursor < len(pieces):
                piece = pieces[cursor]
                if piece in self.units:
                    output.append(piece)
                    offset += len(self.units[piece]["text"])
                    cursor += 1
                    continue
                matches = []
                for reference, entry in by_first.get(piece, []):
                    literal = entry["text"]
                    if pieces[cursor:cursor + len(literal)] != list(literal):
                        continue
                    if entry["source_start"] and offset != 0:
                        continue
                    if entry["bounded"] and not word_boundaries(text, offset, offset + len(literal)):
                        continue
                    if open_tail and offset + len(literal) == len(text):
                        continue
                    matches.append(reference)
                if matches:
                    reference = min(matches, key=lambda ref: (-len(self.units[ref]["text"]), ref))
                    literal = self.units[reference]["text"]
                    output.append(reference)
                    offset += len(literal)
                    cursor += len(literal)
                else:
                    output.append(piece)
                    offset += 1
                    cursor += 1
            pieces = output
        if "".join(self.units[p]["text"] if p in self.units else p for p in pieces) != text:
            raise AssertionError("Candidate extraction changed source text")
        return pieces


class CandidateIndex(PlacementIndex):
    def set_features(self, features: dict) -> None:
        if set(features) != set(self.codec.units):
            raise ValueError("Protected features must cover every indexed reference exactly")
        checked = {}
        for reference, vector in features.items():
            if len(vector) != HIDDEN or any(not math.isfinite(value) for value in vector):
                raise ValueError("Protected reference vector has invalid shape or values")
            checked[reference] = tuple(vector)
        self.features = checked

    def reference_conditions(self, pieces: list[str]) -> list[tuple[float, ...] | None]:
        features = getattr(self, "features", {})
        current = None
        conditions = []
        for piece in pieces:
            if piece in self.codec.units:
                current = features.get(piece)
            else:
                conditions.append(current)
        return conditions

    def snapshot(self) -> dict:
        state = super().snapshot()
        if hasattr(self, "features"):
            state["features"] = {ref: list(vector) for ref, vector in self.features.items()}
        return state

    @classmethod
    def build(cls, bank: dict, training: str, depth: int, *, selective: bool = False,
              l2_only: bool = False, savings_fraction: float | None = None,
              possibilities: set[str] | None = None) -> "CandidateIndex":
        if depth not in (0, 1, 2):
            raise ValueError("Candidate depth must be 0, 1 or 2")
        if selective and depth != 1:
            raise ValueError("Selective L1 requires depth 1")
        if savings_fraction is not None and (depth != 1 or selective):
            raise ValueError("Savings selection requires depth 1 without the context-purity gate")
        if l2_only and (depth != 2 or selective):
            raise ValueError("Standalone L2 requires depth 2 without selective L1")
        if possibilities is not None and (not possibilities or selective or savings_fraction is not None or l2_only):
            raise ValueError("Unique-fragment L1 requires nonempty possibilities without legacy selection modes")
        units = {}
        selection = None

        def add(text, layer, kind, *, start=False, bounded=False, evidence=None):
            reference = f"{REFERENCE}{layer}:{len(units)}"
            units[reference] = {"text": text, "layer": layer, "kind": kind,
                                "source_start": start, "bounded": bounded, "evidence": evidence}

        for seed in (() if l2_only else LAYER_ZERO_SEEDS):
            add(seed.removeprefix(TERMINUS), 0, "foundation",
                start=seed.startswith(TERMINUS), evidence=seed)
        if depth >= 1 and not l2_only:
            selected = (discriminating_forms(training, possibilities)
                        if possibilities is not None else UnitCodec.from_bank(bank, training).units)
            if selective or savings_fraction is not None:
                selection = selective_forms(training, set(selected), savings_fraction=savings_fraction)
            for word, evidence in selected.items():
                if selection is not None and not selection[word]["selected"]:
                    continue
                add(word, 1, "unique_fragment" if possibilities is not None else "whole_form",
                    bounded=True, evidence=evidence)
        if depth >= 2:
            before = CandidateCodec(units.copy(), {}).encode(training)
            visible = []
            segment = ""
            for piece in before:
                if piece in units:
                    visible.append(segment)
                    segment = ""
                else:
                    segment += piece
            visible.append(segment)
            counts = Counter(block for segment in visible for size in (3, 4)
                             for offset in range(len(segment) - size + 1)
                             if (block := segment[offset:offset + size]).isalpha())
            chosen = sorted((block for block, count in counts.items() if count >= 3),
                            key=lambda block: (-counts[block] * (len(block) - 1), block))[:128]
            for block in chosen:
                add(block, 2, "repeated_block", evidence={"support": counts[block]})
            evidence = {}
            for kind, table in sorted(bank["patterns"].items()):
                if kind not in ("hidden1", "hidden2"):
                    continue
                for pattern, word in sorted(table.items()):
                    evidence.setdefault(word, {"pattern": pattern, "kind": kind})
            remaining_words = Counter(m.group() for segment in visible for m in WORD.finditer(segment)
                                      if word_boundaries(segment, m.start(), m.end()))
            hidden = sorted(remaining_words.keys() & evidence.keys(),
                            key=lambda word: (-remaining_words[word] * (len(word) - 1), word))[:128]
            for word in hidden:
                add(word, 2, "hidden_backed_complete", bounded=True, evidence=evidence[word])
        return cls(CandidateCodec(units, {
            "depth": depth,
            "l2_only": l2_only,
            "selective_l1": selection,
            "savings_fraction": savings_fraction,
            "unique_fragment_l1": possibilities is not None,
            "selection_policy": (
                f"Training only: select ceil({savings_fraction} * number of forms with >=2 occurrences), "
                "ranked by occurrences * (length - 1), lexical tie break. No context-purity requirement."
            ) if savings_fraction is not None else (
                "Training only: >=2 occurrences, >=6 potential saved characters; "
                ">=80% of occurrences have an exact preceding up-to-four-character context "
                "observed with this word >=2 times and >=90% next-word purity."
            ) if selective else None,
            "policy": "Experimental candidates, not admissions. Layer 0 constants unchanged. "
                      "Layers 1/2 require exact complete text. Literal * is never an unknown marker. "
                      "Repeated alphabetic blocks: lengths 3/4, support >=3, cap 128. "
                      "Hidden-backed complete forms: cap 128, frozen bank evidence only. "
                      "No extraction crosses an earlier reference; longest exact match wins within a stage.",
        }))

    def extract(self, text: str, *, open_tail: bool = False) -> dict:
        pieces = self.codec.encode(text, open_tail=open_tail)
        remainder = []
        occurrences = []
        offset = 0
        for position, piece in enumerate(pieces):
            entry = self.codec.units.get(piece)
            literal = entry["text"] if entry else piece
            if entry:
                occurrences.append({"reference": piece, "pattern": literal, "layer": entry["layer"],
                                    "kind": entry["kind"], "event": position,
                                    "start": offset, "end": offset + len(literal)})
            else:
                remainder.append(piece)
            offset += len(literal)
        return {"pieces": pieces, "remainder": "".join(remainder), "occurrences": occurrences}

    def learn(self, pieces: list[str]) -> int:
        history = []
        updates = 0
        for piece in pieces:
            action = piece if piece in self.codec.units else ""
            for size in range(min(self.context, len(history)) + 1):
                key = tuple(history[-size:]) if size else ()
                self.counts.setdefault(key, Counter())[action] += 1
                updates += 1
            history.append(piece)
        return updates

    def choices(self, history: list[str], remaining: int) -> Counter:
        for size in range(min(self.context, len(history)), -1, -1):
            observed = self.counts.get(tuple(history[-size:]) if size else ())
            if observed:
                options = Counter({ref: count for ref, count in observed.items() if not ref or (
                    len(self.codec.units[ref]["text"]) <= remaining and
                    (not self.codec.units[ref]["source_start"] or not history))})
                if options:
                    return options
        return Counter({"": 1})

    @classmethod
    def restore(cls, state: dict) -> "CandidateIndex":
        if state.get("atomic"):
            return AtomicCandidateIndex.restore(state)
        index = cls(CandidateCodec(state["codec"]["units"], state["codec"]["provenance"]), state["context"])
        index.counts = {tuple(row["history"]): Counter(row["actions"]) for row in state["counts"]}
        if "features" in state:
            index.set_features(state["features"])
        return index

    def sample(self, model, prompt, *, length=120, seed=23, greedy=False):
        if type(length) is not int or not 1 <= length <= 500:
            raise ValueError("Generation length must be an integer from 1 to 500 characters")
        encoded = self.extract(prompt, open_tail=True)
        history = encoded["pieces"]
        features = getattr(self, "features", {})
        current = None
        for piece in history:
            if piece in self.codec.units:
                current = features.get(piece)
        model.set_reference_condition(current)
        context = ([0] * model.context + [model.tokens.get(c, 1) for c in encoded["remainder"]])[-model.context:]
        rng = random.Random(seed)
        output = ""
        emissions = []
        calls = lookups = 0
        conditioned_calls = 0
        injections = sum(piece in self.codec.units for piece in history) if features else 0
        while len(output) < length:
            options = self.choices(history, length - len(output))
            lookups += 1
            action = (min(options, key=lambda ref: (-options[ref], ref)) if greedy else
                      rng.choices(list(options), weights=list(options.values()))[0])
            if action:
                entry = self.codec.units[action]
                emissions.append({"reference": action, "pattern": entry["text"],
                                  "layer": entry["layer"], "offset": len(output)})
                output += entry["text"]
                history.append(action)
                current = features.get(action)
                model.set_reference_condition(current)
                if features:
                    injections += 1
                continue
            probabilities = model.forward(context)[2]
            calls += 1
            conditioned_calls += current is not None
            if not all(math.isfinite(p) and p >= 0 for p in probabilities[2:]) or not any(probabilities[2:]):
                raise ArithmeticError("Candidate remainder generation has no finite character mass")
            token = (max(range(2, model.size), key=lambda k: (probabilities[k], -k)) if greedy else
                     rng.choices(range(2, model.size), weights=probabilities[2:])[0])
            char = model.characters[token - 2]
            output += char
            history.append(char)
            context = context[1:] + [token]
        result = {"text": output, "neural_calls": calls, "index_lookups": lookups,
                  "indexed_emissions": len(emissions), "emissions": emissions}
        if features:
            result.update(conditioned_neural_calls=conditioned_calls, reference_injections=injections)
        return result


class AtomicCandidateIndex(CandidateIndex):
    def snapshot(self) -> dict:
        return {**super().snapshot(), "atomic": True}

    @classmethod
    def restore(cls, state: dict) -> "AtomicCandidateIndex":
        if not state.get("atomic") or state["counts"] or "features" in state:
            raise ValueError("Atomic codec requires its mode flag and no placement counts or injected features")
        return cls(CandidateCodec(state["codec"]["units"], state["codec"]["provenance"]), state["context"])

    def examples(self, model, text: str) -> list:
        pieces = self.codec.encode(text)
        if any(piece not in model.tokens for piece in pieces):
            raise ValueError("Atomic training stream contains a token outside the model vocabulary")
        sequence = [0] * model.context + [model.tokens[piece] for piece in pieces]
        return [(sequence[i - model.context:i], sequence[i])
                for i in range(model.context, len(sequence))]

    def sample(self, model, prompt, *, length=120, seed=23, greedy=False):
        if type(length) is not int or not 1 <= length <= 500:
            raise ValueError("Generation length must be an integer from 1 to 500 characters")
        model.set_reference_condition(None)
        pieces = self.codec.encode(prompt, open_tail=True)
        context = ([0] * model.context + [model.tokens.get(piece, 1) for piece in pieces])[-model.context:]
        rng = random.Random(seed)
        output = ""
        emissions = []
        calls = 0
        while len(output) < length:
            probabilities = model.forward(context)[2]
            calls += 1
            for reference, entry in self.codec.units.items():
                if (len(entry["text"]) > length - len(output)
                        or (entry["source_start"] and (prompt or output))):
                    probabilities[model.tokens[reference]] = 0.0
            mass = sum(probabilities[2:])
            if not math.isfinite(mass) or mass <= 0:
                raise ArithmeticError("Atomic generation has no finite allowed character/reference mass")
            token = (max(range(2, model.size), key=lambda k: (probabilities[k], -k)) if greedy else
                     rng.choices(range(2, model.size), weights=probabilities[2:])[0])
            piece = model.characters[token - 2]
            if piece in self.codec.units:
                entry = self.codec.units[piece]
                emissions.append({"reference": piece, "pattern": entry["text"],
                                  "layer": entry["layer"], "offset": len(output)})
                output += entry["text"]
            else:
                output += piece
            context = context[1:] + [token]
        return {"text": output, "neural_calls": calls, "index_lookups": 0,
                "indexed_emissions": len(emissions), "emissions": emissions}
