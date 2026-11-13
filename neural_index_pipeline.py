"""Development test: separate placement index and remainder-only character training."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
from uuid import uuid4

from neural_experiment import NeuralModel, RATE
from neural_patterns import STAGED_DIRECTORY, load_chapters
from neural_skeleton_runtime import UnitCodec, load_bank, word_character
from neural_skeletons import WORD
from neural_unit_benchmark import DIRECTORY, edit_distance, probes, text_hash
from original_generator import atomic_json


class PlacementIndex:
    """Observed boundary actions with longest-context backoff, not a neural output list."""

    def __init__(self, codec: UnitCodec, context: int = 4):
        self.codec = codec
        self.context = context
        self.counts: dict[tuple[str, ...], Counter] = {}

    def extract(self, text: str, *, open_tail: bool = False) -> dict:
        pieces = self.codec.encode(text, open_tail=open_tail)
        remainder = []
        occurrences = []
        offset = 0
        for position, piece in enumerate(pieces):
            if piece in self.codec.units:
                occurrences.append({"pattern": piece, "event": position,
                                    "start": offset, "end": offset + len(piece)})
            else:
                remainder.append(piece)
            offset += len(piece)
        return {"pieces": pieces, "remainder": "".join(remainder), "occurrences": occurrences}

    def learn(self, pieces: list[str]) -> int:
        history = []
        updates = 0
        for piece in pieces:
            if not history or not word_character(history[-1][-1:]):
                action = piece if piece in self.codec.units else ""
                for size in range(min(self.context, len(history)) + 1):
                    key = tuple(history[-size:]) if size else ()
                    self.counts.setdefault(key, Counter())[action] += 1
                    updates += 1
            history.append(piece)
        return updates

    def choices(self, history: list[str], remaining: int) -> Counter:
        for size in range(min(self.context, len(history)), -1, -1):
            key = tuple(history[-size:]) if size else ()
            observed = self.counts.get(key)
            if observed:
                allowed = Counter({word: count for word, count in observed.items()
                                   if not word or len(word) <= remaining})
                if allowed:
                    return allowed
        # Defined unseen-context policy: defer to the remainder character model.
        return Counter({"": 1})

    def snapshot(self) -> dict:
        return {"codec": self.codec.snapshot(), "context": self.context,
                "counts": [{"history": list(key), "actions": dict(value)}
                           for key, value in sorted(self.counts.items())]}

    @classmethod
    def restore(cls, state: dict) -> "PlacementIndex":
        index = cls(UnitCodec(state["codec"]["units"], state["codec"]["provenance"]), state["context"])
        index.counts = {tuple(row["history"]): Counter(row["actions"]) for row in state["counts"]}
        return index

    def sample(self, model: NeuralModel, prompt: str, *, length: int = 120,
               seed: int = 23, greedy: bool = False) -> dict:
        if type(length) is not int or not 1 <= length <= 500:
            raise ValueError("Generation length must be an integer from 1 to 500 characters")
        encoded = self.extract(prompt, open_tail=True)
        history = encoded["pieces"]
        context = ([0] * model.context + [model.tokens.get(c, 1)
                   for c in encoded["remainder"]])[-model.context:]
        rng = random.Random(seed)
        output = ""
        emissions = []
        calls = lookups = 0
        while len(output) < length:
            at_boundary = not history or not word_character(history[-1][-1:])
            if at_boundary:
                options = self.choices(history, length - len(output))
                lookups += 1
                action = (min(options, key=lambda word: (-options[word], word)) if greedy else
                          rng.choices(list(options), weights=list(options.values()))[0])
                if action:
                    emissions.append({"pattern": action, "offset": len(output)})
                    output += action
                    history.append(action)
                    continue
            probabilities = model.forward(context)[2]
            calls += 1
            if history and history[-1] in self.codec.units:
                for char, token in model.tokens.items():
                    if word_character(char):
                        probabilities[token] = 0
            weights = probabilities[2:]
            if not all(math.isfinite(value) and value >= 0 for value in weights) or not any(weights):
                raise ArithmeticError("Remainder generation has no finite allowed character")
            token = (max(range(2, model.size), key=lambda k: (probabilities[k], -k)) if greedy else
                     rng.choices(range(2, model.size), weights=weights)[0])
            char = model.characters[token - 2]
            output += char
            history.append(char)
            context = context[1:] + [token]
        return {"text": output, "neural_calls": calls, "index_lookups": lookups,
                "indexed_emissions": len(emissions), "emissions": emissions}


def evaluate(model: NeuralModel, index: PlacementIndex | None, items: list[dict], *,
             selection_metrics: bool = False, reference_codec: UnitCodec | None = None) -> dict:
    rows = []
    for item in items:
        if index:
            generated = index.sample(model, item["prefix"], length=32, greedy=True)
            text = generated["text"]
            emissions = generated["indexed_emissions"]
        else:
            context = ([0] * model.context + [model.tokens.get(c, 1)
                       for c in item["prefix"]])[-model.context:]
            text = ""
            for _ in range(32):
                probabilities = model.forward(context)[2]
                token = max(range(2, model.size), key=lambda k: (probabilities[k], -k))
                text += model.characters[token - 2]
                context = context[1:] + [token]
            emissions = 0
        match = WORD.match(text)
        prediction = match.group() if match and match.end() < len(text) else ""
        rows.append({"offset": item["offset"], "target": item["target"],
                     "prediction": prediction, "indexed_emissions": emissions})
        if selection_metrics:
            first = next((e for e in generated["emissions"] if e["offset"] == 0 and e["layer"] == 1), None) if index else None
            rows[-1]["first_l1_emission"] = first["pattern"] if first else None
            rows[-1]["correct_l1_selection"] = bool(first and first["pattern"] == item["target"]
                                                  and prediction == item["target"])
    result = {
        "exact_next_word_accuracy": statistics.fmean(row["prediction"] == row["target"] for row in rows),
        "normalized_edit_distance": statistics.fmean(
            edit_distance(row["prediction"], row["target"]) /
            max(len(row["prediction"]), len(row["target"]), 1) for row in rows),
        "rows": rows,
    }
    if selection_metrics:
        selected = [row for row in rows if row["first_l1_emission"] is not None]
        codec = reference_codec if reference_codec is not None else index.codec if index else None
        eligible = {entry["text"] for entry in codec.units.values()
                    if entry["layer"] == 1} if codec else set()
        correct = sum(row["correct_l1_selection"] for row in rows)
        following = [row for row, item in zip(rows, items) if codec and any(
            piece in codec.units for piece in codec.encode(item["prefix"], open_tail=True)[-4:])]
        result["selection_metrics"] = {
            "l1_first_emissions": len(selected), "correct_l1_first_emissions": correct,
            "eligible_l1_target_probes": sum(item["target"] in eligible for item in items),
            "l1_selection_precision": correct / len(selected) if selected else None,
            "recent_reference_probes": len(following),
            "recent_reference_word_accuracy": statistics.fmean(row["prediction"] == row["target"]
                                                               for row in following) if following else None,
        }
    return result


def pattern_examples(model: NeuralModel, patterns: list[str]) -> list:
    if not patterns or any(not pattern for pattern in patterns):
        raise ValueError("Pretraining requires nonempty literal patterns")
    if any(char not in model.tokens for pattern in patterns for char in pattern):
        raise ValueError("Pretraining alphabet does not contain every pattern character")
    return [example for pattern in patterns for example in model.examples(pattern)]


def pattern_recall(model: NeuralModel, patterns: list[str]) -> dict:
    model.set_reference_condition(None)
    examples = pattern_examples(model, patterns)
    rows = []
    for pattern in patterns:
        expected = model.examples(pattern)
        predictions = []
        for context, _ in expected:
            probabilities = model.forward(context)[2]
            predictions.append(max(range(2, model.size), key=probabilities.__getitem__))
        rows.append({"pattern": pattern,
                     "correct_characters": sum(predicted == target
                                               for predicted, (_, target) in zip(predictions, expected)),
                     "characters": len(expected)})
    return {"loss": model.loss(examples),
            "teacher_forced_character_accuracy": sum(row["correct_characters"] for row in rows) / len(examples),
            "rows": rows}

def protected_features(model: NeuralModel, units: dict) -> dict:
    model.set_reference_condition(None)
    features = {}
    for reference, entry in units.items():
        context = [0] * model.context
        states = []
        for char in entry["text"]:
            if char not in model.tokens:
                raise ValueError("Protected payload contains a character outside the pretrained alphabet")
            context = context[1:] + [model.tokens[char]]
            states.append(model.forward(context)[1])
        features[reference] = [statistics.fmean(state[j] for state in states)
                               for j in range(len(states[0]))]
    return features


def reinforced_schedule(model: NeuralModel, training: str, extracted: dict, *, remainder: bool) -> list:
    ordinary = model.examples(extracted["remainder"] if remainder else training)
    schedule = []
    cursor = 0
    offset = 0
    occurrences = {row["start"]: row for row in extracted["occurrences"]}
    while offset < len(training):
        occurrence = occurrences.get(offset)
        if occurrence:
            literal = training[offset:occurrence["end"]]
            schedule.extend((context, target, True) for context, target in model.examples(literal))
            if not remainder:
                schedule.extend((context, target, False) for context, target in
                                ordinary[cursor:cursor + len(literal)])
                cursor += len(literal)
            offset += len(literal)
        else:
            context, target = ordinary[cursor]
            schedule.append((context, target, False))
            cursor += 1
            offset += 1
    if cursor != len(ordinary):
        raise AssertionError("Reinforcement schedule lost ordinary targets")
    return schedule


def whole_acts(root: Path) -> tuple[str, dict]:
    from neural_rounds import load_acts
    from neural_proof import load_fresh
    early = load_acts(root, True)
    fresh, _, fresh_hash = load_fresh(root)
    later, later_hash = load_chapters(root, STAGED_DIRECTORY)
    last, last_hash = load_chapters(root, ".data/acts-patterns-locked")
    if set(later) != set(range(6, 28)) or set(last) != {28}:
        raise ValueError("Whole Acts requires every chapter 1-28 exactly once")
    text = "\n".join([early["training"], early["validation"], early["final_test"], fresh,
                      *(later[c] for c in range(6, 28)), last[28]])
    return text, {"chapters": list(range(1, 29)),
                  "manifest_hashes": [early["manifest_sha256"], fresh_hash, later_hash, last_hash],
                  "evaluation_scope": "In-sample learning checks on training text, NOT held-out evaluation. "
                                      "User explicitly authorized whole Acts including former test chapters. "
                                      "Earlier locked claims remain historical; these chapters are no longer "
                                      "unseen for this experiment. Generated text is not scripture."}


def l1_target_probes(text: str, extracted: dict, minimum: int = 20) -> list[dict]:
    targets = [row for row in extracted["occurrences"] if row["layer"] == 1 and row["start"] >= 32]
    if len(targets) < minimum:
        raise ValueError(f"Need at least {minimum} distinct L1 target occurrences; found {len(targets)}")
    chosen = [targets[i * len(targets) // minimum] for i in range(minimum)]
    return [{"prefix": text[:row["start"]], "target": row["pattern"], "offset": row["start"]}
            for row in chosen]


def run(root: Path, *, acknowledge_review: bool, large: bool = False,
        corpus: Path | None = None, candidate_layers: bool = False,
        selective_l1: bool = False, l2_only: bool = False, savings_l1: bool = False,
        unique_l1: bool = False, pretrain_l01: bool = False, inject_l01: bool = False,
        atomic_l01: bool = False, matched_reinforcement: bool = False, all_acts: bool = False) -> dict:
    if acknowledge_review is not True:
        raise ValueError("Explicit approval to use the selected texts is required")
    if corpus is not None and large:
        raise ValueError("A custom corpus cannot be combined with the Acts large profile")
    if selective_l1 and not candidate_layers:
        raise ValueError("Selective L1 comparison requires candidate layers")
    if l2_only and (not candidate_layers or selective_l1):
        raise ValueError("Standalone L2 requires candidate layers without selective L1")
    if savings_l1 and (not candidate_layers or selective_l1 or l2_only):
        raise ValueError("Savings L1 requires candidate layers without selective L1 or standalone L2")
    if unique_l1 and (not candidate_layers or selective_l1 or l2_only or savings_l1):
        raise ValueError("Unique-fragment L1 requires candidate layers without legacy selection modes")
    if pretrain_l01 and not unique_l1:
        raise ValueError("L01 pretraining requires corrected unique-fragment L1")
    if inject_l01 and not pretrain_l01:
        raise ValueError("Protected injection requires L01 pretraining")
    if atomic_l01 and (not unique_l1 or pretrain_l01 or inject_l01):
        raise ValueError("Atomic comparison requires unique L1 without pretraining or injection")
    if matched_reinforcement and (not pretrain_l01 or inject_l01 or atomic_l01):
        raise ValueError("Matched reinforcement requires unique L01 pretraining without injection or atomic mode")
    if candidate_layers and large:
        raise ValueError("Candidate-layer comparison is a small-run experiment")
    if all_acts and (not matched_reinforcement or corpus is not None or large):
        raise ValueError("Whole Acts requires matched reinforcement without custom corpus or large split")
    setup_cpu, setup_wall = time.process_time(), time.perf_counter()
    corpus_record = None
    whole_source = None
    if all_acts:
        train, whole_source = whole_acts(root)
        held = train
        manifest_hash = text_hash(json.dumps(whole_source, sort_keys=True))
    elif corpus is not None:
        raw = corpus.read_bytes()
        corpus_record = json.loads(raw)
        if not isinstance(corpus_record, dict) or any(
            not isinstance(corpus_record.get(key), str) or not corpus_record[key].strip()
            for key in ("title", "provenance", "rights_note", "limitations", "training", "evaluation")
        ):
            raise ValueError("Corpus requires nonempty title, provenance, rights_note, limitations, training and evaluation strings")
        train, held = corpus_record["training"], corpus_record["evaluation"]
        if min(len(train), len(held)) < 64 or max(len(train), len(held)) > 50_000 or train == held:
            raise ValueError("Corpus needs distinct training/evaluation passages of 64-50000 characters each")
        manifest_hash = hashlib.sha256(raw).hexdigest()
    else:
        texts, manifest_hash = load_chapters(root, STAGED_DIRECTORY)
        if large:
            train = "\n".join(texts[chapter] for chapter in range(6, 13))
            held = "\n".join(texts[chapter] for chapter in (13, 14))
        else:
            train = texts[6][:3000].rsplit(" ", 1)[0] + " "
            held = texts[7][:1500].rsplit(" ", 1)[0] + " "
    bank = load_bank(root)
    possibilities = None
    inventory = None
    if unique_l1:
        inventory_path = root / ".data/possible-words-expanded-scowl-2020.12.07"
        inventory = json.loads((inventory_path / "manifest.json").read_text())
        raw_words = (inventory_path / "words.txt").read_bytes()
        if hashlib.sha256(raw_words).hexdigest() != inventory["words_sha256"]:
            raise ValueError("Expanded possibility inventory hash mismatch")
        possibilities = set(raw_words.decode("utf-8").splitlines())
        if len(possibilities) != inventory["entries"]:
            raise ValueError("Expanded possibility inventory count mismatch")
    index = PlacementIndex(UnitCodec.from_bank(bank, train))
    candidates = {}
    if candidate_layers:
        from neural_candidate_layers import CandidateIndex, AtomicCandidateIndex

        candidates = ({"layers_2": CandidateIndex.build(bank, train, 2, l2_only=True)} if l2_only else
                      {f"layers_0{'1' if depth >= 1 else ''}{'2' if depth >= 2 else ''}":
                       CandidateIndex.build(bank, train, depth, possibilities=possibilities)
                       for depth in ((0, 1) if selective_l1 or savings_l1 or unique_l1 else (0, 1, 2))})
        if selective_l1:
            candidates = {
                "layers_0": candidates["layers_0"],
                "layers_01_selective": CandidateIndex.build(bank, train, 1, selective=True),
                "layers_01": candidates["layers_01"],
            }
        if savings_l1:
            candidates = {
                "layers_0": candidates["layers_0"],
                "layers_01_quarter": CandidateIndex.build(bank, train, 1, savings_fraction=0.25),
                "layers_01_half": CandidateIndex.build(bank, train, 1, savings_fraction=0.5),
                "layers_01": candidates["layers_01"],
            }
        if pretrain_l01:
            candidates["layers_01_pretrained"] = CandidateIndex.restore(candidates["layers_01"].snapshot())
        if inject_l01:
            candidates["layers_01_injected"] = CandidateIndex.restore(candidates["layers_01"].snapshot())
        if atomic_l01:
            original = candidates["layers_01"]
            candidates = {"layers_01": original,
                          "layers_01_atomic": AtomicCandidateIndex(original.codec)}
        if matched_reinforcement:
            candidates = {"layers_01": candidates["layers_01"]}
    patterns = sorted({entry["text"] for entry in candidates["layers_01"].codec.units.values()}) if pretrain_l01 else []
    alphabet = sorted(set(train) | set("".join(patterns)))
    arms = ("baseline", *candidates) if candidate_layers else ("baseline", "separated")
    extractions = {arm: candidate.extract(train) for arm, candidate in candidates.items()}
    extracted = index.extract(train)
    if not extracted["occurrences"] or not extracted["remainder"]:
        raise ValueError("Comparison requires both indexed occurrences and remainder text")
    items = probes(held, 256 if large else 48)
    focused_items = []
    if all_acts:
        focused_items = l1_target_probes(train, extractions["layers_01"])
        items = probes(held, 256)
        items = sorted({item["offset"]: item for item in items + focused_items}.values(),
                       key=lambda item: item["offset"])
    run_id = uuid4().hex
    directory = root / DIRECTORY / run_id
    directory.mkdir(parents=True)
    report = {
        "method": "candidate_layer_comparison" if candidate_layers else "separate_index_remainder",
        "run_id": run_id, "status": "running",
        "profile": "whole_acts" if all_acts else "custom" if corpus_record else "large" if large else "small",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": {"seeds": [17, 117, 217, 317, 417] if large else [17, 117, 217],
                     "passes": [1, 2, 4], "rate": RATE,
                     "selector_context": 4, "probes": len(items)},
        "source": {"training_characters": len(train), "evaluation_characters": len(held),
                   "training": f"{corpus_record['title']} training" if corpus_record else
                               "Acts 6-12 complete" if large else "Acts 6 prefix",
                   "evaluation": f"{corpus_record['title']} evaluation" if corpus_record else
                                 "Acts 13-14 complete" if large else "Acts 7 prefix",
                   "training_sha256": text_hash(train), "evaluation_sha256": text_hash(held),
                   "manifest_sha256": manifest_hash, "bank": index.codec.provenance,
                   "approval": "User approved the selected corpus for this separated-index experiment.",
                   "corpus": {key: corpus_record[key] for key in
                              ("title", "provenance", "rights_note", "limitations")} if corpus_record else None},
        "extraction": {"indexed_forms": len(index.codec.units), "occurrences": len(extracted["occurrences"]),
                       "remainder_characters": len(extracted["remainder"]),
                       "removed_characters": len(train) - len(extracted["remainder"])},
        "policy": "Base sees only concatenated remainder characters: no indexed targets, embeddings or contexts. "
                  "Separate standard-library boundary count index learns reference-or-character actions using "
                  "up to four preceding stream events, including reference identities. Longest seen suffix "
                  "backs off to global boundary counts; unavailable actions defer to base. Generation preserves "
                  "order, emitting an entire indexed pattern or one remainder character. Following an indexed "
                  "form requires a delimiter. Extraction runs ahead; no concurrent throughput claim. "
                  "Reused development data, not locked evidence or proof of generalization.",
        "code_hashes": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
                        ("neural_index_pipeline.py", "neural_experiment.py", "neural_skeleton_runtime.py",
                         "neural_unit_benchmark.py")},
        "setup_cpu_seconds": time.process_time() - setup_cpu,
        "setup_wall_seconds": time.perf_counter() - setup_wall,
        "bank_build_included": False, "runs": [],
    }
    if all_acts:
        report["source"].update(training="Acts 1-28 complete", evaluation="Acts 1-28 IN-SAMPLE learning checks",
                                whole_acts=whole_source)
        report["opportunity_check"] = {
            "minimum": 20, "focused_probes": len(focused_items),
            "distinct_l1_words": len({item["target"] for item in focused_items}),
            "offsets": [item["offset"] for item in focused_items],
            "policy": "Twenty distinct recognized L1 occurrence offsets selected evenly BEFORE training; "
                      "merged with 256 evenly distributed general probes, deduplicated by offset. "
                      "Occurrence opportunities, not necessarily twenty different words. In-sample only.",
        }
    if candidate_layers:
        report["settings"]["arms"] = list(arms)
        report["settings"]["selective_l1"] = selective_l1
        report["settings"]["l2_only"] = l2_only
        report["settings"]["savings_l1"] = savings_l1
        report["settings"]["unique_l1"] = unique_l1
        report["settings"]["pretrain_l01"] = pretrain_l01
        report["settings"]["inject_l01"] = inject_l01
        report["settings"]["atomic_l01"] = atomic_l01
        report["settings"]["matched_reinforcement"] = matched_reinforcement
        if matched_reinforcement:
            report["reinforcement"] = {
                "policy": "Both arms restore the identical per-seed L0/L1-only pretraining checkpoint. "
                          "For every corrected L0/L1 occurrence in original training text, both receive "
                          "one extra isolated character pass over its full literal with reset context. "
                          "Refresh immediately before that occurrence's ordinary targets (or skipped span). "
                          "Ordinary passage targets proceed in SOURCE ORDER in BOTH arms to align refresh "
                          "timing, unlike historical shuffled experiments. Ordinary contexts remain full text "
                          "for baseline, concatenated remainder for indexed arm. No evaluation refresh or "
                          "end-of-pass extra replay, no injection/atomic tokens. This is explicit reinforcement, "
                          "not learn-once/no-retraining. All pretraining and replay costs included.",
                "occurrences_per_pass": len(extractions["layers_01"]["occurrences"]),
                "updates_per_pass": sum(row["end"] - row["start"]
                                        for row in extractions["layers_01"]["occurrences"]),
            }
        if atomic_l01:
            report["atomic"] = {
                "policy": "Corrected unique-fragment L0/L1 references become individual neural tokens, "
                          "targets and context positions in the original ordered stream. Fixed payload decoding; "
                          "no internal spelling targets, separate placement learner, pretraining or injection in "
                          "the atomic arm. Four positions mean events, not characters, for that arm. Same seed, "
                          "MLP dimensions, rate and source passes; larger output vocabulary/parameter count means "
                          "atomic and character initial weights are not identical. Retain all nine L0/L1 reference "
                          "identities, including overlapping/source-start variants. No end target or new boundary "
                          "guard; only source-start and remaining-length masks, as in the candidate control. "
                          "Literal exact spelling does not guarantee correct joins or usage. Evaluation selection "
                          "precision requires L1 emission at offset zero AND exact next-word match; absent probes "
                          "or emissions give null, not perfect scores. Recent-reference accuracy uses probes with "
                          "an observed reference in the preceding four encoded events. No evaluation tuning.",
            }
        if pretrain_l01:
            report["pretraining"] = {
                "patterns": patterns, "passes": 200, "rate": RATE,
                "alphabet": alphabet,
                "policy": "Fresh seeded character base, isolated literal payloads only, each with reset context. "
                          "No partial-key targets, evaluation text or previous model weights. Fixed 200 shuffled "
                          "passes, not evaluation-tuned. Save and restore that pretraining-only checkpoint before "
                          "remainder training. No pattern replay during remainder training. Neural retention is "
                          "measured, not guaranteed; shared weights remain trainable. Indexed emission stays exact. "
                          "All arms share the union training/pattern alphabet and original seeded initialization. "
                          "Total training CPU includes pretraining; its cost is also reported separately.",
            }
        if inject_l01:
            report["injection"] = {
                "policy": "Frozen pretraining checkpoint encodes each full literal separately. "
                          "Mean of hidden states after each payload character gives one protected 16-value vector "
                          "per reference. On recognition/emission the latest reference vector is added to active "
                          "hidden preactivation, persisting until the next reference; no future references used. "
                          "Zero conditioning before any reference; no new trainable parameters. Active base starts "
                          "from the same pretrained weights as the unprotected arm, then learns only remainder targets. "
                          "No spelling replay, gradients into the bank, or pretrained-weight merging. "
                          "Frozen features persist, but useful interpretation/quality is not guaranteed. "
                          "Save the full frozen model, reference vectors and hashes; charge pretraining once per arm, "
                          "feature construction and each conditioned forward. Frozen-model recall is distinct from active recall.",
            }
        if unique_l1:
            report["unique_l1"] = {
                "inventory": inventory,
                "forms": {entry["text"]: entry["evidence"]
                          for entry in candidates["layers_01"].codec.units.values() if entry["layer"] == 1},
                "policy": "Shortest nonempty contiguous substring shorter than the whole word; "
                          "lexical tie break. Exactly one distinct lowercase/apostrophe-normalized entry "
                          "in the expanded inventory union all training words. No boundary or length hints. "
                          "No whole-word fallback, no cap. Exact bounded source spelling is fully extracted; "
                          "the key identifies the payload, not a prediction or substring-only deletion. "
                          "Placement remains statistical; recognition does not guarantee correct insertion.",
            }
        if savings_l1:
            report["selections"] = {arm: candidate.codec.provenance for arm, candidate in candidates.items()
                                    if candidate.codec.provenance["savings_fraction"] is not None}
        if selective_l1:
            report["selection"] = candidates["layers_01_selective"].codec.provenance
        report["code_hashes"]["neural_candidate_layers.py"] = hashlib.sha256(
            (root / "neural_candidate_layers.py").read_bytes()).hexdigest()
        report["layer_extractions"] = {
            arm: {"remainder_characters": len(data["remainder"]),
                  "removed_characters": len(train) - len(data["remainder"]),
                  "occurrences": len(data["occurrences"]),
                  "by_kind": dict(Counter(row["kind"] for row in data["occurrences"])),
                  "by_layer": dict(Counter(row["layer"] for row in data["occurrences"]))}
            for arm, data in extractions.items()}
        report["policy"] = (
            "Experimental recognition cascade; base is separate. Layer 0 uses the unchanged seven exact seed "
            "configurations, including source-start prefixes as eligible observations, not whole-word claims. "
            "Layer 1 uses the existing training-present bank-backed complete forms (cap 128). Layer 2 uses "
            "repeated alphabetic 3/4-character blocks (support >=3, cap 128) and hidden-backed complete "
            "forms remaining after earlier layers (cap 128). Earlier references cannot be crossed. "
            "All reference spellings/offsets are preserved; literal missing characters are never inferred. "
            "Separate placement counts learn reference-or-base actions at EVERY stream event for all layered "
            "arms, with the same four-event backoff. Base trains only concatenated remainder. "
            "No old generation exclusions or extra boundary guard is enabled; statistical placement may create "
            "invalid joins. This is not a guard or concurrency test. Candidate admissions and Layer 0 constants "
            "are unchanged. Development only, no equal-quality claim."
        )
        if l2_only:
            report["policy"] = (
                "Standalone L2 + separate remainder base versus baseline. L0 and L1 are disabled. "
                "L2 discovers repeated alphabetic 3/4-character blocks (support >=3, cap 128) and "
                "training-present hidden-backed complete forms (cap 128) directly from training text. "
                "Longest exact match wins; hidden forms are not missing-letter guesses. "
                "Typed references and source offsets preserve text. Same four-event placement counts "
                "learn reference-or-base actions at every event; base learns only concatenated remainder. "
                "No old generation exclusions or delimiter guard. No permanent admissions or concurrency "
                "claim. Training includes both learners; setup/bank costs reported separately. Development only."
            )
        if unique_l1:
            report["policy"] = report["unique_l1"]["policy"] + (
                " L0 unchanged; L1 replaces the old bank-backed whole-word eligibility rule. "
                "No L2. Base trains only concatenated remainder; separate four-event placement learner. "
                "No old generation exclusions. Historical reports preserved; development evidence only."
            )
        if atomic_l01:
            report["policy"] += " " + report["atomic"]["policy"]
        if matched_reinforcement:
            report["pretraining"]["policy"] = report["reinforcement"]["policy"]
            report["policy"] += " " + report["reinforcement"]["policy"]
    atomic_json(directory / "protocol.json", report)
    atomic_json(directory / "inputs.json", {"training": train, "evaluation": held, "probes": items,
                                          "extracted": extracted, "index": index.snapshot(),
                                          "focused_probes": focused_items,
                                          "corpus_record": corpus_record,
                                          "layer_extractions": extractions,
                                          "candidate_indices": {arm: item.snapshot() for arm, item in candidates.items()}})
    latest = root / ".data/neural-experiments/latest-unit-benchmark.json"
    atomic_json(latest, report)
    for position, seed in enumerate(report["settings"]["seeds"]):
        order = arms[position % len(arms):] + arms[:position % len(arms)]
        if not candidate_layers:
            order = arms if position % 2 == 0 else tuple(reversed(arms))
        for arm in order:
            is_atomic = arm == "layers_01_atomic"
            model = NeuralModel(alphabet + list(candidates[arm].codec.units) if is_atomic else alphabet, seed=seed)
            active = (CandidateIndex.restore(candidates[arm].snapshot()) if arm in candidates else
                      PlacementIndex(index.codec) if arm == "separated" else None)
            arm_extracted = extractions[arm] if arm in extractions else extracted
            if active and not arm_extracted["remainder"]:
                raise ValueError(f"{arm} extracted all characters; remainder training is undefined")
            example_start = time.process_time()
            examples = (active.examples(model, train) if is_atomic else
                        model.examples(arm_extracted["remainder"] if active else train))
            entry = {"seed": seed, "arm": arm, "vocabulary": model.size,
                     "parameters": sum(map(len, model.weights.values())),
                     "initial_weights_sha256": text_hash(json.dumps(model.weights, sort_keys=True)),
                     "examples_sha256": text_hash(json.dumps(examples)),
                     "examples_setup_cpu_seconds": time.process_time() - example_start,
                     "targets_per_pass": len(examples), "checkpoints": []}
            pretraining_cpu = pretraining_wall = 0.0
            feature_cpu = feature_wall = 0.0
            if matched_reinforcement or arm in ("layers_01_pretrained", "layers_01_injected"):
                before = pattern_recall(model, patterns)
                pre_examples = pattern_examples(model, patterns)
                pretraining_file = directory / f"{seed}-pretraining-only.json"
                inherited = pretraining_file.exists()
                if not inherited:
                    start_cpu, start_wall = time.process_time(), time.perf_counter()
                    pre_rng = random.Random(seed + 3)
                    for _ in range(report["pretraining"]["passes"]):
                        sequence = list(range(len(pre_examples)))
                        pre_rng.shuffle(sequence)
                        while sequence:
                            context, target = pre_examples[sequence.pop()]
                            model.update(context, target, learning_rate=RATE)
                    atomic_json(pretraining_file, {
                        "characters": model.characters, "weights": model.weights,
                        "patterns": patterns, "passes": report["pretraining"]["passes"],
                        "updates": len(pre_examples) * report["pretraining"]["passes"],
                        "examples_sha256": text_hash(json.dumps(pre_examples)),
                        "cpu_seconds": time.process_time() - start_cpu,
                        "wall_seconds": time.perf_counter() - start_wall,
                    })
                saved = json.loads(pretraining_file.read_text())
                pretraining_cpu, pretraining_wall = saved["cpu_seconds"], saved["wall_seconds"]
                model = NeuralModel(saved["characters"], seed=seed)
                model.weights = saved["weights"]
                entry["pretraining"] = {
                    "checkpoint": pretraining_file.name,
                    "checkpoint_sha256": hashlib.sha256(pretraining_file.read_bytes()).hexdigest(),
                    "updates": saved["updates"], "cpu_seconds": pretraining_cpu,
                    "wall_seconds": pretraining_wall, "before": before,
                    "after": pattern_recall(model, patterns),
                    "timing_inherited": inherited,
                }
                if arm == "layers_01_injected":
                    start_cpu, start_wall = time.process_time(), time.perf_counter()
                    active.set_features(protected_features(model, active.codec.units))
                    feature_cpu, feature_wall = time.process_time() - start_cpu, time.perf_counter() - start_wall
                    entry["injection"] = {
                        "feature_cpu_seconds": feature_cpu, "feature_wall_seconds": feature_wall,
                        "features_sha256": text_hash(json.dumps(active.snapshot()["features"], sort_keys=True)),
                        "frozen_checkpoint_sha256": entry["pretraining"]["checkpoint_sha256"],
                    }
            if pretrain_l01:
                entry["starting_weights_sha256"] = text_hash(json.dumps(model.weights, sort_keys=True))
            conditions = None
            if arm == "layers_01_injected":
                start_cpu, start_wall = time.process_time(), time.perf_counter()
                conditions = active.reference_conditions(arm_extracted["pieces"])
                feature_cpu += time.process_time() - start_cpu
                feature_wall += time.perf_counter() - start_wall
                entry["injection"]["feature_cpu_seconds"] = feature_cpu
                entry["injection"]["feature_wall_seconds"] = feature_wall
                entry["injection"]["conditioned_targets_per_pass"] = sum(c is not None for c in conditions)
            if conditions is not None and len(conditions) != len(examples):
                raise AssertionError("Reference conditioning must align exactly with remainder targets")
            schedule = None
            if matched_reinforcement:
                start = time.process_time()
                schedule = reinforced_schedule(model, train, extractions["layers_01"], remainder=active is not None)
                entry["reinforcement"] = {
                    "schedule_sha256": text_hash(json.dumps(schedule)),
                    "setup_cpu_seconds": time.process_time() - start,
                    "updates_per_pass": sum(refresh for _, _, refresh in schedule),
                }
            neural_cpu = selector_cpu = wall = 0.0
            reinforcement_cpu = 0.0
            rng = random.Random(seed + 2)
            count_updates = 0
            for passes in range(1, 5):
                wall_start = time.perf_counter()
                if active and not is_atomic:
                    start = time.process_time()
                    count_updates += active.learn(arm_extracted["pieces"])
                    selector_cpu += time.process_time() - start
                start = time.process_time()
                sequence = list(range(len(examples))) if schedule is None else []
                if schedule is None:
                    rng.shuffle(sequence)
                while sequence:
                    example_id = sequence.pop()
                    context, target = examples[example_id]
                    if conditions is not None:
                        model.set_reference_condition(conditions[example_id])
                    model.update(context, target, learning_rate=RATE)
                neural_cpu += time.process_time() - start
                if schedule is not None:
                    for context, target, refresh in schedule:
                        start = time.process_time()
                        model.update(context, target, learning_rate=RATE)
                        elapsed = time.process_time() - start
                        if refresh:
                            reinforcement_cpu += elapsed
                        else:
                            neural_cpu += elapsed
                wall += time.perf_counter() - wall_start
                if passes not in report["settings"]["passes"]:
                    continue
                start = time.process_time()
                measured = evaluate(model, active, items, selection_metrics=atomic_l01,
                                    reference_codec=candidates["layers_01"].codec if atomic_l01 else None)
                if focused_items:
                    focused_offsets = {item["offset"] for item in focused_items}
                    rows = [row for row in measured["rows"] if row["offset"] in focused_offsets]
                    measured["l1_focused"] = {
                        "probes": len(rows),
                        "correct": sum(row["prediction"] == row["target"] for row in rows),
                        "exact_next_word_accuracy": statistics.fmean(row["prediction"] == row["target"] for row in rows),
                        "normalized_edit_distance": statistics.fmean(
                            edit_distance(row["prediction"], row["target"]) /
                            max(len(row["prediction"]), len(row["target"]), 1) for row in rows),
                    }
                row = {"passes": passes, "updates": passes * len(examples),
                       "neural_cpu_seconds": neural_cpu, "selector_cpu_seconds": selector_cpu,
                       "total_training_cpu_seconds": neural_cpu + selector_cpu + pretraining_cpu + feature_cpu + reinforcement_cpu,
                       "training_wall_seconds": wall + pretraining_wall + feature_wall, "selector_count_updates": count_updates,
                       "selector_contexts": len(active.counts) if active else 0,
                       **measured, "evaluation_cpu_seconds": time.process_time() - start}
                if pretrain_l01:
                    row["pretraining_cpu_seconds"] = pretraining_cpu
                    row["pretraining_updates"] = entry.get("pretraining", {}).get("updates", 0)
                    row["pattern_recall"] = pattern_recall(model, patterns)
                if matched_reinforcement:
                    row["reinforcement_cpu_seconds"] = reinforcement_cpu
                    row["reinforcement_updates"] = passes * entry["reinforcement"]["updates_per_pass"]
                    row["total_updates_including_pretraining"] = (row["updates"] + row["reinforcement_updates"]
                                                                 + row["pretraining_updates"])
                if arm == "layers_01_injected":
                    row["feature_cpu_seconds"] = feature_cpu
                    row["features_sha256"] = text_hash(json.dumps(active.snapshot()["features"], sort_keys=True))
                    if row["features_sha256"] != entry["injection"]["features_sha256"]:
                        raise AssertionError("Protected features changed during remainder training")
                entry["checkpoints"].append(row)
                atomic_json(directory / f"{seed}-{arm}-passes-{passes}.json", {
                    "characters": model.characters, "weights": model.weights,
                    "index": active.snapshot() if active else None, "evidence": row})
                print(f"{seed} {arm} passes-{passes}: cpu={row['total_training_cpu_seconds']:.3f}s "
                      f"accuracy={row['exact_next_word_accuracy']:.3f}", flush=True)
            entry["samples"] = []
            for prompt in ("The ", "and ", "They "):
                start = time.process_time()
                generated = active.sample(model, prompt) if active else {
                    "text": model.sample(prompt), "neural_calls": 120,
                    "index_lookups": 0, "indexed_emissions": 0, "emissions": []}
                entry["samples"].append({"prompt": prompt, **generated,
                                         "cpu_seconds": time.process_time() - start})
            report["runs"].append(entry)
            atomic_json(directory / "report.json", report)
            atomic_json(latest, report)
    report["summary"] = {}
    for passes in report["settings"]["passes"]:
        report["summary"][f"passes-{passes}"] = {
            arm: {key: statistics.fmean(row[key] for entry in report["runs"] if entry["arm"] == arm
                                       for row in entry["checkpoints"] if row["passes"] == passes)
                  for key in ("total_training_cpu_seconds", "neural_cpu_seconds", "selector_cpu_seconds",
                              "updates", "exact_next_word_accuracy", "normalized_edit_distance")}
            for arm in arms}
    report["status"] = "completed"
    atomic_json(directory / "report.json", report)
    atomic_json(latest, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acknowledge-review", action="store_true")
    parser.add_argument("--large", action="store_true",
                        help="Complete Acts 6-12 training / 13-14 evaluation, five seeds, 256 probes")
    parser.add_argument("--corpus", type=Path, help="Reviewed JSON corpus with separate passages and provenance")
    parser.add_argument("--candidate-layers", action="store_true",
                        help="Four-arm small test: baseline, L0, L01, L012; old exclusions deferred")
    parser.add_argument("--selective-l1", action="store_true",
                        help="With --candidate-layers compare baseline, L0, selective L01 and full L01")
    parser.add_argument("--l2-only", action="store_true",
                        help="With --candidate-layers compare baseline against standalone L2 + base")
    parser.add_argument("--savings-l1", action="store_true",
                        help="Compare baseline, L0, top 25%/50% repeated L1 forms by savings, and full L01")
    parser.add_argument("--unique-l1", action="store_true",
                        help="Corrected L1: unique shortest fragments against expanded inventory; base/L0/L01")
    parser.add_argument("--pretrain-l01", action="store_true",
                        help="With --unique-l1 add a fresh L0/L1-only pretrained base before remainder training")
    parser.add_argument("--inject-l01", action="store_true",
                        help="With --pretrain-l01 add frozen reference features to the active remainder base")
    parser.add_argument("--atomic-l01", action="store_true",
                        help="With --unique-l1 compare base, remainder L01, and contextual atomic L01")
    parser.add_argument("--matched-reinforcement", action="store_true",
                        help="Both arms share pretraining and occurrence-triggered literal replay, source-order passes")
    parser.add_argument("--all-acts", action="store_true",
                        help="Matched whole Acts training, with >=20 explicit IN-SAMPLE L1 target checks")
    args = parser.parse_args()
    if args.corpus is not None and args.large:
        parser.error("--corpus and --large are mutually exclusive")
    result = run(Path(__file__).resolve().parent, acknowledge_review=args.acknowledge_review,
                 large=args.large, corpus=args.corpus, candidate_layers=args.candidate_layers,
                 selective_l1=args.selective_l1, l2_only=args.l2_only, savings_l1=args.savings_l1,
                 unique_l1=args.unique_l1, pretrain_l01=args.pretrain_l01, inject_l01=args.inject_l01,
                 atomic_l01=args.atomic_l01, matched_reinforcement=args.matched_reinforcement,
                 all_acts=args.all_acts)
    print(json.dumps({"run_id": result["run_id"], "summary": result["summary"]}, indent=2))
