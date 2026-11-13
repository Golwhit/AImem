from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from neural_candidate_layers import CandidateIndex, AtomicCandidateIndex, REFERENCE, selective_forms, discriminating_forms
from neural_experiment import NeuralModel
from neural_index_pipeline import run, pattern_examples, pattern_recall, protected_features, evaluate, reinforced_schedule, l1_target_probes, whole_acts
from seed_life import LAYER_ZERO_SEEDS
from test_neural_skeleton_runtime import fixture_bank


class CandidateLayerTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.bank = fixture_bank(self.root)

    def test_pretraining_examples_reset_context_and_reject_unknown_characters(self):
        model = NeuralModel(sorted(set("A I returnsunplugged")))
        patterns = ["A ", "I ", "returns", "unplugged"]
        examples = pattern_examples(model, patterns)
        self.assertEqual(examples, [row for pattern in patterns for row in model.examples(pattern)])
        self.assertEqual(examples[2][0], [0] * model.context)
        self.assertEqual(pattern_recall(model, patterns)["rows"][2]["characters"], 7)
        for invalid in ([], [""], ["Ω"]):
            with self.assertRaises(ValueError):
                pattern_examples(model, invalid)

    def test_explicit_l1_opportunities_require_twenty_distinct_offsets(self):
        text = "x " * 20 + "that with " * 20
        index = CandidateIndex.build(self.bank, text, 1)
        extracted = index.extract(text)
        selected = l1_target_probes(text, extracted)
        self.assertEqual(len(selected), 20)
        self.assertEqual(len({item["offset"] for item in selected}), 20)
        self.assertTrue(all(item["prefix"] + item["target"] ==
                            text[:item["offset"] + len(item["target"])] for item in selected))
        with self.assertRaisesRegex(ValueError, "at least 20"):
            l1_target_probes(text, {"occurrences": extracted["occurrences"][:19]})
        with self.assertRaises(ValueError):
            run(self.root, acknowledge_review=True, all_acts=True)

    def test_whole_acts_assembles_all_chapters_and_marks_training_overlap(self):
        early = {"training": "one", "validation": "two", "final_test": "three",
                 "manifest_sha256": "early"}
        with patch("neural_rounds.load_acts", return_value=early), \
             patch("neural_proof.load_fresh", return_value=("four five", {}, "fresh")), \
             patch("neural_index_pipeline.load_chapters", side_effect=[
                 ({c: str(c) for c in range(6, 28)}, "later"), ({28: "28"}, "last")]):
            text, source = whole_acts(self.root)
        self.assertTrue(text.startswith("one\ntwo\nthree\nfour five\n6\n"))
        self.assertTrue(text.endswith("\n28"))
        self.assertEqual(source["chapters"], list(range(1, 29)))
        self.assertIn("NOT held-out", source["evaluation_scope"])

    def test_matched_refreshes_preserve_ordinary_targets_and_contexts(self):
        text = "that x with y "
        index = CandidateIndex.build(self.bank, text, 1)
        extracted = index.extract(text)
        model = NeuralModel(sorted(set(text)))
        full = reinforced_schedule(model, text, extracted, remainder=False)
        rest = reinforced_schedule(model, text, extracted, remainder=True)
        self.assertEqual([(c, t) for c, t, refresh in full if not refresh], model.examples(text))
        self.assertEqual([(c, t) for c, t, refresh in rest if not refresh],
                         model.examples(extracted["remainder"]))
        self.assertEqual([(c, t) for c, t, refresh in full if refresh],
                         [(c, t) for c, t, refresh in rest if refresh])
        self.assertEqual(sum(refresh for _, _, refresh in full), 8)
        self.assertEqual(full[0][0], [0] * model.context)

    def test_atomic_targets_and_context_keep_references_not_internal_letters(self):
        text = "that with that "
        original = CandidateIndex.build(self.bank, text, 1)
        atomic = AtomicCandidateIndex(original.codec)
        alphabet = sorted(set(text))
        model = NeuralModel(alphabet + list(atomic.codec.units))
        pieces = atomic.codec.encode(text)
        examples = atomic.examples(model, text)
        self.assertEqual(len(examples), len(pieces))
        self.assertEqual([model.characters[target - 2] for _, target in examples], pieces)
        ref = next(ref for ref, entry in atomic.codec.units.items() if entry["text"] == "that")
        self.assertEqual(examples[1][0][-1], model.tokens[ref])
        self.assertEqual(examples[0][1], model.tokens[ref])
        self.assertEqual(atomic.counts, {})
        restored = CandidateIndex.restore(atomic.snapshot())
        self.assertIsInstance(restored, AtomicCandidateIndex)
        self.assertEqual(restored.examples(model, text), examples)
        start_ref = next(ref for ref, entry in atomic.codec.units.items() if entry["source_start"])
        def fixed(context):
            p = [0.0] * model.size
            p[model.tokens[ref]] = 0.9
            p[model.tokens[start_ref]] = 0.95
            p[model.tokens[" "]] = 0.1
            return [], [], p
        with patch.object(model, "forward", side_effect=fixed):
            sample = atomic.sample(model, "x ", length=5, greedy=True)
            self.assertEqual(sample["text"], "that ")
            self.assertEqual(sample["neural_calls"], 2)
            self.assertEqual(sample["index_lookups"], 0)
            self.assertEqual(sample["emissions"][0]["reference"], ref)
        self.assertEqual(model.size, len(alphabet) + len(atomic.codec.units) + 2)
        with self.assertRaises(ValueError):
            atomic.examples(NeuralModel(alphabet), text)

    def test_selection_metrics_require_correct_word_not_just_literal_emission(self):
        index = CandidateIndex.build(self.bank, "that with", 1)
        model = NeuralModel(list("that wi"))
        item = {"prefix": "x ", "target": "that", "offset": 2}
        for output, correct in (("that ", 1), ("thatx ", 0), ("with ", 0)):
            with patch.object(index, "sample", return_value={
                "text": output, "indexed_emissions": 1,
                "emissions": [{"offset": 0, "layer": 1, "pattern": output.split()[0][:4]}]}):
                result = evaluate(model, index, [item], selection_metrics=True)
                self.assertEqual(result["selection_metrics"]["correct_l1_first_emissions"], correct)
        with patch.object(index, "sample", return_value={"text": "x ", "indexed_emissions": 0, "emissions": []}):
            result = evaluate(model, index, [item], selection_metrics=True)
            self.assertIsNone(result["selection_metrics"]["l1_selection_precision"])
            recent = {"prefix": "that ", "target": "x", "offset": 5}
            result = evaluate(model, index, [recent], selection_metrics=True)
            self.assertEqual(result["selection_metrics"]["recent_reference_probes"], 1)
            self.assertEqual(result["selection_metrics"]["recent_reference_word_accuracy"], 1)

    def test_protected_features_align_with_past_references_and_survive_updates(self):
        text = "that x with y "
        index = CandidateIndex.build(self.bank, text, 1)
        model = NeuralModel(sorted(set(text + "AIa")))
        features = protected_features(model, index.codec.units)
        index.set_features(features)
        before = index.snapshot()
        extracted = index.extract(text)
        conditions = index.reference_conditions(extracted["pieces"])
        self.assertEqual(len(conditions), len(extracted["remainder"]))
        refs = {e["text"]: ref for ref, e in index.codec.units.items()}
        self.assertEqual(conditions[0], tuple(features[refs["that"]]))
        self.assertEqual(conditions[extracted["remainder"].index("y")], tuple(features[refs["with"]]))
        self.assertEqual(index.reference_conditions(list("x")), [None])
        context, target = model.examples("x")[0]
        model.set_reference_condition(conditions[0])
        for _ in range(3):
            model.update(context, target)
        self.assertEqual(index.snapshot(), before)
        restored = CandidateIndex.restore(before)
        self.assertEqual(restored.snapshot(), before)
        index.counts[()] = Counter({"": 1})
        observed = []
        original = model.forward
        def capture(context, **kwargs):
            observed.append(model.reference_condition)
            return original(context, **kwargs)
        with patch.object(model, "forward", side_effect=capture):
            result = index.sample(model, "that x ", length=2, greedy=True)
            self.assertEqual(result["conditioned_neural_calls"], 2)
            self.assertEqual(observed, [tuple(features[refs["that"]])] * 2)
            index.sample(model, "x ", length=1, greedy=True)
            self.assertIsNone(observed[-1])
            index.counts[()] = Counter({refs["that"]: 100})
            observed.clear()
            emitted = index.sample(model, "", length=5, greedy=True)
            self.assertEqual(emitted["indexed_emissions"], 1)
            self.assertEqual(emitted["conditioned_neural_calls"], 1)
            self.assertEqual(emitted["reference_injections"], 1)
            self.assertEqual(observed, [tuple(features[refs["that"]])])
        with self.assertRaises(ValueError):
            restored.set_features({})

    def test_unique_fragments_require_single_owner_and_extract_whole_payload(self):
        possibilities = {"haven't", "haven's", "doesn't", "pretzel", "pretzels", "however", "wevet"}
        text = "A haven't doesn't pretzel however."
        forms = discriminating_forms(text, possibilities)
        self.assertIn("haven't", forms)
        self.assertNotEqual(forms["haven't"]["key"], "haven'")
        self.assertNotIn("pretzel", forms)
        self.assertNotEqual(forms["however"]["key"], "wev")
        for word, evidence in forms.items():
            key = evidence["key"]
            universe = possibilities | {"a", "haven't", "doesn't", "pretzel", "however"}
            self.assertEqual({w for w in universe if key in w}, {word})
            for size in range(1, len(key)):
                for offset in range(len(word) - size + 1):
                    fragment = word[offset:offset + size]
                    self.assertGreater(len([w for w in universe if fragment in w]), 1)
        index = CandidateIndex.build(self.bank, text, 1, possibilities=possibilities)
        extracted = index.extract(text)
        self.assertIn("pretzel", extracted["remainder"])
        self.assertNotIn("haven't", extracted["remainder"])
        self.assertNotIn("doesn't", extracted["remainder"])
        self.assertEqual(index.codec.units, CandidateIndex.restore(index.snapshot()).codec.units)
        self.assertTrue(all(e["kind"] == "unique_fragment" for e in index.codec.units.values()
                            if e["layer"] == 1))
        self.assertNotIn("however", discriminating_forms("however howevers", possibilities))
        self.assertEqual(discriminating_forms("pretzel", {"pretzel", "pretzels"}), {})

    def test_foundation_is_unchanged_typed_lossless_and_source_scoped(self):
        index = CandidateIndex.build(self.bank, "A chair and a table. I sit.", 0)
        self.assertEqual(len(index.codec.units), len(LAYER_ZERO_SEEDS))
        for text in ("A chair and a table. I sit.", "Apple a table.", "I ", " A ", "that*"):
            result = index.extract(text)
            self.assertEqual("".join(index.codec.units[p]["text"] if p in index.codec.units else p
                                     for p in result["pieces"]), text)
            for row in result["occurrences"]:
                self.assertEqual(text[row["start"]:row["end"]], row["pattern"])
        self.assertEqual(index.extract("I ")["remainder"], "")
        self.assertEqual(index.extract(" x A")["remainder"], " x A")
        with self.assertRaises(ValueError):
            index.extract(REFERENCE)

    def test_blocks_only_from_unresolved_spans_and_literal_mask_never_inferred(self):
        train = "that repair repair repair with repair "
        index = CandidateIndex.build(self.bank, train, 2)
        result = index.extract(train)
        self.assertTrue(any(row["layer"] == 2 for row in result["occurrences"]))
        self.assertTrue(any(row["layer"] == 1 for row in result["occurrences"]))
        self.assertFalse(any(row["kind"] == "hidden_backed_complete" for row in result["occurrences"]))
        restored = CandidateIndex.restore(index.snapshot())
        self.assertEqual(restored.extract(train), result)
        masked = index.extract("th*t")
        self.assertEqual(masked["remainder"], "th*t")
        rows = result["occurrences"]
        self.assertTrue(all(a["end"] <= b["start"] for a, b in zip(rows, rows[1:])))
        self.assertFalse(any(entry["text"] == "hat" for entry in index.codec.units.values()
                             if entry["kind"] == "repeated_block"))

    def test_generation_preserves_reference_identity_and_does_not_train_base_on_it(self):
        train = "repair repair repair repair "
        index = CandidateIndex.build(self.bank, train, 2)
        result = index.extract(train)
        index.learn(result["pieces"])
        model = NeuralModel(sorted(set(train)))
        self.assertTrue(all(len(c) == 1 for c in model.tokens))
        self.assertEqual(len(model.examples(result["remainder"])), len(result["remainder"]))
        first = result["pieces"][0]
        self.assertIn(first, index.codec.units)
        literal = index.codec.units[first]["text"]
        index.counts[()] = Counter({first: 100})
        with patch.object(model, "forward", side_effect=AssertionError("Indexed emission needs no forward")):
            sample = index.sample(model, "", length=len(literal), greedy=True)
        self.assertEqual(sample["text"], literal)
        self.assertEqual(sample["neural_calls"], 0)
        self.assertEqual(sample["emissions"][0]["reference"], first)

    def test_hidden_backed_complete_match_when_earlier_layer_leaves_it(self):
        from neural_skeleton_runtime import UnitCodec

        with patch.object(UnitCodec, "from_bank", return_value=UnitCodec({}, {})):
            index = CandidateIndex.build(self.bank, "that repair repair repair ", 2)
        extracted = index.extract("that ")
        self.assertEqual(extracted["occurrences"][0]["kind"], "hidden_backed_complete")
        self.assertEqual(extracted["occurrences"][0]["pattern"], "that")
        self.assertEqual(index.extract("th*t ")["remainder"], "th*t ")
        self.assertEqual(index.extract("thit ")["remainder"], "thit ")

    def test_selective_gate_requires_repeated_predictive_context_and_savings(self):
        strong = selective_forms("use that. use that. use that. ", {"that"})["that"]
        self.assertTrue(strong["selected"])
        self.assertEqual(strong["potential_saved_characters"], 9)
        ambiguous = selective_forms("use that. use with. use that. use with. ", {"that"})["that"]
        self.assertFalse(ambiguous["selected"])
        self.assertEqual(ambiguous["reliable_occurrences"], 0)
        singleton = selective_forms("use that. ", {"that"})["that"]
        self.assertFalse(singleton["selected"])
        short = selective_forms("use it. use it. ", {"it"})["it"]
        self.assertFalse(short["selected"])
        varied = selective_forms("use that. use that. see that. ", {"that"})["that"]
        self.assertFalse(varied["selected"])
        self.assertEqual(selective_forms("use this.", {"that"})["that"]["occurrences"], 0)
        index = CandidateIndex.build(self.bank, "use that. use that. use that. with.", 1, selective=True)
        forms = {entry["text"] for entry in index.codec.units.values() if entry["layer"] == 1}
        self.assertEqual(forms, {"that"})
        self.assertEqual(len([e for e in index.codec.units.values() if e["layer"] == 0]), 7)
        restored = CandidateIndex.restore(index.snapshot())
        self.assertEqual(index.extract("with that."), restored.extract("with that."))
        with self.assertRaises(ValueError):
            CandidateIndex.build(self.bank, "that", 0, selective=True)

    def test_standalone_l2_has_no_foundation_or_l1_and_uses_full_training(self):
        index = CandidateIndex.build(self.bank, "A repair that repair with repair. ", 2, l2_only=True)
        self.assertEqual({entry["layer"] for entry in index.codec.units.values()}, {2})
        extracted = index.extract("A repair that repair with repair. ")
        self.assertEqual({row["kind"] for row in extracted["occurrences"]},
                         {"repeated_block", "hidden_backed_complete"})
        self.assertTrue(extracted["remainder"].startswith("A "))
        self.assertEqual(index.extract("th*t")["remainder"], "th*t")
        self.assertEqual(CandidateIndex.restore(index.snapshot()).extract("that repair"), index.extract("that repair"))
        with self.assertRaises(ValueError):
            CandidateIndex.build(self.bank, "that", 1, l2_only=True)

    def test_savings_selection_is_nested_ranked_and_excludes_singletons(self):
        text = "that with long word tiny that with long word tiny single"
        words = {"that", "with", "long", "word", "tiny", "single", "absent"}
        quarter = selective_forms(text, words, savings_fraction=0.25)
        half = selective_forms(text, words, savings_fraction=0.5)
        picked_quarter = {w for w, row in quarter.items() if row["selected"]}
        picked_half = {w for w, row in half.items() if row["selected"]}
        self.assertEqual(picked_quarter, {"long", "that"})
        self.assertEqual(picked_half, {"long", "that", "tiny"})
        self.assertLess(picked_quarter, picked_half)
        self.assertFalse(half["single"]["selected"])
        self.assertFalse(half["absent"]["selected"])
        self.assertFalse(any(row["selected"] for row in
                             selective_forms("that", {"that"}, savings_fraction=0.5).values()))
        with self.assertRaises(ValueError):
            selective_forms(text, words, savings_fraction=0.75)
        with self.assertRaises(ValueError):
            CandidateIndex.build(self.bank, text, 2, savings_fraction=0.25)

    def test_savings_ranks_savings_before_spelling_and_keeps_rejected_targets(self):
        text = "A with that with that with that with that with that with that with. "
        evidence = selective_forms(text, {"A", "with", "that"}, savings_fraction=0.5)
        self.assertTrue(evidence["with"]["selected"])
        self.assertFalse(evidence["that"]["selected"])
        index = CandidateIndex.build(self.bank, text, 1, savings_fraction=0.5)
        self.assertEqual({e["text"] for e in index.codec.units.values() if e["layer"] == 1},
                         {"with"})
        foundation = CandidateIndex.build(self.bank, text, 0)
        self.assertEqual([e for e in index.codec.units.values() if e["layer"] == 0],
                         list(foundation.codec.units.values()))
        extracted = index.extract(text)
        self.assertEqual(extracted["remainder"].count("that"), 6)
        self.assertNotIn("with", extracted["remainder"])
        restored = CandidateIndex.restore(index.snapshot())
        self.assertEqual(restored.extract(text), extracted)
        before = index.snapshot()
        index.extract("An entirely different evaluation passage with that.")
        self.assertEqual(index.snapshot(), before)

    def test_four_arm_run_persists_paired_bases_and_exact_remaining_targets(self):
        for name in ("neural_index_pipeline.py", "neural_experiment.py", "neural_skeleton_runtime.py",
                     "neural_unit_benchmark.py", "neural_candidate_layers.py"):
            (self.root / name).write_text("fixture")
        text = "A repair that repair with a repair. " * 3
        record = {"title": "test", "provenance": "authored fixture", "rights_note": "test only",
                  "limitations": "artificial", "training": text,
                  "evaluation": "I repair with that table. " * 3}
        corpus = self.root / "corpus.json"
        corpus.write_text(json.dumps(record))
        with patch("neural_index_pipeline.load_bank", return_value=self.bank):
            report = run(self.root, acknowledge_review=True, corpus=corpus, candidate_layers=True)
        self.assertEqual(report["method"], "candidate_layer_comparison")
        self.assertEqual(len(report["runs"]), 12)
        directory = self.root / ".data/neural-unit-benchmark" / report["run_id"]
        self.assertEqual(len(list(directory.glob("*-passes-*.json"))), 36)
        inputs = json.loads((directory / "inputs.json").read_text())
        for seed in report["settings"]["seeds"]:
            group = [entry for entry in report["runs"] if entry["seed"] == seed]
            for key in ("initial_weights_sha256", "vocabulary", "parameters"):
                self.assertEqual(len({entry[key] for entry in group}), 1)
            for entry in group:
                arm = entry["arm"]
                if arm == "baseline":
                    continue
                remainder = inputs["layer_extractions"][arm]["remainder"]
                self.assertEqual(entry["targets_per_pass"], len(remainder))
                saved = json.loads((directory / f"{seed}-{arm}-passes-4.json").read_text())
                model = NeuralModel(saved["characters"], seed=seed)
                model.weights = saved["weights"]
                index = CandidateIndex.restore(saved["index"])
                self.assertEqual(index.sample(model, entry["samples"][0]["prompt"])["text"],
                                 entry["samples"][0]["text"])
        with self.assertRaises(ValueError):
            run(self.root, acknowledge_review=True, large=True, candidate_layers=True)
        with patch("neural_index_pipeline.load_bank", return_value=self.bank):
            selective_report = run(self.root, acknowledge_review=True, corpus=corpus,
                                   candidate_layers=True, selective_l1=True)
        self.assertEqual(selective_report["settings"]["arms"],
                         ["baseline", "layers_0", "layers_01_selective", "layers_01"])
        self.assertTrue(selective_report["settings"]["selective_l1"])
        self.assertEqual(len(selective_report["runs"]), 12)
        evidence = selective_report["selection"]["selective_l1"]
        expected = selective_forms(text, {"that", "with"})
        self.assertEqual(evidence, expected)
        with self.assertRaises(ValueError):
            run(self.root, acknowledge_review=True, corpus=corpus, selective_l1=True)
        with patch("neural_index_pipeline.load_bank", return_value=self.bank):
            standalone = run(self.root, acknowledge_review=True, corpus=corpus,
                             candidate_layers=True, l2_only=True)
        self.assertEqual(standalone["settings"]["arms"], ["baseline", "layers_2"])
        self.assertEqual(len(standalone["runs"]), 6)
        self.assertTrue(standalone["settings"]["l2_only"])
        self.assertEqual(set(standalone["layer_extractions"]["layers_2"]["by_layer"]), {2})
        with patch("neural_index_pipeline.load_bank", return_value=self.bank):
            savings = run(self.root, acknowledge_review=True, corpus=corpus,
                          candidate_layers=True, savings_l1=True)
        self.assertEqual(savings["settings"]["arms"],
                         ["baseline", "layers_0", "layers_01_quarter", "layers_01_half", "layers_01"])
        self.assertEqual(len(savings["runs"]), 15)
        self.assertEqual(set(savings["selections"]), {"layers_01_quarter", "layers_01_half"})
        for arm, fraction in (("layers_01_quarter", 0.25), ("layers_01_half", 0.5)):
            self.assertEqual(savings["selections"][arm]["selective_l1"],
                             selective_forms(text, {"that", "with"}, savings_fraction=fraction))
        with self.assertRaises(ValueError):
            run(self.root, acknowledge_review=True, savings_l1=True)
        inventory = self.root / ".data/possible-words-expanded-scowl-2020.12.07"
        inventory.mkdir()
        raw = b"repair\nrepairs\nthat\nwith\n"
        (inventory / "words.txt").write_bytes(raw)
        (inventory / "manifest.json").write_text(json.dumps({
            "words_sha256": hashlib.sha256(raw).hexdigest(), "entries": 4}))
        with patch("neural_index_pipeline.load_bank", return_value=self.bank):
            unique = run(self.root, acknowledge_review=True, corpus=corpus,
                         candidate_layers=True, unique_l1=True)
        self.assertEqual(unique["settings"]["arms"], ["baseline", "layers_0", "layers_01"])
        self.assertEqual(len(unique["runs"]), 9)
        self.assertNotIn("repair", unique["unique_l1"]["forms"])
        self.assertEqual(unique["unique_l1"]["forms"], discriminating_forms(text, set(raw.decode().splitlines())))
        with patch("neural_index_pipeline.load_bank", return_value=self.bank):
            pretrained = run(self.root, acknowledge_review=True, corpus=corpus,
                             candidate_layers=True, unique_l1=True, pretrain_l01=True, inject_l01=True)
        self.assertEqual(pretrained["settings"]["arms"],
                         ["baseline", "layers_0", "layers_01", "layers_01_pretrained", "layers_01_injected"])
        self.assertEqual(len(pretrained["runs"]), 15)
        saved_dir = self.root / ".data/neural-unit-benchmark" / pretrained["run_id"]
        for seed in pretrained["settings"]["seeds"]:
            group = [entry for entry in pretrained["runs"] if entry["seed"] == seed]
            self.assertEqual(len({entry["initial_weights_sha256"] for entry in group}), 1)
            warm = next(entry for entry in group if entry["arm"] == "layers_01_pretrained")
            cold = next(entry for entry in group if entry["arm"] == "layers_01")
            self.assertEqual(warm["examples_sha256"], cold["examples_sha256"])
            checkpoint_path = saved_dir / warm["pretraining"]["checkpoint"]
            snapshot = json.loads(checkpoint_path.read_text())
            self.assertEqual(warm["pretraining"]["checkpoint_sha256"],
                             hashlib.sha256(checkpoint_path.read_bytes()).hexdigest())
            self.assertEqual(warm["starting_weights_sha256"],
                             hashlib.sha256(json.dumps(snapshot["weights"], sort_keys=True).encode()).hexdigest())
            self.assertNotEqual(warm["starting_weights_sha256"], warm["initial_weights_sha256"])
            for row in warm["checkpoints"]:
                self.assertAlmostEqual(row["total_training_cpu_seconds"],
                                       row["neural_cpu_seconds"] + row["selector_cpu_seconds"]
                                       + warm["pretraining"]["cpu_seconds"])
                self.assertEqual(row["updates"], row["passes"] * cold["targets_per_pass"])
            model = NeuralModel(snapshot["characters"], seed=seed)
            model.weights = snapshot["weights"]
            self.assertEqual(pattern_recall(model, snapshot["patterns"]), warm["pretraining"]["after"])
            self.assertEqual(snapshot["updates"], 200 * sum(map(len, snapshot["patterns"])))
            injected = next(entry for entry in group if entry["arm"] == "layers_01_injected")
            self.assertEqual(injected["starting_weights_sha256"], warm["starting_weights_sha256"])
            self.assertEqual(injected["pretraining"]["checkpoint_sha256"], warm["pretraining"]["checkpoint_sha256"])
            self.assertEqual(injected["examples_sha256"], warm["examples_sha256"])
            final = json.loads((saved_dir / f"{seed}-layers_01_injected-passes-4.json").read_text())
            rebuilt = CandidateIndex.restore(final["index"])
            self.assertEqual(rebuilt.snapshot()["features"], protected_features(model, rebuilt.codec.units))
            for row in injected["checkpoints"]:
                self.assertEqual(row["features_sha256"], injected["injection"]["features_sha256"])
                self.assertAlmostEqual(row["total_training_cpu_seconds"],
                                       row["neural_cpu_seconds"] + row["selector_cpu_seconds"]
                                       + injected["pretraining"]["cpu_seconds"]
                                       + injected["injection"]["feature_cpu_seconds"])
            active_model = NeuralModel(final["characters"], seed=seed)
            active_model.weights = final["weights"]
            self.assertEqual(rebuilt.sample(active_model, injected["samples"][0]["prompt"])["text"],
                             injected["samples"][0]["text"])
        with self.assertRaises(ValueError):
            run(self.root, acknowledge_review=True, pretrain_l01=True)
        with patch("neural_index_pipeline.load_bank", return_value=self.bank):
            atomic = run(self.root, acknowledge_review=True, corpus=corpus,
                         candidate_layers=True, unique_l1=True, atomic_l01=True)
        self.assertEqual(atomic["settings"]["arms"], ["baseline", "layers_01", "layers_01_atomic"])
        self.assertEqual(len(atomic["runs"]), 9)
        atomic_dir = self.root / ".data/neural-unit-benchmark" / atomic["run_id"]
        atomic_inputs = json.loads((atomic_dir / "inputs.json").read_text())
        for entry in atomic["runs"]:
            saved = json.loads((atomic_dir / f"{entry['seed']}-{entry['arm']}-passes-4.json").read_text())
            model = NeuralModel(saved["characters"], seed=entry["seed"])
            model.weights = saved["weights"]
            if entry["arm"] != "layers_01_atomic":
                continue
            codec = CandidateIndex.restore(saved["index"])
            self.assertIsInstance(codec, AtomicCandidateIndex)
            self.assertEqual(codec.counts, {})
            self.assertEqual(entry["targets_per_pass"], len(codec.codec.encode(text)))
            self.assertEqual(entry["examples_sha256"],
                             hashlib.sha256(json.dumps(codec.examples(model, text)).encode()).hexdigest())
            self.assertEqual(codec.sample(model, entry["samples"][0]["prompt"])["text"],
                             entry["samples"][0]["text"])
            self.assertTrue(all(row["selector_cpu_seconds"] == 0 for row in entry["checkpoints"]))
            self.assertEqual(saved["index"]["codec"], atomic_inputs["candidate_indices"]["layers_01"]["codec"])
        with self.assertRaises(ValueError):
            run(self.root, acknowledge_review=True, atomic_l01=True)
        with patch("neural_index_pipeline.load_bank", return_value=self.bank):
            matched = run(self.root, acknowledge_review=True, corpus=corpus,
                          candidate_layers=True, unique_l1=True, pretrain_l01=True,
                          matched_reinforcement=True)
        self.assertEqual(matched["settings"]["arms"], ["baseline", "layers_01"])
        for seed in matched["settings"]["seeds"]:
            group = [e for e in matched["runs"] if e["seed"] == seed]
            self.assertEqual(len({e["starting_weights_sha256"] for e in group}), 1)
            self.assertEqual(len({e["pretraining"]["checkpoint_sha256"] for e in group}), 1)
            self.assertEqual(len({e["reinforcement"]["updates_per_pass"] for e in group}), 1)
            for entry in group:
                for row in entry["checkpoints"]:
                    self.assertEqual(row["reinforcement_updates"],
                                     row["passes"] * matched["reinforcement"]["updates_per_pass"])
                    self.assertAlmostEqual(row["total_training_cpu_seconds"],
                                           row["neural_cpu_seconds"] + row["selector_cpu_seconds"]
                                           + row["pretraining_cpu_seconds"] + row["reinforcement_cpu_seconds"])
        with self.assertRaises(ValueError):
            run(self.root, acknowledge_review=True, matched_reinforcement=True)
        (inventory / "words.txt").write_text("tampered")
        with patch("neural_index_pipeline.load_bank", return_value=self.bank):
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                run(self.root, acknowledge_review=True, corpus=corpus, candidate_layers=True, unique_l1=True)


if __name__ == "__main__":
    unittest.main()
