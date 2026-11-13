import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from neural_base_comparison import ARMS, WordBase, generate, evaluate, run, summarize, analyze_saved_run, RATE
from neural_candidate_layers import CandidateCodec, CandidateIndex
from neural_experiment import NeuralModel


class FullBaseComparisonTests(unittest.TestCase):
    def test_quality_timing_uses_earliest_original_crossing(self):
        runs = []
        for arm in ARMS:
            multiplier = 10 if arm == "original" else 12
            points = [{
                "passes": p, "accuracy": .2 if arm == "original" and p == 1 else .1,
                "edit": .8, "training_cpu_seconds": p * multiplier - 1,
                "total_cpu_seconds": p * multiplier, "updates": p,
                "evaluation_cpu_seconds": 1,
            } for p in (1, 2, 4)]
            runs.append({"seed": 17, "arm": arm, "checkpoints": points,
                         "before": {"passes": 0, "accuracy": 0, "total_cpu_seconds": 1}})
        report = {"seeds": [17], "runs": runs}
        summarize(report)
        candidate = next(row for row in report["time_to_original_accuracy"] if row["arm"] == "new")
        self.assertEqual(candidate["original_earliest_total_cpu_seconds"], 10)
        self.assertEqual(candidate["total_cpu_seconds"], 12)
        self.assertFalse(candidate["time_reduction_30pct"])
        self.assertAlmostEqual(candidate["time_reduction_vs_earliest_original"], -.2)
        best = next(row for row in report["time_to_original_best_accuracy"] if row["arm"] == "new")
        self.assertIsNone(best["crossing_pass"])
        self.assertEqual(best["threshold"], .2)
        for entry in runs:
            entry["before"]["accuracy"] = 0
            for point in entry["checkpoints"]:
                point["accuracy"] = 0
        summarize(report)
        self.assertTrue(all(not row["meaningful_threshold"] and not row["time_reduction_30pct"]
                            for row in report["time_to_original_accuracy"]))
        self.assertFalse(report["endpoint_criteria"]["new"]["accuracy_30pct"])

    def test_lossless_surface_tokens_fallback_and_causal_features(self):
        text = 'Cat, cat! CAT can\'t can’t cant well-known. tac\n'
        model = WordBase(text, 17, lexicon=["cat", "can't", "well-known"])
        tokens = model.encode(text)
        self.assertEqual("".join(model.payloads[token] for token in tokens), text)
        self.assertIn(model.model.tokens["emit:Cat"], tokens)
        self.assertNotIn("emit:tac", model.model.tokens)
        self.assertEqual(model.encode("tac"), [model.model.tokens["char:" + c] for c in "tac"])
        self.assertEqual(model.context("Cat, "), model.context("cat, "))
        self.assertNotEqual(model.context("Cat, "), model.context("Cat! "))
        self.assertNotEqual(model.context("can't "), model.context("cant "))
        examples = model.examples(text)
        prefix = ""
        for context, token, prior in examples:
            self.assertEqual(context, model.context(prefix))
            self.assertTrue(prior[token])
            self.assertEqual(sum(prior[:2]), 0)
            prefix += model.payloads[token]
        self.assertEqual(prefix, text)
        self.assertEqual(model.context(""), [0] * 4)
        for prefix in ("Cat ", "Cat, cat ", "Cat! CAT can't ", "can’t ", "hello-", "t"):
            # Compare the optimized backward scan against a full regex scan.
            from neural_word_context import WORD, gap_symbol
            matches = list(WORD.finditer(prefix))
            partial = matches[-1] if matches and matches[-1].end() == len(prefix) else None
            complete = matches[:-1] if partial else matches
            identity = lambda match: model.model.tokens.get(
                "identity:" + match.group().lower().replace("’", "'"),
                model.model.tokens["identity:<other>"])
            if partial:
                prev = complete[-1] if complete else None
                gap = prefix[prev.end() if prev else 0:partial.start()]
                tail = partial.group()[-2:]
                expected = [identity(prev) if prev else 0, model.model.tokens.get(gap_symbol(gap), 1),
                            *([0] * (2 - len(tail))), *(model.model.tokens.get("char:" + c, 1) for c in tail)]
            else:
                expected = []
                for i in (-2, -1):
                    if len(complete) < -i:
                        expected.extend([0, 0])
                    else:
                        following = complete[i + 1].start() if i == -2 else len(prefix)
                        expected.extend([identity(complete[i]), model.model.tokens.get(
                            gap_symbol(prefix[complete[i].end():following]), 1)])
            self.assertEqual(model.context(prefix), expected)
        state = model.snapshot()
        restored = WordBase.restore(state)
        self.assertEqual(restored.snapshot(), state)
        self.assertEqual(generate(model, "Cat, "), generate(restored, "Cat, "))
        self.assertEqual(state, model.snapshot())

    def test_generation_can_spell_a_word_absent_from_lexicon(self):
        model = WordBase("cat tac ", 17, lexicon=["cat"])
        counter = [0]
        payload = "tac "
        def forward(context, *, output_prior=None):
            token = model.model.tokens["char:" + payload[counter[0] % 4]]
            counter[0] += 1
            probabilities = [0.0] * model.model.size
            probabilities[token] = 1.0
            return [], [], probabilities
        with patch.object(model.model, "forward", side_effect=forward):
            result = generate(model, "", length=4)
        self.assertEqual(result["text"], "tac ")
        self.assertEqual(result["neural_calls"], 4)
        with self.assertRaises(ValueError):
            generate(model, "\x00layer:1:0")
        with self.assertRaises(ValueError):
            generate(model, "", length=0)
        for generator in (model, NeuralModel(["a", " "])):
            network = generator.model if isinstance(generator, WordBase) else generator
            with patch.object(network, "forward", return_value=([], [], [float("nan")] * network.size)):
                with self.assertRaises(ArithmeticError):
                    generate(generator, "")

    def test_placement_can_interrupt_a_queued_word_payload(self):
        model = WordBase("cat XYZ ", 17, lexicon=["cat"])
        ref = "\x00layer:2:0"
        index = CandidateIndex(CandidateCodec({
            ref: {"text": "XYZ", "layer": 2, "kind": "repeated_block",
                  "source_start": False, "bounded": False}}, {}))
        index.learn(["c", ref, "a", "t", " "])
        token = model.model.tokens["emit:cat"]
        def forward(context, *, output_prior=None):
            probabilities = [0.0] * model.model.size
            probabilities[token] = 1.0
            return [], [], probabilities
        with patch.object(model.model, "forward", side_effect=forward):
            result = generate(model, "", index, length=6)
        self.assertEqual(result["text"], "cXYZat")
        self.assertEqual(result["neural_calls"], 1)
        self.assertEqual(result["indexed_emissions"], 1)
        self.assertEqual(result["index_lookups"], 4)

    def test_five_arm_run_replays_all_checkpoints_and_isolates_active_state(self):
        text = "A returns unplugged. A returns unplugged. " * 2
        held = "A returns unplugged. A unplugged returns. "
        items = [{"prefix": held[:2], "target": "returns", "offset": 2},
                 {"prefix": held[:10], "target": "unplugged", "offset": 10}]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            active = root / ".data/neural-chat/active.json"
            active.parent.mkdir(parents=True)
            active.write_text("unchanged")
            words = b"A\nreturns\nunplugged\n"
            inventory = root / ".data/possible-words-expanded-scowl-2020.12.07"
            inventory.mkdir()
            (inventory / "words.txt").write_bytes(words)
            (inventory / "manifest.json").write_text(json.dumps({
                "entries": 3, "words_sha256": hashlib.sha256(words).hexdigest()}))
            with self.assertRaises(ValueError):
                run(root, acknowledge_review=False)
            with patch("neural_base_comparison.load_inputs", return_value=(text, held, items, {"fixture": True})), \
                 patch("neural_base_comparison.load_bank", return_value={"patterns": {}}):
                report = run(root, acknowledge_review=True, seeds=(17,), include_retired_layers=True)
            self.assertEqual(report["status"], "completed")
            self.assertEqual({row["arm"] for row in report["runs"]}, set(ARMS))
            new = [row for row in report["runs"] if row["arm"] != "original"]
            self.assertEqual(len({row["initial_weights_sha256"] for row in new}), 1)
            self.assertEqual(len({row["parameters"] for row in new}), 1)
            directory = root / ".data/neural-base-comparison" / report["run_id"]
            inputs = json.loads((directory / "inputs.json").read_text())
            for entry in report["runs"]:
                arm = entry["arm"]
                model = NeuralModel(sorted(set(text)), seed=17) if arm == "original" else WordBase(text, 17)
                index = CandidateIndex.restore(inputs["indices"][arm]) if arm in inputs["indices"] else None
                extracted = index.extract(text) if index else None
                remainder = extracted["remainder"] if extracted else text
                self.assertEqual(len(remainder), entry["remainder_characters"])
                examples = model.examples(remainder)
                rng = random.Random(19)
                for epoch in range(1, 5):
                    if index:
                        index.learn(extracted["pieces"])
                    order = list(range(len(examples)))
                    rng.shuffle(order)
                    for i in reversed(order):
                        if isinstance(model, WordBase):
                            context, token, prior = examples[i]
                            model.model.update(context, token, learning_rate=RATE, output_prior=prior)
                        else:
                            model.update(*examples[i], learning_rate=RATE)
                    if epoch not in (1, 2, 4):
                        continue
                    saved = json.loads((directory / f"17-{arm}-passes-{epoch}.json").read_text())
                    state = model.snapshot() if isinstance(model, WordBase) else {
                        "characters": model.characters, "weights": model.weights}
                    self.assertEqual(state, saved["state"])
                    self.assertEqual(index.snapshot() if index else None, saved["index"])
                    result = evaluate(model, items, index)
                    for key in ("accuracy", "edit", "rows", "probes"):
                        self.assertEqual(result[key], saved["evidence"][key])
            self.assertEqual(active.read_text(), "unchanged")
            self.assertEqual(len(report["time_to_original_accuracy"]), 5)
            self.assertTrue(all(row["crossing_pass"] in (None, 0, 1, 2, 4)
                                for row in report["time_to_original_accuracy"]))
            raw_report = (directory / "report.json").read_bytes()
            analysis = analyze_saved_run(directory)
            self.assertEqual(analysis["restoration_validation"]["verified_checkpoints_including_initial"], 20)
            self.assertEqual((directory / "report.json").read_bytes(), raw_report)
            inputs["training"] += "changed"
            (directory / "inputs.json").write_text(json.dumps(inputs))
            with self.assertRaises(ValueError):
                analyze_saved_run(directory)

    def test_working_comparison_does_not_load_or_construct_layers(self):
        text = "cat cat. "
        items = [{"prefix": "cat ", "target": "cat", "offset": 4}]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch("neural_base_comparison.load_inputs", return_value=(text, text, items, {})), \
                 patch("neural_base_comparison.load_bank", side_effect=AssertionError("Retired bank loaded")), \
                 patch.object(CandidateIndex, "build", side_effect=AssertionError("Retired layers built")):
                report = run(root, acknowledge_review=True, seeds=(17,))
            self.assertEqual(report["arms"], ["original", "new"])
            self.assertFalse(report["retired_layers_enabled"])
            self.assertEqual(set(report["summary"]["4"]), {"original", "new"})
            self.assertTrue(all(not row["layer_occurrences"] for row in report["runs"]))
            self.assertEqual(analyze_saved_run(
                root / ".data/neural-base-comparison" / report["run_id"]
            )["restoration_validation"]["verified_checkpoints_including_initial"], 8)


if __name__ == "__main__":
    unittest.main()
