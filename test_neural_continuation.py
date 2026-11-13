from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from neural_candidate_layers import CandidateIndex
from neural_chat import activate, generate, status
from neural_continuation import (
    BOOKS, CORPUS, chapter_text, extend_alphabet, extend_index, load_corpus, run, stage,
)
from neural_experiment import NeuralModel, EMBEDDING, HIDDEN
from neural_index_pipeline import evaluate, reinforced_schedule
from neural_unit_benchmark import text_hash
from test_neural_chat import chat_fixture


class ContinuationTests(unittest.TestCase):
    def test_complete_source_staging_and_tamper_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / ".data/acts-web-staged"
            source.mkdir(parents=True)
            (source / "manifest.json").write_text(json.dumps({
                "rights_url": "https://ebible.org/engwebp/copyright.htm",
                "translation": "WEB fixture", "rights_statement": "fixture",
                "preprocessing": "strip markup",
            }))

            def fetcher(url):
                if url.endswith("copyright.htm"):
                    return b"The World English Bible is in the Public Domain."
                name = url.rsplit("/", 1)[1]
                count = BOOKS[name[:3]][int(name[3:5]) - 1]
                return "".join(f'<span class="verse" id="V{v}">{v}</span>Fixture verse {v}. '
                               for v in range(1, count + 1)).encode()

            with self.assertRaises(ValueError):
                stage(root, acknowledge_review=False, fetcher=fetcher)
            manifest = stage(root, acknowledge_review=True, fetcher=fetcher)
            self.assertEqual(len(manifest["sources"]), 29)
            self.assertEqual(sum(row["verses"] for row in manifest["sources"]), 694)
            training, record = load_corpus(root)
            self.assertTrue(training.strip())
            self.assertEqual(len(record["manifest_sha256"]), 64)
            with self.assertRaises(ValueError):
                stage(root, acknowledge_review=True, fetcher=fetcher)
            (root / CORPUS / "2CO13.txt").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "integrity"):
                load_corpus(root)
        with self.assertRaisesRegex(ValueError, "inventory"):
            chapter_text(b'<span class="verse" id="V1">1</span>one', 2)
        with self.assertRaisesRegex(ValueError, "empty"):
            chapter_text(b'<span class="verse" id="V1">1</span>', 1)

    def test_alphabet_growth_preserves_old_parameters_and_token_ids(self):
        model = NeuralModel(list("ab "))
        before = deepcopy(model.weights)
        expanded = extend_alphabet(model, list("b az"))
        self.assertEqual(expanded.characters, list("ab z"))
        self.assertEqual({char: expanded.tokens[char] for char in model.characters}, model.tokens)
        self.assertEqual(expanded.weights["embedding"][:model.size * EMBEDDING], before["embedding"])
        self.assertEqual(expanded.weights["hidden"], before["hidden"])
        self.assertEqual(expanded.weights["hidden_bias"], before["hidden_bias"])
        self.assertEqual(expanded.weights["output_bias"][:model.size], before["output_bias"])
        for j in range(HIDDEN):
            self.assertEqual(expanded.weights["output"][j * expanded.size:j * expanded.size + model.size],
                             before["output"][j * model.size:(j + 1) * model.size])
        self.assertEqual(model.weights, before)
        self.assertIs(extend_alphabet(model, list("a")), model)
        self.assertTrue(all(target >= 2 for _, target in expanded.examples("az")))

    def test_index_growth_preserves_reference_ids_and_trained_counts(self):
        prior = "A returns unplugged returns unplugged. "
        new = "amazingly amazingly returns. "
        possibilities = {"returns", "unplugged", "amazingly", "A"}
        parent = CandidateIndex.build({}, prior, 1, possibilities=possibilities)
        parent.learn(parent.extract(prior)["pieces"])
        before = parent.snapshot()
        updated = extend_index(parent, prior, new, possibilities)
        for reference, entry in parent.codec.units.items():
            self.assertEqual(updated.codec.units[reference]["text"], entry["text"])
        self.assertEqual(updated.counts, parent.counts)
        self.assertGreater(len(updated.codec.units), len(parent.codec.units))
        extracted = updated.extract(new)
        model = NeuralModel(sorted(set(prior + new)))
        base = reinforced_schedule(model, new, extracted, remainder=False)
        layered = reinforced_schedule(model, new, extracted, remainder=True)
        self.assertEqual([(c, t) for c, t, refresh in base if refresh],
                         [(c, t) for c, t, refresh in layered if refresh])
        self.assertEqual(sum(not refresh for _, _, refresh in base), len(new))
        self.assertEqual(sum(not refresh for _, _, refresh in layered), len(extracted["remainder"]))
        self.assertEqual(parent.snapshot(), before)
        ambiguous = CandidateIndex.build({}, "A cat. ", 1, possibilities={"cat"})
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            extend_index(ambiguous, "A cat. ", "scat catty bobcat", {"cat", "scat", "catty", "bobcat"})

    def test_small_paired_continuation_restores_and_activates_without_reset(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parent_id = chat_fixture(root)
            directory = root / ".data/neural-unit-benchmark" / parent_id
            prior = "A returns unplugged returns unplugged. " * 2
            training = "amazingly returns unplugged. " * 3
            baseline_path = directory / "17-baseline-passes-4.json"
            checkpoint_path = directory / "17-layers_01-passes-4.json"
            saved = json.loads(checkpoint_path.read_text())
            baseline = {**saved, "index": None}
            baseline_path.write_text(json.dumps(baseline))
            report_path = directory / "report.json"
            report = json.loads(report_path.read_text())
            report["source"]["training_sha256"] = text_hash(prior)
            report["runs"].append({"arm": "baseline", "seed": 17, "checkpoints": [saved["evidence"]]})
            report_path.write_text(json.dumps(report))
            (directory / "inputs.json").write_text(json.dumps({"training": prior}))
            inventory = root / ".data/possible-words-expanded-scowl-2020.12.07"
            inventory.mkdir()
            words = b"A\nreturns\nunplugged\namazingly\n"
            (inventory / "words.txt").write_bytes(words)
            (inventory / "manifest.json").write_text(json.dumps({
                "words_sha256": hashlib.sha256(words).hexdigest(), "entries": 4}))
            active = activate(root, parent_id)
            old_weights = checkpoint_path.read_bytes()
            with self.assertRaises(ValueError):
                run(root, acknowledge_review=False)
            with patch("neural_continuation.load_corpus", return_value=(training, {"fixture": True})):
                continued = run(root, acknowledge_review=True)
            self.assertEqual(continued["status"], "completed")
            self.assertEqual(status(root)["checkpoint_sha256"], active["checkpoint_sha256"])
            self.assertEqual(continued["continuation"]["parent_checkpoint_sha256"]["layers_01"],
                             active["checkpoint_sha256"])
            target = root / ".data/neural-unit-benchmark" / continued["run_id"]
            inputs = json.loads((target / "inputs.json").read_text())
            index = CandidateIndex.restore(inputs["index"])
            extracted = index.extract(training)
            for entry in continued["runs"]:
                model = NeuralModel(saved["characters"])
                model.weights = deepcopy(saved["weights"])
                model = extend_alphabet(model, list(training))
                selected = CandidateIndex.restore(index.snapshot()) if entry["arm"] == "layers_01" else None
                schedule = reinforced_schedule(model, training, extracted, remainder=selected is not None)
                for passes in range(1, 5):
                    if selected:
                        selected.learn(extracted["pieces"])
                    for context, token, _ in schedule:
                        model.update(context, token)
                    if passes in (1, 2, 4):
                        checkpoint = json.loads((target / f"17-{entry['arm']}-passes-{passes}.json").read_text())
                        self.assertEqual(model.weights, checkpoint["weights"])
                        self.assertEqual(checkpoint["index"], selected.snapshot() if selected else None)
                        measured = evaluate(model, selected, inputs["probes"])
                        for key, value in measured.items():
                            self.assertEqual(checkpoint["evidence"][key], value)
                        self.assertEqual(checkpoint["evidence"]["acts_retention"],
                                         evaluate(model, selected, inputs["retention_probes"]))
            chosen = activate(root, continued["run_id"])
            self.assertEqual(chosen["run_id"], continued["run_id"])
            self.assertEqual(chosen["parameters"], continued["runs"][0]["parameters"])
            self.assertTrue(generate(root, {"prompt": "The "})["text"])
            self.assertEqual(checkpoint_path.read_bytes(), old_weights)


if __name__ == "__main__":
    unittest.main()
