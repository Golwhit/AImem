import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import neural_skeletons
from neural_skeletons import Possible, StrictSkeletons, skeletons, upper_bound


DICTIONARY = {"that", "teat", "and", "add", "aid", "said", "sand", "sad", "with", "tat", "them", "then"}
LEARNING = "and that said with them. " * 6 + "and add that. "


class SkeletonTests(unittest.TestCase):
    def test_skeleton_kinds(self):
        self.assertIn("th*t", skeletons("that", "hidden1"))
        self.assertIn("t**t", skeletons("that", "hidden2"))
        self.assertEqual(skeletons("that", "dropped1"), {"hat", "tat", "tht", "tha"})

    def test_possible_words_decide_uniqueness(self):
        possible = Possible(DICTIONARY)
        self.assertEqual(possible.fits("th*t", "hidden1"), {"that"})
        self.assertIsNone(possible.fits("t*at", "hidden1"))  # teat
        self.assertIsNone(possible.fits("a*d", "hidden1"))  # add, aid
        self.assertIsNone(possible.fits("tat", "dropped1"))  # tat is itself a word
        self.assertEqual(possible.fits("tht", "dropped1"), {"that"})
        self.assertIsNone(possible.fits("the*", "hidden1"))  # them, then
        self.assertIsNone(possible.fits("sa**", "hidden2"))  # said, sand

    def test_strict_versus_light(self):
        states = StrictSkeletons(LEARNING, DICTIONARY, support=5)
        self.assertEqual(states.strict["hidden1"]["th*t"], "that")
        self.assertNotIn("sa*d", states.strict["hidden1"])
        self.assertEqual(states.light["hidden1"]["sa*d"], "said")  # only the learning text resolves it
        self.assertIn("sa*d w*th", states.light_phrases)
        self.assertNotIn("a*d th*t", states.light_phrases)  # "add that" was also seen
        self.assertNotIn("w*th th*m", states.light_phrases)  # both words already strict
        result = states.check("sand with that")
        self.assertEqual(result["strict"]["wrong"], 0)
        self.assertGreater(result["light"]["wrong"], 0)  # sa*d -> said, but the text said sand

    def test_names_and_capitals_are_handed_on(self):
        learning = "Mary said that. " * 6 + "with them. " * 6
        states = StrictSkeletons(learning, DICTIONARY | {"mary"}, support=5)
        self.assertIn("mary", states.names)
        self.assertNotIn("m*ry", states.strict["hidden1"])
        result = states.check("That is not claimed but that is")
        self.assertGreater(result["strict"]["kinds"]["hidden1"]["handed_on_capitalized"], 0)
        self.assertEqual(neural_skeletons.words_of("my father’ and don’t"), ["my", "father", "and", "don't"])

    def test_regular_forms_veto_and_hidden_first_letter(self):
        self.assertIn("revealer", neural_skeletons.regular_forms({"reveal"}))
        learning = "it was revealed. " * 6
        states = StrictSkeletons(learning, {"it", "was", "reveal"}, support=5)
        self.assertNotIn("reveale*", states.strict["hidden1"])  # "revealer" is a possible regular form
        self.assertIn("rev*aled", states.strict["hidden1"])
        self.assertFalse(any(s.startswith("*") for s in states.strict["hidden1"]))

    def test_upper_bound(self):
        self.assertAlmostEqual(upper_bound(0, 300_000), 1 - 0.05 ** (1 / 300_000), places=9)
        self.assertLess(upper_bound(0, 300_000), 1e-5)
        self.assertGreater(upper_bound(1, 300_000), upper_bound(0, 300_000))

    def test_locked_books_need_preregistration(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                archive.writestr("engwebp_002_GEN_01_read.txt", "Genesis.\nChapter 1.\nIn the beginning.\n")
            raw = buffer.getvalue()
            target = root / neural_skeletons.BIBLE_DIRECTORY
            target.mkdir(parents=True)
            (target / "engwebp_readaloud.zip").write_bytes(raw)
            import hashlib
            (target / "manifest.json").write_text(json.dumps({"zip_sha256": hashlib.sha256(raw).hexdigest(), "sources": []}))
            with self.assertRaisesRegex(ValueError, "pre-registration"):
                neural_skeletons.load(root, ("locked",))
            (root / neural_skeletons.SKELETON_DIRECTORY).mkdir(parents=True)
            (root / neural_skeletons.SKELETON_DIRECTORY / "locked-evaluation.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "already evaluated"):
                neural_skeletons.evaluate_locked(root)


if __name__ == "__main__":
    unittest.main()
