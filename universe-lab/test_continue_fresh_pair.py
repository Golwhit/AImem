import unittest

from continue_fresh_pair import previous_details


class ContinueFreshPairTests(unittest.TestCase):
    def test_accepts_validated_fresh_and_continued_checkpoints(self):
        scores = {"combined": {"pairs": 1}}
        retention = {"feedback": {"exact": 1, "examples": 1}}
        fresh = {"kind": "private_fresh_vs_active_candidate_mixture",
                 "validation": {"fresh_final_weights_replay_exact": True, "live_selection_unchanged": True},
                 "parent_checkpoint_sha256": "active", "updates": 17,
                 "pair_scores": {"fresh": scores}, "retention": {"fresh": retention}}
        continued = {"kind": "private_fresh_model_continuation",
                     "validation": {"continued_weights_replay_exact": True,
                                    "starting_state_matches_saved_checkpoint": True,
                                    "previous_scores_reproduced": True, "live_selection_unchanged": True},
                     "active_checkpoint_sha256": "active", "total_updates": 34,
                     "continued": {"pair_scores": scores, "retention": retention}}
        self.assertEqual(previous_details(fresh), ("active", 17, scores, retention))
        self.assertEqual(previous_details(continued), ("active", 34, scores, retention))
        continued["validation"]["starting_state_matches_saved_checkpoint"] = False
        with self.assertRaises(ValueError):
            previous_details(continued)


if __name__ == "__main__":
    unittest.main()
