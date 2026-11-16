import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from universe import attempt_connections, connections, execute, new_bond, pattern_groups, recorded_bonds, replay_pattern, step


class UniverseTests(unittest.TestCase):
    def test_batch_matches_individual_steps_and_stops_after_last_cycle(self):
        for mode, fuel in (("legacy", 4), ("legacy", 14), ("three-quark", 4), ("three-quark", 14)):
            with self.subTest(mode=mode, fuel=fuel), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                batch_path, single_path = root / "batch.json", root / "single.json"
                initial = {"schema": 1, "universe": None}
                with patch("universe.secrets.token_hex", return_value="batch-test"):
                    step(initial, "spark", {"fuel": fuel, "bond_mode": mode})
                initial["universe"]["bond_seed"] = "batch-test"
                for path in (batch_path, single_path):
                    path.write_text(json.dumps(initial), encoding="utf-8")
                start = execute("read", {}, batch_path)
                batched = execute("step", {"expected_revision": start["revision"], "steps": 50}, batch_path)
                if batched["state"]["universe"]["energy"] > 0 or batched["state"]["universe"]["phase"] != "predict":
                    batched = execute("step", {"expected_revision": batched["revision"], "steps": 50}, batch_path)
                current = execute("read", {}, single_path)
                for _ in range(4 * (fuel - 1)):
                    current = execute("step", {"expected_revision": current["revision"]}, single_path)
                self.assertEqual(batched["state"], current["state"])
                self.assertEqual(batched["bonds"], current["bonds"])
                self.assertEqual(batched["groups"], current["groups"])
                self.assertEqual(batched["state"]["universe"]["phase"], "predict")
                self.assertEqual(batched["state"]["universe"]["energy"], 0)
                with self.assertRaisesRegex(ValueError, "No energy remains"):
                    execute("step", {"expected_revision": batched["revision"], "steps": 50}, batch_path)

    def test_batch_rejects_bad_counts_and_stale_revisions_without_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.json"
            initial = {"schema": 1, "universe": None}
            step(initial, "spark", {"fuel": 3, "bond_mode": "three-quark"})
            path.write_text(json.dumps(initial), encoding="utf-8")
            current = execute("read", {}, path)
            for count in (0, 51, True, 1.5, "2"):
                with self.subTest(count=count), self.assertRaisesRegex(ValueError, "Steps must be"):
                    execute("step", {"expected_revision": current["revision"], "steps": count}, path)
                self.assertEqual(json.loads(path.read_text(encoding="utf-8")), initial)
            with self.assertRaisesRegex(ValueError, "universe changed"):
                execute("step", {"expected_revision": "stale", "steps": 20}, path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), initial)

    def test_three_quark_candidates_are_disjoint_and_freeze(self):
        state = {"schema": 1, "universe": None}
        with patch("universe.secrets.token_hex", return_value="pattern-test"):
            step(state, "spark", {"fuel": 25, "bond_mode": "three-quark"})
        for _ in range(96):
            step(state, "step", {})
        world = state["universe"]
        pattern = replay_pattern(world)
        groups = pattern_groups(pattern)
        self.assertEqual(sorted(member for group in groups for member in group["members"]), list(range(1, 25)))
        for group in groups:
            members = group["members"]
            self.assertLessEqual(len(members), 3)
            self.assertEqual(group["complete"], len(members) == 3)
            self.assertTrue(all(bond["partner"] in members for bond in pattern["bonds"].values()
                                if bond["quark"] in members))
            if group["complete"]:
                history = [attempt for event in world["events"] for attempt in event.get("pattern_attempts", [])]
                internal = [attempt for attempt in history if attempt["scope"] == "internal" and attempt["quark"] in members]
                self.assertEqual([attempt["quark"] for attempt in internal], members)
                self.assertTrue(all(attempt["partner"] in members for attempt in internal))
                last = history.index(internal[-1])
                self.assertTrue(all(attempt["quark"] not in members and attempt["partner"] not in members
                                    for attempt in history[last + 1:]))
        first = next(event for event in world["events"] if event["stage"] == "reveal")
        self.assertEqual(first["pattern_attempts"][0]["result"], "alone")
        self.assertEqual(first["pattern_attempts"][0]["quark"], 1)

    def test_three_quark_save_replays_and_rejects_changed_attempts(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.json"
            path.write_text('{"schema": 1, "universe": null}', encoding="utf-8")
            def act(action, **payload):
                current = execute("read", {}, path)
                return execute(action, {"expected_revision": current["revision"], **payload}, path)
            with patch("universe.secrets.token_hex", return_value="pattern-test"):
                act("spark", fuel=10, bond_mode="three-quark")
            for _ in range(12):
                act("step")
            saved = act("save", name="candidate")
            identifier = saved["experiments"][0]["id"]
            for _ in range(12):
                expected = act("step")
            act("reload", id=identifier, disposition="discard", confirm=True)
            for _ in range(12):
                replayed = act("step")
            self.assertEqual(replayed["state"], expected["state"])
            self.assertEqual(replayed["bonds"], expected["bonds"])
            self.assertEqual(replayed["groups"], expected["groups"])
            changed = replayed["state"]
            changed["universe"]["events"][-1]["pattern_attempts"].append({"quark": 1})
            with self.assertRaisesRegex(ValueError, "Invalid three-quark attempt history"):
                connections(changed)

    def test_first_popout_is_always_free_and_second_chooses_it(self):
        state = {"schema": 1, "universe": None}
        step(state, "spark", {"fuel": 4})
        step(state, "step", {})
        step(state, "step", {})
        world = state["universe"]
        first = world["events"][-1]
        self.assertEqual(first["bond"], {"partner": None, "outcome": "first", "outgoing": 0.0, "incoming": 0.0})
        self.assertNotIn("bond_seed", world)
        for _ in range(4):
            step(state, "step", {})
        second = world["events"][-1]
        self.assertEqual(second["bond"]["partner"], 1)
        self.assertIn(second["bond"]["outcome"], ("outgoing", "incoming", "both", "free"))
        self.assertIn("bond_seed", world)

    def test_four_outcomes_include_same_bit_partners_and_strengths(self):
        world = {"bond_seed": 17, "events": [
            {"stage": "spark", "bit": 1},
            {"stage": "reveal", "bit": 0},
            {"stage": "reveal", "bit": 1},
            {"stage": "reveal", "bit": 0},
        ]}
        strengths = {"outgoing": (0.5, 0.0), "incoming": (0.0, 0.5),
                     "both": (0.25, 0.25), "free": (0.0, 0.0)}
        for partner in (1, 2):
            for outcome, expected in strengths.items():
                with self.subTest(partner=partner, outcome=outcome):
                    with patch("universe.random.Random") as generator:
                        generator.return_value.randrange.return_value = partner
                        generator.return_value.choice.return_value = outcome
                        bond = new_bond(world)
                        generator.return_value.randrange.assert_called_once_with(1, 4)
                        self.assertEqual(set(generator.return_value.choice.call_args.args[0]), set(strengths))
                    self.assertEqual(bond["partner"], partner)
                    self.assertEqual((bond["outgoing"], bond["incoming"]), expected)
        self.assertEqual(world["events"][1]["bit"], 0)
        self.assertEqual(world["events"][2]["bit"], 1)

    def test_saved_quark_choices_and_future_randomness_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.json"
            path.write_text('{"schema": 1, "universe": null}', encoding="utf-8")
            def act(action, **payload):
                current = execute("read", {}, path)
                return execute(action, {"expected_revision": current["revision"], **payload}, path)
            act("spark", fuel=5)
            for _ in range(8):
                act("step")
            saved = act("save", name="quarks")
            identifier = saved["experiments"][0]["id"]
            original = saved["state"]
            for _ in range(4):
                expected = act("step")["state"]
            restored = act("reload", id=identifier, disposition="discard", confirm=True)
            self.assertEqual(restored["state"], original)
            for _ in range(4):
                replayed = act("step")["state"]
            self.assertEqual(replayed, expected)

    def test_quarks_can_bond_to_same_bit_quarks(self):
        state = {"schema": 1, "universe": None}
        step(state, "spark", {"fuel": 5})
        def bond(partner, outcome):
            return {"partner": partner, "outcome": outcome,
                    "outgoing": 0.5 if outcome == "outgoing" else 0.0, "incoming": 0.0}
        with patch("universe.new_bond", side_effect=[
                bond(None, "first"), bond(1, "outgoing"),
                bond(1, "outgoing"), bond(2, "outgoing")]):
            for _ in range(16):
                step(state, "step", {})
        bits = [event for event in state["universe"]["events"] if event["stage"] in ("spark", "reveal")]
        self.assertEqual((bits[3]["bit"], bits[bits[3]["bond"]["partner"]]["bit"]), (0, 0))
        self.assertEqual((bits[4]["bit"], bits[bits[4]["bond"]["partner"]]["bit"]), (1, 1))
        connections(state)

    def test_every_popout_attempts_each_phase_and_existing_bonds_are_kept(self):
        state = {"schema": 1, "universe": None}
        step(state, "spark", {"fuel": 4})
        step(state, "step", {})
        step(state, "step", {})
        world = state["universe"]
        self.assertEqual(world["events"][-1]["bond"]["outcome"], "first")
        self.assertEqual(world["events"][-1]["attempts"], [])
        step(state, "step", {})
        self.assertEqual(world["events"][-1]["attempts"][0]["result"], "alone")
        step(state, "step", {})
        step(state, "step", {})
        with patch("universe.random.Random") as generator:
            generator.return_value.randrange.return_value = 1
            generator.return_value.choice.return_value = "free"
            step(state, "step", {})
            self.assertEqual(world["events"][-1]["bond"]["outcome"], "free")
            self.assertEqual(world["events"][-1]["attempts"][0]["result"], "free")
            generator.return_value.choice.return_value = "outgoing"
            step(state, "step", {})
            attempts = world["events"][-1]["attempts"]
            self.assertEqual([(a["quark"], a["partner"], a["result"]) for a in attempts],
                             [(1, 2, "formed"), (2, 1, "existing")])
            bond = list(recorded_bonds(world).values())[0].copy()
            generator.return_value.choice.return_value = "both"
            step(state, "step", {})
            self.assertEqual([a["result"] for a in world["events"][-1]["attempts"]],
                             ["existing", "existing"])
            self.assertEqual(list(recorded_bonds(world).values()), [bond])
        connections(state)

    def test_corrupt_attempts_are_rejected(self):
        state = {"schema": 1, "universe": None}
        step(state, "spark", {"fuel": 2})
        step(state, "step", {})
        step(state, "step", {})
        step(state, "step", {})
        state["universe"]["events"][-1]["attempts"][0]["partner"] = 0
        with self.assertRaisesRegex(ValueError, "Invalid quark attempt"):
            connections(state)

    def test_multiple_bonds_and_same_bit_partners_are_recorded(self):
        world = {"bond_seed": "test", "events": [
            {"stage": "spark", "bit": 1},
            {"stage": "reveal", "bit": 0},
            {"stage": "reveal", "bit": 1},
            {"stage": "reveal", "bit": 0},
            {"stage": "predict"},
        ]}
        bonds = {}
        with patch("universe.random.Random") as generator:
            generator.return_value.randrange.side_effect = [2, 2, 1]
            generator.return_value.choice.side_effect = ["outgoing", "incoming", "both"]
            attempt_connections(world, bonds, False)
        attempts = world["events"][-1]["attempts"]
        self.assertEqual([(a["quark"], a["partner"], a["result"]) for a in attempts],
                         [(1, 3, "formed"), (2, 3, "formed"), (3, 1, "existing")])
        self.assertEqual(recorded_bonds(world), bonds)
        self.assertEqual(len(bonds), 2)

    def test_old_quarks_remain_without_invented_bonds(self):
        state = {"schema": 1, "universe": {"events": [
            {"stage": "spark", "bit": 1}, {"stage": "reveal", "bit": 0},
            {"stage": "reveal", "bit": 1},
        ]}}
        before = json.dumps(state)
        self.assertEqual(connections(state)["counts"]["01"], 1)
        self.assertEqual(json.dumps(state), before)
        with patch("universe.secrets.token_hex", return_value="0123456789abcdef"):
            bond = new_bond(state["universe"])
        self.assertIn(bond["partner"], (1, 2))
        self.assertNotIn("bond", state["universe"]["events"][1])
        self.assertNotIn("bond", state["universe"]["events"][2])

    def test_old_history_continues_with_new_attempts(self):
        state = {"schema": 1, "universe": None}
        step(state, "spark", {"fuel": 5})
        for _ in range(8):
            step(state, "step", {})
        world = state["universe"]
        for event in world["events"]:
            event.pop("attempts", None)
        old_bonds = recorded_bonds(world).copy()
        step(state, "step", {})
        self.assertEqual(len(world["events"][-1]["attempts"]), 2)
        for pair, bond in old_bonds.items():
            self.assertEqual(recorded_bonds(world)[pair], bond)
        connections(state)

    def test_invalid_recorded_bonds_are_rejected(self):
        state = {"schema": 1, "universe": {"events": [
            {"stage": "spark", "bit": 1},
            {"stage": "reveal", "bit": 0, "bond": {"partner": None, "outcome": "first", "outgoing": 0.0, "incoming": 0.0}},
            {"stage": "reveal", "bit": 1, "bond": {"partner": 0, "outcome": "outgoing", "outgoing": 0.5, "incoming": 0.0}},
        ]}}
        with self.assertRaisesRegex(ValueError, "Invalid quark bond"):
            connections(state)
        state["universe"]["events"][-1]["bond"]["partner"] = 1
        state["universe"]["events"][-1]["bond"]["outgoing"] = 0.25
        with self.assertRaisesRegex(ValueError, "Invalid quark bond"):
            connections(state)

    def test_connections_follow_actual_reveals_not_guesses_or_other_steps(self):
        state = {"schema": 1, "universe": None}
        self.assertEqual(connections(state), {"seen": {"0": False, "1": False},
                                             "counts": {"00": 0, "01": 0, "10": 0, "11": 0}})
        step(state, "spark", {"fuel": 5})
        self.assertEqual(connections(state)["seen"], {"0": False, "1": True})
        step(state, "step", {})
        self.assertEqual(connections(state)["counts"]["10"], 0)
        step(state, "step", {})
        self.assertEqual(connections(state)["counts"], {"00": 0, "01": 0, "10": 1, "11": 0})
        for _ in range(4):
            step(state, "step", {})
        self.assertEqual(connections(state)["counts"], {"00": 0, "01": 1, "10": 1, "11": 0})
        for _ in range(4):
            step(state, "step", {})
        self.assertEqual(connections(state)["counts"], {"00": 0, "01": 1, "10": 2, "11": 0})

    def test_connections_handle_reverse_only_and_same_bit_observations(self):
        events = [{"stage": "spark", "bit": 0}, {"stage": "reveal", "bit": 1},
                  {"stage": "reveal", "bit": 1}]
        state = {"schema": 1, "universe": {"events": events}}
        self.assertEqual(connections(state)["counts"], {"00": 0, "01": 1, "10": 0, "11": 1})
        self.assertEqual(len(events), 3)

    def test_spark_is_first_bit_and_cannot_overwrite_existing_state(self):
        state = {"schema": 1, "universe": None}
        self.assertIsNone(step(state, "read", {})["universe"])
        with self.assertRaisesRegex(ValueError, "Spark"):
            step(state, "step", {})
        with self.assertRaisesRegex(ValueError, "Fuel"):
            step(state, "spark", {"fuel": 0})
        self.assertIsNone(state["universe"])
        step(state, "spark", {"fuel": 3})
        world = state["universe"]
        self.assertEqual(world["events"], [{"stage": "spark", "bit": 1, "energy": 2}])
        self.assertEqual(world["weights"], {"bias": 0.0, "link": 0.0})
        with self.assertRaisesRegex(ValueError, "already sparked"):
            step(state, "spark", {"fuel": 3})
        self.assertEqual(world["events"][0]["bit"], 1)

    def test_prediction_precedes_reveal_and_batches_grow(self):
        state = {"schema": 1, "universe": None}
        step(state, "spark", {"fuel": 13})
        world = state["universe"]
        step(state, "step", {})
        pending = world["pending"]
        self.assertEqual(world["phase"], "reveal")
        self.assertNotIn("actual", pending)
        self.assertEqual(pending["probability"], 0.5)
        self.assertEqual(pending["guess"], 1)
        step(state, "step", {})
        self.assertEqual(pending["actual"], 0)
        self.assertEqual(world["weights"], {"bias": 0.0, "link": 0.0})
        step(state, "step", {})
        self.assertEqual(pending["error"], -0.5)
        step(state, "step", {})
        self.assertAlmostEqual(world["weights"]["bias"], -0.1)
        self.assertAlmostEqual(world["weights"]["link"], -0.1)
        for _ in range(44):
            step(state, "step", {})
        revealed = [event for event in world["events"] if event["stage"] == "reveal"]
        self.assertEqual([event["bit"] for event in revealed], [0, 1] * 6)
        self.assertEqual([event["batch"] for event in revealed], [1] * 2 + [2] * 4 + [3] * 6)
        self.assertEqual(world["batch"], 4)
        self.assertEqual(world["energy"], 0)
        self.assertEqual(world["phase"], "predict")

    def test_last_paid_cycle_finishes_and_fuel_can_be_added(self):
        state = {"schema": 1, "universe": None}
        step(state, "spark", {"fuel": 2})
        step(state, "step", {})
        world = state["universe"]
        self.assertEqual(world["energy"], 0)
        for _ in range(3):
            step(state, "step", {})
        self.assertEqual(world["phase"], "predict")
        with self.assertRaisesRegex(ValueError, "No energy"):
            step(state, "step", {})
        step(state, "fuel", {"fuel": 4})
        self.assertEqual(world["energy"], 4)
        self.assertEqual(world["events"][-1]["stage"], "fuel")
        step(state, "step", {})
        self.assertEqual(world["energy"], 3)

    def test_file_state_is_persistent_and_concurrent_steps_are_serialized(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.json"
            path.write_text('{"schema": 1, "universe": null}', encoding="utf-8")
            initial = execute("read", {}, path)
            execute("spark", {"fuel": 10, "expected_revision": initial["revision"]}, path)
            def do_step(_):
                while True:
                    current = execute("read", {}, path)
                    try:
                        return execute("step", {"expected_revision": current["revision"]}, path)
                    except ValueError as error:
                        if "universe changed" not in str(error):
                            raise
            with ThreadPoolExecutor(max_workers=5) as pool:
                list(pool.map(do_step, range(20)))
            world = execute("read", {}, path)["state"]["universe"]
            self.assertEqual(len(world["events"]), 21)
            self.assertEqual(world["energy"], 4)
            self.assertEqual(world["phase"], "predict")
            self.assertEqual(json.loads(path.read_text())["universe"], world)
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "Fuel"):
                execute("fuel", {"fuel": True, "expected_revision": execute("read", {}, path)["revision"]}, path)
            self.assertEqual(path.read_bytes(), before)

    def test_save_reload_mid_phase_and_save_before_reset(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.json"
            path.write_text('{"schema": 1, "universe": null}', encoding="utf-8")
            def act(action, **payload):
                current = execute("read", {}, path)
                return execute(action, {"expected_revision": current["revision"], **payload}, path)
            act("spark", fuel=4)
            saved_state = act("step")["state"]
            self.assertEqual(saved_state["universe"]["phase"], "reveal")
            saved = act("save", name="first")
            self.assertEqual(saved["state"], saved_state)
            first = saved["experiments"][0]["id"]
            changed = act("step")["state"]
            self.assertEqual(changed["universe"]["phase"], "measure")
            self.assertEqual(act("read")["connections"]["counts"]["10"], 1)
            reset = act("reset", disposition="save", confirm=True, name="second")
            self.assertIsNone(reset["state"]["universe"])
            second = next(item["id"] for item in reset["experiments"] if item["name"] == "second")
            reloaded = act("reload", id=first)
            self.assertEqual(reloaded["state"], saved_state)
            self.assertEqual(reloaded["connections"]["counts"]["10"], 0)
            restored = act("reload", id=second, disposition="discard", confirm=True)
            self.assertEqual(restored["state"], changed)
            self.assertEqual(restored["connections"]["counts"]["10"], 1)
            self.assertEqual(act("step")["state"]["universe"]["phase"], "reweight")

    def test_replacements_deletion_and_invalid_ids_are_protected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.json"
            path.write_text('{"schema": 1, "universe": null}', encoding="utf-8")
            current = execute("read", {}, path)
            current = execute("spark", {"fuel": 3, "expected_revision": current["revision"]}, path)
            rev = current["revision"]
            with self.assertRaisesRegex(ValueError, "name"):
                execute("save", {"name": " ", "expected_revision": rev}, path)
            saved = execute("save", {"name": "keep", "expected_revision": rev}, path)
            identifier = saved["experiments"][0]["id"]
            before = path.read_bytes()
            for action, payload in (
                ("reset", {}),
                ("reset", {"disposition": "discard"}),
                ("reload", {"id": identifier}),
                ("delete", {"id": identifier}),
                ("reload", {"id": "../state", "disposition": "discard", "confirm": True}),
                ("delete", {"id": "../state", "confirm": True}),
            ):
                with self.subTest(action=action, payload=payload):
                    with self.assertRaises(ValueError):
                        execute(action, {"expected_revision": rev, **payload}, path)
                    self.assertEqual(path.read_bytes(), before)
            with patch("universe.save_archive", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    execute("reset", {"expected_revision": rev, "confirm": True, "disposition": "save", "name": "copy"}, path)
            self.assertEqual(path.read_bytes(), before)
            deleted = execute("delete", {"expected_revision": rev, "id": identifier, "confirm": True}, path)
            self.assertEqual(deleted["experiments"], [])
            self.assertEqual(path.read_bytes(), before)
            reset = execute("reset", {"expected_revision": rev, "disposition": "discard", "confirm": True}, path)
            self.assertIsNone(reset["state"]["universe"])

    def test_stale_actions_cannot_change_current_state(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.json"
            path.write_text('{"schema": 1, "universe": null}', encoding="utf-8")
            old = execute("read", {}, path)["revision"]
            current = execute("spark", {"fuel": 3, "expected_revision": old}, path)
            saved = execute("save", {"name": "kept", "expected_revision": current["revision"]}, path)
            identifier = saved["experiments"][0]["id"]
            before = path.read_bytes()
            for action, payload in (
                ("step", {}),
                ("save", {"name": "stale"}),
                ("reset", {"disposition": "discard", "confirm": True}),
                ("reload", {"id": identifier, "disposition": "discard", "confirm": True}),
                ("delete", {"id": identifier, "confirm": True}),
            ):
                with self.subTest(action=action):
                    with self.assertRaisesRegex(ValueError, "universe changed"):
                        execute(action, {"expected_revision": old, **payload}, path)
                    self.assertEqual(path.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, "Refresh"):
                execute("step", {}, path)
            self.assertEqual(len(execute("read", {}, path)["experiments"]), 1)

    def test_changed_archive_is_not_restored(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.json"
            path.write_text('{"schema": 1, "universe": null}', encoding="utf-8")
            current = execute("read", {}, path)
            current = execute("spark", {"fuel": 3, "expected_revision": current["revision"]}, path)
            saved = execute("save", {"name": "keep", "expected_revision": current["revision"]}, path)
            identifier = saved["experiments"][0]["id"]
            archive_path = path.parent / "experiments" / f"{identifier}.json"
            archive = json.loads(archive_path.read_text())
            archive["state"]["universe"]["energy"] = 999
            archive_path.write_text(json.dumps(archive), encoding="utf-8")
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "checksum"):
                execute("reload", {"expected_revision": current["revision"], "id": identifier, "disposition": "discard", "confirm": True}, path)
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
