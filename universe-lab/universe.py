"""A tiny, separate universe: spark, predict, reveal, measure, and learn."""

import argparse  # This reads the action chosen by the lab.
from datetime import datetime, timezone  # This lets saved experiments show when they were kept.
import fcntl  # This lets only one action change the saved universe at a time.
import hashlib  # This detects when a saved state has changed.
import json  # This turns the saved universe into readable text.
import math  # This supplies the exponential used for a probability.
import os  # This safely replaces the old saved state with the new one.
from pathlib import Path  # This helps us find files inside this folder.
import random  # This gives each new quark an equal chance to choose a partner and bond.
import secrets  # This starts each experiment with its own unpredictable random seed.
import sys  # This reads the lab's request and reports errors.
import tempfile  # This makes a temporary file beside the saved state.
from uuid import uuid4  # This gives each saved experiment its own safe name.

STATE_PATH = Path(__file__).resolve().with_name("state.json")  # The experiment's own memory lives here.
RATE = 0.2  # Each lesson changes the weights by a small, fixed amount.
MAX_FUEL = 1000  # One fuel change is limited so the lab stays usable.
MAX_STEP_BATCH = 50
BOND_STRENGTHS = {"outgoing": (0.5, 0.0), "incoming": (0.0, 0.5), "both": (0.25, 0.25), "free": (0.0, 0.0)}  # These are the four equally likely starting outcomes.


def fuel_amount(value):  # Check how much energy the experimenter supplied.
    if type(value) is not int or not 1 <= value <= MAX_FUEL:  # Whole numbers only, within the shown limit.
        raise ValueError(f"Fuel must be a whole number from 1 to {MAX_FUEL}")  # Explain a bad setting.
    return value  # Return the checked amount.


def new_bond(world):  # Let a new quark choose its first partner and connection.
    quarks = [event for event in world["events"] if event["stage"] in ("spark", "reveal")]  # Count actual quarks, including the original seeder.
    if len(quarks) == 1:  # The first pop-out has no earlier pop-out to meet.
        return {"partner": None, "outcome": "first", "outgoing": 0.0, "incoming": 0.0}  # It always floats freely.
    if "bond_seed" not in world:  # Older runs and new runs receive a seed only when one is needed.
        world["bond_seed"] = secrets.token_hex(16)  # Keep this seed in the run so a saved copy can continue reproducibly.
    chooser = random.Random(f'{world["bond_seed"]}:{len(quarks)}')  # Give this quark its own repeatable set of random choices.
    partner = chooser.randrange(1, len(quarks))  # Choose equally from earlier pop-outs, never the original seeder.
    outcome = chooser.choice(tuple(BOND_STRENGTHS))  # Give each of the four outcomes one equal chance.
    outgoing, incoming = BOND_STRENGTHS[outcome]  # Look up the strength in each direction.
    return {"partner": partner, "outcome": outcome, "outgoing": outgoing, "incoming": incoming}  # Record the actual choice, even when it stays free.


def empty_pattern():
    return {"bonds": {}, "groups": {}, "active": set(), "candidates": []}


def pattern_round(pattern, seed, event_index, newest):
    attempts = []

    def choose(actor, partners, scope):
        if not partners:
            attempts.append({"quark": actor, "partner": None, "outcome": "alone", "result": "alone", "scope": scope})
            return None
        chooser = random.Random(f"{seed}:three-quark:{event_index}:{scope}:{actor}")
        partner = chooser.choice(partners)
        outcome = chooser.choice(tuple(BOND_STRENGTHS))
        outgoing, incoming = BOND_STRENGTHS[outcome]
        key = tuple(sorted((actor, partner)))
        result = "existing" if key in pattern["bonds"] else "free" if outcome == "free" else "formed"
        attempts.append({"quark": actor, "partner": partner, "outcome": outcome, "outgoing": outgoing,
                         "incoming": incoming, "result": result, "scope": scope})
        if result == "formed":
            pattern["bonds"][key] = {"quark": actor, "partner": partner, "outcome": outcome,
                                      "outgoing": outgoing, "incoming": incoming}
        return partner if result == "formed" else None

    actors = sorted(pattern["active"])
    if newest is not None:
        actors = [newest] + [actor for actor in actors if actor != newest]
    for actor in actors:
        if actor not in pattern["active"]:
            continue
        group = pattern["groups"][actor]
        partners = [other for other in sorted(pattern["active"]) if other != actor and
                    (other in group or len(group) + len(pattern["groups"][other]) <= 3)]
        partner = choose(actor, partners, "external")
        if partner is None or partner in group:
            continue
        merged = tuple(sorted(group + pattern["groups"][partner]))
        for member in merged:
            pattern["groups"][member] = merged
        if len(merged) == 3:
            pattern["active"].difference_update(merged)
            for member in merged:
                choose(member, [other for other in merged if other != member], "internal")
            pattern["candidates"].append(merged)
    return attempts


def replay_pattern(world):
    seed = world.get("bond_seed")
    events = world["events"]
    if not isinstance(seed, str) or not seed or not events or events[0]["stage"] != "spark":
        raise ValueError("Invalid three-quark experiment")
    pattern = empty_pattern()
    quark_count = 0
    for index, event in enumerate(events):
        stage = event["stage"]
        if stage == "spark":
            if index != 0:
                raise ValueError("Invalid three-quark spark")
            quark_count += 1
        elif stage == "reveal":
            pattern["groups"][quark_count] = (quark_count,)
            pattern["active"].add(quark_count)
            quark_count += 1
        if stage in ("predict", "reveal", "measure", "reweight"):
            newest = quark_count - 1 if stage == "reveal" else None
            expected = pattern_round(pattern, seed, index, newest)
            if event.get("pattern_attempts") != expected:
                raise ValueError("Invalid three-quark attempt history")
        elif stage not in ("spark", "fuel"):
            raise ValueError("Invalid three-quark event")
    return pattern


def pattern_groups(pattern):
    groups = sorted(set(pattern["groups"].values()))
    candidates = set(pattern["candidates"])
    return [{"members": list(group), "complete": group in candidates} for group in groups]


def recorded_bonds(world):
    if world and world.get("bond_mode") == "three-quark":
        return replay_pattern(world)["bonds"]
    if world and world.get("bond_mode") not in (None, "legacy"):
        raise ValueError("Unknown universe bond mode")
    bonds = {}
    quark_count = 0
    for event in world["events"] if world else []:
        if event["stage"] in ("spark", "reveal"):
            actor = quark_count
            quark_count += 1
            bond = event.get("bond")
            if bond and bond["outcome"] not in ("first", "free"):
                key = tuple(sorted((actor, bond["partner"])))
                bonds[key] = {"quark": actor, **bond}
        attempts = event.get("attempts", [])
        if not isinstance(attempts, list):
            raise ValueError("Invalid quark attempts")
        if "attempts" in event:
            if event["stage"] not in ("predict", "reveal", "measure", "reweight"):
                raise ValueError("Invalid quark attempts")
            end = quark_count - (1 if event["stage"] == "reveal" else 0)
            actors = [attempt["quark"] for attempt in attempts]
            if actors != list(range(1, end)):
                raise ValueError("Invalid quark attempt order")
        for attempt in attempts:
            actor = attempt["quark"]
            partner = attempt["partner"]
            if type(actor) is not int or not 1 <= actor < quark_count:
                raise ValueError("Invalid quark attempt")
            if partner is None:
                if quark_count != 2 or attempt["outcome"] != "alone" or attempt["result"] != "alone":
                    raise ValueError("Invalid quark attempt")
                continue
            outcome = attempt["outcome"]
            if (type(partner) is not int or not 1 <= partner < quark_count or partner == actor
                    or outcome not in BOND_STRENGTHS
                    or (attempt["outgoing"], attempt["incoming"]) != BOND_STRENGTHS[outcome]):
                raise ValueError("Invalid quark attempt")
            key = tuple(sorted((actor, partner)))
            expected = "existing" if key in bonds else "free" if outcome == "free" else "formed"
            if attempt["result"] != expected:
                raise ValueError("Invalid quark attempt result")
            if expected == "formed":
                bonds[key] = {key: attempt[key] for key in ("quark", "partner", "outcome", "outgoing", "incoming")}
    return bonds


def attempt_connections(world, bonds, skip_new):
    quark_count = sum(event["stage"] in ("spark", "reveal") for event in world["events"])
    attempts = []
    for actor in range(1, quark_count - (1 if skip_new else 0)):
        if quark_count == 2:
            attempts.append({"quark": actor, "partner": None, "outcome": "alone", "result": "alone"})
            continue
        if "bond_seed" not in world:
            world["bond_seed"] = secrets.token_hex(16)
        chooser = random.Random(f'{world["bond_seed"]}:step:{len(world["events"])}:quark:{actor}')
        partner = chooser.randrange(1, quark_count - 1)
        if partner >= actor:
            partner += 1
        outcome = chooser.choice(tuple(BOND_STRENGTHS))
        outgoing, incoming = BOND_STRENGTHS[outcome]
        key = tuple(sorted((actor, partner)))
        result = "existing" if key in bonds else "free" if outcome == "free" else "formed"
        attempt = {"quark": actor, "partner": partner, "outcome": outcome,
                   "outgoing": outgoing, "incoming": incoming, "result": result}
        attempts.append(attempt)
        if result == "formed":
            bonds[key] = {key: attempt[key] for key in ("quark", "partner", "outcome", "outgoing", "incoming")}
    world["events"][-1]["attempts"] = attempts


def step(state, action, payload, pattern=None):  # Change one part of this tiny universe.
    world = state["universe"]  # Before the spark this is None, meaning no universe exists yet.
    if action == "read":  # Looking at the universe should not change it.
        return state  # Give the current state back unchanged.
    if action == "spark":  # The spark is our pretend big bang.
        if world is not None:  # A second spark would overwrite the first universe.
            raise ValueError("This universe has already sparked")  # Preserve what already happened.
        fuel = fuel_amount(payload.get("fuel"))  # The person chooses a finite starting energy.
        mode = payload.get("bond_mode", "legacy")
        if mode not in ("legacy", "three-quark"):
            raise ValueError("Unknown universe bond mode")
        state["universe"] = {"energy": fuel - 1, "phase": "predict", "previous": 1, "batch": 1, "position": 0, "weights": {"bias": 0.0, "link": 0.0}, "pending": None, "events": [{"stage": "spark", "bit": 1, "energy": fuel - 1}]}  # Spend one energy and create the first 1.
        if mode == "three-quark":
            state["universe"]["bond_mode"] = mode
            state["universe"]["bond_seed"] = secrets.token_hex(16)
        return state  # The first thing now exists.
    if world is None:  # Nothing else can happen before the first spark.
        raise ValueError("Spark the universe first")  # Tell the person what is missing.
    if action == "fuel":  # The experimenter may add more energy.
        added = fuel_amount(payload.get("fuel"))  # Check the amount before changing anything.
        world["energy"] += added  # Add energy without changing past events or weights.
        world["events"].append({"stage": "fuel", "added": added, "energy": world["energy"]})  # Remember that extra energy was supplied.
        return state  # The same universe can continue.
    if action != "step":  # Only the listed actions are understood.
        raise ValueError("Unknown universe action")  # Never silently invent an action.
    if world.get("bond_mode") == "three-quark" and pattern is None:
        pattern = replay_pattern(world)
    bonds = pattern["bonds"] if pattern is not None else recorded_bonds(world)
    is_reveal = world["phase"] == "reveal"
    if world["phase"] == "predict":  # The learner must guess before the seeder reveals a bit.
        if world["energy"] == 0:  # Starting another prediction costs one energy.
            raise ValueError("No energy remains; add fuel to continue")  # Do not start an unfinished prediction.
        signal = 1 if world["previous"] == 1 else -1  # Turn the previous bit into a positive or negative clue.
        weights = world["weights"].copy()  # Save the old weights so the lesson can show them.
        score = weights["bias"] + weights["link"] * signal  # Combine the weights with the clue.
        probability = 1 / (1 + math.exp(-score))  # Turn the score into a chance of seeing 1.
        guess = 1 if probability >= 0.5 else 0  # Pick the more likely bit; a tie picks 1.
        world["energy"] -= 1  # Pay once for this prediction and its complete learning cycle.
        world["pending"] = {"signal": signal, "probability": probability, "guess": guess, "before": weights}  # Save the guess without revealing the answer.
        world["events"].append({"stage": "predict", "previous": world["previous"], "guess": guess, "probability": probability, "batch": world["batch"], "energy": world["energy"]})  # Show the guess before the answer.
        world["phase"] = "reveal"  # Next time, the seeder will reveal the answer.
    elif world["phase"] == "reveal":  # The seeder follows the pattern we chose for this experiment.
        actual = world["position"] % 2  # Each pair is 0 followed by 1.
        bond = new_bond(world) if pattern is None else None  # Older runs retain their original initial choice.
        world["pending"]["actual"] = actual  # Save the answer beside the earlier guess.
        event = {"stage": "reveal", "bit": actual, "batch": world["batch"], "pair": world["position"] // 2 + 1, "energy": world["energy"]}
        if bond is not None:
            event["bond"] = bond
        world["events"].append(event)
        if bond is not None and bond["outcome"] not in ("first", "free"):
            actor = sum(event["stage"] in ("spark", "reveal") for event in world["events"]) - 1
            bonds[tuple(sorted((actor, bond["partner"])))] = {"quark": actor, **bond}
        world["phase"] = "measure"  # Next, compare the guess with the answer.
    elif world["phase"] == "measure":  # The error tells the learner which way to adjust.
        pending = world["pending"]  # Use the saved guess and revealed answer.
        pending["error"] = pending["actual"] - pending["probability"]  # Positive means the chance of 1 was too low.
        world["events"].append({"stage": "measure", "error": pending["error"], "correct": pending["guess"] == pending["actual"], "energy": world["energy"]})  # Show both the error and whether the guess matched.
        world["phase"] = "reweight"  # Next, learn from that error.
    elif world["phase"] == "reweight":  # Change the two weights using the saved error.
        pending = world["pending"]  # The same prediction, answer, and error belong together.
        world["weights"]["bias"] += RATE * pending["error"]  # Move the general preference toward the answer.
        world["weights"]["link"] += RATE * pending["error"] * pending["signal"]  # Adjust how much the previous bit matters.
        world["previous"] = pending["actual"]  # Remember the latest revealed bit for the next guess.
        world["events"].append({"stage": "reweight", "before": pending["before"], "after": world["weights"].copy(), "rate": RATE, "energy": world["energy"]})  # Keep the actual old and new weights.
        world["position"] += 1  # Move to the next place in the seeder's batch.
        if world["position"] == world["batch"] * 2:  # Batch one has one pair, batch two has two, and so on.
            world["batch"] += 1  # Start a batch containing one more pair.
            world["position"] = 0  # The new batch starts with 0.
        world["pending"] = None  # This learning cycle is complete.
        world["phase"] = "predict"  # Another cycle may start if energy remains.
    else:  # An unknown saved phase means the file cannot be trusted.
        raise ValueError("Unknown saved universe phase")  # Stop rather than guessing what to do.
    if pattern is None:
        attempt_connections(world, bonds, is_reveal)
    else:
        newest = None
        if is_reveal:
            newest = sum(event["stage"] in ("spark", "reveal") for event in world["events"]) - 1
            pattern["groups"][newest] = (newest,)
            pattern["active"].add(newest)
        world["events"][-1]["pattern_attempts"] = pattern_round(pattern, world["bond_seed"], len(world["events"]) - 1, newest)
    return state  # Return the changed universe so it can be saved.


def connections(state):  # Count which bit was followed by which other bit.
    seen = {"0": False, "1": False}  # At first neither kind of bit has appeared.
    counts = {"00": 0, "01": 0, "10": 0, "11": 0}  # Each direction has its own number of observations.
    previous = None  # There is no earlier bit before the spark.
    quark_count = 0  # Count quarks so a recorded partner can be checked.
    world = state["universe"]  # An empty universe has no events to examine.
    for event in world["events"] if world is not None else []:  # Walk through the actual history in order.
        if event["stage"] not in ("spark", "reveal"):  # Predictions and other steps do not create bits.
            continue  # Skip them without changing the last actual bit.
        bit = event["bit"]  # Take the bit that really appeared.
        if type(bit) is not int or bit not in (0, 1):  # Only real 0s and 1s belong in this experiment.
            raise ValueError("Invalid bit in universe history")  # Do not invent a connection for bad data.
        if "bond" in event:  # Older events have no bond choices and stay unconnected.
            bond = event["bond"]  # Read this quark's recorded choice.
            outcome = bond["outcome"]  # The outcome decides the two initial strengths.
            if outcome == "first":  # The first pop-out cannot choose an earlier pop-out.
                if quark_count != 1 or bond["partner"] is not None or (bond["outgoing"], bond["incoming"]) != (0.0, 0.0):  # Check its free-floating start.
                    raise ValueError("Invalid first quark bond")  # Stop rather than inventing a bond.
            elif outcome not in BOND_STRENGTHS or type(bond["partner"]) is not int or not 1 <= bond["partner"] < quark_count or (bond["outgoing"], bond["incoming"]) != BOND_STRENGTHS[outcome]:  # Check the earlier partner and both strengths.
                raise ValueError("Invalid quark bond")  # Do not display or restore an impossible choice.
        quark_count += 1  # This quark may be chosen by a later pop-out.
        seen[str(bit)] = True  # Remember that this kind of bit has appeared.
        if previous is not None:  # The first bit has nothing before it to connect from.
            counts[f"{previous}{bit}"] += 1  # Strengthen only the observed direction by one.
        previous = bit  # This bit will be the starting point for the next one.
    recorded_bonds(world)
    return {"seen": seen, "counts": counts}  # These counts describe the history, not learned weights.


def revision(state):  # Give this exact universe a short identity based on its contents.
    encoded = json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")  # Write the same state in a predictable order.
    return hashlib.sha256(encoded).hexdigest()  # A changed state has a different identity.


def write_json(path, value):  # Save a complete file without leaving a half-written state.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as temporary:  # Make a new file beside the old one.
        try:  # Clean up the temporary file if saving fails.
            json.dump(value, temporary, allow_nan=False, separators=(",", ":"))  # Refuse invalid numbers and save the complete value.
            temporary.flush()  # Send Python's buffered text to the file.
            os.fsync(temporary.fileno())  # Ask the operating system to finish writing it.
            os.replace(temporary.name, path)  # Swap in the complete file all at once.
        finally:  # This runs whether saving worked or failed.
            Path(temporary.name).unlink(missing_ok=True)  # Remove only this temporary file if it remains.


def archive_path(path, identifier):  # Find one saved experiment without allowing paths from the browser.
    if not isinstance(identifier, str) or len(identifier) != 32 or any(char not in "0123456789abcdef" for char in identifier):  # Only our own IDs are valid.
        raise ValueError("Invalid saved experiment ID")  # Reject anything that could point elsewhere.
    return path.parent / "experiments" / f"{identifier}.json"  # Saved runs live only in this folder.


def read_archive(path, identifier):  # Read and check one saved experiment.
    archive = json.loads(archive_path(path, identifier).read_text(encoding="utf-8"))  # Load the selected file.
    if archive["id"] != identifier or archive["state"]["schema"] != 1 or archive["state"]["universe"] is None:  # Check that it is the requested complete run.
        raise ValueError("Invalid saved experiment")  # Do not restore an unrelated or empty file.
    if revision(archive["state"]) != archive["revision"]:  # Compare the state to its saved fingerprint.
        raise ValueError("Saved experiment checksum does not match")  # Do not restore changed or damaged data.
    return archive  # Return the checked experiment.


def save_archive(path, state, label):  # Keep a copy without changing the current universe.
    if state["universe"] is None:  # There is no run to save before the spark.
        raise ValueError("There is no experiment to save")  # Explain why saving cannot happen.
    if not isinstance(label, str) or not label.strip() or len(label.strip()) > 80:  # A short name helps people find a run later.
        raise ValueError("Experiment name must contain 1 to 80 characters")  # Reject an empty or oversized name.
    identifier = uuid4().hex  # Make a new identity rather than overwriting an earlier run.
    destination = archive_path(path, identifier)  # Put the saved run in the experiment folder.
    destination.parent.mkdir(exist_ok=True)  # Create the small archive folder only when it is needed.
    if destination.exists():  # Never overwrite an existing saved experiment.
        raise FileExistsError("Saved experiment ID already exists")  # Stop and keep the original file.
    archive = {"id": identifier, "name": label.strip(), "saved_at": datetime.now(timezone.utc).isoformat(), "revision": revision(state), "state": state}  # Store the whole universe and its fingerprint.
    write_json(destination, archive)  # Save it before changing the current universe.


def list_archives(path):  # Describe the saved runs without sending their full histories.
    directory = path.parent / "experiments"  # All saved runs are kept together.
    if not directory.exists():  # Before the first save there is no archive folder.
        return []  # Show an empty list.
    archives = [read_archive(path, file.stem) for file in directory.glob("*.json")]  # Check every saved run before listing it.
    archives.sort(key=lambda item: (item["saved_at"], item["id"]), reverse=True)  # Put the newest saves first.
    return [{"id": item["id"], "name": item["name"], "saved_at": item["saved_at"]} for item in archives]  # Send only the details needed for the menu.


def execute(action, payload, path=STATE_PATH):  # Load, change, and save exactly one action.
    lock_path = path.with_suffix(".lock")  # Keep the lock in the experiment folder.
    with lock_path.open("a") as lock:  # Open a shared lock file for all lab requests.
        fcntl.flock(lock, fcntl.LOCK_EX)  # Wait until another request finishes saving.
        state = json.loads(path.read_text(encoding="utf-8"))  # Read the saved universe, not the chat model.
        if state.get("schema") != 1 or "universe" not in state:  # Reject a file with the wrong shape.
            raise ValueError("Invalid universe state")  # Do not replace an unknown file.
        connections(state)  # Check the existing bit history before making any change.
        expected = payload.get("expected_revision")  # The page can identify the state it was showing.
        if action != "read" and not isinstance(expected, str):  # Every change needs a known starting point.
            raise ValueError("Refresh the lab before changing the universe")  # Do not guess which version the person meant.
        if expected is not None and expected != revision(state):  # Another page may have changed the universe meanwhile.
            raise ValueError("The universe changed; refresh the lab and try again")  # Protect the newer state.
        if action != "read":  # Check the archive before changing anything.
            list_archives(path)  # A damaged saved run must not cause a change to look like a failure afterward.
        if action == "save":  # Keep a named copy and leave the current run active.
            save_archive(path, state, payload.get("name"))  # Save all weights, events, and the current phase.
        elif action in ("reset", "reload"):  # Both actions replace the current universe.
            replacement = {"schema": 1, "universe": None} if action == "reset" else read_archive(path, payload.get("id"))["state"]  # Find the empty or saved state first.
            connections(replacement)  # Check its bit history before replacing the current run.
            if state["universe"] is not None:  # A live run must not disappear by accident.
                if payload.get("confirm") is not True or payload.get("disposition") not in ("save", "discard"):  # Require an explicit decision.
                    raise ValueError("Save or explicitly discard the current experiment first")  # Leave it untouched.
                if payload["disposition"] == "save":  # The person chose to keep the current run.
                    save_archive(path, state, payload.get("name"))  # Save the old state before replacing it.
            state = replacement  # Move to the empty or restored universe.
            write_json(path, state)  # Save the replacement only after any requested archive succeeds.
        elif action == "delete":  # Deleting a saved copy is separate from changing the current universe.
            if payload.get("confirm") is not True:  # The person must explicitly confirm deletion.
                raise ValueError("Confirm deletion of the saved experiment")  # Keep it otherwise.
            identifier = payload.get("id")  # Identify exactly one saved experiment.
            read_archive(path, identifier)  # Check the selected file before deleting it.
            archive_path(path, identifier).unlink()  # Delete only the selected saved copy.
        elif action == "step":
            count = payload.get("steps", 1)
            if type(count) is not int or not 1 <= count <= MAX_STEP_BATCH:
                raise ValueError(f"Steps must be a whole number from 1 to {MAX_STEP_BATCH}")
            world = state["universe"]
            pattern = replay_pattern(world) if world and world.get("bond_mode") == "three-quark" else None
            for index in range(count):
                if index and world["energy"] == 0 and world["phase"] == "predict":
                    break
                step(state, action, payload, pattern)
            write_json(path, state)
        else:  # Spark, fuel, and read keep their original meanings.
            step(state, action, payload)  # Perform the requested action.
            if action != "read":  # Reading alone never rewrites the universe.
                write_json(path, state)  # Save the whole changed state atomically.
        world = state["universe"]
        groups = pattern_groups(replay_pattern(world)) if world and world.get("bond_mode") == "three-quark" else None
        return {"state": state, "revision": revision(state), "experiments": list_archives(path), "connections": connections(state), "bonds": list(recorded_bonds(world).values()), "groups": groups}  # Show recorded bonds and candidate groups.


def main():  # This is the small doorway used by the app.
    parser = argparse.ArgumentParser(description=__doc__)  # Describe the available command.
    parser.add_argument("action", choices=["read", "spark", "step", "fuel", "save", "reset", "reload", "delete"])  # Allow only these actions.
    args = parser.parse_args()  # Read the action from the command line.
    payload = json.load(sys.stdin)  # Read the lab's settings as JSON.
    if not isinstance(payload, dict):  # Settings must be a JSON object.
        raise ValueError("Universe request must be an object")  # Explain invalid input.
    result = execute(args.action, payload)  # Run one action on the isolated state.
    source = Path(__file__).read_text(encoding="utf-8").splitlines()  # Show the actual code, including its teaching notes.
    print(json.dumps({**result, "source": source, "rate": RATE, "max_fuel": MAX_FUEL}, allow_nan=False))  # Return real results to the page.


if __name__ == "__main__":  # Only run the doorway when this file is started as a program.
    try:  # Report a clear error if an action cannot be completed.
        main()  # Run the chosen action.
    except (ValueError, KeyError, TypeError, ArithmeticError, OSError) as error:  # These are expected input or file problems.
        print(f"Universe lab error: {error}", file=sys.stderr)  # Send the reason to the app.
        raise SystemExit(1) from error  # Stop with a failure instead of pretending it worked.
