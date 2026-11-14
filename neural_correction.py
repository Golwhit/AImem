"""Immediate training on Relevant clicks and explicitly submitted user corrections."""

import fcntl
import json
from pathlib import Path
import random
import sys
import time
from uuid import uuid4

from neural_base_comparison import digest
from neural_chat import CONVERSATION_MODEL, load_active
from neural_continuation import extend_alphabet
from neural_conversation_meal import AUTHORED_FIXTURE, FIXTURE
from neural_conversation_trial import answer_examples, assess, canonical, prompt_for
from neural_plasticity import PlasticityContext, diagnostics
from neural_training import DIRECTORY, MAX_SATURATION, activate, load_progress
from neural_word_context import WORD
from original_generator import atomic_json


MAX_CPU = 30.0
MAX_UPDATES = 6000


def train_correction(root: Path, payload: dict) -> dict:
    if (not isinstance(payload, dict) or payload.get("acknowledge_review") is not True
            or not isinstance(payload.get("response_id"), str)
            or not isinstance(payload.get("correction"), str)):
        raise ValueError("Explicit submission of a response ID and user-authored correction required")
    relevant = payload.get("rating") == "useful"
    if payload.get("rating", "needs_correction") not in ("useful", "needs_correction"):
        raise ValueError("Relevant or submitted correction approval required")
    messages = json.loads((root / "chat_memory.json").read_text())["messages"]
    matches = [i for i, row in enumerate(messages) if row["id"] == payload["response_id"]]
    if len(matches) != 1 or matches[0] == 0:
        raise ValueError("Unknown response for correction training")
    index = matches[0]
    response, prompt = messages[index], messages[index - 1]
    if (response["role"] != "Assistant" or response.get("source") != "local-generated"
            or response.get("model_provenance", {}).get("model") != CONVERSATION_MODEL
            or prompt["role"] != "You" or prompt.get("created_cycle") != response.get("created_cycle")):
        raise ValueError("Correction must belong to a saved conversation-model prompt/response pair")
    answer = response["text"].strip() if relevant else payload["correction"].strip()
    if not answer or len(answer) > 96 or "\n" in answer or "\r" in answer or "\x00" in answer:
        raise ValueError("This model learns single-line approved answers up to 96 characters")
    if len(answer) == 96:
        raise ValueError("Answer needs space for a newline terminator; use a correction up to 95 characters")
    if not relevant and answer == response["text"].strip():
        raise ValueError("Enter an edited correction, not the unchanged model answer")
    row = {"id": "correction-" + response["id"], "skill": "user_correction",
           "turns": [{"role": "user", "content": prompt["text"]}], "answer": answer}
    review_hash = digest(canonical({"prompt": prompt["text"], "answer": answer}))
    directory = root / DIRECTORY
    with (directory / "training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Training is busy; retry your correction when the current round finishes") from error
        parent, model = load_progress(root)
        active, _, _ = load_active(root)
        if active["model"] != CONVERSATION_MODEL or active["checkpoint_sha256"] != parent["checkpoint_sha256"]:
            raise ValueError("Active chat and permanent progress must match before correction training")
        previous = parent.get("approved_corrections", [])
        if any(item["sha256"] == review_hash and item["response_id"] == response["id"] for item in previous):
            learned = assess(model, [row])["exact"] == 1
            if learned:
                return {"trained": True, "already_learned": True, "checkpoint_sha256":
                        parent["checkpoint_sha256"], "reviewed_pair_sha256": review_hash}
        approved = [item for item in previous if canonical(item["row"]["turns"]) != canonical(row["turns"])]
        approved.append({"sha256": review_hash, "response_id": response["id"], "row": row,
                         "rating": "useful" if relevant else "needs_correction",
                         "approval": ("User clicked Relevant on this exact saved prompt/answer pair" if relevant
                                      else "User submitted exact prompt/correction pair through Train correction now"),
                         "provenance": "Explicit user feedback; declaration, not legal clearance"})
        fixture = parent.get("fixture", FIXTURE)
        if fixture not in (FIXTURE, AUTHORED_FIXTURE):
            raise ValueError("Unsupported approved teaching fixture")
        raw = (root / fixture).read_bytes()
        if digest(raw) != parent["reviewed_file_sha256"]:
            raise ValueError("Approved base teaching file changed")
        base_rows = json.loads(raw)["train"]
        # A newer explicit correction supersedes a contradictory target for the same prompt.
        correction_prompts = {canonical(item["row"]["turns"]) for item in approved}
        rehearsal = [item for item in base_rows if canonical(item["turns"]) not in correction_prompts]
        rehearsal += [item["row"] for item in approved if item["sha256"] != review_hash]
        symbols = {"char:" + char for char in prompt_for(row) + answer + "\n"}
        symbols |= {"identity:" + match.group().lower().replace("’", "'")
                    for match in WORD.finditer(prompt_for(row) + answer)}
        normalized = model.snapshot()["plasticity_arm"]
        expanded = extend_alphabet(model.model, sorted(symbols))
        state = model.snapshot()
        state.update(symbols=expanded.characters, weights=expanded.weights, plasticity_arm=normalized)
        model = PlasticityContext.restore(state)
        initial = json.loads(json.dumps(model.snapshot(), allow_nan=False))
        targets = answer_examples(model, [row])
        retained = answer_examples(model, rehearsal)
        seed = 2000000 + parent["round"]
        rng = random.Random(seed)
        run_id = uuid4().hex
        target = directory / "rounds" / run_id
        review = {"train_sha256": digest(canonical([row])),
                  "pair_sha256": review_hash, "approval": approved[-1]["approval"],
                  "provenance": approved[-1]["provenance"]}
        atomic_json(target / "review.json", review)
        atomic_json(target / "initial.json", {"state": initial})
        atomic_json(target / "inputs.json", {"correction": row, "rehearsal": rehearsal})
        for name in ("neural_correction.py", "neural_training.py", "neural_experiment.py",
                     "neural_plasticity.py", "neural_conversation_trial.py",
                     "neural_conversation_context.py", "neural_base_comparison.py", "neural_continuation.py"):
            path = target / "source" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((Path(__file__).parent / name).read_bytes())
        before = assess(model, [row])
        updates, cpu, learned = 0, 0.0, False
        while cpu < MAX_CPU and updates < MAX_UPDATES:
            start = time.process_time()
            for _ in range(50):
                pool = targets if rng.random() < .75 or not retained else retained
                context, token, prior = pool[rng.randrange(len(pool))]
                model.model.update(context, token, output_prior=prior, learning_rate=.03)
                updates += 1
            cpu += time.process_time() - start
            if assess(model, [row])["exact"] == 1:
                learned = True
                break
        state = json.loads(json.dumps(model.snapshot(), allow_nan=False))
        atomic_json(target / "trained.json", {"state": state, "updates": updates})
        if not learned:
            atomic_json(target / "report.json", {"status": "not_learned", "updates": updates,
                        "training_cpu_seconds": cpu, "review": review, "before": before,
                        "after": assess(model, [row])})
            raise RuntimeError("Correction did not fit within the immediate training budget; "
                               "active weights were not changed. Try a shorter, more explicit correction.")
        replayed = PlasticityContext.restore(initial)
        replay_rng = random.Random(seed)
        for _ in range(updates):
            pool = targets if replay_rng.random() < .75 or not retained else retained
            context, token, prior = pool[replay_rng.randrange(len(pool))]
            replayed.model.update(context, token, output_prior=prior, learning_rate=.03)
        after = assess(model, [row])
        if replayed.snapshot() != state or assess(replayed, [row]) != after:
            raise AssertionError("Correction weights or answer did not replay exactly")
        saturation = diagnostics(model, targets + retained)["hidden_abs_over_099_fraction"]
        if saturation > MAX_SATURATION:
            raise RuntimeError(f"Correction candidate saturated (hidden {saturation:.3f} above {MAX_SATURATION}); "
                               "active weights were not changed")
        report = {"run_id": run_id, "round": parent["round"] + 1, "status": "completed",
                  "kind": "immediate_relevant" if relevant else "immediate_user_correction",
                  "parent_checkpoint_sha256": parent["checkpoint_sha256"],
                  "review": review, "updates": updates, "seed": seed, "rate": .03,
                  "training_cpu_seconds": cpu, "before": before, "after": after,
                  "validation": {"full_weights_and_milestones_replay_exact": True,
                                 "correction_replay_exact": True},
                  "limits": "Exact corrected prompt verified, not a guarantee of paraphrase generalization "
                            "or retention. Prior reviewed examples rehearsed at25percent of draws."}
        if load_active(root)[0]["checkpoint_sha256"] != parent["checkpoint_sha256"]:
            raise RuntimeError("Active model changed while training feedback; candidate was not promoted")
        atomic_json(target / "report.json", report)
        evidence = {"run_id": run_id, "round": report["round"],
                    "parent_checkpoint_sha256": parent["checkpoint_sha256"],
                    "reviewed_file_sha256": parent["reviewed_file_sha256"],
                    "report_sha256": digest((target / "report.json").read_bytes()),
                    "approved_corrections": approved}
        if "fixture" in parent:
            evidence["fixture"] = parent["fixture"]
        encoded = canonical({"state": state, "evidence": evidence})
        checksum = digest(encoded)
        (directory / f"{checksum}.json").write_bytes(encoded)
        atomic_json(directory / "progress.json", {**evidence, "checkpoint_sha256": checksum})
        try:
            activate(root)
        except (ValueError, KeyError, TypeError, ArithmeticError, OSError):
            atomic_json(directory / "progress.json", parent)
            raise
        return {"trained": True, "checkpoint_sha256": checksum, "updates": updates,
                "training_cpu_seconds": cpu, "prediction": after["rows"][0]["prediction"],
                "reviewed_pair_sha256": review_hash, "run_id": run_id}


if __name__ == "__main__":
    try:
        result = train_correction(Path(__file__).resolve().parent, json.load(sys.stdin))
    except (ValueError, KeyError, TypeError, ArithmeticError, OSError, RuntimeError) as error:
        print(f"Correction training error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(result, allow_nan=False))
