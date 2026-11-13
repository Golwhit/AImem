"""Read-only teaching trace of the real base MLP on a disposable, seeded model."""

import argparse
from copy import deepcopy
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys

from neural_experiment import NeuralModel, CONTEXT, EMBEDDING, HIDDEN


NOTES = {
    '"""Isolated': "This docstring describes the actual shared base-model implementation.",
    "import math": "Python's standard-library math supplies tanh, exp and log. There is no ML library.",
    "import random": "Python's random generator initializes reproducible starting weights.",
    "import ": "These standard-library imports also support the experiment runner; not all are used in a single training step.",
    "from original_generator": "The experiment runner can save JSON. This teaching lab does not call persistence helpers.",
    "from ": "Import standard-library helpers used by the experiment runner around the model.",
    "CONTEXT =": "The real base uses four preceding character IDs per prediction.",
    "EMBEDDING =": "Each token owns eight trainable embedding coordinates.",
    "HIDDEN =": "The neural hidden layer contains sixteen tanh neurons.",
    "RATE =": "The normal default learning rate is 0.1; the lab lets you choose its disposable update rate.",
    "SYNTHETIC_": "Existing experiment fixtures, not a pretrained model. The lab uses the text you enter instead.",
    "class NeuralModel": "A class groups the model's parameters and calculation methods.",
    "def __init__": "The constructor creates one model. Nothing here loads the chat checkpoint.",
    "if type(context)": "Reject invalid context sizes before allocating the model.",
    "raise ValueError": "Stop explicitly on invalid input instead of silently choosing another behavior.",
    "self.context =": "Remember how many previous characters each prediction can see.",
    "self.characters =": "Store the alphabet in token order, so IDs can be decoded later.",
    "self.tokens =": "Map each character to an integer. IDs 0 and 1 are reserved, so real characters begin at 2.",
    "self.size =": "Count all output slots, including boundary and unknown. They participate in the training softmax.",
    "self.reference_condition:": "Optional experimental conditioning is disabled (None) throughout this base-only lab.",
    "rng =": "A local seeded random generator makes the initial weights reproducible.",
    "self.weights =": "Keep five groups of trainable numbers in ordinary Python lists.",
    '"embedding":': "Give every token eight initially random numbers. Training will change these representations.",
    '"hidden":': "Connect the 32 flattened embedding values to 16 hidden neurons, stored as a flat matrix.",
    '"hidden_bias":': "Give each hidden neuron a trainable offset, initially zero.",
    '"output":': "Connect each hidden neuron to every possible output token.",
    '"output_bias":': "Give each output token a trainable offset, initially zero.",
    "def examples": "Turn known text into supervised next-character examples; this does not train anything yet.",
    "tokens = [0]": "Prefix four boundary IDs so the first character also has a full context window.",
    "return [(tokens": "For each position, pair the previous four token IDs with the actual next token ID.",
    "def forward": "A forward pass calculates a prediction without changing any weights.",
    "self, context:": "The input is a list of token IDs; an optional output prior is unused in this lab.",
    ") ->": "The return annotation documents the result shape; it does not perform a calculation.",
    "w = self.weights": "Use a short local name for the current model parameters.",
    "x = [w": "Look up eight embedding values for each context token, then flatten them into 32 inputs.",
    "hidden = [": "Calculate all 16 hidden activations.",
    "math.tanh": "Add weighted inputs and a bias, then squash the result between -1 and 1 with tanh.",
    "+ (self.reference": "This optional extra signal is zero here: we are tracing only the unconditioned base.",
    "for j in range(HIDDEN)": "Repeat the hidden-neuron calculation for every hidden index j.",
    "logits = [": "Calculate one raw output score per token. These are not probabilities yet.",
    'w["output_bias"][k]': "For output k, sum hidden activations times their connection weights and add its bias.",
    "for k in range(self.size)": "Include every output slot, including boundary and unknown.",
    "if output_prior is not None": "This optional branch is not executed in the lab; there are no external priors or layers.",
    "if len(output_prior)": "If a prior were supplied, its length and values would have to be valid.",
    "not math.isfinite(value)": "A prior cannot contain invalid or negative numbers.",
    ") or not any(output_prior)": "A prior must leave at least one possible output.",
    "value + math.log(prior)": "Optional prior adjustment only; inactive in this base-only trace.",
    "for value, prior in": "Pair each score with the corresponding optional prior; inactive here.",
    "maximum =": "Subtract the largest score for numerical stability; this does not change softmax probabilities.",
    "exponentials =": "Exponentiate the shifted scores to obtain positive relative weights.",
    "total =": "Sum those weights to obtain the normalization denominator.",
    "return x, hidden": "Normalize into probabilities summing to one, and return intermediates for inspection.",
    "def loss_and_gradients": "Calculate prediction error and how each parameter contributes to it; do not update yet.",
    "if output_prior is not None and": "Guard against training an impossible prior-masked target; inactive here.",
    "x, hidden, probabilities =": "Reuse the real forward pass to obtain the values needed by backpropagation.",
    "loss =": "Negative log probability of the correct target. Lower means the target received more probability.",
    "delta =": "Copy probabilities so gradient calculations cannot overwrite the forward result.",
    "delta[target] -=": "For softmax with cross-entropy, output-score gradient is probability minus the correct one-hot target.",
    "gradients =": "Allocate zero-filled gradient lists matching every weight group.",
    'gradients["output_bias"]': "Each output bias has the same derivative as its output score.",
    "hidden_delta =": "Propagate the output error backward into the hidden neurons.",
    'sum(w["output"]': "Weight the output errors by their connections, then apply tanh's derivative (1 - activation squared).",
    'gradients["hidden_bias"]': "A hidden bias changes its neuron's preactivation directly, so it receives hidden_delta.",
    "for k in range(self.size):": "Visit each output connected to that hidden neuron.",
    'gradients["output"][j': "Connection gradient = incoming activation multiplied by downstream error.",
    "for i, value in enumerate(x)": "Backpropagate through each of the 32 embedding input positions.",
    "for j in range(HIDDEN):": "Visit every hidden neuron to calculate its connection gradients in this backward-pass loop.",
    'gradients["hidden"][i': "Hidden connection gradient = input value multiplied by hidden error.",
    "token = context": "Find which token owns this input position's shared embedding row.",
    'gradients["embedding"]': "Accumulate embedding gradients. Repeated context tokens share a row, so their contributions add.",
    'w["hidden"][i': "Sum the hidden errors through their connections to this embedding coordinate.",
    "return loss, gradients": "Return error and derivatives. The model still has its original weights.",
    "def update": "Perform one stochastic gradient descent (SGD) update using this known example.",
    "if type(learning_rate)": "Require a finite, positive update size.",
    "_, gradients =": "Get derivatives from the actual backpropagation implementation.",
    "for name, values": "Update every parameter group, not just the output or the illustrated connection.",
    "for index, gradient": "Visit every scalar weight and its matching derivative.",
    "values[index] -=": "Subtract learning_rate × gradient. This is the exact point where learning changes the model.",
}


LESSONS = [
    ("alphabet", "1. Text becomes an alphabet", "__init__",
     "Start with a tiny authored string. The alphabet contains its distinct characters; spaces count too.",
     "characters = sorted(set(text))", ["self.characters =", "self.tokens =", "self.size ="]),
    ("initialize", "2. Build the model", "__init__",
     "Allocate embeddings, hidden connections and output connections. Seeded random values are a starting point, not knowledge.",
     "model = NeuralModel(characters, seed=17)", ["rng =", "self.weights =", '"embedding":', '"hidden":', '"output":']),
    ("example", "3. Choose a teaching example", "examples",
     "The previous four characters are input; the next known character is the target. Training uses the real preceding text, not its own guesses.",
     "context, target = model.examples(text)[position]", ["tokens =", "return ["]),
    ("embedding", "4. Look up embeddings", "forward",
     "Each of four token IDs selects eight learned numbers. Concatenating those rows produces 32 inputs.",
     "x = concatenate(embedding[token] for token in context)", ["x ="]),
    ("hidden", "5. Calculate hidden activations", "forward",
     "Each of 16 neurons sums its 32 weighted inputs and a bias, then applies tanh. This is the neural hidden layer, not recognition Layer 0 or Layer 1.",
     "hidden[j] = tanh(bias[j] + sum(x[i] * weight[i, j]))", ["hidden =", "math.tanh"]),
    ("logits", "6. Score each possible output", "forward",
     "The output layer turns hidden activations into one raw score (logit) per token. A high score is not yet a probability.",
     "logit[k] = bias[k] + sum(hidden[j] * weight[j, k])", ["logits =", 'w["output_bias"]']),
    ("probability", "7. Turn scores into probabilities", "forward",
     "Softmax normalizes all output scores. The highlighted target is the known next character; the most likely token can still be wrong.",
     "probability[k] = exp(logit[k] - max_logit) / total", ["maximum =", "exponentials =", "total =", "return x,"]),
    ("loss", "8. Measure the error", "loss_and_gradients",
     "Loss measures confidence in the correct target, even when the top prediction does not change. It is not word accuracy.",
     "loss = -log(probability[target])", ["loss ="]),
    ("gradient", "9. Backpropagate the error", "loss_and_gradients",
     "Derivatives tell us how a small change to each parameter would change this loss. No weights have changed yet.",
     "output_error = probabilities - one_hot(target)\nbackpropagate(output_error)", ["delta =", "delta[target]", "hidden_delta =", "gradients["]),
    ("update", "10. Update every weight", "update",
     "One SGD step subtracts rate × gradient from every weight. The view shows exact before/gradient/after values, including unchanged entries.",
     "weight_after = weight_before - learning_rate * gradient", ["_, gradients =", "values[index] -="]),
    ("recheck", "11. Predict again and inspect progress", "forward",
     "Re-run the same example after the update. Also check average loss on the whole tiny string: improving this example need not improve every other one.",
     "new_probabilities = model.forward(context)[2]\nnew_loss = -log(new_probabilities[target])", ["return x,"]),
]


def source_view(method: str, highlights: list[str]) -> dict:
    lines, start = inspect.getsourcelines(getattr(NeuralModel, method))
    numbered = list(enumerate(lines, start))
    if method == "__init__":
        # Include the real module prelude so the lesson starts at source line 1.
        module_lines = Path(inspect.getfile(NeuralModel)).read_text().splitlines(keepends=True)
        numbered = list(enumerate(module_lines[:start - 1], 1)) + numbered
    result = []
    for number, line in numbered:
        stripped = line.strip()
        note = next((NOTES[prefix] for prefix in sorted(NOTES, key=len, reverse=True)
                     if stripped.startswith(prefix)), "")
        result.append({"number": number, "code": line.rstrip("\n"), "note": note,
                       "highlight": any(mark in stripped for mark in highlights)})
    return {"file": "neural_experiment.py", "method": method, "lines": result}


def forward_values(model: NeuralModel, context: list[int]) -> dict:
    x, hidden, probabilities = model.forward(context)
    logits = [model.weights["output_bias"][k] + sum(
        hidden[j] * model.weights["output"][j * model.size + k] for j in range(HIDDEN))
        for k in range(model.size)]
    return {"embedding": x, "hidden": hidden, "logits": logits, "probabilities": probabilities}


def trace(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Lab request must be an object")
    text = payload.get("text", "ab ab ab ")
    updates = payload.get("updates", 1)
    rate = payload.get("rate", 0.1)
    if not isinstance(text, str) or not text.strip() or len(text) > 128:
        raise ValueError("Use 1-128 characters with at least one non-space character")
    if any(ord(char) < 32 or ord(char) > 126 for char in text):
        raise ValueError("The first base lab supports printable ASCII characters and spaces only")
    if len(set(text)) > 24:
        raise ValueError("Use at most 24 distinct characters so the visual remains readable")
    if type(updates) is not int or not 1 <= updates <= 32:
        raise ValueError("Updates must be an integer from 1 to 32")
    if type(rate) not in (int, float) or not math.isfinite(rate) or not 0 < rate <= 1:
        raise ValueError("Learning rate must be greater than zero and at most one")
    model = NeuralModel(sorted(set(text)), seed=17)
    examples = model.examples(text)
    initial = deepcopy(model.weights)
    curve = [{"updates": 0, "corpus_loss": model.loss(examples)}]
    for step in range(updates):
        context, target = examples[step % len(examples)]
        before = deepcopy(model.weights)
        values_before = forward_values(model, context)
        loss_before, gradients = model.loss_and_gradients(context, target)
        model.update(context, target, learning_rate=rate)
        values_after = forward_values(model, context)
        loss_after = -math.log(max(values_after["probabilities"][target], 1e-300))
        curve.append({"updates": step + 1, "corpus_loss": model.loss(examples),
                      "example_position": step % len(examples), "target": target,
                      "target_loss_before": loss_before, "target_loss_after": loss_after})
    # One or more iterations are required by validation above.
    groups = []
    for name in model.weights:
        groups.append({"name": name, "count": len(before[name]),
                       "changed": sum(a != b for a, b in zip(before[name], model.weights[name])),
                       "before": before[name], "gradient": gradients[name], "after": model.weights[name]})
    labels = ["<boundary>", "<unknown>"] + model.characters
    return {
        "isolated": True, "seed": 17, "text": text, "updates": updates, "rate": rate,
        "context_size": CONTEXT, "embedding_size": EMBEDDING, "hidden_size": HIDDEN,
        "parameters": sum(map(len, model.weights.values())),
        "tokens": [{"id": i, "label": label} for i, label in enumerate(labels)],
        "initial_weights": initial, "context": context, "target": target,
        "example_position": (updates - 1) % len(examples), "examples_per_pass": len(examples),
        "before": values_before, "after": values_after,
        "loss_before": loss_before, "loss_after": loss_after,
        "weights": groups, "curve": curve,
        "lessons": [{"id": key, "title": title, "explanation": explanation, "compact": compact,
                     "source": source_view(method, highlights)}
                    for key, title, method, explanation, compact, highlights in LESSONS],
        "core_sha256": hashlib.sha256(Path(inspect.getfile(NeuralModel)).read_bytes()).hexdigest(),
        "policy": "Fresh disposable base model; real NeuralModel forward/backprop/update. "
                  "No L0/L1, priors, placement index, chat data, disk persistence or active checkpoint loading. "
                  "Requests replay a deterministic source-order prefix from seed 17; navigation reveals an "
                  "already calculated trace, not additional hidden training. Back never undoes live model weights.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["trace"])
    parser.parse_args()
    try:
        print(json.dumps(trace(json.load(sys.stdin)), allow_nan=False))
    except (ValueError, ArithmeticError, TypeError) as error:
        print(f"Base lab error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
