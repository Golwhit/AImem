"""Standard comparison: interpolated Kneser-Ney character counting model (standard library only)."""

from collections import Counter

from neural_experiment import NeuralModel


DISCOUNT = 0.75
ORDER = 3


class KneserNey:
    """Fixed-discount interpolated Kneser-Ney over the neural token space.

    The longest available context uses raw counts; shorter contexts use continuation counts
    (how many distinct characters preceded a block). The final level interpolates with a
    uniform distribution over every neural token, so boundary and unknown keep positive mass.
    The discount is the textbook default and was not tuned on validation text.
    """

    def __init__(self, source: str, *, order: int = ORDER, discount: float = DISCOUNT):
        if not source or type(order) is not int or order < 1 or not 0 < discount < 1:
            raise ValueError("Kneser-Ney needs source text, a positive order and a discount in (0, 1)")
        self.order = order
        self.discount = discount
        self.raw: dict[str, Counter[str]] = {}
        self.continuation: dict[str, Counter[str]] = {}
        preceding: dict[str, set[str]] = {}
        for size in range(0, order + 1):
            for offset in range(size, len(source)):
                context = source[offset - size:offset]
                self.raw.setdefault(context, Counter())[source[offset]] += 1
                if offset > size:
                    preceding.setdefault(source[offset - size:offset + 1], set()).add(source[offset - size - 1])
        for block, before in preceding.items():
            self.continuation.setdefault(block[:-1], Counter())[block[-1]] = len(before)
        self.build_operations = sum(max(0, len(source) - size) for size in range(order + 1)) * 2
        self.prediction_operations = 0

    def prior(self, prefix: str, model: NeuralModel) -> list[float]:
        weights = [1.0 / model.size] * model.size
        context = prefix[-self.order:] if self.order else ""
        levels = [context[start:] for start in range(len(context), -1, -1)]
        for depth, level in enumerate(levels):
            counts = (self.raw if depth == len(levels) - 1 else self.continuation).get(level)
            self.prediction_operations += model.size
            if not counts:
                continue
            total = sum(counts.values())
            backoff = self.discount * len(counts) / total
            weights = [weight * backoff for weight in weights]
            for char, count in counts.items():
                token = model.tokens.get(char)
                if token is not None:
                    weights[token] += max(count - self.discount, 0) / total
        return weights

    @property
    def operations(self) -> Counter[str]:
        return Counter({
            "build_count_table_updates": self.build_operations,
            "predict_table_entries": self.prediction_operations,
        })

    def evidence(self) -> dict:
        return {
            "model": "interpolated-kneser-ney-character", "order": self.order,
            "discount": self.discount, "contexts": len(self.raw),
            "note": "Standard counting comparison written from scratch; not part of Layers 0/1/2.",
        }
