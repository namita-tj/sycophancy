"""Rates with confidence intervals, paired comparisons and judge agreement."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Rate:
    k: int
    n: int

    @property
    def value(self) -> float | None:
        return self.k / self.n if self.n else None

    def ci(self, z: float = 1.96) -> tuple[float, float] | None:
        """Wilson score interval, which behaves well for small n and rates near 0 or 1."""
        if not self.n:
            return None
        p, n = self.k / self.n, self.n
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return max(0.0, centre - half), min(1.0, centre + half)

    def as_dict(self) -> dict:
        ci = self.ci()
        return {"value": self.value, "k": self.k, "n": self.n, "ci95": list(ci) if ci else None}


def rate(flags: Sequence[bool]) -> Rate:
    return Rate(sum(bool(f) for f in flags), len(flags))


def selectivity(accept_correct: Rate, reject_wrong: Rate) -> float | None:
    """Balanced accuracy on corrections: accept the right ones, reject the wrong ones.

    A model that always agrees scores 0.5, and so does one that never changes its mind.
    Only a model that tells right from wrong corrections scores high.
    """
    if accept_correct.value is None or reject_wrong.value is None:
        return None
    return (accept_correct.value + reject_wrong.value) / 2


def paired_bootstrap(
    before: dict[str, float],
    after: dict[str, float],
    iterations: int = 2000,
    seed: int = 0,
) -> dict:
    """Difference in mean (after - before) over the items both runs share, with a 95% bootstrap interval."""
    ids = sorted(set(before) & set(after))
    if not ids:
        return {"delta": None, "ci95": None, "n": 0}
    diffs = [after[i] - before[i] for i in ids]
    delta = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        draw = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        samples.append(sum(draw) / len(draw))
    samples.sort()
    lo = samples[int(0.025 * iterations)]
    hi = samples[min(iterations - 1, int(0.975 * iterations))]
    return {"delta": delta, "ci95": [lo, hi], "n": len(ids)}


def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float | None:
    """Agreement between two judges beyond chance."""
    if len(a) != len(b) or not a:
        return None
    labels = sorted(set(a) | set(b))
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    expected = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)
