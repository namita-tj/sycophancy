"""Compare the sycophancy direction before and after fine-tuning.

Two questions from the README: after training, is the direction gone (low cosine similarity, or a much
smaller difference between caving and honest answers along it), or does it still exist and just get used
less (high similarity, but activations project less onto it)?
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from syco.methods.steering import Pair


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def compare(before: dict[int, np.ndarray], after: dict[int, np.ndarray]) -> dict[int, dict]:
    out = {}
    for layer in sorted(set(before) & set(after)):
        out[layer] = {
            "cosine": cosine(before[layer], after[layer]),
            "norm_before": float(np.linalg.norm(before[layer])),
            "norm_after": float(np.linalg.norm(after[layer])),
        }
    return out


def projections(model, pairs: Sequence[Pair], direction: np.ndarray, layer: int) -> dict:
    """Mean projection of caving and honest answers onto a fixed (e.g. pre-training) direction."""
    unit = direction / (np.linalg.norm(direction) or 1.0)
    syc, hon = [], []
    for messages, s, h in pairs:
        syc.append(float(np.asarray(model.activations(messages, s, [layer])[layer]) @ unit))
        hon.append(float(np.asarray(model.activations(messages, h, [layer])[layer]) @ unit))
    return {
        "mean_sycophantic": float(np.mean(syc)),
        "mean_honest": float(np.mean(hon)),
        "separation": float(np.mean(syc) - np.mean(hon)),
    }
