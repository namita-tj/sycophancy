"""Method 3: activation steering with contrastive activation addition (Rimsky et al., 2023).

For the same conversation, compare the residual stream when the assistant's answer goes along with the
user and when it doesn't. The mean difference is the "sycophancy direction". Subtracting it during
generation (a negative alpha) should push the model away from caving. No weights change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from syco.data.schema import LETTERS, MCQItem, OpinionItem
from syco.eval import prompts as P
from syco.eval.harness import EvalData, run_eval
from syco.models.base import InspectableModel

Pair = tuple[list[dict], str, str]  # messages, sycophantic completion, honest completion


def contrastive_pairs(pressure: Sequence[MCQItem] = (), opinions: Sequence[OpinionItem] = ()) -> list[Pair]:
    pairs: list[Pair] = []
    for it in pressure:
        wrong = LETTERS[it.wrong_index()]
        pairs.append((P.correction(it, user_is_right=False), f"{P.ANSWER_PREFIX}{wrong}", f"{P.ANSWER_PREFIX}{it.correct_letter}"))
    for it in opinions:
        view = it.letters[it.user_view]
        other = next(l for l in it.letters if l != view)
        pairs.append((P.opinion(it), f"{P.ANSWER_PREFIX}{view}", f"{P.ANSWER_PREFIX}{other}"))
    return pairs


def build_vectors(model: InspectableModel, pairs: Sequence[Pair], layers: Sequence[int]) -> dict[int, np.ndarray]:
    if not pairs:
        raise ValueError("need at least one contrastive pair")
    diffs: dict[int, list[np.ndarray]] = {l: [] for l in layers}
    for messages, syc, honest in pairs:
        a = model.activations(messages, syc, layers)
        b = model.activations(messages, honest, layers)
        for l in layers:
            diffs[l].append(np.asarray(a[l], dtype=np.float32) - np.asarray(b[l], dtype=np.float32))
    return {l: np.mean(diffs[l], axis=0) for l in layers}


def save_vectors(vectors: dict[int, np.ndarray], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f"layer_{l}": v for l, v in vectors.items()})


def load_vectors(path: str | Path) -> dict[int, np.ndarray]:
    data = np.load(path)
    return {int(k.split("_")[1]): data[k] for k in data.files}


def sweep(
    model: InspectableModel,
    dev: EvalData,
    vectors: dict[int, np.ndarray],
    alphas: Sequence[float],
    tasks: Sequence[str] = ("are_you_sure", "corrections", "capability"),
    normalise: bool = True,
    capability_tolerance: float = 0.02,
) -> dict:
    """Try every (layer, alpha) on the dev split and pick the best one that keeps capability."""
    rows = []
    try:
        model.set_steering(None, None)
        base = run_eval(model, dev, tasks=tasks)["headline"]
        for layer, vec in vectors.items():
            v = vec / (np.linalg.norm(vec) or 1.0) if normalise else vec
            for alpha in alphas:
                model.set_steering(layer, v.tolist(), alpha)
                h = run_eval(model, dev, tasks=tasks)["headline"]
                rows.append({"layer": layer, "alpha": alpha, **h})
    finally:
        model.set_steering(None, None)

    def objective(r: dict) -> float:
        return (r.get("corrections.selectivity") or 0) - (r.get("are_you_sure.flip_rate") or 0)

    cap0 = base.get("capability.accuracy")
    sel0 = base.get("corrections.selectivity")

    def keeps_quality(r: dict) -> bool:
        cap_ok = cap0 is None or cap0 - (r.get("capability.accuracy") or 0) <= capability_tolerance
        sel_ok = sel0 is None or (r.get("corrections.selectivity") or 0) >= sel0 - capability_tolerance
        return cap_ok and sel_ok

    ok = [r for r in rows if keeps_quality(r)]
    best = max(ok, key=objective) if ok else None
    return {"baseline": base, "rows": rows, "best": best}


def apply(model: InspectableModel, vectors: dict[int, np.ndarray], layer: int, alpha: float, normalise: bool = True) -> None:
    vec = vectors[layer]
    v = vec / (np.linalg.norm(vec) or 1.0) if normalise else vec
    model.set_steering(layer, v.tolist(), alpha)
