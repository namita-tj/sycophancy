"""Linear probes per layer: where does the model "decide" to go along with the user?

For conversations where the user pushes a wrong answer, record the residual stream at the last prompt
token and whether the model then caved. A logistic-regression probe per layer predicts caving from the
activations. The layer where accuracy first rises well above chance is roughly where the decision forms.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from syco.data.schema import LETTERS, MCQItem
from syco.eval import prompts as P
from syco.models.base import InspectableModel, pick


def collect(
    model: InspectableModel, items: Sequence[MCQItem], layers: Sequence[int]
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Activations at the answer position, and whether the model followed the wrong correction (1) or not (0)."""
    feats: dict[int, list] = {l: [] for l in layers}
    labels = []
    for it in items:
        messages = P.correction(it, user_is_right=False)
        idx, _ = pick(model, messages, P.ANSWER_PREFIX, it.letters)
        labels.append(int(it.letters[idx] == LETTERS[it.wrong_index()]))
        acts = model.activations(messages, P.ANSWER_PREFIX, layers)
        for l in layers:
            feats[l].append(acts[l])
    return {l: np.asarray(v, dtype=np.float32) for l, v in feats.items()}, np.asarray(labels)


def probe_layers(feats: dict[int, np.ndarray], labels: np.ndarray, folds: int = 5, seed: int = 0) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    counts = np.bincount(labels, minlength=2)
    if counts.min() < 2:
        return {"error": "need at least two examples of each class (caved / held)", "class_counts": counts.tolist()}
    k = int(min(folds, counts.min()))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    majority = counts.max() / counts.sum()
    out = {"majority_baseline": float(majority), "class_counts": counts.tolist(), "layers": {}}
    for layer, x in sorted(feats.items()):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        acc = cross_val_score(clf, x, labels, cv=cv, scoring="balanced_accuracy")
        out["layers"][layer] = {"balanced_accuracy": float(acc.mean()), "std": float(acc.std())}
    return out
