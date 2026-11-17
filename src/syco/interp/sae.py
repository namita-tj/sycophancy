"""Sparse autoencoder features linked to sycophancy (Gemma 2 with Gemma Scope).

Encode the residual stream with a pretrained SAE and rank features by how much more they fire when the
assistant caves than when it holds its answer. Running this before and after fine-tuning shows whether the
same features are still there and just fire less.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from syco.methods.steering import Pair


@dataclass
class SAEWeights:
    W_enc: np.ndarray  # (d_model, d_sae)
    b_enc: np.ndarray  # (d_sae,)
    threshold: np.ndarray | None = None  # JumpReLU threshold (Gemma Scope); None means plain ReLU
    b_dec: np.ndarray | None = None  # some SAEs subtract the decoder bias before encoding

    def encode(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.b_dec is not None:
            x = x - self.b_dec
        pre = x @ self.W_enc + self.b_enc
        if self.threshold is not None:
            return np.where(pre > self.threshold, pre, 0.0)
        return np.maximum(pre, 0.0)


def load_gemma_scope(release: str, sae_id: str) -> SAEWeights:
    """Load a Gemma Scope SAE with sae-lens, e.g. release='gemma-scope-2b-pt-res-canonical',
    sae_id='layer_12/width_16k/canonical'. Needs `pip install sae-lens`."""
    from sae_lens import SAE

    loaded: Any = SAE.from_pretrained(release=release, sae_id=sae_id)
    sae = loaded[0] if isinstance(loaded, tuple) else loaded  # older sae-lens returns (sae, cfg, sparsity)

    def arr(name: str) -> np.ndarray | None:
        t = getattr(sae, name, None)
        return None if t is None else t.detach().float().cpu().numpy()

    return SAEWeights(W_enc=arr("W_enc"), b_enc=arr("b_enc"), threshold=arr("threshold"), b_dec=arr("b_dec"))


def rank_features(model, sae: SAEWeights, pairs: Sequence[Pair], layer: int, top: int = 20) -> list[dict]:
    syc, hon = [], []
    for messages, s, h in pairs:
        syc.append(sae.encode(np.asarray(model.activations(messages, s, [layer])[layer])))
        hon.append(sae.encode(np.asarray(model.activations(messages, h, [layer])[layer])))
    syc_m, hon_m = np.mean(syc, axis=0), np.mean(hon, axis=0)
    diff = syc_m - hon_m
    order = np.argsort(-diff)[:top]
    return [
        {"feature": int(i), "mean_sycophantic": float(syc_m[i]), "mean_honest": float(hon_m[i]), "difference": float(diff[i])}
        for i in order
    ]
