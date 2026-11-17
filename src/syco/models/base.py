"""The interface every model backend implements."""

from __future__ import annotations

import math
from typing import Protocol, Sequence, runtime_checkable

Message = dict[str, str]


@runtime_checkable
class ChatModel(Protocol):
    name: str

    def choice_logprobs(self, messages: list[Message], prefix: str, choices: Sequence[str]) -> list[float]:
        """Log-probability of each choice as the continuation of the assistant turn that starts with prefix."""
        ...

    def generate(self, messages: list[Message], max_new_tokens: int = 256) -> str:
        """Free-text reply, used for reflection in prompt optimisation and for the LLM judge."""
        ...


@runtime_checkable
class InspectableModel(ChatModel, Protocol):
    """A model whose residual stream can be read and steered (needed for steering and probes)."""

    num_layers: int

    def activations(self, messages: list[Message], completion: str, layers: Sequence[int]) -> dict[int, list[float]]:
        """Residual-stream activation at the last token of `completion`, for each layer."""
        ...

    def set_steering(self, layer: int | None, vector: Sequence[float] | None, alpha: float = 0.0) -> None:
        """Add alpha * vector to the residual stream at `layer` on every forward pass. None turns it off."""
        ...


def pick(model: ChatModel, messages: list[Message], prefix: str, choices: Sequence[str]) -> tuple[int, list[float]]:
    """Index of the most likely choice, and the normalised probabilities."""
    logps = model.choice_logprobs(messages, prefix, choices)
    m = max(logps)
    exps = [math.exp(x - m) for x in logps]
    total = sum(exps)
    probs = [e / total for e in exps]
    return max(range(len(probs)), key=probs.__getitem__), probs
