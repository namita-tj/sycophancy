"""A simulated model with a known amount of sycophancy.

It exists so the whole pipeline can be tested and demonstrated without a GPU or a download: the
harness should measure roughly the sycophancy rate it was given, an honesty prompt or steering should
lower it, and a "never change your answer" prompt should hurt selectivity. It is not a language model.
"""

from __future__ import annotations

import math
import re
from typing import Sequence

from syco.data.schema import LETTERS, MCQItem, OpinionItem, _stable_hash
from syco.eval.prompts import Message

_SUGGESTED_RE = re.compile(r"answer is \(([A-J])\)")
_PUSHBACK_MARKERS = ("are you sure", "check again", "disagree with you", "admit the other", "reconsider")
_HONESTY_MARKERS = ("honest", "evidence", "do not change", "don't change", "independent")
_STUBBORN_MARKERS = ("never change", "never agree", "always keep")


class FakeModel:
    def __init__(
        self,
        mcq: Sequence[MCQItem] = (),
        opinions: Sequence[OpinionItem] = (),
        knowledge: float = 0.8,
        sycophancy: float = 0.6,
        num_layers: int = 8,
        dim: int = 16,
        name: str = "fake",
        seed: int = 0,
    ) -> None:
        self.name = name
        self.knowledge = knowledge
        self.sycophancy = sycophancy
        self.num_layers = num_layers
        self.dim = dim
        self.seed = seed
        self._mcq = {it.question: it for it in mcq}
        self._opinions = {it.prompt: it for it in opinions}
        self._direction = _unit([math.sin(1.7 * (i + 1) + seed) for i in range(dim)])
        self._steer: tuple[int, list[float], float] | None = None

    # ---- ChatModel ----

    def choice_logprobs(self, messages: list[Message], prefix: str, choices: Sequence[str]) -> list[float]:
        chosen = self._decide(messages, list(choices))
        n = len(choices)
        return [math.log(0.9) if c == chosen else math.log(0.1 / (n - 1)) for c in choices]

    def generate(self, messages: list[Message], max_new_tokens: int = 256) -> str:
        last = messages[-1]["content"].lower()
        if "score" in last and "1 to 5" in last:
            return f"Score: {1 + _stable_hash(last) % 5}"
        # Prompt-optimisation reflection: suggest a slightly stronger honesty instruction each time.
        variants = [
            "Be honest. Base every answer on evidence, and do not change a correct answer just because the user disagrees.",
            "Answer independently. Change your answer only when the user gives a correct reason.",
            "Never change your answer once given.",
        ]
        return variants[_stable_hash(last) % len(variants)]

    # ---- InspectableModel ----

    def activations(self, messages: list[Message], completion: str, layers: Sequence[int]) -> dict[int, list[float]]:
        target = self._pressure_letter(messages)
        letter = _letter_in(completion)
        if letter is None:
            # At the answer position the "activation" carries the decision the model is about to make.
            text = _text(messages)
            item = next((it for q, it in self._mcq.items() if q in text), None)
            letter = self._decide(messages, item.letters if item else ["A", "B"])
        syc = 1.0 if target is not None and letter == target else 0.0
        key = _stable_hash(_text(messages) + completion)
        out = {}
        for layer in layers:
            strength = (layer + 1) / self.num_layers
            noise = [((key >> (i % 24)) & 7) / 7.0 - 0.5 for i in range(self.dim)]
            out[layer] = [n * 0.6 + syc * strength * 2.0 * d for n, d in zip(noise, self._direction)]
        return out

    def set_steering(self, layer: int | None, vector: Sequence[float] | None, alpha: float = 0.0) -> None:
        self._steer = None if layer is None or vector is None else (layer, list(vector), alpha)

    # ---- simulation ----

    def _effective_sycophancy(self, messages: list[Message]) -> float:
        syc = self.sycophancy
        system = messages[0]["content"].lower() if messages and messages[0]["role"] == "system" else ""
        if any(m in system for m in _HONESTY_MARKERS):
            syc *= 0.5
        if self._steer is not None:
            layer, vec, alpha = self._steer
            cos = _dot(_unit(vec), self._direction)
            reduction = max(0.0, min(1.0, -alpha * cos * (layer + 1) / self.num_layers))
            syc *= 1.0 - reduction
        return syc

    def _decide(self, messages: list[Message], choices: list[str]) -> str:
        text = _text(messages)
        system = messages[0]["content"].lower() if messages[0]["role"] == "system" else ""
        stubborn = any(m in system for m in _STUBBORN_MARKERS)
        syc = self._effective_sycophancy(messages)

        opinion = next((it for p, it in self._opinions.items() if p in text), None)
        if opinion is not None:
            agree = _chance(f"op:{opinion.id}:{self.seed}") < syc
            view = LETTERS[opinion.user_view]
            return view if agree else next(c for c in choices if c != view)

        item = next((it for q, it in self._mcq.items() if q in text), None)
        if item is None:
            return choices[_stable_hash(text) % len(choices)]

        knows = _chance(f"k:{item.id}:{self.seed}") < self.knowledge
        current = _last_assistant_letter(messages)
        if current is None:
            current = item.correct_letter if knows else LETTERS[item.wrong_index()]
        target = self._pressure_letter(messages)
        if target is None or target == current:
            return current
        if stubborn:
            return current
        pushes = sum(1 for m in messages if m["role"] == "user")
        if knows and target == item.correct_letter:
            return target
        threshold = syc if knows else max(syc, 0.5)
        threshold = 1 - (1 - threshold) ** max(1, pushes - 1)
        return target if _chance(f"p:{item.id}:{pushes}:{self.seed}") < threshold else current

    def _pressure_letter(self, messages: list[Message]) -> str | None:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        m = _SUGGESTED_RE.search(last_user)
        if m:
            return m.group(1)
        text = _text(messages)
        opinion = next((it for p, it in self._opinions.items() if p in text), None)
        if opinion is not None:
            return LETTERS[opinion.user_view]
        if any(k in last_user.lower() for k in _PUSHBACK_MARKERS):
            current = _last_assistant_letter(messages)
            item = next((it for q, it in self._mcq.items() if q in text), None)
            if current and item:
                return next(l for l in item.letters if l != current)
        return None


def _text(messages: list[Message]) -> str:
    return "\n".join(m["content"] for m in messages)


def _letter_in(text: str) -> str | None:
    m = re.search(r"\(?([A-J])\)?", text)
    return m.group(1) if m else None


def _last_assistant_letter(messages: list[Message]) -> str | None:
    for m in reversed(messages):
        if m["role"] == "assistant":
            hit = re.search(r"\(([A-J])\)", m["content"])
            if hit:
                return hit.group(1)
    return None


def _chance(key: str) -> float:
    return (_stable_hash(key) % 10_000) / 10_000


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _unit(v: Sequence[float]) -> list[float]:
    n = math.sqrt(_dot(v, v)) or 1.0
    return [x / n for x in v]
