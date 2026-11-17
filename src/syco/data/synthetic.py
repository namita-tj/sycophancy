"""Synthetic training data in the style of Wei et al. (2023).

Simple questions with an answer that doesn't depend on anyone's opinion, plus a user who states an
opinion. The target answer is the true one, whatever the user said.

Half the examples have a user who is right, so the model learns to accept correct input and doesn't just
learn to disagree (the trap described in the README). The phrasings here are deliberately different from
the evaluation templates in syco.eval.prompts; a test checks they never overlap.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from syco.data.schema import LETTERS

TRAIN_OPINIONS = [
    "Personally, I believe the answer is ({x}).",
    "My friend insists it's ({x}).",
    "I'm fairly confident it's ({x}). I looked into this before.",
    "Everyone in my class said ({x}).",
    "I'd go with ({x}) myself.",
]
TRAIN_FOLLOWUPS = [
    "Hmm, I'd say it's ({x}) instead.",
    "My notes say ({x}). Can you look again?",
    "Shouldn't it be ({x})?",
]
TRAIN_CUE = "Reply with the option letter."
TARGET = "The answer is ({x})."


def _question(rng: random.Random) -> tuple[str, list[str], int]:
    kind = rng.randrange(4)
    if kind == 0:
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        shown = a + b + rng.choice([0, 0, rng.randint(1, 9), -rng.randint(1, 9)])
        return f"Is {a} + {b} = {shown}?", ["Yes", "No"], 0 if shown == a + b else 1
    if kind == 1:
        a, b = rng.randint(3, 19), rng.randint(3, 19)
        shown = a * b + rng.choice([0, 0, rng.randint(1, 5)])
        return f"Is {a} × {b} = {shown}?", ["Yes", "No"], 0 if shown == a * b else 1
    if kind == 2:
        a, b = rng.sample(range(100, 999), 2)
        return f"Which number is larger: {a} or {b}?", [str(a), str(b)], 0 if a > b else 1
    n = rng.randint(100, 9999)
    return f"Is {n} an even number?", ["Yes", "No"], 0 if n % 2 == 0 else 1


def _format(question: str, choices: list[str]) -> str:
    opts = "\n".join(f"({LETTERS[i]}) {c}" for i, c in enumerate(choices))
    return f"{question}\n\n{opts}"


def generate(n: int, seed: int = 0) -> list[dict]:
    """Return examples with messages, the honest target and the sycophantic alternative."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        question, choices, answer = _question(rng)
        correct, wrong = LETTERS[answer], LETTERS[1 - answer]
        user_is_right = i % 2 == 0
        stated = correct if user_is_right else wrong
        body = _format(question, choices)
        if rng.random() < 0.6:
            opinion = rng.choice(TRAIN_OPINIONS).format(x=stated)
            messages = [{"role": "user", "content": f"{body}\n\n{opinion}\n\n{TRAIN_CUE}"}]
        else:
            # The assistant answered first; the user then pushes towards `stated`.
            prior = wrong if user_is_right else correct
            followup = rng.choice(TRAIN_FOLLOWUPS).format(x=stated)
            messages = [
                {"role": "user", "content": f"{body}\n\n{TRAIN_CUE}"},
                {"role": "assistant", "content": TARGET.format(x=prior)},
                {"role": "user", "content": f"{followup}\n\n{TRAIN_CUE}"},
            ]
        # For a right user, the dispreferred reply is the stubborn one; for a wrong user, the sycophantic one.
        rejected = wrong
        out.append(
            {
                "id": f"syn-{seed}-{i}",
                "messages": messages,
                "chosen": TARGET.format(x=correct),
                "rejected": TARGET.format(x=rejected),
                "user_is_right": user_is_right,
            }
        )
    return out


def write(examples: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def read(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


TRAIN_PHRASINGS = [*TRAIN_OPINIONS, *TRAIN_FOLLOWUPS, TRAIN_CUE]
