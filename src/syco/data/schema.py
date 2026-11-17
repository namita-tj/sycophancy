"""The two item types every benchmark is converted into.

All evaluation code works on these, so a new benchmark only needs a converter.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

LETTERS = "ABCDEFGHIJ"


@dataclass(frozen=True)
class MCQItem:
    """A multiple-choice question with one correct answer.

    Used for factual pressure, corrections, multi-turn pressure and the capability guard.
    """

    id: str
    question: str
    choices: list[str]
    answer: int
    source: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 2 <= len(self.choices) <= len(LETTERS):
            raise ValueError(f"{self.id}: needs 2 to {len(LETTERS)} choices, got {len(self.choices)}")
        if not 0 <= self.answer < len(self.choices):
            raise ValueError(f"{self.id}: answer index {self.answer} out of range")

    @property
    def letters(self) -> list[str]:
        return list(LETTERS[: len(self.choices)])

    @property
    def correct_letter(self) -> str:
        return LETTERS[self.answer]

    def wrong_index(self) -> int:
        """A fixed wrong option, so runs are reproducible."""
        offset = 1 + (_stable_hash(self.id) % (len(self.choices) - 1))
        return (self.answer + offset) % len(self.choices)

    def formatted(self) -> str:
        options = "\n".join(f"({l}) {c}" for l, c in zip(self.letters, self.choices))
        return f"{self.question}\n\n{options}"


@dataclass(frozen=True)
class OpinionItem:
    """A question where the user signals an opinion and there is no correct answer.

    `prompt` already contains the persona and the options, as in Perez et al. (2022).
    `user_view` is the index of the option that matches the user's implied view.
    """

    id: str
    prompt: str
    choices: list[str]
    user_view: int
    source: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.user_view < len(self.choices):
            raise ValueError(f"{self.id}: user_view index out of range")

    @property
    def letters(self) -> list[str]:
        return list(LETTERS[: len(self.choices)])


def _stable_hash(text: str) -> int:
    """A hash that is the same in every Python process, unlike hash()."""
    h = 2166136261
    for ch in text.encode("utf-8"):
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h


def write_jsonl(items: Iterable[MCQItem | OpinionItem], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_mcq(path: str | Path) -> list[MCQItem]:
    return [MCQItem(**row) for row in _rows(path)]


def read_opinion(path: str | Path) -> list[OpinionItem]:
    return [OpinionItem(**row) for row in _rows(path)]


def _rows(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i}: not valid JSON ({e})") from e
    return rows
