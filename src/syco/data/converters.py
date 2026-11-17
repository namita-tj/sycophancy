"""Convert public benchmarks into MCQItem / OpinionItem.

Supported:
- Perez et al. (2022) sycophancy files from github.com/anthropics/evals (`sycophancy/*.jsonl`):
  each row has `question`, `answer_matching_behavior` and `answer_not_matching_behavior`.
- Any multiple-choice JSONL or Hugging Face dataset with a question, a list of choices and an answer
  given as an index or a letter (MMLU, ARC, TruthfulQA MC1 after flattening, ...).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from syco.data.schema import LETTERS, MCQItem, OpinionItem

_OPTION_RE = re.compile(r"^\s*\(([A-J])\)\s*(.+?)\s*$", re.MULTILINE)


def perez_rows_to_opinion(rows: Iterable[dict], source: str) -> list[OpinionItem]:
    """Perez et al. format: the options are written inside the question text as ' (A) ...'."""
    items = []
    for i, row in enumerate(rows):
        prompt = row["question"].strip()
        options = _OPTION_RE.findall(prompt)
        if len(options) < 2:
            continue
        letters = [l for l, _ in options]
        choices = [text for _, text in options]
        matching = _first_letter(row["answer_matching_behavior"])
        if matching not in letters:
            continue
        # Drop the trailing "Answer:" cue, the harness adds its own.
        prompt = re.sub(r"\s*Answer:\s*$", "", prompt)
        items.append(
            OpinionItem(
                id=f"{source}-{i}",
                prompt=prompt,
                choices=choices,
                user_view=letters.index(matching),
                source=source,
            )
        )
    return items


def mcq_rows_to_items(
    rows: Iterable[dict],
    source: str,
    question_field: str = "question",
    choices_field: str = "choices",
    answer_field: str = "answer",
    id_field: str | None = None,
    offset: int = 0,
) -> list[MCQItem]:
    items = []
    for i, row in enumerate(rows):
        choices = _as_choices(row[choices_field])
        answer = _as_index(row[answer_field], choices)
        item_id = str(row[id_field]) if id_field else f"{source}-{offset + i}"
        meta = {k: row[k] for k in ("subject", "category") if k in row}
        items.append(
            MCQItem(
                id=item_id,
                question=str(row[question_field]).strip(),
                choices=choices,
                answer=answer,
                source=source,
                meta=meta,
            )
        )
    return items


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _first_letter(text: str) -> str:
    m = re.search(r"[A-J]", text)
    if not m:
        raise ValueError(f"no option letter in {text!r}")
    return m.group(0)


def _as_choices(value: Any) -> list[str]:
    if isinstance(value, dict) and "text" in value:  # ARC style: {"text": [...], "label": [...]}
        value = value["text"]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"choices must be a list, got {type(value).__name__}")
    return [str(c).strip() for c in value]


def _as_index(value: Any, choices: list[str]) -> int:
    if isinstance(value, bool):
        raise ValueError("answer must be an index or a letter, not a bool")
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit() and int(text) < len(choices):
        return int(text)
    if len(text) == 1 and text.upper() in LETTERS[: len(choices)]:
        return LETTERS.index(text.upper())
    if text in choices:
        return choices.index(text)
    raise ValueError(f"cannot read answer {value!r}")
