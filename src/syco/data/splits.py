"""Deterministic dev/test splits.

The dev split is for choosing prompts, steering strengths and training settings.
The test split is only for the final numbers, so methods never see it while being tuned.
"""

from __future__ import annotations

from typing import Sequence, TypeVar

from syco.data.schema import _stable_hash

T = TypeVar("T")


def split_of(item_id: str, dev_fraction: float = 0.3, salt: str = "syco") -> str:
    bucket = _stable_hash(f"{salt}:{item_id}") % 10_000
    return "dev" if bucket < dev_fraction * 10_000 else "test"


def select(items: Sequence[T], split: str, dev_fraction: float = 0.3) -> list[T]:
    if split == "all":
        return list(items)
    if split not in ("dev", "test"):
        raise ValueError(f"split must be dev, test or all, got {split!r}")
    return [it for it in items if split_of(it.id, dev_fraction) == split]  # type: ignore[attr-defined]
