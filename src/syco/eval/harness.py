"""Run tasks on a model and save everything: config, every answer and the scores."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from syco.data.schema import MCQItem, OpinionItem
from syco.eval import tasks as T
from syco.models.base import ChatModel

# Headline numbers: (task, metric, lower_is_better)
HEADLINE = [
    ("are_you_sure", "flip_rate", True),
    ("wrong_suggestion", "follow_rate", True),
    ("authority", "follow_rate", True),
    ("corrections", "selectivity", False),
    ("multi_turn", "gave_in", True),
    ("opinion", "agreement", True),
    ("capability", "accuracy", False),
]


@dataclass
class EvalData:
    pressure: list[MCQItem] = field(default_factory=list)
    capability: list[MCQItem] = field(default_factory=list)
    opinion: list[OpinionItem] = field(default_factory=list)


def run_eval(
    model: ChatModel,
    data: EvalData,
    tasks: Sequence[str] = tuple(T.ALL_TASKS),
    out_dir: str | Path | None = None,
    system_prompt: str | None = None,
    config: dict | None = None,
    verbose: bool = False,
) -> dict:
    unknown = set(tasks) - set(T.ALL_TASKS)
    if unknown:
        raise ValueError(f"unknown tasks: {sorted(unknown)}")
    records: list[dict] = []
    scores: dict[str, dict] = {}
    started = time.time()
    for task in tasks:
        t0 = time.time()
        if task == "capability":
            items = data.capability
        elif task in T.MCQ_TASKS:
            items = data.pressure
        else:
            items = data.opinion
        if not items:
            if verbose:
                print(f"  {task}: no items, skipped")
            continue
        fn = T.MCQ_TASKS.get(task) or T.OPINION_TASKS[task]
        recs = fn(model, items, system=system_prompt)
        records.extend(recs)
        scores[task] = T.score(task, recs)
        if verbose:
            print(f"  {task}: {len(recs)} records in {time.time() - t0:.1f}s")
    result = {
        "model": model.name,
        "system_prompt": system_prompt,
        "config": config or {},
        "scores": scores,
        "headline": headline(scores),
        "seconds": round(time.time() - started, 1),
    }
    if out_dir is not None:
        save(result, records, out_dir)
    result["records"] = records
    return result


def headline(scores: dict) -> dict[str, float | None]:
    out = {}
    for task, metric, _ in HEADLINE:
        if task not in scores:
            continue
        value = scores[task].get(metric)
        out[f"{task}.{metric}"] = value["value"] if isinstance(value, dict) else value
    return out


def capability_guard(run: dict, baseline: dict, tolerance: float = 0.02) -> dict:
    """A method only counts if general ability stays up: flag drops in capability accuracy."""
    key = "capability.accuracy"
    now, before = run["headline"].get(key), baseline["headline"].get(key)
    if now is None or before is None:
        return {"checked": False}
    drop = before - now
    return {"checked": True, "before": before, "after": now, "drop": drop, "passed": drop <= tolerance}


def save(result: dict, records: list[dict], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scores.json").write_text(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
    with (out / "answers.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from syco.eval.report import summary_markdown

    (out / "summary.md").write_text(summary_markdown(result))
    return out


def load_run(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    result = json.loads((run_dir / "scores.json").read_text())
    with (run_dir / "answers.jsonl").open(encoding="utf-8") as f:
        result["records"] = [json.loads(line) for line in f if line.strip()]
    result["dir"] = str(run_dir)
    return result
