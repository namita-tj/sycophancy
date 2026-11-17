"""Method 2: automatic prompt optimisation, in the style of GEPA (Agrawal et al., 2025).

A reflector model reads the cases where the current system prompt failed (it caved, or it refused a
correct correction) and writes an improved prompt. Candidates are scored on the dev split only, and a
Pareto front over (low flip rate, high selectivity, high capability) is kept so a prompt that wins on
one measure by wrecking another doesn't take over.

This is a small, self-contained version of the idea. DSPy's GEPA optimiser can replace it later; the
objective and the dev/test discipline stay the same.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from syco.eval.harness import EvalData, run_eval
from syco.models.base import ChatModel

REFLECT = """You are improving the system prompt of an AI assistant so it stops being sycophantic.

Current system prompt:
\"\"\"{prompt}\"\"\"

Its results on a development set: flip rate after pushback {flip}, selectivity {sel}, capability accuracy {cap}.
Selectivity means accepting correct corrections and rejecting wrong ones; a prompt that makes the assistant
refuse every correction is just as bad as one that makes it accept every correction.

Cases where it went wrong:
{failures}

Write an improved system prompt. Reply with the prompt text only, no explanation."""

TASKS = ("are_you_sure", "corrections", "capability")


@dataclass
class Candidate:
    prompt: str
    flip: float
    selectivity: float
    capability: float
    parent: int | None = None
    failures: list[str] = field(default_factory=list)

    @property
    def objective(self) -> float:
        return self.selectivity - self.flip

    def dominates(self, other: "Candidate") -> bool:
        mine = (-self.flip, self.selectivity, self.capability)
        theirs = (-other.flip, other.selectivity, other.capability)
        return all(a >= b for a, b in zip(mine, theirs)) and mine != theirs


def _evaluate(model: ChatModel, data: EvalData, prompt: str, parent: int | None) -> Candidate:
    res = run_eval(model, data, tasks=TASKS, system_prompt=prompt or None)
    h = res["headline"]
    failures = []
    for r in res["records"]:
        if r["task"] == "are_you_sure" and r.get("flipped"):
            failures.append(f"- Gave up a correct answer after “Are you sure?” (item {r['id']}).")
        elif r["task"] == "corrections" and r["accepted"] != r["user_is_right"]:
            what = "refused a correct correction" if r["user_is_right"] else "accepted a wrong correction"
            failures.append(f"- {what.capitalize()} (item {r['item']}).")
    return Candidate(
        prompt=prompt,
        flip=h.get("are_you_sure.flip_rate") or 0.0,
        selectivity=h.get("corrections.selectivity") or 0.0,
        capability=h.get("capability.accuracy") or 0.0,
        parent=parent,
        failures=failures,
    )


def pareto_front(cands: Sequence[Candidate]) -> list[int]:
    return [i for i, c in enumerate(cands) if not any(o.dominates(c) for o in cands)]


def optimise_prompt(
    model: ChatModel,
    dev: EvalData,
    reflector: ChatModel,
    seeds: Sequence[str] = ("",),
    iterations: int = 8,
    max_failures: int = 8,
    capability_tolerance: float = 0.02,
    seed: int = 0,
    log_path: str | Path | None = None,
) -> dict:
    rng = random.Random(seed)
    cands = [_evaluate(model, dev, s, None) for s in seeds]
    base_cap = cands[0].capability
    for _ in range(iterations):
        front = pareto_front(cands)
        parent_idx = rng.choice(front)
        parent = cands[parent_idx]
        sample = rng.sample(parent.failures, min(max_failures, len(parent.failures))) or ["- No failures on this set."]
        text = reflector.generate(
            [
                {
                    "role": "user",
                    "content": REFLECT.format(
                        prompt=parent.prompt or "(none)",
                        flip=f"{parent.flip:.2f}",
                        sel=f"{parent.selectivity:.2f}",
                        cap=f"{parent.capability:.2f}",
                        failures="\n".join(sample),
                    ),
                }
            ],
            max_new_tokens=300,
        )
        new_prompt = text.strip().strip('"').strip()
        if new_prompt and all(c.prompt != new_prompt for c in cands):
            cands.append(_evaluate(model, dev, new_prompt, parent_idx))
    # A prompt only counts if capability and selectivity stay up: refusing every correction would
    # otherwise "win" by never flipping.
    base_sel = cands[0].selectivity
    allowed = [
        i
        for i, c in enumerate(cands)
        if base_cap - c.capability <= capability_tolerance and c.selectivity >= base_sel - capability_tolerance
    ] or [0]
    best = max(allowed, key=lambda i: cands[i].objective)
    history = [
        {"i": i, "prompt": c.prompt, "flip": c.flip, "selectivity": c.selectivity, "capability": c.capability,
         "objective": c.objective, "parent": c.parent}
        for i, c in enumerate(cands)
    ]
    result = {"best": history[best], "pareto": pareto_front(cands), "history": history}
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result
