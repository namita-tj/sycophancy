"""Markdown summaries and comparisons between runs."""

from __future__ import annotations

from syco.eval import tasks as T
from syco.eval.harness import HEADLINE, capability_guard
from syco.eval.metrics import paired_bootstrap

LABELS = {
    "are_you_sure.flip_rate": "Flips after “Are you sure?”",
    "wrong_suggestion.follow_rate": "Follows a wrong suggestion",
    "authority.follow_rate": "Follows a wrong “expert”",
    "corrections.selectivity": "Selectivity (higher is better)",
    "multi_turn.gave_in": "Gives in under repeated pressure",
    "opinion.agreement": "Agrees with the user’s opinion",
    "capability.accuracy": "Capability accuracy (higher is better)",
}


def _fmt(v: float | None) -> str:
    return "–" if v is None else f"{100 * v:.1f}%"


def summary_markdown(result: dict) -> str:
    lines = [f"# {result['model']}", ""]
    if result.get("system_prompt"):
        lines += [f"System prompt: “{result['system_prompt']}”", ""]
    lines += ["| Measure | Value | 95% CI | n |", "| --- | ---: | ---: | ---: |"]
    for task, metric, _ in HEADLINE:
        block = result["scores"].get(task)
        if not block:
            continue
        value = block.get(metric)
        if isinstance(value, dict):
            ci = value.get("ci95")
            ci_text = f"{100 * ci[0]:.1f}–{100 * ci[1]:.1f}%" if ci else "–"
            lines.append(f"| {LABELS[f'{task}.{metric}']} | {_fmt(value['value'])} | {ci_text} | {value['n']} |")
        else:
            lines.append(f"| {LABELS[f'{task}.{metric}']} | {_fmt(value)} | – | – |")
    mt = result["scores"].get("multi_turn")
    if mt and mt.get("survival_by_turn"):
        curve = ", ".join(_fmt(v) for v in mt["survival_by_turn"])
        lines += ["", f"Still correct after each turn of pressure: {curve}"]
    return "\n".join(lines) + "\n"


def compare_markdown(runs: list[dict], baseline: dict | None = None, tolerance: float = 0.02) -> str:
    """One row per measure, one column per run. With a baseline, adds the paired change and its interval."""
    baseline = baseline or runs[0]
    names = [r.get("label") or r["model"] for r in runs]
    lines = ["| Measure | " + " | ".join(names) + " |", "| --- |" + " ---: |" * len(runs)]
    for task, metric, lower_better in HEADLINE:
        key = f"{task}.{metric}"
        if not any(key in r["headline"] for r in runs):
            continue
        cells = []
        for r in runs:
            v = r["headline"].get(key)
            cell = _fmt(v)
            if r is not baseline and v is not None and task in baseline["scores"]:
                d = _paired(task, baseline, r)
                if d["delta"] is not None:
                    sign = "+" if d["delta"] >= 0 else ""
                    lo, hi = d["ci95"]
                    good = (d["delta"] < 0) == lower_better if task != "corrections" else d["delta"] > 0
                    marker = " ✓" if good and (hi < 0 or lo > 0) else ""
                    cell += f" ({sign}{100 * d['delta']:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]){marker}"
            cells.append(cell)
        lines.append(f"| {LABELS[key]} | " + " | ".join(cells) + " |")
    guard_rows = []
    for r in runs:
        if r is baseline:
            continue
        g = capability_guard(r, baseline, tolerance)
        if g.get("checked"):
            status = "passed" if g["passed"] else f"FAILED (dropped {100 * g['drop']:.1f} points)"
            guard_rows.append(f"- {r.get('label') or r['model']}: capability guard {status}")
    text = "\n".join(lines) + "\n"
    if guard_rows:
        text += "\n" + "\n".join(guard_rows) + "\n"
    text += (
        "\nChanges are paired against the first run, on the items both runs share, with a 95% bootstrap interval. "
        "✓ marks an improvement whose interval excludes zero.\n"
    )
    return text


def _paired(task: str, before: dict, after: dict) -> dict:
    a = T.item_scores(task, [r for r in before["records"] if r["task"] == task])
    b = T.item_scores(task, [r for r in after["records"] if r["task"] == task])
    return paired_bootstrap(a, b)
