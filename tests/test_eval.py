import json

import pytest

from syco.eval import tasks as T
from syco.eval.harness import EvalData, capability_guard, load_run, run_eval
from syco.eval.report import compare_markdown, summary_markdown


def test_fake_without_sycophancy_never_caves(make_fake, samples):
    m = make_fake(sycophancy=0.0)
    s = T.score("are_you_sure", T.are_you_sure(m, samples.pressure))
    assert s["initial_accuracy"]["value"] == 1.0
    assert s["flip_rate"]["value"] == 0.0
    sel = T.score("corrections", T.corrections(m, samples.pressure))
    assert sel["selectivity"] == 1.0


def test_fully_sycophantic_fake_caves(make_fake, samples):
    m = make_fake(sycophancy=1.0)
    assert T.score("are_you_sure", T.are_you_sure(m, samples.pressure))["flip_rate"]["value"] == 1.0
    assert T.score("wrong_suggestion", T.wrong_suggestion(m, samples.pressure))["follow_rate"]["value"] == 1.0
    assert T.score("opinion", T.opinion(m, samples.opinion))["agreement"]["value"] == 1.0
    assert T.score("corrections", T.corrections(m, samples.pressure))["selectivity"] == 0.5


def test_harness_measures_roughly_the_planted_rate(make_fake, samples):
    m = make_fake(sycophancy=0.5)
    big = EvalData(pressure=samples.pressure * 1, opinion=samples.opinion)
    res = run_eval(m, big, tasks=["are_you_sure", "opinion"])
    assert 0.1 < res["headline"]["are_you_sure.flip_rate"] < 0.9


def test_multi_turn_survival_is_non_increasing(make_fake, samples):
    s = T.score("multi_turn", T.multi_turn(make_fake(sycophancy=0.3), samples.pressure))
    curve = s["survival_by_turn"]
    assert len(curve) == len(T.P.ESCALATION)
    assert all(a >= b for a, b in zip(curve, curve[1:]))


def test_run_eval_saves_and_loads(tmp_path, make_fake, samples):
    res = run_eval(make_fake(), samples, out_dir=tmp_path / "run", system_prompt="Be honest.")
    for name in ("scores.json", "answers.jsonl", "summary.md"):
        assert (tmp_path / "run" / name).exists()
    loaded = load_run(tmp_path / "run")
    assert loaded["headline"] == res["headline"]
    assert len(loaded["records"]) == len(res["records"])
    assert set(loaded["scores"]) == set(T.ALL_TASKS)
    assert "records" not in json.loads((tmp_path / "run" / "scores.json").read_text())


def test_run_eval_rejects_unknown_task_and_skips_empty(make_fake, samples):
    with pytest.raises(ValueError):
        run_eval(make_fake(), samples, tasks=["nope"])
    res = run_eval(make_fake(), EvalData(pressure=samples.pressure), tasks=["capability", "are_you_sure"])
    assert "capability" not in res["scores"] and "are_you_sure" in res["scores"]


def test_capability_guard():
    base = {"headline": {"capability.accuracy": 0.80}}
    assert capability_guard({"headline": {"capability.accuracy": 0.79}}, base)["passed"]
    assert not capability_guard({"headline": {"capability.accuracy": 0.70}}, base)["passed"]
    assert capability_guard({"headline": {}}, base) == {"checked": False}


def test_reports(make_fake, samples):
    a = run_eval(make_fake(sycophancy=0.9, name="syc"), samples)
    b = run_eval(make_fake(sycophancy=0.0, name="honest"), samples)
    assert "Selectivity" in summary_markdown(a)
    text = compare_markdown([a, b])
    assert "syc" in text and "honest" in text
    assert "✓" in text  # the honest model is a clear improvement on the paired items
    assert "capability guard passed" in text


def test_item_scores_cover_every_task(make_fake, samples):
    res = run_eval(make_fake(), samples)
    for task in T.ALL_TASKS:
        recs = [r for r in res["records"] if r["task"] == task]
        assert T.item_scores(task, recs)
