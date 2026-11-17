import numpy as np

from syco.eval.harness import EvalData, run_eval
from syco.methods import gepa, steering
from syco.methods.prompting import HONESTY_PROMPTS, compare_prompts


def test_honesty_prompt_helps_and_stubborn_prompt_hurts_selectivity(make_fake, samples):
    res = compare_prompts(make_fake(sycophancy=0.8), samples, tasks=["are_you_sure", "corrections"])
    flip = {k: v["headline"]["are_you_sure.flip_rate"] for k, v in res.items()}
    sel = {k: v["headline"]["corrections.selectivity"] for k, v in res.items()}
    assert flip["honest"] <= flip["none"]
    assert flip["stubborn"] == 0.0
    assert sel["stubborn"] == 0.5  # never changing its mind is not honesty
    assert sel["honest"] > sel["stubborn"]
    assert set(res) == set(HONESTY_PROMPTS)


def test_pareto_front():
    a = gepa.Candidate("a", flip=0.1, selectivity=0.9, capability=0.8)
    b = gepa.Candidate("b", flip=0.2, selectivity=0.8, capability=0.8)
    c = gepa.Candidate("c", flip=0.0, selectivity=0.5, capability=0.8)
    assert a.dominates(b) and not b.dominates(a)
    assert gepa.pareto_front([a, b, c]) == [0, 2]


def test_optimise_prompt_improves_on_dev_and_logs(tmp_path, make_fake, samples):
    model = make_fake(sycophancy=0.8)
    res = gepa.optimise_prompt(model, samples, model, seeds=[""], iterations=6, log_path=tmp_path / "g.json")
    assert (tmp_path / "g.json").exists()
    base = res["history"][0]
    assert res["best"]["objective"] >= base["objective"]
    assert "Never change" not in res["best"]["prompt"]  # the stubborn candidate loses on selectivity


def test_steering_vector_finds_the_planted_direction(make_fake, samples):
    model = make_fake(sycophancy=0.8)
    pairs = steering.contrastive_pairs(samples.pressure, samples.opinion)
    assert len(pairs) == len(samples.pressure) + len(samples.opinion)
    vecs = steering.build_vectors(model, pairs, [1, 7])
    planted = np.asarray(model._direction)
    cos = float(vecs[7] @ planted / np.linalg.norm(vecs[7]))
    assert cos > 0.9
    assert np.linalg.norm(vecs[7]) > np.linalg.norm(vecs[1])


def test_steering_sweep_reduces_flips_and_resets(tmp_path, make_fake, samples):
    model = make_fake(sycophancy=0.8)
    vecs = steering.build_vectors(model, steering.contrastive_pairs(samples.pressure), [7])
    steering.save_vectors(vecs, tmp_path / "v.npz")
    loaded = steering.load_vectors(tmp_path / "v.npz")
    assert np.allclose(loaded[7], vecs[7])
    res = steering.sweep(model, samples, loaded, alphas=[-0.5, -1.0, 0.5])
    assert res["best"]["alpha"] < 0
    assert res["best"]["are_you_sure.flip_rate"] < res["baseline"]["are_you_sure.flip_rate"]
    assert model._steer is None
    steering.apply(model, loaded, 7, -1.0)
    steered = run_eval(model, EvalData(pressure=samples.pressure), tasks=["are_you_sure"])
    assert steered["headline"]["are_you_sure.flip_rate"] == 0.0
