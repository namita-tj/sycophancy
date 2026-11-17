import numpy as np
import pytest

from syco.eval.judge import flattery_eval, judge_reply
from syco.interp import directions, probes
from syco.interp.sae import SAEWeights, rank_features
from syco.methods import steering


def test_probe_finds_signal_in_late_layers(make_fake, samples):
    model = make_fake(sycophancy=0.5, knowledge=0.7)
    items = samples.pressure * 3
    feats, labels = probes.collect(model, items, [0, 7])
    res = probes.probe_layers(feats, labels, folds=3)
    assert "error" not in res, res
    assert res["layers"][7]["balanced_accuracy"] > 0.8


def test_probe_reports_single_class():
    feats = {0: np.zeros((4, 3))}
    res = probes.probe_layers(feats, np.array([1, 1, 1, 1]))
    assert "error" in res


def test_direction_comparison(make_fake, samples):
    model = make_fake()
    pairs = steering.contrastive_pairs(samples.pressure)
    vecs = steering.build_vectors(model, pairs, [5])
    same = directions.compare(vecs, vecs)
    assert same[5]["cosine"] == pytest.approx(1.0)
    proj = directions.projections(model, pairs, vecs[5], 5)
    assert proj["separation"] > 0
    assert directions.cosine(np.zeros(3), np.ones(3)) == 0.0


def test_sae_encode_and_rank(make_fake, samples):
    model = make_fake()
    d = model.dim
    w = np.zeros((d, 4), dtype=np.float32)
    w[:, 2] = model._direction  # feature 2 reads the planted sycophancy direction
    w[:, 0] = -np.asarray(model._direction)
    sae = SAEWeights(W_enc=w, b_enc=np.zeros(4, dtype=np.float32), threshold=np.full(4, 0.1, dtype=np.float32))
    assert sae.encode(np.zeros(d)).sum() == 0
    top = rank_features(model, sae, steering.contrastive_pairs(samples.pressure), layer=7, top=2)
    assert top[0]["feature"] == 2 and top[0]["difference"] > 0
    relu = SAEWeights(W_enc=np.eye(2, dtype=np.float32), b_enc=np.zeros(2, dtype=np.float32))
    assert relu.encode(np.array([-1.0, 2.0])).tolist() == [0.0, 2.0]


def test_judge(make_fake):
    m = make_fake()
    assert judge_reply(m, "I quit my job to day-trade.", "Great idea!") in {1, 2, 3, 4, 5}
    res = flattery_eval(m, [m, m], ["Is my plan good?", "Rate my poem."])
    assert res["n_scored"] == 2 and res["kappa"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        flattery_eval(m, [m], ["x"])
