import pytest

from syco.eval.metrics import Rate, cohen_kappa, paired_bootstrap, rate, selectivity


def test_rate_and_wilson():
    r = rate([True, False, True, True])
    assert r.value == 0.75
    lo, hi = r.ci()
    assert 0 <= lo < 0.75 < hi <= 1
    assert Rate(0, 0).value is None and Rate(0, 0).ci() is None
    lo, hi = Rate(0, 20).ci()
    assert lo == 0 and hi > 0


def test_selectivity_extremes():
    always_agree = selectivity(Rate(10, 10), Rate(0, 10))
    never_change = selectivity(Rate(0, 10), Rate(10, 10))
    perfect = selectivity(Rate(10, 10), Rate(10, 10))
    assert always_agree == never_change == 0.5
    assert perfect == 1.0
    assert selectivity(Rate(0, 0), Rate(1, 1)) is None


def test_paired_bootstrap():
    before = {str(i): 1.0 for i in range(50)}
    same = paired_bootstrap(before, dict(before))
    assert same["delta"] == 0 and same["ci95"] == [0, 0]
    after = {str(i): 0.0 for i in range(50)}
    res = paired_bootstrap(before, after)
    assert res["delta"] == -1 and res["n"] == 50
    assert paired_bootstrap({"a": 1}, {"b": 1})["n"] == 0


def test_cohen_kappa():
    assert cohen_kappa([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert cohen_kappa([1, 1, 2, 2], [2, 2, 1, 1]) < 0
    assert cohen_kappa([], []) is None
    assert cohen_kappa([3, 3], [3, 3]) == 1.0
