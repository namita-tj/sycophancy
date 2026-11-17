import re

from syco.data import LETTERS, MCQItem
from syco.data import synthetic
from syco.eval import prompts as P


def mcq():
    return MCQItem(id="q", question="Capital of France?", choices=["Paris", "Rome", "Oslo"], answer=0)


def test_correction_user_right_and_wrong():
    it = mcq()
    right = P.correction(it, user_is_right=True)
    wrong = P.correction(it, user_is_right=False)
    w = LETTERS[it.wrong_index()]
    assert f"({w})" in right[1]["content"] and "(A)" in right[2]["content"]
    assert "(A)" in wrong[1]["content"] and f"({w})" in wrong[2]["content"]


def test_with_system_merges_existing():
    msgs = [{"role": "system", "content": "old"}, {"role": "user", "content": "hi"}]
    out = P.with_system(msgs, "new")
    assert out[0]["content"].startswith("new") and "old" in out[0]["content"]
    assert P.with_system(msgs, None) == msgs


def test_are_you_sure_shape():
    msgs = P.are_you_sure(mcq(), "A")
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert P.ARE_YOU_SURE in msgs[-1]["content"]


def _ngrams(text, n=4):
    words = re.findall(r"[a-z']+", text.lower().replace("{wrong}", "").replace("{suggested}", "").replace("{x}", ""))
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def test_train_and_eval_phrasings_do_not_overlap():
    eval_grams = set().union(*(_ngrams(t) for t in P.EVAL_PHRASINGS + [P.ANSWER_CUE, P.OPINION_CUE]))
    train_grams = set().union(*(_ngrams(t) for t in synthetic.TRAIN_PHRASINGS))
    assert not eval_grams & train_grams


def _truth(question):
    m = re.match(r"Is (\d+) \+ (\d+) = (\d+)\?", question)
    if m:
        a, b, c = map(int, m.groups())
        return "A" if a + b == c else "B"
    m = re.match(r"Is (\d+) × (\d+) = (\d+)\?", question)
    if m:
        a, b, c = map(int, m.groups())
        return "A" if a * b == c else "B"
    m = re.match(r"Which number is larger: (\d+) or (\d+)\?", question)
    if m:
        a, b = map(int, m.groups())
        return "A" if a > b else "B"
    m = re.match(r"Is (\d+) an even number\?", question)
    return "A" if int(m.group(1)) % 2 == 0 else "B"


def test_synthetic_targets_are_true_and_balanced():
    ex = synthetic.generate(400, seed=3)
    assert sum(e["user_is_right"] for e in ex) == 200
    for e in ex:
        question = e["messages"][0]["content"].split("\n")[0]
        assert e["chosen"] == synthetic.TARGET.format(x=_truth(question))
        assert e["chosen"] != e["rejected"]


def test_synthetic_is_deterministic(tmp_path):
    a, b = synthetic.generate(20, seed=1), synthetic.generate(20, seed=1)
    assert a == b
    synthetic.write(a, tmp_path / "s.jsonl")
    assert synthetic.read(tmp_path / "s.jsonl") == a
