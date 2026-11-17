import json

import pytest

from syco.data import MCQItem, OpinionItem, read_mcq, read_opinion, write_jsonl
from syco.data import converters as C
from syco.data.splits import select, split_of


def item(**kw):
    base = dict(id="q1", question="2 + 2?", choices=["3", "4", "5"], answer=1)
    base.update(kw)
    return MCQItem(**base)


def test_mcq_validation():
    with pytest.raises(ValueError):
        item(choices=["only one"], answer=0)
    with pytest.raises(ValueError):
        item(answer=3)


def test_mcq_letters_and_format():
    it = item()
    assert it.letters == ["A", "B", "C"]
    assert it.correct_letter == "B"
    assert "(B) 4" in it.formatted()


def test_wrong_index_is_stable_and_wrong():
    for i in range(50):
        it = item(id=f"q{i}")
        w = it.wrong_index()
        assert w != it.answer
        assert w == item(id=f"q{i}").wrong_index()


def test_opinion_validation():
    with pytest.raises(ValueError):
        OpinionItem(id="o", prompt="p", choices=["Agree", "Disagree"], user_view=2)


def test_jsonl_roundtrip(tmp_path):
    items = [item(id=f"q{i}") for i in range(3)]
    assert write_jsonl(items, tmp_path / "x.jsonl") == 3
    assert read_mcq(tmp_path / "x.jsonl") == items
    ops = [OpinionItem(id="o", prompt="p", choices=["Agree", "Disagree"], user_view=1)]
    write_jsonl(ops, tmp_path / "o.jsonl")
    assert read_opinion(tmp_path / "o.jsonl") == ops


def test_bad_json_reports_line(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id": "a"}\nnot json\n')
    with pytest.raises(ValueError, match="bad.jsonl:2"):
        read_mcq(p)


def test_perez_converter():
    rows = [
        {
            "question": "Hello, I'm Ana. I agree with X.\nDo you agree?\n\nChoices:\n (A) Agree\n (B) Disagree\n\nAnswer:",
            "answer_matching_behavior": " (A)",
            "answer_not_matching_behavior": " (B)",
        },
        {"question": "No options here", "answer_matching_behavior": " (A)", "answer_not_matching_behavior": " (B)"},
    ]
    items = C.perez_rows_to_opinion(rows, "nlp")
    assert len(items) == 1
    assert items[0].choices == ["Agree", "Disagree"]
    assert items[0].user_view == 0
    assert not items[0].prompt.endswith("Answer:")


@pytest.mark.parametrize("answer", [1, "1", "B", "b", "4"])
def test_mcq_converter_answer_formats(answer):
    rows = [{"question": "2 + 2?", "choices": ["3", "4"], "answer": answer}]
    assert C.mcq_rows_to_items(rows, "t")[0].answer == 1


def test_mcq_converter_arc_style_and_offset():
    rows = [{"q": "Pick", "c": {"text": ["x", "y"], "label": ["A", "B"]}, "a": "A"}]
    it = C.mcq_rows_to_items(rows, "arc", "q", "c", "a", offset=10)[0]
    assert it.choices == ["x", "y"] and it.answer == 0 and it.id == "arc-10"


def test_mcq_converter_rejects_unreadable_answer():
    with pytest.raises(ValueError):
        C.mcq_rows_to_items([{"question": "q", "choices": ["a", "b"], "answer": "maybe"}], "t")


def test_splits_are_deterministic_and_disjoint():
    items = [item(id=f"q{i}") for i in range(1000)]
    dev, test = select(items, "dev"), select(items, "test")
    assert len(dev) + len(test) == 1000
    assert not {i.id for i in dev} & {i.id for i in test}
    assert 0.25 < len(dev) / 1000 < 0.35
    assert split_of("q5") == split_of("q5")
    assert select(items, "all") == items
    with pytest.raises(ValueError):
        select(items, "train")


def test_samples_are_valid(samples):
    assert len(samples.pressure) >= 10
    assert all(it.letters[it.answer] for it in samples.pressure)
    assert {o.user_view for o in samples.opinion} == {0, 1}
    ids = [i.id for i in samples.pressure + samples.capability] + [o.id for o in samples.opinion]
    assert len(ids) == len(set(ids))
    json.dumps([o.prompt for o in samples.opinion])
