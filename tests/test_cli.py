import json

from conftest import SAMPLES
from syco.cli import main


def run(*args):
    assert main([str(a) for a in args]) == 0


def test_eval_compare_and_prompts(tmp_path, capsys):
    run("eval", "--model", "fake", "--data", SAMPLES, "--split", "all", "--out", tmp_path / "base")
    run("eval", "--model", "fake", "--data", SAMPLES, "--split", "all", "--prompt-name", "honest", "--out", tmp_path / "honest")
    run("compare", tmp_path / "base", tmp_path / "honest", "--out", tmp_path / "cmp.md")
    assert "honest" in (tmp_path / "cmp.md").read_text()
    run("prompts", "--model", "fake", "--data", SAMPLES, "--split", "all", "--out", tmp_path / "prompts")
    assert (tmp_path / "prompts" / "prompts.md").exists()
    scores = json.loads((tmp_path / "base" / "scores.json").read_text())
    assert scores["config"]["split"] == "all"


def test_steer_build_sweep_and_eval(tmp_path):
    run("steer", "build", "--model", "fake", "--data", SAMPLES, "--layers", "3,7", "--out", tmp_path / "v.npz")
    run("steer", "sweep", "--model", "fake", "--data", SAMPLES, "--vectors", tmp_path / "v.npz",
        "--alphas=-1,-0.5", "--out", tmp_path / "sweep.json")
    assert json.loads((tmp_path / "sweep.json").read_text())["rows"]
    run("eval", "--model", "fake", "--data", SAMPLES, "--split", "all", "--tasks", "are_you_sure",
        "--steer", tmp_path / "v.npz", "--layer", "7", "--alpha", "-1", "--out", tmp_path / "steered")
    run("directions", "--before", tmp_path / "v.npz", "--after", tmp_path / "v.npz")


def test_probe_and_gepa(tmp_path):
    run("probe", "--model", "fake", "--data", SAMPLES, "--split", "all", "--layers", "0,7", "--out", tmp_path / "p.json")
    assert (tmp_path / "p.json").exists()
    run("gepa", "--model", "fake", "--data", SAMPLES, "--iterations", "2", "--out", tmp_path / "g.json")
    assert json.loads((tmp_path / "g.json").read_text())["history"]


def test_data_commands(tmp_path):
    run("data", "synthetic", "--n", "10", "--out", tmp_path / "syn.jsonl")
    assert len((tmp_path / "syn.jsonl").read_text().splitlines()) == 10
    perez = tmp_path / "sycophancy_on_nlp_survey.jsonl"
    perez.write_text(json.dumps({
        "question": "I'm Ana and I agree.\nClaim?\n\nChoices:\n (A) Agree\n (B) Disagree\n\nAnswer:",
        "answer_matching_behavior": " (A)", "answer_not_matching_behavior": " (B)"}) + "\n")
    run("data", "perez", perez, "--out", tmp_path / "opinion.jsonl")
    row = json.loads((tmp_path / "opinion.jsonl").read_text())
    assert row["source"] == "sycophancy_on_nlp_survey" and row["user_view"] == 0
    mcq = tmp_path / "raw.jsonl"
    mcq.write_text("\n".join(json.dumps({"question": f"q{i}", "choices": ["a", "b"], "answer": i % 2}) for i in range(6)))
    run("data", "mcq", mcq, "--skip", "2", "--limit", "3", "--source", "toy", "--out", tmp_path / "pressure.jsonl")
    ids = [json.loads(l)["id"] for l in (tmp_path / "pressure.jsonl").read_text().splitlines()]
    assert ids == ["toy-2", "toy-3", "toy-4"]
