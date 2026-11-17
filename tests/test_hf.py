"""The Hugging Face backend and training loops, on a tiny randomly initialised model.

No download needed: the tokenizer is trained on the fly and the model is a two-layer Llama.
Skipped when torch, transformers or peft aren't installed.
"""

import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from syco.data import synthetic  # noqa: E402
from syco.eval.harness import EvalData, run_eval  # noqa: E402
from syco.methods import finetune, steering  # noqa: E402
from syco.models.base import pick  # noqa: E402
from syco.models.hf import HFModel  # noqa: E402

pytestmark = pytest.mark.torch

TEMPLATE = (
    "{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}</s>{% endfor %}"
    "{% if add_generation_prompt %}<assistant>{% endif %}"
)
NO_SYSTEM_TEMPLATE = (
    "{% for m in messages %}{% if m['role'] == 'system' %}{{ raise_exception('no system role') }}{% endif %}"
    "<{{ m['role'] }}>{{ m['content'] }}</s>{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}"
)


def _tokenizer(template=TEMPLATE):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    corpus = [
        "The answer is (A). The answer is (B). Are you sure? I think that's wrong.",
        "What is the capital of Australia? Sydney Canberra Melbourne Perth",
        "<user><assistant><system></s> Answer with the letter of one option.",
    ] * 20
    trainer = trainers.BpeTrainer(
        vocab_size=400, special_tokens=["<unk>", "<pad>", "</s>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tok.train_from_iterator(corpus, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", eos_token="</s>")
    fast.chat_template = template
    return fast


def _model(template=TEMPLATE, seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    tok = _tokenizer(template)
    cfg = LlamaConfig(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=1024,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
    )
    return HFModel(LlamaForCausalLM(cfg), tok, name="tiny-llama")


@pytest.fixture
def tiny():
    return _model()


def test_choice_logprobs_and_pick(tiny):
    msgs = [{"role": "user", "content": "What is the capital of Australia?"}]
    lp = tiny.choice_logprobs(msgs, "The answer is (", ["A", "B", "C", "D"])
    assert len(lp) == 4 and all(math.isfinite(x) and x < 0 for x in lp)
    idx, probs = pick(tiny, msgs, "The answer is (", ["A", "B"])
    assert idx in (0, 1) and sum(probs) == pytest.approx(1.0)


def test_render_fallbacks():
    folded = _model(NO_SYSTEM_TEMPLATE)
    text = folded.render([{"role": "system", "content": "Be honest."}, {"role": "user", "content": "Hi"}], "The answer is (")
    assert "Be honest." in text and text.endswith("The answer is (")
    plain = _model()
    plain.tokenizer.chat_template = None
    assert plain.render([{"role": "user", "content": "Hi"}]).endswith("Assistant: ")


def test_generate_and_activations(tiny):
    out = tiny.generate([{"role": "user", "content": "Hi"}], max_new_tokens=5)
    assert isinstance(out, str)
    acts = tiny.activations([{"role": "user", "content": "Hi"}], "The answer is (A", [0, 1])
    assert set(acts) == {0, 1} and len(acts[0]) == 32


def test_steering_changes_and_resets(tiny):
    msgs = [{"role": "user", "content": "Are you sure?"}]
    before = tiny.choice_logprobs(msgs, "The answer is (", ["A", "B"])
    tiny.set_steering(1, [1.0] * 32, alpha=5.0)
    steered = tiny.choice_logprobs(msgs, "The answer is (", ["A", "B"])
    tiny.set_steering(None, None)
    after = tiny.choice_logprobs(msgs, "The answer is (", ["A", "B"])
    assert steered != pytest.approx(before)
    assert after == pytest.approx(before)


def test_full_eval_and_steering_vectors_on_tiny_model(tiny, samples):
    data = EvalData(pressure=samples.pressure[:3], capability=samples.capability[:2], opinion=samples.opinion[:2])
    res = run_eval(tiny, data)
    assert "corrections.selectivity" in res["headline"]
    vecs = steering.build_vectors(tiny, steering.contrastive_pairs(data.pressure, data.opinion), [0, 1])
    assert vecs[1].shape == (32,)


def test_sft_trains_and_saves_adapter(tiny, tmp_path):
    ex = synthetic.generate(16, seed=0)
    out = finetune.train_sft(tiny, ex, tmp_path / "sft", batch_size=4, lr=1e-2, r=4, alpha=8, target_modules=["q_proj", "v_proj"])
    assert (out / "adapter_config.json").exists()
    log = (out / "train_log.json").read_text()
    assert '"sft-lora"' in log
    # The wrapped model still works for evaluation and steering.
    tiny.set_steering(0, [0.5] * 32, alpha=1.0)
    assert len(tiny.choice_logprobs([{"role": "user", "content": "x"}], "The answer is (", ["A", "B"])) == 2
    tiny.set_steering(None, None)


def test_sft_reduces_loss(tmp_path):
    hf = _model(seed=1)
    ex = synthetic.generate(8, seed=0)
    out = finetune.train_sft(hf, ex, tmp_path / "sft", epochs=15, batch_size=8, lr=3e-2, r=8, alpha=16,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    import json

    losses = json.loads((out / "train_log.json").read_text())["losses"]
    assert losses[-1] < losses[0]


def test_dpo_starts_at_log2_and_learns(tmp_path):
    hf = _model(seed=2)
    ex = synthetic.generate(8, seed=0)
    out = finetune.train_dpo(hf, ex, tmp_path / "dpo", epochs=10, batch_size=8, lr=3e-2, beta=0.5, r=8, alpha=16,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    import json

    log = json.loads((out / "train_log.json").read_text())
    assert log["losses"][0] == pytest.approx(math.log(2), abs=1e-4)  # policy == reference at the start
    assert log["losses"][-1] < log["losses"][0]


def test_dpo_loss_matches_formula():
    pc, pr, rc, rr = (torch.tensor([v]) for v in (-1.0, -3.0, -2.0, -2.0))
    loss = finetune.dpo_loss(pc, pr, rc, rr, beta=0.5)
    expected = -math.log(1 / (1 + math.exp(-0.5 * ((-1 + 2) - (-3 + 2)))))
    assert float(loss) == pytest.approx(expected)


def test_load_with_adapter(tmp_path):
    hf = _model(seed=3)
    hf.model.save_pretrained(tmp_path / "base")
    hf.tokenizer.save_pretrained(tmp_path / "base")
    finetune.train_sft(hf, synthetic.generate(4), tmp_path / "ad", batch_size=4, r=4, alpha=8, target_modules=["q_proj"])
    loaded = HFModel.load(str(tmp_path / "base"), dtype="fp32", device="cpu", adapter=str(tmp_path / "ad"))
    assert "+" in loaded.name and loaded.num_layers == 2
    assert len(loaded.choice_logprobs([{"role": "user", "content": "x"}], "The answer is (", ["A", "B"])) == 2


def test_cli_train_and_eval_with_adapter(tmp_path):
    from conftest import SAMPLES

    from syco.cli import main

    hf = _model(seed=4)
    hf.model.save_pretrained(tmp_path / "base")
    hf.tokenizer.save_pretrained(tmp_path / "base")
    assert main(["data", "synthetic", "--n", "8", "--out", str(tmp_path / "syn.jsonl")]) == 0
    for method in ("sft", "dpo"):
        assert main(["train", method, "--model", str(tmp_path / "base"), "--dtype", "fp32", "--train",
                     str(tmp_path / "syn.jsonl"), "--batch-size", "4", "--rank", "4", "--out", str(tmp_path / method)]) == 0
        assert (tmp_path / method / "adapter_config.json").exists()
    assert main(["eval", "--model", str(tmp_path / "base"), "--adapter", str(tmp_path / "dpo"), "--dtype", "fp32",
                 "--data", str(SAMPLES), "--split", "all", "--limit", "2", "--out", str(tmp_path / "run")]) == 0
    assert (tmp_path / "run" / "summary.md").exists()
