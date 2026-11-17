"""Methods 4 and 5: LoRA fine-tuning (SFT) and DPO on the synthetic data.

Both train a LoRA adapter, so the base model stays untouched and the adapter can be switched off for
before/after comparisons and for the DPO reference model. The training loops are written out instead of
using a trainer library, so every step of the loss is visible and the code doesn't break when those
libraries change their APIs.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Sequence

from syco.models.hf import HFModel


def _peft_model(hf: HFModel, r: int, alpha: int, dropout: float, target_modules: Any) -> Any:
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=target_modules, task_type="CAUSAL_LM", bias="none"
    )
    model = get_peft_model(hf.model, config)
    # Keep the trainable LoRA weights in fp32 even when the base model is fp16 (a T4 has no bf16):
    # AdamW on fp16 weights is unstable.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return model


def _encode_pair(hf: HFModel, messages: list[dict], completion: str, max_len: int) -> tuple[list[int], int]:
    """Token ids of prompt + completion, and where the completion starts."""
    prompt = hf.render(messages)
    p_ids = hf._encode(prompt)
    full = hf._encode(prompt + completion) + ([hf.tokenizer.eos_token_id] if hf.tokenizer.eos_token_id is not None else [])
    start = 0
    for a, b in zip(p_ids, full):
        if a != b:
            break
        start += 1
    if len(full) > max_len:
        cut = len(full) - max_len
        full, start = full[cut:], max(0, start - cut)
    return full, start


def _batch(hf: HFModel, encoded: Sequence[tuple[list[int], int]]) -> tuple[Any, Any, Any]:
    torch = hf.torch
    width = max(len(ids) for ids, _ in encoded)
    pad = hf.tokenizer.pad_token_id
    ids = torch.full((len(encoded), width), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for i, (row, start) in enumerate(encoded):
        ids[i, : len(row)] = torch.tensor(row)
        mask[i, : len(row)] = 1
        labels[i, start : len(row)] = torch.tensor(row[start:])
    dev = hf.device
    return ids.to(dev), mask.to(dev), labels.to(dev)


def sequence_logps(model: Any, ids: Any, mask: Any, labels: Any) -> Any:
    """Sum of log-probabilities of the labelled (completion) tokens, one value per row."""
    torch = __import__("torch")
    logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
    target = labels[:, 1:]
    keep = target != -100
    logp = torch.log_softmax(logits, dim=-1).gather(-1, target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (logp * keep).sum(-1)


def dpo_loss(pc: Any, pr: Any, rc: Any, rr: Any, beta: float) -> Any:
    """Direct Preference Optimization (Rafailov et al., 2023).

    pc/pr: policy log-probs of chosen/rejected; rc/rr: the same under the frozen reference model.
    """
    torch = __import__("torch")
    margin = beta * ((pc - rc) - (pr - rr))
    return -torch.nn.functional.logsigmoid(margin).mean()


def _save(peft_model: Any, hf: HFModel, out_dir: str | Path, log: dict) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(out)
    hf.tokenizer.save_pretrained(out)
    (out / "train_log.json").write_text(json.dumps(log, indent=2))
    return out


def train_sft(
    hf: HFModel,
    examples: Sequence[dict],
    out_dir: str | Path,
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 2e-4,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Any = "all-linear",
    max_len: int = 512,
    seed: int = 0,
) -> Path:
    torch = hf.torch
    torch.manual_seed(seed)
    model = _peft_model(hf, r, alpha, dropout, target_modules)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    encoded = [_encode_pair(hf, ex["messages"], ex["chosen"], max_len) for ex in examples]
    rng = random.Random(seed)
    losses, started = [], time.time()
    for _ in range(epochs):
        order = list(range(len(encoded)))
        rng.shuffle(order)
        for i in range(0, len(order), batch_size):
            ids, mask, labels = _batch(hf, [encoded[j] for j in order[i : i + batch_size]])
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
    model.eval()
    hf.model = model
    log = {"method": "sft-lora", "examples": len(encoded), "epochs": epochs, "lr": lr, "r": r,
           "losses": losses, "seconds": round(time.time() - started, 1)}
    return _save(model, hf, out_dir, log)


def train_dpo(
    hf: HFModel,
    examples: Sequence[dict],
    out_dir: str | Path,
    epochs: int = 1,
    batch_size: int = 4,
    lr: float = 5e-5,
    beta: float = 0.1,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Any = "all-linear",
    max_len: int = 512,
    seed: int = 0,
) -> Path:
    torch = hf.torch
    torch.manual_seed(seed)
    model = _peft_model(hf, r, alpha, dropout, target_modules)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    chosen = [_encode_pair(hf, ex["messages"], ex["chosen"], max_len) for ex in examples]
    rejected = [_encode_pair(hf, ex["messages"], ex["rejected"], max_len) for ex in examples]
    rng = random.Random(seed)
    losses, accs, started = [], [], time.time()
    for _ in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            c = _batch(hf, [chosen[j] for j in idx])
            rj = _batch(hf, [rejected[j] for j in idx])
            # The reference model is the same network with the adapter switched off.
            model.eval()
            with torch.no_grad(), model.disable_adapter():
                rc, rr = sequence_logps(model, *c), sequence_logps(model, *rj)
            model.train()
            pc, pr = sequence_logps(model, *c), sequence_logps(model, *rj)
            loss = dpo_loss(pc, pr, rc, rr, beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            accs.append(float(((pc - rc) > (pr - rr)).float().mean()))
    model.eval()
    hf.model = model
    log = {"method": "dpo-lora", "examples": len(examples), "epochs": epochs, "lr": lr, "beta": beta, "r": r,
           "losses": losses, "reward_accuracy": accs, "seconds": round(time.time() - started, 1),
           "initial_loss_expected": math.log(2)}
    return _save(model, hf, out_dir, log)
