"""Hugging Face transformers backend.

Loads any causal LM with a chat template (Qwen2.5, Llama 3.2, Gemma 2, OLMo 2, ...), scores answer
letters, generates text, reads the residual stream and adds steering vectors.
torch and transformers are only imported when this module is used.
"""

from __future__ import annotations

from typing import Any, Sequence

from syco.eval.prompts import Message


class HFModel:
    def __init__(self, model: Any, tokenizer: Any, name: str) -> None:
        import torch

        self.torch = torch
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.name = name
        self.layers = _decoder_layers(model)
        self.num_layers = len(self.layers)
        self._hook = None
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @classmethod
    def load(
        cls,
        name: str,
        dtype: str = "auto",
        device: str | None = None,
        adapter: str | None = None,
    ) -> "HFModel":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = {"auto": "auto", "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch_dtype).to(device)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        return cls(model, tokenizer, name if not adapter else f"{name}+{adapter}")

    @property
    def device(self) -> Any:
        return next(self.model.parameters()).device

    # ---- prompt rendering ----

    def render(self, messages: list[Message], prefix: str = "") -> str:
        tok = self.tokenizer
        if getattr(tok, "chat_template", None):
            try:
                text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                # Some templates (Gemma 2) reject a system role: fold it into the first user turn.
                text = tok.apply_chat_template(_fold_system(messages), tokenize=False, add_generation_prompt=True)
        else:
            text = "".join(f"{m['role'].capitalize()}: {m['content']}\n" for m in messages) + "Assistant: "
        return text + prefix

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    # ---- ChatModel ----

    def choice_logprobs(self, messages: list[Message], prefix: str, choices: Sequence[str]) -> list[float]:
        torch = self.torch
        prompt = self.render(messages, prefix)
        prompt_ids = self._encode(prompt)
        rows, spans = [], []
        for choice in choices:
            full = self._encode(prompt + choice)
            start = _common_prefix_len(prompt_ids, full)
            if start == len(full):
                start = len(full) - 1
            rows.append(full)
            spans.append((start, len(full)))
        width = max(len(r) for r in rows)
        pad = self.tokenizer.pad_token_id
        ids = torch.full((len(rows), width), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = torch.tensor(r)
            mask[i, : len(r)] = 1
        with torch.no_grad():
            logits = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device)).logits.float()
        logp = torch.log_softmax(logits, dim=-1)
        out = []
        for i, (start, end) in enumerate(spans):
            total = 0.0
            for pos in range(max(start, 1), end):
                total += float(logp[i, pos - 1, rows[i][pos]])
            out.append(total)
        return out

    def generate(self, messages: list[Message], max_new_tokens: int = 256) -> str:
        torch = self.torch
        ids = torch.tensor([self._encode(self.render(messages))], device=self.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True).strip()

    # ---- InspectableModel ----

    def activations(self, messages: list[Message], completion: str, layers: Sequence[int]) -> dict[int, list[float]]:
        torch = self.torch
        ids = torch.tensor([self._encode(self.render(messages) + completion)], device=self.device)
        with torch.no_grad():
            hidden = self.model(input_ids=ids, output_hidden_states=True).hidden_states
        # hidden[0] is the embedding output, hidden[i + 1] the output of decoder layer i.
        return {layer: hidden[layer + 1][0, -1].float().cpu().tolist() for layer in layers}

    def set_steering(self, layer: int | None, vector: Sequence[float] | None, alpha: float = 0.0) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        if layer is None or vector is None or alpha == 0:
            return
        torch = self.torch
        vec = torch.tensor(list(vector), dtype=torch.float32, device=self.device)
        delta = (alpha * vec)

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            if isinstance(output, tuple):
                return (output[0] + delta.to(output[0].dtype), *output[1:])
            return output + delta.to(output.dtype)

        self._hook = self.layers[layer].register_forward_hook(hook)


def _decoder_layers(model: Any) -> Any:
    paths = (
        "model.layers", "layers", "model.model.layers", "base_model.model.model.layers",
        "transformer.h", "model.transformer.h", "gpt_neox.layers",
    )
    for root in (model, getattr(model, "base_model", model)):
        for path in paths:
            obj = root
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
            except AttributeError:
                continue
            if hasattr(obj, "__len__") and len(obj) > 0:
                return obj
    raise ValueError("could not find the decoder layers of this model")


def _fold_system(messages: list[Message]) -> list[Message]:
    if not messages or messages[0]["role"] != "system":
        return messages
    system, rest = messages[0]["content"], list(messages[1:])
    if rest and rest[0]["role"] == "user":
        rest[0] = {"role": "user", "content": f"{system}\n\n{rest[0]['content']}"}
    else:
        rest.insert(0, {"role": "user", "content": system})
    return rest


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n
