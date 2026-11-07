# sycophancy: measuring and reducing sycophancy in small open LLMs

> Status: planning. Nothing below is a result yet.

A model is sycophantic when it tells people what they want to hear instead of what is true: it drops a correct
answer when the user pushes back, agrees with the opinion it guesses the user holds, or praises a bad plan.
Small open models are a real problem here: a 2026 study of 56 open checkpoints found that below about 7B
parameters, instruction tuning often makes models *more* likely to cave under pressure.

This project measures sycophancy in small open models of several families, tries to reduce it with methods from
cheap to expensive, and looks inside the model to see what changed. It continues the theme of my other projects:
AI systems you can trust, with numbers to show it.

## Questions

1. **How sycophantic are small open models**, by family, size and training stage?
2. **How far can each method reduce it** without making the model stubborn or worse at its job?
3. **What changes inside the model?** Is the sycophancy direction removed by training, or only suppressed?

## The trap I want to avoid

A model that always disagrees scores perfectly on most sycophancy tests. So the training and test data include
cases where **the user is right**, and the headline metric is **selectivity**: does the model accept a correct
correction and reject a wrong one? A method only counts as an improvement if sycophancy goes down while
selectivity and general ability stay up.

## Evaluation

| What | Source | Metric |
| --- | --- | --- |
| Opinion sycophancy | persona prompts on NLP, philosophy and politics (Perez et al.) | agreement with the user's implied view |
| Factual pressure | "Are you sure?", wrong suggestions, authority framing (Sharma et al.; SycoBench) | flip rate on correct answers |
| Selectivity | correct vs incorrect user corrections (SycoBench) | accepts correct, rejects wrong |
| Multi-turn pressure | a persistent user over several turns (SPINE-style, shortened) | turn at which the model gives in |
| Social flattery (stretch) | advice and feedback (ELEPHANT) | LLM-judge score, with judge agreement checked |
| Capability guard | a fixed subset of a general benchmark | accuracy must not drop |

Everything except the stretch goal is scored automatically, without an LLM judge. Benchmarks used for evaluation
are never used for training, and training templates are kept apart from test templates.

## Models

Small enough for a free Colab T4, from different families:

- Qwen2.5 (0.5B, 1.5B, 3B)
- Llama 3.2 (1B, 3B)
- Gemma 2 (2B); Gemma Scope's sparse autoencoders make it the model for the interpretability part
- OLMo 2 (1B), whose SFT and DPO checkpoints are public, so I can see which training stage adds sycophancy

## Methods, from cheap to expensive

| # | Method | Changes the weights? |
| --- | --- | --- |
| 0 | Baseline | no |
| 1 | System prompt asking for honesty | no |
| 2 | Prompt optimised automatically (DSPy with GEPA) against the dev split | no |
| 3 | Activation steering (contrastive steering vectors, persona vectors) | no |
| 4 | LoRA fine-tuning on synthetic data (Wei et al.) | yes |
| 5 | DPO on preference pairs, including "the user is right" pairs | yes |

Every method runs on the same harness, on the same held-out questions, before and after.

## Looking inside

- **Linear probes** per layer: where does the model "decide" to agree?
- **Steering direction** before and after fine-tuning: is it gone, or does it still exist and just get used less?
- **Sparse autoencoder features** (Gemma 2 with Gemma Scope) that rise with sycophancy, before and after.

## Plan

- [ ] Evaluation harness: datasets, automatic scoring, selectivity, capability guard
- [ ] Baseline profile for every model
- [ ] Prompting and GEPA prompt optimisation
- [ ] Activation steering
- [ ] LoRA and DPO fine-tuning on one model per family
- [ ] Probes and steering-direction analysis before and after
- [ ] Write-up: what works, at what cost, and what changes inside

## Planned layout

```
src/syco/
  data/        loaders for the benchmarks, synthetic training data
  eval/        harness, metrics (flip rate, selectivity, capability), report
  methods/     prompts, GEPA, steering, LoRA, DPO
  interp/      probes, steering directions, SAE features
notebooks/     Colab runs, one per stage
results/       one folder per run: config, scores, every answer
```

## Limits I already know

- Small models only, on one GPU. Results may not carry over to large models.
- Synthetic training data can teach the model the template instead of the behaviour, which is why test templates
  are kept apart.
- An LLM judge (only for social flattery) has its own biases. I report how often two judges agree.

## References

- Sharma et al. (2023), [Towards Understanding Sycophancy in Language Models](https://arxiv.org/abs/2310.13548)
- Perez et al. (2022), [Discovering Language Model Behaviors with Model-Written Evaluations](https://arxiv.org/abs/2212.09251)
- Wei et al. (2023), [Simple synthetic data reduces sycophancy in large language models](https://arxiv.org/abs/2308.03958)
- [SycoBench-600: Measuring Sycophancy and Correction Selectivity in LLM Assistants](https://aclanthology.org/2026.findings-acl.1759/) (ACL Findings 2026)
- [ELEPHANT: Measuring and understanding social sycophancy in LLMs](https://arxiv.org/abs/2505.13995)
- [Measuring LLM Sycophancy under Sustained Multi-Turn Pressure](https://arxiv.org/pdf/2609.09090) (SPINE)
- [Decomposing Factual Sycophancy in Language Models: How Size and Instruction Tuning Shape Robustness](https://arxiv.org/pdf/2606.06306)
- Rimsky et al. (2023), [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681)
- Chen et al. (2025), [Persona Vectors: Monitoring and Controlling Character Traits in Language Models](https://arxiv.org/abs/2507.21509)
- [A Few Bad Neurons: Isolating and Surgically Correcting Sycophancy](https://arxiv.org/abs/2601.18939)
- [Detecting and Controlling Sycophancy with Cascading Linear Features](https://arxiv.org/pdf/2606.26155)
- Agrawal et al. (2025), [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457)
- Rafailov et al. (2023), [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
