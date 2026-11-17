"""Command line: `syco <command> --help` for details.

    syco data perez sycophancy_on_nlp_survey.jsonl ... --out data/eval/opinion.jsonl
    syco data mcq --hf cais/mmlu:all:test --limit 600 --out data/eval/pressure.jsonl
    syco data synthetic --n 4000 --out data/train/synthetic.jsonl
    syco eval --model Qwen/Qwen2.5-1.5B-Instruct --data data/eval --split test --out results/qwen1.5b-baseline
    syco compare results/qwen1.5b-baseline results/qwen1.5b-dpo --out results/compare.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from syco.data import read_mcq, read_opinion, write_jsonl
from syco.data.splits import select
from syco.eval.harness import EvalData, load_run, run_eval
from syco.eval.tasks import ALL_TASKS


# ---------------------------------------------------------------- helpers


def _load_data(data_dir: str, split: str, limit: int | None) -> EvalData:
    d = Path(data_dir)

    def take(items):
        items = select(items, split)
        return items[:limit] if limit else items

    data = EvalData()
    if (d / "pressure.jsonl").exists():
        data.pressure = take(read_mcq(d / "pressure.jsonl"))
    if (d / "capability.jsonl").exists():
        data.capability = take(read_mcq(d / "capability.jsonl"))
    if (d / "opinion.jsonl").exists():
        data.opinion = take(read_opinion(d / "opinion.jsonl"))
    if not (data.pressure or data.capability or data.opinion):
        raise SystemExit(f"no pressure.jsonl, capability.jsonl or opinion.jsonl in {d}")
    return data


def _load_model(args, data: EvalData | None = None):
    if args.model.startswith("fake"):
        from syco.models.fake import FakeModel

        d = data or EvalData()
        return FakeModel(
            mcq=[*d.pressure, *d.capability], opinions=d.opinion, sycophancy=args.fake_sycophancy, name=args.model
        )
    from syco.models.hf import HFModel

    return HFModel.load(args.model, dtype=args.dtype, adapter=getattr(args, "adapter", None))


def _model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="Hugging Face model id, or 'fake' for the simulated model")
    p.add_argument("--adapter", help="path to a LoRA adapter trained with `syco train`")
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--fake-sycophancy", type=float, default=0.6, help=argparse.SUPPRESS)


def _data_args(p: argparse.ArgumentParser, default_split: str = "test") -> None:
    p.add_argument("--data", required=True, help="folder with pressure.jsonl, capability.jsonl and/or opinion.jsonl")
    p.add_argument("--split", default=default_split, choices=["dev", "test", "all"])
    p.add_argument("--limit", type=int, help="use only the first N items of each file (for quick checks)")


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


# ---------------------------------------------------------------- commands


def cmd_eval(args) -> None:
    from syco.methods.prompting import HONESTY_PROMPTS

    data = _load_data(args.data, args.split, args.limit)
    model = _load_model(args, data)
    system = args.system_prompt
    if args.prompt_name:
        system = HONESTY_PROMPTS[args.prompt_name]
    if args.steer:
        from syco.methods.steering import apply, load_vectors

        apply(model, load_vectors(args.steer), args.layer, args.alpha)
    config = {
        "split": args.split,
        "limit": args.limit,
        "adapter": args.adapter,
        "steer": args.steer and {"vectors": args.steer, "layer": args.layer, "alpha": args.alpha},
        "prompt_name": args.prompt_name,
    }
    tasks = args.tasks.split(",") if args.tasks else ALL_TASKS
    res = run_eval(model, data, tasks=tasks, out_dir=args.out, system_prompt=system, config=config, verbose=True)
    print(json.dumps(res["headline"], indent=2))
    if args.out:
        print(f"saved to {args.out}")


def cmd_compare(args) -> None:
    from syco.eval.report import compare_markdown

    runs = []
    for d in args.runs:
        run = load_run(d)
        run["label"] = Path(d).name
        runs.append(run)
    text = compare_markdown(runs, tolerance=args.tolerance)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


def cmd_data(args) -> None:
    from syco.data import converters as C

    if args.kind == "perez":
        import random

        items = []
        for path in args.inputs:
            file_items = C.perez_rows_to_opinion(C.load_jsonl(path), source=Path(path).stem)
            if args.shuffle_seed is not None:
                random.Random(args.shuffle_seed).shuffle(file_items)
            items += file_items[: args.limit] if args.limit else file_items
    elif args.kind == "mcq":
        if args.hf:
            from datasets import load_dataset

            name, config, split = (args.hf.split(":") + [None, None])[:3]
            rows = list(load_dataset(name, config, split=split or "test"))
            source = name.split("/")[-1]
        else:
            rows = [r for p in args.inputs for r in C.load_jsonl(p)]
            source = args.source or Path(args.inputs[0]).stem
        if args.shuffle_seed is not None:
            import random

            random.Random(args.shuffle_seed).shuffle(rows)
        if args.skip:
            rows = rows[args.skip :]
        if args.limit:
            rows = rows[: args.limit]
        items = C.mcq_rows_to_items(
            rows, source, args.question_field, args.choices_field, args.answer_field, offset=args.skip
        )
    elif args.kind == "synthetic":
        from syco.data import synthetic

        examples = synthetic.generate(args.n, seed=args.seed)
        synthetic.write(examples, args.out)
        print(f"wrote {len(examples)} training examples to {args.out}")
        return
    else:
        raise SystemExit(f"unknown data kind {args.kind}")
    n = write_jsonl(items, args.out)
    print(f"wrote {n} items to {args.out}")


def cmd_prompts(args) -> None:
    from syco.eval.report import compare_markdown
    from syco.methods.prompting import compare_prompts

    data = _load_data(args.data, args.split, args.limit)
    model = _load_model(args, data)
    results = compare_prompts(model, data, out_root=args.out)
    text = compare_markdown(list(results.values()))
    print(text)
    if args.out:
        (Path(args.out) / "prompts.md").write_text(text)


def cmd_gepa(args) -> None:
    from syco.methods.gepa import optimise_prompt
    from syco.methods.prompting import HONESTY_PROMPTS

    data = _load_data(args.data, "dev", args.limit)
    model = _load_model(args, data)
    reflector = model
    if args.reflector and args.reflector != args.model:
        from syco.models.hf import HFModel

        reflector = HFModel.load(args.reflector, dtype=args.dtype)
    seeds = ["", HONESTY_PROMPTS["honest"]]
    res = optimise_prompt(model, data, reflector, seeds=seeds, iterations=args.iterations, log_path=args.out)
    print(json.dumps(res["best"], indent=2, ensure_ascii=False))


def cmd_steer(args) -> None:
    from syco.methods import steering as S

    data = _load_data(args.data, "dev", args.limit)
    model = _load_model(args, data)
    if args.action == "build":
        pairs = S.contrastive_pairs(data.pressure, data.opinion)
        vectors = S.build_vectors(model, pairs, _ints(args.layers))
        S.save_vectors(vectors, args.out)
        print(f"saved {len(vectors)} vectors from {len(pairs)} pairs to {args.out}")
    else:
        res = S.sweep(model, data, S.load_vectors(args.vectors), _floats(args.alphas))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(json.dumps(res["best"], indent=2))


def cmd_train(args) -> None:
    from syco.data import synthetic
    from syco.methods import finetune
    from syco.models.hf import HFModel

    examples = synthetic.read(args.train)
    if args.limit:
        examples = examples[: args.limit]
    hf = HFModel.load(args.model, dtype=args.dtype)
    kwargs = dict(epochs=args.epochs, batch_size=args.batch_size, r=args.rank)
    if args.lr:
        kwargs["lr"] = args.lr
    if args.method == "sft":
        out = finetune.train_sft(hf, examples, args.out, **kwargs)
    else:
        out = finetune.train_dpo(hf, examples, args.out, beta=args.beta, **kwargs)
    print(f"adapter saved to {out}")


def cmd_probe(args) -> None:
    from syco.interp.probes import collect, probe_layers

    data = _load_data(args.data, args.split, args.limit)
    model = _load_model(args, data)
    feats, labels = collect(model, data.pressure, _ints(args.layers))
    res = probe_layers(feats, labels)
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2))


def cmd_directions(args) -> None:
    from syco.interp.directions import compare
    from syco.methods.steering import load_vectors

    res = compare(load_vectors(args.before), load_vectors(args.after))
    print(json.dumps(res, indent=2))


# ---------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="syco", description="Measure and reduce sycophancy in small open LLMs.")
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("eval", help="run the evaluation harness")
    _model_args(e)
    _data_args(e)
    e.add_argument("--tasks", help=f"comma-separated subset of: {', '.join(ALL_TASKS)}")
    e.add_argument("--system-prompt")
    e.add_argument("--prompt-name", choices=["none", "honest", "independent", "stubborn"])
    e.add_argument("--steer", help="steering vectors (.npz) from `syco steer build`")
    e.add_argument("--layer", type=int)
    e.add_argument("--alpha", type=float, default=0.0)
    e.add_argument("--out", help="results folder")
    e.set_defaults(func=cmd_eval)

    c = sub.add_parser("compare", help="compare runs, paired against the first one")
    c.add_argument("runs", nargs="+")
    c.add_argument("--tolerance", type=float, default=0.02, help="allowed capability drop")
    c.add_argument("--out")
    c.set_defaults(func=cmd_compare)

    d = sub.add_parser("data", help="convert benchmarks or make synthetic training data")
    d.add_argument("kind", choices=["perez", "mcq", "synthetic"])
    d.add_argument("inputs", nargs="*")
    d.add_argument("--out", required=True)
    d.add_argument("--hf", help="Hugging Face dataset as name:config:split, e.g. cais/mmlu:all:test")
    d.add_argument("--source")
    d.add_argument("--question-field", default="question")
    d.add_argument("--choices-field", default="choices")
    d.add_argument("--answer-field", default="answer")
    d.add_argument("--limit", type=int)
    d.add_argument("--skip", type=int, default=0, help="skip the first N rows (to keep files disjoint)")
    d.add_argument("--shuffle-seed", type=int)
    d.add_argument("--n", type=int, default=2000)
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(func=cmd_data)

    pr = sub.add_parser("prompts", help="method 1: compare honesty system prompts")
    _model_args(pr)
    _data_args(pr, "dev")
    pr.add_argument("--out")
    pr.set_defaults(func=cmd_prompts)

    g = sub.add_parser("gepa", help="method 2: optimise the system prompt on the dev split")
    _model_args(g)
    g.add_argument("--data", required=True)
    g.add_argument("--limit", type=int)
    g.add_argument("--reflector", help="model that writes new prompts (default: the model itself)")
    g.add_argument("--iterations", type=int, default=8)
    g.add_argument("--out", default="results/gepa.json")
    g.set_defaults(func=cmd_gepa)

    s = sub.add_parser("steer", help="method 3: build or sweep steering vectors on the dev split")
    s.add_argument("action", choices=["build", "sweep"])
    _model_args(s)
    s.add_argument("--data", required=True)
    s.add_argument("--limit", type=int)
    s.add_argument("--layers", default="4,8,12")
    s.add_argument("--vectors")
    s.add_argument("--alphas", default="-2,-4,-8")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_steer)

    t = sub.add_parser("train", help="methods 4 and 5: LoRA SFT or DPO on synthetic data")
    t.add_argument("method", choices=["sft", "dpo"])
    t.add_argument("--model", required=True)
    t.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    t.add_argument("--train", required=True, help="synthetic JSONL from `syco data synthetic`")
    t.add_argument("--limit", type=int)
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--rank", type=int, default=16)
    t.add_argument("--lr", type=float)
    t.add_argument("--beta", type=float, default=0.1)
    t.add_argument("--out", required=True)
    t.set_defaults(func=cmd_train)

    pb = sub.add_parser("probe", help="linear probes per layer")
    _model_args(pb)
    _data_args(pb, "dev")
    pb.add_argument("--layers", default="0,4,8,12,16,20")
    pb.add_argument("--out")
    pb.set_defaults(func=cmd_probe)

    dr = sub.add_parser("directions", help="compare steering vectors from before and after training")
    dr.add_argument("--before", required=True)
    dr.add_argument("--after", required=True)
    dr.set_defaults(func=cmd_directions)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
