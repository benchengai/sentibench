"""Command-line entry point.

    sentibench fetch && sentibench prepare            # download the gold sets, write splits
    sentibench baselines                              # VADER, TF-IDF + LR, embeddings + LR
    sentibench llm -m openai:gpt-6-luna -p zs-guided fs-guided --split dev -n 150
    sentibench examples                               # selected prompts on the demo's example sentences
    sentibench report                                 # docs/benchmark.md and the web demo data
    sentibench classify "Kill the process and restart the server."
    sentibench spend
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data import DATASETS
from .llm import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# The OpenAI key lives in the ElicitSense checkout; variables already set win.
ENV_FILES = (PROJECT_ROOT / ".env", PROJECT_ROOT.parent / "elicitsense" / ".env")


def _env() -> None:
    for path in ENV_FILES:
        load_dotenv(path)


def cmd_fetch(args) -> None:
    from .data import fetch

    for path in fetch():
        print(f"ok {path}")


def cmd_prepare(args) -> None:
    from .data import prepare

    manifest = prepare()
    for name, info in manifest["datasets"].items():
        s = info["splits"]
        print(f"{name}: train {s['train']['n']} (fit {s['fit']['n']}, dev {s['dev']['n']}), test {s['test']['n']}; "
              f"{info['test_texts_also_in_train']} test texts also occur in train")


def cmd_baselines(args) -> None:
    from . import bench

    print("VADER thresholds tuned on dev:", json.dumps(bench.run_vader()))
    for ds in DATASETS:
        bench.train_tfidf(ds)
        print(f"tfidf-lr@{ds} done")
    if args.roberta:
        bench.run_roberta()
        print("roberta-twitter done")
    if not args.skip_embed:
        _env()
        for ds in DATASETS:
            bench.train_embed(ds)
            print(f"embed-lr@{ds} done")
        print(f"embedding spend ${bench.embedding_spend():.4f}")


def _resolve_prompt(prompt: str, eval_ds: str) -> str:
    """A few-shot prompt without a source uses examples from the dataset being scored."""
    return f"{prompt}@{eval_ds}" if prompt.startswith("fs-") and "@" not in prompt else prompt


def cmd_llm(args) -> None:
    from . import bench

    _env()
    for model in args.models:
        for prompt in args.prompts:
            for ds in args.datasets:
                spec = f"llm:{model}/{_resolve_prompt(prompt, ds)}"
                for run in range(args.first_run, args.first_run + args.runs):
                    try:
                        r = bench.run_llm(spec, ds, args.split, args.n, run, args.workers, args.budget)
                    except bench.BudgetExceeded as e:
                        sys.exit(str(e))
                    print(f"{spec} {ds}/{args.split} run {run}: {r['items']} items ({r['new']} new), "
                          f"{r['failed']} failed, ${r['cost_usd']:.4f}", flush=True)
    print(f"total LLM + embedding spend so far: ${bench.llm_spend() + bench.embedding_spend():.4f}")


def cmd_examples(args) -> None:
    from . import bench
    from .report import EXAMPLES, classifier_rows, select_prompts

    _env()
    sentences = json.loads(EXAMPLES.read_text(encoding="utf-8"))["sentences"]
    bench.run_examples_baselines(sentences, roberta=not args.skip_roberta)
    print("wrote RoBERTa and embeddings + LR outputs for the examples")
    _, selected = select_prompts()
    for row in classifier_rows(selected):
        if row["family"] != "LLM":
            continue
        sources = DATASETS if row["kind"] == "few-shot" else DATASETS[:1]
        for ds in sources:
            key = ds if row["kind"] == "few-shot" else "zs"
            print(f"wrote {bench.run_examples(sentences, row['spec'](ds), key)}")


def cmd_spend(args) -> None:
    from . import bench

    print(f"LLM ${bench.llm_spend():.4f}, embeddings ${bench.embedding_spend():.4f}")


def cmd_report(args) -> None:
    from .report import build

    for path in build(Path(args.out), Path(args.web), args.partial):
        print(f"wrote {path}")


def cmd_classify(args) -> None:
    from . import bench
    from . import classifiers as clf
    from .data import load_split

    text = args.text
    compound = clf.vader_compound([text])[0]
    print(f"vader: {clf.vader_label(compound)} (compound {compound:+.3f})")
    for ds in DATASETS:
        best = json.loads((bench.RUNS / "models" / f"tfidf_{ds}.json").read_text())["best"]
        train = load_split(ds, "train")
        fitted = clf.fit_tfidf([it["text"] for it in train], [it["label"] for it in train], **best)
        print(f"tfidf-lr@{ds}: {clf.predict_tfidf(fitted, [text])[0]}")
    if args.model:
        _env()
        model, prompt = args.model.partition("/")[::2]
        r = bench.classify(text, model, prompt or "zs-guided")
        print(f"{args.model}: {r['label'] if r['ok'] else 'FAILED ' + r['error']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sentibench", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="download the raw gold sets").set_defaults(func=cmd_fetch)
    sub.add_parser("prepare", help="clean and split the gold sets").set_defaults(func=cmd_prepare)
    p = sub.add_parser("baselines", help="run VADER, TF-IDF + LR and embeddings + LR")
    p.add_argument("--skip-embed", action="store_true", help="skip the embedding baseline (needs an API key)")
    p.add_argument("--roberta", action="store_true",
                   help="also run the general-domain RoBERTa model (needs `uv sync --extra hf`, downloads ~500 MB)")
    p.set_defaults(func=cmd_baselines)
    p = sub.add_parser("llm", help="classify a sample with LLMs (cached, resumable)")
    p.add_argument("-m", "--models", nargs="+", required=True, help="provider:model, e.g. openai:gpt-6-luna")
    p.add_argument("-p", "--prompts", nargs="+", default=["zs-guided"],
                   help="zs-basic, zs-guided, fs-guided (examples from the scored dataset) or fs-guided@<dataset>")
    p.add_argument("-d", "--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.add_argument("--split", default="dev", choices=("dev", "test"))
    p.add_argument("-n", type=int, default=None, help="first n items of the split's stratified order (default: all)")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--first-run", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--budget", type=float, default=6.0, help="stop when total recorded spend passes this (USD)")
    p.set_defaults(func=cmd_llm)
    p = sub.add_parser("examples", help="run the selected classifiers on the demo's example sentences")
    p.add_argument("--skip-roberta", action="store_true", help="skip RoBERTa (needs the hf extra)")
    p.set_defaults(func=cmd_examples)
    sub.add_parser("spend", help="total recorded API spend").set_defaults(func=cmd_spend)
    p = sub.add_parser("report", help="write the benchmark report and the web demo data")
    p.add_argument("--out", default="docs/benchmark.md")
    p.add_argument("--web", default="web/public/data")
    p.add_argument("--partial", action="store_true", help="leave out classifiers whose test runs are incomplete (development only)")
    p.set_defaults(func=cmd_report)
    p = sub.add_parser("classify", help="classify one sentence with VADER, TF-IDF and optionally an LLM")
    p.add_argument("text")
    p.add_argument("-m", "--model", help="e.g. openai:gpt-6-luna/zs-guided")
    p.set_defaults(func=cmd_classify)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
