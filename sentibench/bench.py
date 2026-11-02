"""Run classifiers over the splits and cache every prediction on disk.

Layout: runs/preds/<eval dataset>/<split>/<classifier slug>/run<k>.jsonl, one
line per item. LLM runs append a line as each item finishes, so an
interrupted run resumes where it stopped, and a larger sample reuses the
items a smaller one already classified (samples are prefixes of one order).
An item whose call fails (invalid or truncated JSON, a refusal, an API error
after retries) is recorded with its error and counted, not retried.

Classifier specs:
    vader                              VADER with the authors' thresholds
    roberta-twitter                    general-domain RoBERTa sentiment model, no SE training
    vader-tuned@<ds>                   VADER with thresholds tuned on <ds> dev
    tfidf-lr@<ds>                      TF-IDF + logistic regression trained on <ds> train
    embed-lr@<ds>                      text-embedding-3-small + logistic regression trained on <ds> train
    llm:<provider>:<model>/<prompt>    e.g. llm:openai:gpt-6-luna/fs-guided@github
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import classifiers as clf
from .data import DATASETS, LABELS, load_sample, load_split
from .llm import LLMError, complete_json, embed
from .prompts import SCHEMA, SCHEMA_NAME, system_prompt, user_prompt

RUNS = Path("runs")
REASONING_EFFORT = {"openai": "low"}  # the lowest setting all three API models accept


def slug(spec: str) -> str:
    return spec.replace("/", "__").replace(":", "_").replace("@", "-at-")


def pred_path(eval_ds: str, split: str, spec: str, run: int = 1, root: Path = RUNS) -> Path:
    return root / "preds" / eval_ds / split / slug(spec) / f"run{run}.jsonl"


def read_preds(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {r["id"]: r for r in rows}


def _write_preds(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- baselines


def _timed(fn, *args):
    started = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - started


def _record_timing(root: Path, name: str, items: int, seconds: float) -> None:
    """Wall-clock time to classify `items` texts on this machine, single process."""
    path = root / "models" / "timing.json"
    timing = json.loads(path.read_text()) if path.exists() else {}
    timing[name] = {"items": items, "seconds": round(seconds, 3)}
    _write_json(path, timing)


def run_vader(root: Path = RUNS) -> dict:
    """Default and dev-tuned VADER for every dataset; tuned thresholds also applied across datasets."""
    compounds, seconds = {}, 0.0
    for ds in DATASETS:
        for split in ("dev", "test"):
            items = load_split(ds, split)
            comp, elapsed = _timed(clf.vader_compound, [it["text"] for it in items])
            compounds[(ds, split)] = (items, comp)
            seconds += elapsed
    _record_timing(root, "vader", sum(len(v[0]) for v in compounds.values()), seconds)
    tuned = {}
    for ds in DATASETS:
        items, comp = compounds[(ds, "dev")]
        tuned[ds], f1 = clf.tune_vader(comp, [it["label"] for it in items])
        tuned[ds] = {"thresholds": tuned[ds], "dev_macro_f1": f1}
    for (eval_ds, split), (items, comp) in compounds.items():
        _write_preds(pred_path(eval_ds, split, "vader", root=root),
                     [{"id": it["id"], "label": clf.vader_label(c), "score": c} for it, c in zip(items, comp)])
        for ds in DATASETS:
            th = tuple(tuned[ds]["thresholds"])
            _write_preds(pred_path(eval_ds, split, f"vader-tuned@{ds}", root=root),
                         [{"id": it["id"], "label": clf.vader_label(c, th), "score": c} for it, c in zip(items, comp)])
    _write_json(root / "models" / "vader_tuned.json", tuned)
    return tuned


def run_roberta(root: Path = RUNS) -> None:
    total, seconds = 0, 0.0
    for ds in DATASETS:
        for split in ("dev", "test"):
            items = load_split(ds, split)
            out, elapsed = _timed(clf.roberta_predict, [it["text"] for it in items])
            total, seconds = total + len(items), seconds + elapsed
            _write_preds(pred_path(ds, split, "roberta-twitter", root=root),
                         [{"id": it["id"], "label": label, "probs": probs} for it, (label, probs) in zip(items, out)])
    _record_timing(root, "roberta-twitter", total, seconds)


def _xy(items: list[dict]) -> tuple[list[str], list[str]]:
    return [it["text"] for it in items], [it["label"] for it in items]


def train_tfidf(train_ds: str, root: Path = RUNS):
    """Grid search on dev, refit on train, predict dev and test of every dataset."""
    best, dev_f1, table = clf.grid_search(clf.fit_tfidf, clf.predict_tfidf, _xy(load_split(train_ds, "fit")),
                                          _xy(load_split(train_ds, "dev")), clf.TFIDF_GRID)
    # Dev predictions come from the model fitted on fit (dev unseen); test from the refit on train.
    fitted_fit = clf.fit_tfidf(*_xy(load_split(train_ds, "fit")), **best)
    fitted = clf.fit_tfidf(*_xy(load_split(train_ds, "train")), **best)
    spec = f"tfidf-lr@{train_ds}"
    total, seconds = 0, 0.0
    for eval_ds in DATASETS:
        for split, model in (("dev", fitted_fit), ("test", fitted)):
            items = load_split(eval_ds, split)
            labels, elapsed = _timed(clf.predict_tfidf, model, [it["text"] for it in items])
            total, seconds = total + len(items), seconds + elapsed
            _write_preds(pred_path(eval_ds, split, spec, root=root),
                         [{"id": it["id"], "label": label} for it, label in zip(items, labels)])
    _record_timing(root, spec, total, seconds)
    _write_json(root / "models" / f"tfidf_{train_ds}.json",
                {"best": best, "dev_macro_f1": dev_f1, "grid": table, "options": clf.TFIDF_OPTIONS,
                 "vocabulary": len(fitted[0].vocabulary_)})
    return fitted


EMBED_MODEL = "text-embedding-3-small"
EMBED_BATCH = 200


def embeddings(dataset: str, split: str, root: Path = RUNS) -> np.ndarray:
    """Cached embeddings for a split, in split order. Calls the API for missing items only."""
    items = load_split(dataset, split)
    path = root / "embeddings" / EMBED_MODEL / f"{dataset}_{split}.npz"
    log_path = path.with_suffix(".calls.jsonl")
    cached: dict[str, np.ndarray] = {}
    if path.exists():
        with np.load(path) as z:
            cached = dict(zip(z["ids"].tolist(), z["vectors"]))
    todo = [it for it in items if it["id"] not in cached]
    for start in range(0, len(todo), EMBED_BATCH):
        batch = todo[start:start + EMBED_BATCH]
        vectors, meta = embed([it["text"][:8000] for it in batch], EMBED_MODEL)
        cached.update({it["id"]: np.asarray(v, dtype=np.float32) for it, v in zip(batch, vectors)})
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, ids=np.array(list(cached)), vectors=np.stack(list(cached.values())))
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"items": len(batch), **meta}) + "\n")
    return np.stack([cached[it["id"]] for it in items])


def train_embed(train_ds: str, root: Path = RUNS) -> None:
    X = {split: embeddings(train_ds, split, root) for split in ("fit", "dev")}
    y = {split: [it["label"] for it in load_split(train_ds, split)] for split in ("fit", "dev")}
    best, dev_f1, table = clf.grid_search(clf.fit_embed, lambda m, X_: list(m.predict(X_)),
                                          (X["fit"], y["fit"]), (X["dev"], y["dev"]), clf.EMBED_GRID)
    fitted_fit = clf.fit_embed(X["fit"], y["fit"], **best)
    fitted = clf.fit_embed(np.vstack([X["fit"], X["dev"]]), y["fit"] + y["dev"], **best)
    spec = f"embed-lr@{train_ds}"
    for eval_ds in DATASETS:
        for split, model in (("dev", fitted_fit), ("test", fitted)):
            items = load_split(eval_ds, split)
            labels = model.predict(embeddings(eval_ds, split, root))
            _write_preds(pred_path(eval_ds, split, spec, root=root),
                         [{"id": it["id"], "label": str(label)} for it, label in zip(items, labels)])
    _write_json(root / "models" / f"embed_{train_ds}.json", {"best": best, "dev_macro_f1": dev_f1, "grid": table})


def embedding_spend(root: Path = RUNS) -> float:
    total = 0.0
    for log in (root / "embeddings").glob("*/*.calls.jsonl"):
        total += sum(json.loads(line)["cost_usd"] or 0 for line in log.read_text().splitlines() if line)
    return total


# ---------------------------------------------------------------- LLMs


def parse_llm_spec(spec: str) -> tuple[str, str]:
    """'llm:openai:gpt-6-luna/fs-guided@github' -> ('openai:gpt-6-luna', 'fs-guided@github')."""
    if not spec.startswith("llm:") or "/" not in spec:
        raise ValueError(f"LLM spec must look like llm:<provider>:<model>/<prompt>: {spec!r}")
    model_spec, _, prompt_id = spec.removeprefix("llm:").partition("/")
    return model_spec, prompt_id


def classify(text: str, model_spec: str, prompt_id: str) -> dict:
    """One item. Never raises for model-side failures: they are returned as ok=False."""
    effort = REASONING_EFFORT.get(model_spec.partition(":")[0])
    started = time.perf_counter()
    try:
        raw, meta = complete_json(system_prompt(prompt_id), user_prompt(text), SCHEMA, SCHEMA_NAME,
                                  model_spec, effort)
    except LLMError as e:
        return {"label": None, "ok": False, "error": str(e)[:300],
                "latency_s": round(time.perf_counter() - started, 3), "cost_usd": None}
    row = {k: meta[k] for k in ("model", "input_tokens", "cached_input_tokens", "output_tokens",
                                "reasoning_tokens", "latency_s", "cost_usd")}
    try:
        label = json.loads(raw)["label"]
        if label not in LABELS:
            raise ValueError(label)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return {"label": None, "ok": False, "error": f"invalid output: {raw[:200]!r}", **row}
    return {"label": label, "ok": True, **row}


def llm_spend(root: Path = RUNS) -> float:
    total = 0.0
    for path in (root / "preds").glob("*/*/llm_*/run*.jsonl"):
        total += sum(r.get("cost_usd") or 0 for r in read_preds(path).values())
    return total


def run_examples(sentences: list[dict], spec: str, key: str, root: Path = RUNS) -> Path:
    """Classify the demo's example sentences (cached by sentence id). `key` names the
    few-shot source dataset, or "zs" for zero-shot prompts."""
    model_spec, prompt_id = parse_llm_spec(spec)
    path = root / "preds" / "examples" / key / slug(spec) / "run1.jsonl"
    done = read_preds(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for ex in sentences:
            if ex["id"] not in done:
                f.write(json.dumps({"id": ex["id"], **classify(ex["text"], model_spec, prompt_id)}) + "\n")
    return path


def run_examples_baselines(sentences: list[dict], root: Path = RUNS, roberta: bool = True) -> None:
    """RoBERTa and embeddings + LR on the demo's example sentences (real model outputs for the try-box)."""
    texts = [ex["text"] for ex in sentences]
    if roberta:
        out = clf.roberta_predict(texts)
        _write_preds(root / "preds" / "examples" / "all" / "roberta-twitter" / "run1.jsonl",
                     [{"id": ex["id"], "label": label, "probs": probs} for ex, (label, probs) in zip(sentences, out)])
    cache = root / "embeddings" / EMBED_MODEL / "examples.npz"
    cached = {}
    if cache.exists():
        with np.load(cache) as z:
            cached = dict(zip(z["ids"].tolist(), z["vectors"]))
    todo = [ex for ex in sentences if ex["id"] not in cached]
    if todo:
        vectors, meta = embed([ex["text"] for ex in todo], EMBED_MODEL)
        cached.update({ex["id"]: np.asarray(v, dtype=np.float32) for ex, v in zip(todo, vectors)})
        np.savez(cache, ids=np.array(list(cached)), vectors=np.stack(list(cached.values())))
        with cache.with_suffix(".calls.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"items": len(todo), **meta}) + "\n")
    X = np.stack([cached[ex["id"]] for ex in sentences])
    for ds in DATASETS:
        best = json.loads((root / "models" / f"embed_{ds}.json").read_text())["best"]
        y = [it["label"] for it in load_split(ds, "fit") + load_split(ds, "dev")]
        model = clf.fit_embed(np.vstack([embeddings(ds, "fit", root), embeddings(ds, "dev", root)]), y, **best)
        _write_preds(root / "preds" / "examples" / ds / f"embed-lr-at-{ds}" / "run1.jsonl",
                     [{"id": ex["id"], "label": str(label)} for ex, label in zip(sentences, model.predict(X))])


class BudgetExceeded(RuntimeError):
    pass


def run_llm(spec: str, eval_ds: str, split: str, n: int | None, run: int = 1, workers: int = 4,
            budget_usd: float | None = None, root: Path = RUNS) -> dict:
    """Classify the first n items of a split's stratified order; cached per item."""
    model_spec, prompt_id = parse_llm_spec(spec)
    if model_spec.startswith("ollama:"):
        workers = 1  # one request at a time on the shared laptop
    items = load_sample(eval_ds, split, n)
    path = pred_path(eval_ds, split, spec, run, root)
    done = read_preds(path)
    todo = [it for it in items if it["id"] not in done]
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    spent = {"start": llm_spend(root) + embedding_spend(root), "run": 0.0}
    stop = threading.Event()

    def job(item):
        if stop.is_set():
            return
        row = {"id": item["id"], **classify(item["text"], model_spec, prompt_id)}
        with lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            spent["run"] += row.get("cost_usd") or 0
            if budget_usd is not None and spent["start"] + spent["run"] > budget_usd:
                stop.set()

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(job, todo))
    if stop.is_set():
        raise BudgetExceeded(f"stopped: total spend ${spent['start'] + spent['run']:.3f} passed the ${budget_usd} cap")
    rows = read_preds(path)
    selected = [rows[it["id"]] for it in items]
    return {"spec": spec, "dataset": eval_ds, "split": split, "run": run, "items": len(selected),
            "new": len(todo), "failed": sum(not r["ok"] for r in selected),
            "cost_usd": sum(r.get("cost_usd") or 0 for r in selected)}
