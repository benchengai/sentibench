"""Turn the cached predictions into the benchmark report and the web demo data.

Everything here is computed from runs/ (predictions, call logs, model
selection files); nothing is typed in by hand. docs/benchmark.template.md holds
the prose; {{name}} placeholders are replaced with the generated tables.

The dev selection rule is applied in code (`select_prompts`): per model, the
zero-shot and the few-shot prompt with the best mean dev macro-F1. The report
fails if the test predictions for a selected prompt are missing.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

from . import bench, classifiers as clf
from .analysis import TAGS, mask, tags_for
from .data import DATASETS, LABELS, NAMES, load_sample, load_split
from .metrics import bootstrap, encode, kappa, paired_difference, run_consistency, scores
from .pricing import AS_OF, PRICES, SOURCE
from .prompts import BASIC, GUIDED, PROMPT_IDS, SHOTS_PER_CLASS, shots

RUNS = bench.RUNS
TEMPLATE = Path("docs/benchmark.template.md")
SAMPLE_N = 300     # LLM test sample per dataset (prefix of the stratified order)
VARIANCE_N = 100   # repeated-run sample per dataset (prefix of the LLM sample)
RUNS_PER_LLM = 3
N_BOOT = 1000

LLM_MODELS = {
    "openai:gpt-6-luna": "gpt-6-luna",
    "openai:gpt-6.1-terra": "gpt-6.1-terra",
    "openai:gpt-6.1-sol": "gpt-6.1-sol",
    "ollama:qwen3:8b": "qwen3:8b (local)",
}
FULL_TEST_LLMS = ("openai:gpt-6-luna",)  # cheap enough to run on every test item


# ---------------------------------------------------------------- loading


def _preds(eval_ds: str, split: str, spec: str, run: int = 1) -> dict[str, dict]:
    return bench.read_preds(bench.pred_path(eval_ds, split, spec, run))


def _labels(items: list[dict], preds: dict[str, dict]) -> np.ndarray | None:
    if not all(it["id"] in preds for it in items):
        return None
    return encode([preds[it["id"]]["label"] for it in items])


def _gold(items: list[dict]) -> np.ndarray:
    return encode([it["label"] for it in items])


def _fs(prompt: str, ds: str) -> str:
    return f"{prompt}@{ds}" if prompt.startswith("fs-") else prompt


def select_prompts() -> tuple[dict, dict]:
    """Per model: dev macro-F1 for every prompt variant, and the selected zero- and few-shot variants."""
    table, selected = {}, {}
    for model in LLM_MODELS:
        rows = {}
        for prompt in PROMPT_IDS:
            per_ds = {}
            for ds in DATASETS:
                preds = _preds(ds, "dev", f"llm:{model}/{_fs(prompt, ds)}")
                if not preds:
                    continue
                items = [it for it in load_split(ds, "dev") if it["id"] in preds]
                s = scores(_gold(items), _labels(items, preds))
                per_ds[ds] = {"n": len(items), "macro_f1": s["macro_f1"],
                              "cost_per_1k": 1000 * statistics.fmean(preds[it["id"]].get("cost_usd") or 0 for it in items)}
            if len(per_ds) == len(DATASETS):
                rows[prompt] = {"datasets": per_ds, "mean": statistics.fmean(v["macro_f1"] for v in per_ds.values())}
        if not rows:
            continue
        table[model] = rows
        pick = lambda kind: max((p for p in rows if p.startswith(kind)), key=lambda p: rows[p]["mean"])
        selected[model] = {"zero-shot": pick("zs-"), "few-shot": pick("fs-")}
    return table, selected


def classifier_rows(selected: dict) -> list[dict]:
    """Display rows. `spec(ds)` gives the within-platform spec for scoring dataset ds."""
    rows = [
        {"key": "vader", "name": "VADER", "family": "Lexicon", "spec": lambda ds: "vader"},
        {"key": "vader-tuned", "name": "VADER, dev-tuned thresholds", "family": "Lexicon",
         "spec": lambda ds: f"vader-tuned@{ds}"},
        {"key": "roberta", "name": "RoBERTa-Twitter", "family": "General pre-trained",
         "spec": lambda ds: "roberta-twitter"},
        {"key": "tfidf-lr", "name": "TF-IDF + LR", "family": "Supervised (SE train split)",
         "spec": lambda ds: f"tfidf-lr@{ds}"},
        {"key": "embed-lr", "name": "Embeddings + LR", "family": "Supervised (SE train split)",
         "spec": lambda ds: f"embed-lr@{ds}"},
    ]
    for model, short in LLM_MODELS.items():
        if model not in selected:
            continue
        for kind, prompt in selected[model].items():
            key = f"{short.split(' ')[0]}-{'zs' if kind == 'zero-shot' else 'fs'}"
            rows.append({"key": key, "name": f"{short}, {kind} ({prompt})", "family": "LLM", "model": model,
                         "prompt": prompt, "kind": kind,
                         "spec": (lambda m, p: lambda ds: f"llm:{m}/{_fs(p, ds)}")(model, prompt)})
    return rows


# ---------------------------------------------------------------- formatting


def _f(x: float | None, digits: int = 2) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def _ci(point: float, ci: list[float]) -> str:
    return f"{point:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------- sections


def evaluate(rows: list[dict], items_by_ds: dict[str, list[dict]], split: str = "test") -> dict:
    """{key: {ds: scores + CI}} for every row whose predictions cover the items."""
    out = {}
    for row in rows:
        per_ds = {}
        for ds, items in items_by_ds.items():
            pred = _labels(items, _preds(ds, split, row["spec"](ds)))
            if pred is None:
                continue
            gold = _gold(items)
            s = scores(gold, pred)
            s["ci"] = bootstrap(gold, pred, N_BOOT, seed=hash_seed(row["key"], ds))
            per_ds[ds] = s
        if per_ds:
            out[row["key"]] = per_ds
    return out


def hash_seed(*parts: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate("|".join(parts))) % (2**31)


def results_table(rows: list[dict], results: dict) -> str:
    header = ["Classifier", "Family"] + [f"{NAMES[ds]} macro-F1 [95% CI]" for ds in DATASETS] + \
             [f"{NAMES[ds]} κ" for ds in DATASETS] + ["Mean macro-F1"]
    body = []
    for row in rows:
        r = results.get(row["key"])
        if not r or len(r) < len(DATASETS):
            continue
        body.append([row["name"], row["family"]]
                    + [_ci(r[ds]["macro_f1"], r[ds]["ci"]["macro_f1"]) for ds in DATASETS]
                    + [_f(r[ds]["kappa"]) for ds in DATASETS]
                    + [_f(statistics.fmean(r[ds]["macro_f1"] for ds in DATASETS))])
    return _table(header, body)


def per_class_table(rows: list[dict], results: dict) -> str:
    header = ["Classifier"] + [f"{NAMES[ds]} {label[:3]}." for ds in DATASETS for label in LABELS]
    body = []
    for row in rows:
        r = results.get(row["key"])
        if not r or len(r) < len(DATASETS):
            continue
        body.append([row["name"]] + [_f(r[ds]["per_class"][label]["f1"]) for ds in DATASETS for label in LABELS])
    return _table(header, body)


def paired_table(rows: list[dict], sample: dict[str, list[dict]], pairs: list[tuple[str, str]]) -> tuple[str, list]:
    by_key = {r["key"]: r for r in rows}
    body, data = [], []
    for a, b in pairs:
        if a not in by_key or b not in by_key:
            continue
        cells = [f"{by_key[a]['name']} − {by_key[b]['name']}"]
        for ds in DATASETS:
            items = sample[ds]
            pa = _labels(items, _preds(ds, "test", by_key[a]["spec"](ds)))
            pb = _labels(items, _preds(ds, "test", by_key[b]["spec"](ds)))
            if pa is None or pb is None:
                cells.append("–")
                continue
            d = paired_difference(_gold(items), pa, pb, N_BOOT, seed=hash_seed(a, b, ds))
            cells.append(f"{d['diff']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]")
            data.append({"a": a, "b": b, "dataset": ds, **d})
        body.append(cells)
    return _table(["Difference in macro-F1 (paired bootstrap 95% CI)"] + [NAMES[ds] for ds in DATASETS], body), data


def cross_platform(selected: dict) -> tuple[str, dict]:
    """Train or tune on one dataset (rows), test on another (columns)."""
    families = [("TF-IDF + LR", "tfidf-lr@{src}", None), ("Embeddings + LR", "embed-lr@{src}", None),
                ("VADER, thresholds tuned on source dev", "vader-tuned@{src}", None)]
    luna = selected.get("openai:gpt-6-luna")
    if luna:
        families.append((f"gpt-6-luna, few-shot examples from source ({luna['few-shot']})",
                         f"llm:openai:gpt-6-luna/{luna['few-shot']}@{{src}}", SAMPLE_N))
    blocks, data = [], {}
    for name, template, n in families:
        header = [f"{name}: trained on ↓ / tested on →"] + [NAMES[ds] for ds in DATASETS]
        body = []
        for src in DATASETS:
            cells = [NAMES[src]]
            for tgt in DATASETS:
                items = load_sample(tgt, "test", n)
                pred = _labels(items, _preds(tgt, "test", template.format(src=src)))
                if pred is None:
                    cells.append("–")
                    continue
                s = scores(_gold(items), pred)
                data.setdefault(name, {}).setdefault(src, {})[tgt] = {"macro_f1": s["macro_f1"], "kappa": s["kappa"],
                                                                      "n": len(items)}
                cell = f"{s['macro_f1']:.2f}"
                cells.append(f"**{cell}**" if src == tgt else cell)
            body.append(cells)
        note = f" (LLM test sample, {n} per dataset)" if n else " (full test splits)"
        blocks.append(f"**{name}**{note}\n\n" + _table(header, body))
    return "\n\n".join(blocks), data


def cost_table(rows: list[dict]) -> tuple[str, list]:
    timing = json.loads((RUNS / "models" / "timing.json").read_text()) if (RUNS / "models" / "timing.json").exists() else {}
    body, data = [], []
    for row in rows:
        if row["family"] != "LLM":
            key = {"vader": "vader", "vader-tuned": "vader", "roberta": "roberta-twitter"}.get(row["key"])
            t = timing.get(key) or (timing.get(f"{row['key']}@github") if row["key"] == "tfidf-lr" else None)
            if row["key"] == "embed-lr":
                t = _embed_timing()
            ms = 1000 * t["seconds"] / t["items"] if t else None
            cost = "0 (local)" if row["key"] != "embed-lr" else _embed_cost_per_1k()
            entry = {"key": row["key"], "cost_per_1k": 0.0 if row["key"] != "embed-lr" else float(cost.split()[0]),
                     "ms_per_item": ms}
            body.append([row["name"], cost, _f(ms, 3) + " ms per item, batch" if ms is not None else "–",
                         "–", "0", "–", "–"])
            data.append(entry)
            continue
        calls = []
        for ds in DATASETS:
            for run in range(1, RUNS_PER_LLM + 1):
                calls += list(_preds(ds, "test", row["spec"](ds), run).values())
        if not calls:
            continue
        ok = [c for c in calls if c.get("cost_usd") is not None]
        lat = sorted(c["latency_s"] for c in calls)
        cost_1k = 1000 * statistics.fmean(c["cost_usd"] for c in ok) if ok else None
        failed = sum(not c["ok"] for c in calls)
        entry = {"key": row["key"], "calls": len(calls), "cost_per_1k": cost_1k,
                 "latency_median_s": statistics.median(lat), "latency_p95_s": lat[int(0.95 * (len(lat) - 1))],
                 "failed": failed, "failure_rate": failed / len(calls),
                 "input_tokens": statistics.fmean(c.get("input_tokens", 0) for c in ok),
                 "output_tokens": statistics.fmean(c.get("output_tokens", 0) for c in ok)}
        data.append(entry)
        body.append([row["name"], "0 (local)" if row["model"].startswith("ollama") else f"${cost_1k:.3f}",
                     f"{entry['latency_median_s']:.2f} s per call", f"{entry['latency_p95_s']:.2f} s",
                     f"{failed} of {len(calls)}", f"{entry['input_tokens']:.0f}", f"{entry['output_tokens']:.0f}"])
    header = ["Classifier", "Cost per 1,000 items", "Median latency", "p95 latency", "Failed calls",
              "Input tokens / item", "Output tokens / item"]
    return _table(header, body), data


def _embed_timing() -> dict | None:
    """API time per embedded item, from the batched calls (200 texts per request)."""
    items = seconds = 0
    for log in (RUNS / "embeddings").glob("*/*.calls.jsonl"):
        for line in log.read_text().splitlines():
            r = json.loads(line)
            items += r["items"]
            seconds += r["latency_s"]
    return {"items": items, "seconds": seconds} if items else None


def _embed_cost_per_1k() -> str:
    total_items, total_cost = 0, 0.0
    for log in (RUNS / "embeddings").glob("*/*.calls.jsonl"):
        for line in log.read_text().splitlines():
            r = json.loads(line)
            total_items += r["items"]
            total_cost += r["cost_usd"] or 0
    return f"{1000 * total_cost / total_items:.4f} (API)" if total_items else "–"


def variance_table(rows: list[dict]) -> tuple[str, list]:
    body, data = [], []
    for row in rows:
        if row["family"] != "LLM":
            continue
        per_run_f1 = {k: [] for k in range(1, RUNS_PER_LLM + 1)}
        runs_all = []
        ok = True
        for ds in DATASETS:
            items = load_sample(ds, "test", VARIANCE_N)
            labels = [_labels(items, _preds(ds, "test", row["spec"](ds), k)) for k in range(1, RUNS_PER_LLM + 1)]
            if any(lab is None for lab in labels):
                ok = False
                break
            for k, lab in enumerate(labels, 1):
                per_run_f1[k].append(scores(_gold(items), lab)["macro_f1"])
            runs_all.append(labels)
        if not ok:
            continue
        stacked = [np.concatenate([r[k] for r in runs_all]) for k in range(RUNS_PER_LLM)]
        c = run_consistency(stacked)
        means = [statistics.fmean(v) for v in per_run_f1.values()]
        entry = {"key": row["key"], **c, "macro_f1_runs": means}
        data.append(entry)
        body.append([row["name"], f"{statistics.fmean(means):.3f} ({min(means):.3f}–{max(means):.3f})",
                     f"{c['items_changed']} of {c['items']} ({100 * c['flip_rate']:.1f}%)", _f(c["mean_pairwise_kappa"])])
    header = ["Classifier", f"Mean macro-F1 over datasets, {RUNS_PER_LLM} runs (range)", "Items whose label changed",
              "Mean κ between runs"]
    return _table(header, body), data


def agreement(rows: list[dict], sample: dict[str, list[dict]]) -> tuple[str, dict]:
    keys, preds = [], {}
    for row in rows:
        parts = [_labels(sample[ds], _preds(ds, "test", row["spec"](ds))) for ds in DATASETS]
        if any(p is None for p in parts):
            continue
        keys.append(row["key"])
        preds[row["key"]] = np.concatenate(parts)
    gold = np.concatenate([_gold(sample[ds]) for ds in DATASETS])
    matrix = {a: {b: kappa(preds[a], preds[b]) for b in keys} for a in keys}
    for a in keys:
        matrix[a]["gold"] = kappa(preds[a], gold)
    header = ["κ"] + keys + ["gold"]
    body = [[a] + [_f(matrix[a][b]) for b in keys] + [_f(matrix[a]["gold"])] for a in keys]
    return _table(header, body), {"keys": keys, "kappa": matrix}


def error_analysis(rows: list[dict], sample: dict[str, list[dict]]) -> tuple[str, dict, list[dict]]:
    """Tagged hard cases on the LLM test sample: how often each classifier gets them right."""
    keys = []
    items_out = []
    preds_by_key = {}
    for row in rows:
        p = {ds: _preds(ds, "test", row["spec"](ds)) for ds in DATASETS}
        if all(all(it["id"] in p[ds] for it in sample[ds]) for ds in DATASETS):
            keys.append(row["key"])
            preds_by_key[row["key"]] = p
    vader_scores = {ds: _preds(ds, "test", "vader") for ds in DATASETS}
    for ds in DATASETS:
        for it in sample[ds]:
            compound = vader_scores[ds][it["id"]]["score"]
            items_out.append({"id": it["id"], "dataset": ds, "text": mask(it["text"]), "gold": it["label"],
                              "tags": tags_for(it["text"], it["label"], compound), "vader_compound": compound,
                              "preds": {k: preds_by_key[k][ds][it["id"]]["label"] for k in keys}})
    stats = {}
    for tag in TAGS:
        tagged = [x for x in items_out if tag in x["tags"]]
        stats[tag] = {"n": len(tagged), "by_dataset": dict(Counter(x["dataset"] for x in tagged)),
                      "gold": dict(Counter(x["gold"] for x in tagged)), "classifiers": {}}
        for k in keys:
            correct = sum(x["preds"][k] == x["gold"] for x in tagged)
            s = {"accuracy": correct / len(tagged) if tagged else None}
            if tag == "technical":
                s["called_negative"] = sum(x["preds"][k] == "negative" for x in tagged) / len(tagged) if tagged else None
            if tag == "sarcasm":
                s["called_positive"] = sum(x["preds"][k] == "positive" for x in tagged) / len(tagged) if tagged else None
            stats[tag]["classifiers"][k] = s
    # Baseline for the technical tag: other items whose gold label is not negative.
    others = [x for x in items_out if x["gold"] != "negative" and "technical" not in x["tags"]]
    stats["technical"]["baseline_called_negative"] = {
        k: sum(x["preds"][k] == "negative" for x in others) / len(others) for k in keys}
    stats["technical"]["baseline_n"] = len(others)
    # Politeness by dataset: conventions differ between the gold sets.
    stats["politeness"]["accuracy_by_dataset"] = {
        k: {ds: (lambda xs: sum(x["preds"][k] == x["gold"] for x in xs) / len(xs) if xs else None)(
            [x for x in items_out if "politeness" in x["tags"] and x["dataset"] == ds]) for ds in DATASETS}
        for k in keys}
    stats["politeness"]["gold_by_dataset"] = {
        ds: dict(Counter(x["gold"] for x in items_out if "politeness" in x["tags"] and x["dataset"] == ds))
        for ds in DATASETS}
    # Items where every supervised and API model agrees with each other but not with gold.
    strong = [k for k in keys if k in ("tfidf-lr", "embed-lr") or k.startswith(("gpt-6.1-sol", "gpt-6.1-terra"))]
    for x in items_out:
        votes = {x["preds"][k] for k in strong}
        x["all_strong_disagree_with_gold"] = len(votes) == 1 and x["gold"] not in votes
    stats["label_noise_candidates"] = {"classifiers": strong,
                                       "n": sum(x["all_strong_disagree_with_gold"] for x in items_out),
                                       "of": len(items_out)}

    header = ["Classifier"] + [f"technical: called negative (others {stats['technical']['baseline_n']})",
                               "politeness: accuracy", "sarcasm candidates: accuracy", "sarcasm: called positive"]
    body = []
    for k in keys:
        t, p, s = (stats[tag]["classifiers"][k] for tag in ("technical", "politeness", "sarcasm"))
        body.append([k, f"{_f(t['called_negative'])} ({_f(stats['technical']['baseline_called_negative'][k])})",
                     _f(p["accuracy"]), _f(s["accuracy"]), _f(s["called_positive"])])
    counts = (f"Tagged items in the LLM test sample ({len(items_out)} items): technical {stats['technical']['n']}, "
              f"politeness {stats['politeness']['n']}, sarcasm candidates {stats['sarcasm']['n']}.")
    return counts + "\n\n" + _table(header, body), stats, items_out


def politeness_table(stats: dict, keys: list[str]) -> str:
    header = ["Classifier"] + [f"{NAMES[ds]} accuracy" for ds in DATASETS]
    body = [[k] + [_f(stats["politeness"]["accuracy_by_dataset"][k][ds]) for ds in DATASETS] for k in keys]
    gold = "; ".join(f"{NAMES[ds]}: " + ", ".join(f"{v} {lab}" for lab, v in sorted(stats["politeness"]["gold_by_dataset"][ds].items()))
                     for ds in DATASETS)
    return f"Gold labels of the politeness-tagged items: {gold}.\n\n" + _table(header, body)


# Sarcasm is rare in the gold sets. These two items were found by searching the negative test
# items for markers ("nice", "great", "yeah right", "/s", ...) and reading the hits; both are
# in the LLM test sample.
SARCASM_IDS = ("gh-00296-2431386", "gh-01618-126398")


def examples_md(items: list[dict], tag: str, keys: list[str], limit: int = 6) -> str:
    """Tagged items on which the classifiers split, most contested first."""
    if tag == "noise":
        tagged = [x for x in items if x["all_strong_disagree_with_gold"]]
    elif tag == "sarcasm-real":
        tagged = [x for x in items if x["id"] in SARCASM_IDS]
    else:
        tagged = [x for x in items if tag in x["tags"]]
    def spread(x):
        c = Counter(x["preds"][k] for k in keys)
        return (-len(c), -min(c.values()), x["id"])
    chosen = sorted(tagged, key=spread)[:limit]
    show = [k for k in ("vader", "roberta", "tfidf-lr", "embed-lr", "gpt-6-luna-zs", "gpt-6.1-terra-fs", "gpt-6.1-sol-fs", "qwen3:8b-fs") if k in keys]
    header = ["Dataset", "Text", "Gold"] + show
    body = []
    for x in chosen:
        text = x["text"] if len(x["text"]) <= 160 else x["text"][:157] + "..."
        text = text.replace("|", "\\|").replace("\n", " ")
        body.append([NAMES[x["dataset"]], text, x["gold"]] + [x["preds"][k] for k in show])
    return _table(header, body)


def coverage(all_rows: list[dict], selected: dict) -> list[dict]:
    """Every planned LLM run that is not complete: what is missing and how much is done."""
    def done(ds: str, spec: str, n: int | None, run: int = 1) -> tuple[int, int]:
        items = load_sample(ds, "test", n)
        preds = _preds(ds, "test", spec, run)
        return sum(it["id"] in preds for it in items), len(items)

    gaps = []
    for row in all_rows:
        if row["family"] != "LLM":
            continue
        plans = [("LLM test sample", SAMPLE_N, 1)]
        if row["model"] in FULL_TEST_LLMS:
            plans.append(("full test split", None, 1))
        plans += [(f"variance run {k}", VARIANCE_N, k) for k in range(2, RUNS_PER_LLM + 1)]
        for what, n, run in plans:
            for ds in DATASETS:
                got, need = done(ds, row["spec"](ds), n, run)
                if got < need:
                    gaps.append({"classifier": row["name"], "what": what, "dataset": ds, "done": got,
                                 "needed": need, "spec": row["spec"](ds), "n": n, "run": run})
    luna = selected.get("openai:gpt-6-luna")
    if luna:
        for src in DATASETS:
            for tgt in DATASETS:
                if src != tgt:
                    spec = f"llm:openai:gpt-6-luna/{luna['few-shot']}@{src}"
                    got, need = done(tgt, spec, SAMPLE_N)
                    if got < need:
                        gaps.append({"classifier": f"gpt-6-luna, few-shot examples from {NAMES[src]}",
                                     "what": "cross-platform", "dataset": tgt, "done": got, "needed": need,
                                     "spec": spec, "n": SAMPLE_N, "run": 1})
    # The demo's example sentences: every LLM setting, and Embeddings + LR per training set.
    if EXAMPLES.exists():
        ids = [ex["id"] for ex in json.loads(EXAMPLES.read_text(encoding="utf-8"))["sentences"]]
        targets = [("Embeddings + LR", RUNS / "preds" / "examples" / ds / f"embed-lr-at-{ds}" / "run1.jsonl")
                   for ds in DATASETS]
        for row in all_rows:
            if row["family"] == "LLM":
                sources = DATASETS if row["kind"] == "few-shot" else DATASETS[:1]
                targets += [(row["name"], RUNS / "preds" / "examples" / (ds if row["kind"] == "few-shot" else "zs")
                             / bench.slug(row["spec"](ds)) / "run1.jsonl") for ds in sources]
        for name, path in targets:
            got = sum(i in bench.read_preds(path) for i in ids)
            if got < len(ids):
                gaps.append({"classifier": name, "what": "example sentences", "dataset": None, "done": got,
                             "needed": len(ids), "spec": None, "n": None, "run": 1})
    return gaps


def coverage_md(gaps: list[dict]) -> str:
    if not gaps:
        return "All planned runs completed."
    grouped: dict[tuple, list] = {}
    for g in gaps:
        grouped.setdefault((g["classifier"], g["what"].rstrip(" 0123456789")), []).append(g)
    body = [[name, what, ", ".join(sorted({NAMES[g["dataset"]] for g in gs if g["dataset"]})) or "–",
             f"{sum(g['done'] for g in gs)} of {sum(g['needed'] for g in gs)}"]
            for (name, what), gs in grouped.items()]
    return _table(["Classifier", "Run (not done: OpenAI credits ran out)", "Datasets affected",
                   "Items classified so far"], body)


def dataset_table(manifest: dict) -> str:
    header = ["Dataset", "Source", "Split", "Train", "of which dev", "Test", "Test classes (neg / neu / pos)",
              "Test texts also in train"]
    sources = {"github": "Novielli et al. 2020, figshare (CC BY 4.0)",
               "jira": "Ortu et al. 2016, polarity version from EMTk (MIT)",
               "so": "Calefato et al. 2018, Senti4SD / EMTk (MIT)"}
    splits = {"github": "seeded stratified 70/30 (no official split)", "jira": "official", "so": "official"}
    body = []
    for ds in DATASETS:
        s = manifest["datasets"][ds]["splits"]
        t = s["test"]["classes"]
        body.append([NAMES[ds], sources[ds], splits[ds], str(s["train"]["n"]), str(s["dev"]["n"]), str(s["test"]["n"]),
                     f"{t['negative']} / {t['neutral']} / {t['positive']}",
                     str(manifest["datasets"][ds]["test_texts_also_in_train"])])
    return _table(header, body)


def prompt_table(dev: dict, selected: dict) -> str:
    header = ["Model", "Prompt"] + [f"{NAMES[ds]} (n)" for ds in DATASETS] + ["Mean", "Cost / 1k", "Selected"]
    body = []
    for model, rows in dev.items():
        for prompt, r in rows.items():
            sel = [kind for kind, p in selected[model].items() if p == prompt]
            cost = statistics.fmean(v["cost_per_1k"] for v in r["datasets"].values())
            body.append([LLM_MODELS[model], prompt]
                        + [f"{r['datasets'][ds]['macro_f1']:.3f} ({r['datasets'][ds]['n']})" for ds in DATASETS]
                        + [f"{r['mean']:.3f}", "0 (local)" if model.startswith("ollama") else f"${cost:.3f}",
                           "**" + ", ".join(sel) + "**" if sel else ""])
    return _table(header, body)


def spend_breakdown() -> tuple[str, dict]:
    by = {}
    for path in (RUNS / "preds").glob("*/*/llm_*/run*.jsonl"):
        # preds/<dataset>/<split>/... or preds/examples/<few-shot source or zs>/...
        split = "demo example sentences" if path.parent.parent.parent.name == "examples" else path.parent.parent.name
        rows = bench.read_preds(path).values()
        spec_slug = path.parent.name
        model = next((m for m in LLM_MODELS if bench.slug(f"llm:{m}/").rstrip("_") in spec_slug), "other")
        key = (LLM_MODELS.get(model, model), split)
        cur = by.setdefault(key, {"calls": 0, "cost": 0.0})
        cur["calls"] += len(rows)
        cur["cost"] += sum(r.get("cost_usd") or 0 for r in rows)
    embed = bench.embedding_spend()
    total = sum(v["cost"] for v in by.values()) + embed
    body = [[m, split, str(v["calls"]), f"${v['cost']:.3f}"] for (m, split), v in sorted(by.items())]
    body.append(["text-embedding-3-small", "all splits", "–", f"${embed:.3f}"])
    body.append(["**Total**", "", str(sum(v["calls"] for v in by.values())), f"**${total:.2f}**"])
    return _table(["Model", "Split", "Calls", "Cost (USD)"], body), {"total_usd": total, "embeddings_usd": embed}


# ---------------------------------------------------------------- web export


def export_models(web: Path) -> list[Path]:
    """Weights for the browser try-box: TF-IDF + LR per dataset and the VADER lexicons."""
    from vaderSentiment import vaderSentiment as vs

    out = []
    for ds in DATASETS:
        best = json.loads((RUNS / "models" / f"tfidf_{ds}.json").read_text())["best"]
        train = load_split(ds, "train")
        fitted = clf.fit_tfidf([it["text"] for it in train], [it["label"] for it in train], **best)
        path = web / "models" / f"tfidf_{ds}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(clf.export_tfidf(fitted), separators=(",", ":")), encoding="utf-8")
        out.append(path)
    analyzer = vs.SentimentIntensityAnalyzer()
    path = web / "models" / "vader.json"
    path.write_text(json.dumps({"lexicon": analyzer.lexicon, "emojis": analyzer.emojis,
                                "booster": vs.BOOSTER_DICT, "negate": vs.NEGATE, "special": vs.SPECIAL_CASES},
                               ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    out.append(path)
    return out


def export_parity(web: Path) -> Path:
    """Python VADER compounds and TF-IDF labels (all three models) on every test item,
    so web/scripts/parity.ts can check the browser re-implementations against them."""
    rows = []
    for ds in DATASETS:
        vader = _preds(ds, "test", "vader")
        tfidf = {src: _preds(ds, "test", f"tfidf-lr@{src}") for src in DATASETS}
        rows += [{"dataset": ds, "text": it["text"], "vader": vader[it["id"]]["score"],
                  "tfidf": {src: tfidf[src][it["id"]]["label"] for src in DATASETS}}
                 for it in load_split(ds, "test")]
    path = web.parent.parent / "scripts" / "parity_fixture.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return path


EXAMPLES = Path("data/examples.json")


def export_examples(web: Path, selected: dict) -> Path:
    examples = json.loads(EXAMPLES.read_text(encoding="utf-8"))
    rows = classifier_rows(selected)
    out = []
    for ex in examples["sentences"]:
        compound = clf.vader_compound([ex["text"]])[0]
        llm = {}
        for row in rows:
            if row["family"] != "LLM":
                continue
            for ds in DATASETS:
                key = ds if row["kind"] == "few-shot" else "zs"
                path = RUNS / "preds" / "examples" / key / bench.slug(row["spec"](ds)) / "run1.jsonl"
                p = bench.read_preds(path).get(ex["id"])
                if p:
                    llm.setdefault(row["key"], {})[ds] = p["label"] if p["ok"] else None
        roberta = bench.read_preds(RUNS / "preds" / "examples" / "all" / "roberta-twitter" / "run1.jsonl").get(ex["id"])
        embed_lr = {ds: (bench.read_preds(RUNS / "preds" / "examples" / ds / f"embed-lr-at-{ds}" / "run1.jsonl")
                         .get(ex["id"]) or {}).get("label") for ds in DATASETS}
        out.append({**ex, "python_vader_compound": compound, "llm": llm,
                    "roberta": roberta["label"] if roberta else None, "embed_lr": embed_lr})
    path = web / "examples.json"
    path.write_text(json.dumps({"note": examples["note"], "sentences": out}, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


# ---------------------------------------------------------------- build


def build(out: Path, web: Path, partial: bool = False) -> list[Path]:
    manifest = json.loads(Path("data/processed/manifest.json").read_text())
    dev, selected = select_prompts()
    rows = classifier_rows(selected)
    full = {ds: load_split(ds, "test") for ds in DATASETS}
    sample = {ds: load_sample(ds, "test", SAMPLE_N) for ds in DATASETS}
    gaps = coverage(rows, selected)
    if gaps and not partial:
        raise FileNotFoundError(f"{len(gaps)} planned runs are incomplete (first: {gaps[0]}); "
                                "finish them or pass --partial to report what is there")
    complete = []
    for row in rows:
        missing = [ds for ds in DATASETS if _labels(sample[ds], _preds(ds, "test", row["spec"](ds))) is None]
        if missing and not partial:
            raise FileNotFoundError(f"missing test predictions for {row['spec'](missing[0])}")
        if not missing:
            complete.append(row)
    rows = complete

    full_rows = [r for r in rows if r["family"] != "LLM" or r["model"] in FULL_TEST_LLMS]
    res_full = evaluate(full_rows, full)
    res_sample = evaluate(rows, sample)
    # Descriptive comparisons, fixed when the report was written (no model choice depends on them).
    pairs = [("gpt-6.1-terra-fs", "embed-lr"), ("gpt-6.1-sol-fs", "embed-lr"), ("gpt-6-luna-fs", "embed-lr"),
             ("gpt-6.1-terra-fs", "gpt-6.1-sol-fs"), ("gpt-6.1-terra-fs", "gpt-6.1-terra-zs"),
             ("gpt-6.1-sol-fs", "gpt-6.1-sol-zs"), ("gpt-6-luna-fs", "gpt-6-luna-zs"), ("qwen3:8b-fs", "qwen3:8b-zs"),
             ("gpt-6.1-terra-fs", "tfidf-lr"), ("embed-lr", "tfidf-lr"), ("roberta", "vader")]
    paired_md, paired = paired_table(rows, sample, pairs)
    cross_md, cross = cross_platform(selected)
    cost_md, cost = cost_table(rows)
    var_md, var = variance_table(rows)
    agree_md, agree = agreement(rows, sample)
    errors_md, err_stats, err_items = error_analysis(rows, sample)
    spend_md, spend = spend_breakdown()
    keys = agree["keys"]

    tables = {
        "datasets": dataset_table(manifest),
        "prompt_dev": prompt_table(dev, selected),
        "main_full": results_table(full_rows, res_full),
        "per_class_full": per_class_table(full_rows, res_full),
        "main_sample": results_table(rows, res_sample),
        "per_class_sample": per_class_table(rows, res_sample),
        "paired": paired_md,
        "cross": cross_md,
        "cost": cost_md,
        "variance": var_md,
        "agreement": agree_md,
        "errors": errors_md,
        "politeness": politeness_table(err_stats, keys),
        "examples_technical": examples_md(err_items, "technical", keys),
        "examples_politeness": examples_md(err_items, "politeness", keys),
        "examples_sarcasm": examples_md(err_items, "sarcasm", keys),
        "examples_noise": examples_md(err_items, "noise", keys),
        "examples_sarcasm_real": examples_md(err_items, "sarcasm-real", keys),
        "spend": spend_md,
        "incomplete": coverage_md(gaps),
        "llm_calls": f"{sum(c.get('calls', 0) for c in cost):,}",
        "llm_failed": str(sum(c.get("failed", 0) for c in cost)),
        "spend_total": f"${spend['total_usd']:.2f}",
        "noise_n": f"{err_stats['label_noise_candidates']['n']} of {err_stats['label_noise_candidates']['of']}",
        "sample_n": str(SAMPLE_N),
        "variance_n": str(VARIANCE_N),
        "shots": str(3 * SHOTS_PER_CLASS),
        "prices": ", ".join(f"`{m}` ${p[0]:.2f} / ${p[1]:.2f} / ${p[2]:.2f}" for m, p in PRICES.items()),
        "prices_as_of": AS_OF,
        "prices_source": SOURCE,
        "generated": date.today().isoformat(),
    }
    template = TEMPLATE.read_text(encoding="utf-8")
    missing = set(re.findall(r"\{\{(\w+)\}\}", template)) - set(tables)
    if missing:
        raise KeyError(f"template placeholders without a table: {sorted(missing)}")
    report = re.sub(r"\{\{(\w+)\}\}", lambda m: tables[m.group(1)], template)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")

    web.mkdir(parents=True, exist_ok=True)
    meta = [{k: v for k, v in r.items() if k != "spec"} for r in rows]
    strip = lambda res: {k: {ds: {kk: vv for kk, vv in s.items()} for ds, s in v.items()} for k, v in res.items()}
    results = {"generated": tables["generated"], "datasets": manifest["datasets"], "names": NAMES,
               "classifiers": meta, "selected_prompts": selected, "prompt_dev": dev,
               "full": strip(res_full), "sample": strip(res_sample), "sample_n": SAMPLE_N, "variance_n": VARIANCE_N,
               "paired": paired, "cross": cross, "cost": cost, "variance": var, "agreement": agree,
               "errors": err_stats, "tags": TAGS, "spend": spend, "incomplete": gaps,
               "vader_tuned": json.loads((RUNS / "models" / "vader_tuned.json").read_text()),
               "prompts": {"zs-basic": BASIC, "zs-guided": GUIDED,
                           "shots": {ds: [{"text": mask(s["text"]), "label": s["label"]} for s in shots(ds)]
                                     for ds in DATASETS}},
               "prices": {"as_of": AS_OF, "source": SOURCE, "usd_per_million": PRICES}}
    paths = [out]
    for name, payload in (("results.json", results), ("errors.json", {"items": err_items, "keys": keys})):
        path = web / name
        path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        paths.append(path)
    paths += export_models(web)
    paths.append(export_parity(web))
    if EXAMPLES.exists():
        paths.append(export_examples(web, selected))
    return paths
