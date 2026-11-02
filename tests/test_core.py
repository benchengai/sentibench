import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from sentibench import bench, classifiers as clf, data, metrics, prompts
from sentibench.analysis import tags_for
from sentibench.pricing import cost_usd, embedding_cost_usd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed"


def test_github_text_cleaning():
    assert data.clean_github_text('Reverted."') == "Reverted."
    assert data.clean_github_text('It\'s only a ""duplicate"" to improve performance."') == \
        'It\'s only a "duplicate" to improve performance.'
    assert data.clean_github_text("no artefact here") == "no artefact here"


def test_processed_splits_match_published_sizes():
    manifest = json.loads((PROCESSED / "manifest.json").read_text())
    sizes = {name: info["splits"]["train"]["n"] + info["splits"]["test"]["n"]
             for name, info in manifest["datasets"].items()}
    # Novielli et al. (MSR 2020), Table 1; one Jira test row is empty and dropped.
    assert sizes == {"github": 7122, "jira": 5868, "so": 4423}
    for name in data.DATASETS:
        fit, dev, test = (data.load_split(name, s, PROCESSED) for s in ("fit", "dev", "test"))
        ids = [it["id"] for it in fit + dev + test]
        assert len(ids) == len(set(ids)), name
        assert {it["label"] for it in fit + dev + test} == set(data.LABELS)


def test_stratified_split_keeps_class_balance():
    items = [{"id": str(i), "text": "t", "label": data.LABELS[i % 3]} for i in range(300)]
    rest, held = data.stratified_split(items, 0.2, seed=1)
    assert len(held) == 60 and len(rest) == 240
    assert Counter(it["label"] for it in held) == {label: 20 for label in data.LABELS}


def test_stratified_order_prefixes_are_balanced_and_nested():
    items = [{"id": str(i), "text": "t", "label": "neutral" if i < 600 else ("positive" if i < 800 else "negative")}
             for i in range(1000)]
    order = data.stratified_order(items, seed=3)
    assert sorted(order) == sorted(it["id"] for it in items)
    label = {it["id"]: it["label"] for it in items}
    for n in (50, 100, 300):
        c = Counter(label[i] for i in order[:n])
        assert abs(c["neutral"] / n - 0.6) <= 0.03 and abs(c["positive"] / n - 0.2) <= 0.03


def test_samples_are_prefixes():
    small = data.load_sample("jira", "test", 60, PROCESSED)
    large = data.load_sample("jira", "test", 250, PROCESSED)
    assert [it["id"] for it in large[:60]] == [it["id"] for it in small]


def test_scores_known_values():
    gold = metrics.encode(["positive", "positive", "neutral", "negative"])
    perfect = metrics.scores(gold, gold)
    assert perfect["macro_f1"] == 1.0 and perfect["kappa"] == 1.0
    pred = metrics.encode(["positive", "neutral", "neutral", "positive"])
    s = metrics.scores(gold, pred)
    # positive: P 1/2 R 1/2; neutral: P 1/2 R 1; negative: 0.
    assert s["per_class"]["positive"]["f1"] == pytest.approx(0.5)
    assert s["per_class"]["neutral"]["f1"] == pytest.approx(2 / 3)
    assert s["macro_f1"] == pytest.approx((0.5 + 2 / 3) / 3)
    assert s["severe_error_rate"] == pytest.approx(0.25)


def test_failures_count_against_the_classifier():
    gold = metrics.encode(["positive", "neutral", "negative", "neutral"])
    pred = metrics.encode(["positive", None, "negative", "neutral"])
    s = metrics.scores(gold, pred)
    assert s["failed"] == 1
    assert s["per_class"]["neutral"]["recall"] == 0.5
    assert s["per_class"]["neutral"]["precision"] == 1.0
    assert s["macro_f1"] < 1.0


def test_kappa_matches_hand_computation():
    a = metrics.encode(["positive"] * 20 + ["negative"] * 30)
    b = metrics.encode(["positive"] * 15 + ["negative"] * 5 + ["positive"] * 10 + ["negative"] * 20)
    # observed 0.7; expected 0.4*0.5 + 0.6*0.5 = 0.5 -> kappa 0.4
    assert metrics.kappa(a, b) == pytest.approx(0.4)


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(0)
    gold = rng.integers(0, 3, 400)
    pred = np.where(rng.random(400) < 0.75, gold, rng.integers(0, 3, 400))
    point = metrics.scores(gold, pred)["macro_f1"]
    ci = metrics.bootstrap(gold, pred, n_boot=300)["macro_f1"]
    assert ci[0] < point < ci[1]
    diff = metrics.paired_difference(gold, gold, pred, n_boot=300)
    assert diff["diff"] > 0 and diff["ci"][0] > 0


def test_run_consistency():
    r1 = metrics.encode(["positive", "neutral", "negative", "neutral"])
    r2 = metrics.encode(["positive", "neutral", "negative", "positive"])
    c = metrics.run_consistency([r1, r1, r2])
    assert c["items_changed"] == 1 and c["flip_rate"] == 0.25


def test_vader_thresholds():
    assert clf.vader_label(0.05) == "positive"
    assert clf.vader_label(-0.05) == "negative"
    assert clf.vader_label(0.0) == "neutral"
    assert clf.vader_label(0.3, (-0.2, 0.45)) == "neutral"
    (lo, hi), f1 = clf.tune_vader([0.9, 0.6, 0.1, -0.1, -0.7], ["positive", "positive", "neutral", "neutral", "negative"])
    assert f1 == 1.0 and lo < -0.1 and 0.1 < hi <= 0.6


def test_tfidf_export_reproduces_sklearn_scores():
    texts = ["thanks this is great", "great work thanks", "this is broken and awful", "awful crash again",
             "the build runs on linux", "update the docs on linux"] * 3
    labels = ["positive", "positive", "negative", "negative", "neutral", "neutral"] * 3
    fitted = clf.fit_tfidf(texts, labels, C=4.0, class_weight=None)
    exported = clf.export_tfidf(fitted, decimals=8)
    vectorizer, model = fitted
    x = vectorizer.transform(["thanks, great build"]).toarray()[0]
    # Recompute the decision function from the exported weights alone.
    idx = {t: i for i, t in enumerate(exported["terms"])}
    tokens = ["thanks", "great", "build"]
    grams = tokens + [" ".join(p) for p in zip(tokens, tokens[1:])]
    v = np.zeros(len(idx))
    for g in grams:
        if g in idx:
            v[idx[g]] += 1
    v = np.where(v > 0, 1 + np.log(np.where(v > 0, v, 1)), 0) * np.array(exported["idf"])
    v = v / np.linalg.norm(v)
    assert np.allclose(v, x, atol=1e-6)
    decision = np.array(exported["coef"]) @ v + np.array(exported["intercept"])
    assert np.allclose(decision, model.decision_function([x])[0], atol=1e-5)


def test_prompts():
    assert prompts.parse_prompt_id("fs-guided@jira") == ("fs-guided", "jira")
    with pytest.raises(ValueError):
        prompts.parse_prompt_id("fs-guided")
    with pytest.raises(ValueError):
        prompts.parse_prompt_id("cot")
    shots = prompts.shots("github")
    assert Counter(s["label"] for s in shots) == {label: prompts.SHOTS_PER_CLASS for label in data.LABELS}
    fit_ids = {it["id"] for it in data.load_split("github", "fit", PROCESSED)}
    assert all(s["id"] in fit_ids for s in shots)  # examples never come from dev or test
    system = prompts.system_prompt("fs-basic@github")
    assert system.startswith(prompts.BASIC) and all(s["text"] in system for s in shots)
    assert prompts.user_prompt("x" * 3000).endswith("[...]\"\"\"")


def test_slug_is_filesystem_safe():
    s = bench.slug("llm:ollama:qwen3:8b/fs-guided@jira")
    assert "/" not in s and ":" not in s and "@" not in s


def test_classify_records_failures(monkeypatch):
    def fake(system, prompt, schema, name, spec, effort):
        if "bad" in prompt:
            return '{"label": "angry"}', {"model": "m", "input_tokens": 10, "cached_input_tokens": 0,
                                          "output_tokens": 3, "reasoning_tokens": 0, "latency_s": 0.1,
                                          "cost_usd": 0.0}
        if "boom" in prompt:
            raise bench.LLMError("output hit the token limit")
        return '{"label": "neutral"}', {"model": "m", "input_tokens": 10, "cached_input_tokens": 0,
                                        "output_tokens": 3, "reasoning_tokens": 0, "latency_s": 0.1, "cost_usd": 1e-5}

    monkeypatch.setattr(bench, "complete_json", fake)
    assert bench.classify("fine", "openai:gpt-6-luna", "zs-basic")["label"] == "neutral"
    bad = bench.classify("bad", "openai:gpt-6-luna", "zs-basic")
    assert not bad["ok"] and "invalid output" in bad["error"]
    boom = bench.classify("boom", "openai:gpt-6-luna", "zs-basic")
    assert not boom["ok"] and boom["label"] is None


def test_run_llm_resumes_from_cache(monkeypatch, tmp_path):
    calls = []

    def fake(text, model_spec, prompt_id):
        calls.append(text)
        return {"label": "neutral", "ok": True, "cost_usd": 0.001}

    monkeypatch.setattr(bench, "classify", fake)
    monkeypatch.chdir(ROOT)
    spec = "llm:openai:gpt-6-luna/zs-basic"
    first = bench.run_llm(spec, "so", "dev", 5, root=tmp_path)
    again = bench.run_llm(spec, "so", "dev", 8, root=tmp_path)
    assert first["new"] == 5 and again["new"] == 3 and len(calls) == 8
    with pytest.raises(bench.BudgetExceeded):
        bench.run_llm("llm:openai:gpt-6-luna/zs-guided", "so", "dev", 20, budget_usd=0.012, root=tmp_path)


def test_costs():
    # 1,000 input (200 cached) and 100 output tokens on gpt-6.1-sol: 800*2 + 200*0.1 + 100*10 per million.
    assert cost_usd("openai", "gpt-6.1-sol", 1000, 200, 100) == pytest.approx(0.00262)
    assert cost_usd("ollama", "qwen3:8b", 1000, 0, 100) == 0.0
    assert cost_usd("openai", "unknown-model", 1, 0, 1) is None
    assert embedding_cost_usd("text-embedding-3-small", 1_000_000) == pytest.approx(0.02)


def test_error_tags():
    assert "technical" in tags_for("Kill the process and restart the server.", "neutral")
    assert "technical" not in tags_for("Kill the process and restart the server.", "negative")
    assert "politeness" in tags_for("Thanks for the patch!", "positive")
    assert "politeness" in tags_for("Could you please add a test?", "neutral")
