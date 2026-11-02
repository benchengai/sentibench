"""The non-LLM classifiers: VADER, TF-IDF + logistic regression, embeddings + logistic regression.

- VADER (Hutto & Gilbert 2014) is a general-purpose lexicon with rules for
  negation, intensifiers, capitals and punctuation. Its compound score is
  mapped to a label with the authors' thresholds (+/-0.05), or with thresholds
  tuned on a dev split ("vader-tuned").
- TF-IDF + logistic regression is the supervised SE baseline: word uni- and
  bigrams, trained on one dataset's train split. It is the same family as
  SentiCR (bag of words), and small enough to export to the browser.
- Embeddings + logistic regression uses OpenAI's text-embedding-3-small as a
  general pre-trained representation with a supervised SE head.
- RoBERTa-Twitter (cardiffnlp/twitter-roberta-base-sentiment-latest, Loureiro
  et al. 2022) is a general-domain pre-trained sentiment model, fine-tuned on
  tweets, used off the shelf on CPU with no SE training.

Hyperparameters are chosen on dev with the model fitted on fit; the chosen
setting is then refitted on train (fit + dev) before test is scored.
"""

from __future__ import annotations

import itertools

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .data import LABELS
from .metrics import encode, macro_f1_cm, confusion

VADER_DEFAULT = (-0.05, 0.05)


_vader = None


def vader_compound(texts: list[str]) -> list[float]:
    global _vader
    if _vader is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _vader = SentimentIntensityAnalyzer()
    return [_vader.polarity_scores(t)["compound"] for t in texts]


def vader_label(compound: float, thresholds: tuple[float, float] = VADER_DEFAULT) -> str:
    lo, hi = thresholds
    if compound >= hi:
        return "positive"
    if compound <= lo:
        return "negative"
    return "neutral"


def tune_vader(compounds: list[float], gold: list[str]) -> tuple[tuple[float, float], float]:
    """Grid search of (negative, positive) thresholds that maximise macro-F1 on dev."""
    g = encode(gold)
    grid = np.round(np.arange(0.0, 0.951, 0.05), 2)
    best = (VADER_DEFAULT, -1.0)
    for lo, hi in itertools.product(-grid, grid):
        if lo == 0 and hi == 0:
            continue
        pred = encode([vader_label(c, (lo, hi)) for c in compounds])
        f1 = macro_f1_cm(confusion(g, pred))
        if f1 > best[1] + 1e-12:
            best = ((float(lo), float(hi)), f1)
    return best


ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def roberta_predict(texts: list[str], batch_size: int = 32) -> list[tuple[str, list[float]]]:
    """(label, [p_negative, p_neutral, p_positive]) per text. Needs the `hf` extra."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL).eval()
    id2label = model.config.id2label
    assert [id2label[i] for i in range(3)] == list(LABELS), id2label
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                              max_length=256, return_tensors="pt")
            probs = model(**batch).logits.softmax(-1).tolist()
            out += [(LABELS[max(range(3), key=p.__getitem__)], [round(x, 4) for x in p]) for p in probs]
    return out


TFIDF_GRID = {"C": (0.5, 1.0, 2.0, 4.0, 8.0, 16.0), "class_weight": (None, "balanced")}
TFIDF_OPTIONS = {"ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True, "max_features": 20000}


def fit_tfidf(texts: list[str], labels: list[str], C: float, class_weight: str | None):
    vectorizer = TfidfVectorizer(**TFIDF_OPTIONS)
    X = vectorizer.fit_transform(texts)
    model = LogisticRegression(C=C, class_weight=class_weight, max_iter=5000)
    model.fit(X, labels)
    return vectorizer, model


def predict_tfidf(fitted, texts: list[str]) -> list[str]:
    vectorizer, model = fitted
    return list(model.predict(vectorizer.transform(texts)))


EMBED_GRID = {"C": (0.25, 0.5, 1.0, 2.0, 4.0, 8.0), "class_weight": (None, "balanced")}


def fit_embed(X: np.ndarray, labels: list[str], C: float, class_weight: str | None):
    model = LogisticRegression(C=C, class_weight=class_weight, max_iter=5000)
    model.fit(X, labels)
    return model


def grid_search(fit, predict, train: tuple, dev: tuple, grid: dict) -> tuple[dict, float, list]:
    """Fit on train, score macro-F1 on dev for every setting; returns the best and the full table."""
    table = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid, values))
        fitted = fit(*train, **params)
        f1 = macro_f1_cm(confusion(encode(dev[1]), encode(predict(fitted, dev[0]))))
        table.append({**params, "dev_macro_f1": f1})
    best = max(table, key=lambda r: r["dev_macro_f1"])
    return {k: best[k] for k in grid}, best["dev_macro_f1"], table


def export_tfidf(fitted, decimals: int = 4) -> dict:
    """Vocabulary, idf and per-class weights for the browser re-implementation in web/lib/tfidf.ts."""
    vectorizer, model = fitted
    vocab = vectorizer.vocabulary_
    terms = sorted(vocab, key=vocab.get)
    assert list(model.classes_) == sorted(LABELS)
    return {
        "classes": list(model.classes_),
        "ngram_range": list(TFIDF_OPTIONS["ngram_range"]),
        "sublinear_tf": TFIDF_OPTIONS["sublinear_tf"],
        "terms": terms,
        "idf": [round(float(v), decimals) for v in vectorizer.idf_],
        "coef": [[round(float(v), decimals) for v in row] for row in model.coef_],
        "intercept": [round(float(v), decimals) for v in model.intercept_],
    }
