"""Agreement metrics: per-class and macro F1, Cohen's kappa, bootstrap intervals.

A failed prediction (no valid label) is kept as a fourth category that never
matches gold: it lowers recall and kappa and is never credited to a class. So a
model cannot raise its score by failing on hard items; failure rates are also
reported on their own.
"""

from __future__ import annotations

import numpy as np

from .data import LABELS

FAILED = len(LABELS)  # code for a missing or invalid prediction
K = FAILED + 1


def encode(labels: list[str | None]) -> np.ndarray:
    index = {label: i for i, label in enumerate(LABELS)}
    return np.array([index.get(label, FAILED) if label else FAILED for label in labels], dtype=np.int64)


def confusion(gold: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """K x K counts, rows gold, columns predicted (last column: failed)."""
    return np.bincount(gold * K + pred, minlength=K * K).reshape(K, K)


def _per_class(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(LABELS)
    tp = np.diag(cm)[:n].astype(float)
    predicted = cm[:, :n].sum(axis=0).astype(float)
    actual = cm[:n, :].sum(axis=1).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros(n), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros(n), where=actual > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(n), where=denom > 0)
    return precision, recall, f1


def macro_f1_cm(cm: np.ndarray) -> float:
    return float(_per_class(cm)[2].mean())


def kappa_cm(cm: np.ndarray) -> float:
    """Cohen's kappa from a square confusion matrix."""
    total = cm.sum()
    if total == 0:
        return float("nan")
    observed = np.trace(cm) / total
    expected = float((cm.sum(axis=0) * cm.sum(axis=1)).sum()) / total**2
    return 1.0 if expected == 1 else float((observed - expected) / (1 - expected))


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    return kappa_cm(confusion(a, b))


def scores(gold: np.ndarray, pred: np.ndarray) -> dict:
    cm = confusion(gold, pred)
    precision, recall, f1 = _per_class(cm)
    neg, pos = LABELS.index("negative"), LABELS.index("positive")
    return {
        "n": int(len(gold)),
        "macro_f1": float(f1.mean()),
        "kappa": kappa_cm(cm),
        "accuracy": float(np.trace(cm) / cm.sum()) if cm.sum() else float("nan"),
        "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
                      for i, label in enumerate(LABELS)},
        # Positive read as negative or the reverse: the "severe" disagreement of Novielli et al.
        "severe_error_rate": float((cm[neg, pos] + cm[pos, neg]) / cm.sum()) if cm.sum() else float("nan"),
        "failed": int(cm[:, FAILED].sum()),
        "confusion": cm[: len(LABELS)].tolist(),
    }


def bootstrap(gold: np.ndarray, pred: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """95 % percentile intervals for macro-F1 and kappa, resampling items."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(gold), size=(n_boot, len(gold)))
    f1s, kappas = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        cm = confusion(gold[idx[b]], pred[idx[b]])
        f1s[b], kappas[b] = macro_f1_cm(cm), kappa_cm(cm)
    return {"macro_f1": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
            "kappa": [float(np.percentile(kappas, 2.5)), float(np.percentile(kappas, 97.5))]}


def paired_difference(gold: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                      n_boot: int = 1000, seed: int = 0) -> dict:
    """Macro-F1 of a minus b on the same items, with a paired bootstrap interval."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(gold), size=(n_boot, len(gold)))
    diffs = np.array([macro_f1_cm(confusion(gold[i], pred_a[i])) - macro_f1_cm(confusion(gold[i], pred_b[i]))
                      for i in idx])
    point = macro_f1_cm(confusion(gold, pred_a)) - macro_f1_cm(confusion(gold, pred_b))
    return {"diff": float(point), "ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            # Share of resamples where the sign flips: a rough two-sided check.
            "p_sign": float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))}


def run_consistency(runs: list[np.ndarray]) -> dict:
    """Label stability across repeated runs on the same items."""
    stacked = np.stack(runs)
    unanimous = (stacked == stacked[0]).all(axis=0)
    pairs = [kappa(runs[i], runs[j]) for i in range(len(runs)) for j in range(i + 1, len(runs))]
    return {"runs": len(runs), "items": int(stacked.shape[1]),
            "items_changed": int((~unanimous).sum()), "flip_rate": float((~unanimous).mean()),
            "mean_pairwise_kappa": float(np.mean(pairs)) if pairs else float("nan")}
