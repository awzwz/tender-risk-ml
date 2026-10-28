"""
Ranking metrics in plain NumPy, for code that runs next to torch.

On macOS torch and scikit-learn / LightGBM ship different copies of libomp, and loading both in one process
segfaults, so the torch notebooks and scripts cannot import sklearn.metrics. These functions reproduce
`average_precision_score` and `roc_auc_score` exactly (checked against scikit-learn in tests/test_metrics.py).
"""
from __future__ import annotations

import numpy as np


def average_precision(y: np.ndarray, p: np.ndarray) -> float:
    """Average precision (= PR-AUC as reported by scikit-learn): sum over thresholds of ΔRecall × Precision."""
    y = np.asarray(y).astype(float); p = np.asarray(p, dtype=float)
    order = np.argsort(-p, kind="mergesort")
    y, p = y[order], p[order]
    last = np.r_[np.where(np.diff(p))[0], len(p) - 1]        # last index of every block of tied scores
    tp = np.cumsum(y)[last]
    fp = (last + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC-AUC via the rank-sum (Mann–Whitney) statistic, ties counted as one half."""
    y = np.asarray(y).astype(bool); p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]                               # average rank within ties
    n_pos, n_neg = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def top_k(y: np.ndarray, p: np.ndarray, k: float) -> tuple[float, float]:
    """Precision among the top k share of scores and the share of all positives found there."""
    y = np.asarray(y); n = int(len(p) * k)
    idx = np.argsort(-np.asarray(p), kind="mergesort")[:n]
    return float(y[idx].mean()), float(y[idx].sum() / y.sum())
