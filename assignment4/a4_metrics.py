"""Metrics in plain numpy, for the torch notebooks (P4–P6).

On macOS, torch and scikit-learn/LightGBM in one process load two OpenMP runtimes and can crash, so the torch
notebooks do not import scikit-learn. Checked against sklearn in tests/test_a4_metrics.py.
"""
import numpy as np


def accuracy(y, pred):
    return float(np.mean(np.asarray(y) == np.asarray(pred)))


def confusion(y, pred, k):
    m = np.zeros((k, k), dtype=int)
    for t, p in zip(y, pred):
        m[t, p] += 1
    return m


def f1_per_class(y, pred, k):
    m = confusion(y, pred, k)
    tp = np.diag(m).astype(float)
    prec = np.divide(tp, m.sum(0), out=np.zeros(k), where=m.sum(0) > 0)
    rec = np.divide(tp, m.sum(1), out=np.zeros(k), where=m.sum(1) > 0)
    return np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)


def macro_f1(y, pred, k):
    return float(f1_per_class(y, pred, k).mean())


def bootstrap_ci(y, pred, metric, n=1000, seed=42):
    """95 % bootstrap interval of metric(y, pred) over resampled test rows."""
    y, pred = np.asarray(y), np.asarray(pred)
    rng = np.random.default_rng(seed)
    vals = [metric(y[i], pred[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
