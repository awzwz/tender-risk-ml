"""src.metrics must reproduce scikit-learn exactly (it replaces it in torch code). Run: python -m pytest tests"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.metrics import average_precision, roc_auc


def test_matches_sklearn_with_ties_and_imbalance():
    rng = np.random.default_rng(0)
    for n, rate, levels in [(1000, 0.066, None), (5000, 0.3, 20), (200, 0.5, 3), (39000, 0.066, 500)]:
        y = (rng.random(n) < rate).astype(int)
        p = rng.random(n) + y * 0.3
        if levels:
            p = np.round(p * levels) / levels                  # many tied scores
        assert np.isclose(average_precision(y, p), average_precision_score(y, p), rtol=0, atol=1e-12)
        assert np.isclose(roc_auc(y, p), roc_auc_score(y, p), rtol=0, atol=1e-12)
