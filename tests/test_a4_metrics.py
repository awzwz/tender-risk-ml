import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assignment4"))
from a4_metrics import accuracy, confusion, macro_f1  # noqa: E402


def test_against_sklearn():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 3, 500), rng.integers(0, 3, 500)
    p[:5] = 0                                       # make sure every class appears in predictions
    assert np.isclose(accuracy(y, p), accuracy_score(y, p))
    assert (confusion(y, p, 3) == confusion_matrix(y, p, labels=[0, 1, 2])).all()
    assert np.isclose(macro_f1(y, p, 3), f1_score(y, p, average="macro"))
    q = np.zeros_like(y)                            # a class never predicted -> its F1 is 0, as in sklearn
    assert np.isclose(macro_f1(y, q, 3), f1_score(y, q, average="macro", zero_division=0))
