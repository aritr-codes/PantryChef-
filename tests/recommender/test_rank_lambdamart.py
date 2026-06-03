import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm")

from pantrychef.recommender.rank import LambdaMARTRanker  # noqa: E402


def test_lambdamart_ranks_signal_above_noise():
    # train == test; smoke test that the ranker fits + learns the signal
    rng = np.random.default_rng(0)
    X_rows, y_rows, groups = [], [], []
    for _ in range(20):
        feats = rng.normal(0, 1, (5, 3))
        gold = rng.integers(0, 5)
        feats[gold, 0] += 5.0  # signal in feature 0
        labels = np.zeros(5)
        labels[gold] = 1
        X_rows.append(feats)
        y_rows.append(labels)
        groups.append(5)
    X = np.vstack(X_rows)
    y = np.concatenate(y_rows)
    m = LambdaMARTRanker(seed=0).fit(X, y, groups=groups)
    scores = m.score(X)
    group_size, n_groups = 5, 20
    mrr = 0.0
    for i in range(n_groups):
        s = scores[i * group_size : (i + 1) * group_size]
        gold_idx = int(np.argmax(y[i * group_size : (i + 1) * group_size]))
        rank = np.argsort(-s).tolist().index(gold_idx) + 1
        mrr += 1.0 / rank
    mrr /= n_groups
    assert mrr >= 0.8  # +5 sigma signal; well-trained LambdaMART approaches 1.0


def test_lambdamart_unfitted_raises():
    with pytest.raises(RuntimeError):
        LambdaMARTRanker().score(np.zeros((1, 3)))
