import numpy as np

from pantrychef.recommender.rank import LinearRanker


def test_linear_learns_separable_signal():
    rng = np.random.default_rng(0)
    pos = np.column_stack([rng.normal(3, 0.5, 100), rng.normal(0, 1, 100)])
    neg = np.column_stack([rng.normal(-3, 0.5, 100), rng.normal(0, 1, 100)])
    X = np.vstack([pos, neg])
    y = np.array([1] * 100 + [0] * 100)
    m = LinearRanker(seed=0).fit(X, y, groups=[200])
    scores = m.score(X)
    assert scores[:100].min() > scores[100:].max()


def test_linear_scaler_is_train_only():
    X = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    y = np.array([0, 1, 1])
    m = LinearRanker(seed=0).fit(X, y, groups=[3])
    assert m.mean_ is not None and m.std_ is not None
    assert np.allclose(m.mean_, X.mean(axis=0))
    assert np.allclose(m.std_, X.std(axis=0))


def test_score_raises_before_fit():
    import pytest

    with pytest.raises(RuntimeError, match="not fitted"):
        LinearRanker().score(np.zeros((1, 2)))


def test_score_length_matches_rows():
    X = np.random.default_rng(1).normal(size=(7, 3))
    m = LinearRanker(seed=0).fit(X, np.array([0, 1, 0, 1, 1, 0, 1]), groups=[7])
    assert len(m.score(X)) == 7
