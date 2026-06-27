from __future__ import annotations

import numpy as np
import pytest

from pantrychef.common.types import Recipe
from pantrychef.recommender.rank import LambdaMARTRanker, LinearRanker
from pantrychef.recommender.recommend import candidate_pool
from pantrychef.retrieval.index import InvertedIndex


def test_linear_state_roundtrip_preserves_scores() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 4))
    y = np.array([0, 1] * 6)

    fitted = LinearRanker(seed=7).fit(X, y, groups=[12])
    restored = LinearRanker.from_state(fitted.to_state())

    assert np.allclose(restored.score(X), fitted.score(X))


def test_lambdamart_state_roundtrip_preserves_scores() -> None:
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(0)
    X_rows, y_rows, groups = [], [], []
    for _ in range(8):
        feats = rng.normal(0, 1, (4, 3))
        gold = rng.integers(0, 4)
        feats[gold, 0] += 5.0
        labels = np.zeros(4)
        labels[gold] = 1
        X_rows.append(feats)
        y_rows.append(labels)
        groups.append(4)
    X = np.vstack(X_rows)
    y = np.concatenate(y_rows)

    fitted = LambdaMARTRanker(seed=11).fit(X, y, groups)
    restored = LambdaMARTRanker.from_state(fitted.to_state())

    assert np.allclose(restored.score(X), fitted.score(X))


def test_inverted_index_state_roundtrip_preserves_candidate_pool() -> None:
    recipes = [
        Recipe(recipe_id=str(i), title=str(i), canonical=["egg", "flour", "milk", "butter"])
        for i in range(6)
    ] + [
        Recipe(recipe_id=f"o{i}", title=str(i), canonical=["egg", "oil", "salt", "pepper"])
        for i in range(6)
    ]
    fitted = InvertedIndex.build(recipes)
    restored = InvertedIndex.from_state(fitted.to_state())

    expected = [r.recipe_id for r in candidate_pool(fitted, {"egg", "flour"}, cap=5)]
    got = [r.recipe_id for r in candidate_pool(restored, {"egg", "flour"}, cap=5)]

    assert got == expected
