import numpy as np

from pantrychef.common.types import Recipe
from pantrychef.recommender.recommend import candidate_pool, rerank
from pantrychef.retrieval.index import InvertedIndex


def _idx():
    recipes = [
        Recipe(recipe_id="a", title="A", canonical=["egg", "flour", "milk"]),
        Recipe(recipe_id="b", title="B", canonical=["egg", "sugar"]),
        Recipe(recipe_id="c", title="C", canonical=["beef", "onion"]),  # no overlap
    ]
    return InvertedIndex.build(recipes)


def test_candidate_pool_overlap_ordered():
    idx = _idx()
    pool = candidate_pool(idx, {"egg", "flour", "milk"}, cap=10)
    ids = [r.recipe_id for r in pool]
    assert "c" not in ids  # no overlap
    assert ids[0] == "a"  # full coverage (3/3) ranks above b (1/2)


def test_candidate_pool_respects_cap():
    idx = _idx()
    pool = candidate_pool(idx, {"egg"}, cap=1)
    assert len(pool) == 1


def test_rerank_uses_model_scores():
    idx = _idx()
    pantry = {"egg", "flour", "milk"}

    class FakeModel:
        # score by n_matched column (index 2) so b (1) -> 5, a (3) -> 1; b ranks first
        def score(self, X):
            return np.array([1.0 if row[2] == 3.0 else 5.0 for row in X])

    def feat_fn(p, recipe):
        from pantrychef.recommender.features import extract_features

        return extract_features(p, recipe, sub_lookup=None)

    ranked = rerank(idx, pantry, FakeModel(), feat_fn, k=2, cap=10)
    assert ranked[0].recipe_id == "b"
    assert isinstance(ranked[0].score, float)
    assert ranked[0].missing  # b is missing sugar


def test_rerank_empty_pantry_returns_empty():
    idx = _idx()

    class FakeModel:
        def score(self, X):
            return np.zeros(len(X))

    out = rerank(idx, set(), FakeModel(), lambda p, r: {}, k=5, cap=10)
    assert out == []
