from pantrychef.common.types import Recipe
from pantrychef.recommender.config import RecConfig
from pantrychef.recommender.features import FEATURE_NAMES
from pantrychef.recommender.train import build_dataset, train_ranker
from pantrychef.retrieval.index import InvertedIndex


def _corpus():
    return [
        Recipe(
            recipe_id=str(i),
            title=str(i),
            canonical=["egg", "flour", "milk", "sugar", "butter"][: 4 + (i % 2)],
        )
        for i in range(20)
    ]


def test_build_dataset_shapes_and_labels():
    corpus = _corpus()
    idx = InvertedIndex.build(corpus)
    X, y, groups, stats = build_dataset(
        corpus, idx, sub_lookup=None, cfg=RecConfig(seed=1), columns=FEATURE_NAMES
    )
    assert X.shape[1] == len(FEATURE_NAMES)
    assert X.shape[0] == len(y) == sum(groups)
    # every kept group has exactly one positive (the gold recipe)
    start = 0
    for g in groups:
        assert y[start : start + g].sum() == 1
        start += g
    assert stats["n_queries_kept"] == len(groups)
    # n_dropped_no_gold stays 0 here: masking keeps gold reachable in its own pool
    assert "n_dropped_no_gold" in stats


def test_build_dataset_drops_when_pool_empty():
    # candidate_cap=0 forces every pool empty -> all queries dropped
    corpus = _corpus()
    idx = InvertedIndex.build(corpus)
    X, y, groups, stats = build_dataset(
        corpus, idx, sub_lookup=None, cfg=RecConfig(seed=1, candidate_cap=0), columns=FEATURE_NAMES
    )
    assert len(groups) == 0
    assert stats["n_queries_kept"] == 0
    assert X.shape == (0, len(FEATURE_NAMES))


def test_train_ranker_smoke():
    from pantrychef.recommender.rank import LinearRanker

    corpus = _corpus()
    idx = InvertedIndex.build(corpus)
    model, stats = train_ranker(LinearRanker(), corpus, idx, sub_lookup=None, cfg=RecConfig(seed=1))
    assert model.w_ is not None  # fitted
    assert stats["n_queries_kept"] > 0


def test_build_dataset_respects_max_train_queries():
    corpus = _corpus()
    idx = InvertedIndex.build(corpus)
    _, _, groups, stats = build_dataset(
        corpus,
        idx,
        sub_lookup=None,
        cfg=RecConfig(seed=1, max_train_queries=3),
        columns=FEATURE_NAMES,
    )
    assert stats["n_queries_total"] <= 3
    assert len(groups) <= 3
