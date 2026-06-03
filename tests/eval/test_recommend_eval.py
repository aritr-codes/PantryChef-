from pantrychef.common.types import Recipe
from pantrychef.eval.recommend_eval import evaluate, mrr_at_k, recall_at_k
from pantrychef.recommender.config import RecConfig
from pantrychef.recommender.features import FEATURE_NAMES
from pantrychef.recommender.rank import LinearRanker
from pantrychef.recommender.train import train_ranker
from pantrychef.retrieval.index import InvertedIndex


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], "b", 3) == 1.0
    assert recall_at_k(["a", "b", "c"], "b", 1) == 0.0
    assert recall_at_k(["a", "b", "c"], "z", 3) == 0.0


def test_mrr_at_k():
    assert mrr_at_k(["a", "b", "c"], "a", 10) == 1.0
    assert mrr_at_k(["a", "b", "c"], "b", 10) == 0.5
    assert mrr_at_k(["a", "b", "c"], "b", 1) == 0.0


def test_evaluate_reports_ceiling_and_conditional():
    corpus = [
        Recipe(
            recipe_id=str(i),
            title=str(i),
            canonical=["egg", "flour", "milk", "sugar", "butter", "salt"][: 4 + (i % 3)],
        )
        for i in range(60)
    ]
    idx = InvertedIndex.build(corpus)
    cfg = RecConfig(seed=2)
    model, _ = train_ranker(LinearRanker(seed=0), corpus, idx, None, cfg)
    m = evaluate(model, corpus, idx, None, cfg, k=10, columns=FEATURE_NAMES)
    for key in (
        "recall@10",
        "mrr@10",
        "ceiling",
        "recall@10|in_pool",
        "mrr@10|in_pool",
        "n_queries",
        "n_in_pool",
    ):
        assert key in m
    assert 0.0 <= m["ceiling"] <= 1.0
    assert m["recall@10|in_pool"] >= m["recall@10"] - 1e-9
    assert m["mrr@10|in_pool"] >= m["mrr@10"] - 1e-9


def test_evaluate_test_only_filter():
    corpus = [
        Recipe(
            recipe_id=str(i),
            title=str(i),
            canonical=["egg", "flour", "milk", "sugar", "butter", "salt"][: 4 + (i % 3)],
        )
        for i in range(60)
    ]
    idx = InvertedIndex.build(corpus)
    cfg = RecConfig(seed=2)
    model, _ = train_ranker(LinearRanker(seed=0), corpus, idx, None, cfg)
    test_split = evaluate(model, corpus, idx, None, cfg, k=10)
    all_recipes = evaluate(model, corpus, idx, None, cfg, k=10, test_only=False)
    assert all_recipes["n_queries"] >= test_split["n_queries"]


def test_evaluate_respects_max_eval_queries():
    corpus = [
        Recipe(
            recipe_id=str(i),
            title=str(i),
            canonical=["egg", "flour", "milk", "sugar", "butter", "salt"][: 4 + (i % 3)],
        )
        for i in range(60)
    ]
    idx = InvertedIndex.build(corpus)
    cfg = RecConfig(seed=2, max_eval_queries=5)
    model, _ = train_ranker(LinearRanker(seed=0), corpus, idx, None, cfg)
    m = evaluate(model, corpus, idx, None, cfg, k=10)
    assert m["n_queries"] <= 5
