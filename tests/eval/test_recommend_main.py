import numpy as np

from pantrychef.common.types import Recipe
from pantrychef.eval.recommend_main import OverlapModel, run_leaderboard
from pantrychef.recommender.config import RecConfig
from pantrychef.retrieval.index import InvertedIndex


def test_overlap_model_preserves_pool_order():
    m = OverlapModel()
    s = m.score(np.zeros((3, 2)))
    assert s[0] > s[1] > s[2]


def test_run_leaderboard_smoke():
    corpus = [
        Recipe(
            recipe_id=str(i),
            title=str(i),
            canonical=["egg", "flour", "milk", "sugar", "butter", "salt", "oil"][: 4 + (i % 4)],
        )
        for i in range(80)
    ]
    idx = InvertedIndex.build(corpus)
    rows = run_leaderboard(
        corpus, idx, sub_lookup=None, cfg=RecConfig(seed=3), use_lambdamart=False
    )
    names = {r["model"] for r in rows}
    assert "overlap" in names and "linear" in names
    for r in rows:
        assert "recall@10" in r and "mrr@10" in r
