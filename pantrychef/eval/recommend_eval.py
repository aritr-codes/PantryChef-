"""Recovery-task metrics: recall@k, MRR@k, candidate ceiling, and the same
metrics conditional on the gold recipe being reachable in the candidate pool.

NDCG@k is intentionally omitted: with a single gold per query it is a monotone
transform of MRR (see design spec 2026-06-02) and would not be independent
evidence.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

from pantrychef.common import get_logger
from pantrychef.common.types import Recipe
from pantrychef.recommender.config import RecConfig
from pantrychef.recommender.features import FEATURE_NAMES, SubLookup, extract_features
from pantrychef.recommender.query_sim import is_train, make_query
from pantrychef.recommender.recommend import _Scorer, candidate_pool, rerank
from pantrychef.retrieval.index import InvertedIndex

log = get_logger(__name__)


def recall_at_k(ranked_ids: list[str], gold_id: str, k: int) -> float:
    if k <= 0:
        return 0.0
    return 1.0 if gold_id in ranked_ids[:k] else 0.0


def mrr_at_k(ranked_ids: list[str], gold_id: str, k: int) -> float:
    if k <= 0:
        return 0.0
    for rank, rid in enumerate(ranked_ids[:k], 1):
        if rid == gold_id:
            return 1.0 / rank
    return 0.0


def evaluate(
    model: _Scorer,
    recipes: Iterable[Recipe],
    index: InvertedIndex,
    sub_lookup: SubLookup,
    cfg: RecConfig,
    k: int = 10,
    columns: tuple[str, ...] = FEATURE_NAMES,
    test_only: bool = True,
) -> dict[str, float]:
    """Evaluate the recovery task over test-split masked queries.

    Returns overall recall@k / mrr@k, the candidate-recall ceiling (fraction of
    queries where gold is in the pool at all), and recall|in_pool / mrr|in_pool
    conditioned on that subset.
    """
    rng = random.Random(cfg.seed + 1)  # distinct from the train mask stream
    n = n_in_pool = 0
    recall = mrr = recall_ip = mrr_ip = 0.0

    def feat_fn(p, r):
        return extract_features(p, r, sub_lookup)

    for recipe in recipes:
        if test_only and is_train(recipe.recipe_id):
            continue
        q = make_query(recipe, cfg, rng)
        if q is None:
            continue
        n += 1
        pantry = set(q.pantry)
        pool = candidate_pool(index, pantry, cfg.candidate_cap)
        in_pool = q.gold_id in {r.recipe_id for r in pool}
        ranked = rerank(
            index, pantry, model, feat_fn, k=k, cap=cfg.candidate_cap, columns=columns, pool=pool
        )
        ids = [r.recipe_id for r in ranked]
        r_at = recall_at_k(ids, q.gold_id, k)
        m_at = mrr_at_k(ids, q.gold_id, k)
        recall += r_at
        mrr += m_at
        if in_pool:
            n_in_pool += 1
            recall_ip += r_at
            mrr_ip += m_at

    out = {
        f"recall@{k}": recall / n if n else 0.0,
        f"mrr@{k}": mrr / n if n else 0.0,
        "ceiling": n_in_pool / n if n else 0.0,
        f"recall@{k}|in_pool": recall_ip / n_in_pool if n_in_pool else 0.0,
        f"mrr@{k}|in_pool": mrr_ip / n_in_pool if n_in_pool else 0.0,
        "n_queries": float(n),
        "n_in_pool": float(n_in_pool),
    }
    log.info("evaluate: %s", {key: round(v, 4) for key, v in out.items()})
    return out
