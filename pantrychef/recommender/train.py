"""Build LightGBM-style (X, y, groups) training data from masked queries.

Only train-split recipes are used. Queries whose candidate pool is empty or
lacks the gold recipe are dropped (counted) — they carry no positive label.
"""

from __future__ import annotations

import random

import numpy as np

from pantrychef.common import get_logger
from pantrychef.common.types import Recipe
from pantrychef.recommender.config import RecConfig
from pantrychef.recommender.features import FEATURE_NAMES, extract_features, to_matrix
from pantrychef.recommender.query_sim import is_train, make_query
from pantrychef.recommender.recommend import candidate_pool

log = get_logger(__name__)


def build_dataset(
    recipes: list[Recipe],
    index,
    sub_lookup,
    cfg: RecConfig,
    columns: tuple[str, ...] = FEATURE_NAMES,
    train_only: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[int], dict[str, int]]:
    """Return (X, y, group_sizes, stats) over masked train-split queries."""
    rng = random.Random(cfg.seed)
    feats: list[dict[str, float]] = []
    labels: list[int] = []
    groups: list[int] = []
    n_total = n_no_pool = n_no_gold = 0

    for recipe in recipes:
        if train_only and not is_train(recipe.recipe_id):
            continue
        for _ in range(cfg.queries_per_recipe):
            q = make_query(recipe, cfg, rng)
            if q is None:
                continue
            n_total += 1
            pool = candidate_pool(index, set(q.pantry), cfg.candidate_cap)
            if not pool:
                n_no_pool += 1
                continue
            ids = [r.recipe_id for r in pool]
            if q.gold_id not in ids:
                n_no_gold += 1
                continue
            for r in pool:
                feats.append(extract_features(set(q.pantry), r, sub_lookup))
                labels.append(1 if r.recipe_id == q.gold_id else 0)
            groups.append(len(pool))

    stats = {
        "n_queries_total": n_total,
        "n_queries_kept": len(groups),
        "n_dropped_empty_pool": n_no_pool,
        "n_dropped_no_gold": n_no_gold,
    }
    log.info("build_dataset: %s", stats)
    X = to_matrix(feats, columns)
    return X, np.array(labels, dtype=int), groups, stats


def train_ranker(model, recipes, index, sub_lookup, cfg, columns=FEATURE_NAMES):
    """Build the dataset and fit `model` in place; returns (model, stats)."""
    X, y, groups, stats = build_dataset(recipes, index, sub_lookup, cfg, columns)
    model.fit(X, y, groups)
    return model, stats
