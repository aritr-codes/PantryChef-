"""Candidate generation (shared pool) + learned rerank.

candidate_pool: all recipes sharing >=1 pantry ingredient, ordered by coverage
(the P1 baseline order), capped. Both the overlap baseline and every learned
model rank this SAME pool, so the comparison isolates the ordering function.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np

from pantrychef.common.types import Recipe, ScoredRecipe
from pantrychef.recommender.features import FEATURE_NAMES, to_matrix
from pantrychef.retrieval.index import InvertedIndex

FeatureFn = Callable[[set[str], Recipe], dict[str, float]]


class _Scorer(Protocol):
    """Anything with a score(X) -> ndarray method (LinearRanker, LambdaMARTRanker, OverlapModel)."""

    def score(self, X: np.ndarray) -> np.ndarray: ...


def candidate_pool(index: InvertedIndex, pantry: set[str], cap: int) -> list[Recipe]:
    """Overlap candidates ordered by coverage desc, then fewer-missing, then id.

    Vectorized over the index's row-int columnar view: concatenating the pantry
    ingredients' postings rows and ``np.bincount``-ing them yields, per candidate
    row, the match count = |pantry ∩ recipe.canonical| (each ingredient's
    postings array holds a row at most once). Coverage and missing follow by
    array arithmetic with ``canon_len_by_row``.

    The ranking key matches baseline.recommend exactly — coverage desc, fewer
    missing, recipe_id ascending-as-string — so the pool stays the P1 baseline
    order and reranker-vs-baseline remains apples-to-apples. The string tie-break
    is reproduced via ``id_rank_by_row`` (precomputed string-sort rank), and the
    ``[:cap]`` slice mirrors the reference list slice (incl. negative cap). This
    is byte-identical to the previous dict-based scoring but ~50x faster at the
    full-corpus scale where common ingredients touch hundreds of thousands of
    postings (see tests/recommender/test_recommend.py::
    test_candidate_pool_matches_reference_randomized).
    """
    arrs = [index.postings_rows[ing] for ing in pantry if ing in index.postings_rows]
    if not arrs:
        return []
    counts = np.bincount(np.concatenate(arrs), minlength=index.n)
    rows = np.flatnonzero(counts)
    matched = counts[rows]
    canon_len = index.canon_len_by_row[rows]
    keep = canon_len > 0
    rows, matched, canon_len = rows[keep], matched[keep], canon_len[keep]
    coverage = matched / canon_len
    missing = canon_len - matched
    # lexsort: last key is primary -> (-coverage) primary, missing, then id_rank.
    order = np.lexsort((index.id_rank_by_row[rows], missing, -coverage))
    return [index.recipe_by_row[r] for r in rows[order[:cap]]]


def rerank(
    index: InvertedIndex,
    pantry: set[str],
    model: _Scorer,
    feature_fn: FeatureFn,
    k: int = 10,
    cap: int = 200,
    columns: tuple[str, ...] = FEATURE_NAMES,
    pool: list[Recipe] | None = None,
) -> list[ScoredRecipe]:
    """Rerank the shared candidate pool by model score (tie-break recipe_id)."""
    if k <= 0 or cap <= 0:
        return []
    pool = pool if pool is not None else candidate_pool(index, pantry, cap)
    if not pool:
        return []
    feats = [feature_fn(pantry, r) for r in pool]
    scores = model.score(to_matrix(feats, columns))
    order = sorted(range(len(pool)), key=lambda i: (-float(scores[i]), pool[i].recipe_id))
    out: list[ScoredRecipe] = []
    for i in order[:k]:
        r = pool[i]
        canon = index.canon_sets[r.recipe_id]
        out.append(
            ScoredRecipe(
                recipe_id=r.recipe_id,
                score=float(scores[i]),
                title=r.title,
                matched=sorted(pantry & canon),
                missing=sorted(canon - pantry),
            )
        )
    return out
