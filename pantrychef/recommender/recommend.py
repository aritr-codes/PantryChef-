"""Candidate generation (shared pool) + learned rerank.

candidate_pool: all recipes sharing >=1 pantry ingredient, ordered by coverage
(the P1 baseline order), capped. Both the overlap baseline and every learned
model rank this SAME pool, so the comparison isolates the ordering function.
"""

from __future__ import annotations

from collections.abc import Callable

from pantrychef.common.types import Recipe, ScoredRecipe
from pantrychef.recommender.features import FEATURE_NAMES, to_matrix
from pantrychef.retrieval.index import InvertedIndex

FeatureFn = Callable[[set, Recipe], dict]


def candidate_pool(index: InvertedIndex, pantry: set[str], cap: int) -> list[Recipe]:
    """Overlap candidates ordered by coverage desc, then fewer-missing, then id."""
    scored: list[tuple[float, int, str, Recipe]] = []
    for rid in index.candidates(pantry):
        recipe = index.recipes[rid]
        canon = set(recipe.canonical)
        if not canon:
            continue
        matched = len(pantry & canon)
        coverage = matched / len(canon)
        scored.append((coverage, len(canon - pantry), rid, recipe))
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [t[3] for t in scored[:cap]]


def rerank(
    index: InvertedIndex,
    pantry: set[str],
    model,
    feature_fn: FeatureFn,
    k: int = 10,
    cap: int = 200,
    columns: tuple[str, ...] = FEATURE_NAMES,
) -> list[ScoredRecipe]:
    """Rerank the shared candidate pool by model score (tie-break recipe_id)."""
    pool = candidate_pool(index, pantry, cap)
    if not pool:
        return []
    feats = [feature_fn(pantry, r) for r in pool]
    scores = model.score(to_matrix(feats, columns))
    order = sorted(range(len(pool)), key=lambda i: (-float(scores[i]), pool[i].recipe_id))
    out: list[ScoredRecipe] = []
    for i in order[:k]:
        r = pool[i]
        canon = set(r.canonical)
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
