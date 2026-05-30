"""Set-overlap retrieval baseline.

Score = coverage = |pantry ∩ recipe| / |recipe ingredients|. Rank by score,
then fewer missing ingredients, then recipe_id (deterministic).
"""

from __future__ import annotations

from collections.abc import Iterable

from pantrychef.common.types import ScoredRecipe
from pantrychef.ingredients.normalize import canonicalize
from pantrychef.retrieval.index import InvertedIndex


def recommend(index: InvertedIndex, pantry: Iterable[str], k: int = 5) -> list[ScoredRecipe]:
    pantry_set = {c for c in (canonicalize(p) for p in pantry) if c}
    results: list[ScoredRecipe] = []
    for rid in index.candidates(pantry_set):
        recipe = index.recipes[rid]
        canon = set(recipe.canonical)
        if not canon:
            continue
        matched = sorted(pantry_set & canon)
        missing = sorted(canon - pantry_set)
        results.append(
            ScoredRecipe(
                recipe_id=rid,
                score=len(matched) / len(canon),
                title=recipe.title,
                matched=matched,
                missing=missing,
            )
        )
    results.sort(key=lambda r: (-r.score, len(r.missing), r.recipe_id))
    return results[:k]
