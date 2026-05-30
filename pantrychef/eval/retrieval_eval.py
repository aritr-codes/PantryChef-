"""Retrieval evaluation: leave-one-ingredient-out recall@k.

For each recipe with >=2 canonical ingredients, drop one at random and query
the index with the rest; count a hit if the source recipe is in the top-k.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from pantrychef.common.types import Recipe
from pantrychef.retrieval.baseline import recommend
from pantrychef.retrieval.index import InvertedIndex


def evaluate_retrieval(
    index: InvertedIndex, recipes: Sequence[Recipe], k: int = 10, seed: int = 42
) -> dict[str, float]:
    rng = random.Random(seed)
    hits = total = 0
    for r in recipes:
        # Dedup so leave-one-out holds out one ingredient type, independent of
        # clean.py's dedup (order preserved).
        canon = list(dict.fromkeys(r.canonical))
        if len(canon) < 2:
            continue
        held = rng.choice(canon)
        pantry = [c for c in canon if c != held]
        ranked = recommend(index, pantry, k=k)
        total += 1
        if any(s.recipe_id == r.recipe_id for s in ranked):
            hits += 1
    return {
        "recall_at_k": hits / total if total else 0.0,
        "k": float(k),
        "n": float(total),
    }
