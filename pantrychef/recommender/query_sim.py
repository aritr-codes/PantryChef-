"""Leave-ingredients-out query construction + deterministic train/test split.

A query is built by hiding a fraction of a real recipe's canonical ingredients;
the masked recipe is the single gold-relevant target (recovery task). The split
is by recipe_id hash so a recipe is never in both train and test.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from pantrychef.common.types import Recipe
from pantrychef.recommender.config import RecConfig


@dataclass(frozen=True)
class QuerySim:
    pantry: tuple[str, ...]  # kept ingredients (the simulated pantry)
    hidden: tuple[str, ...]  # masked-out ingredients
    gold_id: str  # recipe_id of the recipe these came from


def is_train(recipe_id: str, train_frac: int = 80) -> bool:
    """Deterministic ~80/20 split by recipe_id hash. No global RNG state."""
    h = int(hashlib.md5(recipe_id.encode(), usedforsecurity=False).hexdigest(), 16)
    return (h % 100) < train_frac


def make_query(recipe: Recipe, cfg: RecConfig, rng: random.Random) -> QuerySim | None:
    """Mask cfg.mask_fraction of a recipe's unique canonical ingredients.

    Returns None for recipes shorter than cfg.min_recipe_len. At least one
    ingredient is always hidden and at least one always kept.
    """
    canon = sorted(set(recipe.canonical))  # dedupe + deterministic order
    if len(canon) < cfg.min_recipe_len:
        return None
    n_hidden = max(1, round(cfg.mask_fraction * len(canon)))
    n_hidden = min(n_hidden, len(canon) - 1)  # always keep >=1
    hidden = sorted(rng.sample(canon, n_hidden))
    pantry = sorted(set(canon) - set(hidden))
    return QuerySim(pantry=tuple(pantry), hidden=tuple(hidden), gold_id=recipe.recipe_id)
