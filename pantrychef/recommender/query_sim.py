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


def sample_order(recipes: list[Recipe], seed: int) -> list[Recipe]:
    """Deterministic recipe order for capped ("first N") query sampling.

    Ordering by md5(seed:recipe_id) decouples capped runs from corpus file
    order (a raw prefix of the corpus is not a representative sample) and
    makes any smaller cap a true subset of a larger one under the same seed
    (the 10k sample is a prefix of the 20k sample), so learning-curve points
    nest instead of being disjoint slices.
    """

    def key(recipe: Recipe) -> str:
        return hashlib.md5(f"{seed}:{recipe.recipe_id}".encode(), usedforsecurity=False).hexdigest()

    return sorted(recipes, key=key)


def make_query(recipe: Recipe, cfg: RecConfig, rng: random.Random) -> QuerySim | None:
    """Mask cfg.mask_fraction of a recipe's unique canonical ingredients.

    Returns None for recipes shorter than cfg.min_recipe_len. At least one
    ingredient is always hidden and at least one always kept. The hidden count
    is round(mask_fraction * n) (Python round-half-to-even), floored at 1 and
    capped at n-1.
    """
    canon = sorted(set(recipe.canonical))  # dedupe + deterministic order
    if len(canon) < cfg.min_recipe_len:
        return None
    n_hidden = max(1, round(cfg.mask_fraction * len(canon)))
    n_hidden = min(n_hidden, len(canon) - 1)  # always keep >=1
    assert n_hidden >= 1, "n_hidden must stay >=1; check min_recipe_len"
    hidden = sorted(rng.sample(canon, n_hidden))
    pantry = sorted(set(canon) - set(hidden))
    return QuerySim(pantry=tuple(pantry), hidden=tuple(hidden), gold_id=recipe.recipe_id)
