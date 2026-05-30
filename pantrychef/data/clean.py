"""Turn RawRecipe records into cleaned Recipe objects with canonical sets.

Dedup is by normalized title (RecipeNLG is largely pre-deduplicated; this is a
cheap safeguard). recipe_id is a stable sequential id assigned at clean time.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from pantrychef.common.types import Recipe
from pantrychef.data.schemas import RawRecipe
from pantrychef.ingredients.parser import parse


def to_recipe(raw: RawRecipe, recipe_id: str, vocab: list[str]) -> Recipe:
    canonical: list[str] = []
    seen: set[str] = set()
    for line in raw.ingredients:
        pi = parse(line, vocab)
        if pi.canonical and pi.canonical not in seen:
            seen.add(pi.canonical)
            canonical.append(pi.canonical)
    return Recipe(
        recipe_id=recipe_id,
        title=raw.title,
        ingredients_raw=list(raw.ingredients),
        canonical=canonical,
    )


def clean_recipes(raws: Iterable[RawRecipe], vocab: list[str]) -> Iterator[Recipe]:
    seen_titles: set[str] = set()
    idx = 0
    for raw in raws:
        key = raw.title.strip().lower()
        if not key or key in seen_titles:
            continue
        seen_titles.add(key)
        yield to_recipe(raw, f"r{idx}", vocab)
        idx += 1
