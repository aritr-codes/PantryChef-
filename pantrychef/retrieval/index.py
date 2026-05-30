"""Inverted index: canonical ingredient -> set of recipe ids."""

from __future__ import annotations

from collections.abc import Iterable

from pantrychef.common.types import Recipe


class InvertedIndex:
    def __init__(self) -> None:
        self.postings: dict[str, set[str]] = {}
        self.recipes: dict[str, Recipe] = {}

    def add(self, recipe: Recipe) -> None:
        self.recipes[recipe.recipe_id] = recipe
        for ing in recipe.canonical:
            self.postings.setdefault(ing, set()).add(recipe.recipe_id)

    @classmethod
    def build(cls, recipes: Iterable[Recipe]) -> InvertedIndex:
        idx = cls()
        for r in recipes:
            idx.add(r)
        return idx

    def candidates(self, pantry: Iterable[str]) -> set[str]:
        out: set[str] = set()
        for ing in pantry:
            out |= self.postings.get(ing, set())
        return out
