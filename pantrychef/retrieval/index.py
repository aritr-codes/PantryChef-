"""Inverted index: canonical ingredient -> set of recipe ids."""

from __future__ import annotations

from collections.abc import Iterable

from pantrychef.common.types import Recipe


class InvertedIndex:
    def __init__(self) -> None:
        self.postings: dict[str, set[str]] = {}
        self.recipes: dict[str, Recipe] = {}
        # Canonical ingredient sets, precomputed once per recipe. The reranker
        # scores a recipe across many queries; rebuilding set(recipe.canonical)
        # each time dominates candidate-pool cost, so cache it here.
        self.canon_sets: dict[str, frozenset[str]] = {}

    def add(self, recipe: Recipe) -> None:
        self.recipes[recipe.recipe_id] = recipe
        canon = frozenset(recipe.canonical)
        self.canon_sets[recipe.recipe_id] = canon
        for ing in canon:
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
