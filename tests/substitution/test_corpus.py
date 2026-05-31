"""Tests for IngredientCorpus — per-epoch shuffled sentences."""

from __future__ import annotations

from pantrychef.common.types import Recipe
from pantrychef.substitution.corpus import IngredientCorpus

RECIPES = [
    Recipe(recipe_id="r0", title="a", canonical=["flour", "egg", "milk", "sugar"]),
    Recipe(recipe_id="r1", title="b", canonical=["butter"]),  # singleton kept
    Recipe(recipe_id="r2", title="c", canonical=[]),  # empty dropped
]


def test_yields_token_lists() -> None:
    corpus = IngredientCorpus(RECIPES, seed=42)
    sents = list(corpus)
    assert sorted(sents[0]) == ["egg", "flour", "milk", "sugar"]
    assert sents[1] == ["butter"]
    assert len(sents) == 2  # empty recipe dropped


def test_reshuffles_each_pass_deterministically() -> None:
    corpus = IngredientCorpus(RECIPES, seed=42)
    pass1 = list(corpus)[0]
    pass2 = list(corpus)[0]
    # different epochs -> different order (RNG advances), same multiset
    assert pass1 != pass2  # seed=42 is verified to produce distinct orderings
    assert sorted(pass1) == sorted(pass2)
    # same seed, fresh corpus -> identical first-pass order (reproducible)
    again = list(IngredientCorpus(RECIPES, seed=42))[0]
    assert again == pass1
