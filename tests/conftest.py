"""Shared offline test corpus — mirrors the RecipeNLG schema in-code so tests
need no dataset download. `ner` plays the role of RecipeNLG's NER column
(ground-truth canonical entities)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pantrychef.data.schemas import RawRecipe


@pytest.fixture
def sample_raws() -> list[RawRecipe]:
    # Imported lazily: pantrychef.data.schemas.RawRecipe is created in Task 2.
    from pantrychef.data.schemas import RawRecipe

    return [
        RawRecipe(
            title="Pancakes",
            ingredients=["2 cups flour", "1 egg", "1 cup milk", "2 tbsp sugar"],
            ner=["flour", "egg", "milk", "sugar"],
        ),
        RawRecipe(
            title="Omelette",
            ingredients=["3 eggs", "1/2 cup milk", "1 pinch salt"],
            ner=["egg", "milk", "salt"],
        ),
        RawRecipe(
            title="Sugar Cookies",
            ingredients=["2 cups flour", "1 cup sugar", "2 eggs", "1 cup butter"],
            ner=["flour", "sugar", "egg", "butter"],
        ),
        RawRecipe(
            title="Tomato Salad",
            ingredients=["2 tomatoes", "1 cucumber"],
            ner=["tomato", "cucumber"],
        ),
    ]
