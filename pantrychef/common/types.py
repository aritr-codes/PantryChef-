"""Shared data contracts between modules.

These are the interfaces the ARCHITECTURE.md data-flow describes. Phase 0 ships
the skeletons; fields get filled/validated as each phase lands. Keeping them
here (not inside each module) makes the boundaries explicit and testable.
"""

from __future__ import annotations

from pydantic import BaseModel


class ParsedIngredient(BaseModel):
    """Output of `ingredients.parse` (Phase 1)."""

    raw: str
    canonical: str | None = None
    quantity: float | None = None
    unit: str | None = None
    modifier: str | None = None


class Substitute(BaseModel):
    """A candidate substitution (Phase 2)."""

    ingredient: str
    score: float
    dietary_valid: bool = True


class Recipe(BaseModel):
    """A cleaned recipe with a canonical ingredient set (Phase 1)."""

    recipe_id: str
    title: str
    ingredients_raw: list[str] = []
    canonical: list[str] = []


class ScoredRecipe(BaseModel):
    """A ranked recipe result (Phase 1/3)."""

    recipe_id: str
    score: float
    title: str = ""
    matched: list[str] = []
    missing: list[str] = []


class NutritionFacts(BaseModel):
    """Per-serving nutrition (Phase 4)."""

    calories: float | None = None
    protein_g: float | None = None
    carbs_g: float | None = None
    fat_g: float | None = None
    micros: dict[str, float] = {}
