"""Raw recipe schema (pre-cleaning), mirroring RecipeNLG columns."""

from __future__ import annotations

from pydantic import BaseModel


class RawRecipe(BaseModel):
    title: str = ""
    ingredients: list[str] = []
    directions: list[str] = []
    ner: list[str] = []
    link: str = ""
    source: str = ""
