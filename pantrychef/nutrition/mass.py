"""Resolve (quantity, unit, ingredient, USDA food) -> grams, coverage-honest.

Mass units convert by fixed factor. Volume units use the food's USDA portion
gram-weight when present, else a per-ingredient-class density fallback. Count /
portion units use the food's "each" portion or a curated per-item table.
Anything unresolved returns (None, False) and is reported as a coverage gap."""

from __future__ import annotations

MASS_TO_G: dict[str, float] = {"g": 1.0, "kg": 1000.0, "oz": 28.3495, "lb": 453.592}
VOL_TO_ML: dict[str, float] = {
    "ml": 1.0,
    "l": 1000.0,
    "cup": 236.588,
    "tbsp": 14.787,
    "tsp": 4.929,
    "quart": 946.353,
    "pint": 473.176,
}
TINY_G: dict[str, float] = {"pinch": 0.36, "dash": 0.6}
_PORTION_UNITS = {"clove", "can", "slice", "stick"}

# density g/ml by ingredient keyword (token match on canonical).
_DENSITY: dict[str, float] = {
    "water": 1.0,
    "milk": 1.03,
    "juice": 1.05,
    "broth": 1.0,
    "stock": 1.0,
    "wine": 0.99,
    "vinegar": 1.01,
    "oil": 0.92,
    "honey": 1.42,
    "syrup": 1.37,
    "flour": 0.53,
    "sugar": 0.85,
    "salt": 1.22,
    "rice": 0.85,
    "butter": 0.96,
    "cream": 1.0,
    "sauce": 1.05,
}
# per-item grams by ingredient keyword (for count/None units when USDA lacks it).
_PORTION_G: dict[str, float] = {
    "egg": 50.0,
    "clove": 3.0,
    "onion": 110.0,
    "tomato": 123.0,
    "apple": 182.0,
    "banana": 118.0,
    "carrot": 61.0,
    "potato": 213.0,
    "lemon": 58.0,
}


def _keyword(canonical: str, table: dict[str, float]) -> float | None:
    for w in canonical.split():
        if w in table:
            return table[w]
    return None


def to_grams(
    qty: float | None, unit: str | None, canonical: str, food: dict
) -> tuple[float | None, bool]:
    unit_grams = food.get("unit_grams", {})
    if unit in TINY_G:
        return (qty or 1.0) * TINY_G[unit], True
    if qty is None:
        return None, False
    if unit in MASS_TO_G:
        return qty * MASS_TO_G[unit], True
    if unit in VOL_TO_ML:
        if unit in unit_grams:
            return qty * unit_grams[unit], True
        density = _keyword(canonical, _DENSITY)
        if density is not None:
            return qty * VOL_TO_ML[unit] * density, True
        return None, False
    if unit is None or unit in _PORTION_UNITS:
        if unit in unit_grams:
            return qty * unit_grams[unit], True
        if "each" in unit_grams:
            return qty * unit_grams["each"], True
        per_item = _keyword(canonical, _PORTION_G)
        if per_item is not None:
            return qty * per_item, True
        return None, False
    return None, False  # e.g. "package" — variable, unresolved
