"""Aggregate raw recipe ingredient lines into per-recipe + per-100g nutrition.

Re-parses each raw line (the cleaned recipe store keeps only canonical sets, no
quantities), matches it to USDA, resolves grams, and sums whitelisted nutrients.
Unmatched or unresolved lines are excluded from totals and counted toward the
reported coverage gap — totals are estimates, never claimed as ground truth."""

from __future__ import annotations

from dataclasses import dataclass

from pantrychef.common.types import NutritionFacts
from pantrychef.ingredients.parser import MatchIndex, parse
from pantrychef.nutrition.mass import to_grams
from pantrychef.nutrition.match import IngredientMatcher

# nutrient keys that map onto NutritionFacts' named fields; the rest go to micros.
_NAMED = {"kcal": "calories", "protein_g": "protein_g", "carbs_g": "carbs_g", "fat_g": "fat_g"}


@dataclass
class RecipeNutrition:
    facts_total: NutritionFacts
    facts_per100g: NutritionFacts
    total_grams: float
    n_lines: int
    n_matched: int
    n_massed: int

    @property
    def match_coverage(self) -> float:
        return self.n_matched / self.n_lines if self.n_lines else 0.0

    @property
    def mass_coverage(self) -> float:
        return self.n_massed / self.n_lines if self.n_lines else 0.0


def _facts_from(totals: dict[str, float]) -> NutritionFacts:
    named = {field_: totals.get(key, 0.0) for key, field_ in _NAMED.items()}
    micros = {k: v for k, v in totals.items() if k not in _NAMED}
    return NutritionFacts(**named, micros=micros)


def aggregate(
    raw_lines: list[str],
    vocab: list[str] | None,
    index: MatchIndex | None,
    matcher: IngredientMatcher,
) -> RecipeNutrition:
    totals: dict[str, float] = {}
    total_grams = 0.0
    n_matched = n_massed = 0
    for line in raw_lines:
        pi = parse(line, vocab, index)
        if pi.canonical is None:
            continue
        res = matcher.match(pi.canonical)
        if res.fdc_id is None:
            continue
        n_matched += 1
        food = matcher.table[res.fdc_id]
        grams, ok = to_grams(pi.quantity, pi.unit, pi.canonical, food)
        if not ok or grams is None:
            continue
        n_massed += 1
        total_grams += grams
        for key, per100 in food["per100g"].items():
            totals[key] = totals.get(key, 0.0) + per100 * grams / 100.0

    facts_total = _facts_from(totals)
    scaled = {k: v / total_grams * 100.0 for k, v in totals.items()} if total_grams > 0 else {}
    return RecipeNutrition(
        facts_total=facts_total,
        facts_per100g=_facts_from(scaled),
        total_grams=total_grams,
        n_lines=len(raw_lines),
        n_matched=n_matched,
        n_massed=n_massed,
    )
