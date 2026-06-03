"""Coverage + dietary-class accuracy metrics for Phase 4.

No recipe-level nutrition gold exists, so this measures coverage (match + mass)
and dietary-class accuracy vs a hand-labeled set — NOT recipe-macro accuracy."""

from __future__ import annotations

from statistics import median

from pantrychef.dietary import DietTagger
from pantrychef.ingredients.parser import MatchIndex
from pantrychef.nutrition.aggregate import aggregate
from pantrychef.nutrition.match import IngredientMatcher


def coverage_report(
    recipes: list[list[str]],
    vocab: list[str] | None,
    index: MatchIndex | None,
    matcher: IngredientMatcher,
    usable_mass_fraction: float = 0.50,
) -> dict:
    n_lines = n_matched = n_massed = usable = 0
    unresolved_fracs: list[float] = []
    for raw_lines in recipes:
        res = aggregate(raw_lines, vocab, index, matcher)
        n_lines += res.n_lines
        n_matched += res.n_matched
        n_massed += res.n_massed
        if res.mass_coverage >= usable_mass_fraction:
            usable += 1
        unresolved_fracs.append(1.0 - res.mass_coverage)
    return {
        "n_recipes": len(recipes),
        "n_lines": n_lines,
        "match_coverage": n_matched / n_lines if n_lines else 0.0,
        "mass_coverage": n_massed / n_lines if n_lines else 0.0,
        "nutrition_completeness": usable / len(recipes) if recipes else 0.0,
        "median_unresolved_mass": median(unresolved_fracs) if unresolved_fracs else 0.0,
    }


def dietary_accuracy(tagger: DietTagger, labels: dict[str, list[str]]) -> dict:
    correct = 0
    for ing, gold in labels.items():
        if tagger.tags(ing) == set(gold):
            correct += 1
    return {
        "accuracy": correct / len(labels) if labels else 0.0,
        "n": len(labels),
    }
