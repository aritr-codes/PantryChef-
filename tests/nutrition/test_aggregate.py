import math

from pantrychef.nutrition.aggregate import aggregate
from pantrychef.nutrition.match import IngredientMatcher

TABLE = {
    100: {
        "description": "Butter",
        "per100g": {"kcal": 717.0, "fat_g": 81.0},
        "unit_grams": {"cup": 227.0},
    },
    200: {
        "description": "Egg",
        "per100g": {"kcal": 143.0, "protein_g": 12.5},
        "unit_grams": {"each": 50.0},
    },
}
VOCAB = ["butter", "egg"]
ALIASES = {"butter": 100, "egg": 200}


def test_aggregate_totals_and_per100g():
    m = IngredientMatcher(TABLE, aliases=ALIASES)
    res = aggregate(["1 cup butter", "2 eggs"], VOCAB, None, m)
    assert res.n_lines == 2 and res.n_matched == 2 and res.n_massed == 2
    assert math.isclose(res.total_grams, 327.0)
    # kcal total = 717*2.27 + 143*1.0 = 1770.59
    assert math.isclose(res.facts_total.calories, 1770.59, rel_tol=1e-4)
    assert math.isclose(res.facts_per100g.calories, 1770.59 / 327 * 100, rel_tol=1e-4)


def test_unresolved_line_excluded_and_counted():
    m = IngredientMatcher(TABLE, aliases=ALIASES)
    res = aggregate(["1 package butter", "2 eggs"], VOCAB, None, m)  # package unresolved
    assert res.n_matched == 2 and res.n_massed == 1
    assert math.isclose(res.total_grams, 100.0)  # only the eggs


def test_empty_lines_no_facts_and_zero_coverage():
    m = IngredientMatcher(TABLE, aliases=ALIASES)
    res = aggregate([], VOCAB, None, m)
    assert res.n_lines == 0 and res.total_grams == 0.0
    assert res.facts_total.calories is None  # unknown, not zero
    assert res.facts_per100g.calories is None
    assert res.match_coverage == 0.0 and res.mass_coverage == 0.0


def test_missing_nutrient_is_none_not_zero():
    # Butter table entry has no protein_g/carbs_g -> those stay None, not 0.0
    m = IngredientMatcher(TABLE, aliases=ALIASES)
    res = aggregate(["1 cup butter"], VOCAB, None, m)
    assert res.facts_total.calories is not None  # kcal present
    assert res.facts_total.protein_g is None  # absent in source -> unknown
    assert res.facts_total.carbs_g is None
