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
