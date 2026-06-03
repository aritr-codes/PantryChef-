from pantrychef.dietary import DietTagger
from pantrychef.eval.nutrition_eval import coverage_report, dietary_accuracy
from pantrychef.nutrition.match import IngredientMatcher

TABLE = {
    100: {"description": "Butter", "per100g": {"kcal": 717.0}, "unit_grams": {"cup": 227.0}},
    200: {"description": "Egg", "per100g": {"kcal": 143.0}, "unit_grams": {"each": 50.0}},
}


def test_coverage_report_counts():
    m = IngredientMatcher(TABLE, aliases={"butter": 100, "egg": 200})
    recipes = [["1 cup butter", "2 eggs"], ["1 package butter"]]
    rep = coverage_report(recipes, ["butter", "egg"], None, m)
    assert rep["n_recipes"] == 2
    assert rep["match_coverage"] == 1.0  # all 3 lines matched
    assert 0.0 < rep["mass_coverage"] < 1.0  # the "package" line is unresolved


def test_dietary_accuracy_perfect_on_labeled():
    t = DietTagger(known=["butter", "eggplant"])
    labels = {"butter": ["dairy", "animal_product"], "eggplant": []}
    acc = dietary_accuracy(t, labels)
    assert acc["accuracy"] == 1.0
