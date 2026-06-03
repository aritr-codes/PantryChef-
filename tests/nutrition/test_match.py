from pantrychef.nutrition.match import IngredientMatcher

TABLE = {
    100: {"description": "Butter, salted", "per100g": {}, "unit_grams": {}},
    200: {"description": "Egg, whole, raw", "per100g": {}, "unit_grams": {}},
    300: {"description": "Onions, raw", "per100g": {}, "unit_grams": {}},
}


def test_alias_tier_wins():
    m = IngredientMatcher(TABLE, aliases={"butter": 100})
    res = m.match("butter")
    assert res.fdc_id == 100 and res.method == "alias"


def test_exact_normalized_match():
    m = IngredientMatcher(TABLE)
    res = m.match("onion")  # canonical("Onions, raw") before-comma -> "onion"
    assert res.fdc_id == 300 and res.method == "exact"


def test_jaccard_fallback():
    m = IngredientMatcher(TABLE, jaccard_threshold=0.3)
    res = m.match("whole egg")  # tokens {whole, egg} vs {egg, whole, raw}
    assert res.fdc_id == 200 and res.method == "jaccard"


def test_unmatched_returns_none():
    m = IngredientMatcher(TABLE, jaccard_threshold=0.9)
    res = m.match("xyzzy")
    assert res.fdc_id is None and res.method == "none"


def test_alias_overrides_exact_to_different_fdc():
    # alias points "butter" at 200 even though 100 is its exact description match
    m = IngredientMatcher(TABLE, aliases={"butter": 200})
    res = m.match("butter")
    assert res.fdc_id == 200 and res.method == "alias"


def test_jaccard_tiebreak_picks_smallest_fdc():
    # two foods with identical descriptions => equal Jaccard => smaller fdc_id wins,
    # regardless of dict insertion order
    table = {
        900: {"description": "Pepper, green, raw", "per100g": {}, "unit_grams": {}},
        400: {"description": "Pepper, green, raw", "per100g": {}, "unit_grams": {}},
    }
    m = IngredientMatcher(table, jaccard_threshold=0.3)
    res = m.match("green pepper")  # exact won't hit ("green pepper" != "pepper green raw")
    assert res.method == "jaccard"
    assert res.fdc_id == 400  # smallest fdc among the tie
