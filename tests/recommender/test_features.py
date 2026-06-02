import numpy as np

from pantrychef.common.types import Recipe
from pantrychef.recommender.features import (
    FEATURE_NAMES,
    SUB_FEATURES,
    extract_features,
    to_matrix,
)


def _recipe(rid, ings):
    return Recipe(recipe_id=rid, title=rid, canonical=ings)


def test_basic_overlap_features():
    pantry = {"flour", "egg", "milk"}
    recipe = _recipe("r", ["flour", "egg", "butter", "sugar"])  # len 4
    f = extract_features(pantry, recipe, sub_lookup=None)
    assert f["coverage"] == 2 / 4
    assert f["match_frac_pantry"] == 2 / 3
    assert f["n_matched"] == 2
    assert f["n_missing"] == 2  # butter, sugar
    assert f["recipe_len"] == 4
    assert f["sub_fill_max"] == 0.0  # no sub_lookup
    assert f["sub_fill_mean"] == 0.0


def test_sub_fill_uses_lookup():
    pantry = {"margarine", "honey"}
    recipe = _recipe("r", ["butter", "sugar", "margarine"])
    # missing = {butter, sugar}; margarine substitutes butter (0.9), nothing for sugar
    lookup = {"butter": {"margarine": 0.9, "oil": 0.4}, "sugar": {"stevia": 0.7}}
    f = extract_features(pantry, recipe, sub_lookup=lambda m: lookup.get(m, {}))
    assert f["sub_fill_max"] == 0.9
    assert f["sub_fill_mean"] == (0.9 + 0.0) / 2


def test_dietary_ok_flag():
    pantry = {"tofu"}
    recipe = _recipe("r", ["tofu", "rice", "soy sauce", "ginger"])
    f = extract_features(pantry, recipe, sub_lookup=None)
    assert f["dietary_ok"] == 1.0


def test_empty_recipe_is_safe():
    f = extract_features({"egg"}, _recipe("r", []), sub_lookup=None)
    assert f["coverage"] == 0.0
    assert f["recipe_len"] == 0


def test_to_matrix_column_order():
    feats = [
        {n: float(i) for i, n in enumerate(FEATURE_NAMES)},
        {n: float(i) * 2 for i, n in enumerate(FEATURE_NAMES)},
    ]
    m = to_matrix(feats, FEATURE_NAMES)
    assert m.shape == (2, len(FEATURE_NAMES))
    assert np.allclose(m[0], np.arange(len(FEATURE_NAMES), dtype=float))


def test_to_matrix_drops_sub_columns():
    cols = [n for n in FEATURE_NAMES if n not in SUB_FEATURES]
    feats = [{n: 1.0 for n in FEATURE_NAMES}]  # noqa: C420
    m = to_matrix(feats, cols)
    assert m.shape == (1, len(FEATURE_NAMES) - len(SUB_FEATURES))
