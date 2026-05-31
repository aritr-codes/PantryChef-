from pantrychef.common.types import Recipe
from scripts.fetch_subs_eval import convert_rows, write_mined


def test_convert_rows_maps_columns() -> None:
    rows = [
        {"ingredient": "Butter", "substitution": "Olive Oil"},
        {"ingredient": "Butter", "substitution": "Margarine"},
    ]
    out = convert_rows(rows, src_col="ingredient", tgt_col="substitution")
    assert ("butter", "olive oil") in out
    assert ("butter", "margarine") in out


def test_write_mined(tmp_path) -> None:
    recipes = [
        Recipe(recipe_id="r0", title="Cake", canonical=["flour", "sugar", "butter", "egg"]),
        Recipe(recipe_id="r1", title="Cake", canonical=["flour", "sugar", "oil", "egg"]),
    ]
    p = tmp_path / "mined.csv"
    n = write_mined(recipes, p, min_overlap=3)
    assert n > 0
    text = p.read_text(encoding="utf-8")
    assert "source,target" in text
