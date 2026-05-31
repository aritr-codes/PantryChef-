import json

from pantrychef.common.types import Recipe
from pantrychef.eval.subs_gold import load_curated, load_pairs_csv, mine_pairs


def test_load_pairs_csv_canonicalizes(tmp_path) -> None:
    p = tmp_path / "gold.csv"
    p.write_text("source,target\nButter,Olive-Oil\nButter,Margarine\n", encoding="utf-8")
    gold = load_pairs_csv(p)
    assert gold["butter"] == {"olive oil", "margarine"}


def test_load_curated(tmp_path) -> None:
    p = tmp_path / "curated.json"
    p.write_text(json.dumps({"milk": ["soy milk", "almond milk"]}), encoding="utf-8")
    gold = load_curated(p)
    assert gold["milk"] == {"soy milk", "almond milk"}


def test_mine_pairs_from_near_duplicate_recipes() -> None:
    # Two recipes identical except one ingredient differs => mined sub pair.
    recipes = [
        Recipe(recipe_id="r0", title="Cake", canonical=["flour", "sugar", "butter", "egg"]),
        Recipe(recipe_id="r1", title="Cake", canonical=["flour", "sugar", "oil", "egg"]),
    ]
    gold = mine_pairs(recipes, min_overlap=3)
    assert "oil" in gold.get("butter", set()) or "butter" in gold.get("oil", set())
