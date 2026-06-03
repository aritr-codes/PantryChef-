from pathlib import Path

from pantrychef.nutrition.usda import build_artifact, load_artifact, save_artifact


def _write_fixture(d: Path) -> Path:
    (d / "measure_unit.csv").write_text("id,name\n1,cup\n2,tablespoon\n3,large\n", encoding="utf-8")
    (d / "food.csv").write_text(
        "fdc_id,data_type,description\n"
        '100,sr_legacy_food,"Butter, salted"\n'
        '200,sr_legacy_food,"Egg, whole, raw"\n',
        encoding="utf-8",
    )
    (d / "food_nutrient.csv").write_text(
        "id,fdc_id,nutrient_id,amount\n"
        "1,100,1008,717\n2,100,1004,81.11\n3,100,1003,0.85\n4,100,1005,0.06\n"
        "5,200,1008,143\n6,200,1003,12.56\n7,200,1004,9.51\n8,200,1005,0.72\n",
        encoding="utf-8",
    )
    (d / "food_portion.csv").write_text(
        "id,fdc_id,amount,measure_unit_id,modifier,gram_weight\n1,100,1,1,,227\n2,200,1,3,,50\n",
        encoding="utf-8",
    )
    return d


def test_build_artifact_shape(tmp_path):
    table = build_artifact(_write_fixture(tmp_path))
    assert set(table) == {100, 200}
    assert table[100]["description"] == "Butter, salted"
    assert table[100]["per100g"]["kcal"] == 717.0
    assert table[100]["per100g"]["fat_g"] == 81.11
    assert table[100]["unit_grams"]["cup"] == 227.0
    assert table[200]["unit_grams"]["each"] == 50.0  # "large" measure -> each


def test_save_load_roundtrip(tmp_path):
    table = build_artifact(_write_fixture(tmp_path))
    p = tmp_path / "usda.json"
    save_artifact(table, p)
    loaded = load_artifact(p)
    assert loaded[100]["per100g"]["kcal"] == 717.0  # keys coerced back to int
    assert isinstance(next(iter(loaded)), int)


# ---------------------------------------------------------------------------
# New edge-case tests (written before implementation fixes — TDD)
# ---------------------------------------------------------------------------


def _write_minimal(d: Path, food_nutrient_rows: str = "", food_portion_rows: str = "") -> Path:
    """Write minimal CSVs for a single food (fdc_id=300)."""
    (d / "measure_unit.csv").write_text("id,name\n1,cup\n", encoding="utf-8")
    (d / "food.csv").write_text(
        "fdc_id,data_type,description\n300,sr_legacy_food,TestFood\n",
        encoding="utf-8",
    )
    header_n = "id,fdc_id,nutrient_id,amount\n"
    (d / "food_nutrient.csv").write_text(header_n + food_nutrient_rows, encoding="utf-8")
    header_p = "id,fdc_id,amount,measure_unit_id,modifier,gram_weight\n"
    (d / "food_portion.csv").write_text(header_p + food_portion_rows, encoding="utf-8")
    return d


def test_food_with_no_portions_has_empty_unit_grams(tmp_path):
    """A food with no food_portion rows must produce an empty unit_grams dict."""
    table = build_artifact(_write_minimal(tmp_path, food_nutrient_rows="1,300,1008,100\n"))
    assert table[300]["unit_grams"] == {}


def test_non_whitelisted_nutrient_skipped(tmp_path):
    """nutrient_id 9999 is not in WHITELIST and must never appear in per100g."""
    table = build_artifact(_write_minimal(tmp_path, food_nutrient_rows="1,300,9999,42.0\n"))
    assert "9999" not in table[300]["per100g"]
    # Confirm no stray keys beyond the whitelisted names
    allowed = {
        "kcal",
        "protein_g",
        "fat_g",
        "carbs_g",
        "fiber_g",
        "sugar_g",
        "sodium_mg",
        "calcium_mg",
        "iron_mg",
    }
    assert set(table[300]["per100g"].keys()).issubset(allowed)


def test_zero_gram_weight_portion_skipped(tmp_path):
    """A food_portion row with gram_weight 0 must not produce a unit_grams entry."""
    rows = "1,300,1,1,,0\n"  # gram_weight = 0
    table = build_artifact(_write_minimal(tmp_path, food_portion_rows=rows))
    assert table[300]["unit_grams"] == {}


def test_sugar_prefers_2000_over_1063_order_2000_first(tmp_path):
    """When nutrient 2000 appears before 1063, per100g["sugar_g"] == 5.0 (2000's value)."""
    rows = "1,300,2000,5.0\n2,300,1063,9.9\n"
    table = build_artifact(_write_minimal(tmp_path, food_nutrient_rows=rows))
    assert table[300]["per100g"]["sugar_g"] == 5.0


def test_sugar_prefers_2000_over_1063_order_1063_first(tmp_path):
    """When 1063 appears before 2000, per100g["sugar_g"] must still be 5.0 (2000 wins)."""
    rows = "1,300,1063,9.9\n2,300,2000,5.0\n"
    table = build_artifact(_write_minimal(tmp_path, food_nutrient_rows=rows))
    assert table[300]["per100g"]["sugar_g"] == 5.0


def test_excludes_non_consumable_data_types(tmp_path):
    d = _write_fixture(tmp_path)
    # Append a sub_sample_food row (lab metadata, must be excluded) + its nutrient.
    (d / "food.csv").write_text(
        "fdc_id,data_type,description\n"
        '100,sr_legacy_food,"Butter, salted"\n'
        '200,sr_legacy_food,"Egg, whole, raw"\n'
        '999,sub_sample_food,"Butter sub-sample lab"\n',
        encoding="utf-8",
    )
    table = build_artifact(d)
    assert set(table) == {100, 200}  # 999 excluded
    assert 999 not in table
