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
