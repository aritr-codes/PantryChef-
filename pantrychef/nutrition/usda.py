"""Build a compact in-memory artifact from USDA FoodData Central bulk CSVs.

Output: ``{fdc_id: {"description": str,
                    "per100g": {nutrient_key: float},
                    "unit_grams": {our_unit_symbol: grams_per_one_unit}}}``.

The messy FDC portion->unit normalization happens here, once, so mass.py is a
simple lookup. Nutrients are restricted to a macro+key-micro whitelist; deeper
micros are deferred (data sparse)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

# USDA nutrient_id -> our key (values already per 100g, in the nutrient's unit).
WHITELIST: dict[int, str] = {
    1008: "kcal",
    1003: "protein_g",
    1004: "fat_g",
    1005: "carbs_g",
    1079: "fiber_g",
    2000: "sugar_g",
    1063: "sugar_g",
    1093: "sodium_mg",
    1087: "calcium_mg",
    1089: "iron_mg",
}

# FDC measure_unit.name -> our normalized unit symbol (matches units.UNITS values).
_UNIT_MAP: dict[str, str] = {
    "cup": "cup",
    "tablespoon": "tbsp",
    "tbsp": "tbsp",
    "teaspoon": "tsp",
    "tsp": "tsp",
    "g": "g",
    "gram": "g",
    "kg": "kg",
    "oz": "oz",
    "ounce": "oz",
    "lb": "lb",
    "pound": "lb",
    "ml": "ml",
    "milliliter": "ml",
    "l": "l",
    "liter": "l",
    "quart": "quart",
    "pint": "pint",
    "clove": "clove",
    "can": "can",
    "slice": "slice",
    "stick": "stick",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_artifact(csv_dir: str | Path) -> dict[int, dict]:
    d = Path(csv_dir)
    units = {int(r["id"]): r["name"].strip().lower() for r in _read_csv(d / "measure_unit.csv")}

    table: dict[int, dict] = {}
    for r in _read_csv(d / "food.csv"):
        fdc = int(r["fdc_id"])
        table[fdc] = {"description": r["description"], "per100g": {}, "unit_grams": {}}

    for r in _read_csv(d / "food_nutrient.csv"):
        nid = int(float(r["nutrient_id"]))
        fdc = int(r["fdc_id"])
        if nid in WHITELIST and fdc in table and r["amount"]:
            table[fdc]["per100g"][WHITELIST[nid]] = float(r["amount"])

    for r in _read_csv(d / "food_portion.csv"):
        fdc = int(r["fdc_id"])
        if fdc not in table:
            continue
        amount = float(r["amount"]) if r.get("amount") else 1.0
        if amount <= 0:
            continue
        gram_weight = float(r["gram_weight"]) if r.get("gram_weight") else 0.0
        if gram_weight <= 0:
            continue
        name = units.get(int(r["measure_unit_id"]), "") if r.get("measure_unit_id") else ""
        symbol = _UNIT_MAP.get(name, "each")  # unmapped measure -> a countable "each"
        per_one = gram_weight / amount
        table[fdc]["unit_grams"].setdefault(symbol, per_one)  # first wins (deterministic)
    return table


def save_artifact(table: dict[int, dict], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({str(k): v for k, v in table.items()}), encoding="utf-8")


def load_artifact(path: str | Path) -> dict[int, dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}
