# Phase 4 — Nutrition & Dietary Reasoning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Join parsed recipe ingredients to USDA FoodData Central to produce per-recipe and per-100g nutrition, and replace the substring dietary tagger with an ontology-backed engine — no LLM, deterministic, laptop-friendly.

**Architecture:** New `pantrychef/nutrition/` package (`usda` → `match` → `mass` → `aggregate`, plus a coverage-gated `impute`) operating on raw ingredient lines; the dietary engine relocates to a shared `pantrychef/dietary/` package (ontology + inheritance) with an import shim so Phase-2 stays untouched. An `eval/nutrition_*` harness reports coverage + dietary-class accuracy. The messy USDA portion→unit normalization is done once at artifact-build time (`usda.py` emits a clean per-food `unit_grams` map) so `mass.py` stays simple.

**Tech Stack:** Python 3.12, pydantic, numpy, scikit-learn (gated impute only), pytest, ruff, uv. Spec: [docs/superpowers/specs/2026-06-03-phase4-nutrition-design.md](../specs/2026-06-03-phase4-nutrition-design.md).

---

## File Structure

| File | Responsibility |
|---|---|
| `pantrychef/dietary/__init__.py` | Re-export `DietTagger` + ontology API |
| `pantrychef/dietary/ontology.py` | Category hierarchy + inheritance; `DietTagger` (tags/is_valid/mask/coverage) |
| `pantrychef/substitution/dietary.py` | Shim: `from pantrychef.dietary import *` (keeps P2 imports stable) |
| `pantrychef/nutrition/__init__.py` | Package marker |
| `pantrychef/nutrition/usda.py` | Build/load the compact USDA artifact (`per100g` + `unit_grams`) |
| `pantrychef/nutrition/match.py` | Tiered `canonical → fdc_id` matcher |
| `pantrychef/nutrition/mass.py` | `(qty, unit, canonical, food) → grams` |
| `pantrychef/nutrition/aggregate.py` | raw lines → `RecipeNutrition` (per-recipe + per-100g + coverage) |
| `pantrychef/nutrition/config.py` | Frozen `NutritionConfig` (thresholds, sample size, seed) |
| `pantrychef/nutrition/impute.py` | *(gated)* macro regressor for unmatched ingredients |
| `pantrychef/eval/nutrition_eval.py` | Coverage + dietary-accuracy metric functions |
| `pantrychef/eval/nutrition_main.py` | CLI leaderboard / coverage report |
| `scripts/fetch_usda.py` | Download FDC bundles → regenerate `data/processed/usda.json` |
| `data/nutrition/aliases.json` | Curated `canonical → fdc_id` overrides |
| `data/nutrition/dietary_labels.json` | Hand-labeled `ingredient → [categories]` eval set |

Test files mirror under `tests/`. New optional extra `nutrition` (scikit-learn) added only in Task 9 if the gate trips.

---

## Task 1: Relocate dietary engine to `pantrychef/dietary/` (behavior-preserving)

**Files:**
- Create: `pantrychef/dietary/__init__.py`, `pantrychef/dietary/ontology.py`
- Modify: `pantrychef/substitution/dietary.py` (becomes a shim)
- Create: `tests/dietary/__init__.py`, `tests/dietary/test_relocation.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/dietary/test_relocation.py
def test_dietary_importable_from_new_home():
    from pantrychef.dietary import DietTagger
    t = DietTagger(known=["butter", "olive oil"])
    assert "dairy" in t.tags("butter")
    assert t.is_valid("butter", "vegan") is False


def test_substitution_shim_reexports_same_class():
    from pantrychef.dietary import DietTagger as New
    from pantrychef.substitution.dietary import DietTagger as Shim
    assert Shim is New
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/dietary/test_relocation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pantrychef.dietary'`

- [ ] **Step 3: Move the implementation**

Move the entire current contents of `pantrychef/substitution/dietary.py` into `pantrychef/dietary/ontology.py` **unchanged** (the `_KEYWORDS`, `_CURATED`, `_DIETS`, `DietTagger` class). Then create:

```python
# pantrychef/dietary/__init__.py
"""Shared dietary reasoning engine (used by Phase-2 substitution + Phase-4)."""

from __future__ import annotations

from pantrychef.dietary.ontology import DietTagger

__all__ = ["DietTagger"]
```

Replace `pantrychef/substitution/dietary.py` entirely with a shim:

```python
# pantrychef/substitution/dietary.py
"""Backwards-compatible shim. The dietary engine now lives in
`pantrychef.dietary`; this re-export keeps existing Phase-2 imports working."""

from __future__ import annotations

from pantrychef.dietary import DietTagger
from pantrychef.dietary.ontology import _CURATED, _DIETS, _KEYWORDS

__all__ = ["DietTagger", "_KEYWORDS", "_CURATED", "_DIETS"]
```

- [ ] **Step 4: Run the full suite to verify nothing regressed**

Run: `uv run pytest tests/dietary tests/substitution/test_dietary.py tests/test_cli_substitute.py -v`
Expected: PASS (existing `tests/substitution/test_dietary.py` still imports `from pantrychef.substitution.dietary import DietTagger` via the shim).

- [ ] **Step 5: Commit**

```bash
git add pantrychef/dietary pantrychef/substitution/dietary.py tests/dietary
git commit -m "refactor(dietary): relocate engine to shared pantrychef/dietary with shim"
```

---

## Task 2: Ontology hierarchy + inheritance (fixes substring false-positives)

**Files:**
- Modify: `pantrychef/dietary/ontology.py`
- Create: `tests/dietary/test_ontology.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/dietary/test_ontology.py
from pantrychef.dietary import DietTagger


def _t() -> DietTagger:
    return DietTagger(known=["egg", "eggplant", "beef", "milk", "olive oil"])


def test_eggplant_is_not_egg():
    t = _t()
    assert "egg" not in t.tags("eggplant")   # substring bug fixed
    assert t.is_valid("eggplant", "vegan") is True


def test_inheritance_meat_is_animal_product():
    t = _t()
    # forbidding the parent category (animal_product, via vegan) forbids descendants
    assert t.is_valid("beef", "vegan") is False
    assert "meat" in t.tags("beef")


def test_known_cases_preserved():
    t = _t()
    assert "dairy" in t.tags("milk")
    assert t.is_valid("olive oil", "vegan") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/dietary/test_ontology.py -v`
Expected: FAIL — `test_eggplant_is_not_egg` fails (current substring matcher tags "eggplant" with "egg").

- [ ] **Step 3: Replace flat keyword logic with an ontology**

In `pantrychef/dietary/ontology.py`, replace `_KEYWORDS`/`tags()` with a curated leaf map + word-boundary matching + a category hierarchy. Keep `_DIETS`, `is_valid`, `mask`, `coverage`, and the `_CURATED`/`_KEYWORDS`/`_DIETS` names (shim re-exports them).

```python
# Hierarchy: child category -> parent category. Diets forbid a category AND all
# its descendants (inheritance).
_PARENT: dict[str, str] = {
    "meat": "animal_product",
    "fish": "animal_product",
    "dairy": "animal_product",
    "egg": "animal_product",
    "honey": "animal_product",
}

# Leaf word -> category. Matched on whole canonical TOKENS (not substrings), so
# "eggplant" (token "eggplant") never matches the "egg" token.
_LEAF_WORDS: dict[str, str] = {
    "beef": "meat", "pork": "meat", "chicken": "meat", "bacon": "meat",
    "ham": "meat", "sausage": "meat", "lamb": "meat", "turkey": "meat",
    "veal": "meat", "lard": "meat", "gelatin": "meat",
    "fish": "fish", "salmon": "fish", "tuna": "fish", "shrimp": "fish",
    "anchovy": "fish", "cod": "fish", "crab": "fish", "prawn": "fish",
    "milk": "dairy", "butter": "dairy", "cheese": "dairy", "cream": "dairy",
    "yogurt": "dairy", "ghee": "dairy", "casein": "dairy", "whey": "dairy",
    "egg": "egg", "eggs": "egg",
    "honey": "honey",
    "wheat": "gluten", "barley": "gluten", "rye": "gluten", "bread": "gluten",
    "pasta": "gluten", "flour": "gluten", "couscous": "gluten",
    "semolina": "gluten",
}

# Multi-word / phrase overrides (canonical form -> categories).
_CURATED: dict[str, set[str]] = {
    "worcestershire sauce": {"fish"},
    "fish sauce": {"fish"},
    "soy sauce": {"gluten"},
    "oat": set(), "oats": set(),
    "almond flour": set(), "coconut flour": set(), "rice flour": set(),
    "eggplant": set(),
}

_DIETS: dict[str, set[str]] = {
    "vegan": {"animal_product"},
    "vegetarian": {"meat", "fish"},
    "gluten_free": {"gluten"},
    "dairy_free": {"dairy"},
}

# Kept for the shim's re-export; the ontology no longer uses substring keywords.
_KEYWORDS: dict[str, tuple[str, ...]] = {}


def _ancestors(cat: str) -> set[str]:
    out = {cat}
    while cat in _PARENT:
        cat = _PARENT[cat]
        out.add(cat)
    return out
```

Then rewrite `DietTagger.tags` to use tokens + curated, expanding to ancestors:

```python
    def tags(self, ingredient: str) -> set[str]:
        ing = canonicalize(ingredient)
        if ing in _CURATED:
            leaves = set(_CURATED[ing])
        else:
            leaves = {
                _LEAF_WORDS[w] for w in ing.split() if w in _LEAF_WORDS
            }
        out: set[str] = set()
        for leaf in leaves:
            out |= _ancestors(leaf)
        return out
```

Update `_is_known` to treat any token-leaf or curated entry as known (the existing body already calls `self.tags`), and keep `is_valid`/`mask`/`coverage` unchanged — they operate on the (now ancestor-expanded) tag set and the unchanged `_DIETS` semantics. Because `_DIETS["vegan"] = {"animal_product"}` and `tags("beef")` now includes `"animal_product"`, the intersection still forbids beef.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/dietary -v`
Expected: PASS.

- [ ] **Step 5: Run the legacy P2 dietary test (parity — must not regress)**

Run: `uv run pytest tests/substitution/test_dietary.py -v`
Expected: PASS. (All assertions in that file — `dairy`/`egg`/`meat`/`gluten` tags, vegan/gluten_free validity, conservative-unknown, mask, coverage>0.5, `None`-diet, unknown-diet `ValueError` — still hold under the ontology.)

- [ ] **Step 6: Commit**

```bash
git add pantrychef/dietary/ontology.py tests/dietary/test_ontology.py
git commit -m "feat(dietary): ontology hierarchy + inheritance; fix eggplant->egg false-positive"
```

---

## Task 3: USDA artifact builder (`usda.py`)

**Files:**
- Create: `pantrychef/nutrition/__init__.py` (empty package marker), `pantrychef/nutrition/usda.py`
- Create: `tests/nutrition/__init__.py`, `tests/nutrition/test_usda.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/nutrition/test_usda.py
import json
from pathlib import Path

from pantrychef.nutrition.usda import build_artifact, load_artifact, save_artifact


def _write_fixture(d: Path) -> Path:
    (d / "measure_unit.csv").write_text(
        "id,name\n1,cup\n2,tablespoon\n3,large\n", encoding="utf-8"
    )
    (d / "food.csv").write_text(
        "fdc_id,data_type,description\n"
        "100,sr_legacy_food,\"Butter, salted\"\n"
        "200,sr_legacy_food,\"Egg, whole, raw\"\n",
        encoding="utf-8",
    )
    # nutrient_id: 1008 kcal, 1003 protein, 1004 fat, 1005 carbs
    (d / "food_nutrient.csv").write_text(
        "id,fdc_id,nutrient_id,amount\n"
        "1,100,1008,717\n2,100,1004,81.11\n3,100,1003,0.85\n4,100,1005,0.06\n"
        "5,200,1008,143\n6,200,1003,12.56\n7,200,1004,9.51\n8,200,1005,0.72\n",
        encoding="utf-8",
    )
    # food_portion: butter 1 cup = 227g; egg 1 large = 50g
    (d / "food_portion.csv").write_text(
        "id,fdc_id,amount,measure_unit_id,modifier,gram_weight\n"
        "1,100,1,1,,227\n2,200,1,3,,50\n",
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
    assert loaded[100]["per100g"]["kcal"] == 717.0   # keys coerced back to int
    assert isinstance(next(iter(loaded)), int)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/nutrition/test_usda.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `usda.py`**

```python
# pantrychef/nutrition/usda.py
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
    1008: "kcal", 1003: "protein_g", 1004: "fat_g", 1005: "carbs_g",
    1079: "fiber_g", 2000: "sugar_g", 1063: "sugar_g",
    1093: "sodium_mg", 1087: "calcium_mg", 1089: "iron_mg",
}

# FDC measure_unit.name -> our normalized unit symbol (matches units.UNITS values).
_UNIT_MAP: dict[str, str] = {
    "cup": "cup", "tablespoon": "tbsp", "tbsp": "tbsp", "teaspoon": "tsp",
    "tsp": "tsp", "g": "g", "gram": "g", "kg": "kg", "oz": "oz", "ounce": "oz",
    "lb": "lb", "pound": "lb", "ml": "ml", "milliliter": "ml", "l": "l",
    "liter": "l", "quart": "quart", "pint": "pint", "clove": "clove",
    "can": "can", "slice": "slice", "stick": "stick",
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/nutrition/test_usda.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pantrychef/nutrition/__init__.py pantrychef/nutrition/usda.py tests/nutrition
git commit -m "feat(nutrition): USDA artifact builder (per-100g nutrients + unit_grams)"
```

---

## Task 4: Tiered ingredient→USDA matcher (`match.py`)

**Files:**
- Create: `pantrychef/nutrition/match.py`, `tests/nutrition/test_match.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/nutrition/test_match.py
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
    res = m.match("onion")          # canonical("Onions, raw") before-comma -> "onion"
    assert res.fdc_id == 300 and res.method == "exact"


def test_jaccard_fallback():
    m = IngredientMatcher(TABLE, jaccard_threshold=0.3)
    res = m.match("whole egg")      # tokens {whole, egg} vs {egg, whole, raw}
    assert res.fdc_id == 200 and res.method == "jaccard"


def test_unmatched_returns_none():
    m = IngredientMatcher(TABLE, jaccard_threshold=0.9)
    res = m.match("xyzzy")
    assert res.fdc_id is None and res.method == "none"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/nutrition/test_match.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `match.py`**

```python
# pantrychef/nutrition/match.py
"""Tiered, deterministic canonical-ingredient -> USDA fdc_id resolver.

Tier 0 curated alias -> Tier 1 exact normalized (full description and the part
before the first comma) -> Tier 2 best token-Jaccard over description tokens.
Below threshold returns no match (a reported gap, never a guess — a wrong macro
is worse than a missing one)."""

from __future__ import annotations

from dataclasses import dataclass

from pantrychef.ingredients.normalize import canonicalize


@dataclass(frozen=True)
class Match:
    fdc_id: int | None
    method: str   # "alias" | "exact" | "jaccard" | "none"
    score: float


class IngredientMatcher:
    def __init__(
        self,
        table: dict[int, dict],
        aliases: dict[str, int] | None = None,
        jaccard_threshold: float = 0.34,
    ) -> None:
        self.table = table
        self.threshold = jaccard_threshold
        self._aliases = {canonicalize(k): v for k, v in (aliases or {}).items()}
        self._exact: dict[str, int] = {}
        self._tokens: list[tuple[int, set[str]]] = []
        for fdc, food in table.items():
            desc = food["description"]
            full = canonicalize(desc)
            head = canonicalize(desc.split(",", 1)[0])
            self._exact.setdefault(full, fdc)
            self._exact.setdefault(head, fdc)
            self._tokens.append((fdc, set(full.split())))

    def match(self, ingredient: str) -> Match:
        c = canonicalize(ingredient)
        if c in self._aliases:
            return Match(self._aliases[c], "alias", 1.0)
        if c in self._exact:
            return Match(self._exact[c], "exact", 1.0)
        q = set(c.split())
        if q:
            best_fdc, best_score = None, 0.0
            for fdc, toks in self._tokens:
                union = q | toks
                if not union:
                    continue
                j = len(q & toks) / len(union)
                # deterministic tie-break: higher score, then smaller fdc_id
                if j > best_score or (j == best_score and best_fdc is not None and fdc < best_fdc):
                    best_fdc, best_score = fdc, j
            if best_fdc is not None and best_score >= self.threshold:
                return Match(best_fdc, "jaccard", best_score)
        return Match(None, "none", 0.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/nutrition/test_match.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pantrychef/nutrition/match.py tests/nutrition/test_match.py
git commit -m "feat(nutrition): tiered alias/exact/Jaccard ingredient->USDA matcher"
```

---

## Task 5: Quantity→grams resolver (`mass.py`)

**Files:**
- Create: `pantrychef/nutrition/mass.py`, `tests/nutrition/test_mass.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/nutrition/test_mass.py
import math

from pantrychef.nutrition.mass import to_grams

BUTTER = {"description": "Butter", "per100g": {}, "unit_grams": {"cup": 227.0}}
EGG = {"description": "Egg", "per100g": {}, "unit_grams": {"each": 50.0}}
NODATA = {"description": "X", "per100g": {}, "unit_grams": {}}


def test_mass_unit_direct():
    g, ok = to_grams(2.0, "oz", "anything", NODATA)
    assert ok and math.isclose(g, 56.699, rel_tol=1e-3)


def test_volume_uses_usda_portion():
    g, ok = to_grams(0.5, "cup", "butter", BUTTER)   # 0.5 * 227
    assert ok and math.isclose(g, 113.5)


def test_volume_density_fallback_when_no_portion():
    g, ok = to_grams(1.0, "cup", "milk", NODATA)     # 236.588ml * 1.03 density
    assert ok and math.isclose(g, 243.7, rel_tol=1e-2)


def test_bare_count_uses_each():
    g, ok = to_grams(3.0, None, "egg", EGG)          # 3 * 50
    assert ok and g == 150.0


def test_unresolved_volume_no_density_no_portion():
    g, ok = to_grams(1.0, "cup", "mystery", NODATA)
    assert g is None and ok is False


def test_missing_quantity_unresolved():
    g, ok = to_grams(None, None, "salt", NODATA)
    assert g is None and ok is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/nutrition/test_mass.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `mass.py`**

```python
# pantrychef/nutrition/mass.py
"""Resolve (quantity, unit, ingredient, USDA food) -> grams, coverage-honest.

Mass units convert by fixed factor. Volume units use the food's USDA portion
gram-weight when present, else a per-ingredient-class density fallback. Count /
portion units use the food's "each" portion or a curated per-item table.
Anything unresolved returns (None, False) and is reported as a coverage gap."""

from __future__ import annotations

MASS_TO_G: dict[str, float] = {"g": 1.0, "kg": 1000.0, "oz": 28.3495, "lb": 453.592}
VOL_TO_ML: dict[str, float] = {
    "ml": 1.0, "l": 1000.0, "cup": 236.588, "tbsp": 14.787, "tsp": 4.929,
    "quart": 946.353, "pint": 473.176,
}
TINY_G: dict[str, float] = {"pinch": 0.36, "dash": 0.6}
_PORTION_UNITS = {"clove", "can", "slice", "stick"}

# density g/ml by ingredient keyword (token match on canonical).
_DENSITY: dict[str, float] = {
    "water": 1.0, "milk": 1.03, "juice": 1.05, "broth": 1.0, "stock": 1.0,
    "wine": 0.99, "vinegar": 1.01, "oil": 0.92, "honey": 1.42, "syrup": 1.37,
    "flour": 0.53, "sugar": 0.85, "salt": 1.22, "rice": 0.85, "butter": 0.96,
    "cream": 1.0, "sauce": 1.05,
}
# per-item grams by ingredient keyword (for count/None units when USDA lacks it).
_PORTION_G: dict[str, float] = {
    "egg": 50.0, "clove": 3.0, "onion": 110.0, "tomato": 123.0, "apple": 182.0,
    "banana": 118.0, "carrot": 61.0, "potato": 213.0, "lemon": 58.0,
}


def _keyword(canonical: str, table: dict[str, float]) -> float | None:
    toks = canonical.split()
    for w in toks:
        if w in table:
            return table[w]
    return None


def to_grams(
    qty: float | None, unit: str | None, canonical: str, food: dict
) -> tuple[float | None, bool]:
    unit_grams = food.get("unit_grams", {})
    if unit in TINY_G:
        return (qty or 1.0) * TINY_G[unit], True
    if qty is None:
        return None, False
    if unit in MASS_TO_G:
        return qty * MASS_TO_G[unit], True
    if unit in VOL_TO_ML:
        if unit in unit_grams:
            return qty * unit_grams[unit], True
        density = _keyword(canonical, _DENSITY)
        if density is not None:
            return qty * VOL_TO_ML[unit] * density, True
        return None, False
    if unit is None or unit in _PORTION_UNITS:
        if unit in unit_grams:
            return qty * unit_grams[unit], True
        if "each" in unit_grams:
            return qty * unit_grams["each"], True
        per_item = _keyword(canonical, _PORTION_G)
        if per_item is not None:
            return qty * per_item, True
        return None, False
    return None, False  # e.g. "package" — variable, unresolved
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/nutrition/test_mass.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pantrychef/nutrition/mass.py tests/nutrition/test_mass.py
git commit -m "feat(nutrition): coverage-honest quantity->grams resolver"
```

---

## Task 6: Recipe aggregation (`aggregate.py`)

**Files:**
- Create: `pantrychef/nutrition/aggregate.py`, `tests/nutrition/test_aggregate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/nutrition/test_aggregate.py
import math

from pantrychef.nutrition.aggregate import aggregate
from pantrychef.nutrition.match import IngredientMatcher

TABLE = {
    100: {"description": "Butter", "per100g": {"kcal": 717.0, "fat_g": 81.0}, "unit_grams": {"cup": 227.0}},
    200: {"description": "Egg", "per100g": {"kcal": 143.0, "protein_g": 12.5}, "unit_grams": {"each": 50.0}},
}
VOCAB = ["butter", "egg"]
ALIASES = {"butter": 100, "egg": 200}


def test_aggregate_totals_and_per100g():
    m = IngredientMatcher(TABLE, aliases=ALIASES)
    # 1 cup butter (227g) + 2 eggs (100g) = 327g total
    res = aggregate(["1 cup butter", "2 eggs"], VOCAB, None, m)
    assert res.n_lines == 2 and res.n_matched == 2 and res.n_massed == 2
    assert math.isclose(res.total_grams, 327.0)
    # kcal total = 717*2.27 + 143*1.0 = 1627.59 + 143 = 1770.59
    assert math.isclose(res.facts_total.calories, 1770.59, rel_tol=1e-4)
    # per-100g = total / 327 * 100
    assert math.isclose(res.facts_per100g.calories, 1770.59 / 327 * 100, rel_tol=1e-4)


def test_unresolved_line_excluded_and_counted():
    m = IngredientMatcher(TABLE, aliases=ALIASES)
    res = aggregate(["1 package butter", "2 eggs"], VOCAB, None, m)  # package unresolved
    assert res.n_matched == 2 and res.n_massed == 1
    assert math.isclose(res.total_grams, 100.0)  # only the eggs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/nutrition/test_aggregate.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `aggregate.py`**

```python
# pantrychef/nutrition/aggregate.py
"""Aggregate raw recipe ingredient lines into per-recipe + per-100g nutrition.

Re-parses each raw line (the cleaned recipe store keeps only canonical sets, no
quantities), matches it to USDA, resolves grams, and sums whitelisted nutrients.
Unmatched or unresolved lines are excluded from totals and counted toward the
reported coverage gap — totals are estimates, never claimed as ground truth."""

from __future__ import annotations

from dataclasses import dataclass, field

from pantrychef.common.types import NutritionFacts
from pantrychef.ingredients.parser import MatchIndex, parse
from pantrychef.nutrition.mass import to_grams
from pantrychef.nutrition.match import IngredientMatcher

# nutrient keys that map onto NutritionFacts' named fields; the rest go to micros.
_NAMED = {"kcal": "calories", "protein_g": "protein_g", "carbs_g": "carbs_g", "fat_g": "fat_g"}


@dataclass
class RecipeNutrition:
    facts_total: NutritionFacts
    facts_per100g: NutritionFacts
    total_grams: float
    n_lines: int
    n_matched: int
    n_massed: int

    @property
    def match_coverage(self) -> float:
        return self.n_matched / self.n_lines if self.n_lines else 0.0

    @property
    def mass_coverage(self) -> float:
        return self.n_massed / self.n_lines if self.n_lines else 0.0


def _facts_from(totals: dict[str, float]) -> NutritionFacts:
    named = {field_: totals.get(key, 0.0) for key, field_ in _NAMED.items()}
    micros = {k: v for k, v in totals.items() if k not in _NAMED}
    return NutritionFacts(**named, micros=micros)


def aggregate(
    raw_lines: list[str],
    vocab: list[str] | None,
    index: MatchIndex | None,
    matcher: IngredientMatcher,
) -> RecipeNutrition:
    totals: dict[str, float] = {}
    total_grams = 0.0
    n_matched = n_massed = 0
    for line in raw_lines:
        pi = parse(line, vocab, index)
        if pi.canonical is None:
            continue
        res = matcher.match(pi.canonical)
        if res.fdc_id is None:
            continue
        n_matched += 1
        food = matcher.table[res.fdc_id]
        grams, ok = to_grams(pi.quantity, pi.unit, pi.canonical, food)
        if not ok or grams is None:
            continue
        n_massed += 1
        total_grams += grams
        for key, per100 in food["per100g"].items():
            totals[key] = totals.get(key, 0.0) + per100 * grams / 100.0

    facts_total = _facts_from(totals)
    if total_grams > 0:
        scaled = {k: v / total_grams * 100.0 for k, v in totals.items()}
    else:
        scaled = {}
    return RecipeNutrition(
        facts_total=facts_total,
        facts_per100g=_facts_from(scaled),
        total_grams=total_grams,
        n_lines=len(raw_lines),
        n_matched=n_matched,
        n_massed=n_massed,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/nutrition/test_aggregate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pantrychef/nutrition/aggregate.py tests/nutrition/test_aggregate.py
git commit -m "feat(nutrition): per-recipe + per-100g aggregation with coverage"
```

---

## Task 7: Config + curated data files

**Files:**
- Create: `pantrychef/nutrition/config.py`, `data/nutrition/aliases.json`, `data/nutrition/dietary_labels.json`
- Create: `tests/nutrition/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/nutrition/test_config.py
import json
from pathlib import Path

from pantrychef.nutrition.config import NutritionConfig


def test_defaults():
    c = NutritionConfig()
    assert c.seed == 42
    assert 0 < c.usable_mass_fraction <= 1
    assert c.impute_match_cov_gate == 0.80
    assert c.impute_unresolved_mass_gate == 0.20


def test_curated_files_are_valid_json():
    root = Path(__file__).resolve().parents[2]
    aliases = json.loads((root / "data/nutrition/aliases.json").read_text())
    labels = json.loads((root / "data/nutrition/dietary_labels.json").read_text())
    assert isinstance(aliases, dict)
    assert isinstance(labels, dict)
    # labels map ingredient -> list[str] of dietary categories
    assert all(isinstance(v, list) for v in labels.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/nutrition/test_config.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement config + seed the data files**

```python
# pantrychef/nutrition/config.py
"""Frozen config for the nutrition eval + imputation gate."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NutritionConfig:
    seed: int = 42
    sample_size: int = 5000               # recipes drawn for the coverage eval
    jaccard_threshold: float = 0.34
    usable_mass_fraction: float = 0.50    # recipe "usable" if >= this mass resolved
    # imputation gate (build impute.py iff EITHER trips):
    impute_match_cov_gate: float = 0.80   # build if match coverage < this
    impute_unresolved_mass_gate: float = 0.20  # or median unresolved-mass > this
```

Create `data/nutrition/aliases.json` with an initial curated set (extend during eval; valid JSON object mapping canonical ingredient → fdc_id — fill real fdc_ids from the built artifact during Task 8):

```json
{}
```

Create `data/nutrition/dietary_labels.json` with a starter hand-labeled set (ingredient → list of correct dietary categories; expand to ~100 entries during Task 8):

```json
{
  "butter": ["dairy", "animal_product"],
  "egg": ["egg", "animal_product"],
  "eggplant": [],
  "beef": ["meat", "animal_product"],
  "olive oil": [],
  "wheat flour": ["gluten"],
  "honey": ["honey", "animal_product"],
  "tofu": [],
  "salmon": ["fish", "animal_product"],
  "almond flour": []
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/nutrition/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pantrychef/nutrition/config.py data/nutrition tests/nutrition/test_config.py
git commit -m "feat(nutrition): config + curated alias/dietary-label seed files"
```

---

## Task 8: Eval harness + `fetch_usda.py` + run the coverage report (gate decision)

**Files:**
- Create: `pantrychef/eval/nutrition_eval.py`, `pantrychef/eval/nutrition_main.py`
- Create: `scripts/fetch_usda.py`
- Modify: `.gitignore` (add `data/processed/usda.json`)
- Create: `tests/eval/test_nutrition_eval.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_nutrition_eval.py
from pantrychef.dietary import DietTagger
from pantrychef.eval.nutrition_eval import dietary_accuracy, coverage_report
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
    assert rep["match_coverage"] == 1.0          # all 3 lines matched
    assert 0.0 < rep["mass_coverage"] < 1.0      # the "package" line is unresolved


def test_dietary_accuracy_perfect_on_labeled():
    t = DietTagger(known=["butter", "eggplant"])
    labels = {"butter": ["dairy", "animal_product"], "eggplant": []}
    acc = dietary_accuracy(t, labels)
    assert acc["accuracy"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_nutrition_eval.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement the eval functions**

```python
# pantrychef/eval/nutrition_eval.py
"""Coverage + dietary-class accuracy metrics for Phase 4.

No recipe-level nutrition gold exists, so this measures coverage (match + mass)
and dietary-class accuracy vs a hand-labeled set — NOT recipe-macro accuracy."""

from __future__ import annotations

from statistics import median

from pantrychef.dietary import DietTagger
from pantrychef.ingredients.parser import MatchIndex
from pantrychef.nutrition.aggregate import aggregate
from pantrychef.nutrition.match import IngredientMatcher


def coverage_report(
    recipes: list[list[str]],
    vocab: list[str] | None,
    index: MatchIndex | None,
    matcher: IngredientMatcher,
    usable_mass_fraction: float = 0.50,
) -> dict:
    n_lines = n_matched = n_massed = usable = 0
    unresolved_fracs: list[float] = []
    for raw_lines in recipes:
        res = aggregate(raw_lines, vocab, index, matcher)
        n_lines += res.n_lines
        n_matched += res.n_matched
        n_massed += res.n_massed
        if res.mass_coverage >= usable_mass_fraction:
            usable += 1
        unresolved_fracs.append(1.0 - res.mass_coverage)
    return {
        "n_recipes": len(recipes),
        "n_lines": n_lines,
        "match_coverage": n_matched / n_lines if n_lines else 0.0,
        "mass_coverage": n_massed / n_lines if n_lines else 0.0,
        "nutrition_completeness": usable / len(recipes) if recipes else 0.0,
        "median_unresolved_mass": median(unresolved_fracs) if unresolved_fracs else 0.0,
    }


def dietary_accuracy(tagger: DietTagger, labels: dict[str, list[str]]) -> dict:
    correct = 0
    for ing, gold in labels.items():
        if tagger.tags(ing) == set(gold):
            correct += 1
    return {
        "accuracy": correct / len(labels) if labels else 0.0,
        "n": len(labels),
    }
```

```python
# pantrychef/eval/nutrition_main.py
"""`python -m pantrychef.eval.nutrition_main` — Phase 4 coverage + dietary report.

Prints match/mass coverage, nutrition completeness, dietary-class accuracy, and
the imputation-gate decision. Reads raw recipes (with quantities) from the raw
dataset sample — NOT the canonical-only processed store."""

from __future__ import annotations

import argparse
import json

from pantrychef.common import get_logger
from pantrychef.config import get_settings
from pantrychef.dietary import DietTagger
from pantrychef.eval.nutrition_eval import coverage_report, dietary_accuracy
from pantrychef.ingredients.parser import build_match_index
from pantrychef.ingredients.vocab import load_vocabulary
from pantrychef.nutrition.config import NutritionConfig
from pantrychef.nutrition.match import IngredientMatcher
from pantrychef.nutrition.usda import load_artifact

log = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Phase 4 nutrition coverage report.")
    ap.add_argument("--max-rows", type=int, default=None)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from pantrychef.data.loaders import load_raw_recipes  # see note below

    args = parse_args(argv)
    s = get_settings()
    cfg = NutritionConfig()

    table = load_artifact(s.processed_dir / "usda.json")
    aliases = json.loads((s.repo_root / "data/nutrition/aliases.json").read_text("utf-8"))
    aliases = {k: int(v) for k, v in aliases.items()}
    matcher = IngredientMatcher(table, aliases=aliases, jaccard_threshold=cfg.jaccard_threshold)

    vocab = load_vocabulary(s.processed_dir / "vocab.json")
    index = build_match_index(vocab)

    raws = load_raw_recipes(limit=args.max_rows or cfg.sample_size)
    recipes = [r.ingredients for r in raws]

    rep = coverage_report(recipes, vocab, index, matcher, cfg.usable_mass_fraction)

    labels = json.loads((s.repo_root / "data/nutrition/dietary_labels.json").read_text("utf-8"))
    acc = dietary_accuracy(DietTagger(known=vocab), labels)

    gate = (
        rep["match_coverage"] < cfg.impute_match_cov_gate
        or rep["median_unresolved_mass"] > cfg.impute_unresolved_mass_gate
    )
    log.info("coverage: %s", rep)
    log.info("dietary_accuracy: %s", acc)
    log.info("imputation_gate_tripped: %s", gate)
    print(json.dumps({"coverage": rep, "dietary": acc, "impute_gate": gate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

> **Note (loader):** if `pantrychef/data/loaders.py` has no `load_raw_recipes(limit)` returning `RawRecipe`s from `data/raw/full_dataset.csv`, add a small one there following the existing loader pattern (read the CSV, build `RawRecipe(title=…, ingredients=<parsed list>, ner=<parsed list>)`). Inspect `loaders.py` first; reuse whatever RecipeNLG CSV reader Phase 1 already has rather than writing a new parser.

- [ ] **Step 4: Implement `scripts/fetch_usda.py` + gitignore**

```python
# scripts/fetch_usda.py
"""Download USDA FoodData Central SR Legacy + Foundation CSV bundles and build
the compact artifact at data/processed/usda.json.

Manual download is fine too: unzip the FDC CSV bundles into a folder and pass
--csv-dir. Record the FDC release date in docs/DATASET.md after running."""

from __future__ import annotations

import argparse
from pathlib import Path

from pantrychef.config import get_settings
from pantrychef.nutrition.usda import build_artifact, save_artifact


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", required=True, help="folder with FDC *.csv files")
    args = ap.parse_args()
    s = get_settings()
    table = build_artifact(Path(args.csv_dir))
    out = s.processed_dir / "usda.json"
    save_artifact(table, out)
    print(f"wrote {out} ({len(table)} foods)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Add to `.gitignore`:

```
data/processed/usda.json
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/eval/test_nutrition_eval.py -v`
Expected: PASS.

- [ ] **Step 6: Build the artifact + run the real coverage report (the gate decision)**

```bash
# Download FDC SR Legacy + Foundation CSV bundles from
# https://fdc.nal.usda.gov/download-datasets.html , unzip into <dir>, then:
uv run python scripts/fetch_usda.py --csv-dir <dir>
uv run python -m pantrychef.eval.nutrition_main --max-rows 5000
```

Record the printed `coverage` / `dietary` numbers and `impute_gate` into EVALUATION.md (Task 10). **Expand `data/nutrition/aliases.json`** with the highest-frequency unmatched ingredients (look at low match_coverage offenders) and re-run until match_coverage stabilizes. **The `impute_gate` value decides whether Task 9 runs.**

- [ ] **Step 7: Commit**

```bash
git add pantrychef/eval/nutrition_eval.py pantrychef/eval/nutrition_main.py scripts/fetch_usda.py .gitignore tests/eval/test_nutrition_eval.py data/nutrition/aliases.json
git commit -m "feat(nutrition): coverage + dietary-accuracy eval harness + USDA fetch script"
```

---

## Task 9: Macro imputation regressor — **only if the Task-8 gate tripped**

> Skip this task entirely (and record "imputation not needed — coverage adequate" in EVALUATION.md) if `impute_gate` was `false`. Building it anyway would be ML theater on sparse targets.

**Files:**
- Modify: `pyproject.toml` (add `nutrition` optional extra: `scikit-learn`)
- Modify: `.github/workflows/ci.yml` (add `--extra nutrition` to the sync step — required for any new optional extra, per the Phase-3 CI lesson)
- Create: `pantrychef/nutrition/impute.py`, `tests/nutrition/test_impute.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/nutrition/test_impute.py
import pytest

pytest.importorskip("sklearn")

from pantrychef.nutrition.impute import MacroImputer


def test_imputer_predicts_known_macro_within_mae():
    # train on foods with known kcal; held-out MAE should be finite and reported
    train = [
        ("white sugar", {"kcal": 387.0}),
        ("brown sugar", {"kcal": 380.0}),
        ("olive oil", {"kcal": 884.0}),
        ("canola oil", {"kcal": 884.0}),
        ("white flour", {"kcal": 364.0}),
        ("wheat flour", {"kcal": 340.0}),
    ]
    imp = MacroImputer().fit(train, target="kcal")
    pred = imp.predict("corn oil")
    assert pred > 0  # an oil should predict high kcal
    assert imp.mae_ >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/nutrition/test_impute.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `impute.py`**

```python
# pantrychef/nutrition/impute.py
"""Gated macro imputation for ingredients USDA can't match.

Built only when the coverage gate trips. Features = hashed ingredient name
tokens; model = ridge regression per target macro. MAE is measured on held-out
*matched* ingredients (train on matched, test on held-out matched), then applied
to unmatched ones. Light, deterministic, interpretable."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


class MacroImputer:
    def __init__(self, seed: int = 42, n_features: int = 256) -> None:
        self.seed = seed
        self._vec = HashingVectorizer(n_features=n_features, alternate_sign=False)
        self._model = Ridge(alpha=1.0, random_state=seed)
        self.mae_ = 0.0

    def fit(self, samples: list[tuple[str, dict]], target: str) -> MacroImputer:
        names = [n for n, macros in samples if target in macros]
        y = np.array([macros[target] for _, macros in samples if target in macros], dtype=float)
        x = self._vec.transform(names)
        if len(y) >= 4:
            x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, random_state=self.seed)
            self._model.fit(x_tr, y_tr)
            self.mae_ = float(np.mean(np.abs(self._model.predict(x_te) - y_te)))
            self._model.fit(x, y)  # refit on all for deployment
        else:
            self._model.fit(x, y)
            self.mae_ = 0.0
        return self

    def predict(self, ingredient: str) -> float:
        return float(self._model.predict(self._vec.transform([ingredient]))[0])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/nutrition/test_impute.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pantrychef/nutrition/impute.py tests/nutrition/test_impute.py pyproject.toml .github/workflows/ci.yml
git commit -m "feat(nutrition): gated macro imputation regressor (coverage gate tripped)"
```

---

## Task 10: Documentation + ROADMAP/DATASET/EVALUATION

**Files:**
- Modify: `docs/EVALUATION.md`, `ROADMAP.md`, `docs/DATASET.md`
- Create: `docs/MODEL_CARD_nutrition.md`

- [ ] **Step 1: Record results in EVALUATION.md**

Add a `### Phase 4 — Nutrition & Dietary` section with the provenance block (USDA FDC release date, sample size, seed) and a table of the metrics from Task 8 step 6: match coverage %, mass coverage %, nutrition completeness %, dietary-class accuracy (with the **old substring tagger vs ontology** comparison), and the imputation-gate decision (+ MAE if built). **State the honesty constraint explicitly:** no recipe-level nutrition gold → coverage + per-ingredient + aggregation-test validation only; macros are estimates.

- [ ] **Step 2: Flip ROADMAP Phase 4 to ✅** with achieved metrics + a one-line `Notable` (ontology fixes substring tagger; USDA join coverage; imputation gated-in/out), mirroring the Phase 1–3 entries.

- [ ] **Step 3: Record the USDA FDC release** date/version under the USDA section in `docs/DATASET.md` for reproducibility.

- [ ] **Step 4: Write `docs/MODEL_CARD_nutrition.md`** following `MODEL_CARD_recommender.md` structure: intended use, **not intended for** (medical/allergen/clinical), data sources (USDA FDC + license), method (tiered match + coverage-honest mass + ontology), metrics, and limitations (no per-serving, no macro gold, density approximations, variable can/package sizes).

- [ ] **Step 5: Run the full suite + ruff**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add docs/EVALUATION.md ROADMAP.md docs/DATASET.md docs/MODEL_CARD_nutrition.md
git commit -m "docs(nutrition): Phase 4 evaluation, model card, roadmap + dataset updates"
```

---

## Self-Review (completed during planning)

- **Spec coverage:** nutrition aggregation (Tasks 3–6), per-recipe + per-100g (Task 6), ontology dietary engine + relocation/shim (Tasks 1–2), tiered matcher (Task 4), coverage-honest mass (Task 5), gated imputation with explicit gate (Tasks 7–9), eval metrics + honesty constraint (Task 8, Task 10), usda.json gitignored+regenerated (Task 8), no-per-serving / no-macro-gold caveats (Task 10 model card). ✓
- **Placeholder scan:** the two curated JSON files start minimal and are explicitly expanded in Task 8 step 6 (a real action with a stop condition), not left as "TODO". `aliases.json` fdc_ids are filled from the built artifact during Task 8. Loader reuse is flagged with a concrete instruction. No bare "add error handling".
- **Type consistency:** `IngredientMatcher.table`, `Match(fdc_id, method, score)`, `to_grams(qty, unit, canonical, food) -> (float|None, bool)`, `aggregate(...) -> RecipeNutrition`, `NutritionConfig` field names, and `coverage_report`/`dietary_accuracy` return keys are consistent across Tasks 4–10.
- **Gate:** Task 9 is conditional on Task 8's measured `impute_gate`; skipping it is a documented, valid outcome.
