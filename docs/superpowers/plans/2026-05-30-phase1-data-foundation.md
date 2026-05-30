# Phase 1 — Data Foundation & Ingredient Intelligence — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible recipe corpus loader, a hybrid (rules + vocab) free-text ingredient parser, a set-overlap retrieval baseline, and an evaluation harness — exposed via a `pantrychef cook` CLI.

**Architecture:** Deterministic pipeline, no ML yet. `data` ingests RecipeNLG CSV → `RawRecipe`. `ingredients` normalizes text, parses quantity/unit by rules, and resolves a canonical ingredient by longest-match against a vocabulary built from RecipeNLG's `NER` column. `data.clean` turns raw recipes into `Recipe` objects (canonical ingredient sets). `retrieval` builds an inverted index and ranks recipes by ingredient coverage. `eval` measures parser P/R/F1 (vs the `NER` ground truth) and retrieval recall@k (leave-one-ingredient-out).

**Tech Stack:** Python 3.11, pydantic v2, pandas, pytest, ruff, uv. All tests run offline against an in-code fixture corpus (no dataset download needed for CI).

**Conventions:** TDD (test first, watch it fail, minimal impl, watch it pass, commit). Exact paths. Run commands from repo root. Use `uv run` for everything. Commit messages use Conventional Commits.

---

## File Structure

**Create:**
- `tests/conftest.py` — shared `sample_raws` fixture (in-code corpus).
- `pantrychef/data/schemas.py` — `RawRecipe` pydantic model.
- `pantrychef/data/loaders.py` — `load_recipenlg` CSV reader.
- `pantrychef/data/clean.py` — `to_recipe`, `clean_recipes`.
- `pantrychef/data/store.py` — JSONL save/load of `Recipe`.
- `pantrychef/ingredients/normalize.py` — `normalize`, `singularize`.
- `pantrychef/ingredients/units.py` — unit table, `parse_quantity`, `parse_unit`.
- `pantrychef/ingredients/vocab.py` — `build_vocabulary`, save/load, `_canon_token`.
- `pantrychef/ingredients/parser.py` — `match_canonical`, `parse`.
- `pantrychef/retrieval/index.py` — `InvertedIndex`.
- `pantrychef/retrieval/baseline.py` — `recommend`.
- `pantrychef/eval/__init__.py`, `parser_eval.py`, `retrieval_eval.py`.
- `tests/test_loaders.py`, `test_normalize.py`, `test_units.py`, `test_vocab.py`, `test_parser.py`, `test_clean_store.py`, `test_retrieval.py`, `test_eval.py`, `test_cli_cook.py`.

**Modify:**
- `pantrychef/common/types.py` — add `Recipe`; extend `ScoredRecipe`.
- `pantrychef/cli.py` — add `cook` subcommand.
- `scripts/download_data.py` — real RecipeNLG fetch instructions + run pipeline.
- `pyproject.toml` — add `pandas` runtime dep.
- `Makefile` — wire `data` and `eval` targets.
- `docs/DATASET.md` — fill preprocessing + reproducibility steps.

---

## Task 1: Types + shared test fixture + pandas dep

**Files:**
- Modify: `pyproject.toml`
- Modify: `pantrychef/common/types.py`
- Create: `tests/conftest.py`
- Test: `tests/conftest.py` is exercised by later tasks; this task adds a direct type test in `tests/test_smoke.py` (modify).

- [ ] **Step 1: Add pandas dependency**

Edit `pyproject.toml`, change the `dependencies` list to:

```toml
dependencies = [
    "pydantic>=2.6",
    "pydantic-settings>=2.2",
    "pandas>=2.2",
]
```

Then sync:

Run: `uv sync --extra dev`
Expected: resolves and installs pandas.

- [ ] **Step 2: Write failing test for new types**

Append to `tests/test_smoke.py`:

```python
def test_recipe_and_scored_recipe_types() -> None:
    from pantrychef.common.types import Recipe, ScoredRecipe

    r = Recipe(recipe_id="r0", title="Pancakes", ingredients_raw=["2 cups flour"], canonical=["flour"])
    assert r.canonical == ["flour"]

    s = ScoredRecipe(recipe_id="r0", score=1.0, title="Pancakes", matched=["flour"], missing=["egg"])
    assert s.title == "Pancakes"
    assert s.matched == ["flour"]
    assert s.missing == ["egg"]
```

Run: `uv run pytest tests/test_smoke.py::test_recipe_and_scored_recipe_types -v`
Expected: FAIL — `ImportError: cannot import name 'Recipe'` (and ScoredRecipe lacks `title`/`matched`).

- [ ] **Step 3: Add/extend the types**

Edit `pantrychef/common/types.py`. Replace the `ScoredRecipe` class and add `Recipe`:

```python
class Recipe(BaseModel):
    """A cleaned recipe with a canonical ingredient set (Phase 1)."""

    recipe_id: str
    title: str
    ingredients_raw: list[str] = []
    canonical: list[str] = []


class ScoredRecipe(BaseModel):
    """A ranked recipe result (Phase 1/3)."""

    recipe_id: str
    score: float
    title: str = ""
    matched: list[str] = []
    missing: list[str] = []
```

- [ ] **Step 4: Create the shared fixture**

Create `tests/conftest.py`:

```python
"""Shared offline test corpus — mirrors the RecipeNLG schema in-code so tests
need no dataset download. `ner` plays the role of RecipeNLG's NER column
(ground-truth canonical entities)."""

from __future__ import annotations

import pytest

from pantrychef.data.schemas import RawRecipe


@pytest.fixture
def sample_raws() -> list[RawRecipe]:
    return [
        RawRecipe(
            title="Pancakes",
            ingredients=["2 cups flour", "1 egg", "1 cup milk", "2 tbsp sugar"],
            ner=["flour", "egg", "milk", "sugar"],
        ),
        RawRecipe(
            title="Omelette",
            ingredients=["3 eggs", "1/2 cup milk", "1 pinch salt"],
            ner=["egg", "milk", "salt"],
        ),
        RawRecipe(
            title="Sugar Cookies",
            ingredients=["2 cups flour", "1 cup sugar", "2 eggs", "1 cup butter"],
            ner=["flour", "sugar", "egg", "butter"],
        ),
        RawRecipe(
            title="Tomato Salad",
            ingredients=["2 tomatoes", "1 cucumber"],
            ner=["tomato", "cucumber"],
        ),
    ]
```

> Note: this imports `RawRecipe` (Task 2). Run the fixture-consuming tests only after Task 2 exists. The type test in Step 2 does not use the fixture.

- [ ] **Step 5: Run the type test and commit**

Run: `uv run pytest tests/test_smoke.py -v`
Expected: PASS (all smoke tests).

```bash
git add pyproject.toml uv.lock pantrychef/common/types.py tests/conftest.py tests/test_smoke.py
git commit -m "feat(types): add Recipe, extend ScoredRecipe; add test fixture; add pandas"
```

---

## Task 2: RawRecipe schema + RecipeNLG loader

**Files:**
- Create: `pantrychef/data/schemas.py`
- Create: `pantrychef/data/loaders.py`
- Test: `tests/test_loaders.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_loaders.py`:

```python
from __future__ import annotations

import json

import pandas as pd

from pantrychef.data.loaders import load_recipenlg


def test_load_recipenlg_parses_json_columns(tmp_path) -> None:
    csv = tmp_path / "mini.csv"
    pd.DataFrame(
        [
            {
                "title": "Pancakes",
                "ingredients": json.dumps(["2 cups flour", "1 egg"]),
                "directions": json.dumps(["Mix.", "Cook."]),
                "link": "http://x",
                "source": "Gathered",
                "NER": json.dumps(["flour", "egg"]),
            }
        ]
    ).to_csv(csv, index=False)

    rows = list(load_recipenlg(csv))
    assert len(rows) == 1
    r = rows[0]
    assert r.title == "Pancakes"
    assert r.ingredients == ["2 cups flour", "1 egg"]
    assert r.ner == ["flour", "egg"]


def test_load_recipenlg_max_rows(tmp_path) -> None:
    csv = tmp_path / "mini.csv"
    pd.DataFrame(
        [{"title": f"R{i}", "ingredients": "[]", "directions": "[]", "NER": "[]"} for i in range(5)]
    ).to_csv(csv, index=False)
    assert len(list(load_recipenlg(csv, max_rows=2))) == 2
```

Run: `uv run pytest tests/test_loaders.py -v`
Expected: FAIL — `ModuleNotFoundError: pantrychef.data.loaders`.

- [ ] **Step 2: Implement schema**

Create `pantrychef/data/schemas.py`:

```python
"""Raw recipe schema (pre-cleaning), mirroring RecipeNLG columns."""

from __future__ import annotations

from pydantic import BaseModel


class RawRecipe(BaseModel):
    title: str = ""
    ingredients: list[str] = []
    directions: list[str] = []
    ner: list[str] = []
    link: str = ""
    source: str = ""
```

- [ ] **Step 3: Implement loader**

Create `pantrychef/data/loaders.py`:

```python
"""Read RecipeNLG (CSV) into RawRecipe records.

RecipeNLG stores `ingredients`, `directions`, and `NER` as JSON-encoded lists
inside CSV cells. We parse those defensively (bad/empty cells -> []).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from pantrychef.data.schemas import RawRecipe


def _parse_list(cell: object) -> list[str]:
    if isinstance(cell, list):
        return [str(x) for x in cell]
    if not isinstance(cell, str):
        return []
    try:
        val = json.loads(cell)
    except (json.JSONDecodeError, TypeError):
        return []
    return [str(x) for x in val] if isinstance(val, list) else []


def load_recipenlg(path: str | Path, max_rows: int | None = None) -> Iterator[RawRecipe]:
    df = pd.read_csv(path, nrows=max_rows, keep_default_na=False)
    for _, row in df.iterrows():
        yield RawRecipe(
            title=str(row.get("title", "")).strip(),
            ingredients=_parse_list(row.get("ingredients")),
            directions=_parse_list(row.get("directions")),
            ner=_parse_list(row.get("NER")),
            link=str(row.get("link", "")),
            source=str(row.get("source", "")),
        )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_loaders.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add pantrychef/data/schemas.py pantrychef/data/loaders.py tests/test_loaders.py
git commit -m "feat(data): RawRecipe schema + RecipeNLG CSV loader"
```

---

## Task 3: Text normalization

**Files:**
- Create: `pantrychef/ingredients/normalize.py`
- Test: `tests/test_normalize.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_normalize.py`:

```python
from pantrychef.ingredients.normalize import normalize, singularize


def test_normalize_lowercases_and_strips_parens_punct() -> None:
    assert normalize("All-Purpose Flour (sifted)!") == "all-purpose flour"


def test_normalize_collapses_whitespace() -> None:
    assert normalize("  olive   oil ") == "olive oil"


def test_singularize_common_plurals() -> None:
    assert singularize("eggs") == "egg"
    assert singularize("tomatoes") == "tomato"
    assert singularize("berries") == "berry"


def test_singularize_leaves_short_and_double_s() -> None:
    assert singularize("oil") == "oil"
    assert singularize("molasses") == "molasses"
```

Run: `uv run pytest tests/test_normalize.py -v`
Expected: FAIL — module not found.

- [ ] **Step 2: Implement**

Create `pantrychef/ingredients/normalize.py`:

```python
"""Deterministic text normalization for ingredient strings."""

from __future__ import annotations

import re
import unicodedata

_PARENS = re.compile(r"\([^)]*\)")
_KEEP = re.compile(r"[^a-z0-9/.\-\s]")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = _PARENS.sub(" ", text)
    text = _KEEP.sub(" ", text)
    return _WS.sub(" ", text).strip()


def singularize(word: str) -> str:
    if len(word) <= 3:
        return word
    if word.endswith("ss"):
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "ses", "xes", "zes")):
        return word[:-2]
    if word.endswith("oes"):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_normalize.py -v`
Expected: PASS (4 tests).

- [ ] **Step 4: Commit**

```bash
git add pantrychef/ingredients/normalize.py tests/test_normalize.py
git commit -m "feat(ingredients): text normalize + lightweight singularize"
```

---

## Task 4: Quantity + unit parsing (rules)

**Files:**
- Create: `pantrychef/ingredients/units.py`
- Test: `tests/test_units.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_units.py`:

```python
from pantrychef.ingredients.units import parse_quantity, parse_unit, replace_unicode_fractions


def test_replace_unicode_fractions() -> None:
    assert replace_unicode_fractions("½ cup").strip().startswith("1/2")


def test_parse_quantity_integer_decimal_fraction() -> None:
    assert parse_quantity("2 cups flour")[0] == 2.0
    assert parse_quantity("1.5 cups flour")[0] == 1.5
    assert parse_quantity("1/2 cup milk")[0] == 0.5
    assert parse_quantity("1 1/2 cups sugar")[0] == 1.5


def test_parse_quantity_unicode_and_range() -> None:
    assert parse_quantity("½ cup milk")[0] == 0.5
    assert parse_quantity("1 to 2 cups water")[0] == 1.0
    assert parse_quantity("1-2 cups water")[0] == 1.0


def test_parse_quantity_none_when_absent() -> None:
    qty, rest = parse_quantity("pinch of salt")
    assert qty is None
    assert "salt" in rest


def test_parse_unit_known_and_unknown() -> None:
    assert parse_unit("cups flour") == ("cup", "flour")
    assert parse_unit("tbsp sugar") == ("tbsp", "sugar")
    assert parse_unit("flour") == (None, "flour")
```

Run: `uv run pytest tests/test_units.py -v`
Expected: FAIL — module not found.

- [ ] **Step 2: Implement**

Create `pantrychef/ingredients/units.py`:

```python
"""Rule-based quantity and unit extraction.

Design choices (documented in docs/DECISIONS later if revisited):
- Ranges ("1-2", "1 to 2") take the FIRST number for determinism.
- Mixed numbers ("1 1/2") and unicode fractions ("½") are supported.
"""

from __future__ import annotations

import re
from fractions import Fraction

UNICODE_FRACTIONS = {
    "½": "1/2", "⅓": "1/3", "⅔": "2/3", "¼": "1/4", "¾": "3/4",
    "⅕": "1/5", "⅖": "2/5", "⅗": "3/5", "⅘": "4/5", "⅙": "1/6",
    "⅛": "1/8", "⅜": "3/8", "⅝": "5/8", "⅞": "7/8",
}

UNITS = {
    "cup": "cup", "cups": "cup", "c": "cup",
    "tablespoon": "tbsp", "tablespoons": "tbsp", "tbsp": "tbsp", "tbs": "tbsp", "tbsps": "tbsp",
    "teaspoon": "tsp", "teaspoons": "tsp", "tsp": "tsp", "tsps": "tsp",
    "gram": "g", "grams": "g", "g": "g", "kg": "kg", "kilogram": "kg", "kilograms": "kg",
    "ounce": "oz", "ounces": "oz", "oz": "oz",
    "pound": "lb", "pounds": "lb", "lb": "lb", "lbs": "lb",
    "milliliter": "ml", "milliliters": "ml", "ml": "ml",
    "liter": "l", "liters": "l", "litre": "l", "litres": "l", "l": "l",
    "pinch": "pinch", "dash": "dash", "clove": "clove", "cloves": "clove",
    "can": "can", "cans": "can", "package": "package", "packages": "package", "pkg": "package",
    "slice": "slice", "slices": "slice", "stick": "stick", "sticks": "stick",
    "quart": "quart", "quarts": "quart", "pint": "pint", "pints": "pint",
}

_NUM = re.compile(r"^\s*(\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)")
_RANGE_TAIL = re.compile(r"^\s*(?:-|to)\s*(?:\d+\.\d+|\d+/\d+|\d+)")


def replace_unicode_fractions(text: str) -> str:
    for u, a in UNICODE_FRACTIONS.items():
        text = text.replace(u, " " + a + " ")
    return text


def _to_float(token: str) -> float:
    token = token.strip()
    if " " in token:  # mixed number "1 1/2"
        whole, frac = token.split(maxsplit=1)
        return float(whole) + float(Fraction(frac))
    if "/" in token:
        return float(Fraction(token))
    return float(token)


def parse_quantity(text: str) -> tuple[float | None, str]:
    text = replace_unicode_fractions(text)
    m = _NUM.match(text)
    if not m:
        return None, text.strip()
    qty = _to_float(m.group(1))
    rest = text[m.end() :]
    rest = _RANGE_TAIL.sub("", rest)
    return qty, rest.strip()


def parse_unit(text: str) -> tuple[str | None, str]:
    parts = text.strip().split()
    if not parts:
        return None, ""
    first = parts[0].lower().rstrip(".")
    if first in UNITS:
        return UNITS[first], " ".join(parts[1:]).strip()
    return None, text.strip()
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_units.py -v`
Expected: PASS (5 tests).

- [ ] **Step 4: Commit**

```bash
git add pantrychef/ingredients/units.py tests/test_units.py
git commit -m "feat(ingredients): rule-based quantity + unit parsing"
```

---

## Task 5: Canonical vocabulary builder

**Files:**
- Create: `pantrychef/ingredients/vocab.py`
- Test: `tests/test_vocab.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_vocab.py`:

```python
from pantrychef.ingredients.vocab import (
    build_vocabulary,
    load_vocabulary,
    save_vocabulary,
)


def test_build_vocabulary_min_count(sample_raws) -> None:
    vocab = build_vocabulary(sample_raws, min_count=2)
    # flour, egg, milk, sugar each appear >= 2 times across NER columns
    assert "flour" in vocab and "egg" in vocab and "milk" in vocab and "sugar" in vocab
    # salt/butter/tomato/cucumber appear once -> excluded
    assert "salt" not in vocab and "tomato" not in vocab


def test_build_vocabulary_keeps_singletons_when_min_one(sample_raws) -> None:
    vocab = build_vocabulary(sample_raws, min_count=1)
    assert {"flour", "egg", "milk", "sugar", "salt", "butter", "tomato", "cucumber"} <= set(vocab)


def test_build_vocabulary_sorted_longest_first(sample_raws) -> None:
    vocab = build_vocabulary(sample_raws, min_count=1)
    word_counts = [len(v.split()) for v in vocab]
    assert word_counts == sorted(word_counts, reverse=True)


def test_save_and_load_roundtrip(tmp_path, sample_raws) -> None:
    vocab = build_vocabulary(sample_raws, min_count=1)
    p = tmp_path / "vocab.json"
    save_vocabulary(vocab, p)
    assert load_vocabulary(p) == vocab
```

Run: `uv run pytest tests/test_vocab.py -v`
Expected: FAIL — module not found.

- [ ] **Step 2: Implement**

Create `pantrychef/ingredients/vocab.py`:

```python
"""Canonical ingredient vocabulary, built from RecipeNLG's NER entities.

The vocabulary is the bridge between messy free text and a controlled set of
canonical ingredients. Sorted longest-phrase-first so the parser can do
longest-match.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from pantrychef.data.schemas import RawRecipe
from pantrychef.ingredients.normalize import normalize, singularize


def _canon_token(entity: str) -> str:
    return " ".join(singularize(t) for t in normalize(entity).split()).strip()


def build_vocabulary(recipes: Iterable[RawRecipe], min_count: int = 2) -> list[str]:
    counts: Counter[str] = Counter()
    for r in recipes:
        for ent in r.ner:
            c = _canon_token(ent)
            if c:
                counts[c] += 1
    vocab = [w for w, n in counts.items() if n >= min_count]
    vocab.sort(key=lambda w: (-len(w.split()), -len(w), w))
    return vocab


def save_vocabulary(vocab: list[str], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")


def load_vocabulary(path: str | Path) -> list[str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_vocab.py -v`
Expected: PASS (4 tests).

- [ ] **Step 4: Commit**

```bash
git add pantrychef/ingredients/vocab.py tests/test_vocab.py
git commit -m "feat(ingredients): canonical vocabulary builder from NER"
```

---

## Task 6: The ingredient parser

**Files:**
- Create: `pantrychef/ingredients/parser.py`
- Test: `tests/test_parser.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_parser.py`:

```python
from pantrychef.ingredients.parser import match_canonical, parse

VOCAB = ["all purpose flour", "flour", "olive oil", "egg", "milk", "sugar", "salt"]


def test_match_canonical_longest_first() -> None:
    assert match_canonical("2 cups all-purpose flour", VOCAB) == "all purpose flour"
    assert match_canonical("plain flour", VOCAB) == "flour"


def test_match_canonical_none_when_absent() -> None:
    assert match_canonical("dragonfruit", VOCAB) is None


def test_parse_full_line() -> None:
    pi = parse("2 cups all-purpose flour, sifted", VOCAB)
    assert pi.quantity == 2.0
    assert pi.unit == "cup"
    assert pi.canonical == "all purpose flour"
    assert pi.modifier == "sifted"


def test_parse_plural_and_unicode() -> None:
    pi = parse("3 eggs", VOCAB)
    assert pi.quantity == 3.0
    assert pi.canonical == "egg"

    pi2 = parse("½ cup milk", VOCAB)
    assert pi2.quantity == 0.5
    assert pi2.unit == "cup"
    assert pi2.canonical == "milk"


def test_parse_without_vocab_returns_none_canonical() -> None:
    pi = parse("2 cups flour")
    assert pi.quantity == 2.0
    assert pi.unit == "cup"
    assert pi.canonical is None
```

Run: `uv run pytest tests/test_parser.py -v`
Expected: FAIL — module not found.

- [ ] **Step 2: Implement**

Create `pantrychef/ingredients/parser.py`:

```python
"""Hybrid ingredient parser: rules for quantity/unit, vocabulary longest-match
for the canonical ingredient.

raw line --> ParsedIngredient(quantity, unit, canonical, modifier)
"""

from __future__ import annotations

from pantrychef.common.types import ParsedIngredient
from pantrychef.ingredients.normalize import normalize, singularize
from pantrychef.ingredients.units import parse_quantity, parse_unit


def _canon_phrase(text: str) -> str:
    return " ".join(singularize(t) for t in normalize(text).split())


def match_canonical(phrase: str, vocab: list[str]) -> str | None:
    tokens = set(_canon_phrase(phrase).split())
    if not tokens:
        return None
    # vocab is pre-sorted longest-phrase-first, so the first full match wins.
    for entry in vocab:
        words = entry.split()
        if words and all(w in tokens for w in words):
            return entry
    return None


def parse(raw: str, vocab: list[str] | None = None) -> ParsedIngredient:
    vocab = vocab or []
    body = raw
    modifier: str | None = None
    if "," in raw:
        body, mod = raw.split(",", 1)
        modifier = mod.strip() or None
    qty, rest = parse_quantity(body)
    unit, rest = parse_unit(rest)
    canonical = match_canonical(rest, vocab) if vocab else None
    return ParsedIngredient(
        raw=raw.strip(),
        canonical=canonical,
        quantity=qty,
        unit=unit,
        modifier=modifier,
    )
```

> Note: `match_canonical` is called with the post-quantity/unit remainder (it still re-normalizes internally, which is harmless and keeps it usable standalone, as the tests do).

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_parser.py -v`
Expected: PASS (5 tests).

- [ ] **Step 4: Commit**

```bash
git add pantrychef/ingredients/parser.py tests/test_parser.py
git commit -m "feat(ingredients): hybrid ingredient parser"
```

---

## Task 7: Recipe cleaning + JSONL store

**Files:**
- Create: `pantrychef/data/clean.py`
- Create: `pantrychef/data/store.py`
- Test: `tests/test_clean_store.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_clean_store.py`:

```python
from pantrychef.data.clean import clean_recipes, to_recipe
from pantrychef.data.store import load_recipes, save_recipes
from pantrychef.ingredients.vocab import build_vocabulary


def test_to_recipe_builds_canonical_set(sample_raws) -> None:
    vocab = build_vocabulary(sample_raws, min_count=1)
    recipe = to_recipe(sample_raws[0], "r0", vocab)
    assert recipe.recipe_id == "r0"
    assert recipe.title == "Pancakes"
    assert set(recipe.canonical) == {"flour", "egg", "milk", "sugar"}


def test_clean_recipes_dedups_titles_and_ids(sample_raws) -> None:
    vocab = build_vocabulary(sample_raws, min_count=1)
    dup = sample_raws + [sample_raws[0]]  # duplicate Pancakes
    cleaned = list(clean_recipes(dup, vocab))
    titles = [r.title for r in cleaned]
    assert titles.count("Pancakes") == 1
    assert [r.recipe_id for r in cleaned] == [f"r{i}" for i in range(len(cleaned))]


def test_store_roundtrip(tmp_path, sample_raws) -> None:
    vocab = build_vocabulary(sample_raws, min_count=1)
    cleaned = list(clean_recipes(sample_raws, vocab))
    p = tmp_path / "recipes.jsonl"
    save_recipes(cleaned, p)
    loaded = load_recipes(p)
    assert [r.recipe_id for r in loaded] == [r.recipe_id for r in cleaned]
    assert loaded[0].canonical == cleaned[0].canonical
```

Run: `uv run pytest tests/test_clean_store.py -v`
Expected: FAIL — modules not found.

- [ ] **Step 2: Implement clean**

Create `pantrychef/data/clean.py`:

```python
"""Turn RawRecipe records into cleaned Recipe objects with canonical sets.

Dedup is by normalized title (RecipeNLG is largely pre-deduplicated; this is a
cheap safeguard). recipe_id is a stable sequential id assigned at clean time.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from pantrychef.common.types import Recipe
from pantrychef.data.schemas import RawRecipe
from pantrychef.ingredients.parser import parse


def to_recipe(raw: RawRecipe, recipe_id: str, vocab: list[str]) -> Recipe:
    canonical: list[str] = []
    seen: set[str] = set()
    for line in raw.ingredients:
        pi = parse(line, vocab)
        if pi.canonical and pi.canonical not in seen:
            seen.add(pi.canonical)
            canonical.append(pi.canonical)
    return Recipe(
        recipe_id=recipe_id,
        title=raw.title,
        ingredients_raw=list(raw.ingredients),
        canonical=canonical,
    )


def clean_recipes(raws: Iterable[RawRecipe], vocab: list[str]) -> Iterator[Recipe]:
    seen_titles: set[str] = set()
    idx = 0
    for raw in raws:
        key = raw.title.strip().lower()
        if not key or key in seen_titles:
            continue
        seen_titles.add(key)
        yield to_recipe(raw, f"r{idx}", vocab)
        idx += 1
```

- [ ] **Step 3: Implement store**

Create `pantrychef/data/store.py`:

```python
"""Persist cleaned recipes as JSONL (one Recipe per line)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pantrychef.common.types import Recipe


def save_recipes(recipes: Iterable[Recipe], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in recipes:
            f.write(r.model_dump_json() + "\n")


def load_recipes(path: str | Path) -> list[Recipe]:
    out: list[Recipe] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Recipe.model_validate_json(line))
    return out
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_clean_store.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add pantrychef/data/clean.py pantrychef/data/store.py tests/test_clean_store.py
git commit -m "feat(data): recipe cleaning + JSONL store"
```

---

## Task 8: Inverted index + retrieval baseline

**Files:**
- Create: `pantrychef/retrieval/index.py`
- Create: `pantrychef/retrieval/baseline.py`
- Test: `tests/test_retrieval.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_retrieval.py`:

```python
from pantrychef.data.clean import clean_recipes
from pantrychef.ingredients.vocab import build_vocabulary
from pantrychef.retrieval.baseline import recommend
from pantrychef.retrieval.index import InvertedIndex


def _index(sample_raws):
    vocab = build_vocabulary(sample_raws, min_count=1)
    recipes = list(clean_recipes(sample_raws, vocab))
    return InvertedIndex.build(recipes)


def test_index_postings(sample_raws) -> None:
    idx = _index(sample_raws)
    # "flour" appears in Pancakes (r0) and Sugar Cookies (r2)
    assert len(idx.postings["flour"]) == 2


def test_recommend_ranks_full_match_first(sample_raws) -> None:
    idx = _index(sample_raws)
    ranked = recommend(idx, ["flour", "egg", "milk", "sugar"], k=5)
    assert ranked[0].title == "Pancakes"
    assert ranked[0].score == 1.0
    assert ranked[0].missing == []


def test_recommend_reports_missing(sample_raws) -> None:
    idx = _index(sample_raws)
    ranked = recommend(idx, ["flour", "sugar"], k=5)
    cookies = next(r for r in ranked if r.title == "Sugar Cookies")
    assert set(cookies.matched) == {"flour", "sugar"}
    assert set(cookies.missing) == {"egg", "butter"}


def test_recommend_canonicalizes_pantry(sample_raws) -> None:
    idx = _index(sample_raws)
    # "Eggs" (capitalized, plural) must canonicalize to "egg"
    ranked = recommend(idx, ["Eggs"], k=5)
    assert any(r.title == "Omelette" for r in ranked)
```

Run: `uv run pytest tests/test_retrieval.py -v`
Expected: FAIL — modules not found.

- [ ] **Step 2: Implement index**

Create `pantrychef/retrieval/index.py`:

```python
"""Inverted index: canonical ingredient -> set of recipe ids."""

from __future__ import annotations

from collections.abc import Iterable

from pantrychef.common.types import Recipe


class InvertedIndex:
    def __init__(self) -> None:
        self.postings: dict[str, set[str]] = {}
        self.recipes: dict[str, Recipe] = {}

    def add(self, recipe: Recipe) -> None:
        self.recipes[recipe.recipe_id] = recipe
        for ing in recipe.canonical:
            self.postings.setdefault(ing, set()).add(recipe.recipe_id)

    @classmethod
    def build(cls, recipes: Iterable[Recipe]) -> "InvertedIndex":
        idx = cls()
        for r in recipes:
            idx.add(r)
        return idx

    def candidates(self, pantry: Iterable[str]) -> set[str]:
        out: set[str] = set()
        for ing in pantry:
            out |= self.postings.get(ing, set())
        return out
```

- [ ] **Step 3: Implement baseline**

Create `pantrychef/retrieval/baseline.py`:

```python
"""Set-overlap retrieval baseline.

Score = coverage = |pantry ∩ recipe| / |recipe ingredients|. Rank by score,
then fewer missing ingredients, then recipe_id (deterministic).
"""

from __future__ import annotations

from collections.abc import Iterable

from pantrychef.common.types import ScoredRecipe
from pantrychef.ingredients.normalize import normalize, singularize
from pantrychef.retrieval.index import InvertedIndex


def _canon(item: str) -> str:
    return " ".join(singularize(t) for t in normalize(item).split())


def recommend(index: InvertedIndex, pantry: Iterable[str], k: int = 5) -> list[ScoredRecipe]:
    pantry_set = {c for c in (_canon(p) for p in pantry) if c}
    results: list[ScoredRecipe] = []
    for rid in index.candidates(pantry_set):
        recipe = index.recipes[rid]
        canon = set(recipe.canonical)
        if not canon:
            continue
        matched = sorted(pantry_set & canon)
        missing = sorted(canon - pantry_set)
        results.append(
            ScoredRecipe(
                recipe_id=rid,
                score=len(matched) / len(canon),
                title=recipe.title,
                matched=matched,
                missing=missing,
            )
        )
    results.sort(key=lambda r: (-r.score, len(r.missing), r.recipe_id))
    return results[:k]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_retrieval.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add pantrychef/retrieval/index.py pantrychef/retrieval/baseline.py tests/test_retrieval.py
git commit -m "feat(retrieval): inverted index + set-overlap baseline"
```

---

## Task 9: Evaluation harness

**Files:**
- Create: `pantrychef/eval/__init__.py`
- Create: `pantrychef/eval/parser_eval.py`
- Create: `pantrychef/eval/retrieval_eval.py`
- Test: `tests/test_eval.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval.py`:

```python
from pantrychef.data.clean import clean_recipes
from pantrychef.eval.parser_eval import evaluate_parser
from pantrychef.eval.retrieval_eval import evaluate_retrieval
from pantrychef.ingredients.vocab import build_vocabulary
from pantrychef.retrieval.index import InvertedIndex


def test_evaluate_parser_perfect_on_fixture(sample_raws) -> None:
    vocab = build_vocabulary(sample_raws, min_count=1)
    metrics = evaluate_parser(sample_raws, vocab)
    # Fixture is constructed so the parser recovers exactly the NER sets.
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_evaluate_parser_returns_counts(sample_raws) -> None:
    vocab = build_vocabulary(sample_raws, min_count=1)
    metrics = evaluate_parser(sample_raws, vocab)
    assert metrics["tp"] > 0
    assert set(metrics) == {"precision", "recall", "f1", "tp", "fp", "fn"}


def test_evaluate_retrieval_recall(sample_raws) -> None:
    vocab = build_vocabulary(sample_raws, min_count=1)
    recipes = list(clean_recipes(sample_raws, vocab))
    idx = InvertedIndex.build(recipes)
    metrics = evaluate_retrieval(idx, recipes, k=5, seed=42)
    assert 0.0 <= metrics["recall_at_k"] <= 1.0
    assert metrics["k"] == 5.0
    assert metrics["n"] >= 1.0
```

Run: `uv run pytest tests/test_eval.py -v`
Expected: FAIL — modules not found.

- [ ] **Step 2: Create the eval package init**

Create `pantrychef/eval/__init__.py`:

```python
"""Offline evaluation harness (Phase 1: parser P/R/F1, retrieval recall@k)."""
```

- [ ] **Step 3: Implement parser eval**

Create `pantrychef/eval/parser_eval.py`:

```python
"""Parser evaluation: predicted canonical set vs NER ground truth.

Micro-averaged precision/recall/F1 over the recipe corpus (set-level per
recipe, summed).
"""

from __future__ import annotations

from collections.abc import Iterable

from pantrychef.data.schemas import RawRecipe
from pantrychef.ingredients.parser import parse
from pantrychef.ingredients.vocab import _canon_token


def evaluate_parser(recipes: Iterable[RawRecipe], vocab: list[str]) -> dict[str, float]:
    tp = fp = fn = 0
    for r in recipes:
        gold = {_canon_token(e) for e in r.ner if _canon_token(e)}
        pred = {pi.canonical for line in r.ingredients if (pi := parse(line, vocab)).canonical}
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }
```

- [ ] **Step 4: Implement retrieval eval**

Create `pantrychef/eval/retrieval_eval.py`:

```python
"""Retrieval evaluation: leave-one-ingredient-out recall@k.

For each recipe with >=2 canonical ingredients, drop one at random and query
the index with the rest; count a hit if the source recipe is in the top-k.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from pantrychef.common.types import Recipe
from pantrychef.retrieval.baseline import recommend
from pantrychef.retrieval.index import InvertedIndex


def evaluate_retrieval(
    index: InvertedIndex, recipes: Sequence[Recipe], k: int = 10, seed: int = 42
) -> dict[str, float]:
    rng = random.Random(seed)
    hits = total = 0
    for r in recipes:
        if len(r.canonical) < 2:
            continue
        held = rng.choice(r.canonical)
        pantry = [c for c in r.canonical if c != held]
        ranked = recommend(index, pantry, k=k)
        total += 1
        if any(s.recipe_id == r.recipe_id for s in ranked):
            hits += 1
    return {
        "recall_at_k": hits / total if total else 0.0,
        "k": float(k),
        "n": float(total),
    }
```

- [ ] **Step 5: Run tests + commit**

Run: `uv run pytest tests/test_eval.py -v`
Expected: PASS (3 tests).

```bash
git add pantrychef/eval/ tests/test_eval.py
git commit -m "feat(eval): parser P/R/F1 + retrieval recall@k harness"
```

---

## Task 10: CLI `cook` + data pipeline script + docs/Makefile

**Files:**
- Modify: `pantrychef/cli.py`
- Modify: `scripts/download_data.py`
- Modify: `Makefile`
- Modify: `docs/DATASET.md`
- Test: `tests/test_cli_cook.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli_cook.py`:

```python
from pantrychef.cli import cook_command
from pantrychef.data.clean import clean_recipes
from pantrychef.data.store import save_recipes
from pantrychef.ingredients.vocab import build_vocabulary


def test_cook_command_outputs_ranked(tmp_path, sample_raws, capsys) -> None:
    vocab = build_vocabulary(sample_raws, min_count=1)
    recipes = list(clean_recipes(sample_raws, vocab))
    store = tmp_path / "recipes.jsonl"
    save_recipes(recipes, store)

    rc = cook_command(have="flour,egg,milk,sugar", k=3, recipes_path=str(store))
    out = capsys.readouterr().out

    assert rc == 0
    assert "Pancakes" in out
    assert "%" in out  # coverage percentage rendered


def test_cook_command_missing_store_errors(tmp_path, capsys) -> None:
    rc = cook_command(have="flour", k=3, recipes_path=str(tmp_path / "nope.jsonl"))
    out = capsys.readouterr().out
    assert rc == 1
    assert "not found" in out.lower()
```

Run: `uv run pytest tests/test_cli_cook.py -v`
Expected: FAIL — `cook_command` does not exist.

- [ ] **Step 2: Rewrite the CLI**

Replace the entire contents of `pantrychef/cli.py`:

```python
"""Command-line entrypoint.

`pantrychef cook --have eggs,flour,milk --k 5` ranks recipes by how much of
each recipe your pantry covers, listing what you're missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pantrychef import __version__
from pantrychef.config import get_settings
from pantrychef.data.store import load_recipes
from pantrychef.retrieval.baseline import recommend
from pantrychef.retrieval.index import InvertedIndex


def _default_recipes_path() -> str:
    return str(get_settings().processed_dir / "recipes.jsonl")


def cook_command(have: str, k: int, recipes_path: str | None = None) -> int:
    path = Path(recipes_path or _default_recipes_path())
    if not path.exists():
        print(f"Recipe store not found at {path}. Run `make data` first.")
        return 1
    pantry = [p.strip() for p in have.split(",") if p.strip()]
    index = InvertedIndex.build(load_recipes(path))
    ranked = recommend(index, pantry, k=k)
    if not ranked:
        print("No matching recipes. Try more/different ingredients.")
        return 0
    for r in ranked:
        miss = ", ".join(r.missing) if r.missing else "nothing — you can make this!"
        print(f"[{r.score * 100:5.1f}%] {r.title}  (missing: {miss})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pantrychef", description="Pantry-aware recipe finder.")
    parser.add_argument("--version", action="version", version=f"pantrychef {__version__}")
    sub = parser.add_subparsers(dest="command")

    cook = sub.add_parser("cook", help="Find recipes you can make.")
    cook.add_argument("--have", required=True, help="Comma-separated pantry ingredients.")
    cook.add_argument("--k", type=int, default=5, help="How many recipes to show.")
    cook.add_argument("--recipes", default=None, help="Path to recipes.jsonl.")

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.command == "cook":
        return cook_command(have=args.have, k=args.k, recipes_path=args.recipes)

    print(f"PantryChef v{__version__}. Try: pantrychef cook --have eggs,flour,milk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run the CLI test**

Run: `uv run pytest tests/test_cli_cook.py -v`
Expected: PASS (2 tests).

- [ ] **Step 4: Implement the data pipeline script**

Replace the entire contents of `scripts/download_data.py`:

```python
"""Download RecipeNLG and build the Phase-1 processed artifacts.

Manual step (RecipeNLG requires accepting terms): download
`full_dataset.csv` from https://recipenlg.cs.put.poznan.pl/ and place it at
data/raw/full_dataset.csv  (see docs/DATASET.md). Then run:

    uv run python scripts/download_data.py --max-rows 50000

Outputs:
    data/processed/vocab.json     canonical ingredient vocabulary
    data/processed/recipes.jsonl  cleaned recipes with canonical sets
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pantrychef.common import get_logger
from pantrychef.config import get_settings
from pantrychef.data.clean import clean_recipes
from pantrychef.data.loaders import load_recipenlg
from pantrychef.data.store import save_recipes
from pantrychef.ingredients.vocab import build_vocabulary, save_vocabulary

log = get_logger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--min-count", type=int, default=5)
    args = ap.parse_args()

    s = get_settings()
    raw_csv = s.raw_dir / "full_dataset.csv"
    if not raw_csv.exists():
        log.error("Missing %s — see docs/DATASET.md for the manual download.", raw_csv)
        return 1

    log.info("Building vocabulary (pass 1)...")
    vocab = build_vocabulary(load_recipenlg(raw_csv, max_rows=args.max_rows), min_count=args.min_count)
    save_vocabulary(vocab, s.processed_dir / "vocab.json")
    log.info("Vocabulary size: %d", len(vocab))

    log.info("Cleaning recipes (pass 2)...")
    recipes = clean_recipes(load_recipenlg(raw_csv, max_rows=args.max_rows), vocab)
    out = Path(s.processed_dir / "recipes.jsonl")
    save_recipes(recipes, out)
    log.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Wire the Makefile**

In `Makefile`, replace the `data:` and `eval:` target bodies:

```makefile
data:         ## download/prepare datasets (Phase 1)
	uv run python scripts/download_data.py

eval:         ## run Phase 1 evaluation harness
	uv run python -m pantrychef.eval
```

Then create `pantrychef/eval/__main__.py` so `make eval` works on real data:

```python
"""`python -m pantrychef.eval` — run Phase 1 eval on processed artifacts."""

from __future__ import annotations

from pantrychef.common import get_logger
from pantrychef.config import get_settings
from pantrychef.data.store import load_recipes
from pantrychef.eval.retrieval_eval import evaluate_retrieval
from pantrychef.ingredients.vocab import load_vocabulary
from pantrychef.retrieval.index import InvertedIndex

log = get_logger(__name__)


def main() -> int:
    s = get_settings()
    recipes_path = s.processed_dir / "recipes.jsonl"
    if not recipes_path.exists():
        log.error("No processed recipes at %s. Run `make data` first.", recipes_path)
        return 1
    # vocab loaded to confirm artifacts exist; parser eval needs raw NER and is
    # run from notebooks against the raw corpus (see docs/EVALUATION.md).
    load_vocabulary(s.processed_dir / "vocab.json")
    recipes = load_recipes(recipes_path)
    index = InvertedIndex.build(recipes)
    metrics = evaluate_retrieval(index, recipes, k=10)
    log.info("Retrieval recall@%d = %.3f (n=%d)", int(metrics["k"]), metrics["recall_at_k"], int(metrics["n"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Update DATASET.md preprocessing section**

In `docs/DATASET.md`, under the RecipeNLG "Preprocessing" line, replace `_tbd Phase 1_` with:

```markdown
- **Preprocessing (Phase 1):**
  1. Manual download `full_dataset.csv` → `data/raw/` (terms acceptance).
  2. `uv run python scripts/download_data.py --max-rows N --min-count 5`:
     - Pass 1 builds `data/processed/vocab.json` from the `NER` column
       (canonicalized, frequency-filtered at `min_count`).
     - Pass 2 writes `data/processed/recipes.jsonl` (cleaned, deduped by title,
       canonical ingredient sets).
  3. Record `N`, vocab size, and recipe count of each run here.
- **Splits:** assigned at training time (Phase 2+); Phase 1 retrieval eval is
  leave-one-ingredient-out over the full processed set (seed=42).
```

- [ ] **Step 7: Full test suite + commit**

Run: `uv run ruff check . && uv run ruff format --check . && uv run pytest`
Expected: All checks pass; all tests green.

```bash
git add pantrychef/cli.py scripts/download_data.py Makefile docs/DATASET.md \
        pantrychef/eval/__main__.py tests/test_cli_cook.py
git commit -m "feat(cli): cook command + data pipeline script + eval entrypoint"
```

- [ ] **Step 8: Tag the Phase 1 release**

```bash
git tag -a v0.1.0 -m "Phase 1: data foundation, ingredient parser, retrieval baseline"
```

Then update `README.md` phase badge `phase-0%20scaffold-lightgrey` → `phase-1%20foundation-blue` and `ROADMAP.md` Phase 1 status ⬜ → ✅, and record real numbers in `docs/EVALUATION.md`. Commit:

```bash
git add README.md ROADMAP.md docs/EVALUATION.md
git commit -m "docs: mark Phase 1 complete, record baseline metrics"
```

---

## Self-Review

**Spec coverage (vs design spec §5 Phase 1):**
- Ingest RecipeNLG → Task 2 (loader) + Task 10 (pipeline). ✓
- Canonical ingredient vocab → Task 5. ✓
- Parser (qty/unit/canonical/modifier) → Tasks 3,4,6. ✓
- Set-overlap retrieval → Task 8. ✓
- Eval-harness skeleton → Task 9 (+ `__main__` in Task 10). ✓
- CLI (pantry→ranked recipes) → Task 10. ✓
- Parser P/R/F1 + retrieval recall@k metrics → Task 9. ✓
- Deferred (embeddings, ML ranking, images, nutrition) → not present. ✓

**Type consistency check:**
- `RawRecipe` fields (`title, ingredients, directions, ner, link, source`) — consistent across loader, conftest, vocab, clean, parser_eval. ✓
- `Recipe` (`recipe_id, title, ingredients_raw, canonical`) — consistent across clean, store, index, baseline, eval. ✓
- `ScoredRecipe` (`recipe_id, score, title, matched, missing`) — consistent across baseline, retrieval tests, CLI. ✓
- `ParsedIngredient` (`raw, canonical, quantity, unit, modifier`) — matches existing `common/types.py`; parser returns exactly these. ✓
- Function names stable: `build_vocabulary`, `_canon_token`, `match_canonical`, `parse`, `recommend`, `InvertedIndex.build/.candidates/.postings/.recipes`, `evaluate_parser`, `evaluate_retrieval`, `cook_command`. ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to Task N" in steps. All code steps show full code. ✓

**Known baseline limitations (intentional, documented, not gaps):** title-only dedup; range-quantity takes first number; subset-token canonical match can over-match rare phrasings — these are what the EVALUATION metrics expose and Phase 2 improves on.
