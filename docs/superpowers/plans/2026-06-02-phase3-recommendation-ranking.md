# Phase 3 — Recommendation & Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A learned reranker (`pantrychef/recommender/`) that reorders Phase-1 overlap candidates to recover a pantry-masked recipe, beating the overlap sort on recall@10 / MRR@10, with a substitution-feature ablation as the headline.

**Architecture:** Hold-out queries are built by masking ingredients from real recipes; the masked recipe is the single gold target. The P1 inverted index generates a shared candidate pool (capped, coverage-ordered); both the overlap baseline and the learned models rank the *same* pool. A pure `features.py` extracts tabular features (including a Phase-2 `sub_fill` signal via a `sub_lookup` callable); a numpy logistic `LinearRanker` and a LightGBM `LambdaMARTRanker` score them. Labels never use substitution → no circularity.

**Tech Stack:** Python 3.12, numpy (existing), LightGBM (new optional `[recommend]` extra), pydantic, pytest, ruff (line 100). Reuses `pantrychef.retrieval.index.InvertedIndex`, `pantrychef.common.types.{Recipe,ScoredRecipe}`, `pantrychef.substitution.substitute.Substitutor`.

---

## File Structure

**Create:**
- `pantrychef/recommender/config.py` — `RecConfig` dataclass (knobs: mask_fraction, min_recipe_len, candidate_cap, seed, sub_pool).
- `pantrychef/recommender/query_sim.py` — `QuerySim`, `is_train`, `make_query`.
- `pantrychef/recommender/features.py` — `FEATURE_NAMES`, `SUB_FEATURES`, `extract_features`, `to_matrix`.
- `pantrychef/recommender/rank.py` — `Ranker` protocol, `LinearRanker`, `LambdaMARTRanker`.
- `pantrychef/recommender/recommend.py` — `candidate_pool`, `rerank`.
- `pantrychef/recommender/train.py` — `build_dataset`, `train_ranker`.
- `pantrychef/eval/recommend_eval.py` — `recall_at_k`, `mrr_at_k`, `evaluate`.
- `pantrychef/eval/recommend_main.py` — CLI leaderboard.
- Tests mirroring each under `tests/recommender/` and `tests/eval/`.

**Modify:**
- `pyproject.toml` — add `[recommend]` optional extra (`lightgbm`).
- `docs/EVALUATION.md` — fill Phase 3 table.
- `docs/MODEL_CARD_substitution.md` or new `docs/MODEL_CARD_recommender.md` — recovery-scope wording, caveats.

**Config defaults (locked):** `mask_fraction=0.3`, `min_recipe_len=4`, `candidate_cap=200`, `seed=13`, `sub_pool=20`, `queries_per_recipe=1`.

---

### Task 1: Config + module skeleton + dependency

**Files:**
- Create: `pantrychef/recommender/config.py`
- Modify: `pyproject.toml`
- Modify: `pantrychef/recommender/__init__.py`
- Test: `tests/recommender/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/recommender/test_config.py
from pantrychef.recommender.config import RecConfig


def test_defaults():
    c = RecConfig()
    assert c.mask_fraction == 0.3
    assert c.min_recipe_len == 4
    assert c.candidate_cap == 200
    assert c.seed == 13
    assert c.sub_pool == 20
    assert c.queries_per_recipe == 1


def test_override():
    c = RecConfig(mask_fraction=0.5, candidate_cap=50)
    assert c.mask_fraction == 0.5
    assert c.candidate_cap == 50
    assert c.min_recipe_len == 4  # unchanged
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/recommender/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: pantrychef.recommender.config`

- [ ] **Step 3: Write minimal implementation**

```python
# pantrychef/recommender/config.py
"""Phase 3 reranker knobs. Defaults locked in the design spec (2026-06-02)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecConfig:
    mask_fraction: float = 0.3  # fraction of a recipe's ingredients hidden to form the pantry
    min_recipe_len: int = 4  # skip recipes shorter than this (need >=1 kept after masking)
    candidate_cap: int = 200  # max candidates per query, top by coverage (shared pool)
    seed: int = 13  # deterministic masking
    sub_pool: int = 20  # P2 substitutes fetched per missing ingredient for sub_fill
    queries_per_recipe: int = 1  # masked variants generated per train recipe
```

Ensure `pantrychef/recommender/__init__.py` exists (it does, empty stub — leave as-is).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/recommender/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Add the optional dependency**

In `pyproject.toml`, under `[project.optional-dependencies]`, add alongside the existing `substitution` extra:

```toml
recommend = ["lightgbm>=4.0"]
```

Run: `uv sync --extra recommend` — Expected: lightgbm resolves/installs.

- [ ] **Step 6: Commit**

```bash
git add pantrychef/recommender/config.py tests/recommender/test_config.py pyproject.toml uv.lock
git commit -m "feat(recommender): Phase 3 config + lightgbm extra"
```

---

### Task 2: Query simulation (split + masking)

**Files:**
- Create: `pantrychef/recommender/query_sim.py`
- Test: `tests/recommender/test_query_sim.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/recommender/test_query_sim.py
import random

from pantrychef.common.types import Recipe
from pantrychef.recommender.config import RecConfig
from pantrychef.recommender.query_sim import QuerySim, is_train, make_query


def _recipe(rid: str, ings: list[str]) -> Recipe:
    return Recipe(recipe_id=rid, title=rid, canonical=ings)


def test_is_train_deterministic_and_split():
    # same id -> same bucket every call
    assert is_train("abc") == is_train("abc")
    # ~80/20 over many ids, all bool
    ids = [str(i) for i in range(2000)]
    frac = sum(is_train(i) for i in ids) / len(ids)
    assert 0.7 < frac < 0.9


def test_make_query_masks_fraction_and_sets_gold():
    r = _recipe("r1", ["flour", "egg", "milk", "sugar", "butter"])  # len 5
    q = make_query(r, RecConfig(mask_fraction=0.4, seed=1), random.Random(1))
    assert isinstance(q, QuerySim)
    assert q.gold_id == "r1"
    # 0.4 * 5 = 2 hidden, 3 kept; partition is exact and disjoint
    assert len(q.hidden) == 2
    assert len(q.pantry) == 3
    assert set(q.pantry) | set(q.hidden) == set(r.canonical)
    assert not (set(q.pantry) & set(q.hidden))


def test_make_query_min_one_hidden():
    r = _recipe("r2", ["a", "b", "c", "d"])  # 0.3*4 = 1.2 -> round 1
    q = make_query(r, RecConfig(mask_fraction=0.3, seed=2), random.Random(2))
    assert len(q.hidden) == 1


def test_make_query_skips_short_recipe():
    r = _recipe("r3", ["a", "b", "c"])  # < min_recipe_len 4
    assert make_query(r, RecConfig(), random.Random(3)) is None


def test_make_query_dedupes_canonical():
    r = _recipe("r4", ["egg", "egg", "milk", "flour", "sugar"])  # 4 unique
    q = make_query(r, RecConfig(mask_fraction=0.25, seed=4), random.Random(4))
    assert q is not None
    assert len(set(q.pantry) | set(q.hidden)) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/recommender/test_query_sim.py -v`
Expected: FAIL — `ModuleNotFoundError: pantrychef.recommender.query_sim`

- [ ] **Step 3: Write minimal implementation**

```python
# pantrychef/recommender/query_sim.py
"""Leave-ingredients-out query construction + deterministic train/test split.

A query is built by hiding a fraction of a real recipe's canonical ingredients;
the masked recipe is the single gold-relevant target (recovery task). The split
is by recipe_id hash so a recipe is never in both train and test.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from pantrychef.common.types import Recipe
from pantrychef.recommender.config import RecConfig


@dataclass(frozen=True)
class QuerySim:
    pantry: tuple[str, ...]  # kept ingredients (the simulated pantry)
    hidden: tuple[str, ...]  # masked-out ingredients
    gold_id: str  # recipe_id of the recipe these came from


def is_train(recipe_id: str, train_frac: int = 80) -> bool:
    """Deterministic ~80/20 split by recipe_id hash. No global RNG state."""
    h = int(hashlib.md5(recipe_id.encode()).hexdigest(), 16)
    return (h % 100) < train_frac


def make_query(recipe: Recipe, cfg: RecConfig, rng: random.Random) -> QuerySim | None:
    """Mask cfg.mask_fraction of a recipe's unique canonical ingredients.

    Returns None for recipes shorter than cfg.min_recipe_len. At least one
    ingredient is always hidden and at least one always kept.
    """
    canon = sorted(set(recipe.canonical))  # dedupe + deterministic order
    if len(canon) < cfg.min_recipe_len:
        return None
    n_hidden = max(1, round(cfg.mask_fraction * len(canon)))
    n_hidden = min(n_hidden, len(canon) - 1)  # always keep >=1
    hidden = sorted(rng.sample(canon, n_hidden))
    pantry = sorted(set(canon) - set(hidden))
    return QuerySim(pantry=tuple(pantry), hidden=tuple(hidden), gold_id=recipe.recipe_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/recommender/test_query_sim.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add pantrychef/recommender/query_sim.py tests/recommender/test_query_sim.py
git commit -m "feat(recommender): held-out query simulation + recipe_id split"
```

---

### Task 3: Feature extraction (pure)

**Files:**
- Create: `pantrychef/recommender/features.py`
- Test: `tests/recommender/test_features.py`

`sub_lookup` is a callable `missing_ingredient -> {substitute_name: score}` so this module never imports the heavy Phase-2 model. `sub_fill_max` = best score of any pantry item appearing as a substitute for any missing ingredient; `sub_fill_mean` = mean over missing ingredients of their best pantry-substitute score (0 if none).

- [ ] **Step 1: Write the failing test**

```python
# tests/recommender/test_features.py
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
    # butter best pantry sub = margarine 0.9; sugar has no pantry sub -> 0
    assert f["sub_fill_max"] == 0.9
    assert f["sub_fill_mean"] == (0.9 + 0.0) / 2


def test_dietary_ok_flag():
    pantry = {"tofu"}
    recipe = _recipe("r", ["tofu", "rice", "soy sauce", "ginger"])
    # no diet implied -> dietary_ok defaults to 1.0
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
    feats = [{n: 1.0 for n in FEATURE_NAMES}]
    m = to_matrix(feats, cols)
    assert m.shape == (1, len(FEATURE_NAMES) - len(SUB_FEATURES))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/recommender/test_features.py -v`
Expected: FAIL — `ModuleNotFoundError: pantrychef.recommender.features`

- [ ] **Step 3: Write minimal implementation**

```python
# pantrychef/recommender/features.py
"""Pure per-(query, candidate) feature extraction for the reranker.

No I/O, no model imports. The Phase-2 substitution signal enters via a
`sub_lookup` callable (missing_ingredient -> {substitute_name: score}) so this
module stays cheap to unit-test and free of circular label dependence.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from pantrychef.common.types import Recipe

FEATURE_NAMES: tuple[str, ...] = (
    "coverage",
    "match_frac_pantry",
    "n_matched",
    "n_missing",
    "recipe_len",
    "sub_fill_max",
    "sub_fill_mean",
    "dietary_ok",
)
SUB_FEATURES: frozenset[str] = frozenset({"sub_fill_max", "sub_fill_mean"})

SubLookup = Callable[[str], Mapping[str, float]] | None


def extract_features(
    pantry: set[str],
    recipe: Recipe,
    sub_lookup: SubLookup = None,
) -> dict[str, float]:
    """Compute the feature dict for one pantry/candidate-recipe pair.

    `dietary_ok` is a placeholder 1.0 here (no diet constraint passed in the
    recovery task); it is wired to DietTagger at the call site when a diet is
    supplied. Kept in the vector so the column exists for later phases.
    """
    canon = set(recipe.canonical)
    recipe_len = len(canon)
    matched = pantry & canon
    missing = canon - pantry
    n_matched = len(matched)

    coverage = n_matched / recipe_len if recipe_len else 0.0
    match_frac_pantry = n_matched / len(pantry) if pantry else 0.0

    sub_max = 0.0
    per_missing_best: list[float] = []
    if sub_lookup is not None and missing:
        for m in missing:
            subs = sub_lookup(m)
            best = max((subs[p] for p in pantry if p in subs), default=0.0)
            per_missing_best.append(best)
            sub_max = max(sub_max, best)
    sub_mean = (sum(per_missing_best) / len(per_missing_best)) if per_missing_best else 0.0

    return {
        "coverage": coverage,
        "match_frac_pantry": match_frac_pantry,
        "n_matched": float(n_matched),
        "n_missing": float(len(missing)),
        "recipe_len": float(recipe_len),
        "sub_fill_max": float(sub_max),
        "sub_fill_mean": float(sub_mean),
        "dietary_ok": 1.0,
    }


def to_matrix(feats: list[dict[str, float]], columns: tuple[str, ...] | list[str]) -> np.ndarray:
    """Stack feature dicts into a (n, len(columns)) float array in column order."""
    if not feats:
        return np.empty((0, len(columns)), dtype=float)
    return np.array([[f[c] for c in columns] for f in feats], dtype=float)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/recommender/test_features.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add pantrychef/recommender/features.py tests/recommender/test_features.py
git commit -m "feat(recommender): pure feature extractor + sub_fill via lookup"
```

---

### Task 4: LinearRanker (numpy logistic, train-only scaling)

**Files:**
- Create: `pantrychef/recommender/rank.py`
- Test: `tests/recommender/test_rank_linear.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/recommender/test_rank_linear.py
import numpy as np

from pantrychef.recommender.rank import LinearRanker


def test_linear_learns_separable_signal():
    rng = np.random.default_rng(0)
    # feature 0 is the signal: positives have higher values
    pos = np.column_stack([rng.normal(3, 0.5, 100), rng.normal(0, 1, 100)])
    neg = np.column_stack([rng.normal(-3, 0.5, 100), rng.normal(0, 1, 100)])
    X = np.vstack([pos, neg])
    y = np.array([1] * 100 + [0] * 100)
    m = LinearRanker(seed=0).fit(X, y, groups=[200])
    scores = m.score(X)
    # positives should score higher on average
    assert scores[:100].mean() > scores[100:].mean()


def test_linear_scaler_is_train_only():
    X = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    y = np.array([0, 1, 1])
    m = LinearRanker(seed=0).fit(X, y, groups=[3])
    # mean/std captured from train; stored for reuse
    assert m.mean_ is not None and m.std_ is not None
    assert np.allclose(m.mean_, X.mean(axis=0))


def test_score_length_matches_rows():
    X = np.random.default_rng(1).normal(size=(7, 3))
    m = LinearRanker(seed=0).fit(X, np.array([0, 1, 0, 1, 1, 0, 1]), groups=[7])
    assert len(m.score(X)) == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/recommender/test_rank_linear.py -v`
Expected: FAIL — `ImportError: cannot import name 'LinearRanker'`

- [ ] **Step 3: Write minimal implementation**

```python
# pantrychef/recommender/rank.py
"""Reranker model wrappers behind one interface.

LinearRanker: numpy logistic regression (pointwise baseline), standardized with
TRAIN-only mean/std. LambdaMARTRanker: LightGBM lambdarank flagship. Both expose
fit(X, y, groups) / score(X); `groups` is the per-query candidate counts (used by
lambdarank; ignored by the linear model).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Ranker(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray, groups: list[int]) -> "Ranker": ...
    def score(self, X: np.ndarray) -> np.ndarray: ...


class LinearRanker:
    """Logistic regression via gradient descent. Ordering uses the raw logit."""

    def __init__(self, lr: float = 0.1, epochs: int = 500, seed: int = 13) -> None:
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.w_: np.ndarray | None = None
        self.b_: float = 0.0
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit(self, X: np.ndarray, y: np.ndarray, groups: list[int]) -> "LinearRanker":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0  # guard constant columns
        Xs = self._standardize(X)
        rng = np.random.default_rng(self.seed)
        n, d = Xs.shape
        self.w_ = rng.normal(0, 0.01, d)
        self.b_ = 0.0
        for _ in range(self.epochs):
            z = Xs @ self.w_ + self.b_
            p = 1.0 / (1.0 + np.exp(-z))
            err = p - y
            self.w_ -= self.lr * (Xs.T @ err) / n
            self.b_ -= self.lr * err.mean()
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.w_ is None or self.mean_ is None:
            raise RuntimeError("LinearRanker not fitted")
        return self._standardize(X) @ self.w_ + self.b_
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/recommender/test_rank_linear.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add pantrychef/recommender/rank.py tests/recommender/test_rank_linear.py
git commit -m "feat(recommender): numpy logistic LinearRanker (train-only scaling)"
```

---

### Task 5: LambdaMARTRanker (LightGBM)

**Files:**
- Modify: `pantrychef/recommender/rank.py`
- Test: `tests/recommender/test_rank_lambdamart.py`

- [ ] **Step 1: Write the failing test** (skips cleanly if lightgbm absent)

```python
# tests/recommender/test_rank_lambdamart.py
import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm")

from pantrychef.recommender.rank import LambdaMARTRanker


def test_lambdamart_ranks_signal_above_noise():
    rng = np.random.default_rng(0)
    # build 20 groups of 5 candidates; in each, the high-signal row is the gold
    X_rows, y_rows, groups = [], [], []
    for _ in range(20):
        feats = rng.normal(0, 1, (5, 3))
        gold = rng.integers(0, 5)
        feats[gold, 0] += 5.0  # signal in feature 0
        labels = np.zeros(5)
        labels[gold] = 1
        X_rows.append(feats)
        y_rows.append(labels)
        groups.append(5)
    X = np.vstack(X_rows)
    y = np.concatenate(y_rows)
    m = LambdaMARTRanker(seed=0).fit(X, y, groups=groups)
    scores = m.score(X)
    # within the first group, the gold row should rank top
    first = scores[:5]
    assert int(np.argmax(first)) == int(np.argmax(y[:5]))


def test_lambdamart_unfitted_raises():
    with pytest.raises(RuntimeError):
        LambdaMARTRanker().score(np.zeros((1, 3)))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/recommender/test_rank_lambdamart.py -v`
Expected: FAIL — `ImportError: cannot import name 'LambdaMARTRanker'` (or skip if lightgbm missing — install via `uv sync --extra recommend`)

- [ ] **Step 3: Write minimal implementation** (append to `rank.py`)

```python
# append to pantrychef/recommender/rank.py


class LambdaMARTRanker:
    """LightGBM LambdaMART (objective='lambdarank'). Lazy import keeps core clean."""

    def __init__(self, num_leaves: int = 31, n_estimators: int = 200, seed: int = 13) -> None:
        self.num_leaves = num_leaves
        self.n_estimators = n_estimators
        self.seed = seed
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray, groups: list[int]) -> "LambdaMARTRanker":
        try:
            import lightgbm as lgb
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("LambdaMARTRanker needs the [recommend] extra: uv sync --extra recommend") from e
        self._model = lgb.LGBMRanker(
            objective="lambdarank",
            num_leaves=self.num_leaves,
            n_estimators=self.n_estimators,
            random_state=self.seed,
            min_child_samples=5,
            verbose=-1,
        )
        self._model.fit(X, y.astype(int), group=groups)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("LambdaMARTRanker not fitted")
        return np.asarray(self._model.predict(X))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/recommender/test_rank_lambdamart.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add pantrychef/recommender/rank.py tests/recommender/test_rank_lambdamart.py
git commit -m "feat(recommender): LightGBM LambdaMART flagship ranker"
```

---

### Task 6: Candidate pool + rerank glue

**Files:**
- Create: `pantrychef/recommender/recommend.py`
- Test: `tests/recommender/test_recommend.py`

`candidate_pool` produces the shared, coverage-ordered, capped pool. `rerank` applies a fitted model's scores to that pool (tie-break by recipe_id for determinism). The overlap baseline is exactly `candidate_pool` order.

- [ ] **Step 1: Write the failing test**

```python
# tests/recommender/test_recommend.py
import numpy as np

from pantrychef.common.types import Recipe
from pantrychef.recommender.recommend import candidate_pool, rerank
from pantrychef.retrieval.index import InvertedIndex


def _idx():
    recipes = [
        Recipe(recipe_id="a", title="A", canonical=["egg", "flour", "milk"]),
        Recipe(recipe_id="b", title="B", canonical=["egg", "sugar"]),
        Recipe(recipe_id="c", title="C", canonical=["beef", "onion"]),  # no overlap
    ]
    return InvertedIndex.build(recipes)


def test_candidate_pool_overlap_ordered():
    idx = _idx()
    pool = candidate_pool(idx, {"egg", "flour", "milk"}, cap=10)
    ids = [r.recipe_id for r in pool]
    assert "c" not in ids  # no overlap
    assert ids[0] == "a"  # full coverage (3/3) ranks above b (1/2)


def test_candidate_pool_respects_cap():
    idx = _idx()
    pool = candidate_pool(idx, {"egg"}, cap=1)
    assert len(pool) == 1


def test_rerank_uses_model_scores():
    idx = _idx()
    pantry = {"egg", "flour", "milk"}

    class FakeModel:
        # score b above a regardless of coverage
        def score(self, X):
            return np.array([1.0 if row[2] == 1.0 else 5.0 for row in X])  # n_matched col

    def feat_fn(p, recipe):
        from pantrychef.recommender.features import extract_features

        return extract_features(p, recipe, sub_lookup=None)

    ranked = rerank(idx, pantry, FakeModel(), feat_fn, k=2, cap=10)
    # b has n_matched=1 -> score 5; a has n_matched=3 -> score 1; so b first
    assert ranked[0].recipe_id == "b"
    assert ranked[0].matched and isinstance(ranked[0].score, float)


def test_rerank_empty_pantry_returns_empty():
    idx = _idx()

    class FakeModel:
        def score(self, X):
            return np.zeros(len(X))

    out = rerank(idx, set(), FakeModel(), lambda p, r: {}, k=5, cap=10)
    assert out == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/recommender/test_recommend.py -v`
Expected: FAIL — `ModuleNotFoundError: pantrychef.recommender.recommend`

- [ ] **Step 3: Write minimal implementation**

```python
# pantrychef/recommender/recommend.py
"""Candidate generation (shared pool) + learned rerank.

candidate_pool: all recipes sharing >=1 pantry ingredient, ordered by coverage
(the P1 baseline order), capped. Both the overlap baseline and every learned
model rank this SAME pool, so the comparison isolates the ordering function.
"""

from __future__ import annotations

from collections.abc import Callable

from pantrychef.common.types import Recipe, ScoredRecipe
from pantrychef.recommender.features import FEATURE_NAMES, to_matrix
from pantrychef.retrieval.index import InvertedIndex

FeatureFn = Callable[[set, Recipe], dict]


def candidate_pool(index: InvertedIndex, pantry: set[str], cap: int) -> list[Recipe]:
    """Overlap candidates ordered by coverage desc, then fewer-missing, then id."""
    scored: list[tuple[float, int, str, Recipe]] = []
    for rid in index.candidates(pantry):
        recipe = index.recipes[rid]
        canon = set(recipe.canonical)
        if not canon:
            continue
        matched = len(pantry & canon)
        coverage = matched / len(canon)
        scored.append((coverage, len(canon - pantry), rid, recipe))
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [t[3] for t in scored[:cap]]


def rerank(
    index: InvertedIndex,
    pantry: set[str],
    model,
    feature_fn: FeatureFn,
    k: int = 10,
    cap: int = 200,
    columns: tuple[str, ...] = FEATURE_NAMES,
) -> list[ScoredRecipe]:
    """Rerank the shared candidate pool by model score (tie-break recipe_id)."""
    pool = candidate_pool(index, pantry, cap)
    if not pool:
        return []
    feats = [feature_fn(pantry, r) for r in pool]
    scores = model.score(to_matrix(feats, columns))
    order = sorted(range(len(pool)), key=lambda i: (-float(scores[i]), pool[i].recipe_id))
    out: list[ScoredRecipe] = []
    for i in order[:k]:
        r = pool[i]
        canon = set(r.canonical)
        out.append(
            ScoredRecipe(
                recipe_id=r.recipe_id,
                score=float(scores[i]),
                title=r.title,
                matched=sorted(pantry & canon),
                missing=sorted(canon - pantry),
            )
        )
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/recommender/test_recommend.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add pantrychef/recommender/recommend.py tests/recommender/test_recommend.py
git commit -m "feat(recommender): shared candidate pool + learned rerank glue"
```

---

### Task 7: Training dataset builder

**Files:**
- Create: `pantrychef/recommender/train.py`
- Test: `tests/recommender/test_train.py`

`build_dataset` turns train-split recipes into (X, y, groups). A query is dropped (logged) when its candidate pool is empty or does not contain the gold recipe (no positive → useless lambdarank group). The dropped count feeds the candidate-recall ceiling reported at eval.

- [ ] **Step 1: Write the failing test**

```python
# tests/recommender/test_train.py
import numpy as np

from pantrychef.common.types import Recipe
from pantrychef.recommender.config import RecConfig
from pantrychef.recommender.features import FEATURE_NAMES
from pantrychef.recommender.train import build_dataset
from pantrychef.retrieval.index import InvertedIndex


def _corpus():
    return [
        Recipe(recipe_id=str(i), title=str(i),
               canonical=["egg", "flour", "milk", "sugar", "butter"][: 4 + (i % 2)])
        for i in range(20)
    ]


def test_build_dataset_shapes_and_labels():
    corpus = _corpus()
    idx = InvertedIndex.build(corpus)
    X, y, groups, stats = build_dataset(
        corpus, idx, sub_lookup=None, cfg=RecConfig(seed=1), columns=FEATURE_NAMES
    )
    assert X.shape[1] == len(FEATURE_NAMES)
    assert X.shape[0] == len(y) == sum(groups)
    # every kept group has exactly one positive (the gold recipe)
    pos = 0
    start = 0
    for g in groups:
        assert y[start : start + g].sum() == 1
        pos += 1
        start += g
    assert stats["n_queries_kept"] == len(groups)
    assert "n_dropped_no_gold" in stats


def test_build_dataset_drops_when_gold_absent(monkeypatch):
    # a recipe whose ingredients are all unique -> still in its own pool, so
    # to force a drop we shrink the cap to 0
    corpus = _corpus()
    idx = InvertedIndex.build(corpus)
    X, y, groups, stats = build_dataset(
        corpus, idx, sub_lookup=None, cfg=RecConfig(seed=1, candidate_cap=0), columns=FEATURE_NAMES
    )
    assert len(groups) == 0
    assert stats["n_queries_kept"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/recommender/test_train.py -v`
Expected: FAIL — `ModuleNotFoundError: pantrychef.recommender.train`

- [ ] **Step 3: Write minimal implementation**

```python
# pantrychef/recommender/train.py
"""Build LightGBM-style (X, y, groups) training data from masked queries.

Only train-split recipes are used. Queries whose candidate pool is empty or
lacks the gold recipe are dropped (counted) — they carry no positive label.
"""

from __future__ import annotations

import random

import numpy as np

from pantrychef.common import get_logger
from pantrychef.common.types import Recipe
from pantrychef.recommender.config import RecConfig
from pantrychef.recommender.features import FEATURE_NAMES, extract_features, to_matrix
from pantrychef.recommender.query_sim import is_train, make_query
from pantrychef.recommender.recommend import candidate_pool

log = get_logger(__name__)


def build_dataset(
    recipes: list[Recipe],
    index,
    sub_lookup,
    cfg: RecConfig,
    columns: tuple[str, ...] = FEATURE_NAMES,
    train_only: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[int], dict[str, int]]:
    """Return (X, y, group_sizes, stats) over masked train-split queries."""
    rng = random.Random(cfg.seed)
    feats: list[dict[str, float]] = []
    labels: list[int] = []
    groups: list[int] = []
    n_total = n_no_pool = n_no_gold = 0

    for recipe in recipes:
        if train_only and not is_train(recipe.recipe_id):
            continue
        for _ in range(cfg.queries_per_recipe):
            q = make_query(recipe, cfg, rng)
            if q is None:
                continue
            n_total += 1
            pool = candidate_pool(index, set(q.pantry), cfg.candidate_cap)
            if not pool:
                n_no_pool += 1
                continue
            ids = [r.recipe_id for r in pool]
            if q.gold_id not in ids:
                n_no_gold += 1
                continue
            for r in pool:
                feats.append(extract_features(set(q.pantry), r, sub_lookup))
                labels.append(1 if r.recipe_id == q.gold_id else 0)
            groups.append(len(pool))

    stats = {
        "n_queries_total": n_total,
        "n_queries_kept": len(groups),
        "n_dropped_empty_pool": n_no_pool,
        "n_dropped_no_gold": n_no_gold,
    }
    log.info("build_dataset: %s", stats)
    X = to_matrix(feats, columns)
    return X, np.array(labels, dtype=int), groups, stats


def train_ranker(model, recipes, index, sub_lookup, cfg, columns=FEATURE_NAMES):
    """Build the dataset and fit `model` in place; returns (model, stats)."""
    X, y, groups, stats = build_dataset(recipes, index, sub_lookup, cfg, columns)
    model.fit(X, y, groups)
    return model, stats
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/recommender/test_train.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add pantrychef/recommender/train.py tests/recommender/test_train.py
git commit -m "feat(recommender): masked-query training dataset builder"
```

---

### Task 8: Evaluation metrics

**Files:**
- Create: `pantrychef/eval/recommend_eval.py`
- Test: `tests/eval/test_recommend_eval.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_recommend_eval.py
import numpy as np

from pantrychef.common.types import Recipe
from pantrychef.recommender.config import RecConfig
from pantrychef.recommender.features import FEATURE_NAMES
from pantrychef.recommender.recommend import rerank
from pantrychef.recommender.rank import LinearRanker
from pantrychef.recommender.train import train_ranker
from pantrychef.eval.recommend_eval import evaluate, mrr_at_k, recall_at_k
from pantrychef.retrieval.index import InvertedIndex


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], "b", 3) == 1.0
    assert recall_at_k(["a", "b", "c"], "b", 1) == 0.0
    assert recall_at_k(["a", "b", "c"], "z", 3) == 0.0


def test_mrr_at_k():
    assert mrr_at_k(["a", "b", "c"], "a", 10) == 1.0
    assert mrr_at_k(["a", "b", "c"], "b", 10) == 0.5
    assert mrr_at_k(["a", "b", "c"], "b", 1) == 0.0


def test_evaluate_reports_ceiling_and_conditional():
    corpus = [
        Recipe(recipe_id=str(i), title=str(i),
               canonical=["egg", "flour", "milk", "sugar", "butter", "salt"][: 4 + (i % 3)])
        for i in range(60)
    ]
    idx = InvertedIndex.build(corpus)
    cfg = RecConfig(seed=2)
    model, _ = train_ranker(LinearRanker(seed=0), corpus, idx, None, cfg)
    m = evaluate(model, corpus, idx, None, cfg, k=10, columns=FEATURE_NAMES)
    for key in ("recall@10", "mrr@10", "ceiling", "recall@10|in_pool", "n_queries", "n_in_pool"):
        assert key in m
    assert 0.0 <= m["ceiling"] <= 1.0
    # conditional recall >= overall recall (denominator excludes unreachable golds)
    assert m["recall@10|in_pool"] >= m["recall@10"] - 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_recommend_eval.py -v`
Expected: FAIL — `ModuleNotFoundError: pantrychef.eval.recommend_eval`

- [ ] **Step 3: Write minimal implementation**

```python
# pantrychef/eval/recommend_eval.py
"""Recovery-task metrics: recall@k, MRR@k, candidate ceiling, and the same
metrics conditional on the gold recipe being reachable in the candidate pool.

NDCG@k is intentionally omitted: with a single gold per query it is a monotone
transform of MRR (see design spec 2026-06-02) and would not be independent
evidence.
"""

from __future__ import annotations

import random

from pantrychef.common import get_logger
from pantrychef.common.types import Recipe
from pantrychef.recommender.config import RecConfig
from pantrychef.recommender.features import FEATURE_NAMES, extract_features
from pantrychef.recommender.query_sim import is_train, make_query
from pantrychef.recommender.recommend import candidate_pool, rerank

log = get_logger(__name__)


def recall_at_k(ranked_ids: list[str], gold_id: str, k: int) -> float:
    return 1.0 if gold_id in ranked_ids[:k] else 0.0


def mrr_at_k(ranked_ids: list[str], gold_id: str, k: int) -> float:
    for rank, rid in enumerate(ranked_ids[:k], 1):
        if rid == gold_id:
            return 1.0 / rank
    return 0.0


def evaluate(model, recipes, index, sub_lookup, cfg: RecConfig, k: int = 10,
             columns: tuple[str, ...] = FEATURE_NAMES, test_only: bool = True) -> dict[str, float]:
    """Evaluate the recovery task over test-split masked queries.

    Returns overall recall@k / mrr@k, the candidate-recall ceiling (fraction of
    queries where gold is in the pool at all), and recall|in_pool / mrr|in_pool
    conditioned on that subset.
    """
    rng = random.Random(cfg.seed + 1)  # distinct from train stream
    n = n_in_pool = 0
    recall = mrr = recall_ip = mrr_ip = 0.0

    def feat_fn(p, r):
        return extract_features(p, r, sub_lookup)

    for recipe in recipes:
        if test_only and is_train(recipe.recipe_id):
            continue
        q = make_query(recipe, cfg, rng)
        if q is None:
            continue
        n += 1
        pantry = set(q.pantry)
        pool_ids = [r.recipe_id for r in candidate_pool(index, pantry, cfg.candidate_cap)]
        in_pool = q.gold_id in pool_ids
        ranked = rerank(index, pantry, model, feat_fn, k=k, cap=cfg.candidate_cap, columns=columns)
        ids = [r.recipe_id for r in ranked]
        r_at = recall_at_k(ids, q.gold_id, k)
        m_at = mrr_at_k(ids, q.gold_id, k)
        recall += r_at
        mrr += m_at
        if in_pool:
            n_in_pool += 1
            recall_ip += r_at
            mrr_ip += m_at

    out = {
        f"recall@{k}": recall / n if n else 0.0,
        f"mrr@{k}": mrr / n if n else 0.0,
        "ceiling": n_in_pool / n if n else 0.0,
        f"recall@{k}|in_pool": recall_ip / n_in_pool if n_in_pool else 0.0,
        f"mrr@{k}|in_pool": mrr_ip / n_in_pool if n_in_pool else 0.0,
        "n_queries": float(n),
        "n_in_pool": float(n_in_pool),
    }
    log.info("evaluate: %s", {k_: round(v, 4) for k_, v in out.items()})
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_recommend_eval.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add pantrychef/eval/recommend_eval.py tests/eval/test_recommend_eval.py
git commit -m "feat(eval): recovery metrics + candidate ceiling + conditional"
```

---

### Task 9: CLI leaderboard + overlap baseline row

**Files:**
- Create: `pantrychef/eval/recommend_main.py`
- Test: `tests/eval/test_recommend_main.py`

The CLI builds the index + (optional) P2 substitutor, trains overlap (no model — pool order), linear, LambdaMART, and LambdaMART−sub, then prints the comparison. The overlap baseline is scored with an identity model (score = -pool_position) so it reuses the same `evaluate` path on the same pool.

- [ ] **Step 1: Write the failing test** (smoke on a tiny synthetic corpus, no disk/model)

```python
# tests/eval/test_recommend_main.py
import numpy as np

from pantrychef.common.types import Recipe
from pantrychef.eval.recommend_main import OverlapModel, run_leaderboard
from pantrychef.recommender.config import RecConfig
from pantrychef.retrieval.index import InvertedIndex


def test_overlap_model_preserves_pool_order():
    # OverlapModel scores rows by descending position so pool order is kept
    m = OverlapModel()
    X = np.zeros((3, 2))
    s = m.score(X)
    assert s[0] > s[1] > s[2]


def test_run_leaderboard_smoke():
    corpus = [
        Recipe(recipe_id=str(i), title=str(i),
               canonical=["egg", "flour", "milk", "sugar", "butter", "salt", "oil"][: 4 + (i % 4)])
        for i in range(80)
    ]
    idx = InvertedIndex.build(corpus)
    rows = run_leaderboard(corpus, idx, sub_lookup=None, cfg=RecConfig(seed=3), use_lambdamart=False)
    names = {r["model"] for r in rows}
    assert "overlap" in names and "linear" in names
    for r in rows:
        assert "recall@10" in r and "mrr@10" in r
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_recommend_main.py -v`
Expected: FAIL — `ModuleNotFoundError: pantrychef.eval.recommend_main`

- [ ] **Step 3: Write minimal implementation**

```python
# pantrychef/eval/recommend_main.py
"""`python -m pantrychef.eval.recommend_main` — Phase 3 reranker leaderboard.

Compares overlap (P1) < linear < LambdaMART, plus a LambdaMART-without-subs
ablation, on the masked recipe-recovery task. Prints recall@10 / MRR@10, the
candidate ceiling, and in-pool conditional metrics.
"""

from __future__ import annotations

import argparse

import numpy as np

from pantrychef.common import get_logger
from pantrychef.recommender.config import RecConfig
from pantrychef.recommender.features import FEATURE_NAMES, SUB_FEATURES
from pantrychef.recommender.rank import LambdaMARTRanker, LinearRanker
from pantrychef.recommender.train import train_ranker
from pantrychef.eval.recommend_eval import evaluate

log = get_logger(__name__)

NO_SUB_COLUMNS = tuple(c for c in FEATURE_NAMES if c not in SUB_FEATURES)


class OverlapModel:
    """Identity ranker: keeps the candidate-pool (coverage) order. score = -row_index."""

    def fit(self, X, y, groups):  # noqa: D401 - no-op
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -np.arange(len(X), dtype=float)


def run_leaderboard(recipes, index, sub_lookup, cfg: RecConfig, use_lambdamart: bool = True):
    """Train + evaluate each model; return a list of metric rows."""
    rows = []

    overlap = OverlapModel()
    rows.append({"model": "overlap", **evaluate(overlap, recipes, index, sub_lookup, cfg, k=10)})

    linear, _ = train_ranker(LinearRanker(seed=cfg.seed), recipes, index, sub_lookup, cfg)
    rows.append({"model": "linear", **evaluate(linear, recipes, index, sub_lookup, cfg, k=10)})

    if use_lambdamart:
        lm, _ = train_ranker(LambdaMARTRanker(seed=cfg.seed), recipes, index, sub_lookup, cfg)
        rows.append({"model": "lambdamart", **evaluate(lm, recipes, index, sub_lookup, cfg, k=10)})

        lm_ns, _ = train_ranker(
            LambdaMARTRanker(seed=cfg.seed), recipes, index, sub_lookup, cfg, columns=NO_SUB_COLUMNS
        )
        rows.append({
            "model": "lambdamart-nosub",
            **evaluate(lm_ns, recipes, index, sub_lookup, cfg, k=10, columns=NO_SUB_COLUMNS),
        })
    return rows


def _build_substitutor():
    """Load Phase-2 artifacts; return (sub_lookup, vocab) or (None, None) if absent."""
    from pantrychef.config import get_settings
    from pantrychef.data.store import load_recipes
    from pantrychef.ingredients.vocab import load_vocabulary
    from pantrychef.substitution.config import SubConfig
    from pantrychef.substitution.cooccur import build_cooccurrence, sppmi
    from pantrychef.substitution.dietary import DietTagger
    from pantrychef.substitution.embeddings import EmbeddingModel
    from pantrychef.substitution.graph import ContextGraph
    from pantrychef.substitution.substitute import Substitutor

    s = get_settings()
    model = s.models_dir / "substitution" / "word2vec.kv"
    if not model.exists():
        return None, None, None
    recipes = load_recipes(s.processed_dir / "recipes.jsonl")
    vocab = load_vocabulary(s.processed_dir / "vocab.json")
    cmat, _, _, _ = build_cooccurrence(recipes, vocab)
    cfg = SubConfig()
    sub = Substitutor(
        emb=EmbeddingModel.load(model),
        graph=ContextGraph(sppmi(cmat, cfg.sppmi_shift), vocab, cmat, lam=cfg.lam,
                           overlap_shrink=cfg.overlap_shrink),
        tagger=DietTagger(known=vocab),
        cfg=cfg,
    )

    def lookup(missing: str) -> dict[str, float]:
        return {sub_.ingredient: sub_.score for sub_ in sub.substitutes(missing, k=20)}

    return lookup, recipes, vocab


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 3 reranker leaderboard.")
    ap.add_argument("--max-rows", type=int, default=None, help="cap corpus size for fast dev")
    ap.add_argument("--mask-fraction", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no-subs", action="store_true", help="skip Phase-2 sub_fill features")
    args = ap.parse_args(argv)

    from pantrychef.config import get_settings
    from pantrychef.data.store import load_recipes
    from pantrychef.retrieval.index import InvertedIndex

    s = get_settings()
    sub_lookup, recipes, _ = (None, None, None) if args.no_subs else _build_substitutor()
    if recipes is None:
        recipes = load_recipes(s.processed_dir / "recipes.jsonl")
    if args.max_rows:
        recipes = recipes[: args.max_rows]

    index = InvertedIndex.build(recipes)
    cfg = RecConfig(mask_fraction=args.mask_fraction, seed=args.seed)
    rows = run_leaderboard(recipes, index, sub_lookup, cfg)
    for r in rows:
        log.info("%s", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_recommend_main.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full reranker test suite + lint**

Run: `uv run pytest tests/recommender tests/eval/test_recommend_eval.py tests/eval/test_recommend_main.py -v && uv run ruff check pantrychef/recommender pantrychef/eval/recommend_eval.py pantrychef/eval/recommend_main.py && uv run ruff format --check .`
Expected: all PASS, no lint errors.

- [ ] **Step 6: Commit**

```bash
git add pantrychef/eval/recommend_main.py tests/eval/test_recommend_main.py
git commit -m "feat(eval): Phase 3 reranker leaderboard CLI + overlap baseline row"
```

---

### Task 10: Real-corpus run + docs + memory

**Files:**
- Modify: `docs/EVALUATION.md` (Phase 3 table)
- Create: `docs/MODEL_CARD_recommender.md`
- Modify: memory `project-pantrychef-state.md` + `MEMORY.md`

- [ ] **Step 1: Run the leaderboard on the 50k sample**

Regenerate the 50k sample if needed (the `--max-rows` clean), then:
Run: `uv run python -m pantrychef.eval.recommend_main --max-rows 50000`
Capture: overlap / linear / lambdamart / lambdamart-nosub rows (recall@10, mrr@10, ceiling, in_pool).
Expected: `overlap < linear <= lambdamart`; `lambdamart` ≥ `lambdamart-nosub` if subs help.

- [ ] **Step 2: Run the full-corpus scale row**

Run: `uv run python -m pantrychef.eval.recommend_main` (full 1.27M on disk)
Capture the same metrics as a scale row. Note runtime (this may be long — per the "estimate long runs" memory, time it rather than guessing).

- [ ] **Step 3: Fill `docs/EVALUATION.md` Phase 3 table**

Replace the `_tbd_` table (currently NDCG@10/recall@10) with the recovery-task results:

```markdown
### Phase 3 — Recommendation (recipe-recovery task)
_Task: mask 30% of a recipe's ingredients, recover that recipe by reranking the
shared overlap candidate pool. Single gold per query; NDCG omitted as redundant
with MRR under single-gold (see design spec 2026-06-02)._

| Metric | overlap (P1) | linear | LambdaMART | LambdaMART −sub | Run |
| ------ | ------------ | ------ | ---------- | --------------- | --- |
| recall@10 | <v> | <v> | <v> | <v> | 50k |
| MRR@10 | <v> | <v> | <v> | <v> | 50k |

Candidate-recall ceiling: <v>. In-pool recall@10 (LambdaMART): <v>.
Full-corpus (1.27M) scale row: recall@10 <v> / MRR@10 <v>.
```

- [ ] **Step 4: Write `docs/MODEL_CARD_recommender.md`**

Cover: recovery-task scope (NOT general relevance), held-out label construction, feature list + dropped popularity, the sub_fill ablation result, frozen-P2 disclosure, candidate-ceiling honesty, near-duplicate caveat, train/test split. Mirror the structure of `docs/MODEL_CARD_substitution.md`.

- [ ] **Step 5: Update memory**

In `project-pantrychef-state.md`, add a Phase 3 line with the headline numbers + ablation verdict. Add a one-line pointer in `MEMORY.md`. If the sub_fill ablation shows lift, record the magnitude as a finding.

- [ ] **Step 6: Commit + finish branch**

```bash
git add docs/EVALUATION.md docs/MODEL_CARD_recommender.md
git commit -m "docs(eval): Phase 3 recovery-task results + recommender model card"
```

Then invoke `superpowers:finishing-a-development-branch` to decide merge/PR/tag (`v0.3.0`).

---

## Self-Review

**Spec coverage:**
- Task framing (rerank, same pool) → Task 6 `candidate_pool`/`rerank`, `OverlapModel` in Task 9. ✓
- Held-out labels, feature-independent → Task 2 `make_query`, Task 7 labels gold=1 only. ✓
- Linear + LambdaMART, 3-way compare → Tasks 4, 5, 9. ✓
- Feature list incl. sub_fill via frozen P2, dropped popularity → Task 3, `_build_substitutor` in Task 9. ✓
- Sub ablation headline → `lambdamart-nosub` in Task 9, reported Task 10. ✓
- Split no-leakage, full-corpus index → Task 2 `is_train`, Task 7 `train_only`, Task 8 `test_only`. ✓
- recall@10 + MRR@10, NDCG omitted, ceiling + conditional → Task 8 `evaluate`. ✓
- Same-candidate-set fairness → `OverlapModel` evaluated on identical pool. ✓
- Error handling (empty pantry, OOV, degenerate, lightgbm missing) → Tasks 6, 3, 2, 5. ✓
- 50k headline + full-corpus scale row → Task 10 steps 1–2. ✓
- `[recommend]` extra → Task 1. ✓

**Placeholder scan:** No TBD/TODO in code steps; `<v>` markers in Task 10 docs are result placeholders filled at run time (intentional — values unknown until the run). No "handle edge cases" hand-waving.

**Type consistency:** `extract_features(pantry, recipe, sub_lookup)` signature consistent across Tasks 3/6/7/8. `to_matrix(feats, columns)` consistent. Ranker `fit(X, y, groups)` / `score(X)` consistent across LinearRanker, LambdaMARTRanker, OverlapModel. `RecConfig` field names consistent. `evaluate(...)` returns the keys the tests assert. `candidate_pool`/`rerank` signatures match call sites.
