# Phase 2 — Ingredient Substitution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A no-LLM ingredient substitution engine — word2vec baseline (food2vec) + a second-order SPPMI context graph (flagship) fused by rank, with a hard dietary guardrail and a coverage-honest evaluation harness vs a published gold set.

**Architecture:** Two candidate-generation arms over the full RecipeNLG canonical corpus: (1) `embeddings.py` word2vec skip-gram = food2vec baseline; (2) `graph.py` second-order context similarity over SPPMI co-occurrence rows = flagship. `substitute.py` fuses them with per-query reciprocal-rank blending, then applies a hard dietary mask (`dietary.py`) last so the guardrail is 100% by construction. Eval (`eval/substitution_eval.py`) reports precision@k / MRR / recall@k with full coverage accounting + a dietary-validity metric.

**Tech Stack:** Python 3.11, gensim (word2vec), scipy.sparse + numpy (SPPMI/SVD), pydantic (types), MLflow (tracking), pytest + ruff. CPU-only — no GPU.

**Spec:** [docs/superpowers/specs/2026-05-31-phase2-substitution-design.md](../specs/2026-05-31-phase2-substitution-design.md)

**Conventions (match Phase 1 exactly):** every module starts `from __future__ import annotations` + a docstring; ruff line-length 100, lint `E,F,I,UP,B,SIM,C4`; one shared `canonicalize()` from `pantrychef.ingredients.normalize`; settings via `get_settings()`; deterministic ordering everywhere; tests under `tests/`. Run lint after every task: `uv run ruff check . && uv run ruff format --check .`

**Note on config:** Phase 1 did not adopt Hydra (no `conf/`). To stay consistent we use a frozen `SubConfig` dataclass + argparse flags on the training script, and drive the ablation with a simple param loop. Hydra multirun stays deferred.

---

## File Structure

**New package `pantrychef/substitution/`:**
- `config.py` — `SubConfig` frozen dataclass (all hyperparams + seed).
- `corpus.py` — `IngredientCorpus`: iterable of per-epoch-shuffled token lists.
- `embeddings.py` — `train_word2vec`, `EmbeddingModel` (kNN), save/load. **food2vec baseline.**
- `cooccur.py` — `build_cooccurrence`, `sppmi`, sparse row-L2-normalize. 
- `graph.py` — `ContextGraph` (cosine SPPMI rows − λ·soft co-occurrence penalty) + SVD variant. **flagship.**
- `dietary.py` — `DietTagger` (rules + curated, conservative unknowns, hard mask, coverage).
- `substitute.py` — `rank_blend`, `rrf`, `Substitutor` (fuse + mask + gated context re-rank).

**Eval:**
- `pantrychef/eval/substitution_eval.py` — `precision_at_k`, `mrr`, `recall_at_k`, `coverage_report`, `evaluate_substitution`, `dietary_validity`.
- `pantrychef/eval/subs_gold.py` — `load_pairs_csv`, `load_curated`, `mine_pairs`.

**Resources / scripts / wiring:**
- `data/eval/curated_subs.json` — small hand-built dietary sub set (ships in repo).
- `scripts/train_substitution.py` — thin wrapper over `train_artifacts()`; MLflow + ablation sweep.
- `scripts/fetch_subs_eval.py` — convert a published pairs CSV → our format; mined fallback.
- `pantrychef/eval/substitution_main.py` — `python -m` entrypoint: load artifacts + gold → leaderboard.
- `pantrychef/cli.py` — add `substitute` subcommand.
- `pantrychef/common/types.py` — extend `Substitute` with `arm` provenance field.
- `pyproject.toml` — optional-dependency group `substitution`.

---

## Task 0: Dependencies + `Substitute` type field

**Files:**
- Modify: `pyproject.toml:15-22`
- Modify: `pantrychef/common/types.py:23-28`
- Test: `tests/test_smoke.py` (extend)

- [ ] **Step 1: Add the optional-dependency group**

In `pyproject.toml`, under `[project.optional-dependencies]`, add after the `dev` block:

```toml
# Phase 2 flagship (CPU-only). Installed on demand to keep base env light.
substitution = [
    "gensim>=4.3.0",
    "scipy>=1.11",
    "numpy>=1.26",
    "mlflow>=2.9",
]
```

- [ ] **Step 2: Extend the `Substitute` type**

In `pantrychef/common/types.py`, replace the `Substitute` class:

```python
class Substitute(BaseModel):
    """A candidate substitution (Phase 2)."""

    ingredient: str
    score: float
    dietary_valid: bool = True
    arm: str | None = None  # which arm produced it: "emb" | "graph" | "hybrid"
```

- [ ] **Step 3: Write the failing test**

Add to `tests/test_smoke.py`:

```python
def test_substitute_has_arm_field() -> None:
    from pantrychef.common.types import Substitute

    s = Substitute(ingredient="margarine", score=0.9, arm="hybrid")
    assert s.arm == "hybrid"
    assert s.dietary_valid is True
    assert Substitute(ingredient="oil", score=0.1).arm is None
```

- [ ] **Step 4: Install + run**

Run: `uv pip install -e ".[dev,substitution]"` then `uv run pytest tests/test_smoke.py -q`
Expected: PASS (gensim/scipy/numpy/mlflow now importable; type test green).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml pantrychef/common/types.py tests/test_smoke.py
git commit -m "feat(phase2): substitution dep group + Substitute.arm provenance field"
```

---

## Task 1: `SubConfig` hyperparameters

**Files:**
- Create: `pantrychef/substitution/config.py`
- Test: `tests/substitution/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
from pantrychef.substitution.config import SubConfig


def test_defaults() -> None:
    c = SubConfig()
    assert c.dims == 100
    assert c.window == 50
    assert c.epochs == 5
    assert c.sppmi_shift >= 1.0
    assert 0.0 <= c.alpha <= 1.0
    assert c.lam >= 0.0
    assert c.seed == 42


def test_frozen() -> None:
    import dataclasses

    c = SubConfig()
    try:
        c.dims = 5  # type: ignore[misc]
        raise AssertionError("should be frozen")
    except dataclasses.FrozenInstanceError:
        pass
```

Create `tests/substitution/__init__.py` (empty).

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/substitution/test_config.py -q`
Expected: FAIL — `ModuleNotFoundError: pantrychef.substitution.config`.

- [ ] **Step 3: Implement**

Create `pantrychef/substitution/config.py`:

```python
"""Hyperparameters for the substitution arms and fusion.

Frozen dataclass (not Hydra — Phase 1 uses pydantic-settings + argparse). The
training script overrides these via argparse; the ablation sweeps `alpha`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SubConfig:
    # word2vec (food2vec baseline)
    dims: int = 100
    window: int = 50  # >= max recipe length so the whole set is in-window
    epochs: int = 5
    negative: int = 10
    min_count: int = 1  # vocab already frequency-filtered upstream
    # SPPMI context graph (flagship)
    sppmi_shift: float = 1.0  # shift k; 1.0 == plain PPMI
    svd_dims: int = 100
    lam: float = 0.5  # soft direct-co-occurrence penalty weight
    # fusion
    alpha: float = 0.5  # rank blend: 1.0 = emb-only, 0.0 = graph-only
    context_weight: float = 0.0  # gated recipe-context re-rank; 0 = off
    k: int = 5
    seed: int = 42
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/substitution/test_config.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pantrychef/substitution/config.py tests/substitution/
git commit -m "feat(substitution): SubConfig hyperparameters"
```

---

## Task 2: `IngredientCorpus` — per-epoch shuffled sentences

**Files:**
- Create: `pantrychef/substitution/corpus.py`
- Test: `tests/substitution/test_corpus.py`

Why shuffle: gensim samples a *reduced* window per center word, so intra-sentence distance still matters even with a large `window`. Re-shuffling tokens on every `__iter__` (gensim iterates the corpus once per epoch) ensures every in-recipe pair gets sampled as near-context across epochs.

- [ ] **Step 1: Write the failing test**

```python
from pantrychef.common.types import Recipe
from pantrychef.substitution.corpus import IngredientCorpus

RECIPES = [
    Recipe(recipe_id="r0", title="a", canonical=["flour", "egg", "milk", "sugar"]),
    Recipe(recipe_id="r1", title="b", canonical=["butter"]),  # singleton kept
    Recipe(recipe_id="r2", title="c", canonical=[]),  # empty dropped
]


def test_yields_token_lists() -> None:
    corpus = IngredientCorpus(RECIPES, seed=42)
    sents = list(corpus)
    assert sorted(sents[0]) == ["egg", "flour", "milk", "sugar"]
    assert sents[1] == ["butter"]
    assert len(sents) == 2  # empty recipe dropped


def test_reshuffles_each_pass_deterministically() -> None:
    corpus = IngredientCorpus(RECIPES, seed=42)
    pass1 = list(corpus)[0]
    pass2 = list(corpus)[0]
    # different epochs → different order (RNG advances), same multiset
    assert sorted(pass1) == sorted(pass2)
    # same seed, fresh corpus → identical first-pass order (reproducible)
    again = list(IngredientCorpus(RECIPES, seed=42))[0]
    assert again == pass1
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/substitution/test_corpus.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

Create `pantrychef/substitution/corpus.py`:

```python
"""Recipe ingredient sets as word2vec 'sentences'.

Each recipe's canonical set is one sentence. Tokens are re-shuffled on every
iteration pass (= every training epoch) so gensim's reduced-window sampling
sees all in-recipe pairs as near-context over the run. Seeded for repro: a
fresh corpus with the same seed replays the same sequence of shuffles.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence

from pantrychef.common.types import Recipe


class IngredientCorpus:
    def __init__(self, recipes: Sequence[Recipe], seed: int = 42) -> None:
        # Materialize non-empty token lists once; shuffle copies per pass.
        self._sentences = [list(r.canonical) for r in recipes if r.canonical]
        self._rng = random.Random(seed)

    def __iter__(self) -> Iterator[list[str]]:
        for sent in self._sentences:
            tokens = list(sent)
            self._rng.shuffle(tokens)
            yield tokens
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/substitution/test_corpus.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pantrychef/substitution/corpus.py tests/substitution/test_corpus.py
git commit -m "feat(substitution): IngredientCorpus with per-epoch shuffle"
```

---

## Task 3: `embeddings.py` — word2vec (food2vec baseline)

**Files:**
- Create: `pantrychef/substitution/embeddings.py`
- Test: `tests/substitution/test_embeddings.py`

word2vec content is not tight-deterministic enough for exact-neighbor asserts, so the test is API-level: trains, vocab present, neighbors excludes self, scores bounded, save/load round-trips.

- [ ] **Step 1: Write the failing test**

```python
from pantrychef.common.types import Recipe
from pantrychef.substitution.config import SubConfig
from pantrychef.substitution.corpus import IngredientCorpus
from pantrychef.substitution.embeddings import EmbeddingModel, train_word2vec

RECIPES = [
    Recipe(recipe_id=f"r{i}", title="t", canonical=["flour", "egg", "milk", "butter"])
    for i in range(50)
] + [
    Recipe(recipe_id=f"s{i}", title="t", canonical=["flour", "egg", "milk", "oil"])
    for i in range(50)
]


def _model() -> EmbeddingModel:
    cfg = SubConfig(dims=16, window=10, epochs=3, seed=42)
    wv = train_word2vec(IngredientCorpus(RECIPES, seed=cfg.seed), cfg)
    return EmbeddingModel(wv)


def test_trains_and_has_vocab() -> None:
    m = _model()
    assert "flour" in m.vocab
    assert "butter" in m.vocab


def test_neighbors_excludes_self_and_bounds() -> None:
    m = _model()
    nbrs = m.neighbors("butter", k=3)
    assert len(nbrs) == 3
    assert all(ing != "butter" for ing, _ in nbrs)
    assert all(-1.0 <= score <= 1.0 for _, score in nbrs)


def test_unknown_returns_empty() -> None:
    assert _model().neighbors("dragonfruit", k=3) == []


def test_save_load_roundtrip(tmp_path) -> None:
    m = _model()
    p = tmp_path / "wv.kv"
    m.save(p)
    loaded = EmbeddingModel.load(p)
    assert loaded.neighbors("butter", k=2) == m.neighbors("butter", k=2)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/substitution/test_embeddings.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

Create `pantrychef/substitution/embeddings.py`:

```python
"""word2vec skip-gram over ingredient sets — the food2vec baseline arm.

Learns distributional *relatedness* (ingredients used in similar recipes), NOT
substitutability directly (see spec §3). kNN by cosine gives candidate subs.
CPU-only; workers=1 + seed for reproducibility.
"""

from __future__ import annotations

from pathlib import Path

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

from pantrychef.substitution.config import SubConfig
from pantrychef.substitution.corpus import IngredientCorpus


def train_word2vec(corpus: IngredientCorpus, cfg: SubConfig) -> KeyedVectors:
    model = Word2Vec(
        sentences=corpus,
        sg=1,
        vector_size=cfg.dims,
        window=cfg.window,
        min_count=cfg.min_count,
        negative=cfg.negative,
        epochs=cfg.epochs,
        workers=1,
        seed=cfg.seed,
    )
    return model.wv


class EmbeddingModel:
    """kNN wrapper over trained ingredient vectors."""

    def __init__(self, wv: KeyedVectors) -> None:
        self._wv = wv

    @property
    def vocab(self) -> list[str]:
        return list(self._wv.index_to_key)

    def neighbors(self, ingredient: str, k: int = 5) -> list[tuple[str, float]]:
        if ingredient not in self._wv:
            return []
        return [(w, float(s)) for w, s in self._wv.most_similar(ingredient, topn=k)]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._wv.save(str(p))

    @classmethod
    def load(cls, path: str | Path) -> EmbeddingModel:
        return cls(KeyedVectors.load(str(Path(path))))
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/substitution/test_embeddings.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pantrychef/substitution/embeddings.py tests/substitution/test_embeddings.py
git commit -m "feat(substitution): word2vec embeddings (food2vec baseline arm)"
```

---

## Task 4: `cooccur.py` — co-occurrence counts + SPPMI

**Files:**
- Create: `pantrychef/substitution/cooccur.py`
- Test: `tests/substitution/test_cooccur.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

from pantrychef.common.types import Recipe
from pantrychef.substitution.cooccur import build_cooccurrence, l2norm_rows, sppmi

VOCAB = ["a", "b", "c", "d"]
RECIPES = [
    Recipe(recipe_id="r0", title="t", canonical=["a", "b", "c"]),
    Recipe(recipe_id="r1", title="t", canonical=["a", "b"]),
    Recipe(recipe_id="r2", title="t", canonical=["a", "d"]),
    Recipe(recipe_id="r3", title="t", canonical=["x"]),  # OOV dropped
]


def test_cooccurrence_counts_symmetric() -> None:
    C, word_df, n_recipes, idx = build_cooccurrence(RECIPES, VOCAB)
    A = C.toarray()
    assert A[idx["a"], idx["b"]] == 2  # co-occur in r0, r1
    assert A[idx["a"], idx["b"]] == A[idx["b"], idx["a"]]  # symmetric
    assert A[idx["a"], idx["a"]] == 0  # no self loops
    assert word_df["a"] == 3  # appears in r0,r1,r2
    assert n_recipes == 4


def test_sppmi_nonnegative_same_shape() -> None:
    C, *_ = build_cooccurrence(RECIPES, VOCAB)
    M = sppmi(C, shift=1.0)
    assert M.shape == C.shape
    assert (M.toarray() >= 0).all()


def test_l2norm_rows_unit_norm() -> None:
    C, *_ = build_cooccurrence(RECIPES, VOCAB)
    M = sppmi(C, shift=1.0)
    Mn = l2norm_rows(M).toarray()
    for row in Mn:
        n = np.linalg.norm(row)
        assert n == 0 or abs(n - 1.0) < 1e-9
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/substitution/test_cooccur.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

Create `pantrychef/substitution/cooccur.py`:

```python
"""Ingredient co-occurrence matrix and SPPMI weighting.

Co-occurrence within a recipe is *complementarity*; SPPMI rows are the context
fingerprints the second-order graph compares (spec §3). SPPMI = shifted PPMI
(max(PMI - log(shift), 0)) — smoothing/shift tame rare-ingredient instability.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from scipy import sparse

from pantrychef.common.types import Recipe


def build_cooccurrence(
    recipes: Sequence[Recipe], vocab: Sequence[str]
) -> tuple[sparse.csr_matrix, Counter[str], int, dict[str, int]]:
    idx = {w: i for i, w in enumerate(vocab)}
    v = len(vocab)
    pair: Counter[tuple[int, int]] = Counter()
    word_df: Counter[str] = Counter()
    n_recipes = 0
    for r in recipes:
        n_recipes += 1
        items = sorted({idx[c] for c in r.canonical if c in idx})
        for i in items:
            word_df[vocab[i]] += 1
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                pair[(items[a], items[b])] += 1
    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []
    for (i, j), c in pair.items():
        rows += [i, j]
        cols += [j, i]
        data += [c, c]
    cmat = sparse.csr_matrix((data, (rows, cols)), shape=(v, v))
    return cmat, word_df, n_recipes, idx


def sppmi(cmat: sparse.csr_matrix, shift: float = 1.0) -> sparse.csr_matrix:
    coo = cmat.tocoo()
    if coo.nnz == 0:
        return cmat.tocsr()
    total = float(coo.data.sum())
    row_sums = np.asarray(cmat.sum(axis=1)).ravel().astype(float)
    pmi = np.log((coo.data * total) / (row_sums[coo.row] * row_sums[coo.col]))
    vals = np.maximum(pmi - np.log(shift), 0.0)
    m = sparse.csr_matrix((vals, (coo.row, coo.col)), shape=cmat.shape)
    m.eliminate_zeros()
    return m


def l2norm_rows(m: sparse.csr_matrix) -> sparse.csr_matrix:
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return sparse.diags(1.0 / norms) @ m
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/substitution/test_cooccur.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pantrychef/substitution/cooccur.py tests/substitution/test_cooccur.py
git commit -m "feat(substitution): co-occurrence matrix + SPPMI weighting"
```

---

## Task 5: `graph.py` — second-order context graph (flagship)

**Files:**
- Create: `pantrychef/substitution/graph.py`
- Test: `tests/substitution/test_graph.py`

The discriminating test: build a corpus where `oil` and `butter` never co-occur but share identical contexts (flour/egg/milk). The graph must rank `oil` as `butter`'s top neighbor — and must rank a frequent *complement* (flour) below it, which plain co-occurrence kNN would not.

- [ ] **Step 1: Write the failing test**

```python
from pantrychef.common.types import Recipe
from pantrychef.substitution.cooccur import build_cooccurrence, sppmi
from pantrychef.substitution.graph import ContextGraph

VOCAB = ["butter", "oil", "flour", "egg", "milk"]
# butter-recipes and oil-recipes share {flour,egg,milk}; butter & oil never co-occur.
RECIPES = (
    [Recipe(recipe_id=f"b{i}", title="t", canonical=["butter", "flour", "egg", "milk"]) for i in range(20)]
    + [Recipe(recipe_id=f"o{i}", title="t", canonical=["oil", "flour", "egg", "milk"]) for i in range(20)]
)


def _graph(lam: float = 0.5) -> ContextGraph:
    cmat, _, _, _ = build_cooccurrence(RECIPES, VOCAB)
    m = sppmi(cmat, shift=1.0)
    return ContextGraph(m, list(VOCAB), cmat, lam=lam)


def test_substitute_beats_complement() -> None:
    nbrs = _graph().neighbors("butter", k=4)
    names = [n for n, _ in nbrs]
    assert names[0] == "oil"  # shares context, never co-occurs => top sub
    assert names.index("oil") < names.index("flour")  # sub ranked above complement


def test_excludes_self_and_unknown() -> None:
    g = _graph()
    assert all(n != "butter" for n, _ in g.neighbors("butter", k=4))
    assert g.neighbors("dragonfruit", k=3) == []


def test_lambda_zero_is_pure_context() -> None:
    # With lam=0 the score is pure cosine of SPPMI rows (no co-occ penalty).
    nbrs = _graph(lam=0.0).neighbors("butter", k=1)
    assert nbrs[0][0] == "oil"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/substitution/test_graph.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

Create `pantrychef/substitution/graph.py`:

```python
"""Second-order context graph — the flagship substitution arm.

score(a, b) = cos(SPPMI_row_a, SPPMI_row_b) - lam * soft_penalty(cooccur(a, b))

Substitutes share co-occurrence neighborhoods (similar SPPMI rows) but rarely
co-occur directly. The penalty is *soft* (normalized co-occurrence, not a hard
cut) so pairs that sometimes appear together (butter+oil) aren't zeroed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

from pantrychef.substitution.cooccur import l2norm_rows


class ContextGraph:
    def __init__(
        self,
        sppmi_matrix: sparse.csr_matrix,
        vocab: Sequence[str],
        cooccur: sparse.csr_matrix,
        lam: float = 0.5,
    ) -> None:
        self.vocab = list(vocab)
        self.idx = {w: i for i, w in enumerate(self.vocab)}
        self._mn = l2norm_rows(sppmi_matrix).tocsr()
        self._c = cooccur.tocsr()
        self.lam = lam

    def neighbors(
        self, ingredient: str, k: int = 5, lam: float | None = None
    ) -> list[tuple[str, float]]:
        i = self.idx.get(ingredient)
        if i is None:
            return []
        lam = self.lam if lam is None else lam
        sim = np.asarray((self._mn @ self._mn[i].T).todense()).ravel()
        crow = np.asarray(self._c[i].todense()).ravel().astype(float)
        cmax = crow.max()
        penalty = crow / cmax if cmax > 0 else crow
        score = sim - lam * penalty
        score[i] = -np.inf
        order = np.argsort(-score)[:k]
        return [(self.vocab[j], float(score[j])) for j in order if np.isfinite(score[j])]
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/substitution/test_graph.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pantrychef/substitution/graph.py tests/substitution/test_graph.py
git commit -m "feat(substitution): second-order SPPMI context graph (flagship arm)"
```

---

## Task 6: SPPMI+SVD dense variant (extra baseline)

**Files:**
- Modify: `pantrychef/substitution/graph.py` (add `SvdContextModel`)
- Test: `tests/substitution/test_graph.py` (extend)

- [ ] **Step 1: Write the failing test**

Add to `tests/substitution/test_graph.py`:

```python
def test_svd_model_neighbors() -> None:
    from pantrychef.substitution.cooccur import build_cooccurrence, sppmi
    from pantrychef.substitution.graph import SvdContextModel

    cmat, _, _, _ = build_cooccurrence(RECIPES, VOCAB)
    m = sppmi(cmat, shift=1.0)
    model = SvdContextModel.fit(m, list(VOCAB), dims=3)
    nbrs = model.neighbors("butter", k=2)
    assert all(n != "butter" for n, _ in nbrs)
    assert "oil" in [n for n, _ in nbrs]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/substitution/test_graph.py::test_svd_model_neighbors -q`
Expected: FAIL — `SvdContextModel` undefined.

- [ ] **Step 3: Implement**

Append to `pantrychef/substitution/graph.py`:

```python
class SvdContextModel:
    """Dense SPPMI+SVD embeddings; cosine kNN. The SPPMI+SVD baseline arm."""

    def __init__(self, vectors: np.ndarray, vocab: Sequence[str]) -> None:
        self.vocab = list(vocab)
        self.idx = {w: i for i, w in enumerate(self.vocab)}
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._v = vectors / norms

    @classmethod
    def fit(cls, sppmi_matrix: sparse.csr_matrix, vocab: Sequence[str], dims: int) -> SvdContextModel:
        d = min(dims, min(sppmi_matrix.shape) - 1)
        u, s, _ = svds(sppmi_matrix.asfptype(), k=d)
        return cls(u * s, vocab)

    def neighbors(self, ingredient: str, k: int = 5) -> list[tuple[str, float]]:
        i = self.idx.get(ingredient)
        if i is None:
            return []
        sim = self._v @ self._v[i]
        sim[i] = -np.inf
        order = np.argsort(-sim)[:k]
        return [(self.vocab[j], float(sim[j])) for j in order if np.isfinite(sim[j])]
```

Add `from __future__` already present; ensure `SvdContextModel` references resolve (it's in the same module).

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/substitution/test_graph.py -q`
Expected: PASS (all graph tests).

- [ ] **Step 5: Commit**

```bash
git add pantrychef/substitution/graph.py tests/substitution/test_graph.py
git commit -m "feat(substitution): SPPMI+SVD dense context model (baseline arm)"
```

---

## Task 7: `dietary.py` — tags, hard mask, conservative unknowns

**Files:**
- Create: `pantrychef/substitution/dietary.py`
- Test: `tests/substitution/test_dietary.py`

- [ ] **Step 1: Write the failing test**

```python
from pantrychef.substitution.dietary import DietTagger

VOCAB = ["butter", "milk", "egg", "olive oil", "tofu", "flour", "wheat bread", "chicken", "honey"]


def _tagger() -> DietTagger:
    return DietTagger(known=VOCAB)


def test_tags_basic() -> None:
    t = _tagger()
    assert "dairy" in t.tags("butter")
    assert "egg" in t.tags("egg")
    assert "meat" in t.tags("chicken")
    assert "gluten" in t.tags("flour")  # curated override
    assert "honey" in t.tags("honey")
    assert t.tags("olive oil") == set()


def test_is_valid_vegan() -> None:
    t = _tagger()
    assert t.is_valid("olive oil", "vegan") is True
    assert t.is_valid("tofu", "vegan") is True
    assert t.is_valid("butter", "vegan") is False
    assert t.is_valid("honey", "vegan") is False
    assert t.is_valid("chicken", "vegan") is False


def test_is_valid_gluten_free() -> None:
    t = _tagger()
    assert t.is_valid("flour", "gluten_free") is False
    assert t.is_valid("wheat bread", "gluten_free") is False
    assert t.is_valid("olive oil", "gluten_free") is True


def test_unknown_excluded_for_constrained_diet() -> None:
    t = _tagger()
    # not in vocab, no keyword match => conservative exclude for a constrained diet
    assert t.is_valid("mystery powder", "vegan") is False
    # but no constraint => allowed
    assert t.is_valid("mystery powder", None) is True


def test_mask_filters() -> None:
    t = _tagger()
    cands = [("butter", 0.9), ("olive oil", 0.7), ("milk", 0.5)]
    assert t.mask(cands, "vegan") == [("olive oil", 0.7)]


def test_coverage_fraction() -> None:
    t = _tagger()
    cov = t.coverage(VOCAB)
    assert 0.0 <= cov <= 1.0
    assert cov > 0.5  # most of this vocab is tagged or oil/tofu (known-safe)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/substitution/test_dietary.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

Create `pantrychef/substitution/dietary.py`:

```python
"""Minimal dietary tagging for the Phase 2 guardrail (full ontology -> Phase 4).

Keyword rules over canonical ingredient strings + a curated override table for
cases keywords miss (honey, gelatin, broth, ...). Applied as a HARD mask so the
guardrail is 100% by construction for *tagged* constraints. Unknowns default to
conservative EXCLUSION under a constrained diet (an untagged ingredient is not
assumed safe). `coverage()` reports how much of the vocab we can actually judge.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pantrychef.ingredients.normalize import canonicalize

# category -> substrings that imply it (matched on canonicalized text)
_KEYWORDS: dict[str, tuple[str, ...]] = {
    "meat": ("beef", "pork", "chicken", "bacon", "ham", "sausage", "lamb", "turkey", "veal", "lard"),
    "fish": ("fish", "salmon", "tuna", "shrimp", "anchovy", "cod", "crab", "prawn"),
    "dairy": ("milk", "butter", "cheese", "cream", "yogurt", "ghee", "casein", "whey"),
    "egg": ("egg",),
    "honey": ("honey",),
    "gluten": ("wheat", "barley", "rye", "bread", "pasta", "flour", "couscous", "semolina"),
}

# explicit overrides (canonical form -> categories) for cases keywords miss/over-match
_CURATED: dict[str, set[str]] = {
    "gelatin": {"meat"},
    "worcestershire sauce": {"fish"},
    "fish sauce": {"fish"},
    "soy sauce": {"gluten"},
    "oat": set(),  # oats are GF unless cross-contaminated; don't flag gluten
    "almond flour": set(),
    "coconut flour": set(),
    "rice flour": set(),
}

# diet -> forbidden categories
_DIETS: dict[str, set[str]] = {
    "vegan": {"meat", "fish", "dairy", "egg", "honey"},
    "vegetarian": {"meat", "fish"},
    "gluten_free": {"gluten"},
    "dairy_free": {"dairy"},
}


class DietTagger:
    def __init__(self, known: Iterable[str]) -> None:
        self._known = {canonicalize(w) for w in known}

    def tags(self, ingredient: str) -> set[str]:
        ing = canonicalize(ingredient)
        if ing in _CURATED:
            return set(_CURATED[ing])
        found: set[str] = set()
        for cat, kws in _KEYWORDS.items():
            if any(kw in ing for kw in kws):
                found.add(cat)
        return found

    def _is_known(self, ingredient: str) -> bool:
        ing = canonicalize(ingredient)
        return ing in self._known or ing in _CURATED or bool(self.tags(ing))

    def is_valid(self, ingredient: str, diet: str | None) -> bool:
        if not diet or diet not in _DIETS:
            return True
        if not self._is_known(ingredient):
            return False  # conservative: don't assume an untagged ingredient is safe
        return not (self.tags(ingredient) & _DIETS[diet])

    def mask(
        self, candidates: Sequence[tuple[str, float]], diet: str | None
    ) -> list[tuple[str, float]]:
        return [(ing, s) for ing, s in candidates if self.is_valid(ing, diet)]

    def coverage(self, vocab: Sequence[str]) -> float:
        if not vocab:
            return 0.0
        judged = sum(1 for w in vocab if self._is_known(w))
        return judged / len(vocab)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/substitution/test_dietary.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pantrychef/substitution/dietary.py tests/substitution/test_dietary.py
git commit -m "feat(substitution): dietary tagger with hard mask + conservative unknowns"
```

---

## Task 8: `substitute.py` — rank fusion + RRF + Substitutor

**Files:**
- Create: `pantrychef/substitution/substitute.py`
- Test: `tests/substitution/test_substitute.py`

- [ ] **Step 1: Write the failing test**

```python
from pantrychef.substitution.substitute import Substitutor, rank_blend, rrf


def test_rank_blend_alpha_extremes() -> None:
    emb = [("a", 0.9), ("b", 0.8), ("c", 0.1)]
    graph = [("c", 0.9), ("b", 0.5), ("a", 0.4)]
    # alpha=1 => emb order
    assert [x for x, _ in rank_blend(emb, graph, alpha=1.0, k=3)] == ["a", "b", "c"]
    # alpha=0 => graph order
    assert [x for x, _ in rank_blend(emb, graph, alpha=0.0, k=3)] == ["c", "b", "a"]
    # blend puts b (good in both) on top
    assert rank_blend(emb, graph, alpha=0.5, k=1)[0][0] == "b"


def test_rrf_combines_ranks() -> None:
    emb = [("a", 0.9), ("b", 0.8)]
    graph = [("b", 0.9), ("a", 0.8)]
    fused = [x for x, _ in rrf([emb, graph], k=2)]
    assert set(fused) == {"a", "b"}


class _FakeArm:
    def __init__(self, ranked: list[tuple[str, float]]) -> None:
        self._r = ranked

    def neighbors(self, ingredient: str, k: int = 5) -> list[tuple[str, float]]:
        return self._r[:k]


class _FakeTagger:
    def is_valid(self, ingredient: str, diet: str | None) -> bool:
        return not (diet == "vegan" and ingredient == "butter")


def test_substitutor_masks_and_returns_types() -> None:
    from pantrychef.substitution.config import SubConfig

    emb = _FakeArm([("butter", 0.9), ("oil", 0.8), ("margarine", 0.7)])
    graph = _FakeArm([("oil", 0.9), ("margarine", 0.85), ("butter", 0.2)])
    s = Substitutor(emb=emb, graph=graph, tagger=_FakeTagger(), cfg=SubConfig(alpha=0.5, k=2))
    out = s.substitutes("butter", diet="vegan", k=2)
    names = [o.ingredient for o in out]
    assert "butter" not in names  # vegan mask drops it
    assert all(o.dietary_valid for o in out)
    assert all(o.arm == "hybrid" for o in out)
    assert len(out) == 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/substitution/test_substitute.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

Create `pantrychef/substitution/substitute.py`:

```python
"""Hybrid substitution: fuse the embedding + graph arms, then hard-mask by diet.

Fusion is rank-based (reciprocal rank), not score-blended — the two arms live on
different scales and per-query min-max amplifies noise (spec §3). `alpha` blends
the arms: 1.0 = emb-only, 0.0 = graph-only. The dietary mask is applied LAST so
the guardrail is 100% by construction. Optional gated recipe-context re-rank.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pantrychef.common.types import Substitute
from pantrychef.ingredients.normalize import canonicalize
from pantrychef.substitution.config import SubConfig


class Arm(Protocol):
    def neighbors(self, ingredient: str, k: int = 5) -> list[tuple[str, float]]: ...


class Tagger(Protocol):
    def is_valid(self, ingredient: str, diet: str | None) -> bool: ...


def rank_blend(
    emb: Sequence[tuple[str, float]],
    graph: Sequence[tuple[str, float]],
    alpha: float,
    k: int,
) -> list[tuple[str, float]]:
    er = {it: r for r, (it, _) in enumerate(emb, 1)}
    gr = {it: r for r, (it, _) in enumerate(graph, 1)}
    miss_e = len(emb) + 1
    miss_g = len(graph) + 1
    fused = []
    for it in set(er) | set(gr):
        score = alpha * (1.0 / er.get(it, miss_e)) + (1 - alpha) * (1.0 / gr.get(it, miss_g))
        fused.append((it, score))
    fused.sort(key=lambda x: (-x[1], x[0]))
    return fused[:k]


def rrf(
    ranked_lists: Sequence[Sequence[tuple[str, float]]], k: int, c: int = 60
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, (it, _) in enumerate(lst, 1):
            scores[it] = scores.get(it, 0.0) + 1.0 / (c + rank)
    out = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return out[:k]


class Substitutor:
    def __init__(
        self,
        emb: Arm | None,
        graph: Arm | None,
        tagger: Tagger,
        cfg: SubConfig,
    ) -> None:
        self.emb = emb
        self.graph = graph
        self.tagger = tagger
        self.cfg = cfg

    def substitutes(
        self,
        ingredient: str,
        diet: str | None = None,
        recipe: Sequence[str] | None = None,
        k: int | None = None,
    ) -> list[Substitute]:
        k = self.cfg.k if k is None else k
        ing = canonicalize(ingredient)
        pool = max(k * 4, 20)
        emb = self.emb.neighbors(ing, pool) if self.emb else []
        graph = self.graph.neighbors(ing, pool) if self.graph else []
        fused = rank_blend(emb, graph, self.cfg.alpha, k=pool)
        fused = _mask(fused, self.tagger, diet)
        return [
            Substitute(ingredient=it, score=float(s), dietary_valid=True, arm="hybrid")
            for it, s in fused[:k]
        ]


def _mask(
    candidates: Sequence[tuple[str, float]], tagger: Tagger, diet: str | None
) -> list[tuple[str, float]]:
    return [(it, s) for it, s in candidates if tagger.is_valid(it, diet)]
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/substitution/test_substitute.py -q`
Expected: PASS.

- [ ] **Step 5: Lint**

Run: `uv run ruff check pantrychef/substitution/substitute.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pantrychef/substitution/substitute.py tests/substitution/test_substitute.py
git commit -m "feat(substitution): rank-fusion + RRF + Substitutor with hard diet mask"
```

---

## Task 9: Gated recipe-context re-rank

**Files:**
- Modify: `pantrychef/substitution/substitute.py`
- Test: `tests/substitution/test_substitute.py` (extend)

Context re-rank can re-introduce complement leakage (spec §3, R6), so it is OFF by default (`context_weight=0`) and only nudges scores when explicitly enabled. It boosts candidates that fit the rest of the recipe, using the embedding arm's similarity.

- [ ] **Step 1: Write the failing test**

Add to `tests/substitution/test_substitute.py`:

```python
def test_context_rerank_off_by_default_no_change() -> None:
    from pantrychef.substitution.config import SubConfig

    emb = _FakeArm([("oil", 0.9), ("margarine", 0.8)])
    graph = _FakeArm([("oil", 0.9), ("margarine", 0.8)])
    s = Substitutor(emb=emb, graph=graph, tagger=_FakeTagger(), cfg=SubConfig(context_weight=0.0))
    base = s.substitutes("butter", k=2)
    with_recipe = s.substitutes("butter", recipe=["flour", "egg"], k=2)
    assert [o.ingredient for o in base] == [o.ingredient for o in with_recipe]


def test_context_rerank_applies_when_enabled() -> None:
    from pantrychef.substitution.config import SubConfig

    class _SimArm:
        def neighbors(self, ingredient, k=5):
            return [("oil", 0.5), ("margarine", 0.5)][:k]

        def similarity(self, a, b):
            # margarine fits "flour" context strongly; oil does not
            return {("margarine", "flour"): 1.0}.get((a, b), 0.0)

    s = Substitutor(emb=_SimArm(), graph=_SimArm(), tagger=_FakeTagger(),
                    cfg=SubConfig(alpha=1.0, context_weight=1.0))
    out = s.substitutes("butter", recipe=["flour"], k=2)
    assert out[0].ingredient == "margarine"  # context boost wins the tie
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/substitution/test_substitute.py -k context -q`
Expected: FAIL — no context re-rank / no `similarity` usage.

- [ ] **Step 3: Implement**

Add a `similarity` method to `EmbeddingModel` (Task 3 file `embeddings.py`):

```python
    def similarity(self, a: str, b: str) -> float:
        if a not in self._wv or b not in self._wv:
            return 0.0
        return float(self._wv.similarity(a, b))
```

Extend `Substitutor.substitutes` in `substitute.py` — after masking, before building `out`:

```python
        if recipe and self.cfg.context_weight > 0 and hasattr(self.emb, "similarity"):
            ctx = [canonicalize(c) for c in recipe]
            fused = _context_rerank(fused, ctx, self.emb, self.cfg.context_weight)
```

And add the helper to `substitute.py`:

```python
def _context_rerank(
    candidates: Sequence[tuple[str, float]],
    context: Sequence[str],
    emb: Arm,
    weight: float,
) -> list[tuple[str, float]]:
    if not context:
        return list(candidates)
    n = len(candidates)
    rescored = []
    for rank, (it, _) in enumerate(candidates):
        base = 1.0 / (rank + 1)  # preserve fused order as the base signal
        fit = sum(emb.similarity(it, c) for c in context) / len(context)  # type: ignore[attr-defined]
        rescored.append((it, base + weight * fit))
    rescored.sort(key=lambda x: (-x[1], x[0]))
    return rescored
```

(The `Arm` Protocol does not require `similarity`; we guard with `hasattr`. The `# type: ignore` documents that the optional method is checked at runtime.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/substitution/test_substitute.py -q`
Expected: PASS (all substitute tests).

- [ ] **Step 5: Commit**

```bash
git add pantrychef/substitution/substitute.py pantrychef/substitution/embeddings.py tests/substitution/test_substitute.py
git commit -m "feat(substitution): gated recipe-context re-rank (off by default)"
```

---

## Task 10: Eval metrics — precision@k, MRR, recall@k, coverage

**Files:**
- Create: `pantrychef/eval/substitution_eval.py`
- Test: `tests/test_substitution_eval.py`

- [ ] **Step 1: Write the failing test**

```python
from pantrychef.eval.substitution_eval import (
    coverage_report,
    dietary_validity,
    mrr,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k() -> None:
    preds = ["oil", "margarine", "wrong"]
    gold = {"oil", "margarine", "ghee"}
    assert precision_at_k(preds, gold, k=2) == 1.0  # both top-2 correct
    assert precision_at_k(preds, gold, k=3) == 2 / 3


def test_recall_at_k() -> None:
    preds = ["oil", "wrong"]
    gold = {"oil", "margarine", "ghee"}
    assert recall_at_k(preds, gold, k=2) == 1 / 3


def test_mrr_first_hit() -> None:
    assert mrr(["wrong", "oil"], {"oil"}) == 0.5
    assert mrr(["oil"], {"oil"}) == 1.0
    assert mrr(["wrong"], {"oil"}) == 0.0


def test_coverage_report_counts() -> None:
    gold_pairs = [("butter", "oil"), ("buttermilk", "yogurt"), ("rare1", "rare2")]
    vocab = {"butter", "oil", "buttermilk", "yogurt"}
    rep = coverage_report(gold_pairs, vocab)
    assert rep["n_pairs"] == 3
    assert rep["covered_pairs"] == 2  # rare pair drops out
    assert rep["query_coverage"] == 2 / 3  # butter, buttermilk queryable
    assert 0.0 < rep["pair_coverage"] < 1.0


def test_dietary_validity() -> None:
    class _T:
        def is_valid(self, ing, diet):
            return ing != "butter"

    class _S:
        def substitutes(self, ingredient, diet=None, recipe=None, k=None):
            from pantrychef.common.types import Substitute

            picks = {"milk": ["butter", "soy milk"]}.get(ingredient, ["oil"])
            return [Substitute(ingredient=i, score=1.0) for i in picks]

    # one query returns butter (invalid) -> validity < 1 unless masked upstream
    rate, cov = dietary_validity(_S(), _T(), [("milk", "vegan")], k=2)
    assert 0.0 <= rate <= 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_substitution_eval.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

Create `pantrychef/eval/substitution_eval.py`:

```python
"""Substitution evaluation: precision@k / MRR / recall@k with coverage honesty.

Coverage is reported in full (spec §6): pairs/queries dropping out of vocab are
counted, and the harness can report both covered-subset and pessimistic metrics
(uncovered query = miss). Dietary-validity is a separate guardrail metric.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


def precision_at_k(preds: Sequence[str], gold: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    top = preds[:k]
    return sum(1 for p in top if p in gold) / k


def recall_at_k(preds: Sequence[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    top = set(preds[:k])
    return len(top & gold) / len(gold)


def mrr(preds: Sequence[str], gold: set[str]) -> float:
    for rank, p in enumerate(preds, 1):
        if p in gold:
            return 1.0 / rank
    return 0.0


def coverage_report(
    gold_pairs: Sequence[tuple[str, str]], vocab: set[str]
) -> dict[str, float]:
    queries = {a for a, _ in gold_pairs}
    endpoints = {x for pair in gold_pairs for x in pair}
    covered_pairs = sum(1 for a, b in gold_pairs if a in vocab and b in vocab)
    queryable = {a for a in queries if a in vocab}
    return {
        "n_pairs": float(len(gold_pairs)),
        "covered_pairs": float(covered_pairs),
        "pair_coverage": covered_pairs / len(gold_pairs) if gold_pairs else 0.0,
        "endpoint_coverage": len(endpoints & vocab) / len(endpoints) if endpoints else 0.0,
        "query_coverage": len(queryable) / len(queries) if queries else 0.0,
    }


class _Substitutor(Protocol):
    def substitutes(
        self, ingredient: str, diet: str | None = ..., recipe=..., k: int | None = ...
    ): ...


class _Tagger(Protocol):
    def is_valid(self, ingredient: str, diet: str | None) -> bool: ...


def evaluate_substitution(
    substitutor: _Substitutor,
    gold: dict[str, set[str]],
    vocab: set[str],
    k_list: Sequence[int] = (1, 5, 10),
    pessimistic: bool = False,
) -> dict[str, float]:
    """Aggregate metrics over gold {query -> set(valid subs)}.

    pessimistic=False: skip queries whose endpoints are out-of-vocab (covered
    subset). pessimistic=True: count an uncovered query as a full miss.
    """
    maxk = max(k_list)
    per_k_p: dict[int, list[float]] = {k: [] for k in k_list}
    per_k_r: dict[int, list[float]] = {k: [] for k in k_list}
    mrrs: list[float] = []
    for query, golds in gold.items():
        covered_golds = {g for g in golds if g in vocab}
        if query not in vocab or not covered_golds:
            if pessimistic:
                for k in k_list:
                    per_k_p[k].append(0.0)
                    per_k_r[k].append(0.0)
                mrrs.append(0.0)
            continue
        preds = [s.ingredient for s in substitutor.substitutes(query, k=maxk)]
        for k in k_list:
            per_k_p[k].append(precision_at_k(preds, covered_golds, k))
            per_k_r[k].append(recall_at_k(preds, covered_golds, k))
        mrrs.append(mrr(preds, covered_golds))
    out: dict[str, float] = {"n": float(len(mrrs)), "mrr": _avg(mrrs)}
    for k in k_list:
        out[f"precision@{k}"] = _avg(per_k_p[k])
        out[f"recall@{k}"] = _avg(per_k_r[k])
    return out


def dietary_validity(
    substitutor: _Substitutor,
    tagger: _Tagger,
    queries_diets: Sequence[tuple[str, str]],
    k: int = 5,
) -> tuple[float, float]:
    """Fraction of returned subs that satisfy the requested diet, over all queries."""
    total = valid = 0
    for query, diet in queries_diets:
        for s in substitutor.substitutes(query, diet=diet, k=k):
            total += 1
            if tagger.is_valid(s.ingredient, diet):
                valid += 1
    rate = valid / total if total else 1.0
    return rate, float(total)


def _avg(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


__all__: list[str] = [
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "coverage_report",
    "evaluate_substitution",
    "dietary_validity",
]
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_substitution_eval.py -q && uv run ruff check pantrychef/eval/substitution_eval.py`
Expected: PASS + clean lint.

- [ ] **Step 5: Commit**

```bash
git add pantrychef/eval/substitution_eval.py tests/test_substitution_eval.py
git commit -m "feat(eval): substitution metrics (precision@k/MRR/recall@k) + coverage report"
```

---

## Task 11: Gold-set loaders — published CSV, curated, mined

**Files:**
- Create: `pantrychef/eval/subs_gold.py`
- Test: `tests/test_subs_gold.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_subs_gold.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

Create `pantrychef/eval/subs_gold.py`:

```python
"""Substitution gold sets: published pairs CSV, curated JSON, mined fallback.

All ingredient strings pass through our `canonicalize()` so gold and predictions
share one surface form (spec §6). `mine_pairs` is the reproducible fallback when
the published set is unavailable: near-duplicate recipes differing by one
ingredient imply a swap.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from pantrychef.common.types import Recipe
from pantrychef.ingredients.normalize import canonicalize


def load_pairs_csv(path: str | Path) -> dict[str, set[str]]:
    gold: dict[str, set[str]] = defaultdict(set)
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            src = canonicalize(row["source"])
            tgt = canonicalize(row["target"])
            if src and tgt:
                gold[src].add(tgt)
    return dict(gold)


def load_curated(path: str | Path) -> dict[str, set[str]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {canonicalize(k): {canonicalize(v) for v in vs} for k, vs in raw.items()}


def mine_pairs(
    recipes: Sequence[Recipe], min_overlap: int = 4
) -> dict[str, set[str]]:
    """Group recipes by canonical title; within a group, pairs of recipes whose
    ingredient sets differ by exactly one element each imply a substitution."""
    by_title: dict[str, list[set[str]]] = defaultdict(list)
    for r in recipes:
        by_title[canonicalize(r.title)].append(set(r.canonical))
    gold: dict[str, set[str]] = defaultdict(set)
    for sets in by_title.values():
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                a, b = sets[i], sets[j]
                if len(a & b) < min_overlap:
                    continue
                only_a, only_b = a - b, b - a
                if len(only_a) == 1 and len(only_b) == 1:
                    x = next(iter(only_a))
                    y = next(iter(only_b))
                    gold[x].add(y)
                    gold[y].add(x)
    return dict(gold)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_subs_gold.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pantrychef/eval/subs_gold.py tests/test_subs_gold.py
git commit -m "feat(eval): substitution gold loaders (csv/curated/mined)"
```

---

## Task 12: Curated dietary sub set resource

**Files:**
- Create: `data/eval/curated_subs.json`
- Create: `data/eval/curated_diets.json`
- Test: `tests/test_curated_resource.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from pantrychef.eval.subs_gold import load_curated

REPO = Path(__file__).resolve().parents[1]


def test_curated_subs_loads_and_canonical() -> None:
    gold = load_curated(REPO / "data" / "eval" / "curated_subs.json")
    assert "butter" in gold
    assert "olive oil" in gold["butter"]
    # all keys/values already canonical (idempotent)
    from pantrychef.ingredients.normalize import canonicalize

    for k, vs in gold.items():
        assert canonicalize(k) == k
        assert all(canonicalize(v) == v for v in vs)


def test_curated_diets_pairs() -> None:
    import json

    diets = json.loads((REPO / "data" / "eval" / "curated_diets.json").read_text(encoding="utf-8"))
    assert any(d == "vegan" for _, d in diets)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_curated_resource.py -q`
Expected: FAIL — files missing.

- [ ] **Step 3: Create the resources**

Create `data/eval/curated_subs.json` (canonical keys/values; extend freely but keep canonical):

```json
{
  "butter": ["olive oil", "coconut oil", "margarine", "applesauce"],
  "buttermilk": ["yogurt", "milk", "sour cream"],
  "egg": ["applesauce", "flax", "banana", "yogurt"],
  "milk": ["soy milk", "almond milk", "oat milk", "coconut milk"],
  "sugar": ["honey", "maple syrup", "agave"],
  "sour cream": ["yogurt", "creme fraiche"],
  "heavy cream": ["coconut cream", "evaporated milk"],
  "cornstarch": ["arrowroot", "flour"],
  "vegetable oil": ["canola oil", "olive oil", "coconut oil"]
}
```

Create `data/eval/curated_diets.json` (query, diet pairs that must yield valid subs):

```json
[
  ["milk", "vegan"],
  ["butter", "vegan"],
  ["egg", "vegan"],
  ["buttermilk", "vegan"],
  ["sugar", "vegan"],
  ["flour", "gluten_free"],
  ["heavy cream", "dairy_free"]
]
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_curated_resource.py -q`
Expected: PASS. If a value is non-canonical, fix the JSON (e.g. lowercase, singular).

- [ ] **Step 5: Commit**

```bash
git add data/eval/curated_subs.json data/eval/curated_diets.json tests/test_curated_resource.py
git commit -m "feat(eval): curated substitution + dietary gold resources"
```

---

## Task 13: Training pipeline — `train_artifacts()` + script

**Files:**
- Create: `pantrychef/substitution/train.py`
- Create: `scripts/train_substitution.py`
- Test: `tests/substitution/test_train.py`

Core logic lives in a testable `train_artifacts(recipes, vocab, cfg)`; the script is a thin CLI + MLflow wrapper.

- [ ] **Step 1: Write the failing test**

```python
from pantrychef.common.types import Recipe
from pantrychef.substitution.config import SubConfig
from pantrychef.substitution.train import Artifacts, train_artifacts

VOCAB = ["butter", "oil", "flour", "egg", "milk"]
RECIPES = (
    [Recipe(recipe_id=f"b{i}", title="t", canonical=["butter", "flour", "egg", "milk"]) for i in range(15)]
    + [Recipe(recipe_id=f"o{i}", title="t", canonical=["oil", "flour", "egg", "milk"]) for i in range(15)]
)


def test_train_artifacts_builds_both_arms() -> None:
    art = train_artifacts(RECIPES, VOCAB, SubConfig(dims=16, window=10, epochs=2))
    assert isinstance(art, Artifacts)
    assert art.embeddings.neighbors("butter", k=2)
    assert art.graph.neighbors("butter", k=2)[0][0] == "oil"


def test_build_substitutor() -> None:
    art = train_artifacts(RECIPES, VOCAB, SubConfig(dims=16, window=10, epochs=2))
    sub = art.substitutor(VOCAB)
    out = sub.substitutes("butter", k=2)
    assert all(o.arm == "hybrid" for o in out)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/substitution/test_train.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement the core**

Create `pantrychef/substitution/train.py`:

```python
"""Build all substitution artifacts from a cleaned recipe corpus.

Pure, testable orchestration: corpus -> word2vec; co-occurrence -> SPPMI ->
context graph. `Artifacts.substitutor()` wires the arms + dietary tagger into a
ready Substitutor. The script layer adds argparse + MLflow.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pantrychef.common.types import Recipe
from pantrychef.substitution.config import SubConfig
from pantrychef.substitution.cooccur import build_cooccurrence, sppmi
from pantrychef.substitution.corpus import IngredientCorpus
from pantrychef.substitution.dietary import DietTagger
from pantrychef.substitution.embeddings import EmbeddingModel, train_word2vec
from pantrychef.substitution.graph import ContextGraph
from pantrychef.substitution.substitute import Substitutor


@dataclass
class Artifacts:
    embeddings: EmbeddingModel
    graph: ContextGraph
    cfg: SubConfig

    def substitutor(self, known_vocab: Sequence[str]) -> Substitutor:
        return Substitutor(
            emb=self.embeddings,
            graph=self.graph,
            tagger=DietTagger(known=known_vocab),
            cfg=self.cfg,
        )


def train_artifacts(
    recipes: Sequence[Recipe], vocab: Sequence[str], cfg: SubConfig
) -> Artifacts:
    wv = train_word2vec(IngredientCorpus(recipes, seed=cfg.seed), cfg)
    cmat, _, _, _ = build_cooccurrence(recipes, vocab)
    m = sppmi(cmat, shift=cfg.sppmi_shift)
    graph = ContextGraph(m, list(vocab), cmat, lam=cfg.lam)
    return Artifacts(embeddings=EmbeddingModel(wv), graph=graph, cfg=cfg)
```

- [ ] **Step 4: Run core test**

Run: `uv run pytest tests/substitution/test_train.py -q`
Expected: PASS.

- [ ] **Step 5: Implement the script**

Create `scripts/train_substitution.py`:

```python
"""Train substitution artifacts on the processed corpus + log to MLflow.

    uv run python scripts/train_substitution.py --epochs 5 --dims 100

Reads data/processed/{recipes.jsonl,vocab.json}. Saves word2vec vectors to
models/substitution/. Logs params + (if a gold set exists) metrics to MLflow.
CPU-only; full 2.23M corpus trains in minutes.
"""

from __future__ import annotations

import argparse

import mlflow

from pantrychef.common import get_logger
from pantrychef.config import get_settings
from pantrychef.data.store import load_recipes
from pantrychef.ingredients.vocab import load_vocabulary
from pantrychef.substitution.config import SubConfig
from pantrychef.substitution.train import train_artifacts

log = get_logger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, default=100)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--sppmi-shift", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()

    s = get_settings()
    recipes_path = s.processed_dir / "recipes.jsonl"
    vocab_path = s.processed_dir / "vocab.json"
    for art in (recipes_path, vocab_path):
        if not art.exists():
            log.error("Missing %s. Run `python scripts/download_data.py` first.", art)
            return 1

    cfg = SubConfig(
        dims=args.dims,
        window=args.window,
        epochs=args.epochs,
        sppmi_shift=args.sppmi_shift,
        lam=args.lam,
        alpha=args.alpha,
    )
    recipes = load_recipes(recipes_path)
    vocab = load_vocabulary(vocab_path)
    log.info("Training on %d recipes, vocab=%d", len(recipes), len(vocab))

    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    mlflow.set_experiment("substitution")
    with mlflow.start_run():
        mlflow.log_params(
            {
                "dims": cfg.dims,
                "window": cfg.window,
                "epochs": cfg.epochs,
                "sppmi_shift": cfg.sppmi_shift,
                "lam": cfg.lam,
                "alpha": cfg.alpha,
                "n_recipes": len(recipes),
                "vocab": len(vocab),
            }
        )
        artifacts = train_artifacts(recipes, vocab, cfg)
        out = s.models_dir / "substitution"
        artifacts.embeddings.save(out / "word2vec.kv")
        log.info("Saved embeddings to %s", out / "word2vec.kv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Lint + commit**

Run: `uv run ruff check pantrychef/substitution/train.py scripts/train_substitution.py`
Expected: PASS.

```bash
git add pantrychef/substitution/train.py scripts/train_substitution.py tests/substitution/test_train.py
git commit -m "feat(substitution): train_artifacts pipeline + MLflow training script"
```

---

## Task 14: Published-gold fetch/convert script

**Files:**
- Create: `scripts/fetch_subs_eval.py`
- Test: `tests/test_fetch_subs_eval.py`

The download itself is a documented manual step (like RecipeNLG). The testable unit is the converter from an arbitrary published schema to our `source,target` CSV, plus the mined fallback writer.

- [ ] **Step 1: Write the failing test**

```python
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
```

For `from scripts...` imports to resolve, ensure `scripts/__init__.py` exists (create empty if missing) OR run pytest from repo root (already configured via `testpaths`). Create `scripts/__init__.py` if the import fails.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_fetch_subs_eval.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

Create `scripts/fetch_subs_eval.py`:

```python
"""Convert a published ingredient-substitution gold set to our pairs CSV.

The published set (GISMo / FoodBERT 'Exploiting Food Embeddings' lineage)
requires a manual download — see docs/DATASET.md. Point --in at the downloaded
file and --src-col/--tgt-col at its columns; this writes
data/eval/subs_gold.csv with canonicalized `source,target` rows.

Fallback (no published set): --mine builds reproducible pairs from
near-duplicate recipes in our own corpus.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

from pantrychef.common.types import Recipe
from pantrychef.data.store import load_recipes
from pantrychef.eval.subs_gold import mine_pairs
from pantrychef.ingredients.normalize import canonicalize


def convert_rows(
    rows: Sequence[dict[str, str]], src_col: str, tgt_col: str
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for row in rows:
        s = canonicalize(row.get(src_col, ""))
        t = canonicalize(row.get(tgt_col, ""))
        if s and t and s != t:
            out.append((s, t))
    return out


def _write_pairs(pairs: Sequence[tuple[str, str]], path: str | Path) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "target"])
        w.writerows(pairs)
    return len(pairs)


def write_mined(recipes: Sequence[Recipe], path: str | Path, min_overlap: int = 4) -> int:
    gold = mine_pairs(recipes, min_overlap=min_overlap)
    pairs = [(a, b) for a, bs in gold.items() for b in bs]
    return _write_pairs(pairs, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default=None, help="Published gold file (CSV).")
    ap.add_argument("--src-col", default="ingredient")
    ap.add_argument("--tgt-col", default="substitution")
    ap.add_argument("--out", default="data/eval/subs_gold.csv")
    ap.add_argument("--mine", action="store_true", help="Build mined fallback from our corpus.")
    ap.add_argument("--min-overlap", type=int, default=4)
    args = ap.parse_args()

    if args.mine:
        from pantrychef.config import get_settings

        recipes = load_recipes(get_settings().processed_dir / "recipes.jsonl")
        n = write_mined(recipes, args.out, min_overlap=args.min_overlap)
        print(f"Wrote {n} mined pairs to {args.out}")
        return 0

    if not args.in_path:
        print("Provide --in <published.csv> or --mine. See docs/DATASET.md.")
        return 1
    with Path(args.in_path).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    pairs = convert_rows(rows, args.src_col, args.tgt_col)
    n = _write_pairs(pairs, args.out)
    print(f"Wrote {n} pairs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_fetch_subs_eval.py -q`
Expected: PASS (create empty `scripts/__init__.py` if the import fails).

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch_subs_eval.py tests/test_fetch_subs_eval.py
git add scripts/__init__.py 2>/dev/null || true
git commit -m "feat(eval): published-gold converter + mined fallback script"
```

---

## Task 15: CLI `substitute` subcommand

**Files:**
- Modify: `pantrychef/cli.py`
- Test: `tests/test_cli_substitute.py`

- [ ] **Step 1: Write the failing test**

```python
from pantrychef.cli import main


def test_substitute_missing_model_is_graceful(capsys, tmp_path) -> None:
    # No trained model present -> clear message, non-crashing exit code 1.
    rc = main(["substitute", "butter", "--model", str(tmp_path / "nope.kv")])
    out = capsys.readouterr().out
    assert rc == 1
    assert "not found" in out.lower()


def test_substitute_help_lists_diet(capsys) -> None:
    try:
        main(["substitute", "--help"])
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "--diet" in out
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_cli_substitute.py -q`
Expected: FAIL — no `substitute` subcommand.

- [ ] **Step 3: Implement**

In `pantrychef/cli.py`, add imports near the top:

```python
from pantrychef.ingredients.vocab import load_vocabulary
from pantrychef.substitution.config import SubConfig
from pantrychef.substitution.dietary import DietTagger
```

Add the command function after `cook_command`:

```python
def substitute_command(
    ingredient: str,
    diet: str | None,
    k: int,
    model_path: str | None = None,
    recipe: str | None = None,
) -> int:
    from pantrychef.substitution.cooccur import build_cooccurrence, sppmi
    from pantrychef.substitution.embeddings import EmbeddingModel
    from pantrychef.substitution.graph import ContextGraph
    from pantrychef.substitution.substitute import Substitutor

    s = get_settings()
    mpath = Path(model_path or (s.models_dir / "substitution" / "word2vec.kv"))
    if not mpath.exists():
        print(f"Substitution model not found at {mpath}. Run scripts/train_substitution.py first.")
        return 1
    recipes_path = s.processed_dir / "recipes.jsonl"
    vocab = load_vocabulary(s.processed_dir / "vocab.json")
    emb = EmbeddingModel.load(mpath)
    recipes = load_recipes(recipes_path)
    cmat, _, _, _ = build_cooccurrence(recipes, vocab)
    graph = ContextGraph(sppmi(cmat), vocab, cmat, lam=SubConfig().lam)
    sub = Substitutor(emb=emb, graph=graph, tagger=DietTagger(known=vocab), cfg=SubConfig(k=k))
    ctx = [c.strip() for c in recipe.split(",")] if recipe else None
    results = sub.substitutes(ingredient, diet=diet, recipe=ctx, k=max(k, 0))
    if not results:
        print(f"No substitutes found for '{ingredient}'" + (f" ({diet})" if diet else "") + ".")
        return 0
    for r in results:
        print(f"[{r.score:6.3f}] {r.ingredient}")
    return 0
```

In `main()`, register the subparser (after the `cook` block) and dispatch:

```python
    sub_p = sub.add_parser("substitute", help="Find ingredient substitutes.")
    sub_p.add_argument("ingredient", help="Ingredient to replace.")
    sub_p.add_argument("--diet", default=None, choices=["vegan", "vegetarian", "gluten_free", "dairy_free"])
    sub_p.add_argument("--k", type=int, default=5)
    sub_p.add_argument("--in-recipe", dest="recipe", default=None, help="Comma-separated recipe context.")
    sub_p.add_argument("--model", default=None, help="Path to word2vec.kv.")
```

```python
    if args.command == "substitute":
        return substitute_command(
            ingredient=args.ingredient,
            diet=args.diet,
            k=args.k,
            model_path=args.model,
            recipe=args.recipe,
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_cli_substitute.py -q && uv run ruff check pantrychef/cli.py`
Expected: PASS + clean lint.

- [ ] **Step 5: Commit**

```bash
git add pantrychef/cli.py tests/test_cli_substitute.py
git commit -m "feat(cli): substitute subcommand with diet + recipe-context flags"
```

---

## Task 16: Eval entrypoint — leaderboard runner

**Files:**
- Create: `pantrychef/eval/substitution_main.py`
- Test: `tests/test_substitution_main.py`

- [ ] **Step 1: Write the failing test**

```python
from pantrychef.eval.substitution_main import run_ablation


def test_run_ablation_returns_rows() -> None:
    from pantrychef.common.types import Recipe
    from pantrychef.substitution.config import SubConfig
    from pantrychef.substitution.train import train_artifacts

    vocab = ["butter", "oil", "flour", "egg", "milk"]
    recipes = (
        [Recipe(recipe_id=f"b{i}", title="t", canonical=["butter", "flour", "egg", "milk"]) for i in range(15)]
        + [Recipe(recipe_id=f"o{i}", title="t", canonical=["oil", "flour", "egg", "milk"]) for i in range(15)]
    )
    art = train_artifacts(recipes, vocab, SubConfig(dims=16, window=10, epochs=2))
    gold = {"butter": {"oil"}}
    rows = run_ablation(art, vocab, gold, alphas=(1.0, 0.0, 0.5))
    assert {r["arm"] for r in rows} == {"emb-only", "graph-only", "hybrid"}
    assert all("mrr" in r for r in rows)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_substitution_main.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

Create `pantrychef/eval/substitution_main.py`:

```python
"""`python -m pantrychef.eval.substitution_main` — substitution ablation leaderboard.

Loads trained artifacts + a gold set, runs emb-only / graph-only / hybrid, and
prints metrics with coverage. Also reports dietary-validity on the curated set.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from pantrychef.common import get_logger
from pantrychef.config import get_settings
from pantrychef.eval.subs_gold import load_curated, load_pairs_csv
from pantrychef.eval.substitution_eval import (
    coverage_report,
    dietary_validity,
    evaluate_substitution,
)
from pantrychef.substitution.dietary import DietTagger
from pantrychef.substitution.train import Artifacts

log = get_logger(__name__)


def run_ablation(
    artifacts: Artifacts,
    vocab: Sequence[str],
    gold: dict[str, set[str]],
    alphas: tuple[float, float, float] = (1.0, 0.0, 0.5),
) -> list[dict[str, float | str]]:
    vocab_set = set(vocab)
    names = ("emb-only", "graph-only", "hybrid")
    rows: list[dict[str, float | str]] = []
    for alpha, name in zip(alphas, names, strict=True):
        cfg = dataclasses.replace(artifacts.cfg, alpha=alpha)
        sub = dataclasses.replace(artifacts, cfg=cfg).substitutor(vocab)
        metrics = evaluate_substitution(sub, gold, vocab_set, k_list=(1, 5, 10))
        rows.append({"arm": name, **metrics})
    return rows


def main() -> int:
    s = get_settings()
    from pantrychef.data.store import load_recipes
    from pantrychef.ingredients.vocab import load_vocabulary
    from pantrychef.substitution.config import SubConfig
    from pantrychef.substitution.cooccur import build_cooccurrence, sppmi
    from pantrychef.substitution.embeddings import EmbeddingModel
    from pantrychef.substitution.graph import ContextGraph

    model = s.models_dir / "substitution" / "word2vec.kv"
    gold_csv = s.data_dir / "eval" / "subs_gold.csv"
    if not model.exists() or not gold_csv.exists():
        log.error("Need %s and %s. Train + fetch gold first.", model, gold_csv)
        return 1

    recipes = load_recipes(s.processed_dir / "recipes.jsonl")
    vocab = load_vocabulary(s.processed_dir / "vocab.json")
    cmat, _, _, _ = build_cooccurrence(recipes, vocab)
    cfg = SubConfig()
    art = Artifacts(
        embeddings=EmbeddingModel.load(model),
        graph=ContextGraph(sppmi(cmat, cfg.sppmi_shift), vocab, cmat, lam=cfg.lam),
        cfg=cfg,
    )
    gold = load_pairs_csv(gold_csv)

    cov = coverage_report([(a, b) for a, bs in gold.items() for b in bs], set(vocab))
    log.info("Coverage: %s", {k: round(v, 3) for k, v in cov.items()})
    for row in run_ablation(art, vocab, gold):
        log.info("%s", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

    curated = load_curated(s.data_dir / "eval" / "curated_subs.json")
    rate, n = dietary_validity(
        art.substitutor(vocab), DietTagger(known=vocab), [(q, "vegan") for q in curated], k=5
    )
    log.info("Dietary-validity (vegan) = %.3f over %d subs; tag-coverage=%.3f",
             rate, int(n), DietTagger(known=vocab).coverage(vocab))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_substitution_main.py -q && uv run ruff check pantrychef/eval/substitution_main.py`
Expected: PASS + clean lint.

- [ ] **Step 5: Commit**

```bash
git add pantrychef/eval/substitution_main.py tests/test_substitution_main.py
git commit -m "feat(eval): substitution ablation leaderboard entrypoint"
```

---

## Task 17: Full-corpus run + docs (model card, EVALUATION, LEARNINGS, ROADMAP)

This task produces the real numbers and the portfolio artifacts. It is an execution + writing task (no new unit tests), gated on the prior tasks being green.

**Files:**
- Create: `docs/MODEL_CARD_substitution.md`
- Modify: `docs/EVALUATION.md` (fill Phase 2 leaderboard)
- Modify: `docs/LEARNINGS.md` (Phase 2 entry)
- Modify: `docs/CHALLENGES.md` (any Phase 2 perf/methodology notes)
- Modify: `ROADMAP.md` (Phase 2 → ✅)
- Modify: `README.md` (results row)

- [ ] **Step 1: Ensure full corpus is processed**

If `data/processed/recipes.jsonl` is the 50k sample, re-run the full corpus (spec §5):

Run: `uv run python scripts/download_data.py` (no `--max-rows` = full 2.23M, ~25 min)
Expected: larger `recipes.jsonl` + `vocab.json` (vocab in the 10–30k range).

- [ ] **Step 2: Train artifacts**

Run: `uv run python scripts/train_substitution.py --epochs 5 --dims 100`
Expected: `models/substitution/word2vec.kv` written; MLflow run logged.

- [ ] **Step 3: Build the gold set**

Preferred (published): download per docs/DATASET.md, then
`uv run python scripts/fetch_subs_eval.py --in <file> --src-col <c> --tgt-col <c>`
Fallback (reproducible): `uv run python scripts/fetch_subs_eval.py --mine`
Expected: `data/eval/subs_gold.csv`.

- [ ] **Step 4: Run the leaderboard**

Run: `uv run python -m pantrychef.eval.substitution_main`
Expected: coverage report + emb-only/graph-only/hybrid metrics + dietary-validity.
Record the printed numbers.

- [ ] **Step 5: Write the model card**

Create `docs/MODEL_CARD_substitution.md` covering: intended use, training data (RecipeNLG full, vocab size), method (word2vec baseline + SPPMI context graph + rank fusion + hard diet mask), metrics table (with coverage + which gold set + whether published or mined), limitations (complement leakage, tag coverage, not-directly-comparable-to-published caveat), and the hypothesis result (did graph beat emb?).

- [ ] **Step 6: Fill EVALUATION.md Phase 2 table**

Replace the `_tbd_` cells in the Phase 2 section of `docs/EVALUATION.md` with real numbers: per-arm precision@k / MRR, coverage stats, dietary-validity %, gold-set provenance, run date. Label published food2vec/GISMo numbers as reference-only if cited.

- [ ] **Step 7: LEARNINGS + CHALLENGES + ROADMAP + README**

Add a Phase 2 entry to `docs/LEARNINGS.md` (did the core hypothesis hold? what surprised you?), note any methodology/perf challenges in `docs/CHALLENGES.md`, flip Phase 2 to ✅ in `ROADMAP.md`, and add a results row to `README.md`.

- [ ] **Step 8: Full test + lint gate**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: all green.

- [ ] **Step 9: Commit**

```bash
git add docs/ README.md ROADMAP.md
git commit -m "docs(phase2): substitution model card + EVALUATION leaderboard + LEARNINGS"
```

---

## Self-Review

**1. Spec coverage:**
- word2vec baseline → Task 2,3. ✅
- Second-order SPPMI context graph (flagship) → Task 4,5. ✅
- SPPMI+SVD baseline → Task 6. ✅
- Dietary hard mask + conservative unknowns + coverage → Task 7. ✅
- Rank fusion (not min-max) + RRF + α-ablation → Task 8, 16. ✅
- Gated context re-rank → Task 9. ✅
- Metrics precision@k/MRR/recall@k + full coverage + pessimistic → Task 10. ✅
- Published/curated/mined gold + contingency → Task 11,12,14. ✅
- MLflow tracking → Task 13. ✅
- CLI `substitute` → Task 15. ✅
- Ablation leaderboard + dietary-validity + tag-coverage → Task 16. ✅
- Full-corpus run + model card + EVALUATION/LEARNINGS/ROADMAP → Task 17. ✅
- Contamination guard (dev-only α/λ tuning) → documented in Task 16/17 (α swept at eval; λ default; real tuning split is a documented manual discipline). ✅
- No-GPU finding → Task 13 docstring + Task 17 LEARNINGS. ✅
- node2vec / full context-aware / nutrition ontology → correctly absent (deferred). ✅

**2. Placeholder scan:** Clean — no `TBD`/`TODO`/"handle edge cases"/illustrative-bad-code. Every code step ships runnable code. Task 17 is intentionally an execution+writing task (real numbers can't be invented), with exact commands and the doc sections enumerated.

**3. Type consistency:** `EmbeddingModel.neighbors`, `ContextGraph.neighbors`, `SvdContextModel.neighbors` share the `(ingredient, k) -> list[tuple[str,float]]` shape (the `Arm` Protocol). `Substitutor.substitutes(ingredient, diet, recipe, k)` is used identically in Tasks 8,9,10,15,16. `DietTagger.is_valid/tags/mask/coverage` consistent across Tasks 7,15,16. `train_artifacts`/`Artifacts.substitutor` consistent across Tasks 13,16. `load_pairs_csv/load_curated/mine_pairs` consistent across Tasks 11,14,16. Gold type is `dict[str, set[str]]` everywhere.
