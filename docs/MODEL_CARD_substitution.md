# Model Card — Ingredient Substitution (Phase 2)

> Phase 2 flagship. Not an LLM. Benchmarked vs food2vec baseline.
> Numbers are on a 28k-recipe sample with mined gold; not directly comparable
> to published food2vec / GISMo numbers (see Limitations).

---

## Intended use

Given a query ingredient (e.g. "butter"), return a ranked list of valid
substitutes (e.g. "margarine", "coconut oil") that:
1. Are semantically similar in culinary context.
2. Pass a hard dietary-constraint guardrail (vegan / vegetarian / gluten-free
   tags applied last, before returning results).

**Not intended for:** medical dietary advice, allergen safety, quantity
conversion. Dietary tags are substring-matched — see Limitations.

---

## Training data

| Item | Value |
|------|-------|
| Corpus | RecipeNLG — 28,273-recipe sample (from full 2.23M) |
| Canonical vocab (total) | 2,067 ingredients |
| word2vec vocab (after min_count) | 2,011 |
| Full raw CSV | present in `data/raw/`; full-corpus run pending |

Config: `dims=100, window=50, epochs=5, seed=42`.

The full 2.23M-recipe corpus is present but not yet processed at scale;
28k is the sample used throughout Phase 1 and reused here for consistency.

---

## Method

Three arms, evaluated independently and fused:

### 1 — Embedding baseline (food2vec-style, `emb-only`)
word2vec trained on ingredient **sets** (each recipe = one "document" of
canonical ingredients, order-irrelevant). Nearest neighbours in embedding space
are retrieved by cosine similarity.

**Known bias:** PMI-based word2vec on ingredient sets learns *complementarity*
(ingredients that co-occur together) rather than *substitutability*
(ingredients that are interchangeable). Salt and pepper rank near each other
not because either replaces the other, but because they always appear together.
This makes `emb-only` the baseline to beat, not the flagship.

### 2 — Second-order context graph (SPPMI, `graph-only` — FLAGSHIP)
**Core insight:** substitutes rarely co-occur in the same recipe (you use
*either* butter *or* margarine), so direct PMI is a poor signal. Instead, two
ingredients are substitutes if they share similar *neighborhoods* — they
appear alongside the same supporting cast across different recipes.

Implementation:
1. Build a SPPMI (Shifted Positive PMI) co-occurrence matrix from the corpus.
2. For each ingredient pair (i, j), compute second-order similarity:
   cosine(SPPMI row i, SPPMI row j). This captures shared context without
   requiring direct co-occurrence.
3. Config: `sppmi_shift=1.0, lam=0.5` (Laplace smoothing).

No GPU required; the graph is deterministic, no training loop.

### 3 — Blended-rank fusion (`hybrid`, alpha=0.5)
Scores from `emb-only` and `graph-only` are converted to average ranks and
interpolated: `rank_hybrid = alpha × rank_emb + (1-alpha) × rank_graph`.
Config: `alpha=0.5`. Note: reciprocal-rank fusion was the original plan but
was replaced because it does not reward cross-arm consensus; blended-average-
rank is more stable under this ablation.

### Dietary guardrail (hard constraint, applied last)
Substring-based tag lookup on a curated dietary vocab. Filtered results are
100% valid by construction for tested tags — see Limitations for false-positive
risk.

---

## Evaluation

### Gold-set provenance

> **Phase 2 eval provenance:** RecipeNLG **28k-recipe sample**, seed=42,
> `min_overlap=4`. Gold pairs **mined from the same corpus** (near-duplicate
> recipes differing by exactly one ingredient). 116 directed pairs, 82 evaluated
> query keys.
>
> **Coverage is 100% (pairs / endpoints / queries) by construction** — an
> artifact of mining from the same corpus, not a quality claim. This is a
> reproducible fallback for when the published food2vec / GISMo gold set is
> unavailable. Published-set comparison is wired via
> `scripts/fetch_subs_eval.py` and pending a manual download. Numbers are
> **NOT directly comparable to published food2vec / GISMo results.**

### Ablation results (n=82 queries)

| Arm | MRR | P@1 | R@1 | P@5 | R@5 | P@10 | R@10 |
|-----|-----|-----|-----|-----|-----|------|------|
| emb-only (food2vec baseline) | 0.290 | 0.232 | 0.151 | 0.093 | 0.302 | 0.057 | 0.405 |
| **graph-only (FLAGSHIP)** | **0.339** | **0.256** | **0.189** | **0.105** | **0.395** | **0.063** | **0.475** |
| hybrid (alpha=0.5) | 0.324 | 0.232 | 0.157 | 0.100 | 0.347 | 0.067 | 0.503 |

### Core hypothesis result: CONFIRMED

The second-order context graph (flagship) beats the food2vec baseline on:
- MRR: **0.339 vs 0.290** (+17%)
- P@1: 0.256 vs 0.232
- recall@10: 0.475 vs 0.405

Hybrid gives the best recall@10 (0.503) but trades top-rank precision — lower
MRR and P@1 than graph-only. Fusion helps coverage, hurts precision@1 here.

### Dietary guardrail

| Tag | Substitutes tested | Valid % |
|-----|--------------------|---------|
| vegan | 45 | **100%** |

Tag-coverage is 100% for the curated dietary vocab used in this test.
Full-vocab coverage will be lower — see Limitations.

### Published reference (pending)

| Metric | food2vec | GISMo | PantryChef (graph) |
|--------|----------|-------|--------------------|
| MRR | _pending manual download_ | _pending_ | 0.339 (mined gold) |
| P@5 | _pending_ | _pending_ | 0.105 (mined gold) |

---

## Limitations

1. **Small mined sample (n=82).** Wide confidence intervals; treat as a baseline
   to beat rather than a definitive benchmark.
2. **Mined gold, not published.** 100% coverage is a mining artifact (same
   corpus → all ingredients known). Numbers are not paper-comparable. Published-
   set evaluation is wired and pending.
3. **28k sample, not full corpus.** The full 2.23M-recipe run is a pending
   quality upgrade; the raw CSV is present locally.
4. **word2vec learns complementarity.** Ingredients that always co-occur
   (salt + pepper) rank as similar even though neither substitutes the other.
   The second-order graph mitigates this, but doesn't eliminate it entirely.
5. **Substring dietary tags can false-positive.** Example: "eggplant" contains
   "egg" — a naive substring match could mis-tag it non-vegan. Phase 4 replaces
   substring matching with an ontology-backed dietary engine.
6. **No GPU used.** word2vec trained CPU-only (~4s); graph is deterministic.
   MX150 is idle until Phase 5 (CV).
