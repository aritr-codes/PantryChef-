# Model Card — Nutrition & Dietary Reasoning (Phase 4)

> Phase 4. Not an LLM. A mostly-deterministic pipeline: tiered USDA matcher +
> coverage-honest mass resolution + ontology dietary engine + a gated macro
> imputer. Numbers are coverage metrics on a 5,000-recipe RecipeNLG sample —
> **there is no recipe-level nutrition gold**, so every recipe macro is an
> **estimate**, not a validated value (see Limitations).

---

## Intended use

Given parsed ingredient lines (name + quantity) for a recipe, produce:
1. Per-ingredient nutrition by joining to USDA FoodData Central, with each line's
   quantity resolved to grams where possible.
2. Per-recipe aggregate macros/micros (totals + per-100g), with unmatched or
   unresolved lines **excluded** from totals and counted toward coverage.
3. Dietary tags (vegan / vegetarian / gluten / animal-product classes) via an
   ontology engine.

**Not intended for:**
- **Medical, clinical, or therapeutic dietary advice.** These are corpus-derived
  estimates, not validated nutrition figures.
- **Allergen safety.** Dietary tags are a best-effort ontology lookup, not an
  allergen-safety guarantee; do not rely on them to keep an allergic person safe.
- **Per-serving nutrition.** RecipeNLG has **no servings field**, so the pipeline
  emits per-recipe and per-100g values only — never per-serving.
- **Precise calorie counting.** Mass resolution uses density and per-item gram
  approximations; unresolved quantities are reported as gaps, not guessed.

---

## Data sources

| Source | Role | License |
|--------|------|---------|
| USDA FoodData Central — **SR Legacy** (release 2018-04, 7,793 foods) | nutrition join | public domain (U.S. Gov) |
| USDA FoodData Central — **Foundation Foods** (release 2025-04-24, 411 foods) | nutrition join | public domain (U.S. Gov) |
| RecipeNLG — 5,000-recipe raw sample (with quantities), 37,472 ingredient lines, seed=42 | ingredient lines | research use (see DATASET.md) |

The two USDA bundles combine into an **8,204-food** artifact
(`data/processed/usda.json`, gitignored; regenerate via
`scripts/fetch_usda.py --csv-dir <dir>`). The Foundation bundle also ships ~73.7k
lab-sampling metadata rows that `fetch_usda.py` filters **out** via a `data_type`
whitelist (only `sr_legacy_food` + `foundation_food` are kept). USDA FDC is public
domain and freely redistributable but requests citation of the release + date; we
gitignore the artifact only because it is large and regenerable. Full provenance
in [DATASET.md](DATASET.md).

---

## Method

### 1 — Tiered ingredient → USDA matcher
Resolution proceeds in order; the first hit wins:
1. **Curated alias** (`data/nutrition/aliases.json`, 34 high-frequency aliases) →
   exact FDC id.
2. **Exact normalized** match on the full ingredient name, then on the
   before-comma head ("flour, all-purpose" → "flour").
3. **Token-Jaccard** similarity at **threshold 0.34**.

Below threshold is a **reported gap, never a guess**. Curating the 34 aliases
lifted match coverage from a **0.736** baseline to **0.800**.

### 2 — Coverage-honest mass resolution
Quantity → grams by route:
- **Mass units** by fixed conversion factor.
- **Volume** via USDA portion gram-weight, else a per-class density fallback.
- **Count / portion** via USDA "each" gram-weight, else curated per-item grams.

Anything unresolved returns `(None, False)` and counts toward the coverage gap —
it is never imputed as `0`. Absent nutrients are reported as `None` (unknown),
never `0.0`.

### 3 — Aggregation
Per-recipe totals + per-100g. Unmatched and unresolved lines are **excluded** from
totals and counted toward coverage, so a recipe with low resolution reads as
low-coverage rather than as a falsely-small total.

### 4 — Ontology dietary engine (`pantrychef/dietary/`)
Token-leaf-word matching with category inheritance
(`animal_product` > `meat` / `fish` / `dairy` / `egg` / `honey`), replacing the
Phase-2 flat-substring keyword tagger. This fixes the substring class of false
positives — concretely **"eggplant" is no longer tagged as containing "egg"** —
plus curated phrase overrides (worcestershire / fish sauce → `fish`, soy sauce →
`gluten`). The old tagger was removed (not run side-by-side), so the fix is
reported qualitatively, not as an accuracy delta.

### 5 — Gated macro imputer (`pantrychef/nutrition/impute.py`)
A HashingVectorizer + Ridge regressor that predicts macros from an ingredient
name, with MAE measured on held-out matched ingredients. It is built **only when a
gate trips**: `match_coverage < 0.80` **and** `median_unresolved_mass > 0.20`. On
the 5k run the gate tripped (0.800 < 0.80 boundary, median unresolved 0.50 > 0.20)
so the imputer was built and validated at unit level. **It is not yet wired into
`RecipeNutrition` totals** — see Limitations.

---

## Metrics

> 5,000-recipe RecipeNLG sample, 37,472 ingredient lines, USDA 8,204-food join,
> seed=42. Run: `uv run python -m pantrychef.eval.nutrition_main --max-rows 5000`.

| Metric | Value |
|--------|-------|
| USDA match coverage | **0.800** (lifted from 0.736 baseline via 34 curated aliases) |
| Mass coverage | 0.487 |
| Nutrition completeness (recipes with ≥50% of lines mass-resolved) | 0.560 |
| Median unresolved-mass fraction | 0.50 |
| Dietary accuracy (n=10 hand-labeled, **sanity check only**) | 1.0 |
| Impute gate | **TRIPPED → imputer built** |

There is **no recipe-macro accuracy number** because no recipe-level nutrition
gold exists. Validation is via coverage metrics, per-ingredient unit tests, and
aggregation arithmetic tests. See [EVALUATION.md](EVALUATION.md).

---

## Limitations

1. **No recipe-macro gold.** RecipeNLG has no nutrition labels, so all recipe
   macros are **estimates** validated only by coverage + unit + arithmetic tests,
   never by ground-truth recipe nutrition. Treat outputs accordingly.
2. **No per-serving values.** No servings field in RecipeNLG; only per-recipe and
   per-100g are emitted.
3. **Mass coverage is partial (0.487).** Just under half of ingredient lines
   resolve to grams; the rest are reported gaps, so per-recipe totals systematically
   undercount mass-unresolved lines rather than guessing them.
4. **Density and per-item-gram approximations.** Volume→gram and count→gram
   conversions use per-class density fallbacks and curated per-item grams; these
   are approximations, not exact for any specific ingredient.
5. **Variable can / package sizes.** "1 can" or "1 package" quantities have no
   universal gram weight; resolution is best-effort and frequently falls to a gap.
6. **Dietary labels are a sanity set (n=10).** The 1.0 dietary accuracy is a smoke
   check, **not** a robust accuracy estimate. Expanding
   `data/nutrition/dietary_labels.json` to ~100 entries is future work.
7. **Imputer not wired into totals.** The gated macro imputer is a standalone,
   unit-validated module. Its predictions are **not** currently fed into
   `RecipeNutrition` aggregates at corpus scale — doing so is future work. Do not
   read the recipe totals as containing imputed macros.
8. **Not for medical / clinical / allergen-safety use.** See Intended use.
