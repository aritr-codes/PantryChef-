# Phase 4 — Nutrition & Dietary Reasoning — Design

> Status: design approved 2026-06-03. Next: implementation plan (writing-plans).
> Roadmap: [ROADMAP.md](../../../ROADMAP.md) Phase 4. Project spec:
> [2026-05-30-pantrychef-design.md](2026-05-30-pantrychef-design.md).

## Goal

Given a recipe's raw ingredient lines (typed or pasted — the real use case),
produce **per-recipe** and **per-100g** macro/micro nutrition by joining parsed
ingredients to USDA FoodData Central, and replace the brittle substring dietary
tagger with an **ontology-backed dietary engine**. Stay no-LLM, deterministic,
laptop-friendly. The thesis ("ML engineering, not CRUD") is carried by the
ingredient→USDA entity-resolution matcher, the ontology dietary classifier
(accuracy metric), and a **gated** macro-imputation regressor.

## Scope decision (approved, codex-reviewed)

Build **nutrition aggregation + ontology dietary engine**; build the imputation
regressor **only if a measured coverage gate trips** (below). This separates
validated need from speculative modeling — an imputer trained on sparse, noisy
targets when USDA already covers common ingredients would be "ML theater."

- **Denominator:** per-recipe total **and** per-100g. **No per-serving** —
  RecipeNLG has no servings field, and inferring it from directions adds noisy
  NLP and fakes precision the data can't support.
- **Mass resolution:** USDA portion gram-weights + a density-table fallback for
  volume units; ingredients we cannot mass-resolve are **dropped from the total
  and reported as a coverage gap** (coverage-honest).
- **Matcher:** tiered deterministic (alias → exact → token-Jaccard) in v1;
  escalate to a heavier **TF-IDF / embedding nearest-neighbor** fallback only if
  measured match-coverage from the v1 tiers is poor (same measure-then-escalate
  discipline; the Jaccard tier is *part of* v1, the NN fallback is the escalation).

## Honesty constraint (load-bearing)

RecipeNLG has **no recipe-level nutrition labels**. We therefore **cannot** claim
recipe-macro accuracy. The evaluation validates: (1) **coverage** (match + mass),
(2) **per-ingredient correctness** against USDA (authoritative source), (3)
**aggregation correctness** via unit tests on hand-built recipes, and (4) face
validity on spot-checks. This is the Phase-4 analog of Phase-2's coverage
honesty. Absolute recipe macros are presented as **estimates**, never as
validated ground truth.

## Architecture

New package `pantrychef/nutrition/` (on-demand module) + an eval harness, mirroring
the P1–P3 "module + `eval/*_main.py`" convention. The dietary engine moves to a
**neutral shared package** `pantrychef/dietary/` (used by both P2 substitution and
P4), with an import shim so P2 is behaviorally untouched.

### Module layout

| File | Responsibility | Depends on |
|---|---|---|
| `nutrition/usda.py` | Load USDA FDC bulk CSVs (SR Legacy + Foundation Foods — generic foods, ~few-thousand entries; **not** the 1M+ Branded set) → compact artifact `fdc_id → {description, per-100g nutrients, portions:[{modifier, gram_weight}]}`. Cached to `data/processed/usda.json`. | — |
| `nutrition/match.py` | Tiered `canonical_ingredient → fdc_id`: alias table → exact normalized → token-Jaccard over descriptions. Returns `(fdc_id|None, method, score)`. Unmatched = reported gap, **no guessing** in v1. | `usda`, P1 `normalize` |
| `nutrition/mass.py` | `(qty, unit, fdc_id) → (grams|None, resolved)`: fixed mass/volume factors → ml → grams via USDA portion gram-weights, else density-by-class fallback, else unresolved. | `usda`, P1 `units` |
| `nutrition/aggregate.py` | raw ingredient lines → `NutritionFacts` per-recipe **and** per-100g + coverage stats. Re-parses raw lines (quantities), matches, masses, sums. | `match`, `mass`, P1 `parser` |
| `nutrition/impute.py` *(gated)* | Light regression (ridge/GBT) predicting per-100g macros for unmatched ingredients. Built only if the gate trips. | `match`, `aggregate`, `dietary` |
| `dietary/ontology.py` | Category **hierarchy + inheritance** → `tags()`, `is_valid(diet)`, `mask()`, `coverage()`. Ontology data in `dietary/ontology_data.py` (or JSON). | P1 `vocab/normalize` |
| `dietary/__init__.py` | Re-exports `DietTagger`/ontology API. | — |
| `substitution/dietary.py` | **Shim** re-exporting from `pantrychef.dietary` (keeps P2 imports + behavior stable). | `dietary` |

### Data sources & artifacts

- `scripts/fetch_usda.py` — download FDC SR Legacy + Foundation bundles, parse
  `food.csv` / `food_nutrient.csv` / `nutrient.csv` / `food_portion.csv`, emit the
  compact `usda.json`. Raw CSVs gitignored.
- **Commit the derived `usda.json`** — it is small (~MB, only whitelisted
  nutrients for a few-thousand generic foods) and **public domain (US Gov)**, so
  committing it makes the eval reproducible with no download. (Exception to the
  "data gitignored" norm, justified by size + license + reproducibility.)
- `data/nutrition/aliases.json` — curated `canonical → fdc_id` overrides for
  high-frequency ingredients (committed, hand-maintained).
- `data/nutrition/dietary_labels.json` — hand-labeled `ingredient → {categories}`
  eval set for dietary-class accuracy (committed).

### Data flow

```
raw line → parse(qty, unit, canonical)
         → match: canonical → fdc_id            (else: unmatched gap)
         → mass: (qty, unit, fdc_id) → grams     (else: unresolved gap)
         → per-100g nutrients × grams/100
Σ over resolved lines = recipe total
per-100g = recipe total / Σ grams × 100
coverage = (matched ∧ massed lines) / all lines
```

## Component internals

### `usda.py`
Whitelist nutrients (by USDA `nutrient_id`): Energy (kcal), Protein, Total lipid
(fat), Carbohydrate by difference; + fiber, total sugars; + micros sodium,
calcium, iron. (Defer deeper micros if sparse.) Normalize all to **per-100g**
(USDA `food_nutrient.amount` is already per 100g for SR/Foundation). Portions:
keep `(modifier, gram_weight)` rows for mass resolution.

### `match.py` (tiered)
1. **Alias table** (Tier 0) — curated `canonical → fdc_id`.
2. **Exact normalized** (Tier 1) — `canonicalize(name)` against a prebuilt
   normalized-description index.
3. **Token-Jaccard** (Tier 2) — best Jaccard over description token sets above a
   threshold; deterministic tie-break (score, then fdc_id). Below threshold →
   `None`.
Returns `(fdc_id|None, method, score)`. Match-precision measured on a ~50–100
hand-labeled set.

### `mass.py`
- Mass units → grams: g 1, kg 1000, oz 28.3495, lb 453.592.
- Volume → ml: ml 1, l 1000, cup 236.588, tbsp 14.787, tsp 4.929, quart 946.353,
  pint 473.176 → grams via USDA portion gram-weight (match the modifier) else
  **density by ingredient class** (liquid ~1.0, flour ~0.53, sugar ~0.85, …) else
  unresolved.
- Portion/count: clove, can, slice, stick, **bare count** ("2 eggs") → USDA
  `food_portion` gram-weight by modifier → curated portion table → unresolved.
- pinch/dash → fixed ~0 estimate (immaterial to macros).
- Returns `(grams|None, resolved)`. Unresolved excluded from totals + counted in
  the mass-coverage gap.

### `dietary/ontology.py`
Category hierarchy with inheritance, e.g.:
```
animal_product → meat → {beef, pork, chicken, …}
animal_product → {dairy → {milk, butter, cheese, …}, egg, honey, fish → {salmon, …}}
gluten_grain → {wheat, barley, rye}
```
Ingredient → leaf(s) via curated mapping + inherited ancestors. Diets are
forbidden **category** sets; forbidding a parent forbids all descendants. Fixes
substring false-positives (`eggplant` is mapped explicitly / via word-boundary,
not a substring of "egg"). `coverage()` = vocab fraction mapped to ≥1 leaf.
**Same public API** (`tags/is_valid/mask/coverage`) as today's `DietTagger`, so
the P2 guardrail keeps working — with a **parity test** guaranteeing P2's 100%
dietary-validity cannot regress.

### `impute.py` (gated)
**Gate:** build the imputer **iff**, on the validation sample, match coverage
< **80%** OR median per-recipe unresolved-mass fraction > **20%**. Else
EVALUATION records "imputation not needed — coverage adequate" (honest non-build,
like the sub_fill null). **If built:** features = ontology category + name tokens
+ density class; model = ridge or gradient-boosted trees; **MAE measured on
held-out matched ingredients** (train on matched, test on held-out matched),
then applied to unmatched.

## Evaluation

`pantrychef/eval/nutrition_eval.py` + `nutrition_main.py`, on a seeded sample of
raw recipes from `data/raw`:

| Metric | Definition |
|---|---|
| **Match coverage %** | ingredient occurrences mapped to an `fdc_id` |
| **Mass coverage %** | matched lines resolved to grams (+ per-recipe resolved-fraction distribution) |
| **Nutrition completeness %** | recipes with ≥ 50% of ingredient mass resolved (usable nutrition) |
| **Dietary-class accuracy** | vs `dietary_labels.json`, with ontology coverage %, **compared to the old substring tagger** (improvement story) |
| **Imputation MAE** | only if the gate trips |

Results recorded in [EVALUATION.md](../../EVALUATION.md) with the honesty
constraint stated. Numbers logged to MLflow per project convention.

## Testing (TDD)

- `usda`: fixture CSVs → artifact shape + nutrient whitelist + per-100g values.
- `match`: alias/exact/Jaccard tiers; unmatched → `None`; tricky cases.
- `mass`: each unit class → grams (exact factors); volume via portion vs density
  fallback; portion/count path; unresolved path.
- `aggregate`: hand-built recipe with known grams + per-100g → **exact** expected
  totals, per-100g, and coverage; unresolved ingredient excluded from totals.
- `ontology`: inheritance; `eggplant ≠ egg`; vegan forbids descendants;
  conservative-unknown; `coverage()`; **parity/superset vs old `DietTagger`** on
  its existing cases.
- shim: `substitution` imports still resolve; P2 tests stay green.

## Out of scope / risks

- **Per-serving** nutrition (no servings field) → per-recipe + per-100g only.
- **Absolute macro accuracy** (no recipe-level gold) → coverage + per-ingredient +
  face validity only; macros are estimates.
- **Density approximations** → volume→mass is approximate; report mass coverage,
  don't overclaim precision.
- **Branded foods, allergen safety, medical/clinical advice** → explicitly out.
- **Variable "1 can"/"1 package" sizes** → USDA portion / curated best-effort,
  else gap.

## Deliverables

`pantrychef/nutrition/` (usda, match, mass, aggregate, [impute]), shared
`pantrychef/dietary/` (ontology + shim), `scripts/fetch_usda.py`, committed
`data/processed/usda.json` + `data/nutrition/{aliases,dietary_labels}.json`,
`pantrychef/eval/nutrition_{eval,main}.py`, EVALUATION.md + ROADMAP updates, full
test suite. Standalone-taggable as `v0.4.0`.
