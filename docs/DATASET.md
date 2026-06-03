# Datasets

Provenance, licenses, schema, and preprocessing for every data source. Data
files themselves are gitignored — this file is the source of truth for *what*
the data is and *how to reproduce* it.

> Update whenever a dataset is added, a version changes, or preprocessing
> changes.

## Sources

### RecipeNLG  (Phase 1 — primary recipe corpus)
- **Size:** ~2.23M recipes; text-only (small footprint).
- **Why:** ships food-entity NER; large + deduplicated; fits free tier.
- **Access:** HuggingFace `mbien/recipe_nlg`; original at
  `recipenlg.cs.put.poznan.pl`.
- **License:** check dataset card before redistribution (research use).
- **Schema (raw):** `title`, `ingredients` (free text), `directions`, `NER`
  (extracted food entities), `link`, `source`.
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

### USDA FoodData Central  (Phase 4 — nutrition)
- **Why:** authoritative free nutrition data (macros + micros).
- **Access:** free bulk CSV download (FoodData Central full-download datasets).
- **Bundles used (record release + date):**
  - **SR Legacy** bundle release **2018-04** — 7,793 `sr_legacy_food` entries.
  - **Foundation Foods** bundle release **2025-04-24** — 411 `foundation_food`
    entries. (This bundle also ships ~73.7k lab-sampling metadata rows —
    `sub_sample_food` / `market_acquisition` / `sample_food` /
    `agricultural_acquisition` — which `scripts/fetch_usda.py` filters **out**
    via a `data_type` whitelist; only `sr_legacy_food` + `foundation_food` are
    kept.)
- **Artifact:** combined → `data/processed/usda.json` = **8,204 foods**
  (gitignored — large and regenerable).
- **License:** **public domain** (U.S. Government work, no copyright). FDC is
  freely redistributable; it requests citation of the **release + date**. (Unlike
  the GISMo benchmark below, which is CC BY-NC.) We still gitignore the artifact
  because it is large and regenerable.
- **Use:** join parsed ingredient + quantity → per-recipe nutrition **estimates**
  (no per-serving — RecipeNLG has no servings field).
- **Regenerate:** `uv run python scripts/fetch_usda.py --csv-dir <dir>` (point at
  the unzipped FDC CSV directory). Record release dates + food count of each run
  here.

### Roboflow fridge / grocery detection sets  (Phase 5 — CV)
- **Why:** YOLO-ready labeled images, free, small (~1–5k images).
- **Access:** Roboflow Universe (fridge-objects, fruits-and-vegetables).
- **License:** per-dataset on Roboflow — record exact one chosen here.
- **Use:** fine-tune detector → ONNX → local inference.

### Recipe1M+  (deferred — multimodal)
- **Size:** 1M recipes + 13M images (image dump huge).
- **Status:** **deferred** — storage risk on free tier. Sample only if needed.

### Substitution ground-truth test set  (Phase 2 — eval)
- **Source:** "Exploiting Food Embeddings for Ingredient Substitution" repo.
- **Use:** precision@k / MRR evaluation of substitution model.

## Reproducibility

- Download via `scripts/download_data.py` (Phase 1).
- Raw → `data/raw/`, cleaned → `data/interim/`, final → `data/processed/`.
- Record dataset version + download date + row counts per source here.
