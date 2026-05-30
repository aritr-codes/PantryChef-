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
- **Preprocessing:** _tbd Phase 1_ — dedup, canonical-ingredient mapping,
  train/val/test splits (record split seed + sizes here).

### USDA FoodData Central  (Phase 4 — nutrition)
- **Why:** authoritative free nutrition data (macros + micros).
- **Access:** free API + bulk CSV download.
- **License:** public domain (US Gov).
- **Use:** join parsed ingredient + quantity → per-serving nutrition.

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
