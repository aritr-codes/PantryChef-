# Architecture

> Living document. Update whenever module boundaries, data flow, or tech
> choices change. Diagram is ASCII for now; replace with an image in Phase 6.

## Principles

1. **Logic lives in the package, not notebooks.** Notebooks `pip install` the
   repo and call it. Same code runs local / Colab / Kaggle.
2. **Each module has one purpose, a typed interface, and is testable alone.**
3. **ML where it earns its place; rules where they suffice.** (See README table.)
4. **Reproducible by default:** `uv` lock + Hydra configs + MLflow runs.

## Module map

```
                        ┌──────────────────────────┐
   text pantry  ─────▶  │ ingredients (parse/canon) │
   photo pantry ──┐     └────────────┬──────────────┘
                  │                  │ canonical ingredients
        ┌─────────▼─────────┐        │
        │ vision (detect →  │────────┘
        │ canonical map)    │
        └───────────────────┘        ▼
                          ┌──────────────────────┐
                          │ retrieval (overlap /  │
                          │ FAISS candidate set)  │
                          └───────────┬───────────┘
                                      ▼
        ┌──────────────┐   ┌──────────────────────┐   ┌───────────────────┐
        │ substitution │──▶│ recommender (LTR /    │◀──│ nutrition (USDA + │
        │ (emb + graph)│   │ two-tower rerank)     │   │ dietary ontology) │
        └──────────────┘   └───────────┬───────────┘   └───────────────────┘
                                      ▼
                          ┌──────────────────────┐
                          │ serving (FastAPI API) │ ──▶ hosted demo
                          └──────────────────────┘
```

## Package modules (`pantrychef/`)

| Module          | Purpose                                            | Phase |
| --------------- | -------------------------------------------------- | ----- |
| `config`        | Typed runtime settings (pydantic-settings)         | 0     |
| `common`        | Logging, IO, shared types                          | 0     |
| `data`          | Ingest, cleaning, schemas, splits                  | 1     |
| `ingredients`   | Free-text parser + canonical vocabulary            | 1     |
| `retrieval`     | Set-overlap baseline + FAISS candidate retrieval   | 1/3   |
| `substitution`  | Ingredient embeddings + relation graph (flagship)  | 2     |
| `recommender`   | Learning-to-rank + two-tower reranking             | 3     |
| `nutrition`     | USDA mapping + dietary ontology (deterministic)    | 4     |
| `vision`        | Detector training glue + ONNX inference            | 5     |
| `serving`       | FastAPI inference API                              | 6     |

## Data flow contracts

- **Pantry in:** `list[str]` raw ingredient strings (typed) **or** image →
  `vision` → `list[str]`.
- **`ingredients.parse`** → `list[ParsedIngredient]` (qty, unit, canonical, modifier).
- **`retrieval`** → candidate `list[RecipeId]` + match metadata.
- **`substitution`** → per missing ingredient, ranked `list[Substitute]` with
  dietary-validity flag.
- **`recommender`** → ranked `list[ScoredRecipe]`.
- **`nutrition`** → `NutritionFacts` per serving + dietary labels.

> Concrete dataclasses/pydantic models defined in `pantrychef/common/types.py`
> as each phase lands. Phase 0 ships placeholders only.

## Tooling

- **Env/deps:** `uv` (+ `uv.lock`).
- **Lint/format:** `ruff`.
- **Tests:** `pytest`.
- **Config:** `pydantic-settings` (runtime) + Hydra (experiments, `configs/`).
- **Tracking:** MLflow (self-hosted, free).
- **Serving/demo:** FastAPI + Docker + HF Spaces / Streamlit.
