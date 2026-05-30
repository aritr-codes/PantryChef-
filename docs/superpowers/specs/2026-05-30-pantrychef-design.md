# PantryChef — Design Spec

**Date:** 2026-05-30
**Status:** Approved
**Type:** Portfolio ML-engineering project (multi-phase)

---

## 1. Problem

Given what's in a user's kitchen (typed or photographed) plus dietary constraints,
PantryChef surfaces *makeable* recipes, fills ingredient gaps with *valid*
substitutions, and quantifies nutrition — **without** delegating the hard ML
problems to a black-box LLM API.

### Why it is ML-interesting (the anti-CRUD moat)

The valuable behaviour resists a `WHERE ingredient IN (...)` query. "What
substitutes for buttermilk in *this* recipe, given vegan + gluten-free?" is a
learned-representation + constrained-graph problem. That is where we build,
not call.

### ML vs deterministic split (deliberate)

Keeping nutrition/dietary logic rule-based is itself a signal: we do not
ML-everything.

| Built with ML                                   | Deterministic on purpose            |
| ----------------------------------------------- | ----------------------------------- |
| Ingredient NER (entity extraction)              | Quantity/unit parsing (rules)       |
| **Substitution: embeddings + graph (flagship)** | Dietary classification (ontology)   |
| Recommendation: learning-to-rank + two-tower    | Nutrition aggregation (USDA join)   |
| CV ingredient detection                         | Constraint filtering / set-overlap  |
| (optional) nutrition imputation regression      |                                     |

## 2. Decisions (locked)

- **Repo:** monorepo + tagged releases; each phase a standalone release.
- **Flagship:** ingredient substitution (embeddings + graph), benchmarked vs
  `food2vec` / GISMo.
- **Deployment:** reproducible local (Docker) + a free hosted demo
  (HF Spaces / Streamlit). No paid infra.
- **Role target:** broad full-stack ML story (undecided specialism).
- **Phase 1 start:** data foundation + ingredient parser + baseline retrieval.
- **Python stack:** `uv` + `ruff` + `pydantic-settings` + `pytest`.
- **ML tooling:** Hydra configs + MLflow tracking (self-hosted, free).

## 3. Technical risks + mitigations

1. **Ingredient parsing mess** (free text — biggest hidden cost) → hybrid
   rules + NER, seed from RecipeNLG NER, labeled eval set early.
2. **Substitution eval validity** (ground truth sparse/subjective) → published
   test sets, precision@k + MRR + dietary-validity guardrail.
3. **Free-tier storage** (Recipe1M = 13M images) → text-only RecipeNLG for
   P1–P4; sample images only in P5.
4. **6-phase scope creep** → each phase ships standalone; enforced defer lists.
5. **Colab/Kaggle limits** (timeouts, GPU quota) → checkpointing, thin
   notebooks, small models, ONNX inference local.
6. **Local↔cloud repro drift** → `uv` lock + same package installed
   everywhere + Makefile.

## 4. Datasets (validated feasible on free tier)

- **RecipeNLG** — 2.23M recipes, ships food-entity NER. Text-only, small.
  HuggingFace mirror `mbien/recipe_nlg`.
- **Recipe1M+** — 1M recipes + 13M images. Multimodal; image dump huge →
  defer/sample.
- **CV detection** — Roboflow fridge/grocery sets (~1–5k images, YOLO-ready).
- **Nutrition** — USDA FoodData Central (free API + bulk download).

### Research lineage to benchmark against

- `food2vec` / `foodBERT` — word2vec/BERT on Recipe1M → nearest-neighbour subs.
- `ingredient2Vec` — vegan substitutes.
- **GISMo** — graph-based substitution (recipe context + ingredient relations).
- Substitution ground-truth test set ships with the "Exploiting Food
  Embeddings" repo → real precision@k, not vibes.

## 5. Roadmap (6 phases, each a tagged standalone release)

### Phase 1 — Data Foundation & Ingredient Intelligence
- **Goal:** reproducible recipe corpus + free-text ingredient parser + baseline retrieval.
- **Scope:** ingest RecipeNLG, canonical ingredient vocab, parser
  (qty/unit/canonical/modifier), set-overlap retrieval, eval-harness skeleton.
- **Deliverables:** data pipeline, `ingredients` parser module, CLI
  (pantry→ranked recipes), parser P/R/F1 report, README demo.
- **Skills:** data engineering, text normalization, hybrid NER, eval discipline, reproducibility.
- **Metrics:** parser F1 on labeled held-out (target ≥0.85 canonical match); retrieval recall@k baseline.
- **Complexity:** Medium.
- **Defer:** embeddings, ML ranking, images, nutrition.

### Phase 2 — FLAGSHIP: Ingredient Substitution (embeddings + graph)
- **Goal:** valid substitutions, no LLM, benchmarked vs food2vec/GISMo.
- **Scope:** train ingredient embeddings (skip-gram on recipe ingredient
  sets)→kNN; co-occurrence + relation graph; role/dietary filtering;
  **ablation** embeddings-only vs graph vs hybrid.
- **Deliverables:** Colab/Kaggle training notebook, `substitution` module,
  MLflow runs, EVALUATION leaderboard vs baselines, model card.
- **Skills:** representation learning, embeddings, graph methods, rigorous eval + ablation, experiment tracking.
- **Metrics:** precision@k / MRR vs published numbers; dietary-validity rate (zero vegan→animal subs).
- **Complexity:** Medium-High.
- **Defer:** full recsys integration, serving.

### Phase 3 — Recommendation & Ranking
- **Goal:** from "contains X" to learned constraint-aware ranking.
- **Scope:** feature LTR (match%, sub-distance, dietary fit, popularity);
  optional two-tower contrastive retrieval; rerank where subs fill gaps.
- **Deliverables:** `recommender` module, FAISS index, LTR model, NDCG/recall report.
- **Skills:** recsys, LTR, contrastive retrieval, FAISS, ranking metrics.
- **Metrics:** NDCG@k, recall@k vs P1 baseline.
- **Complexity:** High.
- **Defer:** personalization, online/AB.

### Phase 4 — Nutrition & Dietary Reasoning (mostly deterministic)
- **Goal:** per-serving macro/micro + dietary engine.
- **Scope:** parsed ingredient+qty → USDA join → per-serving aggregate;
  dietary classification via ontology; optional imputation regression.
- **Deliverables:** `nutrition` module, ontology/rules, optional imputation model + eval.
- **Skills:** knowledge integration, constraint logic, data joins, light regression.
- **Metrics:** nutrition coverage %, dietary-class accuracy, imputation MAE (if built).
- **Complexity:** Medium.
- **Defer:** deep micronutrients if data sparse.

### Phase 5 — Computer Vision: Ingredient Detection (multimodal)
- **Goal:** fridge/pantry photo → detected ingredients → into pipeline.
- **Scope:** fine-tune YOLO on Roboflow fridge/grocery set; export ONNX;
  quantize; local MX150 inference; map detections→canonical vocab (reuse P1).
- **Deliverables:** training notebook, `vision` module, ONNX model, local inference demo, mAP report.
- **Skills:** transfer learning, labeling/augmentation, model export, quantization, edge inference, multimodal glue.
- **Metrics:** mAP@0.5, MX150 inference latency.
- **Complexity:** Medium-High.
- **Defer:** segmentation, quantity-from-image.

### Phase 6 — Serving, Demo & MLOps
- **Goal:** reproducible service + live free demo.
- **Scope:** FastAPI + ONNX runtime + FAISS, Docker Compose, MLflow registry,
  CI eval gates, HF Spaces/Streamlit demo, prediction logging + drift check.
- **Deliverables:** API, Dockerfiles, CI workflows, live demo URL, architecture diagram, model cards.
- **Skills:** serving, containerization, CI/CD for ML, monitoring.
- **Metrics:** API latency, demo uptime, eval-regression gate firing.
- **Complexity:** High.
- **Defer:** k8s/autoscaling/paid infra (out of scope by constraint).

## 6. Team roles / working agreement

- **Senior ML Eng / Tech Lead (Claude):** propose architecture, give
  options+tradeoffs *before* code, write ADRs, review code, set eval bars,
  push back on over-engineering.
- **ML Eng (user):** implement, run experiments, co-decide, log results.
- Every implementation leads with *why + alternatives + tradeoffs*.
  Decisions → ADR. Experiments → EXPERIMENTS.md + MLflow.
  Quality + learning over speed.

## 7. Recruiter-memory hook

"Built a no-LLM ingredient-substitution engine benchmarked against published
research (food2vec/GISMo), wrapped in a reproducible multimodal cooking
assistant with a live demo."

## 8. Out of scope (whole project)

- Any LLM dependency for core features.
- Paid cloud, enterprise GPUs, hosted vector DBs with cost.
- Mobile app, user accounts, real-time collaboration.
