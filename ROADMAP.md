# Roadmap

Each phase is a **standalone tagged release** and could stand alone as a GitHub
project. Status legend: ⬜ not started · 🟡 in progress · ✅ done.

> Full per-phase spec lives in
> [docs/superpowers/specs/2026-05-30-pantrychef-design.md](docs/superpowers/specs/2026-05-30-pantrychef-design.md).

---

## Phase 0 — Engineering Scaffold ✅
- **Goal:** serious-from-day-one foundation before any ML.
- **Deliverables:** repo structure, docs system, `uv`/`ruff`/`pytest` tooling,
  Hydra config skeleton, CI, Docker placeholders.
- **Defer:** all ML.

## Phase 1 — Data Foundation & Ingredient Intelligence ✅
- **Goal:** reproducible recipe corpus + free-text ingredient parser + baseline retrieval.
- **Skills:** data engineering, text normalization, hybrid NER, eval discipline.
- **Metrics (achieved, 50k RecipeNLG sample):** parser **F1 0.864** (target ≥0.85 ✅);
  retrieval **recall@10 0.873**. See [docs/EVALUATION.md](docs/EVALUATION.md).
- **Complexity:** Medium.
- **Notable:** token-pruned match index cut the clean pass ~40× (see
  [docs/CHALLENGES.md](docs/CHALLENGES.md)).
- **Deferred:** embeddings, ML ranking, images, nutrition.

## Phase 2 — FLAGSHIP: Ingredient Substitution (embeddings + graph) ✅
- **Goal:** valid substitutions, no LLM, benchmarked vs food2vec/GISMo.
- **Skills:** representation learning, graph methods, rigorous eval + ablation.
- **Metrics (achieved, 28k RecipeNLG sample, mined gold n=82):**
  graph-only **MRR 0.339** / **recall@10 0.475** (vs food2vec baseline 0.290 / 0.405);
  dietary-validity **100%** (vegan, 45 substitutes). See [docs/EVALUATION.md](docs/EVALUATION.md).
- **Notable:** flagship hypothesis confirmed — second-order context graph beats
  food2vec embedding baseline; no GPU required (word2vec ~4s CPU, graph deterministic).
  Published-gold comparison pending manual download (`scripts/fetch_subs_eval.py` wired).
- **Complexity:** Medium-High.
- **Defer:** recsys integration, serving, full-corpus run (2.23M), published-gold eval.

## Phase 3 — Recommendation & Ranking ✅
- **Goal:** learned constraint-aware ranking (LTR + optional two-tower).
- **Skills:** recsys, LTR, contrastive retrieval, FAISS.
- **Metrics (achieved):** masked recipe-recovery rerank of the P1 candidate pool.
  - **50k sample (seed=13):** LambdaMART **recall@10 0.971 / MRR 0.924** vs P1
    overlap baseline 0.663 / 0.332 (candidate ceiling 0.983).
  - **Full corpus (1.27M, 2026-06-03):** ceiling collapses to **0.706** (retrieval
    becomes the bottleneck at 25× haystack), but the reranker's lead *grows* —
    LambdaMART **recall@10 0.690 vs overlap 0.089 (7.7×)**, recovering 97.7% of
    the ceiling. `sub_fill` (P2) ablation is a confirmed **null**. See
    [docs/EVALUATION.md](docs/EVALUATION.md).
- **Complexity:** High.
- **Notable:** vectorized `candidate_pool` (row-int columnar index + `np.bincount`)
  cut the full-corpus run ~23 h → ~65 min, byte-identical (see
  [docs/CHALLENGES.md](docs/CHALLENGES.md)).
- **Defer:** personalization, online/AB; **first-stage candidate recall** (the
  exposed 0.706 ceiling) — larger/learned candidate generation or ANN.

## Phase 4 — Nutrition & Dietary Reasoning ⬜
- **Goal:** per-serving macro/micro + dietary engine (mostly deterministic).
- **Skills:** knowledge integration, constraint logic, light regression.
- **Metrics:** nutrition coverage %, dietary-class accuracy, imputation MAE.
- **Complexity:** Medium.
- **Defer:** deep micronutrients if data sparse.

## Phase 5 — Computer Vision: Ingredient Detection ⬜
- **Goal:** fridge/pantry photo → detected ingredients → pipeline.
- **Skills:** transfer learning, model export, quantization, edge inference.
- **Metrics:** mAP@0.5, MX150 inference latency.
- **Complexity:** Medium-High.
- **Defer:** segmentation, quantity-from-image.

## Phase 6 — Serving, Demo & MLOps ⬜
- **Goal:** reproducible service + live free demo.
- **Skills:** serving, containerization, CI/CD for ML, monitoring.
- **Metrics:** API latency, demo uptime, eval-regression gate firing.
- **Complexity:** High.
- **Defer:** k8s/autoscaling/paid infra (out of scope).
