# Roadmap

Each phase is a **standalone tagged release** and could stand alone as a GitHub
project. Status legend: ⬜ not started · 🟡 in progress · ✅ done.

> Full per-phase spec lives in
> [docs/superpowers/specs/2026-05-30-pantrychef-design.md](docs/superpowers/specs/2026-05-30-pantrychef-design.md).

---

## Phase 0 — Engineering Scaffold 🟡
- **Goal:** serious-from-day-one foundation before any ML.
- **Deliverables:** repo structure, docs system, `uv`/`ruff`/`pytest` tooling,
  Hydra config skeleton, CI, Docker placeholders.
- **Defer:** all ML.

## Phase 1 — Data Foundation & Ingredient Intelligence ⬜
- **Goal:** reproducible recipe corpus + free-text ingredient parser + baseline retrieval.
- **Skills:** data engineering, text normalization, hybrid NER, eval discipline.
- **Metrics:** parser F1 ≥0.85 (canonical match); retrieval recall@k baseline.
- **Complexity:** Medium.
- **Defer:** embeddings, ML ranking, images, nutrition.

## Phase 2 — FLAGSHIP: Ingredient Substitution (embeddings + graph) ⬜
- **Goal:** valid substitutions, no LLM, benchmarked vs food2vec/GISMo.
- **Skills:** representation learning, graph methods, rigorous eval + ablation.
- **Metrics:** precision@k / MRR vs published numbers; dietary-validity rate.
- **Complexity:** Medium-High.
- **Defer:** recsys integration, serving.

## Phase 3 — Recommendation & Ranking ⬜
- **Goal:** learned constraint-aware ranking (LTR + optional two-tower).
- **Skills:** recsys, LTR, contrastive retrieval, FAISS.
- **Metrics:** NDCG@k, recall@k vs P1 baseline.
- **Complexity:** High.
- **Defer:** personalization, online/AB.

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
