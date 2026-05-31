# Evaluation

The proof that PantryChef is ML engineering, not CRUD. Every ML component has a
defined metric, a protocol, and a number compared against a baseline.

> Update on every evaluation run. Link MLflow runs.

## Protocol principles

- Fixed held-out splits (seed recorded in [DATASET.md](DATASET.md)).
- Compare against a **named baseline** (published or naive), never in a vacuum.
- Report >1 metric where the task is multi-faceted.
- Guardrail metrics (e.g. dietary validity) gate "good" scores.

## Leaderboards

> **Phase 1 eval provenance:** RecipeNLG, **50,000-recipe sample**, `min_count=5`
> (vocab = 2,067 canonical ingredients), seed=42. Sample-based by design:
> parser eval is fast (~37s) but retrieval leave-one-out is O(N²) on shared
> postings (see [CHALLENGES.md](CHALLENGES.md)), so it runs on the sample, not
> the full 2.23M corpus. Numbers are a baseline to beat in Phase 2+.

### Phase 1 — Ingredient parser
| Metric | Baseline | Current | Target | Run |
| ------ | -------- | ------- | ------ | --- |
| Canonical-match precision | — | 0.860 | — | 50k, 2026-05-31 |
| Canonical-match recall | — | 0.869 | — | 50k, 2026-05-31 |
| **Canonical-match F1** | — | **0.864** | **≥0.85 ✅** | 50k, 2026-05-31 |

_Counts: tp=315,224 fp=51,379 fn=47,456 (micro-averaged over the sample).
Gold = RecipeNLG `NER` column; both gold and prediction canonicalized via the
same `canonicalize()` path._

### Phase 1 — Retrieval baseline
| Metric | Baseline | Current | Run |
| ------ | -------- | ------- | --- |
| **recall@10** (leave-one-ingredient-out) | — | **0.873** | 50k (n=28,183), 2026-05-31 |

_Protocol: for each recipe with ≥2 canonical ingredients, drop one at random
(seed=42), query with the rest, hit if the source recipe is in top-10. Set-overlap
coverage scoring. This is the number Phase 3's learned ranker must beat._

### Phase 2 — Substitution (flagship)

> **Phase 2 eval provenance:** RecipeNLG **28,273-recipe sample**, seed=42,
> `min_overlap=4`. Gold pairs **mined from the same corpus** (near-duplicate
> recipes differing by exactly one ingredient): 116 directed pairs, 82 evaluated
> query keys. Coverage is 100% **by construction** (an artifact of mining from
> the same corpus — not a quality claim). This is the reproducible fallback;
> comparison against the published food2vec / GISMo gold set is wired via
> `scripts/fetch_subs_eval.py` and pending a manual download. Numbers are
> **NOT directly comparable to published food2vec / GISMo results.**

#### Ablation (n=82 queries)

| Arm | MRR | P@1 | R@1 | P@5 | R@5 | P@10 | R@10 | Run |
| --- | --- | --- | --- | --- | --- | ---- | ---- | --- |
| emb-only (food2vec baseline) | 0.290 | 0.232 | 0.151 | 0.093 | 0.302 | 0.057 | 0.405 | 28k, 2026-05-31 |
| **graph-only (FLAGSHIP) ★** | **0.339** | **0.256** | **0.189** | **0.105** | **0.395** | **0.063** | **0.475** | 28k, 2026-05-31 |
| hybrid (alpha=0.5) | 0.324 | 0.232 | 0.157 | 0.100 | 0.347 | 0.067 | 0.503 | 28k, 2026-05-31 |

★ **Flagship beats baseline:** graph-only MRR 0.339 vs food2vec 0.290 (+17%);
recall@10 0.475 vs 0.405. Hybrid gives best recall@10 (0.503) but lower MRR/P@1.

#### Dietary guardrail

| Tag | Substitutes tested | Valid % | Run |
| --- | ------------------ | ------- | --- |
| vegan | 45 | **100%** | 2026-05-31 |

_Tag-coverage is 100% only because the curated set uses common, well-tagged
ingredients. Full-vocab coverage is lower — Phase 4 fixes with an ontology._

#### Published reference (pending manual download)

| Metric | food2vec | GISMo | PantryChef (graph, mined gold) |
| ------ | -------- | ----- | ------------------------------ |
| MRR | _pending manual download_ | _pending_ | 0.339 |
| P@5 | _pending_ | _pending_ | 0.105 |

### Phase 3 — Recommendation
| Metric | Baseline (P1 overlap) | Current | Run |
| ------ | --------------------- | ------- | --- |
| NDCG@10 | _tbd_ | _tbd_ | — |
| recall@10 | _tbd_ | _tbd_ | — |

### Phase 5 — Detection
| Metric | Current | Run |
| ------ | ------- | --- |
| mAP@0.5 | _tbd_ | — |
| MX150 latency (ms/img) | _tbd_ | — |
