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
| Metric | food2vec | GISMo | PantryChef | Run |
| ------ | -------- | ----- | ---------- | --- |
| precision@5 | _tbd_ | _tbd_ | _tbd_ | — |
| MRR | _tbd_ | _tbd_ | _tbd_ | — |
| dietary-validity % (guardrail) | — | — | _tbd (target 100%)_ | — |

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
