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

### Phase 1 — Ingredient parser
| Metric | Baseline | Current | Target | Run |
| ------ | -------- | ------- | ------ | --- |
| Canonical-match F1 | rule-only _tbd_ | _tbd_ | ≥0.85 | — |

### Phase 1 — Retrieval baseline
| Metric | Baseline | Current | Run |
| ------ | -------- | ------- | --- |
| recall@10 | random _tbd_ | _tbd_ | — |

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
