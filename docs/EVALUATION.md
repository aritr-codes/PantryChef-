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
> comparison against the **published GISMo gold set** is implemented via
> `scripts/fetch_gismo_gold.py` — see the "vs GISMo published benchmark" table
> below. Mined-gold numbers here are **NOT directly comparable to published
> food2vec / GISMo results** (different gold, 100%-by-construction coverage).

#### Ablation (n=82 queries)

| Arm | MRR | P@1 | R@1 | P@5 | R@5 | P@10 | R@10 | Run |
| --- | --- | --- | --- | --- | --- | ---- | ---- | --- |
| emb-only (food2vec baseline) | 0.290 | 0.232 | 0.151 | 0.093 | 0.302 | 0.057 | 0.405 | 28k, 2026-05-31 |
| **graph-only (FLAGSHIP) ★** | **0.339** | **0.256** | **0.189** | **0.105** | **0.395** | **0.063** | **0.475** | 28k, 2026-05-31 |
| hybrid (alpha=0.5) | 0.324 | 0.232 | 0.157 | 0.100 | 0.347 | 0.067 | 0.503 | 28k, 2026-05-31 |

★ **Flagship beats baseline:** graph-only MRR 0.339 vs food2vec 0.290 (+17%);
recall@10 0.475 vs 0.405. Hybrid gives best recall@10 (0.503) but lower MRR/P@1.

#### Ablation — full corpus (1,274,290 recipes, vocab 30,481, n=82)

> **Provenance:** full RecipeNLG 2.23M cleaned (1.27M after title-dedup),
> `min_count=5` → vocab 30,481, seed=42. **Same** mined gold (116 pairs / 82
> query keys) as the 28k row, so the arms are comparable **to each other**.
> Absolute numbers are **not** comparable to the 28k row: the candidate space is
> ~15× larger (a strictly harder ranking task), so all arms score lower.

| Arm | MRR | R@5 | R@10 | Run |
| --- | --- | --- | ---- | --- |
| emb-only (food2vec baseline) | 0.169 | 0.201 | 0.315 | full, 2026-06-01 |
| graph-only, **no** support guard | 0.151 | 0.151 | 0.264 | full, 2026-06-01 |
| **graph-only + overlap-shrink (β=100) ★** | **0.176** | 0.225 | **0.316** | full, 2026-06-01 |
| **hybrid (alpha=0.5) + shrink** | **0.201** | **0.281** | **0.352** | full, 2026-06-01 |

★ **The flagship's lead reverses at full scale without a support guard**
(graph 0.151 < emb 0.169) and is **restored** by overlap-shrinkage
(0.176 > 0.169; hybrid best at 0.201 / 0.352). Root-cause investigation and the
fix are documented in [CHALLENGES.md](CHALLENGES.md) — "Flagship graph advantage
reversed at full-corpus scale". **Caveats:** β=100 was tuned on this same 82-pair
gold (small n → mild overfit risk); the win is a **relative** one (graph > emb at
the same scale), not an absolute gain over the 28k sample.

#### Dietary guardrail

| Tag | Substitutes tested | Valid % | Run |
| --- | ------------------ | ------- | --- |
| vegan | 45 | **100%** | 2026-05-31 |

_Tag-coverage is 100% only because the curated set uses common, well-tagged
ingredients. Full-vocab coverage is lower — Phase 4 fixes with an ontology._

#### vs GISMo published benchmark (CC BY-NC 4.0)

> **Source:** Fatemi et al., "Learning to Substitute Ingredients in Recipes"
> (arXiv:2302.07960), facebookresearch/gismo. Gold = their `test_comments_subs.pkl`
> (`subs` source→target name pairs), canonicalized into our ingredient space via
> `scripts/fetch_gismo_gold.py`. Their data is **CC BY-NC 4.0** and is **not
> committed** (regenerate locally; the script + derived CSV are gitignored).
> **Coverage caveat:** their Recipe1M vocab ≠ our RecipeNLG 30,481 vocab, so only
> the covered subset is scored — **pair_coverage = 78.5%** (5,359/6,829 pairs;
> n=1,391 query keys); uncovered pairs are skipped, **not** penalized. Numbers are
> **our arms on their gold**, NOT a reproduction of GISMo's own model.

| Arm | MRR | R@5 | R@10 | Coverage | Run |
| --- | --- | --- | ---- | -------- | --- |
| emb-only (food2vec baseline) | 0.057 | 0.038 | 0.064 | 78.5% | gismo, 2026-06-01 |
| graph-only + overlap-shrink (β=100) | 0.076 | 0.050 | 0.077 | 78.5% | gismo, 2026-06-01 |
| **hybrid (alpha=0.5) + shrink ★** | **0.083** | **0.060** | **0.087** | 78.5% | gismo, 2026-06-01 |

★ **Flagship arm-ordering holds on an independent published benchmark:**
graph-only MRR 0.076 > emb-only 0.057 (+34%); hybrid best at 0.083 / R@10 0.087 —
the **same** ranking as our mined gold, now on Recipe1M-sourced pairs we did not
mine. Absolute values sit far below the mined-gold rows because this gold is
independently sourced (no construction-coverage; a strictly harder task) and only
the covered 78.5% is scored. _GISMo's own headline metric is MRR (arXiv:2302.07960
§4); their model's number is read from the paper, **not** reproduced here — this is
arm-vs-their-gold on the covered subset, a "benchmarked against published gold"
claim, not a head-to-head model reproduction._

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
