# Phase 3 — Recommendation & Ranking — Design

**Date:** 2026-06-02
**Status:** Approved (design), pending spec review
**Goal:** Move from "contains X" overlap retrieval (Phase 1) to a learned, constraint-aware reranker that recovers the intended recipe from a pantry-masked query — and beats the Phase 1 overlap sort on the same candidate set. Demonstrates that Phase 2 substitution knowledge, injected as a ranking feature, measurably improves ranking.

## Background

Phase 1 ships an inverted-index retriever (`pantrychef/retrieval/`): candidates = recipes sharing ≥1 pantry ingredient, scored by `coverage = |pantry ∩ recipe| / |recipe|`, tie-broken by fewer missing ingredients then `recipe_id` (`baseline.py`). On the 50k Recipe1M sample its recall@10 is 0.873 — the number Phase 3 must beat (see `docs/EVALUATION.md`).

Recipe1M has **no clicks, ratings, or interaction logs**, so there is no natural relevance signal for a learning-to-rank model. Phase 3 constructs the signal from the corpus itself via ingredient masking.

## Task framing — recipe recovery, not general relevance

**Rerank, do not re-retrieve.** The Phase 1 inverted index generates the candidate set; the learned ranker reorders it. Both the baseline and the ranker operate on the **exact same candidate set** (same generation, same depth) so the comparison is fair — the only variable is the ordering function.

**Query construction (leave-ingredients-out).** Take a real recipe R with canonical ingredient set C (|C| ≥ N+2). Hide N ingredients at random (seeded) → the simulated pantry is the kept ingredients C∖hidden. The **single gold-relevant target is R itself**. The ranker must surface R among the overlap candidates generated from that pantry.

**Scope of the claim (important, per cross-model review).** This is a *recipe-recovery* task: "given a partial pantry derived by masking a known recipe, recover that recipe." It is **not** a claim of general recipe relevance — other candidates may be genuinely good or better recipes for the pantry, and they are labeled 0 here. All reporting and the model card must use recovery wording, never "best recipe for your pantry."

This framing is deliberately **feature-independent**: labels come only from "which recipe was masked," never from substitution or any model output, so the substitution feature (below) cannot become circular.

## Components — new `pantrychef/recommender/` module

The package already has an empty `recommender/` stub. New files, each one clear purpose, pure where possible:

- **`query_sim.py`** — held-out query generator. `recipe → QuerySim(pantry, hidden, gold_id)`. Seeded/deterministic. Skips recipes with `|C| < N+2` (logged count, no silent drop).
- **`features.py`** — pure candidate feature extractor. `(pantry_set, recipe, sub_model?) → FeatureVec`. No I/O, fully unit-testable against hand-built inputs.
- **`rank.py`** — model wrappers behind one interface (`fit(groups)`, `score(features)`): `LinearRanker` (pointwise logistic baseline; any fitted scaler is **fit on train split only**) and `LambdaMARTRanker` (LightGBM `lambdarank`, optimizes NDCG internally; persisted to `models/recommender/`).
- **`recommend.py`** — glue: `pantry → P1 candidates → features → model scores → top-k ScoredRecipe`. Reuses `pantrychef.common.types.ScoredRecipe`.
- **`train.py`** — builds LightGBM training groups from **train-split** recipes (M held-out queries each), fits, persists model + feature-name manifest.

Eval (mirrors `substitution_eval.py` / `substitution_main.py`):
- **`pantrychef/eval/recommend_eval.py`** — metric computation over test-split queries.
- **`pantrychef/eval/recommend_main.py`** — CLI entry, prints the comparison table, writes the EVALUATION.md row.

## Features (per query–candidate pair)

| Feature | Definition | Notes |
| ------- | ---------- | ----- |
| `coverage` | \|pantry ∩ recipe\| / \|recipe\| | the P1 baseline score |
| `match_frac_pantry` | \|pantry ∩ recipe\| / \|pantry\| | pantry utilization |
| `n_matched` | \|pantry ∩ recipe\| | size |
| `n_missing` | \|recipe ∖ pantry\| | gap size |
| `recipe_len` | \|recipe\| | length prior |
| **`sub_fill_max`** | over missing ingredients, max P2 substitution similarity to any pantry item | **ablation feature** |
| **`sub_fill_mean`** | same, mean over missing ingredients | **ablation feature** |
| `dietary_ok` | recipe violates no diet tag implied by pantry (P2 dietary module) | binary; cheap |

**`sub_fill_*` uses the frozen Phase 2 model** (graph/embedding similarity). Phase 2 was tuned and evaluated on substitution gold pairs — a different task; it never saw recipe-recovery queries. Disclosed in the model card so the feature is not mistaken for tuned-on-eval.

**Dropped: popularity.** Recipe1M carries no ratings/clicks → no honest popularity signal. Recorded as out-of-scope, not fabricated.

## Headline experiment — substitution ablation

Train LambdaMART **with** vs **without** the `sub_fill_*` features. If the with-subs variant lifts recall@10 / MRR@10 on recovery, that is empirical evidence that substitution knowledge improves ranking — the Phase 2 → Phase 3 narrative, demonstrated rather than asserted. The three-way comparison **overlap (P1) < linear < LambdaMART** is the secondary headline.

## Data flow + train/test split (no leakage)

- Recipes split **train/test by `recipe_id` hash** (deterministic, ~80/20). Test recipes never contribute training groups.
- **Index is built on the full corpus** (so the gold recipe is retrievable at eval — this is required for retrieval, not leakage). Cross-model review confirmed: leakage would require test *labels* or label-fitted statistics entering training; candidate generation over the full corpus does not. Only unsupervised / declared-corpus-built stats (the index, the frozen P2 model) are corpus-wide; any *fitted* ranker stat is train-only.
- **Train:** each train recipe → M seeded held-out queries → P1 candidates → label gold=1, all others=0 → one LightGBM group per query.
- **Eval:** each test recipe → one held-out query → rank the candidate set → metrics vs gold.
- Everything seeded → reproducible.

**Caveat (disclosed):** near-duplicate recipes across the split could inflate results. Noted in the model card; optional dedup is a stretch item, not v1 scope.

## Metrics

Per cross-model review: with a **single gold per query**, NDCG@10 is monotone-equivalent to MRR (NDCG@10 = 1/log₂(rank+1) if rank ≤ 10 else 0; MRR = 1/rank). Reporting both as independent evidence oversells. So:

- **Primary:** `recall@10` (directly comparable to P1's 0.873) and `MRR@10`.
- **NDCG@10 omitted** as redundant with MRR under single-gold; noted in EVALUATION.md so its absence is intentional, not an oversight.
- **Candidate-recall ceiling** reported separately: fraction of queries where gold R is in the candidate pool at all. Ranker metrics reported **both** over all queries **and** conditional on gold-in-pool — so model recall is read against the achievable max (same honesty pattern as GISMo's 78.5% coverage).

### EVALUATION.md Phase 3 table (to fill)

| Metric | overlap (P1) | linear | LambdaMART | LambdaMART −sub |
| ------ | ------------ | ------ | ---------- | --------------- |
| recall@10 | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| MRR@10 | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

Plus: candidate-recall ceiling, and metrics conditional on gold-in-pool. **Corpus:** 50k sample headline (comparable to P1/P2), full-corpus (1.27M) added as a scale row — consistent with how Phase 2 and the GISMo benchmark were reported.

## Error handling

- **Empty pantry / no candidates** → empty ranked list, no crash (mirrors P1 `recommend`).
- **OOV ingredient in sub-feature** → similarity 0, skip (P2 already handles OOV).
- **Degenerate query** (recipe with < N+2 ingredients) → skipped, dropped count logged.
- **LightGBM not installed** → `train`/`recommend` raise a clear "install the `[recommend]` extra" error; core package import stays clean (same pattern as the `[substitution]` extra).

## Testing (TDD)

- `features.py`: pure unit tests — hand-built pantry/recipe → known feature values, including OOV and sub_fill aggregation.
- `query_sim.py`: determinism under fixed seed; degenerate-recipe skip.
- `rank.py`: fit/predict on a tiny fixture; train-only scaler boundary.
- `recommend_eval.py`: metric correctness on a toy gold set with known ranks (recall@k, MRR@k, ceiling, conditional metric).
- End-to-end smoke on a small sample.

## Dependencies

- New optional extra `[recommend]` → `lightgbm` (lighter dep + faster CPU `lambdarank` than xgboost). Core install unchanged.

## Out of scope (YAGNI / deferred)

- Two-tower contrastive dense retrieval → stretch / later phase.
- Personalization, online/AB → Phase 6+.
- Popularity signal → no data; dropped.
- Near-duplicate dedup → optional stretch.
- Full nutrition/dietary ontology → Phase 4.
