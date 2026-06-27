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
> ~15× larger (a strictly harder ranking task). Overlap-shrinkage support guard
> (β=100) is **on** for graph/hybrid. **Re-run 2026-06-05 after the parser
> doubled-token fix** (see [CHALLENGES.md](CHALLENGES.md) — "Reduplicated NER
> artifacts…"); pair_coverage = 1.0.

| Arm | MRR | R@5 | R@10 | Run |
| --- | --- | --- | ---- | --- |
| emb-only (food2vec baseline) | 0.259 | 0.313 | 0.481 | full, 2026-06-05 |
| **graph-only + overlap-shrink (β=100) ★** | **0.289** | 0.344 | 0.466 | full, 2026-06-05 |
| **hybrid (alpha=0.5) + shrink** | **0.317** | **0.434** | **0.532** | full, 2026-06-05 |

★ **Flagship ordering holds at full scale:** graph-only MRR 0.289 > food2vec
0.259; hybrid best (0.317 / R@10 0.532). The support-guard story (the graph's
lead *reverses* without overlap-shrinkage at this scale, then is restored) is
documented in [CHALLENGES.md](CHALLENGES.md) — "Flagship graph advantage reversed
at full-corpus scale". **These numbers supersede the pre-2026-06-05 full-corpus
row (graph 0.176, hybrid 0.201):** that run was on the doubled-token corpus,
where gold pairs keyed on common single-word ingredients ("butter") silently
missed the model and depressed every arm. The parser fix lifted graph-only MRR
**0.176 → 0.289** (eval *validity* was never broken — gold was canonicalized
through the same path — but the absolute scores were artificially low).
**Caveats:** β=100 was tuned on this same 82-pair gold (small n → mild overfit
risk); the flagship win is a **relative** one (graph > emb at the same scale).

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

### Phase 3 — Recommendation & Ranking (learned reranker)

> **Phase 3 eval provenance:** RecipeNLG **50,000-recipe sample**, seed=13
> (eval mask stream = seed+1=14, distinct from train), mask_fraction=0.3,
> candidate_cap=200. **Task = recipe RECOVERY**, not general relevance: mask a
> random 30% of a real recipe's ingredients, treat the rest as the "pantry",
> and rerank the Phase-1 overlap candidate pool to surface that **exact** masked
> recipe (single gold per query). Labels are **held-out** (leave-ingredients-out)
> and never reference the substitution features, so the sub_fill ablation below
> is non-circular. Train/test split by `recipe_id` md5 hash (~80/20). Train =
> 19,667 kept query-groups (20,000 attempted, 333 dropped: gold unreachable in
> pool); eval = 5,000 test queries (4,914 with gold reachable). All arms scored
> against the **same** prebuilt candidate pools (fairness). **NDCG omitted:**
> with a single gold per query it is a monotone transform of MRR — redundant,
> not independent evidence (see design spec 2026-06-02).
>
> **Re-run 2026-06-06 on the parser-fixed corpus** (doubled-token fix, see
> [CHALLENGES.md](CHALLENGES.md)), **uncapped eval = 9,354 queries** (the
> `--max-eval-queries 5000` cap of the original 2026-06-03 run was dropped, so n
> is larger and the estimate more stable). Numbers below supersede the pre-fix
> 50k row.

| Arm | recall@10 | mrr@10 | recall@10\|in_pool | mrr@10\|in_pool | Run |
| --- | --------- | ------ | ------------------ | --------------- | --- |
| **overlap (P1 baseline)** | 0.746 | 0.414 | 0.754 | 0.418 | 50k, 2026-06-06 |
| linear (numpy logistic) | 0.979 | 0.926 | 0.990 | 0.936 | 50k, 2026-06-06 |
| LambdaMART (+sub_fill) | 0.980 | 0.935 | 0.991 | 0.945 | 50k, 2026-06-06 |
| **LambdaMART −sub_fill ★** | **0.980** | **0.935** | **0.991** | **0.945** | 50k, 2026-06-06 |
| _candidate ceiling_ | _0.989_ | — | — | — | _gold-in-pool rate_ |

★ **Learned reranker crushes the P1 overlap baseline:** recall@10 0.746 → 0.980
(+0.234), MRR 0.414 → 0.935 (+0.521) on the identical candidate pools — the core
Phase-3 claim. The **candidate ceiling is 0.989** (fraction of eval queries where
the gold recipe is even present in the 200-candidate pool); the reranker reaches
0.980 of a possible 0.989, i.e. **0.991 recall conditional on the gold being
reachable** — it nearly saturates what the P1 retriever leaves on the table. (The
parser fix lifted every arm slightly — cleaner tokens improve both candidate
coverage and ranking; the pre-fix row was overlap 0.663 / LambdaMART 0.971.)

**sub_fill ablation = NULL.** Injecting Phase-2 substitution knowledge
(`sub_fill_max`/`sub_fill_mean` = best P2 substitute-similarity of a candidate's
missing ingredients to the pantry) does **not** help: LambdaMART −sub
(0.980/0.9349) ≈ LambdaMART +sub (0.980/0.9347) — a ~0.02pt gap, far **within ~1
SE** → no significant effect, and the null **reproduces on the parser-fixed
corpus**. This is an **honest negative result**, reported as-is and **not**
p-hacked positive. Recovery rewards finding
the *exact* masked recipe; substitutability is an orthogonal recommendation-quality
signal — at best noise here, at worst it pulls non-gold-but-substitutable
candidates up. Details + rationale in
[MODEL_CARD_recommender.md](MODEL_CARD_recommender.md).

#### Full corpus (1,274,290 recipes, vocab 30,481, n=5,000 eval)

> ⚠️ **Pre-parser-fix (2026-06-03) — re-eval pending.** These numbers are on the
> doubled-token corpus. The 50k re-run on the fixed corpus (above) lifted every
> arm, so expect these to move up too; the full re-run is RAM-blocked on the dev
> box (~2.8 GB resident vs <0.5 GB free) and deferred to a freed/fresh machine.
> The three findings below (ceiling collapse, reranker lead grows, sub_fill null)
> are structural and expected to hold.
>
> **Provenance:** full RecipeNLG 2.23M cleaned (1.27M after title-dedup),
> seed=13, mask_fraction=0.3, candidate_cap=200 — the **same protocol and the
> same first eval queries** as the 50k row (queries iterate the corpus in file
> order and are capped at 20k train / 5k eval, so they are the same recipes; the
> candidate **pool is drawn from the full 1.27M haystack instead of 50k**). So
> the arms are directly comparable to the 50k row: same task, same queries, a
> 25× larger haystack. Run 2026-06-03 (~65 min on i7-8565U; the
> `candidate_pool` vectorization in [CHALLENGES.md](CHALLENGES.md) made this
> feasible — the naive path projected ~23 h).

| Arm | recall@10 | mrr@10 | recall@10\|in_pool | mrr@10\|in_pool | Run |
| --- | --------- | ------ | ------------------ | --------------- | --- |
| **overlap (P1 baseline)** | 0.089 | 0.028 | 0.127 | — | full, 2026-06-03 |
| linear (numpy logistic) | 0.678 | 0.589 | 0.960 | 0.834 | full, 2026-06-03 |
| LambdaMART (+sub_fill) | 0.690 | 0.632 | 0.977 | 0.896 | full, 2026-06-03 |
| **LambdaMART −sub_fill ★** | **0.690** | **0.632** | **0.977** | **0.896** | full, 2026-06-03 |
| _candidate ceiling_ | _0.706_ | — | — | — | _gold-in-pool rate_ |

★ **The scale-up flips the story from "easy task, saturated" to "retrieval is
the bottleneck, and the reranker is what holds the line."** Three findings, all
honest:

1. **The candidate ceiling collapses 0.983 → 0.706.** A fixed 200-candidate pool
   ordered by coverage now contains the gold recipe only 70.6% of the time
   (was 98.3% at 50k): with 25× more recipes competing for 200 slots, the gold
   is crowded out ~30% of the time. **Candidate generation — not ranking — is
   the dominant error source at full scale.** This is the recovery-task analog
   of the Phase-2 graph reversal: small-sample numbers were optimistic.
2. **The learned reranker's lead grows, and it nearly saturates the (lower)
   ceiling.** LambdaMART recall@10 0.690 vs overlap **0.089 = 7.7×** (the 50k gap
   was 1.46×); MRR 0.632 vs 0.028 ≈ **23×**. Conditional on the gold being in the
   pool, LambdaMART reaches **0.977 recall@10 / 0.896 MRR** — i.e. 0.690 of a
   possible 0.706 (**97.7% of ceiling**). The P1 overlap baseline *collapses* at
   scale (even in-pool it ranks the gold top-10 only 12.7% of the time — tiny
   recipes score coverage 1.0 and bury it); the reranker is exactly what
   recovers it. **The core Phase-3 claim strengthens at full corpus.**
3. **sub_fill ablation stays NULL at full scale** — LambdaMART +sub
   (0.690/0.632) ≈ −sub (0.690/0.632), no significant gap, same as the 50k row.
   The honest "P2 substitution features don't help recipe recovery" result
   holds; not an artifact of the small sample.

_The exposed bottleneck — first-stage candidate recall (ceiling 0.706) — is the
natural next retrieval-quality target (larger/learned candidate generation,
e.g. higher cap or ANN over recipe embeddings); the reranker layer is already
near-saturating what the pool surfaces._

### Phase 4 — Nutrition & Dietary

> **Phase 4 eval provenance:** USDA FoodData Central join — **SR Legacy 2018-04**
> (7,793 foods) + **Foundation Foods 2025-04-24** (411 foods) = **8,204-food**
> artifact (`data/processed/usda.json`, gitignored; regenerate via
> `scripts/fetch_usda.py --csv-dir <dir>`). Evaluated over a **5,000-recipe**
> RecipeNLG raw sample (with quantities), **37,472 ingredient lines**, seed=42.
> Run: `uv run python -m pantrychef.eval.nutrition_main --max-rows 5000`.
> **No recipe-level nutrition gold exists** (RecipeNLG has no macro labels and no
> servings field), so this is a **coverage** evaluation, not a recipe-macro
> accuracy one — see the honesty constraint below.

| Metric | Current | Run |
| ------ | ------- | --- |
| **USDA match coverage** | **0.800** | 5k, 2026-06-03 |
| mass coverage | 0.487 | 5k, 2026-06-03 |
| nutrition completeness (≥50% lines mass-resolved) | 0.560 | 5k, 2026-06-03 |
| median unresolved-mass fraction | 0.50 | 5k, 2026-06-03 |
| dietary accuracy (n=10 hand-labeled) | **1.0** | 5k, 2026-06-03 |

_Match coverage was lifted from a **0.736** baseline to **0.800** by curating 34
high-frequency aliases in `data/nutrition/aliases.json` (the tiered matcher:
curated alias → exact normalized full + before-comma head → token-Jaccard at
threshold 0.34; below threshold is a reported gap, never a guess). Mass coverage
is the honest fraction of lines resolved to grams (mass units by fixed factor;
volume via USDA portion gram-weight else per-class density fallback;
count/portion via USDA "each" or curated per-item grams); anything unresolved is
`(None, False)` and counts toward the coverage gap, never imputed as 0._

**Dietary accuracy is a sanity check, not a robust estimate.** n=10 is a
hand-labeled smoke set (`data/nutrition/dietary_labels.json`); 1.0 means the
ontology engine agrees with all 10 labels, **not** that it is 100% accurate at
scale. Expanding the labeled set to ~100 entries is future work.

**Ontology substring-fix (qualitative).** The Phase-2 flat-substring dietary
tagger was **replaced** (not run side-by-side) by a token-leaf-word ontology with
category inheritance (`animal_product` > `meat`/`fish`/`dairy`/`egg`/`honey`).
This fixes the substring class of false positives — concretely **"eggplant" is no
longer tagged as containing "egg"** — plus curated phrase overrides
(worcestershire / fish sauce → `fish`, soy sauce → `gluten`). Because the old
tagger was removed rather than benchmarked alongside, this is reported as a
**qualitative fix**, not an old-vs-new accuracy delta.

**Impute-gate decision: TRIPPED → macro imputer built.** The gate fires because
`match_coverage` (0.800) is `< 0.80` **and** `median_unresolved_mass` (0.50) is
`> 0.20`. In response, a gated macro imputer (`pantrychef/nutrition/impute.py`:
HashingVectorizer + Ridge, MAE measured on held-out matched ingredients) was
built and validated at unit level. **Honesty note:** the imputer is a standalone
gated module; wiring its predictions into `RecipeNutrition` totals at corpus
scale is future work — recipe totals **do not currently use imputed macros**.

**Honesty constraint.** No recipe-level nutrition gold exists, so Phase 4 is
validated via (1) coverage metrics, (2) per-ingredient unit tests, and (3)
aggregation arithmetic tests — **NOT** recipe-macro accuracy. All recipe macros
are **estimates**. There are **no per-serving values** (RecipeNLG has no servings
field). Absent nutrients are reported as `None` (unknown), never `0.0`. **Not for
medical, clinical, or allergen-safety use.**

### Phase 5 — Detection
| Metric | Current | Run |
| ------ | ------- | --- |
| mAP@0.5 | _tbd_ | — |
| MX150 latency (ms/img) | _tbd_ | — |
