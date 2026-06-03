# Model Card — Recipe Recommendation & Ranking (Phase 3)

> Phase 3. Not an LLM. A learned reranker over Phase-1 overlap candidates.
> Numbers are on a 50k-recipe sample, on a **recipe-recovery** task (not general
> relevance). The headline win over the Phase-1 baseline is large and real; the
> Phase-2 → Phase-3 transfer hypothesis came back **null** and is reported as
> such (see Limitations).

---

## Intended use

Given a "pantry" (a set of ingredients a user has) plus the Phase-1 overlap
candidate pool, return a **reranked** list of recipes the pantry best supports.
The shipped evaluation frames this as **recipe recovery**: hide a fraction of a
real recipe's ingredients, treat the rest as the pantry, and measure whether the
reranker surfaces that exact recipe near the top.

**Not intended for:**
- General "what's a good recipe for me" relevance ranking — the model is
  trained and evaluated only to recover a held-out source recipe from its own
  partial ingredient set, **not** to judge culinary quality, popularity, or
  personal preference.
- Cold-start / novel recipes outside the indexed corpus (it reranks an existing
  candidate pool; it cannot surface what Phase-1 retrieval never returned — see
  the candidate ceiling below).
- Substitution recommendations — that is Phase 2's job. The Phase-2 signal was
  tested as a feature here and did not help recovery (see Limitations §3).

---

## Task definition (why "recovery", not "relevance")

Real relevance labels (which recipes a user would actually want) do not exist in
RecipeNLG. Rather than invent soft labels, Phase 3 uses an **objective,
held-out** task with a single unambiguous gold per query:

1. Take a recipe with ≥ `min_recipe_len` (4) canonical ingredients.
2. Randomly mask `mask_fraction` (0.3) of them → those become the "missing"
   ingredients; the rest become the **pantry**.
3. The masked recipe is the **single gold**. Rerank the Phase-1 candidate pool
   (top-200 by overlap coverage) to put that gold first.

Labels are **leave-ingredients-out and feature-independent** — they are derived
purely from which recipe the pantry came from, and never from any model feature
(critically, never from the substitution features). This is what makes the
sub_fill ablation (below) a clean, non-circular test of the Phase-2 → Phase-3
hypothesis.

---

## Training data

| Item | Value |
|------|-------|
| Corpus | RecipeNLG — 50,000-recipe sample (from full 1.27M cleaned) |
| Split | by `recipe_id` md5 hash, ~80% train / ~20% test |
| Train query-groups | 19,667 kept (20,000 attempted; 333 dropped — gold unreachable in pool, 0 empty pool) |
| Eval queries | 5,000 test (4,914 with gold reachable in pool) |
| Mask fraction | 0.3 |
| Candidate cap | 200 (overlap-coverage order) |
| Seeds | train mask = 13; eval mask = 14 (seed+1, distinct stream) |

A "query-group" is one masked recipe's full candidate pool (≤200 candidates,
exactly one positive). LambdaMART trains on these groups directly; the linear
model trains on the flattened rows.

---

## Method

### Candidates (Phase-1, reused)
The Phase-1 inverted index scores candidates by overlap coverage
(matched-pantry-ingredients / recipe-length) and returns the top-200. Phase 3
**only reranks this pool** — it never expands it. The fraction of queries whose
gold is in the pool at all is the **candidate ceiling** (0.983 here); no reranker
can exceed it.

### Features (8, per pantry × candidate pair)
Pure function `extract_features` — no global state:

| Feature | Meaning |
|---------|---------|
| `coverage` | matched / recipe_len |
| `match_frac_pantry` | matched / pantry_size |
| `n_matched` | # pantry ingredients in the candidate |
| `n_missing` | # candidate ingredients not in the pantry |
| `recipe_len` | # canonical ingredients in the candidate |
| `sub_fill_max` | best Phase-2 substitute-similarity of any missing ingredient to a pantry item |
| `sub_fill_mean` | mean of the above over missing ingredients |
| `dietary_ok` | candidate passes the pantry's inferred dietary constraints |

`sub_fill_*` enter via an injected `sub_lookup` callable so the feature module
stays pure and the Phase-2 substitutor stays a swappable dependency. The two
`sub_fill_*` columns are the **only** features dropped in the `−sub_fill`
ablation arm.

**Deliberately excluded: popularity / frequency features.** A recovery task with
a single gold would let a popularity prior shortcut the ranking without learning
pantry-fit; omitting it keeps the comparison about ingredient coverage, not
corpus frequency.

### Models
- **LinearRanker** — numpy logistic regression, gradient descent, **train-only**
  standardization (mean/std fit on train, applied to eval; no leakage). Score =
  raw logit.
- **LambdaMARTRanker** — LightGBM `lambdarank` objective over the query-groups.
  Lazy import (the `recommend` extra: `lightgbm`, `scikit-learn`).
- **OverlapModel** — identity ranker (score = −row_index), reproducing the
  Phase-1 coverage order through the **same** eval path on the **same** pools.
  This is the honest baseline: any lift over it is attributable to learning, not
  to a different candidate set.

---

## Evaluation

### Metrics
- **recall@10** — gold in top-10 (fraction of queries).
- **MRR@10** — mean reciprocal rank of the gold.
- **candidate ceiling** — fraction of queries where gold is in the pool at all.
- **recall@10 | in_pool**, **MRR@10 | in_pool** — the above conditioned on the
  gold being reachable, isolating reranker quality from retrieval recall.
- **NDCG omitted** — monotone-equivalent to MRR under a single gold; redundant.

### Results (50k sample, n=5,000 eval queries, 4,914 in pool, ceiling 0.983)

| Arm | recall@10 | mrr@10 | recall@10\|in_pool | mrr@10\|in_pool |
| --- | --------- | ------ | ------------------ | --------------- |
| overlap (P1 baseline) | 0.663 | 0.332 | 0.675 | 0.337 |
| linear | 0.966 | 0.910 | 0.983 | 0.926 |
| LambdaMART (+sub_fill) | 0.968 | 0.921 | 0.985 | 0.937 |
| **LambdaMART −sub_fill** | **0.971** | **0.924** | **0.988** | **0.940** |

### Headline result: CONFIRMED
The learned reranker beats the Phase-1 overlap baseline by a wide margin on
identical pools: **recall@10 0.663 → 0.971**, **MRR 0.332 → 0.924**. Conditional
on the gold being reachable, the best arm reaches **0.988 recall** of a **0.983**
ceiling — it nearly saturates what Phase-1 retrieval makes recoverable. Linear
and LambdaMART are close; the tree model edges ahead on MRR (rank quality).

### Phase-2 → Phase-3 transfer hypothesis: NULL (reported honestly)
The question Phase 3 was designed to test: *does injecting Phase-2 substitution
knowledge improve recovery ranking?* **Answer: no.** `−sub_fill` (0.971/0.924) ≥
`+sub_fill` (0.968/0.921) — a ~0.2pt difference **within ~1 standard error**
(≈0.24pt at n=5,000), i.e. no significant effect, marginally negative.

This is a **legitimate negative result**, not a failure to report around:
- The ablation is clean because labels are feature-independent (Task §above).
- It was **not** p-hacked positive — no alpha-tuning the substitutor against
  this eval to manufacture a win.
- The coverage / `match_frac_pantry` / `n_matched` features nearly solve recovery
  on their own (a recipe has near-maximal overlap with its own kept ingredients
  by construction), leaving little headroom for an orthogonal signal.

**Interpretation:** recovery rewards finding the *exact* masked recipe;
"are this candidate's missing ingredients substitutable from the pantry?" is a
*recommendation-quality* signal, not a *recovery* signal. If a positive sub_fill
effect is wanted, it belongs in a recommendation-framed task ("rank recipes you
can almost make"), not exact recovery. The shipped model can run with or without
sub_fill; **`−sub_fill` is the cleaner default** given the null.

---

## Limitations

1. **Recovery ≠ relevance.** The model is validated only to recover a held-out
   source recipe from its partial ingredients. It makes **no** claim about
   culinary quality, user preference, or popularity ranking. Treat 0.97 recall
   as "reconstructs a known recipe from its own pantry", not "recommends good
   recipes".
2. **Capped by the candidate ceiling (0.983).** The reranker cannot surface a
   recipe Phase-1 retrieval never returned. ~1.7% of eval queries have an
   unreachable gold and are unrecoverable by construction; improving that needs
   a better *retriever*, not a better *reranker*.
3. **sub_fill ablation is null.** The Phase-2 → Phase-3 transfer did not pan out
   on this task (see Evaluation). Honest negative; do not cite Phase 3 as
   evidence the substitution model adds ranking value.
4. **50k sample, not full corpus.** Numbers are on a 50,000-recipe slice. The
   full 1.27M-recipe scale row is **deferred future work**: the current
   candidate-pool pass is O(corpus × queries) (~4–8h even capped), and the
   proper fix is a scipy **sparse recipe×ingredient coverage matmul** to compute
   pools in one vectorized pass rather than per-query posting walks. The 50k
   numbers are a representative baseline, not a full-corpus claim.
5. **Phase-2 substitutor built on the same 50k slice.** For capped runs the
   sub_fill cooccurrence graph is rebuilt over the same N-recipe universe (the
   word2vec embedding arm is the shipped full-corpus artifact). This keeps the
   feature self-consistent with the eval corpus but means sub_fill is **not**
   computed against the full-vocabulary substitutor in these numbers. (Moot for
   the headline, given the null.)
6. **Frozen word2vec.** The embedding side of the substitutor is the v0.2.0
   full-corpus artifact loaded from disk, not retrained per Phase-3 run.
7. **Near-duplicate recipes.** RecipeNLG contains near-duplicate recipes; a
   masked recipe's near-twin in the pool is "wrong" under single-gold scoring
   even when ingredient-identical, slightly understating true recall. The
   `recall|in_pool` conditional partly controls for pool composition but not for
   this.
8. **Single gold per query.** Real recommendation has many acceptable answers;
   the metrics here reward exactly one. NDCG was omitted for this reason
   (redundant with MRR under a single relevant item).
