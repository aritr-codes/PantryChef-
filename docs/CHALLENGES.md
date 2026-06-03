# Challenges

Real obstacles hit and how they were attacked — especially free-hardware
workarounds. This is the honest engineering story recruiters remember.

> Update whenever you hit a wall and resolve (or work around) it.

## Template

```
### <Challenge title>   (YYYY-MM-DD, Phase N)
- **Problem:** what blocked progress.
- **Constraint:** hardware / free-tier / data limit that made it hard.
- **Attempts:** what was tried (incl. what failed).
- **Resolution:** what worked, and the tradeoff accepted.
```

---

### Ingredient parser too slow for the full corpus   (2026-05-31, Phase 1)
- **Problem:** First real run on RecipeNLG took 11m44s to clean 50k recipes →
  ~9 hours extrapolated to the full 2.23M. Unusable.
- **Constraint:** Local i7-8565U CPU; no GPU help for pure-Python text work.
- **Cause:** `match_canonical` scanned the *entire* vocabulary for every
  ingredient line — O(lines × V), with V in the thousands.
- **Resolution:** Built a token→vocab-entries index once per vocab
  (`build_match_index`); each line now only checks entries sharing a word with
  it. O(lines × candidates). Clean pass dropped **11m44s → 18s (~40×)** on the
  50k sample. The pruned path is proven equivalent to the full scan by a test
  (any full match has all its words in the line, so it is reached via every one
  of its tokens). ⚠️ The "≈25 min full corpus" estimate first recorded here was
  extrapolated from the 2,067-term sample vocab and proved wrong by ~20× — see
  the next entry.

### Match index didn't scale to full-corpus vocab — superlinear blow-up   (2026-06-01, Phase 1)
- **Problem:** The "~25 min full corpus" estimate above was extrapolated from the
  50k sample (vocab 2,067) and was wrong by ~20×. On the real full 2.23M corpus
  the vocab is **30,481** (min_count=5) and the clean pass projected to **~8–10
  hours**. An aborted background run also clobbered `data/processed/` (multiple
  writers to one file) — see the lesson below.
- **Constraint:** local i7-8565U CPU, 16 GB RAM; pure-Python text matching.
- **Cause:** `match_canonical` seeded candidates from the **union of vocab entries
  sharing ANY token** with the line. At 30k vocab, common words ("cheese",
  "sauce") each map to ~hundreds of entries, almost none real matches — candidate
  lists exploded and the per-line check went superlinear. The token index that
  gave ~40× at 2k vocab degraded badly at 30k.
- **Resolution:** **rarest-token bucketing** — register each vocab entry under
  only its *rarest* word (min document frequency). A full match's rarest word is
  always present in the line, so it is still reached; common-word buckets shrink
  from hundreds to a handful. Provably output-identical to the full scan (locked
  by a test), plus a deterministic tie-break so the indexed and full-scan paths
  agree exactly. **Verified, not extrapolated:** 53k–72k recipes/min on the real
  30k vocab → full clean in **~31–42 min** (measured 27.5 min clean + 7.5 min
  vocab pass). Lesson recorded: never quote a scaled-up runtime from small-sample
  extrapolation, and never run a long job as an unsupervised background writer.

### Flagship graph advantage reversed at full-corpus scale — then recovered   (2026-06-01, Phase 2)
- **Problem:** retraining substitution on the full 1.27M-recipe corpus (vocab
  30,481) **reversed the headline result**: graph-only fell *below* the food2vec
  baseline (MRR 0.151 vs 0.169; recall@10 0.264 vs 0.315), where on the 28k
  sample it had won (0.339 vs 0.290). Both arms dropped; the graph dropped ~2×
  harder.
- **Constraint:** 16 GB RAM; the 30k×30k SPPMI graph is rebuilt at eval time.
- **Attempts (systematic, hypothesis-driven — root cause before any fix):**
  - *Dimensionality?* Swapped raw sparse SPPMI-cosine for dense SPPMI+SVD-100 on
    the same matrix → **worse** (MRR 0.084; gold buried at median rank 752).
    **Refuted.**
  - *Frequency?* corr(log query-frequency, gold-rank) = **−0.61** for the graph —
    the most frequency-sensitive arm.
  - *Distractors?* Top-40 audit: gold substitutes are common (median df 7,586,
    context-overlap 748); the non-gold candidates crowding the ranking are ~7×
    rarer (df 1,016), half the overlap, **28% genuinely rare vs 0% of golds**.
    Masking rare *non-gold* candidates (gold exempt) lifted graph-only back to
    parity+ with emb. **Confirmed.**
- **Cause:** the graph ranks by raw SPPMI cosine with **no support guard**. At 2k
  common-only vocab there were no rare distractors; at 30k vocab the ~28k added
  rare ingredients have sparse rows that yield *coincidentally* high cosine and
  displace the well-supported common substitutes. word2vec's dense, downsampled
  training is robust to this. The 28k-sample "graph beats baseline" headline was
  real but **conditional on a small, common-only vocab**.
- **Resolution:** **overlap-shrinkage** — weight the cosine by `ov/(ov+β)` where
  `ov` = shared nonzero SPPMI context columns and β=100 (empirical-Bayes
  confidence; chosen by a formulation grid over overlap/df floors & shrinkage).
  Downweights low-overlap, low-confidence candidates. At full corpus this
  **restores the flagship**: graph-only MRR 0.176 > emb 0.169, recall@10 0.316 >
  0.315; hybrid best (0.201 / 0.352). *Caveats:* β tuned on the same 82-pair gold
  (small n → mild overfit risk); absolute recall stays below the 28k sample
  because 30k-vocab ranking is a harder task (not apples-to-apples). The claim is
  narrow — the support guard restores the graph's **relative** advantage at scale.

### Retrieval leave-one-out eval is O(N²)   (2026-05-31, Phase 1)
- **Problem:** Retrieval recall@10 over the 50k sample took 75 min.
- **Cause:** common ingredients (salt, flour, egg) have postings covering a
  large fraction of the corpus, so each query scores tens of thousands of
  candidates; × ~28k queries ≈ O(N²).
- **Resolution (accepted tradeoff):** evaluate on a **sample**, not the full
  corpus — standard practice for retrieval eval, and the recall estimate is
  stable. Full-scale speedups (candidate caps, score short-circuit) deferred to
  Phase 3 when the retrieval layer gets its learned ranker.

### Substitutes don't co-occur — word2vec gives the wrong signal   (2026-05-31, Phase 2)
- **Problem:** word2vec trained on ingredient sets ranks complementary
  ingredients (salt + pepper) as similar, not substitutable ones. The embedding
  baseline (food2vec-style) was expected to be the flagship, but the co-occurrence
  geometry is wrong for substitution.
- **Constraint:** no labeled substitution pairs in the training data; only recipe
  ingredient sets are available.
- **Attempts:** tried embedding nearest-neighbours directly (MRR 0.290). Looked at
  score-space hybrid — better recall, worse top-rank precision.
- **Resolution:** second-order SPPMI graph. Substitutes share the same supporting
  cast across recipes even though they don't appear together. Cosine similarity on
  SPPMI rows (shared neighborhood) sidesteps the co-occurrence problem entirely.
  Graph-only MRR 0.339 (+17% vs baseline). No training loop, deterministic, CPU-only.

### Substitution gold-truth scarcity → mined fallback + coverage honesty   (2026-05-31, Phase 2)
- **Problem:** the published food2vec / GISMo substitution test set requires a
  manual download that cannot be scripted (terms gate). Without it, there is no
  ground truth for eval.
- **Constraint:** free-tier, no institutional access.
- **Attempts:** tried automating the download — blocked by form gate. Converter
  script `scripts/fetch_subs_eval.py` is wired and waiting.
- **Resolution:** mined the gold set from our own corpus — near-duplicate recipes
  differing by exactly one ingredient (`min_overlap=4`). Yields 116 directed pairs,
  82 query keys. Fully reproducible. Coverage is 100% **by construction** (mining
  from the same corpus guarantees all ingredients are known) — stated explicitly as
  an artifact, not a strength. Numbers flagged as not paper-comparable.

### Fusion formula bug — reciprocal-rank didn't reward consensus   (2026-05-31, Phase 2)
- **Problem:** the original rank fusion used reciprocal-rank aggregation. A
  candidate ranked #1 in one arm and #50 in the other outscored one ranked #2 in
  *both* arms — penalising consistent mid-rank candidates.
- **Resolution:** caught in peer review before results were recorded. Switched to
  blended-average-rank (`rank_hybrid = alpha × rank_emb + (1-alpha) × rank_graph`,
  alpha=0.5). The corrected formula rewards cross-arm consensus appropriately.

### Full-corpus Phase-3 eval hit a memory wall, then a speed wall   (2026-06-03, Phase 3)
- **Problem:** the Phase-3 reranker leaderboard had only ever run on a 50k
  sample. Scaling it to the full 1.27M-recipe corpus (to confirm the headline
  holds at scale) hit two walls in sequence.
- **Constraint:** local i7-8565U, 16 GB RAM (often <4 GB free).
- **Wall 1 — memory.** `load_recipes` materialized 1.27M **pydantic** `Recipe`
  objects, each carrying `ingredients_raw` (the bulk of the 625 MB file) which
  the reranker never reads, and the inverted index pinned those full objects.
  Projected ~5 GB resident → OOM-tight. **Resolution:** stream-load dropping
  `ingredients_raw` and intern canonical strings (one object per the ~30k
  distinct ingredients, shared across 1.27M recipes). Peak resident **5 GB →
  2.8 GB**, output-identical (nothing downstream reads the dropped field).
- **Wall 2 — speed.** `candidate_pool` scored candidates with a pure-Python dict
  loop over postings. At full scale common pantry ingredients (salt, sugar,
  butter…) have postings spanning hundreds of thousands of recipes, so each pool
  cost **3.3 s** — the full run projected to **~23 h**. Pruning is *not* lossless
  here: coverage normalizes by recipe length, so a 1-ingredient recipe matching
  one common pantry item scores coverage 1.0 and legitimately competes. You must
  count every candidate. **Resolution:** vectorize losslessly — build a row-int
  columnar view of the index (`postings_rows`/`canon_len_by_row`/`id_rank_by_row`)
  and score with `np.bincount` + `np.lexsort` instead of the Python loop. The
  string tie-break (`"r10" < "r2"`) is reproduced via a precomputed string-sort
  rank, so the result is **byte-identical** (locked by a randomized oracle test
  vs the old algorithm across tie-heavy cases). **3279 ms → 89 ms/pool (~37×)**;
  full run 23 h → **~65 min**. The 50k numbers were unaffected (same output).
- **Payoff (the real finding):** at full scale the P1 candidate **ceiling
  collapses 0.983 → 0.706** — retrieval recall, not ranking, becomes the
  bottleneck — yet the learned reranker's lead over the overlap baseline *grows*
  (recall@10 7.7×, MRR ~23×) and it recovers **97.7% of the lower ceiling**. The
  Phase-3 claim strengthens; see [EVALUATION.md](EVALUATION.md). Lesson echoed
  from Phase 2: validate headline metrics at target scale — small-sample numbers
  flatter both the baseline and the absolute scores.

---

_Anticipated (from design risk analysis):_
- Free-text ingredient parsing messiness (Phase 1).
- Substitution eval ground-truth sparsity (Phase 2).
- Colab/Kaggle session timeouts + GPU quota during training.
- 4 GB VRAM (MX150) → inference-only locally, ONNX export from cloud training.
