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
  it. O(lines × candidates). Clean pass dropped **11m44s → 18s (~40×)**; full
  corpus now ≈25 min. The pruned path is proven equivalent to the full scan by a
  test (any full match has all its words in the line, so it is reached via every
  one of its tokens).

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

---

_Anticipated (from design risk analysis):_
- Free-text ingredient parsing messiness (Phase 1).
- Substitution eval ground-truth sparsity (Phase 2).
- Colab/Kaggle session timeouts + GPU quota during training.
- 4 GB VRAM (MX150) → inference-only locally, ONNX export from cloud training.
