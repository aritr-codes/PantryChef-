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

---

_Anticipated (from design risk analysis):_
- Free-text ingredient parsing messiness (Phase 1).
- Substitution eval ground-truth sparsity (Phase 2).
- Colab/Kaggle session timeouts + GPU quota during training.
- 4 GB VRAM (MX150) → inference-only locally, ONNX export from cloud training.
