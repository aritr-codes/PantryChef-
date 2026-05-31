# Learnings

What was learned, per phase — the "learn deeply" goal made visible. Written at
the end of each phase (and tagged with the release).

> Update at the end of each phase.

## Template

```
## Phase N — <name>   (tag: vN.x)
- **New skills:** ...
- **Concepts that clicked:** ...
- **What I'd do differently:** ...
- **Open questions carried forward:** ...
```

---

## Phase 0 — Scaffold
- Setting up uv + src-layout package + Hydra/MLflow before writing ML forces
  clean module boundaries from day one.

## Phase 1 — Data Foundation & Ingredient Intelligence   (tag: v0.1.0)
- **New skills:** free-text → canonical normalization, hybrid rules+vocab NER,
  inverted-index retrieval, micro-averaged P/R/F1 + leave-one-out recall@k eval,
  subagent-driven TDD with spec+quality review gates.
- **Concepts that clicked:**
  - RecipeNLG's `NER` column is *both* the vocab seed and the parser-eval gold —
    free ground truth, no manual labeling.
  - One shared `canonicalize()` is non-negotiable: the first version had three
    near-identical canonicalizers that disagreed on hyphens, silently killing
    multiword vocab entries. Consistency bugs hide between modules, not within.
  - Algorithmic complexity is a *correctness-of-feasibility* issue: an O(lines×V)
    match made the full corpus a 9-hour run; a token index made it 25 min. Same
    output, 40× faster — and a test proved equivalence so the speedup was safe.
- **Results:** parser F1 **0.864** (≥0.85 target met), retrieval recall@10
  **0.873** (50k sample). Baselines for Phase 2/3 to beat.
- **What I'd do differently:** add a hyphenated multiword fixture *before*
  writing the canonicalizers — the asymmetry would have been caught at unit-test
  time instead of in the final cross-cutting review.
- **Open questions → Phase 2/3:** retrieval eval is O(N²) (sample-only for now);
  bag-of-words canonical matching ignores word adjacency (precision risk);
  embeddings should replace vocab longest-match for substitution.

## Phase 2 — Ingredient Substitution (flagship)   (tag: v0.2.0)
- **New skills:** SPPMI co-occurrence matrices, second-order graph similarity,
  blended-rank fusion, ablation study design, mined gold-set construction,
  coverage-honesty methodology (report covered + pessimistic; flag 100% as
  artifact).
- **Concepts that clicked:**
  - **word2vec on ingredient sets learns complementarity, not substitutability.**
    Ingredients that always appear together (salt + pepper) rank as similar, but
    neither replaces the other. This is why `emb-only` is the baseline, not the
    flagship — it answers the wrong question.
  - **The second-order insight:** substitutes rarely co-occur in the same recipe
    (you use *either* butter *or* margarine). But they share the same supporting
    cast across different recipes. Second-order cosine similarity on SPPMI rows
    captures this shared neighborhood without requiring direct co-occurrence.
    This is what makes the graph win.
  - **Rank fusion vs min-max normalization:** blended-average-rank is more
    robust under this ablation than score-space fusion because the two arms
    produce scores on different scales and distributions.
  - **Hard-constraint guardrail = 100% by construction.** Apply dietary filters
    as a hard mask after ranking, not as a score penalty — it guarantees
    validity for tested tags and is easy to audit. Caveat: substring-based
    tagging can false-positive (e.g. "eggplant" containing "egg"). Phase 4 fixes
    this with an ontology.
  - **The original reciprocal-rank fusion formula was buggy** — it did not
    reward cross-arm consensus, so a candidate ranked #2 in both arms scored
    worse than one ranked #1 in one arm and #50 in the other. Caught in review;
    switched to blended-average-rank.
- **Results:** graph-only MRR **0.339** vs food2vec baseline **0.290** (+17%);
  recall@10 **0.475** vs **0.405**. Hybrid best recall@10 (**0.503**) at the
  cost of lower MRR/P@1. Dietary-validity **100%** (45 vegan substitutes).
  Core hypothesis confirmed. Numbers are on n=82 mined-gold queries (28k sample);
  full-corpus + published-gold comparison pending.
- **What I'd do differently:** build the mined gold harness before writing the
  model, not after — having the eval runner first makes ablation iteration much
  faster.
- **Open questions → Phase 3:** does the graph advantage hold on the published
  food2vec / GISMo gold set? Full 2.23M corpus likely shifts neighborhood
  statistics — rerun pending.
