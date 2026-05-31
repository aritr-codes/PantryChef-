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
