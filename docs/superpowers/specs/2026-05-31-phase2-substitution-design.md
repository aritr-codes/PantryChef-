# Phase 2 — Ingredient Substitution (flagship) — Design Spec

**Date:** 2026-05-31
**Status:** Approved
**Phase:** 2 (flagship) — builds on Phase 1 data foundation (`v0.1.0`)
**Type:** No-LLM ingredient substitution engine, benchmarked vs food2vec / GISMo

---

## 1. Goal

Given an ingredient (and optionally the recipe it sits in) plus a dietary
constraint, return a ranked list of **valid substitutes** — without an LLM, on
free-tier hardware. Report `precision@k` / `MRR` against published baselines
with transparent coverage, and guarantee dietary validity by construction.

### Core hypothesis (the thing we are actually testing)

> **Second-order context similarity plus hard constraint filtering beats
> direct co-occurrence embeddings for ingredient substitution.**

This is the defensible thesis. We are *not* claiming "word2vec learns
substitutions" — it does not (see §3).

## 2. Scope

- Train ingredient embeddings (word2vec skip-gram on recipe ingredient sets) →
  **food2vec baseline arm**.
- Build a **second-order context graph** (SPPMI co-occurrence rows + soft
  direct-co-occurrence penalty) → **flagship arm**.
- Minimal rule + curated **dietary tagging** for a hard guardrail filter.
- **Hybrid** rank-fusion of the two arms; optional, gated recipe-context re-rank.
- **Ablation**: emb-only / graph-only / hybrid / RRF, plus extra baselines.
- Eval harness vs a published substitution gold set, with full coverage reporting.

### Context: context-free core + context re-rank

Substitution is **context-free** at the core (`ingredient → ranked subs`) —
this is what the published pair gold set measures and what the benchmark
compares. A **light recipe-context re-rank** layer is added on top for the
"substitute X *in this recipe*" hook, but it is **optional and gated**: it
ships only if it provably improves gold metrics (see §3, risk R6).

## 3. Why this design (decisions + rationale)

Reviewed cross-model (codex) before approval. Key decisions and the reasoning:

### word2vec is a **baseline**, not the flagship

word2vec over ingredient *sets* learns "used in the same recipe" =
**complementarity / cuisine / course**, not substitutability. Butter ends up
near flour / sugar / egg, not oil / margarine. So:

- word2vec kNN is our **in-pipeline reimplementation of food2vec**. Running it
  on *our* corpus/vocab/canonicalization makes the food2vec comparison **fair
  by construction** (apples-to-apples), unlike comparing to published numbers.
- The **second-order context graph** is the arm expected to win.

### Second-order context graph (flagship arm)

Substitutes **rarely co-occur** (you don't use butter *and* its replacement in
one recipe) but **share neighborhoods** (both appear with flour, sugar, eggs).
So:

```
score(a, b) = ctx_sim(a, b) − λ · soft_penalty(cooccur(a, b))
```

- `ctx_sim(a, b)` = cosine of the **SPPMI** (shifted-smoothed PPMI) co-occurrence
  rows of `a` and `b`. High = appear in similar contexts.
- **SPPMI, not raw PPMI**: smoothing + shift controls rare-ingredient instability
  and stops sparse-but-coincident pairs being overrated.
- **Frequency cutoffs** (min/max count) and **row normalization**: stop common
  ingredients (salt, water) dominating neighborhoods.
- **Soft** co-occurrence penalty, **not** a hard subtraction: legitimate
  interchangeable pairs that *sometimes* co-occur (butter + oil, milk + cream,
  lemon + lime) must not be zeroed out.
- Optional: truncated SVD over SPPMI rows for a dense, denoised variant.

### Dietary guardrail = hard constraint, not learned

- Rule-based tagging (meat/fish/dairy/egg/honey → non-vegan; wheat/barley/rye →
  gluten) over canonical vocab, plus a **curated override list** for edge cases
  expanded beyond the obvious: honey, gelatin, lard, broth, fish sauce,
  Worcestershire, casein, ghee, etc.
- Applied as a **hard filter mask, last** → guardrail validity is **100% by
  construction** for *tagged* constraints (safety is a constraint, not a
  learned score).
- **Honesty:** an incomplete tag table is not actually 100% safe. We report
  "**100% on known tags, with X% vocab tag coverage**", and **unknown tags
  default to conservative exclusion** for constrained diets.

### Hybrid fusion = rank-based, not min-max

Naive per-query min-max blending amplifies noise among weak candidates and
distorts rankings when one arm is heavy-tailed. Instead:

- Primary fusion = **per-query rank / percentile blend**:
  `score = α · rank_emb + (1−α) · rank_graph`.
- α **tuned on the dev split only, frozen before touching test gold**.
- **RRF** (reciprocal rank fusion) reported as an alternative, knob-free fusion.
- Ablation maps cleanly: `α=1` → emb-only, `α=0` → graph-only, `α*` → hybrid.

### Recipe-context re-rank — gated

Nudging candidates toward the recipe's *other* ingredients re-introduces
complement leakage (promotes egg/vanilla over oil when subbing butter in cake).
So it is applied **only after role/type filtering, with a small weight**, and
**kept only if it improves substitution gold metrics**. Otherwise cut.

## 4. Architecture

New package: `pantrychef/substitution/`

| Module | Responsibility | Depends on |
| ------ | -------------- | ---------- |
| `corpus.py` | Ingredient-set "sentences" from cleaned recipes; custom gensim iterator that **re-shuffles each epoch** (gensim does not auto-shuffle). | Phase 1 `Recipe` store |
| `embeddings.py` | word2vec skip-gram (full-recipe window); save vectors; kNN cosine query. **= food2vec baseline arm.** | `corpus`, gensim |
| `cooccur.py` | Co-occurrence counts → SPPMI; frequency cutoffs; row normalization; optional truncated SVD. | Phase 1 `Recipe` store |
| `graph.py` | **Flagship.** Second-order context scoring `ctx_sim − λ·soft_penalty`. | `cooccur` |
| `dietary.py` | Rule + curated tag table; `tags(ingredient) → {vegan, vegetarian, gluten_free, dairy_free, ...}`; hard-filter mask; conservative-unknown handling; tag-coverage stat. | Phase 1 vocab |
| `substitute.py` | Hybrid rank-fusion + role filter + hard dietary mask → ranked `Substitute` list; optional gated context re-rank. | `embeddings`, `graph`, `dietary` |

Eval: `pantrychef/eval/substitution_eval.py`
Fetch script: `scripts/fetch_subs_eval.py`
Types: extend `pantrychef/common/types.py` (`Substitute` already exists; add
dietary tags / provenance fields as needed).
CLI: `pantrychef substitute <ingredient> [--diet vegan] [--in-recipe "..."] [-k 5]`
Config: Hydra group for `dims, window, epochs, min_count, max_count, sppmi_shift, lambda, alpha`.

## 5. Data flow

```
cleaned Recipe store (Phase 1, FULL 2.23M corpus)
        │
        ├── corpus.py ──→ shuffled ingredient-set sentences ──→ embeddings.py (word2vec) ──→ vectors ──┐
        │                                                                                              │
        └── cooccur.py ──→ SPPMI matrix ──→ graph.py (second-order ctx score) ───────────────────────┤
                                                                                                       ▼
                                                          substitute.py: rank-fusion(α) ──→ role filter ──→ HARD dietary mask
                                                                                                       │
                                                              (optional, gated) recipe-context re-rank ─┘
                                                                                                       ▼
                                                                                       ranked [Substitute]
```

**Training corpus = full 2.23M RecipeNLG.** The 50k-sample vocab (~2k) drops too
many gold-set endpoints out-of-vocab and makes the benchmark unfair. Re-run
clean on the full corpus first (~25 min with the Phase 1 match-index opt).

## 6. Evaluation

### Gold set

- Fetch the published substitution gold set (GISMo / FoodBERT "Exploiting Food
  Embeddings" lineage) via `scripts/fetch_subs_eval.py`.
- Map their ingredient strings through **our** `canonicalize()`.
- **Contingency:** if the set is unavailable / unmappable / license-blocked,
  fall back to mined pairs (ingredient swaps across near-duplicate recipes) +
  the curated set, and state so explicitly in the model card.

### Coverage — reported in full (no silent dropping)

- Full gold size, **endpoint coverage**, **query coverage**, **pair coverage**.
- Metrics on the **covered subset** AND **pessimistic** metrics (uncovered query
  counts as a miss). Low coverage is a finding, not hidden.
- Canonicalization can leak/merge labels — unresolved/ambiguous mappings are
  reported, not dropped.

### Metrics

- `precision@1/5/10`, `MRR`, `recall@k` per arm/baseline.
- **Dietary-validity %** on a curated dietary sub set + **tag-coverage %**.

### Baselines (ablation leaderboard)

food2vec (our word2vec) · co-occurrence-graph-only · SPPMI+SVD ·
role-constrained NN · taxonomy + frequency prior · RRF fusion ·
**hybrid(α\*)**. GISMo/food2vec **published** numbers listed as
**reference-only, "not directly comparable"** (different corpus/vocab/splits).

### Contamination guard

α, λ, masks, and any canonicalization tweaks are tuned on the **dev split
only**. Test gold is untouched until the final reported run.

### Tracking

MLflow logs every run (params + metrics). EVALUATION.md Phase 2 leaderboard
filled with real numbers + coverage + run dates.

## 7. Hardware / repro note

- word2vec (gensim) = **CPU-only**; second-order graph = **deterministic, no
  training**. → **Phase 2 needs no GPU.** The flagship is no-LLM *and* no-GPU;
  the MX150 stays idle until Phase 5 (CV).
- A Colab/Kaggle notebook is still shipped for reproducibility, but it runs
  CPU-fine locally on 16 GB RAM.
- `uv` lock + Hydra config + fixed seeds → local↔cloud reproducibility.

## 8. Success criteria

- Real `precision@k` / `MRR` reported vs baselines **with transparent coverage**
  — the rigorous, honest comparison *is* the deliverable, even if we don't beat
  published numbers.
- **Dietary-validity = 100%** on tagged constraints, with coverage stated.
- **Ablation table** showing each arm's marginal contribution, validating (or
  refuting) the core hypothesis.
- `substitution` module + CLI + MLflow runs + model card + EVALUATION leaderboard.

## 9. Out of scope (deferred)

- Full dietary / nutrition ontology → **Phase 4**.
- Recsys integration (subs filling recipe gaps in ranking) → **Phase 3**.
- Serving / API / demo → **Phase 6**.
- `node2vec` / GNN graph-embedding variant (second-order graph chosen instead).
- Full recipe-conditioned (GISMo-style) substitution model — only a light gated
  re-rank here.

## 10. Risks

| # | Risk | Mitigation |
| - | ---- | ---------- |
| R1 | word2vec returns complements, not subs | Demoted to baseline; graph is flagship; ablation quantifies it |
| R2 | Gold set unavailable / unmappable | Contingency: mined + curated, stated in model card |
| R3 | Vocab coverage vs gold low | Full-corpus training; coverage reported in full + pessimistic metrics |
| R4 | SPPMI instability on rare ingredients | Smoothing/shift + frequency cutoffs + row norm |
| R5 | Hard penalty nukes valid sometimes-co-occurring pairs | Soft penalty with tuned λ |
| R6 | Context re-rank reintroduces complement leakage | Gated: post-filter, small weight, kept only if it improves gold metrics |
| R7 | Min-max fusion amplifies noise | Rank/percentile fusion + RRF; α tuned on dev, frozen |
| R8 | Incomplete dietary tags ⇒ false "100% safe" | Report tag coverage; unknown → conservative exclude |
| R9 | Benchmark not apples-to-apples | food2vec reimplemented in-pipeline; published numbers labeled reference-only; no test-set tuning |
