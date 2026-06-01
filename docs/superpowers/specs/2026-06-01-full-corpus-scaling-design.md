# Full-Corpus Scaling — Parser Fix + Verified Run + Substitution Retrain

**Date:** 2026-06-01
**Status:** Approved (design), pending spec review
**Objective:** Both — (a) fix & document the parser scaling bottleneck found on 2026-05-31, and (b) retrain the substitution graph on the larger corpus and measure metric deltas vs the 50k-sample v0.2.0 baseline.

## Problem

The 2026-05-31 attempt at the full 2.23M RecipeNLG clean revealed that the token-pruned match index does **not** scale. At the full-corpus vocab (30,481 terms, `min_count=5`), the clean pass runs ~40× slower per line than on the 50k sample (2,067-term vocab), projecting to ~8–10 hr rather than the ~25 min claimed in `docs/CHALLENGES.md`.

**Root cause:** `match_canonical` ([parser.py:55](../../../pantrychef/ingredients/parser.py)) seeds candidates from the **union of vocab entries containing ANY phrase token**. At 30k vocab, common tokens ("cheese", "sauce") map to ~500 entries each, almost none of which are true subset matches, and every candidate is checked per line. Superlinear in vocab size; the pruning that worked at 2k vocab degrades badly at 30k.

This contradicts our own `CHALLENGES.md` ~25-min estimate (extrapolated from small-vocab sample) — a real, portfolio-worthy scaling finding.

## The Fix: Rarest-Token Bucketing

Precompute `df[w]` = number of vocab entries containing word `w` (counted over **unique** tokens per entry). Register each vocab entry under **only its rarest word** (min `df`, tie-broken by the word string for determinism). At query time, candidates = union over phrase tokens `t` of `index.get(t)`, then the **existing** subset-filter + longest-match scoring loop, unchanged.

**Correctness (output-identical to full O(V) scan):** Any true match `e` has all its words ⊆ phrase tokens. `rarest(e)` is one of `e`'s words, so `rarest(e)` ∈ phrase tokens. `e` is registered under `rarest(e)`, therefore `e` ∈ candidates. No true match is excluded; the scoring loop is untouched, so the selected entry is identical.

**Why it's fast:** each entry sits in exactly one bucket (its rarest word) instead of all its word-buckets. The single-word entry `"cheese"` stays in the `cheese` bucket, but multi-word entries move to their *rare* word's bucket — so `index["cheese"]` shrinks from ~500 to a handful.

**Implementation guards (from codex review):**
- Compute `df` over `set(entry.split())` (unique tokens per entry), not raw token list.
- Skip empty entries when building buckets (cannot be matched; existing `if words` guard in the scoring loop already handles them downstream).
- Keep the scoring loop byte-for-byte: `words = entry.split()`, `key = (len(words), len(entry))`, strict `key > best_key`. Word-count uses `len(split())` even though subset matching uses a set.
- Tie-break the rarest-word choice by `(df[w], w)` for reproducibility (does not affect correctness — any in-entry word in the phrase suffices).

**Rejected alternatives:**
- *Frozenset subset lookup* (map `frozenset(tokens) → best entry`, enumerate 2^k phrase subsets): fast for short phrases but blows up on long lines; kept only as a fallback if bucketing under-performs at benchmark.
- *Skip/cap ultra-common tokens during seeding*: silently breaks the most common matches (plain `"cheese"` would never match). Bucketing is the principled version without the correctness loss.

## Scope of the Match-Index API change

- `build_match_index(vocab)` — same signature; internal logic changes from all-token to rarest-token registration. Add a df precompute pass.
- `match_canonical(phrase, vocab, index)` — unchanged body; docstring updated to describe rarest-token bucketing. The `index is None` full-scan fallback stays as the reference path.

## Verification & Run Plan

1. **TDD the fix.** Add a test asserting the bucketed-index path returns results **identical** to the no-index full scan across a vocab deliberately seeded with shared common tokens (locks the consistency invariant `test_canonical_consistency.py` relies on). Then implement. All existing tests stay green (103 baseline).
2. **Benchmark.** Time the fixed parser over a fixed N-line slice of `data/raw/full_dataset.csv` against the real 30,481-term vocab → verified recipes/min. Project wall-clock for full 2.23M. **Hard stop here** — report the verified number before launching any long run (per the 2026-05-31 lesson: no extrapolated estimates, no multi-hour background sub-agent).
3. **Run-size decision + controlled clean.** User picks size from the verified projection. Execute as a **single** foreground/background process owned by the main session — no sub-agent, no second writer to `data/processed/recipes.jsonl` (`save_recipes` opens with `"w"` and truncates; concurrent writers clobber). Regenerate processed data from scratch (current on-disk `recipes.jsonl` is a 171,907-line partial/clobbered clean and must be discarded).
4. **Retrain substitution.** Run `train_artifacts` on the new clean; run `pantrychef.eval.substitution_main`; record MRR / recall@10 deltas vs v0.2.0 (graph-only MRR 0.339, recall@10 0.475; hybrid recall@10 0.503; dietary-validity 100%) into EVALUATION.md.
5. **Document.** Write the scaling bottleneck + fix into `docs/CHALLENGES.md`, correcting the old ~25-min estimate with the verified numbers and the root-cause analysis.

## Out of Scope

- **Phase 1 retrieval eval** is O(N²) and infeasible at full corpus — stays sample-only. Not part of this run.
- No changes to the substitution model architecture; only retraining on a larger corpus.

## Success Criteria

- Parser fix merged with a consistency test proving bucketed == full-scan output; full suite green.
- A **verified** (measured, not extrapolated) full-corpus throughput number and wall-clock projection recorded before any long run.
- Clean completes to a known, uncorrupted `recipes.jsonl` of the chosen size, produced by a single writer.
- Substitution retrained; metric deltas vs v0.2.0 recorded in EVALUATION.md.
- `docs/CHALLENGES.md` corrected with the real scaling story.
