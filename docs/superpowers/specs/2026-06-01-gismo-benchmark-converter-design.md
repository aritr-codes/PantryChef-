# GISMo Published-Benchmark Converter — Design

**Date:** 2026-06-01
**Status:** Approved (design), pending spec review
**Goal:** Evaluate PantryChef's substitution arms against the **published GISMo benchmark** (Fatemi et al., Meta AI, arXiv:2302.07960) so EVALUATION.md can carry a real "vs published baseline" row — alongside, not replacing, the existing mined-gold eval.

## Background

The current Phase-2 substitution numbers use a gold set **mined from our own RecipeNLG corpus** (116 pairs / 82 query keys). Coverage is 100% **by construction** (mining from the same corpus) — honest but not paper-comparable. GISMo introduced the standardized substitution benchmark (pairs mined from Recipe1M comments + MRR metric). Comparing against it is the credible "benchmarked vs published" claim.

**Source (CC BY-NC 4.0, no login, Recipe1M not needed):**
- `https://dl.fbaipublicfiles.com/gismo/test_comments_subs.pkl` (920 KB) — held-out test pairs
- `https://dl.fbaipublicfiles.com/gismo/vocab_ingrs.pkl` (318 KB) — ingredient ID → string decoder
- Reference loader: `github.com/facebookresearch/gismo`, `gismo/data_loader.py::SubsData`.

**License obligation:** CC BY-NC 4.0 — attribute, non-commercial. Their `.pkl` AND any CSV derived from it stay **out of git** (gitignored); reproducible via the committed scripts.

**Schema is verified at implementation time, not assumed.** Each `*_comments_subs.pkl` is a Python pickle: a list of records carrying source ingredient, target substitute, and recipe context as Recipe1M vocab IDs. The exact field names are unknown until `pickle.load` + introspection (step 1 of the plan pins them). `vocab_ingrs.pkl` is a Vocabulary object/dict mapping ID → ingredient string.

## Architecture

Isolate the hard-to-test glue (network + pickle) from the pure, testable transform.

- **`pantrychef/eval/gismo_gold.py`** — pure transform, unit-tested. No I/O.
  ```
  canonical_pairs(raw_pairs: list[tuple[str, str]]) -> dict[str, set[str]]
  ```
  Canonicalize both sides via the project `canonicalize()`; drop empties and self-pairs (source == target after canonicalize); merge into `source -> {targets}`. Deterministic.
- **`scripts/fetch_gismo_gold.py`** — the glue script (mirrors `scripts/download_data.py` style):
  1. Idempotent download of the two pkls to `data/raw/gismo/` (skip if present); clear failure + non-zero exit on network error.
  2. `pickle.load` both; **introspect + assert** the record schema (a clear `ValueError` naming the actual fields if they differ from the pinned ones — forces re-introspection rather than silent miswiring).
  3. Decode source/target IDs → strings via `vocab_ingrs`.
  4. `canonical_pairs(...)` → write `source,target` rows (header `source,target`, one per (source,target)) to `data/eval/subs_gold_gismo.csv`.
  5. Print stats: #records read, #pairs after canonicalize/dedup, #unique sources, and #pairs whose source or all targets are OOV vs our `vocab.json` (the coverage preview).
  CC BY-NC attribution in the file docstring.
- **`pantrychef/eval/substitution_main.py`** — add `argparse`:
  - `--gold <path>` (default `data_dir/eval/subs_gold.csv` — current behavior unchanged).
  - `--gold-name <str>` (default `"mined"`) — label only, for the log line.
  Everything else (artifacts load, ablation, coverage report, dietary) unchanged. The existing `coverage_report` already prints; with GISMo gold it will be **<100%** (Recipe1M vocab ≠ our RecipeNLG 30k vocab) — that is the honest headline.
- **`.gitignore`** — add `data/raw/gismo/` and `data/eval/subs_gold_gismo.csv`.
- **`docs/EVALUATION.md`** — fill the "Published reference" table with the GISMo row(s): our arms' MRR/recall on the GISMo gold, the **stated coverage**, CC BY-NC attribution, and the explicit caveat that RecipeNLG vocab ≠ Recipe1M so uncovered pairs are skipped (not scored).

## Data flow

```
fbaipublicfiles URLs
  → data/raw/gismo/{test_comments_subs,vocab_ingrs}.pkl   (gitignored)
  → pickle.load + introspect schema
  → decode IDs via vocab_ingrs
  → raw (src, tgt) name pairs
  → canonical_pairs()                                       (pure, tested)
  → data/eval/subs_gold_gismo.csv                           (gitignored)
  → substitution_main --gold subs_gold_gismo.csv --gold-name gismo
  → ablation rows + coverage
  → docs/EVALUATION.md "vs GISMo (published)" row
```

## Error handling

- **Download failure** (network/HTTP): clear message, non-zero exit; do not write a partial CSV.
- **Schema mismatch** on introspection: `ValueError` quoting the actual record keys/type — never silently map the wrong field.
- **OOV pairs** (GISMo ingredient not in our vocab after canonicalize): NOT an error — counted and reported as coverage; the eval already skips OOV queries/targets (`pessimistic=False`).
- **Empty after canonicalize**: fail loudly (indicates a decode/schema problem).

## Testing

- **Unit (CI-safe, no network):** `tests/eval/test_gismo_gold.py` for `canonical_pairs` — canonicalization applied to both sides, dedup, self-pair drop, multi-target merge, empty-input handling. Hand-made `raw_pairs`; no pkl needed.
- **`--gold` wiring:** a small test that `substitution_main` reads an alternate gold CSV path (or, lighter, that the argparse default preserves current behavior). Must not depend on the full artifacts — keep minimal or assert arg parsing only.
- **Download + pickle decode:** one-shot integration, run manually (their pkl can't ship in CI). The script's stats printout is the manual verification.

## Out of Scope

- Only the **test** split (`test_comments_subs.pkl`). train/val splits are YAGNI for a benchmark eval.
- No change to the substitution model or the mined-gold eval.
- The exact published GISMo MRR table (from the arXiv PDF §4) is recorded in EVALUATION.md as context but is read manually — not scraped.

## Success Criteria

- `canonical_pairs` unit-tested and green in CI (no network/data dependency).
- `scripts/fetch_gismo_gold.py` downloads, decodes, and writes `subs_gold_gismo.csv` with a printed coverage stat; their pkl + the CSV stay out of git.
- `substitution_main --gold <gismo.csv> --gold-name gismo` runs the ablation and prints coverage <100% honestly.
- EVALUATION.md carries a "vs GISMo (published)" row with coverage + CC BY-NC attribution + vocab-mismatch caveat.
- Existing mined-gold eval and all current tests unchanged.
