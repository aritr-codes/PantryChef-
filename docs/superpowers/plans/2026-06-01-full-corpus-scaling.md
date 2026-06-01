# Full-Corpus Scaling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the parser's superlinear candidate blow-up so the full 30k-vocab clean is viable, verify real throughput, then regenerate processed data, retrain substitution, and document the scaling story.

**Architecture:** Replace `build_match_index`'s all-token registration with rarest-token bucketing (each entry registered under only its rarest word by document frequency). `match_canonical`'s scoring loop is unchanged, so output is provably identical to the full O(V) scan — only the candidate set shrinks. A throughput benchmark on the real 30,481-term vocab gates the long run; the clean runs as a single owned process.

**Tech Stack:** Python 3.12, uv, pytest, ruff (line 100). RecipeNLG raw at `data/raw/full_dataset.csv` (2.2GB, present).

---

## File Structure

- `pantrychef/ingredients/parser.py` — MODIFY `build_match_index` (rarest-token bucketing + df precompute); update `match_canonical` docstring. Scoring loop and `parse` untouched.
- `tests/test_parser.py` — MODIFY `test_build_match_index_maps_tokens` to bucketing semantics; ADD `test_build_match_index_buckets_by_rarest_token` (red→green) and `test_match_index_equivalence_common_tokens` (correctness net on a common-token-heavy vocab). Existing `test_match_index_equivalence` / `test_parse_with_index_matches_without` stay as the output-identical safety net.
- `scripts/bench_clean.py` — CREATE throughput benchmark (verified recipes/min + projected full-corpus wall-clock).
- `data/processed/recipes.jsonl`, `data/processed/vocab.json` — REGENERATED (Task 3, single writer). Current on-disk `recipes.jsonl` is a 171,907-line partial/clobbered clean — discard.
- `docs/CHALLENGES.md`, `docs/EVALUATION.md` — MODIFY (Task 5).

---

## Task 1: Rarest-token bucketing in the match index

**Files:**
- Modify: `pantrychef/ingredients/parser.py:19-31` (`build_match_index`), `pantrychef/ingredients/parser.py:34-47` (docstring only)
- Test: `tests/test_parser.py`

- [ ] **Step 1: Write the failing bucketing-structure test**

Add to `tests/test_parser.py`:

```python
def test_build_match_index_buckets_by_rarest_token() -> None:
    # Each entry is registered under ONLY its rarest word (min document
    # frequency). "cheese" appears in all three, so it is the rarest word of
    # none of the multi-word entries -> its bucket holds only the bare "cheese".
    idx = build_match_index(["cream cheese", "cheddar cheese", "cheese"])
    assert idx["cheese"] == ["cheese"]
    assert idx["cream"] == ["cream cheese"]
    assert idx["cheddar"] == ["cheddar cheese"]
    # The common word does NOT fan out to every entry containing it.
    assert "cream cheese" not in idx["cheese"]
    assert "cheddar cheese" not in idx["cheese"]
```

- [ ] **Step 2: Run it to verify it fails against the current all-token impl**

Run: `uv run pytest tests/test_parser.py::test_build_match_index_buckets_by_rarest_token -v`
Expected: FAIL — current impl puts all three entries in `idx["cheese"]`, so `idx["cheese"] == ["cheese"]` fails (AssertionError).

- [ ] **Step 3: Implement rarest-token bucketing**

Replace `build_match_index` in `pantrychef/ingredients/parser.py`:

```python
def build_match_index(vocab: list[str]) -> MatchIndex:
    """Map each entry's rarest word to the entries it is the rarest word of.

    Built once per vocab. An entry can only be a full match when ALL its words
    are present in the phrase, so it is reachable from any single one of its
    words — in particular its rarest. Registering each entry under only its
    rarest word (least document frequency) keeps `match_canonical` correct
    while preventing common words ("cheese", "sauce") from fanning out to
    hundreds of non-matching candidates per line. Tie broken by the word string
    for deterministic, reproducible buckets.
    """
    df: dict[str, int] = {}
    for entry in vocab:
        for w in set(entry.split()):
            df[w] = df.get(w, 0) + 1
    index: MatchIndex = {}
    for entry in vocab:
        words = set(entry.split())
        if not words:
            continue
        rarest = min(words, key=lambda w: (df[w], w))
        index.setdefault(rarest, []).append(entry)
    return index
```

Also update the `match_canonical` docstring (`parser.py:38-42`): replace the sentence describing candidates as "entries sharing a token with the phrase ... reached via every one of its tokens" with: "Pass a prebuilt `index` (from `build_match_index`) to prune candidates to entries registered under their rarest word; a full match's rarest word is in `tokens`, so it is reached via that bucket. Both paths return the same result."

- [ ] **Step 4: Run the new test to verify it passes**

Run: `uv run pytest tests/test_parser.py::test_build_match_index_buckets_by_rarest_token -v`
Expected: PASS

- [ ] **Step 5: Fix the stale all-token assertion**

The old `test_build_match_index_maps_tokens` (`tests/test_parser.py:41-44`) asserts the all-token mapping and will now fail. Replace it with the bucketing equivalent:

```python
def test_build_match_index_maps_tokens() -> None:
    # df: all=1, purpose=1, flour=2 -> "all purpose flour" buckets under "all"
    # (tie all/purpose broken by string), bare "flour" under "flour".
    idx = build_match_index(["all purpose flour", "flour"])
    assert idx["all"] == ["all purpose flour"]
    assert idx["flour"] == ["flour"]
```

- [ ] **Step 6: Add the common-token correctness net**

Add to `tests/test_parser.py`:

```python
def test_match_index_equivalence_common_tokens() -> None:
    # Bucketing must give byte-identical results to the full scan even when
    # many entries share common words — this is the property the full-corpus
    # speedup depends on.
    vocab = [
        "cheese", "cream cheese", "cheddar cheese", "blue cheese",
        "goat cheese", "cheese sauce", "tomato sauce", "soy sauce",
        "sauce", "cream", "tomato", "cheddar cheese sauce",
    ]
    idx = build_match_index(vocab)
    phrases = [
        "sharp cheddar cheese",
        "cream cheese frosting",
        "a little tomato sauce",
        "cheddar cheese sauce for nachos",
        "just cheese",
        "blue cheese and soy sauce",
        "nothing relevant here",
        "",
    ]
    for p in phrases:
        assert match_canonical(p, vocab, idx) == match_canonical(p, vocab)
```

- [ ] **Step 7: Run the full parser suite**

Run: `uv run pytest tests/test_parser.py -v`
Expected: PASS (all, including the pre-existing `test_match_index_equivalence` and `test_parse_with_index_matches_without` safety nets).

- [ ] **Step 8: Run the full suite + lint**

Run: `uv run pytest -q && uv run ruff check pantrychef tests scripts`
Expected: all green (103 baseline tests + 1 net new = 104), no lint errors.

- [ ] **Step 9: Commit**

```bash
git add pantrychef/ingredients/parser.py tests/test_parser.py
git commit -m "perf(parser): rarest-token bucketing fixes 30k-vocab candidate blow-up"
```

---

## Task 2: Verified throughput benchmark (HARD STOP GATE)

**Files:**
- Create: `scripts/bench_clean.py`

Uses the existing on-disk `data/processed/vocab.json` (the correct full 30,481-term vocab from pass-1) — no rebuild needed for the benchmark. Times `clean_recipes` over the first N raw rows and projects the full-corpus wall-clock.

- [ ] **Step 1: Write the benchmark script**

Create `scripts/bench_clean.py`:

```python
"""Verified throughput benchmark for the full-vocab clean.

Times clean_recipes over the first N raw rows using the real 30k-term vocab,
then projects the full-corpus (2.23M) wall-clock. Print-only; writes nothing.

    uv run python scripts/bench_clean.py --rows 5000
"""

from __future__ import annotations

import argparse
import json
import time

from pantrychef.config import get_settings
from pantrychef.data.clean import clean_recipes
from pantrychef.data.loaders import load_recipenlg

FULL_CORPUS_ROWS = 2_231_142  # RecipeNLG full_dataset.csv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=5000)
    args = ap.parse_args()

    s = get_settings()
    vocab = json.load(open(s.processed_dir / "vocab.json", encoding="utf-8"))
    print(f"vocab size: {len(vocab)}")

    raw_csv = s.raw_dir / "full_dataset.csv"
    raws = load_recipenlg(raw_csv, max_rows=args.rows)

    t0 = time.perf_counter()
    n = sum(1 for _ in clean_recipes(raws, vocab))
    dt = time.perf_counter() - t0

    rpm = n / dt * 60
    eta_min = FULL_CORPUS_ROWS / rpm
    print(f"cleaned {n} recipes in {dt:.1f}s -> {rpm:,.0f} recipes/min")
    print(f"projected full corpus ({FULL_CORPUS_ROWS:,}): {eta_min:.1f} min "
          f"({eta_min / 60:.1f} hr)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Lint the script**

Run: `uv run ruff check scripts/bench_clean.py`
Expected: no errors.

- [ ] **Step 3: Run the benchmark**

Run: `uv run python scripts/bench_clean.py --rows 5000`
Expected: prints `vocab size: 30481`, a recipes/min figure, and a projected full-corpus wall-clock. (Note: vocab dict vs list — `clean_recipes` calls `build_match_index(vocab)` which iterates entries, so a list of strings is required. `vocab.json` is a JSON array, so `json.load` yields a list. Good.)

- [ ] **Step 4: Commit the benchmark tool**

```bash
git add scripts/bench_clean.py
git commit -m "feat(bench): verified full-vocab clean throughput + corpus projection"
```

- [ ] **Step 5: STOP — report to the user**

Report the measured recipes/min and projected full-2.23M wall-clock. Do **not** proceed to Task 3 until the user picks a corpus size from these verified numbers. (Lesson from 2026-05-31: no extrapolated estimates, no multi-hour background sub-agent.)

---

## Task 3: Controlled regeneration (GATED on user's size pick)

**Files:**
- Regenerate: `data/processed/vocab.json`, `data/processed/recipes.jsonl`

- [ ] **Step 1: Confirm no other writer is running**

Run (PowerShell): `Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,StartTime,Path`
Expected: no stray `download_data.py` / clean processes. If any exist from a prior run, stop and give the user `taskkill /F /IM python.exe` — do not launch a second writer.

- [ ] **Step 2: Run the regen as a single owned process**

Substitute `<N>` with the user-chosen `--max-rows` (omit the flag entirely for the full corpus). Run in the background so it is owned by this session, not a sub-agent:

Run (background): `uv run python scripts/download_data.py --max-rows <N>`
Expected on completion: logs "Vocabulary size: ..." and "Wrote .../recipes.jsonl". This is the ONLY process writing `recipes.jsonl`.

- [ ] **Step 3: Verify the output is complete and uncorrupted**

Run: `(Get-Content data/processed/recipes.jsonl | Measure-Object -Line).Lines; uv run python -c "import json,sys; [json.loads(l) for l in open('data/processed/recipes.jsonl',encoding='utf-8')]; print('all lines valid json')"`
Expected: line count matches the run size (minus title-dedup), and every line parses (no truncation/clobber).

- [ ] **Step 4: Commit nothing**

`data/processed/` is gitignored. No commit. Record the final line count + vocab size for the EVALUATION update.

---

## Task 4: Retrain substitution + measure deltas (GATED on Task 3)

**Files:**
- Regenerated substitution artifacts (paths per `scripts/train_substitution.py`)

- [ ] **Step 1: Retrain on the new clean**

Run: `uv run python scripts/train_substitution.py`
Expected: training completes; artifacts written. (If the script takes args for input/output paths, pass the regenerated `data/processed/recipes.jsonl` per its `--help`.)

- [ ] **Step 2: Run the substitution evaluation**

Run: `uv run python -m pantrychef.eval.substitution_main`
Expected: prints the ablation leaderboard (food2vec baseline / graph / hybrid) with MRR, recall@10, dietary-validity, coverage.

- [ ] **Step 3: Record deltas vs v0.2.0**

Capture the new metrics next to the 50k-sample v0.2.0 baseline (graph-only MRR 0.339, recall@10 0.475; hybrid recall@10 0.503; dietary-validity 100%). No commit yet — feeds Task 5.

---

## Task 5: Document the scaling story (GATED on Task 4)

**Files:**
- Modify: `docs/CHALLENGES.md`, `docs/EVALUATION.md`

- [ ] **Step 1: Correct CHALLENGES.md**

Replace the stale ~25-min full-corpus claim with the real story: the all-token candidate union was superlinear in vocab size (at 30k vocab, common words mapped to ~hundreds of candidates/line → ~40× per-line slowdown vs the 2k-sample), the verified pre-fix projection (~8–10 hr), the rarest-token bucketing fix (provably output-identical to full scan), and the verified post-fix throughput from Task 2.

- [ ] **Step 2: Update EVALUATION.md**

Add the full-corpus substitution row(s) from Task 4 next to the 50k-sample v0.2.0 numbers, with the corpus size and coverage. Note retrieval eval remains sample-only (O(N²), out of scope).

- [ ] **Step 3: Commit the docs**

```bash
git add docs/CHALLENGES.md docs/EVALUATION.md
git commit -m "docs(scaling): correct full-corpus estimate + record bucketing fix and retrained metrics"
```

---

## Out of Scope

- Phase 1 retrieval eval (O(N²)) stays sample-only.
- No substitution model-architecture changes — retraining only.
- The frozenset-subset-lookup alternative is a fallback only if Task 2 shows bucketing is still too slow.
