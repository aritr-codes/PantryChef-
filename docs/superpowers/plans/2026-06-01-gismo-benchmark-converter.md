# GISMo Published-Benchmark Converter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate PantryChef's substitution arms against the published GISMo benchmark (arXiv:2302.07960) and add a "vs GISMo (published)" row to EVALUATION.md, alongside (not replacing) the mined-gold eval.

**Architecture:** Isolate the pure, CI-safe transform (`canonical_pairs`) from the network/pickle glue (`scripts/fetch_gismo_gold.py`). Add a `--gold` flag to the eval entrypoint so the GISMo gold is a separate file from the committed mined gold. Their CC BY-NC `.pkl` and the derived CSV stay out of git.

**Tech Stack:** Python 3.11, uv, pytest, ruff (line 100). stdlib `pickle` + `urllib` (no new deps). Reuses `pantrychef.ingredients.normalize.canonicalize` and `pantrychef.eval.subs_gold.load_pairs_csv`.

---

## File Structure

- `pantrychef/eval/gismo_gold.py` — CREATE. Pure `canonical_pairs(raw_pairs)` transform. No I/O, no network. Unit-tested.
- `tests/test_gismo_gold.py` — CREATE. Unit tests for `canonical_pairs` (CI-safe, no pkl/network).
- `pantrychef/eval/substitution_main.py` — MODIFY. Add `argparse` with `--gold` / `--gold-name` (default = current mined path + label "mined").
- `tests/test_substitution_main.py` — MODIFY. Add a test that an alternate `--gold` path is honored / default preserved.
- `scripts/fetch_gismo_gold.py` — CREATE. Download + pickle introspect + decode + write `data/eval/subs_gold_gismo.csv`. Glue; not in CI.
- `.gitignore` — MODIFY. Add `data/raw/gismo/` and `data/eval/subs_gold_gismo.csv`.
- `docs/EVALUATION.md` — MODIFY. Fill the "Published reference" table with the GISMo row + coverage + CC BY-NC attribution.

**Build order rationale:** Tasks 1–2 are pure/testable and schema-independent — do them first under TDD. Task 3 is an introspection **spike** that pins the real pkl schema (unknown until downloaded). Task 4 writes the decode glue against the confirmed schema. Tasks 5–6 wire gitignore + docs.

---

## Task 1: Pure `canonical_pairs` transform (TDD)

**Files:**
- Create: `pantrychef/eval/gismo_gold.py`
- Test: `tests/test_gismo_gold.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_gismo_gold.py`:

```python
from pantrychef.eval.gismo_gold import canonical_pairs


def test_canonicalizes_both_sides() -> None:
    # canonicalize lowercases / singularizes / hyphen-splits via the project path.
    out = canonical_pairs([("Butter", "Margarine"), ("eggs", "egg-whites")])
    assert out["butter"] == {"margarine"}
    assert out["egg"] == {"egg whites"}


def test_merges_multiple_targets_per_source() -> None:
    out = canonical_pairs([("butter", "margarine"), ("butter", "oil")])
    assert out["butter"] == {"margarine", "oil"}


def test_drops_self_pairs_and_empties() -> None:
    # self-pair after canonicalize, and empty/whitespace sides, are dropped.
    out = canonical_pairs([("Butter", "butter"), ("", "oil"), ("salt", "   ")])
    assert "butter" not in out
    assert out == {} or all(v for v in out.values())


def test_empty_input_returns_empty_dict() -> None:
    assert canonical_pairs([]) == {}
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/test_gismo_gold.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pantrychef.eval.gismo_gold'`.

- [ ] **Step 3: Implement the transform**

Create `pantrychef/eval/gismo_gold.py`:

```python
"""Pure transform for the GISMo published substitution benchmark.

Decoded (source, target) ingredient-name pairs -> canonicalized gold mapping.
No I/O or network here so it is unit-testable in CI; the download + pickle
decode live in scripts/fetch_gismo_gold.py.

Source: Fatemi et al., "Learning to Substitute Ingredients in Recipes"
(arXiv:2302.07960). Benchmark data is CC BY-NC 4.0.
"""

from __future__ import annotations

from collections.abc import Iterable

from pantrychef.ingredients.normalize import canonicalize


def canonical_pairs(raw_pairs: Iterable[tuple[str, str]]) -> dict[str, set[str]]:
    """Canonicalize raw (source, target) pairs into a source -> {targets} gold map.

    Both sides are run through the project `canonicalize()`. Pairs are dropped
    when either side is empty after canonicalization or when source == target
    (a self-substitution carries no signal).
    """
    gold: dict[str, set[str]] = {}
    for src_raw, tgt_raw in raw_pairs:
        src = canonicalize(src_raw)
        tgt = canonicalize(tgt_raw)
        if not src or not tgt or src == tgt:
            continue
        gold.setdefault(src, set()).add(tgt)
    return gold
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_gismo_gold.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Lint + full suite**

Run: `uv run ruff check pantrychef tests && uv run ruff format --check pantrychef tests && uv run pytest -q`
Expected: all green. (Note: run BOTH `ruff check` and `ruff format --check` — CI enforces both.)

- [ ] **Step 6: Commit**

```bash
git add pantrychef/eval/gismo_gold.py tests/test_gismo_gold.py
git commit -m "feat(eval): pure canonical_pairs transform for GISMo gold"
```

---

## Task 2: `--gold` / `--gold-name` flag on the eval entrypoint

**Files:**
- Modify: `pantrychef/eval/substitution_main.py` (`main()`, currently takes no args; loads a hardcoded `gold_csv`)
- Test: `tests/test_substitution_main.py`

Current `main()` hardcodes `gold_csv = s.data_dir / "eval" / "subs_gold.csv"` and calls `load_pairs_csv(gold_csv)`. Add argparse so an alternate gold file can be evaluated without touching the mined default.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_substitution_main.py`:

```python
def test_parse_args_defaults_to_mined_gold() -> None:
    from pantrychef.eval.substitution_main import parse_args

    args = parse_args([])
    assert args.gold is None  # None => main() uses the mined default path
    assert args.gold_name == "mined"


def test_parse_args_accepts_gold_override(tmp_path) -> None:
    from pantrychef.eval.substitution_main import parse_args

    p = tmp_path / "subs_gold_gismo.csv"
    args = parse_args(["--gold", str(p), "--gold-name", "gismo"])
    assert args.gold == str(p)
    assert args.gold_name == "gismo"
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/test_substitution_main.py -k parse_args -v`
Expected: FAIL — `ImportError: cannot import name 'parse_args'`.

- [ ] **Step 3: Implement argparse**

In `pantrychef/eval/substitution_main.py`, add an importable `parse_args` and wire it into `main()`. Add at module level (after imports):

```python
import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args. --gold overrides the gold CSV (default: mined gold);
    --gold-name is a label for the log line."""
    ap = argparse.ArgumentParser(description="Substitution ablation leaderboard.")
    ap.add_argument("--gold", default=None, help="Path to a gold CSV (source,target). Default: mined subs_gold.csv")
    ap.add_argument("--gold-name", default="mined", help="Label for the gold set in the log output.")
    return ap.parse_args(argv)
```

Then change `main()` to accept args and use them. Replace the signature `def main() -> int:` with `def main(argv: list[str] | None = None) -> int:`, and inside, replace the hardcoded gold line:

```python
    args = parse_args(argv)
    gold_csv = args.gold if args.gold else (s.data_dir / "eval" / "subs_gold.csv")
```

Update the missing-artifacts check and the coverage log to use `args.gold_name`. Specifically, after `gold = load_pairs_csv(gold_csv)`, change the coverage log line to include the label:

```python
    log.info("[%s] Coverage: %s", args.gold_name, {k: round(v, 3) for k, v in cov.items()})
```

Leave the `__main__` block as `raise SystemExit(main())` — argv defaults to `sys.argv[1:]` via argparse.

**⚠ pytest pitfall:** `parse_args(None)` reads `sys.argv`, which under pytest holds pytest's own args (`-q`, test paths) and would make `main()` raise `SystemExit`. Grep `tests/test_substitution_main.py` for any existing `main()` call; if present, change it to `main([])` so it parses an empty argv. (The new `parse_args` tests already pass explicit lists, so they are safe.)

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_substitution_main.py -k parse_args -v`
Expected: PASS (2 tests). Then `uv run pytest tests/test_substitution_main.py -q` — existing tests still pass (default path unchanged).

- [ ] **Step 5: Lint + format + full suite**

Run: `uv run ruff check pantrychef tests && uv run ruff format --check pantrychef tests && uv run pytest -q`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add pantrychef/eval/substitution_main.py tests/test_substitution_main.py
git commit -m "feat(eval): --gold/--gold-name flag so alternate gold sets can be scored"
```

---

## Task 3: Schema introspection spike (download + inspect, no converter yet)

**Files:**
- Create (temporary spike, will be extended in Task 4): `scripts/fetch_gismo_gold.py`

The exact pkl record schema is unknown until inspected. This task downloads the two pkls and prints their structure so Task 4 maps the correct fields. Note: `vocab_ingrs.pkl` may be a custom `Vocabulary` class instance — if `pickle.load` raises `ModuleNotFoundError`/`AttributeError` for a GISMo class, record that; Task 4 will add a minimal compatibility shim or a custom `Unpickler`.

- [ ] **Step 1: Write the download + introspect spike**

Create `scripts/fetch_gismo_gold.py` with ONLY the download + introspect portion for now:

```python
"""Fetch the GISMo published substitution benchmark and convert it to a gold CSV.

Source: Fatemi et al., "Learning to Substitute Ingredients in Recipes"
(arXiv:2302.07960), facebookresearch/gismo. Benchmark data is **CC BY-NC 4.0** —
attribute, non-commercial. Their .pkl files and the derived CSV are NOT committed.

    uv run python scripts/fetch_gismo_gold.py            # download + convert
    uv run python scripts/fetch_gismo_gold.py --inspect  # print pkl structure only
"""

from __future__ import annotations

import argparse
import pickle
import urllib.request
from pathlib import Path

from pantrychef.config import get_settings

BASE = "https://dl.fbaipublicfiles.com/gismo"
FILES = {
    "test_comments_subs.pkl": f"{BASE}/test_comments_subs.pkl",
    "vocab_ingrs.pkl": f"{BASE}/vocab_ingrs.pkl",
}


def _download(dest_dir: Path) -> dict[str, Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, url in FILES.items():
        out = dest_dir / name
        if not out.exists():
            print(f"downloading {url} -> {out}")
            urllib.request.urlretrieve(url, out)  # noqa: S310 (trusted public URL)
        paths[name] = out
    return paths


def _inspect(paths: dict[str, Path]) -> None:
    for name, p in paths.items():
        print(f"\n=== {name} ({p.stat().st_size} bytes) ===")
        try:
            with open(p, "rb") as f:
                obj = pickle.load(f)
        except Exception as e:  # noqa: BLE001 - spike: report whatever blocks load
            print(f"  pickle.load FAILED: {type(e).__name__}: {e}")
            continue
        print(f"  type={type(obj)}")
        if isinstance(obj, list):
            print(f"  len={len(obj)}  first={obj[0]!r}")
            if len(obj) > 1:
                print(f"  second={obj[1]!r}")
        elif isinstance(obj, dict):
            keys = list(obj)[:5]
            print(f"  dict len={len(obj)}  sample keys={keys}")
            print(f"  sample item={ {k: obj[k] for k in keys} !r}")
        else:
            print(f"  attrs={[a for a in dir(obj) if not a.startswith('__')][:20]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true", help="Print pkl structure and exit.")
    args = ap.parse_args()

    s = get_settings()
    paths = _download(s.raw_dir / "gismo")
    if args.inspect:
        _inspect(paths)
        return 0
    # Conversion implemented in Task 4.
    print("download complete; run with --inspect to view structure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Lint the spike**

Run: `uv run ruff check scripts/fetch_gismo_gold.py && uv run ruff format --check scripts/fetch_gismo_gold.py`
Expected: clean (the `# noqa` comments suppress the urllib/broad-except lint).

- [ ] **Step 3: Run the introspection**

Run: `uv run python scripts/fetch_gismo_gold.py --inspect`
Expected: downloads the two pkls to `data/raw/gismo/`, prints the type/len/first-record of `test_comments_subs.pkl` and the type/keys of `vocab_ingrs.pkl`. **Record the actual structure** — the record field layout (where source/target ingredient IDs live) and the vocab decoder shape (`idx2word` dict vs `.word2idx`). If `vocab_ingrs.pkl` fails to unpickle due to a missing GISMo class, note the exact class/module name in the error — Task 4 needs it.

- [ ] **Step 4: Do NOT commit yet** — the script is incomplete (conversion added in Task 4). The downloaded pkls live under `data/raw/gismo/` and must stay out of git (Task 5 adds the gitignore entry before any `git add`).

---

## Task 4: Decode + write the gold CSV (uses Task 3 findings)

**Files:**
- Modify: `scripts/fetch_gismo_gold.py` (replace the Task-4 stub in `main()`)

Implement the decode against the schema confirmed in Task 3. The code below assumes GISMo's documented layout: `test_comments_subs.pkl` is a list of records each exposing a source and target ingredient **ID**, and `vocab_ingrs.pkl` decodes IDs to strings. **Confirm the exact field access against the Task-3 printout and adjust the two marked lines** (`_extract_pairs` field access and the vocab decode) if they differ; the rest is schema-independent.

- [ ] **Step 1: Add the decode + write logic**

Add these helpers to `scripts/fetch_gismo_gold.py` (above `main()`):

```python
from pantrychef.eval.gismo_gold import canonical_pairs


def _load_vocab_decoder(vocab_path: Path) -> dict[int, str]:
    """Return an id -> ingredient-string map from vocab_ingrs.pkl.

    GISMo's Vocabulary exposes `idx2word` (dict[int, str|list]). If unpickling
    needs a missing GISMo class, a custom Unpickler maps it to a plain dict.
    """
    with open(vocab_path, "rb") as f:
        vocab_obj = pickle.load(f)
    # >>> ADJUST per Task-3 output if the attribute/shape differs <<<
    idx2word = vocab_obj.idx2word if hasattr(vocab_obj, "idx2word") else vocab_obj
    out: dict[int, str] = {}
    for k, v in dict(idx2word).items():
        name = v[0] if isinstance(v, (list, tuple)) and v else v
        out[int(k)] = str(name)
    return out


def _extract_pairs(records: list, decode: dict[int, str]) -> list[tuple[str, str]]:
    """Pull (source_name, target_name) pairs from the test records.

    >>> ADJUST the field access per Task-3 output. <<< GISMo records typically
    carry source/target ingredient IDs; decode each via `decode`. Records whose
    IDs are not in the decoder are skipped.
    """
    pairs: list[tuple[str, str]] = []
    for rec in records:
        # Expected GISMo shape: rec is a dict/tuple with a source id and target id.
        src_id, tgt_id = _record_ids(rec)  # defined below; ADJUST per Task 3
        if src_id in decode and tgt_id in decode:
            pairs.append((decode[src_id], decode[tgt_id]))
    return pairs


def _record_ids(rec) -> tuple[int, int]:
    """Return (source_id, target_id) from one record. ADJUST per Task-3 schema.

    Handles the two most common GISMo shapes; extend per the actual printout:
      - dict with keys like {'id': ..., 'subs': (a, b)} or {'source':, 'target':}
      - tuple/list (source_id, target_id, *context)
    """
    if isinstance(rec, dict):
        if "subs" in rec:
            a, b = rec["subs"]
            return int(a), int(b)
        return int(rec["source"]), int(rec["target"])
    return int(rec[0]), int(rec[1])
```

Then replace the Task-4 stub at the end of `main()` (the `print("download complete; ...")` line) with:

```python
    decode = _load_vocab_decoder(paths["vocab_ingrs.pkl"])
    with open(paths["test_comments_subs.pkl"], "rb") as f:
        records = pickle.load(f)
    raw_pairs = _extract_pairs(list(records), decode)
    gold = canonical_pairs(raw_pairs)

    if not gold:
        print("ERROR: 0 gold pairs after decode/canonicalize — schema mismatch?")
        return 1

    # Coverage preview vs our vocab.
    import json

    vocab = set(json.load(open(s.processed_dir / "vocab.json", encoding="utf-8")))
    covered_src = sum(1 for q in gold if q in vocab)
    out_csv = s.data_dir / "eval" / "subs_gold_gismo.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        f.write("source,target\n")
        for src in sorted(gold):
            for tgt in sorted(gold[src]):
                f.write(f"{src},{tgt}\n")
    n_pairs = sum(len(v) for v in gold.values())
    print(f"wrote {out_csv}: {len(gold)} sources, {n_pairs} pairs; "
          f"sources in our vocab: {covered_src}/{len(gold)}")
    return 0
```

- [ ] **Step 2: Lint + format**

Run: `uv run ruff check scripts/fetch_gismo_gold.py && uv run ruff format --check scripts/fetch_gismo_gold.py`
Expected: clean.

- [ ] **Step 3: Run the converter end-to-end**

Run: `uv run python scripts/fetch_gismo_gold.py`
Expected: prints `wrote .../subs_gold_gismo.csv: N sources, M pairs; sources in our vocab: K/N`. If it prints the 0-pairs error, the field access in `_record_ids`/`_load_vocab_decoder` doesn't match the real schema — re-check Task 3 output and adjust the two marked spots, then rerun.

- [ ] **Step 4: Commit the script (NOT the data — Task 5 gitignores it first)**

Do Task 5 BEFORE this commit so the pkl/CSV can't be staged. Then:

```bash
git add scripts/fetch_gismo_gold.py
git commit -m "feat(eval): GISMo pkl->gold-csv converter (CC BY-NC, data uncommitted)"
```

---

## Task 5: Gitignore the GISMo data

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add ignore entries**

Append to `.gitignore`:

```
# GISMo published benchmark (CC BY-NC 4.0 — do not commit their data or derivations)
data/raw/gismo/
data/eval/subs_gold_gismo.csv
```

- [ ] **Step 2: Verify nothing GISMo is staged**

Run: `git status --short`
Expected: `data/raw/gismo/` and `data/eval/subs_gold_gismo.csv` do NOT appear (ignored). If they appear, the ignore entry path is wrong — fix before any `git add -A`.

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore GISMo benchmark data (CC BY-NC)"
```

---

## Task 6: Run the GISMo eval + document

**Files:**
- Modify: `docs/EVALUATION.md` (the "Published reference (pending manual download)" table, lines ~74-79)

- [ ] **Step 1: Run the ablation against the GISMo gold**

Run: `uv run python -m pantrychef.eval.substitution_main --gold data/eval/subs_gold_gismo.csv --gold-name gismo`
Expected: logs `[gismo] Coverage: {...}` (pair_coverage < 1.0 — Recipe1M vocab ≠ ours) and three arm rows (emb-only / graph-only / hybrid) with MRR + recall@k. **Record the coverage and the three arms' numbers.**

- [ ] **Step 2: Fill the Published-reference table in `docs/EVALUATION.md`**

Replace the placeholder "Published reference (pending manual download)" table with the real GISMo comparison. Use the numbers from Step 1. Structure:

```markdown
#### vs GISMo published benchmark (CC BY-NC 4.0)

> **Source:** Fatemi et al., "Learning to Substitute Ingredients in Recipes"
> (arXiv:2302.07960), facebookresearch/gismo. Gold = their `test_comments_subs.pkl`
> decoded via `vocab_ingrs.pkl`, canonicalized to our ingredient space. Their data
> is **CC BY-NC 4.0** and is NOT committed (regenerate via `scripts/fetch_gismo_gold.py`).
> **Coverage caveat:** their Recipe1M vocab ≠ our RecipeNLG 30k vocab, so only the
> covered subset (pair_coverage = <FILL>%) is scored; uncovered pairs are skipped,
> not penalized. Numbers are our arms on their gold, NOT a reproduction of their model.

| Arm | MRR | R@5 | R@10 | Coverage | Run |
| --- | --- | --- | ---- | -------- | --- |
| emb-only (food2vec baseline) | <FILL> | <FILL> | <FILL> | <FILL> | gismo, 2026-06-01 |
| graph-only + overlap-shrink | <FILL> | <FILL> | <FILL> | <FILL> | gismo, 2026-06-01 |
| hybrid | <FILL> | <FILL> | <FILL> | <FILL> | gismo, 2026-06-01 |

_GISMo's own published headline metric is MRR (see arXiv:2302.07960 §4); their
model's number is read from the paper, not reproduced here — our comparison is
arm-vs-their-gold on the covered subset._
```

Fill every `<FILL>` from Step 1's output. (These are NOT placeholders to leave — they are filled at execution time from the actual run.)

- [ ] **Step 3: Lint docs-adjacent + full suite**

Run: `uv run pytest -q && uv run ruff check pantrychef tests scripts && uv run ruff format --check .`
Expected: all green (the converter run wrote only gitignored data; no test depends on it).

- [ ] **Step 4: Commit the docs**

```bash
git add docs/EVALUATION.md
git commit -m "docs(eval): vs GISMo published benchmark row + coverage caveat"
```

---

## Out of Scope

- Only the `test` split. train/val pkls are YAGNI.
- No reproduction of GISMo's model — we score OUR arms on THEIR gold.
- No change to the mined-gold eval or the substitution model.
