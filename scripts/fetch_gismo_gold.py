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
from pantrychef.eval.gismo_gold import canonical_pairs
from pantrychef.ingredients.vocab import load_vocabulary

BASE = "https://dl.fbaipublicfiles.com/gismo"
FILES = {
    "test_comments_subs.pkl": f"{BASE}/test_comments_subs.pkl",
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
            # S: trusted source — dl.fbaipublicfiles.com (Facebook Research CDN,
            # same domain as official PyTorch checkpoints). Spike/introspect only;
            # no user-supplied paths reach this call.
            with open(p, "rb") as f:
                obj = pickle.load(f)  # noqa: S301
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
            print(f"  sample item={ {k: obj[k] for k in keys}!r}")
        else:
            print(f"  attrs={[a for a in dir(obj) if not a.startswith('__')][:20]}")


def _extract_pairs(records: list) -> list[tuple[str, str]]:
    """Pull (source, target) name pairs from the GISMo test records.

    Each record is a dict whose ``subs`` key holds a (source, target) tuple of
    ingredient-name strings (underscore-joined, e.g. ``long_grain_rice``). The
    project ``canonicalize`` maps underscores to spaces, so these raw names feed
    straight into ``canonical_pairs`` with no decoding.
    """
    pairs: list[tuple[str, str]] = []
    for rec in records:
        subs = rec.get("subs") if isinstance(rec, dict) else None
        if not subs:
            continue
        try:
            src, tgt = subs
        except (TypeError, ValueError):
            # Not a 2-element iterable (schema drift) — skip rather than crash.
            continue
        pairs.append((str(src), str(tgt)))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true", help="Print pkl structure and exit.")
    args = ap.parse_args()

    s = get_settings()
    paths = _download(s.raw_dir / "gismo")
    if args.inspect:
        _inspect(paths)
        return 0
    with open(paths["test_comments_subs.pkl"], "rb") as f:
        records = pickle.load(f)  # noqa: S301
    raw_pairs = _extract_pairs(list(records))
    gold = canonical_pairs(raw_pairs)

    if not gold:
        print("ERROR: 0 gold pairs after canonicalize — schema mismatch?")
        return 1

    out_csv = s.data_dir / "eval" / "subs_gold_gismo.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        f.write("source,target\n")
        for src in sorted(gold):
            for tgt in sorted(gold[src]):
                f.write(f"{src},{tgt}\n")

    n_pairs = sum(len(v) for v in gold.values())
    vocab_path = s.processed_dir / "vocab.json"
    if vocab_path.exists():
        vocab = set(load_vocabulary(vocab_path))
        covered_src = sum(1 for q in gold if q in vocab)
        cov_note = f"sources in our vocab: {covered_src}/{len(gold)}"
    else:
        cov_note = "vocab.json not found — coverage preview skipped"
    print(f"wrote {out_csv}: {len(gold)} sources, {n_pairs} pairs; {cov_note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
