"""Command-line entrypoint.

`pantrychef cook --have eggs,flour,milk --k 5` ranks recipes by how much of
each recipe your pantry covers, listing what you're missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pantrychef import __version__
from pantrychef.config import get_settings
from pantrychef.data.store import load_recipes
from pantrychef.retrieval.baseline import recommend
from pantrychef.retrieval.index import InvertedIndex


def _default_recipes_path() -> str:
    return str(get_settings().processed_dir / "recipes.jsonl")


def cook_command(have: str, k: int, recipes_path: str | None = None) -> int:
    path = Path(recipes_path or _default_recipes_path())
    if not path.exists():
        print(f"Recipe store not found at {path}. Run `make data` first.")
        return 1
    pantry = [p.strip() for p in have.split(",") if p.strip()]
    index = InvertedIndex.build(load_recipes(path))
    ranked = recommend(index, pantry, k=max(k, 0))
    if not ranked:
        print("No matching recipes. Try more/different ingredients.")
        return 0
    for r in ranked:
        miss = ", ".join(r.missing) if r.missing else "nothing — you can make this!"
        print(f"[{r.score * 100:5.1f}%] {r.title}  (missing: {miss})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pantrychef", description="Pantry-aware recipe finder.")
    parser.add_argument("--version", action="version", version=f"pantrychef {__version__}")
    sub = parser.add_subparsers(dest="command")

    cook = sub.add_parser("cook", help="Find recipes you can make.")
    cook.add_argument("--have", required=True, help="Comma-separated pantry ingredients.")
    cook.add_argument("--k", type=int, default=5, help="How many recipes to show.")
    cook.add_argument("--recipes", default=None, help="Path to recipes.jsonl.")

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.command == "cook":
        return cook_command(have=args.have, k=args.k, recipes_path=args.recipes)

    print(f"PantryChef v{__version__}. Try: pantrychef cook --have eggs,flour,milk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
