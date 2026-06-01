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


def substitute_command(
    ingredient: str,
    diet: str | None,
    k: int,
    model_path: str | None = None,
    recipe: str | None = None,
) -> int:
    """Find ingredient substitutes using the trained embedding model."""
    from pantrychef.ingredients.vocab import load_vocabulary
    from pantrychef.substitution.config import SubConfig
    from pantrychef.substitution.cooccur import build_cooccurrence, sppmi
    from pantrychef.substitution.dietary import DietTagger
    from pantrychef.substitution.embeddings import EmbeddingModel
    from pantrychef.substitution.graph import ContextGraph
    from pantrychef.substitution.substitute import Substitutor

    s = get_settings()
    mpath = Path(model_path or (s.models_dir / "substitution" / "word2vec.kv"))
    if not mpath.exists():
        print(f"Substitution model not found at {mpath}. Run scripts/train_substitution.py first.")
        return 1
    recipes_path = s.processed_dir / "recipes.jsonl"
    vocab = load_vocabulary(s.processed_dir / "vocab.json")
    emb = EmbeddingModel.load(mpath)
    recipes = load_recipes(recipes_path)
    cmat, _, _, _ = build_cooccurrence(recipes, vocab)
    cfg = SubConfig(k=k)
    graph = ContextGraph(
        sppmi(cmat, cfg.sppmi_shift),
        vocab,
        cmat,
        lam=cfg.lam,
        overlap_shrink=cfg.overlap_shrink,
    )
    sub = Substitutor(emb=emb, graph=graph, tagger=DietTagger(known=vocab), cfg=cfg)
    ctx = [c.strip() for c in recipe.split(",") if c.strip()] if recipe else None
    results = sub.substitutes(ingredient, diet=diet, recipe=ctx, k=max(k, 0))
    if not results:
        print(f"No substitutes found for '{ingredient}'" + (f" ({diet})" if diet else "") + ".")
        return 0
    for r in results:
        print(f"[{r.score:6.3f}] {r.ingredient}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pantrychef", description="Pantry-aware recipe finder.")
    parser.add_argument("--version", action="version", version=f"pantrychef {__version__}")
    sub = parser.add_subparsers(dest="command")

    cook = sub.add_parser("cook", help="Find recipes you can make.")
    cook.add_argument("--have", required=True, help="Comma-separated pantry ingredients.")
    cook.add_argument("--k", type=int, default=5, help="How many recipes to show.")
    cook.add_argument("--recipes", default=None, help="Path to recipes.jsonl.")

    sub_p = sub.add_parser("substitute", help="Find ingredient substitutes.")
    sub_p.add_argument("ingredient", help="Ingredient to replace.")
    sub_p.add_argument(
        "--diet", default=None, choices=["vegan", "vegetarian", "gluten_free", "dairy_free"]
    )
    sub_p.add_argument("--k", type=int, default=5)
    sub_p.add_argument(
        "--in-recipe", dest="recipe", default=None, help="Comma-separated recipe context."
    )
    sub_p.add_argument("--model", default=None, help="Path to word2vec.kv.")

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.command == "cook":
        return cook_command(have=args.have, k=args.k, recipes_path=args.recipes)

    if args.command == "substitute":
        return substitute_command(
            ingredient=args.ingredient,
            diet=args.diet,
            k=args.k,
            model_path=args.model,
            recipe=args.recipe,
        )

    print(f"PantryChef v{__version__}. Try: pantrychef cook --have eggs,flour,milk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
