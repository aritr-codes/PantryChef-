from pantrychef.common.types import Recipe
from pantrychef.data.clean import clean_recipes, to_recipe
from pantrychef.data.store import load_recipes, save_recipes
from pantrychef.ingredients.vocab import build_vocabulary


def test_to_recipe_builds_canonical_set(sample_raws) -> None:
    vocab = build_vocabulary(sample_raws, min_count=1)
    recipe = to_recipe(sample_raws[0], "r0", vocab)
    assert recipe.recipe_id == "r0"
    assert recipe.title == "Pancakes"
    # Order-preserving: canonical follows first-occurrence order in ingredients.
    assert recipe.canonical == ["flour", "egg", "milk", "sugar"]


def test_clean_recipes_dedups_titles_and_ids(sample_raws) -> None:
    vocab = build_vocabulary(sample_raws, min_count=1)
    # Exact dupe + a case/whitespace variant — both dedup by normalized title.
    variant = sample_raws[0].model_copy(update={"title": "  PANCAKES  "})
    dup = sample_raws + [sample_raws[0], variant]
    cleaned = list(clean_recipes(dup, vocab))
    titles = [r.title for r in cleaned]
    assert titles.count("Pancakes") == 1
    assert [r.recipe_id for r in cleaned] == [f"r{i}" for i in range(len(cleaned))]


def test_clean_recipes_skips_empty_titles(sample_raws) -> None:
    vocab = build_vocabulary(sample_raws, min_count=1)
    blank = sample_raws[0].model_copy(update={"title": "   "})
    cleaned = list(clean_recipes([blank, *sample_raws], vocab))
    assert all(r.title.strip() for r in cleaned)
    assert len(cleaned) == len(sample_raws)


def test_store_roundtrip(tmp_path, sample_raws) -> None:
    vocab = build_vocabulary(sample_raws, min_count=1)
    cleaned = list(clean_recipes(sample_raws, vocab))
    p = tmp_path / "recipes.jsonl"
    save_recipes(cleaned, p)
    loaded = load_recipes(p)
    assert [r.recipe_id for r in loaded] == [r.recipe_id for r in cleaned]
    assert loaded[0].canonical == cleaned[0].canonical


def test_load_recipes_drops_ingredients_raw(tmp_path) -> None:
    # ingredients_raw is the bulk of the full-corpus file and is never read after
    # cleaning, so the loader drops it to keep the resident index lean.
    r = Recipe(
        recipe_id="r0",
        title="Soup",
        ingredients_raw=["1 cup flour", "2 eggs"],
        canonical=["flour", "egg"],
    )
    p = tmp_path / "r.jsonl"
    save_recipes([r], p)
    loaded = load_recipes(p)
    assert loaded[0].ingredients_raw == []  # dropped
    assert loaded[0].canonical == ["flour", "egg"]  # preserved
    assert loaded[0].title == "Soup"
    assert loaded[0].recipe_id == "r0"


def test_load_recipes_interns_canonical_across_recipes(tmp_path) -> None:
    # Equal canonical ingredients share one str object across recipes (memory
    # dedup): ~30k distinct ingredients backing 1.27M recipes.
    a = Recipe(recipe_id="a", title="A", canonical=["flour", "egg"])
    b = Recipe(recipe_id="b", title="B", canonical=["flour", "milk"])
    p = tmp_path / "r.jsonl"
    save_recipes([a, b], p)
    la, lb = load_recipes(p)
    flour_a = la.canonical[la.canonical.index("flour")]
    flour_b = lb.canonical[lb.canonical.index("flour")]
    assert flour_a is flour_b


def test_load_recipes_limit_zero_returns_empty(tmp_path) -> None:
    p = tmp_path / "r.jsonl"
    save_recipes([Recipe(recipe_id="a", title="A", canonical=["flour"])], p)
    assert load_recipes(p, limit=0) == []
