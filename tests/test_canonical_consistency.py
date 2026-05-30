from pantrychef.data.clean import clean_recipes
from pantrychef.data.schemas import RawRecipe
from pantrychef.ingredients.parser import parse
from pantrychef.ingredients.vocab import build_vocabulary
from pantrychef.retrieval.baseline import recommend
from pantrychef.retrieval.index import InvertedIndex


def test_hyphenated_multiword_symmetry() -> None:
    # vocab entry, parsed recipe line, and pantry query all agree.
    raw = RawRecipe(
        title="Bread",
        ingredients=["2 cups all-purpose flour", "1 tsp baking soda"],
        ner=["all-purpose flour", "baking soda"],
    )
    vocab = build_vocabulary([raw], min_count=1)
    assert "all purpose flour" in vocab
    assert "baking soda" in vocab

    pi = parse("2 cups all-purpose flour", vocab)
    assert pi.canonical == "all purpose flour"


def test_hyphenated_pantry_query_matches_recipe() -> None:
    raw = RawRecipe(
        title="Bread",
        ingredients=["2 cups all-purpose flour", "1 tsp baking soda"],
        ner=["all-purpose flour", "baking soda"],
    )
    vocab = build_vocabulary([raw], min_count=1)
    recipes = list(clean_recipes([raw], vocab))
    idx = InvertedIndex.build(recipes)

    # User types it WITH the hyphen — must still match the stored canonical.
    ranked = recommend(idx, ["all-purpose flour", "baking soda"], k=5)
    assert ranked[0].title == "Bread"
    assert ranked[0].score == 1.0
    assert ranked[0].missing == []
