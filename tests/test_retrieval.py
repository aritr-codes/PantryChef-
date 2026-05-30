from pantrychef.data.clean import clean_recipes
from pantrychef.ingredients.vocab import build_vocabulary
from pantrychef.retrieval.baseline import recommend
from pantrychef.retrieval.index import InvertedIndex


def _index(sample_raws):
    vocab = build_vocabulary(sample_raws, min_count=1)
    recipes = list(clean_recipes(sample_raws, vocab))
    return InvertedIndex.build(recipes)


def test_index_postings(sample_raws) -> None:
    idx = _index(sample_raws)
    # "flour" appears in Pancakes (r0) and Sugar Cookies (r2)
    assert len(idx.postings["flour"]) == 2


def test_recommend_ranks_full_match_first(sample_raws) -> None:
    idx = _index(sample_raws)
    ranked = recommend(idx, ["flour", "egg", "milk", "sugar"], k=5)
    assert ranked[0].title == "Pancakes"
    assert ranked[0].score == 1.0
    assert ranked[0].missing == []


def test_recommend_reports_missing(sample_raws) -> None:
    idx = _index(sample_raws)
    ranked = recommend(idx, ["flour", "sugar"], k=5)
    cookies = next(r for r in ranked if r.title == "Sugar Cookies")
    assert set(cookies.matched) == {"flour", "sugar"}
    assert set(cookies.missing) == {"egg", "butter"}


def test_recommend_canonicalizes_pantry(sample_raws) -> None:
    idx = _index(sample_raws)
    # "Eggs" (capitalized, plural) must canonicalize to "egg"
    ranked = recommend(idx, ["Eggs"], k=5)
    assert any(r.title == "Omelette" for r in ranked)
