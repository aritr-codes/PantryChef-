import random

from pantrychef.common.types import Recipe
from pantrychef.recommender.config import RecConfig
from pantrychef.recommender.query_sim import QuerySim, is_train, make_query


def _recipe(rid: str, ings: list[str]) -> Recipe:
    return Recipe(recipe_id=rid, title=rid, canonical=ings)


def test_is_train_deterministic_and_split():
    assert is_train("abc") == is_train("abc")
    ids = [str(i) for i in range(2000)]
    frac = sum(is_train(i) for i in ids) / len(ids)
    assert 0.7 < frac < 0.9


def test_make_query_masks_fraction_and_sets_gold():
    r = _recipe("r1", ["flour", "egg", "milk", "sugar", "butter"])  # len 5
    q = make_query(r, RecConfig(mask_fraction=0.4, seed=1), random.Random(1))
    assert isinstance(q, QuerySim)
    assert q.gold_id == "r1"
    assert len(q.hidden) == 2  # 0.4 * 5 = 2
    assert len(q.pantry) == 3
    assert set(q.pantry) | set(q.hidden) == set(r.canonical)
    assert not (set(q.pantry) & set(q.hidden))


def test_make_query_floor_enforces_min_one_hidden():
    # mask_fraction so tiny it rounds to 0 -> floor should give 1
    r = _recipe("rX", ["a", "b", "c", "d"])
    q = make_query(r, RecConfig(mask_fraction=0.01), random.Random(0))
    assert len(q.hidden) == 1


def test_make_query_cap_enforces_min_one_kept():
    # mask_fraction so large it would hide all -> cap should leave 1 kept
    r = _recipe("rY", ["a", "b", "c", "d"])
    q = make_query(r, RecConfig(mask_fraction=0.9), random.Random(0))
    assert len(q.pantry) == 1  # round(3.6)=4 -> capped to 3 hidden


def test_make_query_skips_short_recipe():
    r = _recipe("r3", ["a", "b", "c"])  # < min_recipe_len 4
    assert make_query(r, RecConfig(), random.Random(3)) is None


def test_make_query_dedupes_canonical():
    r = _recipe("r4", ["egg", "egg", "milk", "flour", "sugar"])  # 4 unique
    q = make_query(r, RecConfig(mask_fraction=0.25, seed=4), random.Random(4))
    assert q is not None
    assert len(set(q.pantry) | set(q.hidden)) == 4
