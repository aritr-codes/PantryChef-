from pantrychef.recommender.config import RecConfig


def test_defaults():
    c = RecConfig()
    assert c.mask_fraction == 0.3
    assert c.min_recipe_len == 4
    assert c.candidate_cap == 200
    assert c.seed == 13
    assert c.sub_pool == 20
    assert c.queries_per_recipe == 1


def test_override():
    c = RecConfig(mask_fraction=0.5, candidate_cap=50)
    assert c.mask_fraction == 0.5
    assert c.candidate_cap == 50
    assert c.min_recipe_len == 4  # unchanged
