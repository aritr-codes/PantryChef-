from pantrychef.dietary import DietTagger


def _t() -> DietTagger:
    return DietTagger(known=["egg", "eggplant", "beef", "milk", "olive oil"])


def test_eggplant_is_not_egg():
    t = _t()
    assert "egg" not in t.tags("eggplant")   # substring bug fixed
    assert t.is_valid("eggplant", "vegan") is True


def test_inheritance_meat_is_animal_product():
    t = _t()
    assert t.is_valid("beef", "vegan") is False
    assert "meat" in t.tags("beef")


def test_known_cases_preserved():
    t = _t()
    assert "dairy" in t.tags("milk")
    assert t.is_valid("olive oil", "vegan") is True
