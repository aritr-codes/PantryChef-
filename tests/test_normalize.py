from pantrychef.ingredients.normalize import normalize, singularize


def test_normalize_lowercases_and_strips_parens_punct() -> None:
    assert normalize("All-Purpose Flour (sifted)!") == "all-purpose flour"


def test_normalize_collapses_whitespace() -> None:
    assert normalize("  olive   oil ") == "olive oil"


def test_singularize_common_plurals() -> None:
    assert singularize("eggs") == "egg"
    assert singularize("tomatoes") == "tomato"
    assert singularize("berries") == "berry"


def test_singularize_leaves_short_and_double_s() -> None:
    assert singularize("oil") == "oil"
    assert singularize("molasses") == "molasses"
