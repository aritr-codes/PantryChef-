from pantrychef.ingredients.parser import match_canonical, parse

VOCAB = ["all purpose flour", "flour", "olive oil", "egg", "milk", "sugar", "salt"]


def test_match_canonical_longest_first() -> None:
    assert match_canonical("2 cups all-purpose flour", VOCAB) == "all purpose flour"
    assert match_canonical("plain flour", VOCAB) == "flour"


def test_match_canonical_none_when_absent() -> None:
    assert match_canonical("dragonfruit", VOCAB) is None


def test_parse_full_line() -> None:
    pi = parse("2 cups all-purpose flour, sifted", VOCAB)
    assert pi.quantity == 2.0
    assert pi.unit == "cup"
    assert pi.canonical == "all purpose flour"
    assert pi.modifier == "sifted"


def test_parse_plural_and_unicode() -> None:
    pi = parse("3 eggs", VOCAB)
    assert pi.quantity == 3.0
    assert pi.canonical == "egg"

    pi2 = parse("½ cup milk", VOCAB)
    assert pi2.quantity == 0.5
    assert pi2.unit == "cup"
    assert pi2.canonical == "milk"


def test_parse_without_vocab_returns_none_canonical() -> None:
    pi = parse("2 cups flour")
    assert pi.quantity == 2.0
    assert pi.unit == "cup"
    assert pi.canonical is None
