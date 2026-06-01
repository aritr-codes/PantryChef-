from pantrychef.ingredients.parser import build_match_index, match_canonical, parse

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


def test_build_match_index_maps_tokens() -> None:
    # df: all=1, purpose=1, flour=2 -> "all purpose flour" buckets under "all"
    # (tie all/purpose broken by string), bare "flour" under "flour".
    idx = build_match_index(["all purpose flour", "flour"])
    assert idx["all"] == ["all purpose flour"]
    assert idx["flour"] == ["flour"]


def test_build_match_index_buckets_by_rarest_token() -> None:
    # Each entry is registered under ONLY its rarest word (min document
    # frequency). "cheese" appears in all three, so it is the rarest word of
    # none of the multi-word entries -> its bucket holds only the bare "cheese".
    idx = build_match_index(["cream cheese", "cheddar cheese", "cheese"])
    assert idx["cheese"] == ["cheese"]
    assert idx["cream"] == ["cream cheese"]
    assert idx["cheddar"] == ["cheddar cheese"]
    # The common word does NOT fan out to every entry containing it.
    assert "cream cheese" not in idx["cheese"]
    assert "cheddar cheese" not in idx["cheese"]


def test_match_index_equivalence() -> None:
    # The token-pruned path MUST return identical results to the full scan.
    idx = build_match_index(VOCAB)
    phrases = [
        "2 cups all-purpose flour",
        "plain flour",
        "3 eggs",
        "extra virgin olive oil",
        "dragonfruit",
        "a pinch of salt and sugar",
        "",
    ]
    for p in phrases:
        assert match_canonical(p, VOCAB, idx) == match_canonical(p, VOCAB)


def test_match_index_equivalence_common_tokens() -> None:
    # Bucketing must give byte-identical results to the full scan even when
    # many entries share common words — this is the property the full-corpus
    # speedup depends on.
    vocab = [
        "cheese",
        "cream cheese",
        "cheddar cheese",
        "blue cheese",
        "goat cheese",
        "cheese sauce",
        "tomato sauce",
        "soy sauce",
        "sauce",
        "cream",
        "tomato",
        "cheddar cheese sauce",
    ]
    idx = build_match_index(vocab)
    phrases = [
        "sharp cheddar cheese",
        "cream cheese frosting",
        "a little tomato sauce",
        "cheddar cheese sauce for nachos",
        "just cheese",
        "blue cheese and soy sauce",
        "nothing relevant here",
        "",
    ]
    for p in phrases:
        assert match_canonical(p, vocab, idx) == match_canonical(p, vocab)


def test_parse_with_index_matches_without() -> None:
    idx = build_match_index(VOCAB)
    with_idx = parse("2 cups all-purpose flour, sifted", VOCAB, idx)
    without = parse("2 cups all-purpose flour, sifted", VOCAB)
    assert with_idx == without
    assert with_idx.canonical == "all purpose flour"


def test_match_index_equivalence_tied_scores() -> None:
    # Two entries with identical (word_count, byte_length) both match the
    # phrase. The indexed path and the full scan MUST agree on the winner, and
    # the result must be deterministic (independent of set iteration order).
    vocab = ["goat cheese", "blue cheese"]
    idx = build_match_index(vocab)
    phrase = "goat blue cheese"
    assert match_canonical(phrase, vocab, idx) == match_canonical(phrase, vocab)
    # Deterministic winner: tie broken by entry string -> "goat cheese" > "blue cheese".
    assert match_canonical(phrase, vocab, idx) == "goat cheese"
    assert match_canonical(phrase, vocab) == "goat cheese"
