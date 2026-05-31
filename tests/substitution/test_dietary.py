from pantrychef.substitution.dietary import DietTagger

VOCAB = ["butter", "milk", "egg", "olive oil", "tofu", "flour", "wheat bread", "chicken", "honey"]


def _tagger() -> DietTagger:
    return DietTagger(known=VOCAB)


def test_tags_basic() -> None:
    t = _tagger()
    assert "dairy" in t.tags("butter")
    assert "egg" in t.tags("egg")
    assert "meat" in t.tags("chicken")
    assert "gluten" in t.tags("flour")  # curated override
    assert "honey" in t.tags("honey")
    assert t.tags("olive oil") == set()


def test_is_valid_vegan() -> None:
    t = _tagger()
    assert t.is_valid("olive oil", "vegan") is True
    assert t.is_valid("tofu", "vegan") is True
    assert t.is_valid("butter", "vegan") is False
    assert t.is_valid("honey", "vegan") is False
    assert t.is_valid("chicken", "vegan") is False


def test_is_valid_gluten_free() -> None:
    t = _tagger()
    assert t.is_valid("flour", "gluten_free") is False
    assert t.is_valid("wheat bread", "gluten_free") is False
    assert t.is_valid("olive oil", "gluten_free") is True


def test_unknown_excluded_for_constrained_diet() -> None:
    t = _tagger()
    # not in vocab, no keyword match => conservative exclude for a constrained diet
    assert t.is_valid("mystery powder", "vegan") is False
    # but no constraint => allowed
    assert t.is_valid("mystery powder", None) is True


def test_mask_filters() -> None:
    t = _tagger()
    cands = [("butter", 0.9), ("olive oil", 0.7), ("milk", 0.5)]
    assert t.mask(cands, "vegan") == [("olive oil", 0.7)]


def test_coverage_fraction() -> None:
    t = _tagger()
    cov = t.coverage(VOCAB)
    assert 0.0 <= cov <= 1.0
    assert cov > 0.5  # most of this vocab is tagged or oil/tofu (known-safe)
