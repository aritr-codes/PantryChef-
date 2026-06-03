def test_dietary_importable_from_new_home():
    from pantrychef.dietary import DietTagger
    t = DietTagger(known=["butter", "olive oil"])
    assert "dairy" in t.tags("butter")
    assert t.is_valid("butter", "vegan") is False


def test_substitution_shim_reexports_same_class():
    from pantrychef.dietary import DietTagger as New
    from pantrychef.substitution.dietary import DietTagger as Shim
    assert Shim is New
