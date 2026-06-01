from pantrychef.eval.gismo_gold import canonical_pairs


def test_canonicalizes_both_sides() -> None:
    out = canonical_pairs([("Butter", "Margarine"), ("eggs", "egg-whites")])
    assert out["butter"] == {"margarine"}
    assert out["egg"] == {"egg white"}


def test_merges_multiple_targets_per_source() -> None:
    out = canonical_pairs([("butter", "margarine"), ("butter", "oil")])
    assert out["butter"] == {"margarine", "oil"}


def test_drops_self_pairs_and_empties() -> None:
    out = canonical_pairs([("Butter", "butter"), ("", "oil"), ("salt", "   ")])
    assert "butter" not in out
    assert out == {} or all(v for v in out.values())


def test_empty_input_returns_empty_dict() -> None:
    assert canonical_pairs([]) == {}
