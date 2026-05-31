from pathlib import Path

from pantrychef.eval.subs_gold import load_curated

REPO = Path(__file__).resolve().parents[1]


def test_curated_subs_loads_and_canonical() -> None:
    gold = load_curated(REPO / "data" / "eval" / "curated_subs.json")
    assert "butter" in gold
    assert "olive oil" in gold["butter"]
    # all keys/values already canonical (idempotent)
    from pantrychef.ingredients.normalize import canonicalize

    for k, vs in gold.items():
        assert canonicalize(k) == k
        assert all(canonicalize(v) == v for v in vs)


def test_curated_diets_pairs() -> None:
    import json

    diets = json.loads((REPO / "data" / "eval" / "curated_diets.json").read_text(encoding="utf-8"))
    assert any(d == "vegan" for _, d in diets)
