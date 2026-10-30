"""Tests for the substitution ablation leaderboard entrypoint."""

from __future__ import annotations

from types import SimpleNamespace

from pantrychef.common.types import Recipe
from pantrychef.eval.substitution_main import main, parse_args, run_ablation
from pantrychef.substitution.bundle import DEFAULT_BUNDLE_NAME
from pantrychef.substitution.config import SubConfig
from pantrychef.substitution.train import train_artifacts

VOCAB = ["butter", "oil", "flour", "egg", "milk"]
CANONICAL_B = ["butter", "flour", "egg", "milk"]
CANONICAL_O = ["oil", "flour", "egg", "milk"]
RECIPES = [Recipe(recipe_id=f"b{i}", title="t", canonical=CANONICAL_B) for i in range(15)] + [
    Recipe(recipe_id=f"o{i}", title="t", canonical=CANONICAL_O) for i in range(15)
]


def _write_bundle(tmp_path):
    art = train_artifacts(RECIPES, VOCAB, SubConfig(dims=16, window=10, epochs=2))
    bundle_dir = tmp_path / "models" / "substitution"
    bundle = bundle_dir / DEFAULT_BUNDLE_NAME
    art.save_bundle(bundle)
    return bundle


def _explode(*_args, **_kwargs):
    raise AssertionError("offline rebuild should not happen during substitution evaluation")


def test_run_ablation_returns_rows() -> None:
    art = train_artifacts(RECIPES, VOCAB, SubConfig(dims=16, window=10, epochs=2))
    gold = {"butter": {"oil"}}
    rows = run_ablation(art, VOCAB, gold, alphas=(1.0, 0.0, 0.5))
    assert {r["arm"] for r in rows} == {"emb-only", "graph-only", "hybrid"}
    assert all("mrr" in r for r in rows)


def test_parse_args_defaults_to_mined_gold() -> None:
    args = parse_args([])
    assert args.gold is None  # None => main() uses the mined default path
    assert args.gold_name == "mined"


def test_parse_args_accepts_gold_override(tmp_path) -> None:
    p = tmp_path / "subs_gold_gismo.csv"
    args = parse_args(["--gold", str(p), "--gold-name", "gismo"])
    assert args.gold == str(p)
    assert args.gold_name == "gismo"


def test_main_uses_bundle_without_corpus_rebuild(tmp_path, monkeypatch) -> None:
    bundle = _write_bundle(tmp_path)
    gold_csv = tmp_path / "gold.csv"
    gold_csv.write_text("source,target\nbutter,oil\n", encoding="utf-8")

    import pantrychef.data.store as store
    import pantrychef.eval.substitution_main as sub_main
    import pantrychef.ingredients.vocab as vocab
    import pantrychef.substitution.cooccur as cooccur

    monkeypatch.setattr(store, "load_recipes", _explode)
    monkeypatch.setattr(vocab, "load_vocabulary", _explode)
    monkeypatch.setattr(cooccur, "build_cooccurrence", _explode)
    monkeypatch.setattr(cooccur, "sppmi", _explode)
    monkeypatch.setattr(
        sub_main,
        "get_settings",
        lambda: SimpleNamespace(models_dir=bundle.parent.parent, data_dir=tmp_path / "data"),
    )
    monkeypatch.setattr(sub_main, "load_curated", lambda _path: {"butter": {"oil"}})

    assert main(["--gold", str(gold_csv), "--gold-name", "test"]) == 0
