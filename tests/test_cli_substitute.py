"""Tests for the `substitute` CLI subcommand."""

from __future__ import annotations

import contextlib

from pantrychef.cli import main
from pantrychef.common.types import Recipe
from pantrychef.substitution.bundle import DEFAULT_BUNDLE_NAME
from pantrychef.substitution.config import SubConfig
from pantrychef.substitution.train import train_artifacts

VOCAB = ["butter", "oil", "flour", "egg", "milk"]
RECIPES = [
    Recipe(recipe_id=f"b{i}", title="t", canonical=["butter", "flour", "egg", "milk"])
    for i in range(15)
] + [
    Recipe(recipe_id=f"o{i}", title="t", canonical=["oil", "flour", "egg", "milk"])
    for i in range(15)
]


def _write_bundle(tmp_path):
    art = train_artifacts(RECIPES, VOCAB, SubConfig(dims=16, window=10, epochs=2))
    bundle = tmp_path / DEFAULT_BUNDLE_NAME
    art.save_bundle(bundle)
    return bundle


def _explode(*_args, **_kwargs):
    raise AssertionError("offline rebuild should not happen during substitution inference")


def test_substitute_missing_model_is_graceful(capsys, tmp_path) -> None:
    # No trained model present -> clear message, non-crashing exit code 1.
    rc = main(["substitute", "butter", "--model", str(tmp_path / "nope.kv")])
    out = capsys.readouterr().out
    assert rc == 1
    assert "not found" in out.lower()


def test_substitute_help_lists_diet(capsys) -> None:
    with contextlib.suppress(SystemExit):
        main(["substitute", "--help"])
    out = capsys.readouterr().out
    assert "--diet" in out


def test_substitute_uses_bundle_without_rebuild(tmp_path, monkeypatch, capsys) -> None:
    bundle = _write_bundle(tmp_path)

    import pantrychef.data.store as store
    import pantrychef.ingredients.vocab as vocab
    import pantrychef.substitution.cooccur as cooccur

    monkeypatch.setattr(store, "load_recipes", _explode)
    monkeypatch.setattr(vocab, "load_vocabulary", _explode)
    monkeypatch.setattr(cooccur, "build_cooccurrence", _explode)
    monkeypatch.setattr(cooccur, "sppmi", _explode)

    rc = main(["substitute", "butter", "--model", str(bundle), "--k", "2"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "oil" in out.lower()
