"""Tests for the substitution ablation leaderboard entrypoint."""

from __future__ import annotations

from pantrychef.eval.substitution_main import run_ablation


def test_run_ablation_returns_rows() -> None:
    from pantrychef.common.types import Recipe
    from pantrychef.substitution.config import SubConfig
    from pantrychef.substitution.train import train_artifacts

    vocab = ["butter", "oil", "flour", "egg", "milk"]
    canonical_b = ["butter", "flour", "egg", "milk"]
    canonical_o = ["oil", "flour", "egg", "milk"]
    recipes = (
        [Recipe(recipe_id=f"b{i}", title="t", canonical=canonical_b) for i in range(15)]
        + [Recipe(recipe_id=f"o{i}", title="t", canonical=canonical_o) for i in range(15)]
    )
    art = train_artifacts(recipes, vocab, SubConfig(dims=16, window=10, epochs=2))
    gold = {"butter": {"oil"}}
    rows = run_ablation(art, vocab, gold, alphas=(1.0, 0.0, 0.5))
    assert {r["arm"] for r in rows} == {"emb-only", "graph-only", "hybrid"}
    assert all("mrr" in r for r in rows)
