"""Phase 0 smoke tests: the scaffold imports and wires together.

These prove the package is installable and the contracts/config load — a green
CI from day one. Real unit tests arrive with each phase's logic.
"""

from __future__ import annotations

import pantrychef
from pantrychef.common import get_logger
from pantrychef.common.types import ParsedIngredient
from pantrychef.config import get_settings


def test_version() -> None:
    assert pantrychef.__version__ == "0.0.0"


def test_settings_load() -> None:
    s = get_settings()
    assert s.processed_dir == s.data_dir / "processed"
    assert s.mlflow_tracking_uri.startswith("file:")


def test_logger() -> None:
    log = get_logger("pantrychef.test")
    assert log.name == "pantrychef.test"


def test_types_contract() -> None:
    pi = ParsedIngredient(raw="2 cups flour", canonical="flour", quantity=2, unit="cup")
    assert pi.canonical == "flour"
    assert pi.dietary_valid if hasattr(pi, "dietary_valid") else True


def test_cli_runs() -> None:
    from pantrychef.cli import main

    assert main([]) == 0


def test_substitute_has_arm_field() -> None:
    from pantrychef.common.types import Substitute

    s = Substitute(ingredient="margarine", score=0.9, arm="hybrid")
    assert s.arm == "hybrid"
    assert s.dietary_valid is True
    assert Substitute(ingredient="oil", score=0.1).arm is None


def test_recipe_and_scored_recipe_types() -> None:
    from pantrychef.common.types import Recipe, ScoredRecipe

    r = Recipe(
        recipe_id="r0",
        title="Pancakes",
        ingredients_raw=["2 cups flour"],
        canonical=["flour"],
    )
    assert r.canonical == ["flour"]
    assert Recipe(recipe_id="x", title="y").canonical == []  # list default, not None

    s = ScoredRecipe(
        recipe_id="r0",
        score=1.0,
        title="Pancakes",
        matched=["flour"],
        missing=["egg"],
    )
    assert s.title == "Pancakes"
    assert s.matched == ["flour"]
    assert s.missing == ["egg"]
