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
