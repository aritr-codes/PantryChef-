import json
from pathlib import Path

from pantrychef.nutrition.config import NutritionConfig


def test_defaults():
    c = NutritionConfig()
    assert c.seed == 42
    assert 0 < c.usable_mass_fraction <= 1
    assert c.impute_match_cov_gate == 0.80
    assert c.impute_unresolved_mass_gate == 0.20


def test_curated_files_are_valid_json():
    root = Path(__file__).resolve().parents[2]
    aliases = json.loads((root / "data/nutrition/aliases.json").read_text())
    labels = json.loads((root / "data/nutrition/dietary_labels.json").read_text())
    assert isinstance(aliases, dict)
    assert isinstance(labels, dict)
    # labels map ingredient -> list[str] of dietary categories
    assert all(isinstance(v, list) for v in labels.values())
