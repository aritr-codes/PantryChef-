from pantrychef.eval.nutrition_main import impute_gate_tripped
from pantrychef.nutrition.config import NutritionConfig


def test_impute_gate_trips_on_low_match_coverage():
    cfg = NutritionConfig()
    rep = {"match_coverage": 0.79, "median_unresolved_mass": 0.10}
    assert impute_gate_tripped(rep, cfg) is True


def test_impute_gate_trips_on_high_unresolved_mass_even_at_threshold_match():
    cfg = NutritionConfig()
    rep = {"match_coverage": 0.80, "median_unresolved_mass": 0.50}
    assert impute_gate_tripped(rep, cfg) is True
