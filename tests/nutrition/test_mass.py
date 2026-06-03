import math

from pantrychef.nutrition.mass import to_grams

BUTTER = {"description": "Butter", "per100g": {}, "unit_grams": {"cup": 227.0}}
EGG = {"description": "Egg", "per100g": {}, "unit_grams": {"each": 50.0}}
NODATA = {"description": "X", "per100g": {}, "unit_grams": {}}


def test_mass_unit_direct():
    g, ok = to_grams(2.0, "oz", "anything", NODATA)
    assert ok and math.isclose(g, 56.699, rel_tol=1e-3)


def test_volume_uses_usda_portion():
    g, ok = to_grams(0.5, "cup", "butter", BUTTER)  # 0.5 * 227
    assert ok and math.isclose(g, 113.5)


def test_volume_density_fallback_when_no_portion():
    g, ok = to_grams(1.0, "cup", "milk", NODATA)  # 236.588ml * 1.03 density
    assert ok and math.isclose(g, 243.7, rel_tol=1e-2)


def test_bare_count_uses_each():
    g, ok = to_grams(3.0, None, "egg", EGG)  # 3 * 50
    assert ok and g == 150.0


def test_unresolved_volume_no_density_no_portion():
    g, ok = to_grams(1.0, "cup", "mystery", NODATA)
    assert g is None and ok is False


def test_missing_quantity_unresolved():
    g, ok = to_grams(None, None, "salt", NODATA)
    assert g is None and ok is False
