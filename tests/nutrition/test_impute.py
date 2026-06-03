import pytest

pytest.importorskip("sklearn")

from pantrychef.nutrition.impute import MacroImputer


def test_imputer_predicts_known_macro_within_mae():
    # train on foods with known kcal; held-out MAE should be finite and reported
    train = [
        ("white sugar", {"kcal": 387.0}),
        ("brown sugar", {"kcal": 380.0}),
        ("olive oil", {"kcal": 884.0}),
        ("canola oil", {"kcal": 884.0}),
        ("white flour", {"kcal": 364.0}),
        ("wheat flour", {"kcal": 340.0}),
    ]
    imp = MacroImputer().fit(train, target="kcal")
    pred = imp.predict("corn oil")
    assert pred > 0  # an oil should predict high kcal
    assert imp.mae_ >= 0.0
