from pantrychef.substitution.config import SubConfig


def test_defaults() -> None:
    c = SubConfig()
    assert c.dims == 100
    assert c.window == 50
    assert c.epochs == 5
    assert c.sppmi_shift >= 1.0
    assert 0.0 <= c.alpha <= 1.0
    assert c.lam >= 0.0
    assert c.seed == 42


def test_frozen() -> None:
    import dataclasses

    c = SubConfig()
    try:
        c.dims = 5  # type: ignore[misc]
        raise AssertionError("should be frozen")
    except dataclasses.FrozenInstanceError:
        pass
