from pantrychef.cli import cook_command
from pantrychef.data.clean import clean_recipes
from pantrychef.data.store import save_recipes
from pantrychef.ingredients.vocab import build_vocabulary


def test_cook_command_outputs_ranked(tmp_path, sample_raws, capsys) -> None:
    vocab = build_vocabulary(sample_raws, min_count=1)
    recipes = list(clean_recipes(sample_raws, vocab))
    store = tmp_path / "recipes.jsonl"
    save_recipes(recipes, store)

    rc = cook_command(have="flour,egg,milk,sugar", k=3, recipes_path=str(store))
    out = capsys.readouterr().out

    assert rc == 0
    assert "Pancakes" in out
    assert "%" in out  # coverage percentage rendered


def test_cook_command_missing_store_errors(tmp_path, capsys) -> None:
    rc = cook_command(have="flour", k=3, recipes_path=str(tmp_path / "nope.jsonl"))
    out = capsys.readouterr().out
    assert rc == 1
    assert "not found" in out.lower()
