from pantrychef.data.clean import clean_recipes
from pantrychef.eval.parser_eval import evaluate_parser
from pantrychef.eval.retrieval_eval import evaluate_retrieval
from pantrychef.ingredients.vocab import build_vocabulary
from pantrychef.retrieval.index import InvertedIndex


def test_evaluate_parser_perfect_on_fixture(sample_raws) -> None:
    vocab = build_vocabulary(sample_raws, min_count=1)
    metrics = evaluate_parser(sample_raws, vocab)
    # Fixture is constructed so the parser recovers exactly the NER sets.
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_evaluate_parser_returns_counts(sample_raws) -> None:
    vocab = build_vocabulary(sample_raws, min_count=1)
    metrics = evaluate_parser(sample_raws, vocab)
    assert metrics["tp"] > 0
    assert set(metrics) == {"precision", "recall", "f1", "tp", "fp", "fn"}


def test_evaluate_retrieval_recall(sample_raws) -> None:
    vocab = build_vocabulary(sample_raws, min_count=1)
    recipes = list(clean_recipes(sample_raws, vocab))
    idx = InvertedIndex.build(recipes)
    metrics = evaluate_retrieval(idx, recipes, k=5, seed=42)
    assert 0.0 <= metrics["recall_at_k"] <= 1.0
    assert metrics["k"] == 5.0
    assert metrics["n"] >= 1.0
