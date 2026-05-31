from pantrychef.eval.substitution_eval import (
    coverage_report,
    dietary_validity,
    mrr,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k() -> None:
    preds = ["oil", "margarine", "wrong"]
    gold = {"oil", "margarine", "ghee"}
    assert precision_at_k(preds, gold, k=2) == 1.0  # both top-2 correct
    assert precision_at_k(preds, gold, k=3) == 2 / 3


def test_recall_at_k() -> None:
    preds = ["oil", "wrong"]
    gold = {"oil", "margarine", "ghee"}
    assert recall_at_k(preds, gold, k=2) == 1 / 3


def test_mrr_first_hit() -> None:
    assert mrr(["wrong", "oil"], {"oil"}) == 0.5
    assert mrr(["oil"], {"oil"}) == 1.0
    assert mrr(["wrong"], {"oil"}) == 0.0


def test_coverage_report_counts() -> None:
    gold_pairs = [("butter", "oil"), ("buttermilk", "yogurt"), ("rare1", "rare2")]
    vocab = {"butter", "oil", "buttermilk", "yogurt"}
    rep = coverage_report(gold_pairs, vocab)
    assert rep["n_pairs"] == 3
    assert rep["covered_pairs"] == 2  # rare pair drops out
    assert rep["query_coverage"] == 2 / 3  # butter, buttermilk queryable
    assert 0.0 < rep["pair_coverage"] < 1.0


def test_dietary_validity() -> None:
    class _T:
        def is_valid(self, ing, diet):
            return ing != "butter"

    class _S:
        def substitutes(self, ingredient, diet=None, recipe=None, k=None):
            from pantrychef.common.types import Substitute

            picks = {"milk": ["butter", "soy milk"]}.get(ingredient, ["oil"])
            return [Substitute(ingredient=i, score=1.0) for i in picks]

    rate, cov = dietary_validity(_S(), _T(), [("milk", "vegan")], k=2)
    assert 0.0 <= rate <= 1.0


# ── Fix 3: vacuous result when substitutor yields nothing locks contract ─────

def test_dietary_validity_empty_substitutor_returns_vacuous_rate() -> None:
    """When the substitutor returns no subs at all, rate=1.0 and total=0.0."""

    class _NullT:
        def is_valid(self, ing, diet):
            return True

    class _NullS:
        def substitutes(self, ingredient, diet=None, recipe=None, k=None):
            return []

    rate, total = dietary_validity(_NullS(), _NullT(), [("butter", "vegan")], k=5)
    assert rate == 1.0
    assert total == 0.0
