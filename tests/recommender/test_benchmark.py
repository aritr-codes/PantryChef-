import numpy as np

from pantrychef.common.types import Recipe
from pantrychef.recommender.benchmark import BenchmarkRecorder, ExampleProgress, render_summary
from pantrychef.recommender.config import RecConfig
from pantrychef.recommender.train import train_bundle
from pantrychef.retrieval.index import InvertedIndex


class _Observer:
    def __init__(self) -> None:
        self.stages: list[tuple[str, str, float]] = []
        self.progress: list[ExampleProgress] = []

    def record_stage(self, key: str, label: str, seconds: float) -> None:
        self.stages.append((key, label, seconds))

    def record_examples_progress(self, progress: ExampleProgress) -> None:
        self.progress.append(progress)


def _corpus() -> list[Recipe]:
    return [
        Recipe(
            recipe_id=str(i),
            title=str(i),
            canonical=["egg", "flour", "milk", "sugar", "butter"][: 4 + (i % 2)],
        )
        for i in range(20)
    ]


def test_benchmark_result_schema_and_summary() -> None:
    recorder = BenchmarkRecorder(
        git_sha="abc123",
        dataset_size=42,
        config={"recommender": {"mask_fraction": 0.3}},
        timestamp="2026-06-28T00:00:00+00:00",
    )
    recorder.record_stage("load_recipes", "Load recipes", 1.25)
    recorder.record_examples_progress(
        ExampleProgress(
            recipes_processed=42,
            recipes_total=42,
            queries_total=100,
            queries_kept=95,
            candidates_total=1_000,
            elapsed_seconds=2.0,
        )
    )
    result = recorder.build_result()
    payload = result.to_dict()
    assert payload["git_sha"] == "abc123"
    assert payload["dataset_size"] == 42
    assert payload["config"]["recommender"]["mask_fraction"] == 0.3
    assert payload["stage_timings"][0]["label"] == "Load recipes"
    assert payload["throughput"]["total_queries"] == 100
    summary = render_summary(result)
    assert "Load recipes" in summary
    assert "Queries/sec" in summary
    assert "Candidates/sec" in summary


def test_train_bundle_observer_preserves_behavior() -> None:
    corpus = _corpus()
    idx = InvertedIndex.build(corpus)
    cfg = RecConfig(seed=1)

    baseline_bundle, baseline_stats = train_bundle(
        corpus,
        idx,
        sub_lookup=None,
        cfg=cfg,
        use_lambdamart=False,
    )
    observer = _Observer()
    observed_bundle, observed_stats = train_bundle(
        corpus,
        idx,
        sub_lookup=None,
        cfg=cfg,
        use_lambdamart=False,
        observer=observer,
    )

    assert observed_stats == baseline_stats
    assert [stage[0] for stage in observer.stages] == [
        "generate_examples",
        "build_matrix_full",
        "train_linear",
    ]
    assert observer.progress
    assert observer.progress[-1].queries_total == observed_stats["n_queries_total"]
    assert observer.progress[-1].queries_kept == observed_stats["n_queries_kept"]

    baseline_model = baseline_bundle.models["linear"].model
    observed_model = observed_bundle.models["linear"].model
    assert np.allclose(observed_model.w_, baseline_model.w_)
    assert np.isclose(observed_model.b_, baseline_model.b_)
    assert np.allclose(observed_model.mean_, baseline_model.mean_)
    assert np.allclose(observed_model.std_, baseline_model.std_)
