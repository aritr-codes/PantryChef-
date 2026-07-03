"""Training-run observability helpers for recommender benchmarking."""

from __future__ import annotations

import dataclasses
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from time import perf_counter, process_time
from typing import Protocol

from pantrychef.common import get_logger

log = get_logger(__name__)


@dataclasses.dataclass(frozen=True)
class ExampleProgress:
    recipes_processed: int
    recipes_total: int
    queries_total: int
    queries_kept: int
    candidates_total: int
    elapsed_seconds: float


class TrainObserver(Protocol):
    def record_stage(self, key: str, label: str, seconds: float) -> None: ...

    def record_examples_progress(self, progress: ExampleProgress) -> None: ...


@dataclasses.dataclass(frozen=True)
class StageTiming:
    key: str
    label: str
    seconds: float

    def to_dict(self) -> dict[str, object]:
        return {"key": self.key, "label": self.label, "seconds": self.seconds}


@dataclasses.dataclass(frozen=True)
class ThroughputStats:
    recipes_per_sec: float | None
    queries_per_sec: float | None
    candidates_per_sec: float | None
    total_queries: int | None
    total_candidates: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "recipes_per_sec": self.recipes_per_sec,
            "queries_per_sec": self.queries_per_sec,
            "candidates_per_sec": self.candidates_per_sec,
            "total_queries": self.total_queries,
            "total_candidates": self.total_candidates,
        }


@dataclasses.dataclass(frozen=True)
class BenchmarkResult:
    git_sha: str | None
    timestamp: str
    dataset_size: int
    config: dict[str, object]
    stage_timings: tuple[StageTiming, ...]
    total_seconds: float
    total_cpu_seconds: float
    throughput: ThroughputStats

    def to_dict(self) -> dict[str, object]:
        return {
            "git_sha": self.git_sha,
            "timestamp": self.timestamp,
            "dataset_size": self.dataset_size,
            "config": dict(self.config),
            "stage_timings": [stage.to_dict() for stage in self.stage_timings],
            "total_seconds": self.total_seconds,
            "total_cpu_seconds": self.total_cpu_seconds,
            "throughput": self.throughput.to_dict(),
        }


class BenchmarkRecorder:
    """Collect stable benchmark metadata without affecting training behavior."""

    def __init__(
        self,
        *,
        git_sha: str | None,
        dataset_size: int,
        config: Mapping[str, object],
        timestamp: str | None = None,
    ) -> None:
        self.git_sha = git_sha
        self.dataset_size = dataset_size
        self.config = dict(config)
        self.timestamp = timestamp or datetime.now(UTC).isoformat()
        self._wall_start = perf_counter()
        self._cpu_start = process_time()
        self._stage_timings: OrderedDict[str, StageTiming] = OrderedDict()
        self._latest_progress: ExampleProgress | None = None

    def record_stage(self, key: str, label: str, seconds: float) -> None:
        self._stage_timings[key] = StageTiming(key=key, label=label, seconds=seconds)

    def record_examples_progress(self, progress: ExampleProgress) -> None:
        self._latest_progress = progress

    @contextmanager
    def stage(self, key: str, label: str):
        start = perf_counter()
        try:
            yield
        finally:
            self.record_stage(key, label, perf_counter() - start)

    def build_result(self) -> BenchmarkResult:
        total_seconds = perf_counter() - self._wall_start
        total_cpu_seconds = process_time() - self._cpu_start
        recipes_per_sec = None
        if total_seconds > 0 and self.dataset_size > 0:
            recipes_per_sec = self.dataset_size / total_seconds
        progress = self._latest_progress
        queries_per_sec = None
        candidates_per_sec = None
        total_queries = None
        total_candidates = None
        if progress is not None and progress.elapsed_seconds > 0:
            total_queries = progress.queries_total
            total_candidates = progress.candidates_total
            queries_per_sec = progress.queries_total / progress.elapsed_seconds
            candidates_per_sec = progress.candidates_total / progress.elapsed_seconds
        return BenchmarkResult(
            git_sha=self.git_sha,
            timestamp=self.timestamp,
            dataset_size=self.dataset_size,
            config=dict(self.config),
            stage_timings=tuple(self._stage_timings.values()),
            total_seconds=total_seconds,
            total_cpu_seconds=total_cpu_seconds,
            throughput=ThroughputStats(
                recipes_per_sec=recipes_per_sec,
                queries_per_sec=queries_per_sec,
                candidates_per_sec=candidates_per_sec,
                total_queries=total_queries,
                total_candidates=total_candidates,
            ),
        )


class ConsoleObserver:
    """TrainObserver that records into a BenchmarkRecorder and logs progress."""

    def __init__(self, recorder: BenchmarkRecorder) -> None:
        self.recorder = recorder
        self._last_logged_progress: tuple[int, int] | None = None

    def record_stage(self, key: str, label: str, seconds: float) -> None:
        self.recorder.record_stage(key, label, seconds)

    def record_examples_progress(self, progress: ExampleProgress) -> None:
        self.recorder.record_examples_progress(progress)
        current = (progress.recipes_processed, progress.queries_total)
        if current == self._last_logged_progress:
            return
        self._last_logged_progress = current
        pct = 0.0
        if progress.recipes_total > 0:
            pct = progress.recipes_processed / progress.recipes_total * 100.0
        msg = (
            f"Generate examples: {progress.recipes_processed}/{progress.recipes_total} "
            f"recipes ({pct:.1f}%), {progress.queries_total} queries, "
            f"{progress.queries_kept} kept"
        )
        if progress.elapsed_seconds > 0:
            msg += f", {progress.queries_total / progress.elapsed_seconds:.1f} queries/s"
            msg += f", {progress.candidates_total / progress.elapsed_seconds:.1f} candidates/s"
        log.info(msg)


def render_summary(result: BenchmarkResult) -> str:
    total = result.total_seconds
    stage_width = max([len("Stage"), *(len(stage.label) for stage in result.stage_timings)])
    lines = [f"{'Stage':<{stage_width}}  {'Time':>10}  {'%':>6}"]
    lines.append(f"{'-' * stage_width}  {'-' * 10}  {'-' * 6}")
    for stage in result.stage_timings:
        pct = (stage.seconds / total * 100.0) if total > 0 else 0.0
        lines.append(f"{stage.label:<{stage_width}}  {stage.seconds:>10.2f} s  {pct:>5.1f}%")
    lines.append(f"{'-' * stage_width}  {'-' * 10}  {'-' * 6}")
    lines.append(f"{'Total':<{stage_width}}  {total:>10.2f} s  {100.0:>5.1f}%")
    lines.append("")
    lines.append(f"Dataset size: {result.dataset_size}")
    if result.throughput.recipes_per_sec is not None:
        lines.append(f"Recipes/sec: {result.throughput.recipes_per_sec:.1f}")
    if result.throughput.total_queries is not None:
        lines.append(f"Queries: {result.throughput.total_queries}")
    if result.throughput.queries_per_sec is not None:
        lines.append(f"Queries/sec: {result.throughput.queries_per_sec:.1f}")
    if result.throughput.total_candidates is not None:
        lines.append(f"Candidates: {result.throughput.total_candidates}")
    if result.throughput.candidates_per_sec is not None:
        lines.append(f"Candidates/sec: {result.throughput.candidates_per_sec:.1f}")
    return "\n".join(lines)
