"""Learning-curve sweep to pick the production --max-train-queries cap.

    uv run python scripts/recommender_learning_curve.py

Trains the production lambdamart-nosub model at increasing train-query caps
and scores every cap against ONE fixed eval slice (hash-sampled test-split
queries, built once). Capped sampling is seed-hash ordered (see
query_sim.sample_order), so each cap's training queries are a true prefix of
the next cap's. Stops once the MRR@10 gain over the previous cap falls below
--stop-delta and recommends the smallest saturating cap as the production
default.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from time import perf_counter

from pantrychef.common import get_logger
from pantrychef.config import get_settings
from pantrychef.data.store import load_recipes
from pantrychef.eval.recommend_eval import build_eval_queries, score_queries
from pantrychef.recommender.benchmark import BenchmarkRecorder, ConsoleObserver
from pantrychef.recommender.config import RecConfig
from pantrychef.recommender.train import train_production_bundle
from pantrychef.retrieval.index import InvertedIndex

log = get_logger(__name__)

_TABLE_COLS = (
    "cap",
    "attempted",
    "kept",
    "mrr@10",
    "recall@10",
    "mrr@10|in_pool",
    "gen_seconds",
    "fit_seconds",
    "eval_seconds",
)


def _format_table(rows: list[dict]) -> str:
    header = "  ".join(c.ljust(16) for c in _TABLE_COLS)
    lines = [header]
    for r in rows:
        cells = []
        for c in _TABLE_COLS:
            v = r.get(c)
            cells.append((f"{v:.4f}" if isinstance(v, float) else str(v)).ljust(16))
        lines.append("  ".join(cells))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Recommender train-cap learning curve.")
    ap.add_argument("--caps", type=int, nargs="+", default=[10_000, 20_000, 40_000])
    ap.add_argument("--eval-queries", type=int, default=2000)
    ap.add_argument(
        "--stop-delta",
        type=float,
        default=0.002,
        help="stop when mrr@10 gain over the previous cap is below this",
    )
    ap.add_argument("--mask-fraction", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max-rows", type=int, default=None, help="cap corpus size for fast dev")
    ap.add_argument("--output", default=None, help="write per-cap results JSON here")
    args = ap.parse_args(argv)

    s = get_settings()
    recipes_path = s.processed_dir / "recipes.jsonl"
    if not recipes_path.exists():
        log.error("Missing %s. Run `make data` first.", recipes_path)
        return 1

    started = perf_counter()
    recipes = load_recipes(recipes_path, limit=args.max_rows)
    index = InvertedIndex.build(recipes)
    log.info("Loaded %d recipes + index in %.1fs", len(recipes), perf_counter() - started)

    eval_cfg = RecConfig(
        mask_fraction=args.mask_fraction, seed=args.seed, max_eval_queries=args.eval_queries
    )
    started = perf_counter()
    eval_queries = build_eval_queries(recipes, index, eval_cfg)
    log.info(
        "Built %d eval queries in %.1fs (one fixed slice, reused across caps)",
        len(eval_queries),
        perf_counter() - started,
    )
    if not eval_queries:
        log.error("0 eval queries; corpus too small for this config")
        return 1

    rows: list[dict] = []
    metrics: dict[str, float] = {}
    recommended: int | None = None
    for cap in sorted(args.caps):
        cfg = RecConfig(mask_fraction=args.mask_fraction, seed=args.seed, max_train_queries=cap)
        recorder = BenchmarkRecorder(
            git_sha=None,
            dataset_size=len(recipes),
            config={"recommender": dataclasses.asdict(cfg)},
        )
        bundle, stats = train_production_bundle(
            recipes, index, cfg, observer=ConsoleObserver(recorder)
        )
        ranker = bundle.models["lambdamart-nosub"]
        started = perf_counter()
        metrics = score_queries(
            ranker.model, eval_queries, index, None, k=10, columns=ranker.columns
        )
        stage_seconds = {t.key: t.seconds for t in recorder.build_result().stage_timings}
        rows.append(
            {
                "cap": cap,
                "attempted": stats["n_queries_total"],
                "kept": stats["n_queries_kept"],
                "mrr@10": metrics["mrr@10"],
                "recall@10": metrics["recall@10"],
                "mrr@10|in_pool": metrics["mrr@10|in_pool"],
                "gen_seconds": stage_seconds.get("generate_examples"),
                "fit_seconds": stage_seconds.get("train_lambdamart_nosub"),
                "eval_seconds": perf_counter() - started,
            }
        )
        log.info(
            "cap=%d: %s",
            cap,
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()},
        )
        if len(rows) >= 2:
            delta = rows[-1]["mrr@10"] - rows[-2]["mrr@10"]
            if delta < args.stop_delta:
                recommended = rows[-2]["cap"]
                log.info(
                    "Saturated: mrr@10 gain %.4f < %.4f at cap=%d -> recommend cap=%d",
                    delta,
                    args.stop_delta,
                    cap,
                    recommended,
                )
                break
        if stats["n_queries_total"] < cap:
            # recommend the effective count, not the never-reached nominal cap
            recommended = stats["n_queries_total"]
            log.info(
                "Corpus exhausted at %d attempted queries; larger caps change nothing",
                stats["n_queries_total"],
            )
            break

    if recommended is None:
        recommended = rows[-1]["cap"]
        log.info("No saturation within tested caps; curve still rising at cap=%d", recommended)

    print(_format_table(rows))
    print(
        f"ceiling (gold in candidate pool): {metrics['ceiling']:.4f} "
        f"on {len(eval_queries)} eval queries"
    )
    print(f"recommended production --max-train-queries: {recommended}")

    if args.output:
        payload = {
            "dataset_size": len(recipes),
            "eval": {
                "n_queries": len(eval_queries),
                "seed": args.seed,
                "mask_fraction": args.mask_fraction,
                "ceiling": metrics["ceiling"],
            },
            "stop_delta": args.stop_delta,
            "rows": rows,
            "recommended_cap": recommended,
        }
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
