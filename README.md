# 🥘 PantryChef

> A no-LLM, multimodal cooking assistant: tell it what's in your kitchen — by
> text or photo — and it finds makeable recipes, suggests **valid** ingredient
> substitutions, and breaks down nutrition. The hard ML is built, not called.

[![Phase](https://img.shields.io/badge/phase--1%20foundation-blue)](ROADMAP.md)
[![Python](https://img.shields.io/badge/python-3.11-blue)](.python-version)
[![Lint](https://img.shields.io/badge/lint-ruff-261230)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Why this exists

Most "AI recipe app" portfolios are a database query with an LLM bolted on.
PantryChef deliberately builds the parts that are genuinely ML:

- **Ingredient substitution** as a learned representation + constrained graph
  (flagship) — benchmarked against published research (`food2vec`, GISMo).
- **Recipe recommendation** as learning-to-rank + two-tower retrieval.
- **Ingredient detection** as a fine-tuned, ONNX-exported, edge-deployable
  computer-vision model.

…and deliberately keeps nutrition + dietary logic deterministic, because not
everything should be a model.

## What's the moat?

| Built with ML                              | Deterministic on purpose          |
| ------------------------------------------ | --------------------------------- |
| Ingredient NER                             | Quantity/unit parsing             |
| **Substitution: embeddings + graph**       | Dietary classification (ontology) |
| Recommendation: LTR + two-tower            | Nutrition aggregation (USDA)      |
| CV ingredient detection                    | Constraint filtering              |

## Status

**Phase 1 complete ✅** — data foundation, hybrid ingredient parser, and a
set-overlap retrieval baseline, validated on real RecipeNLG data. See
[ROADMAP.md](ROADMAP.md). Next: Phase 2 (flagship ingredient substitution).

## Results

_See [docs/EVALUATION.md](docs/EVALUATION.md) for protocol + provenance._

| Phase | Task | Metric | PantryChef |
| ----- | ---- | ------ | ---------- |
| 1 ✅ | Ingredient parser | canonical-match F1 | **0.864** (target ≥0.85) |
| 1 ✅ | Retrieval baseline | recall@10 | **0.873** |
| 2 | Substitution | precision@5 | _tbd_ |
| 3 | Recommendation | NDCG@10 | _tbd_ |
| 5 | Detection | mAP@0.5 | _tbd_ |

<sub>Phase 1 numbers: 50k-recipe RecipeNLG sample, min_count=5, seed=42.</sub>

## Quickstart

```bash
# 1. Install uv  (https://docs.astral.sh/uv/)
# 2. Sync the environment (creates .venv, installs locked deps)
uv sync --extra dev

# 3. Run checks
make lint
make test
```

Runs identically on local, Google Colab, and Kaggle — see
[CONTRIBUTING.md](CONTRIBUTING.md) for the notebook-thin convention.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for module boundaries and data flow.

## Documentation

| Doc | Purpose |
| --- | --- |
| [ROADMAP.md](ROADMAP.md) | Phase plan + status |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System map, module boundaries |
| [docs/decisions/](docs/decisions/) | ADRs — tradeoffs captured at decision time |
| [docs/DATASET.md](docs/DATASET.md) | Data sources, licenses, provenance |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Metrics + leaderboard |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Experiment log |
| [docs/CHALLENGES.md](docs/CHALLENGES.md) | Real obstacles + how solved |
| [docs/LEARNINGS.md](docs/LEARNINGS.md) | What was learned per phase |

## License

MIT — see [LICENSE](LICENSE).
