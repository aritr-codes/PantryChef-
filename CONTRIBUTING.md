# Contributing / Development Guide

This is a personal portfolio project, but it follows real engineering
discipline. Same workflow whether you're on local, Colab, or Kaggle.

## Environment (local)

We use [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                 # create .venv + install locked deps
uv run pytest           # run tests
uv run ruff check .     # lint
uv run ruff format .    # format
```

Or via the Makefile: `make setup`, `make test`, `make lint`, `make format`.

## Environment (Colab / Kaggle)

Notebooks are **thin**: they install this package and call it. No business
logic in cells.

```python
# top cell of any Colab/Kaggle notebook
!pip install -q "git+https://github.com/<you>/pantrychef.git@main"
from pantrychef.ingredients import parse   # example
```

Heavy GPU work (training) runs in the cloud notebook; the resulting artifacts
(weights, ONNX) are pulled down for local inference on the MX150.

## Repo conventions

- **Package code → `pantrychef/`.** Notebooks → `notebooks/` (thin only).
- **Experiment configs → `configs/`** (Hydra). Runtime config →
  `pantrychef/config/`.
- **Data is gitignored.** Track provenance in [docs/DATASET.md](docs/DATASET.md).
- **Models/artifacts gitignored.** Track in MLflow + model cards.

## Decision discipline

- Non-trivial choice? Write an ADR in [docs/decisions/](docs/decisions/)
  (copy `0000-template.md`).
- Ran an experiment? Log it in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) +
  MLflow.
- Hit a wall? Record it in [docs/CHALLENGES.md](docs/CHALLENGES.md).
- Finished a phase? Add to [docs/LEARNINGS.md](docs/LEARNINGS.md), tag a release.

## Commit conventions

Conventional Commits: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`,
`test:`, `exp:` (experiment).

## Pre-commit

```bash
uv run pre-commit install
```

Runs `ruff` lint + format on staged files.
