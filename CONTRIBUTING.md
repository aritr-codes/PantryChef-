# Contributing / Development Guide

This is a personal portfolio project, but it follows real engineering
discipline.

## Environment (local)

We use [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras    # create .venv + install locked deps (tests need all extras)
uv run pytest           # run tests
uv run ruff check .     # lint
uv run ruff format .    # format
```

Or via the Makefile: `make setup`, `make test`, `make lint`, `make format`.

## Kaggle

Use Kaggle for full-corpus research, large training and vision tests. Core logic stays
in `pantrychef/`. A kernel is a thin script at a pinned commit. Never put a token in a
kernel, a notebook or a log. See [CLAUDE.md](CLAUDE.md) for the rules.

## Repo conventions

- **Package code → `pantrychef/`.** Scripts in `scripts/` are thin entry points.
- **Runtime config → `pantrychef/config/`.** Model hyperparameters are dataclasses next
  to each model.
- **Data is gitignored.** Track provenance in [docs/DATASET.md](docs/DATASET.md).
- **Models/artifacts gitignored.** Describe them in model cards.
- **Line endings are LF** ([.gitattributes](.gitattributes)).

## Decision discipline

- Non-trivial choice? Write an ADR in [docs/decisions/](docs/decisions/)
  (copy `0000-template.md`).
- Ran an experiment? Log it in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).
- Hit a wall? Record it in [docs/CHALLENGES.md](docs/CHALLENGES.md).
- Finished a milestone? Add to [docs/LEARNINGS.md](docs/LEARNINGS.md), tag a release.

## Commit conventions

Conventional Commits: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`,
`test:`, `exp:` (experiment).

## Pre-commit

```bash
uv run pre-commit install
```

Runs `ruff` lint + format on staged files.
