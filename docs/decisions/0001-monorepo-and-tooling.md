# 0001. Monorepo, uv/ruff/pytest, Hydra + MLflow

- **Status:** Accepted
- **Date:** 2026-05-30
- **Deciders:** Tech Lead, ML Eng

## Context

PantryChef is a 6-phase project where each phase should stand alone as a GitHub
artifact, built on free hardware (local MX150, free Colab/Kaggle). We need a
structure + toolchain that is reproducible across all three environments and
signals serious engineering, without paid services.

## Options considered

1. **Monorepo + tagged releases** — one history telling the growth story; each
   phase a release. Slight risk of a large repo.
2. **Separate repo per phase** — truly independent, but duplicated tooling/data
   code, fragmented history, harder to read as one system.
3. **Monorepo, split later** — start mono, extract a phase only if it becomes a
   strong standalone showcase.

Tooling axis: `uv` vs `poetry` vs `pip`; `ruff` vs `black`+`flake8`;
Hydra+MLflow vs pydantic+W&B vs plain YAML.

## Decision

**Monorepo + tagged releases.** Toolchain: **`uv`** (fast, lockfile-based
reproducibility, works on Colab/Kaggle), **`ruff`** (lint+format in one),
**`pydantic-settings`** (typed runtime config), **`pytest`**. Experiments:
**Hydra** (composable YAML configs) + **MLflow** (self-hosted, free, no
account). We keep the "split later" door open (option 3 as fallback).

## Consequences

- Positive: single coherent story; one toolchain everywhere; modern signal;
  reproducible via `uv.lock`.
- Negative / accepted: monorepo grows; must enforce thin-notebook + module
  boundaries to avoid a tangle.
- Revisit when: a single phase clearly deserves its own repo for visibility.
