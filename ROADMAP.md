# Roadmap

Status legend: ⬜ not started · 🟡 in progress · ✅ done · 🔁 done, but its public claim
is retracted or under review.

Current state: [docs/PROJECT_STATE.md](docs/PROJECT_STATE.md). Full detail:
[docs/DEVELOPMENT_PLAN.md](docs/DEVELOPMENT_PLAN.md). Direction:
[ADR 0003](docs/decisions/0003-licence-clean-recipe-corpus.md),
[ADR 0004](docs/decisions/0004-static-first-browser-engine.md),
[ADR 0005](docs/decisions/0005-honest-claims-and-eval-reset.md).

## Research foundation (2026-05 to 2026-07)

This work built the Python package. No part of it is usable as a product yet.

| Phase | Result | Status today |
| --- | --- | --- |
| 0 Scaffold | Tooling and CI. M0 removed the unused Hydra and MLflow | ✅ |
| 1 Parser and retrieval | Rule-based parser, F1 0.864 against gold from the same data | 🔁 Metric is circular. No log exists |
| 2 Substitution | Graph MRR 0.289 and embedding 0.259 (n=82) | 🔁 Difference is within noise. Hand check shows mostly name variants |
| 3 Recommendation | LambdaMART recall@10 0.980 and overlap 0.746 at 50k | 🔁 Task is leaked. A no-ML rule scores 0.975 |
| 4 Nutrition and dietary | USDA join, coverage 0.800 on the first rows | 🔁 Matches are wrong for staples. Dietary "100%" cannot fail |

PR #5 (parser fix and training cleanup) merged on 2026-10-30.

## Revival plan (from 2026-10)

| Milestone | Goal | Status |
| --- | --- | --- |
| Audit | Independent audit, research and decisions | ✅ 2026-10-30 |
| M0 Reset | Honest docs, fresh clone works, dead config removed | ✅ 2026-10-30 (PR #6) |
| M1 Thin slice | First hosted page with real search over Wikibooks recipes | ⬜ |
| M2 Evaluation baseline | Realistic protocols, rule baselines, judging protocol and pilot | ⬜ |
| M3 Ingredient foundation | Curated table, resolver, clean vocabulary | ⬜ |
| M4 Search and ranking | IDF ordering and a ranker that must beat it | ⬜ |
| M5 Trust layer | Three-state diet flags, curated and mined swaps | ⬜ |
| M6 Full app | Complete, accessible product | ⬜ |
| M7 Public v1.0 | Reproducible release on GitHub Pages | ⬜ |
| M8 Vision beta | Photo suggestions, only if the test passes | ⬜ Gated |
| M9 Nutrition | Estimates, only if FNDDS validation passes | ⬜ Gated |

## Not planned

Accounts, a hosted LLM, allergen or medical claims, per-serving nutrition without data,
and paid infrastructure.
