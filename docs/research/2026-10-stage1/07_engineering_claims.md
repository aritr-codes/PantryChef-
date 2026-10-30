# 07: Engineering and honesty of claims

Audited `fix/parser-doubled-token-coverage` at d49f02c. Repo unchanged.

**Verdict.** The code is cleaner than the docs. `docs/EVALUATION.md` is careful and mostly accurate. `README.md`, `ARCHITECTURE.md` and `ROADMAP.md` still describe the original plan as if it were built. An interviewer who greps for FAISS, two-tower, NER or Hydra finds nothing. There is no working product path: `pantrychef cook` does not use the learned ranker, and the "production" bundle on disk was trained on 1,000 recipes.

Good news, verified: 246 tests pass, coverage is 92%, `ruff check` is clean, there are no secrets in git history, nothing large is tracked, and the serving path has no pickle.

## 1. Claims ledger

Status counts: see the summary after the table.

| # | Claim (location) | Evidence | Status |
|---|---|---|---|
| 1 | "by text or photo" (README:3) | `vision/__init__.py` is a docstring only | FALSE |
| 2 | "two-tower" (README:21,35; ARCHITECTURE; `recommender/__init__.py`) | No torch, no code | FALSE |
| 3 | FAISS (ARCHITECTURE:29,50; ROADMAP P3 skills) | Not a dependency, no import | FALSE |
| 4 | "fine-tuned, ONNX-exported, edge-deployable" CV (README:23) | Nothing exists; written in present tense | FALSE |
| 5 | "Ingredient NER" under "Built with ML" (README:33) | `parser.py:1` is rules plus vocab longest-match; the NER column is the dataset's | FALSE |
| 6 | Hydra configs (ARCHITECTURE:13,75; ADR-0001; CONTRIBUTING) | Not in `pyproject.toml`, no import; `configs/model/placeholder.yaml` | FALSE |
| 7 | "Reproducible by default: ... MLflow runs"; "Link MLflow runs" (ARCHITECTURE:13; EVALUATION:6) | 4 runs, params only, 0 metrics, 0 artifacts; only `train_substitution.py` logs; no links | FALSE |
| 8 | Quickstart: `uv sync --extra dev`, `make test` (README) | Fresh clone: 7 collection errors, `No module named 'scipy'` | FALSE |
| 9 | `uv sync`, `uv run pytest` (CONTRIBUTING) | pytest sits in the `dev` extra, so a plain sync omits it | FALSE |
| 10 | "All numbers reflect the 2026-06-05 parser fix" (README footnote) | P1 dated 05-31, P4 06-03; neither rerun | FALSE |
| 11 | `dietary_ok`: "candidate passes the pantry's inferred dietary constraints" (recommender card); "constraint-aware ranking" (ROADMAP P3) | `features.py` returns literal `1.0`; `runtime.recommend` has no diet parameter; no call site | FALSE |
| 12 | "default bundle contains only lambdamart-nosub ... runtime serves it" (EVALUATION:244; card) | Disk bundle: 3 arms, `recipe_count` 1000, 767 groups, commit a74280c, `max_train_queries` null. Code path exists; artifact does not | FALSE |
| 13 | `NutritionFacts` "per serving" (ARCHITECTURE:66; `types.py:52`) | Model card and EVALUATION correctly say no per-serving exists | FALSE |
| 14 | Docker exists as a runnable path (`docker/Dockerfile`) | Replicating its `uv sync --no-dev` with only pyproject and uv.lock fails: `Readme file does not exist`. No API, wrong extras (emulated, no `docker build` run) | FALSE |
| 15 | "Runs identically on local, Colab, Kaggle" (README) | No notebook has ever existed; `<you>` placeholder URL; `settings.py:16` derives paths from `parents[2]`, wrong under pip install; `data/` not in wheel | UNSUPPORTED |
| 16 | "benchmarked against published research (food2vec, GISMo)" (README:20; ADR-0002) | "food2vec" is our own word2vec; GISMo's model was never run, only its gold on a 78.5% covered subset. EVALUATION states this; README does not | UNSUPPORTED |
| 17 | Graph "beats" food2vec (README, cards) | 0.289 vs 0.259 at n=82, no CI, β tuned on same gold; graph loses on R@10 (0.466 vs 0.481). My rough SE is ~0.04, so within noise (HYPOTHESIS) | UNSUPPORTED |
| 18 | "100%" dietary validity as a result (README table) | `substitution_main.py:115` uses the same tagger to filter and to validate: a tautology. n=45 curated. Log predates 9a2779e | UNSUPPORTED |
| 19 | "Learned reranker crushes baseline", "core claim" (EVALUATION:163) | Number is real, but see F1: a one-line rule gets nearly all of the lift | UNSUPPORTED |
| 20 | "saturates at or below 10k" (EVALUATION:272; card) | Only 2 points, none below 10k; `git_sha=None` hard-coded (`recommender_learning_curve.py:105`) | UNSUPPORTED |
| 21 | Production train "~25 minutes" (EVALUATION:276) | No completed run. Measured generation 14.1 min and fit 1.2 min; save and reload never measured at scale | UNSUPPORTED |
| 22 | P1 F1 0.864, recall@10 0.873 | No log anywhere on disk; parser has since changed | UNSUPPORTED |
| 23 | P4 coverage 0.800, 0.487, 0.560; dietary 1.0 | Command documented, but no log on disk | UNSUPPORTED |
| 24 | Mined gold is "reproducible" (EVALUATION:49) | `subs_gold.csv` gitignored by `*.csv`, mined 05-31 pre-fix, command undocumented. Contains name variants (`bisquick mix`/`bisquick baking mix`, `all purpose flour`/`flour`) | UNSUPPORTED |
| 25 | RecipeNLG usable for a public app (DATASET: "check dataset card") | Licence question left open | UNSUPPORTED |
| 26 | ROADMAP P2 MRR 0.339, "Defer full-corpus, published-gold" | README 0.289; both since done | STALE |
| 27 | ROADMAP P3 0.971 / 0.663 / ceiling 0.983 | Superseded: 0.980 / 0.746 / 0.989. Full-corpus row lacks the pre-fix caveat | STALE |
| 28 | `MODEL_CARD_substitution.md` entire | 28k numbers, "pending", substring tags. Last touched d5092fa; PR #5 did not update it | STALE |
| 29 | "Ceiling collapses 0.983 → 0.706"; 7.7× (EVALUATION:217-224; ROADMAP) | The same doc says that prefix is biased-hard; unbiased post-fix ceiling is 0.854. Effects conflated | STALE |
| 30 | GISMo table (EVALUATION:119-121) | `eval_gismo.log` 06-01, pre-fix, not flagged. P2 mined table moved 0.176 → 0.289, so this will move too | STALE |
| 31 | `docs/EXPERIMENTS.md` "experiment log" | "No experiments yet" | STALE |
| 32 | ci.yml comment: "Phase 2+: an eval-gate job" | No such job. ROADMAP P6 lists it as future, so only the comment misleads | STALE |
| 33 | `recommender/__init__.py` "Placeholder ... popularity"; `types.py` "skeletons"; ARCHITECTURE "placeholders only" | Package is implemented; popularity was deliberately excluded | STALE |
| 34 | Makefile `train` "Not implemented until Phase 2"; `eval` is P1 only | Scripts exist for P2 and P3 | STALE |
| 35 | Version | `pyproject` and `__version__` 0.0.0 vs tag v0.4.0; bundle manifests record 0.0.0 | STALE |
| 36 | DATASET: "record N per run", substitution gold source, "GISMo below" | No run records, wrong source, no GISMo section | STALE |
| 37 | P2 full: 0.259 / 0.289 / 0.317, R@10 0.532 | `p2_eval_fixed.log` matches. No SHA; log is 7.5h older than commit d4189a8 (uncommitted code) | TRUE |
| 38 | P3 50k: 0.980 / 0.746, n=9,354, seed 13 | `p3_50k_fixed.log` matches | TRUE |
| 39 | P3 full 0.690 / 0.089, labelled pre-fix, rerun pending | `p3_full.log` matches. `p3_full_fixed.log` is a `MemoryError` traceback with **no results**, so "pending" is honest | TRUE |
| 40 | sub_fill null (0.9349 vs 0.9347) | Log matches | TRUE |
| 41 | Learning curve values 0.8275 / 0.9689 / 0.854 | `learning_curve.json` matches | TRUE |
| 42 | Imputer built, not wired; no per-serving; no macro gold | Code and docs agree. Good honesty | TRUE |
| 43 | No LLM; 246 tests; ruff check | Confirmed | TRUE |

**Totals: 14 FALSE, 11 UNSUPPORTED, 11 STALE, 7 TRUE.** Not traced: the "~1000× sub slowdown" and "23h → 65min" timings.

## 2. Findings

### F1. The "learned" reranker is nearly a one-line rule. High. VERIFIED on a slice.
- **What:** Gold recipes contain every pantry item by construction, so `match_frac_pantry` = 1.0 is almost an answer key. In the bundle, the first split on it has gain 10138; the next are under 17. On the first 20k recipes with 1,500 hash-sampled test queries, seed 13:

| Ranker | recall@10 | MRR@10 |
|---|---|---|
| overlap | 0.861 | 0.544 |
| `match_frac_pantry*1000 + coverage` (untrained, first try) | 0.993 | 0.952 |
| LambdaMART trained on only 767 groups | 0.994 | 0.958 |

- **Why it matters:** This is the likely reason 10k and 20k gave metrics identical to full float precision (HYPOTHESIS: no training was run here). A real pantry is not a subset of one recipe, so this signal should not transfer. An interviewer will find this in minutes. The model card half-admits it ("nearly solve recovery ... by construction") but the headline does not.
- **Where:** `recommender/features.py`, `EVALUATION.md:163`.
- **Fix:** Add the rule as a named baseline in the leaderboard and rewrite the claim. Recommender specialist owns a relevance-framed eval.
- **Worth it?** Yes: an hour, and it turns a liability into your best interview story.

### F2. No "pantry → full answer" entry point exists. High for the web-app goal. VERIFIED.
- **What:**
  - `cli.py:18` `cook` calls the P1 baseline and rebuilds the index per call. `RecommenderRuntime` is referenced only by tests.
  - `recipes.jsonl` has no directions or link. `load_recipes` blanks `ingredients_raw` (`store.py`), and the bundle index shows `[]`, so results cannot show steps or quantities, and nutrition cannot run on them.
  - Pantry uses `canonicalize`, not vocab matching: "2 cups all-purpose flour" becomes `2 cup all purpose flour`, and matching is exact-string against postings.
  - `recipe_id` is sequential at clean time, so URLs break on reclean.
- **Fix:** One `answer(pantry, diet, k)` function plus a read-only SQLite recipe store keyed by a stable hash id. Retain directions in the clean step.
- **Worth it?** Required.

### F3. The production artifact does not exist. High. VERIFIED.
- **What:** See ledger 12. `train_recommender.py:139` reloads the bundle while the original is in memory, and `InvertedIndex.from_state` builds pydantic objects per recipe from JSON. This probably contributes to the RAM block (HYPOTHESIS).
- **Fix:** Drop the reload or free first, and store the index as numpy plus SQLite. Do this before hosting, because the load path decides the memory tier.
- **Worth it?** Yes.

### F4. README and ARCHITECTURE describe a different project. High. VERIFIED.
- **What:** Ledger 1-7.
- **Fix:** Rewrite with Built / Planned columns, and link numbers to EVALUATION only so there is one source of truth.
- **Worth it?** Yes, half a day and the highest return on this list.

### F5. Fresh clone fails. High. VERIFIED by clone in a temp dir.
- **What:** `tests/test_cli_substitute.py` and 6 others import scipy at module level, and `cli`/`eval` chains need it. CI hides this by installing all extras.
- **Fix:** Change the quickstart to `uv sync --all-extras`, or use `[dependency-groups]`. Use `--locked` in CI.
- **Worth it?** Yes, 5 minutes.

### F6. Evidence is not public or pinned. Medium-High. VERIFIED.
- **What:**
  - `*.log`, `models/*` and `*.csv` are gitignored, so results cannot be traced outside this laptop, including the 116-row gold CSV.
  - No result log records a SHA, and `_git_commit` ignores dirty state.
  - No P2 or P3 eval command appears in EVALUATION, DATASET or the cards.
  - `recommend_main.main` now only scores a saved bundle, and `run_leaderboard` is test-only, so the published P3 table has no documented command.
  - Raw CSV has no checksum.
  - Positive: bundles carry corpus fingerprints and seeds are recorded.
- **Fix:**
  - Commit `data/eval/subs_gold.csv`, after a licence check (GISMo's stays out).
  - Add `results/*.json` with SHA, dirty flag, config, fingerprint and metrics, written by one helper.
  - Publish bundles as a GitHub or HF release.
  - Add a `make reproduce-*` target with the exact commands.
- **Worth it?** Yes, it replaces MLflow for ~30 lines.

### F7. Dependency bloat and advisories. Medium. VERIFIED for counts; attribution is HYPOTHESIS.
- **What:** `pip-audit` on the locked runtime extras (96 packages) flags 16 packages, including mlflow, gitpython, pillow, aiohttp, starlette, sqlparse, cryptography, urllib3, werkzeug and direct dep pydantic-settings (CVE-2026-58203, fixed in 2.14.2). Reachability not assessed. `mlflow` lives in the `substitution` extra, so serving would drag in the tracking stack, which only logs 8 params.
- **Fix:** Delete mlflow, then `uv lock --upgrade` and rerun the audit. Add `pip-audit` to CI.
- **Worth it?** Yes.

### F8. Security before deployment. Low-Medium. VERIFIED.
- **Good:** Both bundles use JSON, npy with `allow_pickle=False` and npz. There is no shared mutable state on the runtime path: `CandidatePoolWorkspace` is not thread-safe but runtime never shares one, `neighbors` is pure, and `@cache` is keyed by vocab entries only.
- **Fix before public use:**
  - Cap pantry length, string length and `k`. Candidate cost scales with postings summed, and `k` is unbounded in `substitute_command`.
  - Never expose `--model`-style paths through an API.
  - Delete `EmbeddingModel.load` (`embeddings.py:101`, gensim pickle), which has no production caller, plus `word2vec.kv`.
  - `fetch_gismo_gold.py:48,98` unpickles a download. It is dev-only, so note it.
  - LightGBM threads may oversubscribe under a server; set `n_jobs=1` (HYPOTHESIS).
- **Worth it?** Yes, cheap.

### F9. Format mismatch explained. Low. VERIFIED.
- **What:** Same ruff (0.15.15). `git ls-files --eol` shows `features.py` and `recommend.py` as `i/lf w/mixed`; 123 other files are `w/crlf` via `core.autocrlf=true`. ruff sees mixed endings on Windows only, and CI on LF is clean. The committed blobs are fine. Separately, pre-commit pins ruff v0.4.10 vs locked 0.15.15, a real drift.
- **Fix:** Add `.gitattributes` with `* text=auto eol=lf`, re-checkout both files, and bump the pre-commit rev.
- **Worth it?** Yes, 5 minutes.

### F10. Tests. Medium. VERIFIED.
- **Coverage (92%):** gaps are `eval/__main__` 0%, `nutrition_main` 42%. `scripts/` is excluded from measurement.
- **Quality:** Solid unit tests, not tautological on the whole; mocks are limited to CLI and main wiring (7 or fewer hits per file). `test_smoke.py` and `test_relocation.py` add little.
- **Gaps:** No integration test goes pantry → ranked → substitutes → nutrition, because that pipeline does not exist. No metric-regression gate. Nothing tests that bundle `columns` match runtime features across versions, beyond the persistence tests.
- **CI:** One job, no Windows although development is on Windows. `dietary_validity` tests cannot catch a bad tagger by design.
- **Fix:** One end-to-end test on a tiny committed fixture once F2 lands, and a small-fixture metric floor as the eval gate.
- **Worth it?** Yes for those two; a Windows job is optional.

### F11. Dead code and duplication. Low-Medium. VERIFIED.
- **Unused in production:** `rrf`, `SvdContextModel` (tests only), `run_leaderboard` and `build_examples`, `impute.py`.
- **Shim:** `substitution/dietary.py`, 2 production imports.
- **Duplicates:** `corpus_fingerprint` and `_read_json`/`_write_json` in both bundles; `NO_SUB_COLUMNS` in `train.py` and `recommend_main.py`; `_format_table` twice; `train_bundle` vs `train_production_bundle`.
- **Over-abstraction:** `benchmark.py` (206 lines of recorder and observer for stage timings); `train.py:60-90` progress reporting.
- **Note:** Import boundaries are otherwise clean, and `recommender → substitution` is correctly lazy via `sub_lookup`.

### F12. Repo hygiene. Low. VERIFIED.
- **What:** Nothing junk is tracked. Root noise is local only: empty `.agents`, `.codex`, `tmp1hp3_sy0`, `.pytest-tmp`; ignored `tmp_recommender_*.log`; untracked `temp_pytest/`, `TestRun.md`, `scratch_codex_audit_prompt.md`, `CODEX_HANDOFF.md`. `recipes.buggy.jsonl` (625 MB) and `buggy_backup/` are disposable. `uv lock --check` passes.
- **Fix:** Delete locally and ignore `.codegraph/` and `temp_pytest/`.
- **Worth it?** 2 minutes.

### F13. PR #5. VERIFIED except where marked.
- **State:** Open, MERGEABLE, CLEAN, CI green at d49f02c (22 commits). CI logs have expired, so the CI test count is unconfirmed; the local run is 246 passed.
- **Reviewed, no bugs found:**
  - Parser multiset logic.
  - Hash sampling (`sample_order`, nesting).
  - `limit<=0`.
  - Conservative `is_known` (9a2779e): correct, but it invalidates the logged `tag-coverage=1.000`, which was an artifact of `ing in self._known`.
  - `candidate_pool` is covered by persistence and recommend tests; byte-identity is claimed but I did not rerun it.
- **Risk:** Mostly documentation, because the PR adds "what ships" prose for an artifact that does not exist (ledger 12).
- **Recommendation:** Merge, then a follow-up docs PR. Do not rewrite the branch.

## 3. Ponytail cut: justified

Delete outright:
- `configs/`, `notebooks/`, `docker/` (rewrite with serving), `vision/` and `serving/` stubs.
- mlflow and its `experiments/` setup, `EXPERIMENTS.md`.
- `docs/superpowers/` (12 AI planning files, ~300 KB; history keeps them; a reader wants ADRs, not plans).
- `rrf`, `SvdContextModel`, `EmbeddingModel.save/load`, `substitution/dietary.py` and `test_relocation.py`.
- `impute.py` and the `nutrition` extra, unless wired soon.
- `benchmark.py`, replaced by timed log lines.
- `dietary_ok`: it is a constant.

Cut after the full P3 rerun and a tag: `--experimental-subs`, `sub_lookup.py`, `sub_fill_*`, `LinearRanker`, `train_bundle`. The null result stays in docs, the code in the tag.

Open question, not mine to decide: if F1's rule holds up on a relevance-framed eval, lightgbm and most of `train.py` could go too. Do not cut this until that eval exists.

Keep: bundles, tests, EVALUATION.md, CHALLENGES.md, the two ADRs (mark Hydra superseded).

## 4. Ranked list

1. **F4/ledger 1-7**: rewrite README and ARCHITECTURE as Built / Planned. Half a day.
2. **F1**: add rule baseline; restate the P3 claim. An hour.
3. **F2**: single `answer()` plus a recipe store with directions. Blocks the app.
4. **F3**: compact index and a real production bundle. Blocks hosting.
5. **F5**: fix fresh-clone quickstart. 5 minutes.
6. **F6**: results JSON with SHA, commit gold, publish bundles, documented repro commands.
7. **Stale docs**: substitution card, ROADMAP numbers, README footnote, flag GISMo as pre-fix. Rerun P2 GISMo and P1.
8. **F7**: drop mlflow, upgrade lock, audit in CI.
9. **Licence**: settle RecipeNLG before going public (ledger 25).
10. **Ponytail cut** (section 3).
11. **F8**: input caps at the API boundary.
12. **F10**: end-to-end test and small metric gate.
13. **F9, F12, version bump**: 15 minutes in total. Merge PR #5 first.
