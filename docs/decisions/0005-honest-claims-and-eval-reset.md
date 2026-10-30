# 0005. Honest claims and an evaluation reset

- **Status:** Accepted
- **Date:** 2026-10-30
- **Deciders:** Project owner, Tech Lead

## Context

The October 2026 audit (`docs/research/2026-10-stage1/`) found that:

- The recovery task is leaked by construction. A no-ML rule scores close to
  LambdaMART.
- Graph versus embedding at n=82 has a 95% interval that includes zero.
- "100% dietary validity" cannot fail.
- Nutrition matches and masses are wrong for staple ingredients.

## Options considered

1. **Keep the headlines and add caveats.** Little work, but an interviewer can break
   them quickly.
2. **Retract the headlines and rebuild evaluation (chosen).**

## Decision

- Retire "graph beats food2vec" as a headline. Graph and embedding stay as research
  arms until a confidence interval excludes zero on a clean test set.
- Keep masked recovery only as a sanity test, with the no-ML rule on its leaderboard.
- A learned ranker ships only if it beats the IDF rule on hand-judged pantries. If it
  loses, the IDF rule ships and the result is reported.
- Keep nutrition out of the first release. It enters only after validation against
  USDA FNDDS recipes passes the target set in `docs/DEVELOPMENT_PLAN.md`.
- The dietary filter is a preference filter with three states. It makes no allergen
  or medical claim.
- Every published number links to a results file with commit, dirty flag, config,
  corpus fingerprint and metrics.

## Consequences

- Positive: all public claims can survive a code review. The falsified metric becomes
  a strong engineering story.
- Negative: some impressive numbers disappear. The ML depth must come from new work.
- Revisit when: a new evaluation gives evidence for a stronger claim.
