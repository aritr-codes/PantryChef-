# 0004. Static-first app with a browser engine

- **Status:** Accepted
- **Date:** 2026-10-30
- **Deciders:** Project owner, Tech Lead

## Context

The frontend must run on GitHub Pages. GitHub Pages cannot run Python. Free
always-on Python hosting is now rare: Hugging Face Docker and Gradio Spaces need a
paid plan to create, and Render free sleeps and takes about 1 minute to wake. The
current Python runtime peaks at 2.8 GB on the full corpus. ADR 0003 reduces the
display corpus to about 3,800 recipes.

## Options considered

1. **Static frontend and a Python API.** Keeps one implementation, but needs a host,
   adds cold starts and an abuse surface, and stops when free tiers change.
2. **Python in the browser with Pyodide.** Reuses the code, but needs tens of MB and
   seconds of start time on a phone.
3. **Fully static with a small TypeScript engine (chosen).** Precompute artifacts in
   Python. Run search and scoring in the browser.
4. **Hybrid.** Option 3 plus an optional API, only when a feature proves the need.

## Decision

Use option 3. The app calls one module, `web/src/engine.ts`, so option 4 needs only a
replacement for that module. We add no formal interface until a second implementation
exists. Python remains the source of truth for pipelines and for the reference
implementation. A golden-file test proves that the TypeScript engine and the Python
reference give the same results.

## Consequences

- Positive: no server, no secrets, no sleep, no cost, and pantry data stays on the
  device. A vision model can run on the device too.
- Negative: two implementations of search and scoring, held together by parity tests.
  Free-text parsing is limited to what fits in the browser, so the interface uses
  autocomplete, synonyms and fuzzy matching.
- Revisit when: a feature cannot run in a browser, or measured latency on a low-end
  phone fails the target.
