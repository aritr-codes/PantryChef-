# 0002. No-LLM core: embeddings + graph for substitution

- **Status:** Accepted
- **Date:** 2026-05-30
- **Deciders:** Tech Lead, ML Eng

## Context

The flagship capability is ingredient substitution. The easy path is to prompt
an LLM — but that produces a CRUD-app-with-AI-bolted-on portfolio with no
demonstrable ML engineering, and depends on a paid/rate-limited API. We want to
*build* the ML and benchmark it against published work.

## Options considered

1. **LLM prompting** (GPT/Llama) — fast to demo, zero ML-engineering signal,
   API dependency, hard to evaluate rigorously.
2. **Ingredient embeddings** (food2vec-style: skip-gram on recipe ingredient
   co-occurrence) → kNN substitutions. Trainable on CPU/free GPU, citable
   baseline, real eval.
3. **Relation graph** (GISMo-style: ingredient nodes + substitutable edges +
   dietary attributes) → constrained traversal.
4. **Hybrid** embeddings + graph with role/dietary filtering.

## Decision

**No LLM for core features.** Build substitution as **embeddings + graph
(hybrid)** with role/dietary filtering, benchmarked against `food2vec` and
GISMo using a published substitution test set (precision@k, MRR) plus a
dietary-validity guardrail.

## Consequences

- Positive: genuine ML-engineering signal; reproducible on free hardware;
  defensible vs published baselines; no API cost.
- Negative / accepted: more work than prompting; eval ground truth is sparse
  and partly subjective.
- Revisit when: a small *local* model could improve natural-language query
  parsing (an optional, non-core enhancement) — still no hosted LLM.
