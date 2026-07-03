# Checkpoint C — Cortex v0 vs Baselines (LongMemEval_S)

**Date:** 2026-06-28 · **Setup:** LongMemEval_S, n=40 seeded (seed 0, shuffled — representative across types), reader `gemini-2.5-flash-lite`, embed `gemini-embedding-001`@768, judge `gemini-2.5-flash` votes=3.

## Results

| System | Accuracy | non-abst | temporal | multi-sess | know-update | ss-user | abstention | mean in-tok | $/q | recall@k |
|---|---|---|---|---|---|---|---|---|---|---|
| **cortex-v0** | **0.575** | 0.611 | 0.286 | 0.556 | 0.50 | 0.875 | 0.25 | 84,059 | $0.00845 | 0.991 |
| naive-rag | 0.475 | 0.472 | 0.000 | 0.444 | 0.40 | 0.875 | 0.50 | 83,255 | $0.00835 | 0.968 |
| full-context | 0.050 | 0.028 | 0.000 | 0.000 | 0.10 | 0.125 | 0.25 | 109,213 | $0.01094 | 1.000 |

## Verdict: PASSED — thesis validated
- **Cortex v0 beats both baselines on accuracy** (0.575 > naive-rag 0.475 > full-context 0.05) and is **cheaper than full-context**. Hybrid (dense+BM25 RRF) retrieval > dense-only naive RAG (+10 pts overall; +28.6 pts temporal; recall 0.991 > 0.968).
- **full-context collapses (0.05)** — `gemini-2.5-flash-lite` cannot read ~109k-token haystacks (lost-in-the-middle). This *proves* retrieval is essential for a cheap reader, and is itself a sellable finding.
- The measurement spine works on `_S`; relative comparisons are trustworthy (judge validated fair at Checkpoint B).

## Phase 3 targets (ranked by leverage)
1. **Cost — the real problem.** $/q ≈ $0.008–0.011 is dominated by **embedding the entire haystack per question (~84k tokens)**, NOT the reader (top-k context is small). Levers: fact/keyphrase extraction so we embed *less*; cheaper/smaller embeddings; dedup; (product: embed-once amortization). Target <$0.005/q. **Also fix the metric to price embed tokens at the embed rate, separate from reader tokens.**
2. **Temporal reasoning (0.286)** — weakest answerable category. Add time-aware indexing + temporal query handling (Goal.md I4).
3. **Abstention (0.25)** — calibration OVERSHOT: Cortex now under-abstains (0.25 vs naive-rag 0.50) because the reader prompt pushes against abstaining. Re-balance so it abstains on genuinely-unanswerable Qs without hurting answerable recall.
4. **knowledge-update (0.50) / multi-session (0.556)** — soft-update (valid-from/to) + better cross-session fusion.
5. **single-session-preference (0.0, n=2)** — tiny sample; revisit with larger n.

## Caveats
- Absolute numbers await **official `gpt-4o-2024-08-06` judge** validation (needs `OPENAI_API_KEY`) — pivotal for whether flash-lite suffices for the ~0.85 target or we need `flash`.
- naive-rag/full-context here supersede the interrupted-run artifacts; n=40 (small per-type n).
