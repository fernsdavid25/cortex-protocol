# Vertex cross-encoder reranker — validated (2026-07-03)

The L2 listwise reranker was killed (a cheap Gemini listwise model demoted the weak multi-hop bridge
chunks and crushed recall to 0.78). The **real lever** — a cross-encoder — needed the Vertex AI
Ranking API (Discovery Engine `semantic-ranker`), now enabled. This is that reranker, validated.

**Config:** over-retrieve `--top-k 100` → Vertex `semantic-ranker-default-004` ranks the pool →
reader sees the top `--rerank-k 25`. `--rerank --rerank-backend vertex-ranking`, `--tier cheap`
readers, gemini-3.5-flash judge votes=1, seed 0, n=100 same-question A/B vs the cached k=100
(reader-sees-100) baseline.

| Benchmark | baseline (reader sees 100) | vertex-rerank (reader sees 25) | Δ acc | reader $/q |
|---|---|---|---|---|
| **LoCoMo** (n=100) | 0.770 | **0.780** | **+1.0pt** | 0.0034 → **0.0011** (~3×) |
| **LongMemEval_S** (n=100) | 0.990 | **0.990** | 0.0 (ceiling) | 0.0153 → **0.0044** (~3.5×) |

**LoCoMo per-category** (vs baseline): single-hop 0.784→0.824 (**+3.9**), multi-hop 0.615→0.615
(**held** — vs listwise −20pt), temporal 0.917→0.917, adversarial 0.857→0.810 (**−4.8**, the lone
regression), **recall@k 0.998→0.975 held** (vs listwise 0.78). **LME per-category:** every bucket
identical — no regression anywhere at the 0.99 ceiling.

## Reading the result
A real cross-encoder does what the cheap listwise couldn't: it **keeps the answer-bearing chunks in
the top-25** (recall 0.975, not 0.78), so multi-hop is preserved, while the added precision (fewer
noise chunks) lifts single-hop. Feeding the reader **25 clean chunks instead of 100 makes it ~3×
cheaper** on both benchmarks. The Vertex ranking call itself adds only ~$0.0001/q (100 query-doc
pairs at ~$1/1k), so the net is still ~3× cheaper. On LME (subset at a 0.99 ceiling) accuracy can't
climb but holds perfectly at 3.5× lower cost.

Net: **equal-or-better accuracy at ~3× lower cost on both benchmarks** — a clear accuracy-per-dollar
win, and the LoCoMo +1pt clears the L2 gate. Caveats: (1) the +1pt on n=100 is a 1-question margin —
the robust wins are the cost reduction + recall preservation; full-1986 / full-500 confirmation is
part of the L9 flagship campaign. (2) It requires GCP Discovery Engine creds (a service account), so
it is a hosted/flagship feature — pure-BYOK self-host falls back gracefully to no rerank. (3) The
adversarial −4.8pt is the fewer-context tradeoff (the A2 gate that would recover it was itself killed).

## Decision
Stacked into the **flagship** profile (`use_rerank=True`, `rerank_backend="vertex-ranking"`) — pro
reader + cross-encoder, now at reduced context cost. Open question for the hosted **cheap** tier:
adopting it there would improve the LoCoMo headline (~0.82) AND cut $/q ~3×, but adds a GCP
Discovery Engine dependency + one recall-time API call — a product decision (self-host BYOK can't use
it). Runs: `bench/results/locomo_vertex25_n100/`, `bench/results/lme_vertex25_n100/`.
