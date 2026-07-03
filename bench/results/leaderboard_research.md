# LongMemEval_S — Verified Competitor Leaderboard (deep research, 2026-06-29)

Multi-source, adversarially-verified research (99 agents, 17 sources fetched, 25 claims verified).
**Bottom line: the canonical judge is `gpt-4o-2024-08-06` (>97% human agreement); cross-system
comparison is only valid under the same judge. Several systems outscore Cortex with premium readers.**

## Ranked leaderboard (LongMemEval_S, 500 questions) — judge flagged

| # | System | Accuracy | Reader | Judge | Source |
|---|---|---|---|---|---|
| 1 | Mastra (Observational Memory) | 0.9487 / 0.9327 / 0.8920 / 0.8423 | gpt-5-mini / gemini-3-pro / gemini-3-flash / gpt-4o | GPT-4o | mastra.ai/research/observational-memory |
| 2 | ByteRover 2.1.5 (claimed, unverified) | 0.928 | Gemini-3.1-pro | unstated | byterover.dev blog |
| 3 | Hindsight (Vectorize.io, arXiv 2512.12818) | 0.914 / 0.890 / 0.836 | Gemini-3 / GPT-OSS-120B / GPT-OSS-20B | **GPT-OSS-120B** (not GPT-4o) | arxiv.org/html/2512.12818v1 |
| 4 | Emergence AI | 0.860 | gpt-4o | GPT-4o | emergence.ai/blog/sota-on-longmemeval-with-rag |
| 5 | Supermemory | 0.852 / 0.846 / 0.816 | Gemini-3 / GPT-5 / GPT-4o | GPT-4o | Supermemory tech report (via Hindsight table) |
| — | **Cortex (this repo)** | **0.866 / 0.840** | gemini-3.5-flash / 2.5-flash (~$0.008/q) | **gemini-flash** | this repo, full 500q |
| 6 | Zep / Graphiti | 0.712 | gpt-4o | GPT-4o | blog.getzep.com / Supermemory table |
| 7 | LongMemEval paper — full-context | 0.606 / 0.640 (CoN) | gpt-4o | GPT-4o | arXiv 2410.10813 |
| — | LongMemEval paper — ORACLE ceiling | 0.870 / 0.924 (CoN) | gpt-4o | GPT-4o | arXiv 2410.10813 (no-retrieval upper bound) |

## Key facts (verified)

- **Benchmark:** LongMemEval_S = 500 questions, ~115k tokens (~40 sessions) per question. _M = ~500 sessions (~1.5M tokens). _Oracle = gold-evidence sessions only (retrieval upper bound). Same 500 questions across variants.
- **Canonical judge = `gpt-4o-2024-08-06`**, per-question-type prompts, >97% human agreement. Configurable (gpt-4o-mini, llama-3.1-70b also allowed) — which is why cross-system numbers drift. Self-preference bias ≈ +10% when generator == judge.
- **Original paper, GPT-4o reader:** full-context (read entire history) 0.606 (0.640 CoN); oracle 0.870 (0.924 CoN). A memory system's job is to beat the 0.606 full-context number while approaching the oracle.

## Caveats (load-bearing)
1. **Judge inconsistency dominates.** Cortex's gemini-flash judge is NOT comparable to any published number until re-graded with GPT-4o. Gemini judges are often *more lenient* → Cortex's true GPT-4o-judged score could be lower.
2. **Vendor self-reporting.** Every score above the paper baselines (Mastra, Hindsight, Supermemory, Emergence, Zep) is self-reported, not independently reproduced. There is **no enforced public leaderboard**.
3. **Hindsight uses a GPT-OSS-120B judge** — its 0.89/0.914 are not comparable to GPT-4o-judged rows; its *baseline* column is copied from Supermemory's GPT-4o-judged report.
4. **Oracle ≠ end-to-end.** The paper's 0.870/0.924 remove retrieval entirely — not a system to "beat" the same way.
5. **No verifiable Mem0 / Letta / Memary / Memobase LongMemEval_S number** (with variant+judge) survived verification (they often report LoCoMo instead). Earlier "Mem0 ~0.49" / "Zep 63.8%" were refuted.

## Biological-vs-retrieval verdict (deep research, 2026-06-29, 102 agents, cited)

**Question:** would a refined biologically-inspired memory (graph / reflection / hierarchical tiers) be state-of-the-art vs simple hybrid retrieval + a strong reader? **Verdict: No — SOTA is retrieval + reader; graph regresses; reflection only ties.**

- **SOTA on BOTH benchmarks = Cognis** (~92.4% LongMemEval, ~92.5% LoCoMo): dual-store **BM25 + vector, RRF fusion, cross-encoder reranker, Claude-Opus-4.6 reader** — *no graph, reflection, or tiers in core*. Biggest single gain = a better BM25 backend (+20.3%). RRF(vector+BM25) "consistently outperforms any single approach." [arXiv 2604.19771] → **validates Cortex's dense+BM25+RRF design; names our upgrade levers: a reranker + a stronger reader.**
- **Simple combined retrieval beats specialized biological systems:** EMem (dense + LLM filter) = **0.78/0.84 LoCoMo**, **76–83% LongMemEval** — beats Zep (graph) 0.585/0.616 and Mem0 (graph) 0.613/0.663. Authors: graph propagation "helpful but not universally necessary." [arXiv 2511.17208]
- **Graph specifically REGRESSES:** Mem0^g loses on the multi-hop it targets (24.3 vs 28.6 F1) at ~3× latency / ~2× tokens; no-graph variant wins multi-session, knowledge-update, open-domain.
- **Reflection = only competitive biological mechanism** (Mastra 84%, Hindsight 84%, vendor self-reported) — not clearly superior; the "94.87% highest-ever" claim was **refuted**.
- **Hierarchical tiers + consolidation (MemGPT/Letta, MemoryBank): no confirmed benchmark win.**
- **Net (accuracy-per-dollar):** biological mechanisms are at best a tie, often a net loss. HippoRAG-2 is the one graph bright spot (multi-hop +9.5 F1 on 2Wiki) but is **not** evaluated on LongMemEval/LoCoMo.

**Product implication:** build **ONE combined core** (hybrid retrieval + reranker + reader), not multiple specialized protocols. Add biological modules (reflection, graph) only as optional flags *if/when a benchmark proves they earn their cost*.

## What this means for Cortex
- **Targets to beat (under the canonical GPT-4o judge):** Emergence 0.86, Supermemory 0.852, and the leaders Mastra (0.84–0.95) / Hindsight (0.91, diff judge) / ByteRover (0.93, unverified).
- **Step 0 (required): GPT-4o-judge re-grade of Cortex outputs** (`evaluate_qa.py gpt-4o`) — needs an OpenAI key. Until then no ranking claim is defensible.
- **Differentiator: accuracy-per-dollar.** The leaders use premium readers (gpt-5-mini, gemini-3-pro, GPT-5). Cortex matches the gpt-4o-reader tier at ~10× lower reader cost.
- **To reach 0.90+:** the leaders use an *observational-memory / reflection* architecture (an observer model distils sessions into observations/reflections) + premium readers — a different design than Cortex's retrieval-only pipeline.
