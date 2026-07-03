# LoCoMo — Cortex results

**2026-06-30** · full LoCoMo (10 conversations, **1986 QA**) · gemini-3.5-flash judge (votes=1) ·
embed cache · seed 0. Adapter: `cortex_bench/locomo.py` (`--locomo`).

## Cheap config — gemini-2.5-flash + top_k=50 (~$0.0019/query)

| category | accuracy | n |
|---|---|---|
| single-hop | 0.841 | 841 |
| adversarial (abstention) | 0.877 | 446 |
| temporal | 0.801 | 321 |
| **multi-hop** | **0.589** | 282 |
| open-domain | 0.396 | 96 |
| **overall** | **0.785** | 1986 |

recall@k 0.986 · reader $/q **$0.0019** · p50 latency 3.5s.

## Competitor comparison (LoCoMo; ⚠️ judges differ — indicative, not apples-to-apples)

| System | LoCoMo | Reader | Judge | Source |
|---|---|---|---|---|
| Cognis (SOTA) | ~0.925 | Claude Opus 4.6 + reranker | — | arXiv 2604.19771 |
| **Cortex (cheap)** | **0.785** | gemini-2.5-flash · $0.0019/q | gemini-3.5-flash | this repo |
| EMem (simple retrieval baseline) | 0.78 / 0.84 | gpt-4o-mini / gpt-4.1-mini | gpt-4o-mini | arXiv 2511.17208 |
| full-context (no memory) | 0.723 | gpt-4o-mini | gpt-4o-mini | arXiv 2511.17208 |
| Mem0 (graph) | 0.613 / 0.663 | — | — | arXiv 2511.17208 |
| Zep / Graphiti (graph) | 0.585 / 0.616 | — | — | arXiv 2511.17208 |

## Reading the result

1. **Cortex (cheap) beats the graph memory systems on LoCoMo too** — 0.785 vs Mem0 0.61–0.66 and
   Zep 0.585–0.62 — and lands on par with the strong simple-retrieval baseline (EMem 0.78–0.84),
   at **~$0.0019/query**. This is the *same verdict as LongMemEval*: a well-engineered retrieval +
   reader core beats biologically-inspired graph memory, on a second benchmark. (Judge caveat applies.)
2. **Multi-hop (0.589) is the genuine frontier** — relational reasoning across memories. Notably the
   research shows graph memory *regresses* here too (Mem0^g 24.3 vs 28.6 F1), so the lever is a
   **stronger reader / query decomposition**, not a graph. A quality-tier reader should lift this.
3. **Open-domain (0.396) is largely a non-memory skill** — these need world knowledge *outside* the
   conversation; a memory system answering only from stored context will (correctly) score low.
   It drags the average but isn't a memory failure.
4. **Strong where memory matters:** single-hop 0.84, temporal 0.80, adversarial/abstention 0.88,
   retrieval recall 0.99.

## Multi-hop error analysis (282 cases, 19-agent adversarial classification)
An independent Claude panel re-graded + classified all 282 multi-hop cases (panel 0.528 ≈ harness 0.589):

| failure mode (of 133 wrong) | count | fixable by |
|---|---|---|
| **retrieval-miss** (needed fact absent from the reader's notes) | **99 (74%)** | more retrieval depth / reranking |
| reader-reasoning (facts present, wrong synthesis) | 23 (17%) | stronger reader |
| answer-format (right but judge-unfriendly) | 10 | — |
| ambiguous gold | 1 | — |

**Verdict: the multi-hop gap is retrieval precision + output truncation, NOT a deep reasoning limit.**
A recurring pattern in the failures is *"truncated, no answer"* — multi-hop questions enumerate/aggregate
many items, blowing the **2048-token output budget** mid-notes before the ANSWER. So the cheap, high-leverage
fixes are: **(1) larger reader output budget** (kill truncation), **(2) more retrieval depth / a reranker**.
A stronger reader fixes only ~25% (33/133). This matches the deep-research (a cross-encoder reranker is the
named SOTA lever) and is consistent with graph memory *not* helping multi-hop. Next experiments (cheap config):
bump `--max-output-tokens` and `--top-k`; then add a reranker — each gated on this benchmark.

## Quality config — gemini-3.5-flash + top_k=50 (best reader) · COMPLETE

**2026-06-30** · full LoCoMo (1986 QA) · gemini-3.5-flash reader (2048 output) · gemini-3.5-flash
judge (votes=1) · seed 0. Completed via `--resume` on a second key, **0 failures**.

| category | 2.5-flash (cheap) | 3.5-flash (quality) | Δ |
|---|---|---|---|
| single-hop | 0.841 | **0.874** | +0.033 |
| temporal | 0.801 | **0.835** | +0.034 |
| multi-hop | 0.589 | 0.578 | −0.011 |
| open-domain | 0.396 | 0.406 | +0.010 |
| adversarial (abstention) | **0.877** | 0.765 | −0.112 |
| **overall** | **0.785** | 0.778 | −0.007 |

reader $/q **$0.0019** · recall@k 0.990 · p50 latency 4.2s.

### Reading the result — the stronger reader is a wash
A more capable reader (gemini-3.5-flash, thinking) **lifts the answerable categories** (single-hop
+3.3, temporal +3.4) but **regresses on abstention** (−11): it answers unanswerable (adversarial)
questions instead of declining. The two effects cancel, so overall is unchanged (0.778 vs 0.785) at
the same cost. Takeaways:
1. **The cheap config is the sweet spot** — equal accuracy-per-dollar, and it abstains better.
2. **The frontier is not reader capability.** Multi-hop barely moves (0.589→0.578), confirming the
   error analysis above: the lever is retrieval precision + output budget + a reranker, not a bigger
   reader.
3. **Abstention needs explicit tuning for strong readers** — a capable reader over-answers unless
   the prompt/calibration pushes it to decline (cf. preference-mode on LongMemEval).

The `--resume` path proved out here: an interrupted ~2h run finished cleanly on a second key,
re-grading saved work without recomputation and recording zero failures.

## V3 update (2026-07-02): the named fixes tested — depth is the lever, budget is neutral

Following the multi-hop error analysis above ("next experiments: bump `--max-output-tokens` and
`--top-k`; then a reranker"), both were tested at the cheap config (gemini-2.5-flash, seed 0,
gemini-3.5-flash judge votes=1):

- **Output budget + answer-first (A1) = NEUTRAL on LoCoMo.** Raising `--max-output-tokens` to 8192 and
  emitting ANSWER before NOTES (`--answer-first`) held accuracy at ~0.785 (k=50). LoCoMo's contexts are
  small (~5.4k tok) so budget truncation was far less severe than on LongMemEval, where the *same* fix
  gave **+3.8pt (0.894→0.932)**. A1 is context-size-specific — it pays where contexts are large.
- **Retrieval depth (k=50→100) = +3.1pt.** Clean same-question A/B (387 shared Q, both answer-first):
  k=50 **0.783** → k=100 **0.814**. Depth fixed 24 / broke 12 (net +12); k=100 subset-500 overall
  **0.800** vs 0.785 baseline. The "broke 12" is over-retrieval noise — precisely why a **reranker**
  (keep depth's recall, restore top-k precision) is the documented next lever.

### Full-1986 result — k=100 + answer-first (`bench/results/locomo_k100_full/`)

gemini-2.5-flash reader, top_k=100, answer-first, gemini-3.5-flash judge (votes=1), seed 0,
**$0.0034/q, recall 0.998**.

| category | baseline (k=50) | **k=100** | Δ |
|---|---|---|---|
| single-hop | 0.841 | 0.864 | +2.3 |
| **multi-hop** | 0.589 | **0.695** | **+10.6** |
| temporal | 0.801 | 0.857 | +5.6 |
| open-domain | 0.396 | 0.458 | +6.2 |
| adversarial (abstention) | 0.877 | 0.836 | −4.1 |
| **overall** | **0.785** | **0.813** | **+2.8** |

**Reading it:** the prior error analysis is validated end-to-end — the LoCoMo frontier is
**retrieval depth**, not reader capability or truncation. Doubling depth (k=50→100) recovers the
74%-retrieval-miss multi-hop bucket (**+10.6pt**, the largest single gain) plus temporal/open-domain,
because multi-hop needs several sessions in context at once. The lone regression is
adversarial/abstention (−4.1): more context + answer-first makes the reader answer some
*unanswerable* questions — the exact failure the **A2 detect-then-decline** gate (conservative model
owns the unanswerable verdict) is designed to fix. **Next LoCoMo levers:** A2 (recover adversarial),
then a reranker (A9) to restore precision lost to over-retrieval, then a depth sweep (k=150).

## A2 detect-then-decline — tested, does NOT recover the adversarial regression (2026-07-02)

**Motivation:** the k=100 depth win above cost −4.1pt on adversarial/abstention (0.877→0.836 on
full-1986) because more context makes the answer-first reader over-answer genuinely-unanswerable
questions. A2 aimed to recover this via "detect-then-decline": either a separate conservative gate
(`--answerability-gate`) that rules each question ANSWERABLE/UNANSWERABLE before the reader, or a
reader-side strict abstention policy (`--strict-abstain`) that forbids emitting a specific detail no
memory states.

**Methodology / confound (important to record):** the first probe round used the default
`--max-output-tokens 256`, which starved the gemini-2.5-flash *thinking* reader and truncated
answers — collapsing every variant to ~0.40 overall (a measurement artifact, NOT the mechanism).
Re-running at `--max-output-tokens 8192` (matching the baseline) gave the clean numbers below.
Lesson: always match the baseline's output budget for thinking readers.

Config (all rows): gemini-2.5-flash reader, top_k=100, answer-first, `--max-output-tokens 8192`,
gemini-3.5-flash judge (votes=1), seed 0, same n=100 seed-0 subset (same-question A/B; baseline
verdicts reused from the cached full k=100 run, so only the variant arm was freshly run).

| variant | overall | adversarial | temporal | single-hop | multi-hop | $/q |
|---|---|---|---|---|---|---|
| baseline (k=100, no gate) | 0.770 (77/100) | 0.857 (18/21) | 0.917 (11/12) | 0.784 (40/51) | 0.615 (8/13) | 0.0034 |
| `--answerability-gate` | 0.760 (76/100), Δ −0.01 | 0.857 (18/21), Δ 0.00 | 0.833 (10/12), Δ −0.083 | 0.784 (40/51), Δ 0.00 | 0.615 (8/13), Δ 0.00 | 0.0057 (+67%) |
| `--strict-abstain` | 0.710 (71/100), Δ −0.06 | 0.857 (18/21), Δ 0.00 | 0.667 (8/12), Δ −0.25 | 0.725 (37/51), Δ −0.059 | 0.615 (8/13), Δ 0.00 | 0.0034 |

`--answerability-gate` (gemini-3.5-flash gate) declined on 30/100; `--strict-abstain` (reader-side,
free) declined on 27/100.

**Reading the result (the key insight):** neither mechanism recovers a single adversarial question —
adversarial is identical 18/21 in the baseline and BOTH variants. The 3 adversarial questions the
baseline over-answers are *hard* cases: their retrieved memories look answer-bearing, so a gate and a
strict reader both rule them ANSWERABLE for the same reason the reader answers them. Meanwhile both
mechanisms add false declines on genuinely-answerable questions (strict-abstain especially hurts
temporal, −0.25). The gate is strictly dominated: same accuracy at +67% cost (worse
accuracy-per-dollar). Conclusion: at k=100 the reader's willingness to answer from loosely-related
context simultaneously powers the +10.6 multi-hop / +5.6 temporal depth gains AND the adversarial
over-answering — they are entangled, so a blunt abstention lever is net-negative. The −4.1 adversarial
is best understood as an already-paid, non-cheaply-recoverable cost of the depth win.

**Decision:** A2 killed at the n=100 gate (zero adversarial-recovery signal → no justification to
spend on full-1986 runs). The `--answerability-gate` and `--strict-abstain` flags are retained OFF BY
DEFAULT as tested, available levers (they may help a stronger flagship reader where adversarial cases
are more separable). Next LoCoMo lever: a reranker (L2) to restore precision lost to over-retrieval.

## L2 reranker — cheap Gemini listwise rerank FAILS the +1pt gate (2026-07-02)

**Motivation:** the k=100 depth win (above) gave +2.8pt overall but dragged in over-retrieval noise —
it hurt adversarial/abstention and "broke 12" previously-correct answerable questions. L2 tests the
documented SOTA lever for exactly this failure mode: a **reranker** — over-retrieve the deep top-100
pool, rerank, and feed the reader only the top-25 most useful chunks, keeping depth's recall while
restoring top-k precision and shrinking the reader's context. The research names a **cross-encoder**
reranker, but the repo has no Vertex client path, and the AI-Studio-key `GeminiProvider` cannot call
the Vertex AI Ranking API without David enabling the API + granting runtime IAM (a human blocker, not
a code blocker). So this round tests the feasible substitute: a cheap **Gemini listwise reranker** —
one auxiliary call (`gemini-2.5-flash-lite`) returns a ranked id list over the 100 candidates, and the
reader sees only the top `rerank-k=25`.

**Config:** `--top-k 100 --rerank --rerank-k 25 --answer-first --max-output-tokens 8192`; rerank aux
model = gemini-2.5-flash-lite; gemini-3.5-flash judge (votes=1); seed 0; n=100 same-question A/B.
LoCoMo reader gemini-2.5-flash (baseline = cached k=100 run). LME reader gemini-3.5-flash +
preference-mode (baseline = fresh k=100 no-rerank arm).

### LoCoMo (rerank vs k=100 baseline; 98/100 clean after 2 transient 503s)

| category | k=100 baseline | rerank (k=100→25) | Δ |
|---|---|---|---|
| overall | 0.770 (77/100) | 0.765 (75/98) | −0.005 |
| multi-hop | 0.615 (8/13) | 0.417 (5/12) | **−0.199** |
| single-hop | 0.784 (40/51) | **0.840 (42/50)** | **+0.056** |
| temporal | 0.917 (11/12) | 0.833 (10/12) | −0.083 |
| adversarial | 0.857 (18/21) | 0.857 (18/21) | 0.000 |

recall@k (reader's top-25 view) 0.998 (top-100) → 0.78 · $/q 0.0034 → 0.0035 (neutral).

### LongMemEval-S (rerank vs k=100 no-rerank baseline; n=100 subset is near-saturated — a
ceiling-limited accuracy test)

| category | k=100 baseline | rerank (k=100→25) | Δ |
|---|---|---|---|
| overall | 0.990 (99/100) | 0.969 (95/98) | −0.021 |
| temporal-reasoning | 1.000 (25/25) | 0.920 (23/25) | **−0.080** |
| knowledge-update / single-session-* / preference / abstention | 1.000 | 1.000 | 0.000 |
| multi-session | 0.960 | 0.958 | −0.002 |

$/q 0.01535 → 0.01873, Δ **+22%** (MORE expensive).

**Reading the result (the key insight):** a cheap Gemini listwise reranker is NOT a cross-encoder. It
**demotes the weak-but-necessary multi-hop bridge chunks** — multi-hop collapses −19.9pt and recall of
the reader's top-25 view drops to 0.78, i.e. pruning 100→25 loses the answer session ~22% of the time.
It DOES help single-hop (+5.6pt) by cutting noise, confirming the noise-reduction premise is real —
but the multi-hop and temporal losses swamp it, netting −0.5pt on LoCoMo. On LME the n=100 subset sits
at a 0.99 ceiling (no headroom to show +1pt), and worse, the extra rerank call over 100 large LME
chunks isn't offset by the smaller reader context, so $/q *rises* 22%. **Fails the +1pt hard gate on
both benchmarks** — LoCoMo net-negative, LME ceiling-bound and costlier.

**Decision:** the cheap listwise reranker is **killed (off by default)**. The `--rerank`/`--rerank-k`
plumbing (over-retrieve → rerank → top-k) is retained as reusable scaffolding. The real lever remains a
**cross-encoder reranker via the Vertex AI Ranking API** (semantic-ranker) — the documented SOTA move —
which needs David to enable the Vertex Ranking API + grant runtime IAM (no Vertex client path exists
yet). Flagging this as a concrete human-gated action for the flagship phase. Meanwhile the single-hop
+5.6pt hints that noise-reduction with preserved recall (a good cross-encoder) could still pay off.
