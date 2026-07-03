# E1 — Failure Autopsy (LongMemEval headline 0.894)

**Date:** 2026-07-02 (V3 research, overnight). **Source run:** `r35f_k50_pref_n500` (gemini-3.5-flash reader, top_k=50, `--preference-mode`, gemini-3.5-flash judge votes=3, full 500).

## Method + a caveat that matters
The persisted hypotheses store only `{question_id, hypothesis}` (legacy format, no per-question verdict), so correctness must be reconstructed. **The offline judge is unusable for failure-finding**: it is a naive containment check (`gold.lower() in hypothesis.lower()`) and scores the run at **0.650 vs the real 0.894** — it fails on paraphrase and on preference questions:
- GOLD `"The GR-90 trail."` vs ANS `"The hiking trail recommended is the GR-90."` → scored **wrong** (it's right).
- GOLD `"Three times a week."` vs ANS `"Three times a week (previously twice a week)."` → scored **wrong** (it's right).
- `single-session-preference` scores **100% wrong** offline because the gold is a *preference description* ("The user would prefer responses that…"), ungradeable by containment.

⇒ True per-type failure rates require the **gemini judge**; that re-grade is folded into the A1 A/B and the planned full re-run. What the offline pass *does* reveal cleanly is a **format/truncation signal** that needs no judge.

## The dominant, actionable failure: output-budget truncation (→ A1)
**69/500 (13.8%) of headline answers contain no `ANSWER:` marker at all** — the Chain-of-Note `NOTES:` section exhausts the reader's output budget at top_k=50 before it reaches the answer. Evidence (verbatim tails):
- `gpt4_7f6b06db` (trip ordering) ends: `…- Yosemite (first trip): before April 20, 2023\n  - Big Sur and` — cut off mid-enumeration.
- `c8f1aeed` ends: `…for a certain period after drilling is complete." (Memory 1)\n\nPennsylvania` — cut off.
- `1da05512` ends: `…Because a NAS is a convenience and capacity upgrade rather than an emergency,` — cut off.

**The truncation is concentrated in our weakest types** (not random):

| type | truncated / total | % of type |
|---|---|---|
| temporal-reasoning | 24 / 127 | **19%** |
| multi-session | 19 / 121 | **16%** |
| knowledge-update | 9 / 72 | 12% |
| single-session-preference | 4 / 30 | 13% |
| single-session-assistant | 5 / 56 | 9% |
| single-session-user | 5 / 64 | 8% |
| (abstention overlay) | 3 | — |

temporal (0.87) and multi-session (0.835) are exactly the two biggest weighted weaknesses, and they truncate the most — the reasoning-heavy types produce the longest NOTES and run out of budget first. **Fixing truncation directly attacks the weak types.**

## Fix → A1 (implemented + under test)
1. **Raise `--max-output-tokens`** (benchmark default is **256**; even the headline's higher value truncates 69 answers at k=50). A/B: 2048 vs 8192 at n=100 (running).
2. **`--answer-first`** reader variant (committed `f9699fb`): emit `ANSWER:` before `NOTES:` so budget truncation can never eat the answer; thinking-capable readers reason internally then lead with the answer. Opt-in, ablatable.

**Expected:** synthesis estimates +1.5–2.5pt LME (P0). Because the truncations sit in temporal + multi-session, the recovered points land on the weakest types. Kill-criterion: if answer-first + 8k doesn't cut the no-ANSWER rate below ~1% on a 100-q spot check, keep the raised cap and drop the reformat.

## Other buckets (to be sized with the gemini judge, next)
- **Adversarial over-answering (LoCoMo, → A2):** the stronger 3.5-flash reader regressed the adversarial category (0.877→0.765) by answering unanswerable questions — a *representational* over-confidence that needs a policy/architecture gate (detect-then-decline), not a confidence threshold.
- **Judge-format on count/enumerate (multi-session, → A4):** the `_STANDARD` rubric marks a right count wrong if the enumerated detail is omitted.
- **Knowledge-update staleness (→ A7):** occasional old-state answers where a superseded fact wins.

See `Docs/Cortex_V3_Research_Report.md` §A for the full ranked experiment plan.
