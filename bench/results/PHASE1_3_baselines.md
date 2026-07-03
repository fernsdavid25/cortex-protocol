# Phase 1.3 / Checkpoint B — Representative baselines + judge-vs-reader diagnostic

**Status: representative full-context baseline (n=60) verified; naive-rag artifact on disk is a
stale partial run and must be re-run before any relative comparison is published.**

LongMemEval **oracle** variant (evidence-only haystack; the easiest setting — published
GPT-4o-oracle is ≈0.87). Reader `gemini-2.5-flash-lite`, embed `gemini-embedding-001`, judge
`gemini-2.5-flash` with `votes=3` (self-consistency majority). Seeded shuffle (`--seed 0`),
`--limit 60`.

## Representative numbers

| System        | n  | Overall acc | Abstention (n=5) | Recall@k | Mean in-tok | Mean out-tok | $/q (reader) | p50 lat (ms) | p95 lat (ms) |
|---------------|----|-------------|------------------|----------|-------------|--------------|--------------|--------------|--------------|
| full-context  | 60 | **0.400**   | 0.200            | 1.000    | 6189.9      | 68.3         | 0.000646     | 1258.7       | 2147.3       |
| naive-rag\*   | 25 | 0.160       | n=0              | 0.972    | 11194.0     | 69.0         | 0.001147     | 1934.3       | 2442.8       |

\* **naive-rag is NOT comparable.** The on-disk `naive-rag_oracle_results.json` is an earlier
partial/interrupted run of only the first 25 instances, which (the oracle file is grouped by type)
are **all `temporal-reasoning`, zero abstention** — the single hardest slice. Its 0.16 is a
worst-slice smoke number, not a 60-item baseline. **Re-run naive-rag at the same seeded n=60
before publishing any full-context-vs-rag delta.**

### full-context per-type accuracy (gemini judge, n=60)

| Question type             | n  | Acc    |
|---------------------------|----|--------|
| single-session-assistant  | 6  | 1.000  |
| single-session-user       | 7  | 0.857  |
| multi-session             | 18 | 0.389  |
| knowledge-update          | 10 | 0.400  |
| temporal-reasoning        | 17 | 0.059  |
| single-session-preference | 2  | 0.000  |
| **overall**               | 60 | **0.400** |

recall@k = 1.0 for full-context (it feeds every session), so the loss is entirely in
**reading/reasoning or judging**, not retrieval.

## Diagnostic — is the low oracle score the reader or the judge?

Deterministic offline judge (`cortex_bench.judge.offline_label`, substring containment) re-graded
all 60 full-context hypotheses and compared to the gemini-judged headline:

| Judge                                   | Overall acc | Temporal acc |
|-----------------------------------------|-------------|--------------|
| gemini (headline, votes=3)              | **0.400**   | 0.059        |
| offline (substring containment)         | 0.483 (29/60) | 0.471 (8/17) |
| offline + lenient norm (strip trailing punct / leading article) | 0.517 (31/60) | — |

**Offline-vs-gemini gap = +0.083** (offline slightly *over*-accepts, because its substring match
catches a stray number anywhere in a long temporal response that the semantic gemini judge
rejects). This gap is **small** — there is no large `offline >> gemini` blow-out that would point
to a harsh judge. Going from offline to the lenient norm flips only **2 of 60** items
(`561fabcd` gold `"Fissionator."` — trailing period; `f8c5f88b` `"the sports store downtown"` vs
`"a sports store downtown"` — article). Pure judge/format brittleness is therefore ~2/60.

By contrast, **12 of 60 answerable questions got a bare "I don't know"** — the small flash-lite
reader simply gave up (over-abstention on answerable items), e.g. temporal `gpt4_468eb064` (gold
"Emma"), multi-session `e3038f8c` (gold "99"), single-session-user `1faac195` (gold "Denver").
That is unambiguous reader weakness.

### Manual inspection (16 cases across all types) — bucket counts

Buckets: **A** reader genuinely wrong/weak · **B** reader correct but phrased so a strict judge
says no · **C** correct fact buried/omitted in a long Chain-of-Note · **D** judge harshness/format.

| Bucket | Count | Example qids |
|--------|-------|--------------|
| A (reader wrong / "I don't know" / wrong arithmetic) | 8 | `gpt4_59c863d7`, `cc06de0d` ($12 vs $6), `0ddfec37_abs`, `ce6d2d27` (Thu vs Fri), `06878be2`, `1faac195`, `gpt4_b0863698` (partial temporal), `gpt4_68e94287` |
| C (answer buried in long Chain-of-Note) | 4 | `gpt4_7f6b06db`, `8979f9ec` (got facts, fumbled the sum), `5c40ec5b`, `8cf51dda` |
| D / format (literal answer present, judge/substring quibble) | 2 | `561fabcd` (trailing period), `f8c5f88b` (article) |
| pass (offline-correct) | 2 | `15745da0`, plus B `f8c5f88b` overlaps D |

Globally over all 60: 27/60 hypotheses are long Chain-of-Note (>250 chars); of 31 offline-fails,
22 have <0.3 token overlap with the gold (reader genuinely wrong), 9 are long CoN, and only 6 have
the answer likely present (overlap ≥0.6). **The mass is bucket A**, with a meaningful **C** tail on
numeric/temporal counting (the reader retrieves the right facts but botches the final
arithmetic/ordering), and a negligible **D** slice.

## Dominant cause

**A — genuine reader weakness — dominates**, with a secondary **C** (Chain-of-Note arithmetic/
ordering failures on temporal/multi-session) and a negligible **D** (≈2/60 format). The judge is
NOT the problem: offline ≈ gemini (gap +0.08, and offline if anything *over*-accepts). The low
oracle score is real — `gemini-2.5-flash-lite` is too weak for oracle reading: it over-abstains on
answerable questions and fails multi-step temporal/numeric reasoning (temporal acc 0.059).

## CHECKPOINT B VERDICT

1. **Spine works + reproducible.** End-to-end harness (ingest → answer → judge → metrics) runs
   green on a seeded n=60 oracle slice; recall@k=1.0 for full-context isolates the loss to
   reading/judging; all artifacts (results.json, hypotheses.jsonl) persist for offline re-grade.
2. **Dominant cause of low oracle accuracy = reader weakness (bucket A), not the judge.**
   Offline-vs-gemini gap is only +0.08; 12/60 answerable items got "I don't know" and temporal
   reasoning collapses to 0.059. flash-lite is under-powered for oracle reading; a CoN tail (C)
   also costs numeric/temporal items.
3. **Relative comparisons are trustworthy enough to iterate on in Phase 2** *for full-context*: the
   judge tracks the offline label closely, recall is perfect, and per-type structure is sensible.
   **Caveat:** the naive-rag artifact on disk is a stale 25-item single-type partial — re-run it at
   the same seeded n=60 before quoting any full-context-vs-rag or Cortex-vs-baseline delta.
4. **Recommendation on the gpt-4o judge:** the gemini judge is good enough for *relative* Phase-2
   iteration, so do **not** block on it. But because absolute oracle numbers are far below the
   ~0.87 literature mark, validate the gemini judge against the official `gpt-4o-2024-08-06` judge
   on this same 60-item set **before publishing any absolute number**. Request the OpenAI key now so
   the one sanctioned validation pass is ready when Phase-2 candidates exist — but the bigger lever
   for absolute accuracy is upgrading the *reader* (e.g. `gemini-2.5-flash`/`pro`), not the judge.
