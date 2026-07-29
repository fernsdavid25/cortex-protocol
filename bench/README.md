# Cortex Bench — LongMemEval Harness

Reproducible LongMemEval evaluation for Cortex. **Phase 0 verified facts below are confirmed from the official repo + dataset** ([xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval), [HF dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned), [paper arXiv 2410.10813](https://arxiv.org/abs/2410.10813)), verified 2026-06-15.

## Target & version (task 0.2)
- **Target: LongMemEval V1, `LongMemEval_S`** — this is what Mem0/Zep/Supermemory report on. **`LongMemEval-V2` is a separate *agentic* benchmark (May 2026) — NOT our target.**
- Use **Oracle** to isolate the reading stage; **M** (~500 sessions, ~1.5M tokens) as the retrieval stress test.

## Dataset (task 0.1 — VERIFIED)
- Source: HuggingFace `xiaowu0162/longmemeval-cleaned` (the 2025-09 cleaned release). Download:
  ```
  wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
  wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
  wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json
  ```
  (Downloads go to `bench/data/`, which is gitignored.)
- `LongMemEval_S` ≈ 115k tokens / ~40 sessions (Llama-3 tokenizer — **re-measure under Gemini's tokenizer** before publishing token/cost numbers).
- **Instance fields:** `question_id, question_type, question, answer, question_date, haystack_session_ids, haystack_dates, haystack_sessions, answer_session_ids`. Evidence turns carry `has_answer: true`.

### Question-type counts (VERIFIED from `longmemeval_oracle.json`, n=500)
| question_type | count |
|---|---|
| single-session-user | 70 |
| single-session-assistant | 56 |
| single-session-preference | 30 |
| multi-session | 133 |
| knowledge-update | 78 |
| temporal-reasoning | 133 |
| **total** | **500** |

- **Abstention = 30 questions** whose `question_id` ends in `_abs`. These are an **overlay** on the 6 types above (each `_abs` question still has one of the 6 `question_type` values), NOT a 7th disjoint type — so the 6 counts already sum to 500. (This resolves the earlier "suspicious sum" flag.)
- Retrieval eval **skips the 30 `_abs` instances** (no ground-truth answer location).

## Judge protocol (task 0.3 — VERIFIED from `src/evaluation/evaluate_qa.py`)
- **Model:** `gpt-4o` → `gpt-4o-2024-08-06` (also supports `gpt-4o-mini-2024-07-18`, and a local `llama-3.1-70b-instruct` via vLLM). The OpenAI judge is the **only sanctioned non-GCP spend**, headline runs only.
- **Call:** `temperature=0, max_tokens=10, n=1`, exponential backoff. **Label:** `'yes' in response.lower()`.
- **Metric:** plain accuracy overall + per `question_type`.
- **Per-type prompt routing (mandatory):**
  - `single-session-user|single-session-assistant|multi-session` → "does the response contain the correct answer" (subset = no).
  - `temporal-reasoning` → same + **off-by-one-day tolerance** for day/week counts.
  - `knowledge-update` → correct if it contains the **updated** answer (old info alongside is fine).
  - `single-session-preference` → graded against a **rubric** (recalls/uses the user's personal info).
  - `_abs` (abstention) → correct if the model **identifies the question as unanswerable**.
- **Hypothesis file:** JSONL, `{question_id, hypothesis}` per line. Eval appends `autoeval_label`.
- ⚠️ Note: `evaluate_qa.py` buckets per-type by `question_type` only — abstention is NOT a separate bucket there. **Our harness will add a distinct abstention bucket (by `_abs`) + report abstention accuracy vs non-abstention recall** (abstention is gameable by over-abstaining).

## Harness design (Phase 1.2 — next)
- `MemorySystem` ABC: `ingest(history) -> None`, `answer(question, question_date) -> str`. Baselines + Cortex + competitors all implement it.
- Pipeline: load `_S` → per question: `ingest(haystack_sessions)` → `answer(...)` → write hypothesis JSONL → judge (Gemini for iteration; `gpt-4o-2024-08-06` for headline, validated for correlation) → metrics.
- **Metrics report:** accuracy overall + per type + **abstention bucket**; **mean context tokens/query, $/question (ingest extract + read + judge), p50/p95 latency, tokens-per-correct-answer**; plus two retrieval metrics at two different granularities — **recall@k** from `answer_session_ids` (session granularity) and **coverage@k** from LongMemEval's per-turn `has_answer` / LoCoMo's `dia_id` (turn granularity). See "Reader-independent metrics" below.
- **Judge code:** reuse the official prompt templates (verbatim, recorded above). Before vendoring `evaluate_qa.py` into this Apache-2.0 repo, check the upstream LICENSE for compatibility (Phase 1.2 TODO); otherwise reimplement the templates (which are factual prompts) with attribution.
- **Baselines (Phase 1.3/1.4):** full-context (GPT-4o + a Gemini reader), naive-RAG, **Mem0-OSS** (settle the 49%-vs-93.4% contradiction), Zep-OSS/Graphiti.

## Checkpoint A — self-review verdict (autonomous)
All four Phase-0 unknowns resolved from primary sources (counts verified by direct dataset count; judge verified from source; dataset path verified by successful download; V1-vs-V2 resolved). **No blockers. Proceeding to Phase 1.**

## Datasets supported + how to run
- **LongMemEval** (S / M / oracle) — auto-downloaded to `bench/data/`. The 0.932 headline config
  (drop `--limit` / raise it to 500 for the full run):
  ```
  python -m cortex_bench.run --system cortex-v0 --variant s --limit 100 \
    --reader-model gemini-3.5-flash --top-k 50 --preference-mode --answer-first \
    --max-output-tokens 8192 --judge gemini --judge-votes 3
  ```
- **LoCoMo** (multi-hop / temporal / open-domain / single-hop / adversarial) — put `locomo10.json`
  in `bench/data/` (from [snap-research/locomo](https://github.com/snap-research/locomo)), then run
  the 0.813 headline config:
  ```
  python -m cortex_bench.run --system cortex-v0 --locomo --reader-model gemini-2.5-flash \
    --top-k 100 --answer-first --max-output-tokens 8192 --judge gemini
  ```
  Categories map to `locomo-*` question types; adversarial → abstention. See `cortex_bench/locomo.py`.

Per-role models are env-configurable (`CORTEX_READER_MODEL`, `CORTEX_JUDGE_MODEL`, …; see `../.env.example`).

## Reader-independent metrics (coverage@k + MUR)

Accuracy conflates two things: how good the **memory** is at putting the right evidence in
front of the reader, and how good the **reader** is at using it. These two metrics separate
them, and neither one asks an LLM anything about quality.

### coverage@k — turn-level evidence coverage (fully deterministic)

`recall@k` is SESSION-level: "did we surface the right *conversation*". A system that drags
in a whole 40-turn session because one of its turns is gold scores 1.0. That is why the
shipped **0.998 recall@k is not evidence of good memory**.

`coverage@k` is TURN-level: "did we surface the right *memory*" — the fraction of the gold
evidence **turns** contained in the memories the reader actually saw. The gap between the two
is the interesting number: high recall with low coverage means the retriever is finding the
right haystack and handing the reader the wrong needle.

- **Turn identity** is a string `"{session_id}#{position}"`, `position` **0-based** into
  `haystack_sessions[i]`. Defined once, in `cortex_bench/turns.py`, and used by every dataset
  adapter and every system.
- **LongMemEval gold turns** come from the per-turn `has_answer: true` flag (previously read
  by zero lines of this harness). A release without it yields no turn gold, and coverage
  reports `null` rather than a fabricated `0.0`.
- **LoCoMo gold turns** are resolved from `qa[].evidence` (`"D1:3"`). The primary path is
  **exact `dia_id` string equality** — real LoCoMo turns carry a `dia_id` byte-identical to
  the evidence string, so no off-by-one is possible. Only turns lacking `dia_id` fall back to
  positional resolution via `LOCOMO_EVIDENCE_TURN_BASE = 1` (evidence turn numbers are
  1-indexed; the 0-based position is `m - 1`). That value is **documented inference** — from
  the committed fixture in `tests/test_locomo.py` and the upstream snap-research/locomo
  schema — not observation, since `locomo10.json` is gitignored. A fallback id that lands out
  of range is **dropped and warned about**, so a wrong convention reads as a loud zero rather
  than a plausible number.
- **`--fact-keys` raises recall@k without raising coverage@k.** Distilled fact chunks carry
  session provenance but no turn provenance, so they contribute zero coverage. That asymmetry
  is the pair of metrics working as intended, not a bug.
- **Non-retrieval systems report `null`, not `0.0`.** full-context covers every turn by
  construction, so it declines to enumerate them; the report explains the absence in
  `coverage_at_k_note` instead.
- **coverage@k is a LIVE-RUN metric.** Turn ids are not persisted into `*_hypotheses.jsonl`
  (resumed records are excluded from it anyway), so **the two committed headline runs
  (0.932 / 0.813) predate coverage@k and carry no coverage number.**

### MUR — Memory Utility Ratio

```
MUR = accuracy(system-retrieved context) / accuracy(gold-evidence context)
```

with the **same reader and the same judge on both sides**, so reader capability and judge
leniency cancel to first order. `MUR = 1.0` means retrieval cost the reader nothing;
`MUR = 0.6` means the memory layer threw away 40% of the accuracy the reader could have had.
The denominator is a run over LongMemEval's `oracle` variant (already supported by
`--variant oracle`).

MUR is computed **post hoc** by pairing two committed `*_hypotheses.jsonl` files per
`question_id` — it never runs a reader or a judge, and it costs nothing to recompute:

```
python -m cortex_bench.mur \
  --system bench/results/cortex-v0_s_hypotheses.jsonl \
  --oracle bench/results/cortex-v0_oracle_hypotheses.jsonl \
  --system-results bench/results/cortex-v0_s_results.json \
  --oracle-results bench/results/cortex-v0_oracle_results.json \
  --by-type --variant s --out bench/results/mur_s
```

It pairs the hypotheses rather than dividing the two `accuracy` scalars in `*_results.json`,
because those are rounded over whatever `--limit/--seed/--shuffle` subset each run used —
dividing them silently compares two different populations. Pairing gives an exact ratio over
one common population plus per-type MUR, a paired exact McNemar test, and a paired bootstrap
CI. The `*_results.json` files are an optional second input: `aggregate()` now stamps a `run`
block (system, benchmark, variant, reader, judge, tier, top_k, seed), and MUR checks two
preconditions against it.

- **Same reader, judge and benchmark on both sides** — otherwise they do not cancel. Refused
  unless `--allow-provenance-mismatch`. `system` is deliberately *not* compared: the
  gold-evidence denominator is normally served by `full-context` while the numerator is the
  retrieval system under test.
- **The denominator really is the gold-evidence run** (`run.variant == "oracle"`). Without
  this, two ordinary retrieval runs pair into a green-checked number that is a ratio of two
  retrievers, not MUR. Refused unless `--allow-non-oracle-denominator`, which records
  `denominator_is_oracle: false` and prints a warning.

Runs committed before that stamp existed report `provenance_checked: false` — the
preconditions are recorded as *unchecked*, never assumed. `--min-overlap` (default 0.9)
separately refuses to pair two runs that share too few `question_id`s; it measures the
overlap over the same population the ratio uses, so excluding abstention questions from the
ratio also excludes them from the overlap (a run paired with itself is always 1.0, whatever
its `_abs` share).

LoCoMo has no oracle variant, so MUR ships LongMemEval-first; the CLI pairs on `question_id`
only, so a future gold-evidence-context LoCoMo system needs no code change — just
`--allow-non-oracle-denominator`, since such a run cannot carry the `oracle` variant tag.

## Current results
- **LongMemEval_S headline + methodology:** [`results/PHASE3_authoritative.md`](results/PHASE3_authoritative.md) — **0.932** (gemini-3.5-flash + top_k=50 + preference-mode + the A1 fix `--answer-first --max-output-tokens 8192`, ~$0.0396/q; Gemini judge). Backing artifact: [`results/a1_af8k_full500/cortex-v0_s_results.json`](results/a1_af8k_full500/cortex-v0_s_results.json).
- **LoCoMo headline + methodology:** [`results/LOCOMO_results.md`](results/LOCOMO_results.md) — **0.813** (gemini-2.5-flash + top_k=100 + answer-first, ~$0.0034/q; Gemini judge). Backing artifact: [`results/locomo_k100_full/cortex-v0_locomo_results.json`](results/locomo_k100_full/cortex-v0_locomo_results.json).
- **Positioning:** raw #2 on LongMemEval (behind Mastra 0.949), raw #3 on LoCoMo. What is distinct is the cost side — Cortex is the only memory system publishing $/question, and the only one at 0.9+ on LongMemEval_S with a cost-efficient-tier reader. No competitor $/question has been measured, so no cost comparison is claimed. $/q is computed from measured token counts at Google's confirmed list prices (see `cortex_bench/metrics.py`); the only code-labeled ESTIMATE left is the preview reader `gemini-3.1-pro-preview`. **Reader frontier** (same retrieval, full n=500): `gemini-2.5-flash` 0.848 @ ~$0.0079/q vs `gemini-3.5-flash` 0.932 @ ~$0.0396/q — +8.4 points for ~5× the reader cost. Same-judge caveat: our Gemini judge is likely more lenient than the canonical GPT-4o (a GPT-4o re-grade is pending an `OPENAI_API_KEY`); the headline run used the same model family as both reader and judge, so a self-preference bias cannot be excluded and a cross-family re-grade is planned.
- **Verified competitor leaderboard + deep-research verdict:** [`results/leaderboard_research.md`](results/leaderboard_research.md).
