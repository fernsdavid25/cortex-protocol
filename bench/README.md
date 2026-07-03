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
- **Metrics report:** accuracy overall + per type + **abstention bucket**; **mean context tokens/query, $/question (ingest extract + read + judge), p50/p95 latency, tokens-per-correct-answer**; plus **retrieval recall@k** from `answer_session_ids`/`has_answer`.
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

## Current results
- **LongMemEval_S headline + methodology:** [`results/PHASE3_authoritative.md`](results/PHASE3_authoritative.md) — **0.932** (gemini-3.5-flash + top_k=50 + preference-mode + the A1 fix `--answer-first --max-output-tokens 8192`, ~$0.008/q; Gemini judge). Backing artifact: [`results/a1_af8k_full500/cortex-v0_s_results.json`](results/a1_af8k_full500/cortex-v0_s_results.json).
- **LoCoMo headline + methodology:** [`results/LOCOMO_results.md`](results/LOCOMO_results.md) — **0.813** (gemini-2.5-flash + top_k=100 + answer-first, ~$0.0034/q; Gemini judge). Backing artifact: [`results/locomo_k100_full/cortex-v0_locomo_results.json`](results/locomo_k100_full/cortex-v0_locomo_results.json).
- **Positioning:** **#1 on accuracy-per-dollar** — raw #2 on LongMemEval (behind Mastra 0.949), raw #3 on LoCoMo; **not raw SOTA**. Same-judge caveat: our Gemini judge is likely more lenient than the canonical GPT-4o (a GPT-4o re-grade is pending an `OPENAI_API_KEY`).
- **Verified competitor leaderboard + deep-research verdict:** [`results/leaderboard_research.md`](results/leaderboard_research.md).
