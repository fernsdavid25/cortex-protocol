# Agent-uplift eval — results

Does Cortex memory actually make an agent *better*, or is it just a retrieval benchmark? This eval
answers that directly: it runs the SAME multi-session tasks three ways and grades them
deterministically. Each task's answer depends on facts revealed in **earlier** sessions
(cross-session recall), so an agent with no memory of those sessions cannot succeed.

- **memoryless** — the agent sees only the final task (no history).
- **full_context** — the agent sees ALL prior sessions stuffed into the prompt (works, but the
  context grows without bound as history accumulates — it does not scale to decades).
- **cortex** — each session is `memorize()`d into a fresh per-scenario store; at task time the agent
  gets only the top-k `recall()`ed memories.

## Live result (`--provider gemini`, 9 scenarios, top_k=5, 2026-07-03)

```
arm             pass_rate   mean_input_tok  mean_latency_ms
-----------------------------------------------------------
memoryless           0.00             89.8          1372.20
full_context         1.00            168.6           913.57
cortex               1.00            142.6           996.28
```

**Reading it.** Memory is the difference between **0% and 100%** on multi-session tasks — a
memoryless agent fails every one because the needed fact was stated in a session it never sees.
**Cortex matches full-context accuracy (100%)** while feeding the agent only the *relevant* recalled
memories rather than the entire history. On these mostly-3-session scenarios the token gap is modest
(142.6 vs 168.6, ~15%) because top-k=5 already covers a 3-session history; the gap widens sharply
with history length — the `secret-store` scenario (one fact buried in **12** sessions) feeds cortex
5 recalled memories vs full_context's 12 (**125 vs 212 input tokens, 0.59 ratio**). That is the point:
`full_context` cost scales with total history, `cortex` cost stays bounded by top-k — so over a long
relationship cortex holds accuracy at a cost that does not blow up.

## Reproduce

```
uv run python bench/agent_uplift/harness.py --provider gemini      # live (all 9 scenarios)
uv run python bench/agent_uplift/harness.py --provider fake        # offline structure check
```

Scenarios (`scenarios.py`) mix fact-update recall, cross-session aggregation, retrieval-under-noise,
and coding-convention recall; each has a deterministic `check()`. Offline tests
(`tests/test_agent_uplift.py`) prove memoryless fails / full_context passes / cortex passes when
recall surfaces the fact, with no network.
