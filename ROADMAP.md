# Cortex Roadmap — one core, scaled to a lifelong memory

Cortex aims to be the **default memory layer for AI agents**: one memory a user owns, that any
agent reads with consent, that still recalls what you said *years* later and across devices. This is
**one combined core, improved only by what benchmarks prove** — not a pile of specialized memory
systems.

## Principle

**Add a mechanism only when a benchmark shows it earns its cost.** Every change is gated on the
suite below. This already saved us from a reflection layer that *hurt* accuracy (−5 pts at ~2×
cost). Positioning is honest: **not raw SOTA — #1 on accuracy-per-dollar.** Cortex targets the best
accuracy *per dollar* (premium-tier retrieval quality with a cheap Gemini reader), with $/question
published, not raw accuracy at any cost.

## The measuring sticks

- **LongMemEval_S** (bounded long-term QA) — current headline **0.932** with a cheap
  `gemini-3.5-flash` reader (~$0.008/question), retrieval recall@k ≈ 1.0.
- **LoCoMo** (multi-hop / temporal / adversarial) — current headline **0.813** with
  `gemini-2.5-flash` (~$0.0034/question).
- **Agent uplift** (`bench/agent_uplift/`) — does memory make an agent *better*? Memoryless 0.00 vs
  Cortex 1.00 on multi-session tasks, at a cost bounded by top-k instead of growing with history.

See [`README.md`](README.md) and `bench/results/` for full methodology and per-type tables. All
Cortex numbers are graded by a Gemini judge (disclosed); the canonical LongMemEval judge is GPT-4o.

## Stages

### 1. Shipped — the core (this repo)

- **Local memory engine** (`CortexMemory`): chunk → embed → **hybrid retrieval (dense cosine +
  Okapi BM25, fused with Reciprocal Rank Fusion)** at deep top-k. `recall` does **no** server-side
  generation — it embeds the query once and returns the raw memories for the agent to reason over.
- **Local stdio MCP server** (`python -m cortex.mcp.server`): exposes `memorize`, `recall`,
  `recall_about`, `recall_timeline`, `list_memories`, and `forget` to any MCP client (Claude Code,
  Cursor, Claude Desktop, VS Code). Per-user SQLite, BYOK (Gemini), zero phone-home.
  (`recall_about`/`recall_timeline` are live with the opt-in `CORTEX_GRAPH`/`CORTEX_EPISODIC` layers.)
- **Persistent per-user store** (SQLite, WAL, float32-blob vectors) with an embedding-signature
  guard, bounded short-id deletion, and graceful shutdown.
- **Additive, opt-in enrichment layers** — each write-time only, so **recall stays byte-identical**
  and the accuracy-per-dollar guarantee holds:
  - **Episodic memory** — one cheap extraction per `memorize` structures event_time / actor /
    location / event_type, powering a timeline read.
  - **Entity graph + `recall_about`** — the same extraction folds entities and labeled,
    directed relationships into an ego knowledge graph rooted at a synthetic `self`, powering an
    exhaustive per-entity dossier.
  - **Anti-saturation** — embedding-only write-time dedup bounds store growth; a cheap
    contradiction arbiter supersedes stale facts so recall returns only the latest value.
- **Benchmark harness** (`bench/cortex_bench/`): LongMemEval + LoCoMo with per-type accuracy,
  abstention, recall@k, and per-question cost accounting; an offline embedding cache; provider
  retry/backoff.
- **Chain-of-Note reader** with calibrated abstention and a preference-aware mode (benchmark path).
- **`.mcpb` bundle** ([`packaging/mcpb/`](packaging/mcpb/)) for one-click Claude Desktop install.

### 2. Near-term — distribution and cross-device

- **PyPI publish** — `uvx --from cortex-protocol cortex-mcp` with no checkout, so any MCP client wires it in with one
  config snippet.
- **Remote streamable-HTTP server + OAuth ("Sign in with Cortex")** — the cross-device path: point
  ChatGPT, Claude, Cursor, and Claude Code at the *same* memory, authorized per agent with your
  consent. The local stdio server is the stepping stone; this is an infrastructure problem, not an
  accuracy one.
- **Hosted dashboard** — inspect, search, and manage your memory in a browser (proprietary; the
  engine here stays open source).

### 3. Scale — make the one core lifelong-capable

- **ANN vector index (pgvector / HNSW)** replacing brute-force cosine, so recall stays fast from
  thousands → millions of memories. (Today's flat rebuild-per-query is fine at personal scale;
  measured self-host SQLite recall is O(n) — see `docs/decades-scale.md`.)
- **Cross-encoder reranker** over the fused candidates — already an *injected, optional* engine hook
  (`Reranker`), gated on a benchmark before it becomes a default.

### 4. Lifelong memory management — evidence-gated biological ideas

At decade scale you can't retrieve usefully over raw turns, so these become candidates — each
shipped **only if it beats the current core on the suite**:

- **Consolidation / summarization** of old memory, so distant memory is dense, not a haystack.
- **Importance + recency ranking and archival** (prioritization / forgetting), required at scale.

## What we deliberately are NOT doing

Multiple parallel "memory protocols," a knowledge graph as the *primary* store, or nightly
consolidation/decay machinery **on faith** — the evidence says these add cost without beating a
well-engineered retrieval + reader core. They re-enter only through stage 4, gated on a benchmark.
