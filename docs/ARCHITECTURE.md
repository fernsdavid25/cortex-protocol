# Cortex Architecture

A contributor-facing tour of the engine under [`server/cortex/`](../server/cortex/). It is
deliberately small: a provider abstraction, a persistent store, one benchmark-proven retrieval
pipeline, and a thin MCP surface. Three enrichment layers bolt on at **write time only**, so recall
stays byte-identical whether they are on or off.

## The pipeline at a glance

```
                 memorize(content)                         recall(query)
                        │                                       │
                        ▼                                       ▼
         provider.embed(content)  ── BYOK ──►  provider.embed(query)     [ providers/ ]
                        │                                       │
                        ▼                                       ▼
              SQLiteStore.add(memory, vec) ◄─ persist ─► SQLiteStore.build_index(user)
                        │                                       │      [ store/ ]
       (opt-in, write-time only)                                ▼
        L4 episodic · G2 graph · L5           hybrid_retrieve: dense ⊕ BM25 → RRF
        anti-saturation enrichment                             │      [ retrieve/ ]
                                                                ▼
                                              ranked raw memories → the agent reasons
                                              (or the Chain-of-Note reader, bench path)
                                                                       [ reader/ ]
```

`memorize` does **one embed** then persists. `recall` does **one embed** then a pure retrieval — **no
server-side generation**. That is what makes self-hosting effectively free: the only runtime cost a
self-hoster pays is their own embedding tokens against their own key, and nothing phones home.

## Modules

| Path | Role |
|---|---|
| [`memory.py`](../server/cortex/memory.py) | `CortexMemory` — the engine that composes everything: `memorize` / `recall` / `list_memories` / `forget` / `timeline` / `count`. |
| [`providers/`](../server/cortex/providers/) | The `LLMProvider` abstraction (BYOK) and its adapters. |
| [`store/`](../server/cortex/store/) | `SQLiteStore` (the persistent product store) and `InMemoryStore` (the dense + BM25 store the retriever runs over). |
| [`retrieve/`](../server/cortex/retrieve/) | `hybrid_retrieve` — dense + BM25 fused with Reciprocal Rank Fusion. |
| [`reader/`](../server/cortex/reader/) | Chain-of-Note reader prompts + the write-time extraction / arbiter prompts. |
| [`mcp/server.py`](../server/cortex/mcp/server.py) | The stdio MCP server: the five agent-facing tools. |
| [`profiles.py`](../server/cortex/profiles.py) | `CORTEX_TIER` profiles (cheap vs flagship) — the single source of truth for reader/retrieval knobs. |

## The provider abstraction (BYOK)

Everything programs against [`providers/base.py`](../server/cortex/providers/base.py):

```python
class LLMProvider(ABC):
    def generate(self, prompt, *, temperature=0.0, max_output_tokens=512) -> GenResult: ...
    def embed(self, texts: list[str]) -> EmbedResult: ...
```

`GenResult` and `EmbedResult` also carry token counts for cost accounting. Three adapters ship:

- **`GeminiProvider`** — the real embed/generate adapter. The `google-genai` SDK is **lazy-imported
  inside the methods**, so importing the engine (and running the offline tests) needs no SDK and no
  key.
- **`FakeProvider`** — deterministic, offline. Every test uses it; **no test ever calls a live
  model.**
- **`caching.py`** — a wrapper that memoizes embeddings/generations (used by the benchmark harness).

**BYOK:** credentials are read only from the environment (`GEMINI_API_KEY` / `GOOGLE_API_KEY`);
nothing is bundled, defaulted, or hardcoded. Claude/OpenAI/Ollama adapters can drop in behind the
same interface later.

## The store

There are two stores with distinct jobs:

- **`InMemoryStore`** ([`store/memory_store.py`](../server/cortex/store/memory_store.py)) — a
  pure-Python store of `MemoryChunk` records with two retrievers: `dense_search` (cosine over
  precomputed embeddings) and `lexical_search` (Okapi BM25, `k1=1.5`, `b=0.75`, corpus stats rebuilt
  lazily after a mutation). This is what the fusion runs over.
- **`SQLiteStore`** ([`store/sqlite_store.py`](../server/cortex/store/sqlite_store.py)) — the
  **persistent, per-user product store**: one file at `~/.cortex/memory.db`, WAL mode, vectors packed
  as float32 blobs, zero external services (no Docker, no DB server). Tables: `memories`, `meta`
  (schema version + the embedding signature guard), plus the enrichment tables `events`,
  `supersessions`, `entities`, `entity_edges`, `memory_entities`.

At query time `SQLiteStore.build_index(user_id)` materializes that user's rows into a fresh
`InMemoryStore` and hands it to the retriever. Brute-force cosine is ample at personal scale
(thousands of memories); an ANN index (pgvector / HNSW) is a later, hosted-scale concern
(see [ROADMAP.md](../ROADMAP.md) and `docs/L6_Decades_Scale.md`). The engine also probes for an
optional `store.search(...)` pushdown (e.g. a Postgres dense `<=>` + FTS fusion); SQLite has none, so
it takes the `build_index` path.

**Safety properties:** all SQL is parameterized; an embedding-signature guard (model + dim) is
enforced lazily on `memorize`/`recall` so a mismatch never bricks the whole store; short-id deletion
refuses ambiguous prefixes; a non-empty, correct-dimension embedding is required before any write.

## Hybrid retrieval (dense + BM25, RRF)

[`retrieve/hybrid.py`](../server/cortex/retrieve/hybrid.py) is the accuracy core. Dense and lexical
recall fail in *different* ways — dense misses rare exact tokens (names, IDs, dates); BM25 misses
paraphrase — so `hybrid_retrieve` asks each for a candidate pool (`max(top_k * 4, top_k)`) and fuses
their rankings with **Reciprocal Rank Fusion** (Cormack et al., 2009):

```
score(id) = Σ_over_lists  1 / (k + rank_in_list)      # k = RRF_K = 60
```

RRF fuses by **rank, not score**, so it needs no score normalization: a memory ranked highly by
*both* retrievers beats one only a single retriever likes. Ties break deterministically (higher
fused score, then earlier chunk index). Deep top-k drives retrieval recall@k ≈ 1.0 on the benchmarks.

## The reader (benchmark path)

[`reader/reader.py`](../server/cortex/reader/reader.py) builds a **Chain-of-Note** prompt: the model
notes the specific supporting facts from the retrieved memories, then answers from those notes, with
**calibrated abstention** — it replies with the exact `ABSTAIN_SENTINEL` (`"I don't know"`) only when
the memories genuinely lack the answer, rather than hallucinating. An answer-first ordering keeps a
truncated output budget from ever eating the answer.

Note the reader is **not** on the product `recall` path — `recall` returns raw ranked memories and
the client agent reasons over them. The reader is used by the benchmark harness (and mirrors what a
client does). This same module also houses the write-time extraction and arbiter prompts/parsers the
enrichment layers below depend on (`build_episodic_extraction_prompt`, `parse_graph_extraction`,
`build_supersession_arbiter_prompt`, …).

## Additive layers — opt-in, recall byte-identical

All three layers are **off by default** and act at **write time only**. With every flag off (or no
`extractor` configured), `memorize` is exactly one embed + persist and `recall` is exactly one embed
+ hybrid retrieve — no extra model calls, and recall never even queries the enrichment tables. That
byte-identical guarantee is what preserves the accuracy-per-dollar headline numbers.

- **Episodic memory (L4)** — `use_episodic` + an `extractor`. One cheap extraction per `memorize`
  structures `event_time` / `actor` / `location` / `event_type` into the `events` table, powering
  `CortexMemory.timeline(...)`.
- **Entity graph (G2)** — `use_graph`. Folded into the **same** extraction call as episodic (so
  enabling both still spends a single aux call). It upserts entities and labeled, directed
  relationships into an ego knowledge graph rooted at a synthetic `self` node (`ensure_self_entity`,
  `upsert_entity`, `add_entity_edge`, `link_memory_entity`), and links each memory to its subject and
  mentioned entities — powering the `recall_about` dossier read.
- **Anti-saturation (L5)** — `use_dedup` (embedding-only, `dedup_threshold = 0.95`, **no LLM**) drops
  near-identical rewrites to bound store growth; `use_soft_update` spends **one** cheap arbiter call,
  only in the "related but not duplicate" band (cosine `0.83`–`0.95`), to supersede a stale fact via
  the `supersessions` table so recall returns only the latest value.

There is also an injected, optional **cross-encoder reranker** (the `Reranker` Protocol in
`memory.py`): when set, `recall` over-retrieves a deeper pool and asks the reranker to select the top
`k`; any failure degrades gracefully to the RRF top-`k`, so recall can never fail because of
reranking. The engine never imports the Vertex SDK — it duck-types the injected object.

## The MCP tool surface

[`mcp/server.py`](../server/cortex/mcp/server.py) wraps the engine as a
[FastMCP](https://github.com/jlowin/fastmcp) **stdio** server (`uvx --from cortex-protocol cortex-mcp`, or
`python -m cortex.mcp.server` from a checkout). The engine is built lazily from environment config on
first tool call, so importing the module needs no API key. Six tools are exposed:

| Tool | What it does |
|---|---|
| `memorize(content, kind?, tags?)` | Embed + persist one durable memory (optionally one of six `kind`s and short `tags`). |
| `recall(query, limit?)` | Hybrid dense + BM25 (RRF) retrieval; returns ranked raw memories, **no LLM generation**. |
| `recall_about(entity, limit?)` | Exhaustive per-entity dossier from the entity graph — a pure keyed read, no embed/LLM call. **Requires `CORTEX_GRAPH=1`.** |
| `recall_timeline(since?, until?, limit?)` | Dated events in chronological order from the episodic layer — a keyed read, no LLM call. **Requires `CORTEX_EPISODIC=1`.** |
| `list_memories(limit?)` | The most recently saved memories, newest first. |
| `forget(memory_id)` | Delete by full id or an unambiguous short-id prefix (ambiguous prefixes are refused). |

Agent-supplied `limit`s are clamped (`_MAX_LIMIT = 1000`) so a runaway request can't materialize the
whole store. Configuration is all environment-driven (`CORTEX_DB_PATH`, `CORTEX_USER_ID`,
`CORTEX_EMBED_MODEL`, `CORTEX_EMBED_DIM`, `CORTEX_TOP_K`); see [`.env.example`](../.env.example).

## Where to start reading

`memory.py` first — it is the spine that composes the provider, the store, and the retriever, and
gates every optional layer. Then `retrieve/hybrid.py` for the accuracy core, and
`store/sqlite_store.py` for the persistence and enrichment schema.
