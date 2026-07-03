# L6 — Decades-Scale Retrieval

How Cortex recall scales from a personal SQLite store (thousands of memories) to a
decades-long, multi-user hosted store (10^5–10^7+ memories). This is a design + measurement
note; it changes no engine behaviour. Two harnesses accompany it: `bench/scale/latency_harness.py`
(offline — the SQLite ceiling, measured below) and `bench/scale/pgvector_latency.py` (the
ready-to-run Cloud SQL HNSW harness — operator-gated, see §5).

## 1. The ceiling of the current in-memory path

The self-host store is `SQLiteStore`. It has no store-side `search`, so `CortexMemory.recall`
(`server/cortex/memory.py`) takes the in-memory branch: `build_index(user_id)` then
`hybrid_retrieve`. `SQLiteStore.build_index` (`server/cortex/store/sqlite_store.py`) runs
`SELECT ... WHERE user_id = ? ORDER BY seq` with **no limit** — it loads *every* one of a user's
rows, unpacks each float32 blob, and materialises a fresh `InMemoryStore` **on every recall**.
`hybrid_retrieve` then scans all of it: `dense_search` computes a cosine against every chunk and
`lexical_search` rebuilds BM25 statistics across every document (`server/cortex/store/memory_store.py`).
So each recall is **O(n)** in both I/O (blob unpack) and CPU (n×dim cosine + n-doc BM25), with no
caching between calls.

Harness #1 (`FakeProvider`, dim=768, 30 queries/size, one user) makes the trend concrete:

```
# in-memory recall latency  (dim=768, queries=30, top_k=5, seed=1234)
       N    build_ms      p50_ms      p95_ms      rows
------------------------------------------------------
    1000        17.9        95.1       100.5      1000
    5000        91.3       507.4       547.2      5000
   20000       416.5      2364.0      2562.3     20000
   50000      1234.9      7737.7      9700.3     50000
------------------------------------------------------
```

`rows == N` every time — the whole store is loaded and scanned per query — and p50/p95 climb
roughly linearly with N (a 50× store is ~80× slower). This is fine at personal scale (~95 ms at
1k) but falls over well before decades scale: at 50k a single recall already costs ~7.7 s (p50) /
~9.7 s (p95), and 100k/1M are simply not viable on this path. Note `build_ms` — the full-store load
`recall` repeats *on every query* — is itself ~1.2 s at 50k.

## 2. The pgvector HNSW path (O(log n))

The fix is to push retrieval **into** the database with pgvector's HNSW index — the store used by
the **hosted production deployment** and the documented scale-out for self-host (Goal.md I8; this
OSS slice currently ships the SQLite path above, and porting the pgvector store to self-host is on
the roadmap). That `PostgresStore` builds `CREATE INDEX idx_memories_hnsw ON memories USING hnsw
(embedding vector_cosine_ops)`, and `search()` issues dense `ORDER BY embedding <=> %s::vector LIMIT
pool` (HNSW-indexed) plus lexical FTS (`GIN(content_tsv)`, `ts_rank_cd`), fused by the **same**
RRF(k=60). Only a small candidate pool crosses the wire — the full store is never loaded. HNSW is a
navigable small-world graph, so a search visits ~**O(log n)** nodes instead of all n, which is why
latency is expected to stay roughly flat as n grows. Harness #2 (`bench/scale/pgvector_latency.py`)
measures exactly this at 100k/1M against a live Cloud SQL instance — an **operator-gated** run (see
§5), not yet executed here, so the flat-latency figure is HNSW's designed behaviour pending that run.

**Recommended HNSW parameters** (starting points):

- **`m = 16`** (build) — graph connectivity. Higher m lifts recall but enlarges the index and
  slows build. 16 is a solid default; go 24–32 for recall-critical workloads.
- **`ef_construction = 64`** (build) — candidate breadth while building. Higher = better graph
  quality, slower build. 64 default; 128–200 when you can afford the build time.
- **`hnsw.ef_search = 100`** (runtime) — candidate breadth at query time; the recall⇄latency dial.
  pgvector defaults to 40; 100 is a better hosted starting point.

The recall/latency tradeoff lives almost entirely in `ef_search`: raise it until recall@k ≥ 0.95
(measure with harness #2), then raise `m` if you are still short. `pg_store`'s index is created
with pgvector's defaults (m=16, ef_construction=64), so the recommended *build* params already
match — **only `ef_search` needs setting**: `SET hnsw.ef_search = 100;` per session, or
`ALTER DATABASE cortex SET hnsw.ef_search = 100;` as a default. `m`/`ef_construction` are baked at
build time, so changing them means `DROP INDEX` + `CREATE INDEX ... WITH (m=..., ef_construction=...)`
(a `REINDEX` rebuilds with the current settings); `ef_search` needs no rebuild.

## 3. Cold-tier / archival

HNSW build time and RAM working set grow with row count, so keep the *hot* index small. Add a
`cold boolean NOT NULL DEFAULT false` column plus a `last_recalled_at` timestamp (bumped on recall
hits), and build a **partial** HNSW index `WHERE cold = false`. A nightly job flips `cold = true`
for memories not recalled in, say, 12 months. Recall stays hot-only by default
(`WHERE cold = false`); an optional `include_cold` flag runs a second, capped search over the cold
tier — a smaller/cheaper index, native `PARTITION BY LIST (cold)`, or even an `ivfflat` or seq
scan — and RRF-merges the two. This keeps p95 low for the common case while preserving full
decades-scale recall on demand.

## 4. AlloyDB + ScaNN, and when to switch

AlloyDB (Postgres-compatible) ships the **ScaNN** index — a drop-in at the SQL layer: same table,
same `<=>` operator, swap `USING hnsw` for `USING scann`. ScaNN offers lower memory, faster builds,
and strong recall at very large scale. Rough thresholds:

- **SQLite → pgvector:** migrate a user off the self-host in-memory path once their store passes
  ~10k–20k memories, or recall p95 exceeds ~200–300 ms, or durability across Cloud Run `/tmp`
  wipes is needed. Hosted multi-user deployments should start on pgvector.
- **pgvector HNSW → AlloyDB ScaNN:** pgvector/HNSW on Cloud SQL is comfortable to ~1–5M vectors per
  instance. Beyond ~5–10M (or when HNSW index RAM/build time bites), move to AlloyDB ScaNN; past
  ~50–100M, add partitioning and read replicas.

## 5. What needs the human

The 100k/1M pgvector numbers require the Cloud SQL instance — they cannot be produced offline.
`bench/scale/pgvector_latency.py` is ready: point it at the instance via `CORTEX_PG_CONNINFO` (or
`--conninfo`) behind the Cloud SQL Auth Proxy and run it. It measures recall p50/p95 at 100k and 1M
and reports recall@k (HNSW vs exact scan), so we can confirm latency stays flat and pick the
`ef_search` that holds recall@k ≥ 0.95.
