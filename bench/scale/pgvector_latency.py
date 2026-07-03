"""pgvector / HNSW recall-latency + recall@k harness for the decades-scale (Cloud SQL) path.

DO NOT run this in CI or offline — it REQUIRES a live Postgres+pgvector instance (Cloud SQL).
It is the ready-to-run companion to ``bench/scale/latency_harness.py``: where that quantifies the
O(n) ceiling of the SQLite in-memory path, this quantifies the O(log n) HNSW pushdown at
100k / 1M vectors and checks that the approximate index still returns the right neighbours.

Embeddings are still deterministic + offline (:class:`FakeProvider`) — no Gemini call. Only the
STORE is real. Nothing here phones home.

How to run against Cloud SQL
----------------------------
1. Provision a Cloud SQL for PostgreSQL instance and enable pgvector:
       CREATE EXTENSION IF NOT EXISTS vector;   -- the harness also does this
2. Open a local tunnel with the Cloud SQL Auth Proxy (recommended — no public IP):
       cloud-sql-proxy <PROJECT>:<REGION>:<INSTANCE> --port 5432 &
3. Point the harness at it via env or flag (psycopg conninfo / URL):
       export CORTEX_PG_CONNINFO="host=127.0.0.1 port=5432 dbname=cortex user=postgres password=***"
       uv run --with '.[postgres]' python bench/scale/pgvector_latency.py

   Smaller smoke run + explicit HNSW params:
       uv run --with '.[postgres]' python bench/scale/pgvector_latency.py \
           --conninfo "$CORTEX_PG_CONNINFO" --sizes 100000 \
           --m 16 --ef-construction 64 --ef-search 100

WARNING: with ``--truncate`` (default) the harness OWNS the ``memories`` table and TRUNCATEs it.
Point it at a scratch database, never a database holding real user memories.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from collections.abc import Sequence
from pathlib import Path

# Make ``import cortex`` resolve when launched as a plain script from the repo root.
_SERVER = Path(__file__).resolve().parents[2] / "server"
if _SERVER.is_dir() and str(_SERVER) not in sys.path:
    sys.path.insert(0, str(_SERVER))

from cortex.providers.fake import FakeProvider  # noqa: E402  (path shim must run first)
from cortex.retrieve.hybrid import RRF_K, reciprocal_rank_fusion  # noqa: E402

_VOCAB = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november "
    "oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu project budget "
    "meeting travel recipe doctor flight invoice password birthday address phone contract launch"
).split()

_DEFAULT_SIZES = (100_000, 1_000_000)


def _sentence(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(_VOCAB) for _ in range(n))


def _vec_literal(vec: Sequence[float]) -> str:
    """pgvector text literal ``[a,b,c]`` (the format the hosted PostgresStore uses)."""
    return "[" + ",".join(str(float(x)) for x in vec) + "]"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[int(rank)]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS memories (
    seq         bigserial PRIMARY KEY,
    id          text NOT NULL UNIQUE,
    user_id     text NOT NULL,
    content     text NOT NULL,
    metadata    jsonb NOT NULL DEFAULT '{}',
    created_at  text NOT NULL,
    embedding   vector(%(dim)s) NOT NULL,
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);
CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_tsv ON memories USING GIN(content_tsv);
"""


def _ensure_schema(conn: object, dim: int, truncate: bool) -> None:
    conn.execute(_SCHEMA % {"dim": int(dim)})  # type: ignore[attr-defined]
    if truncate:
        conn.execute("TRUNCATE memories")  # type: ignore[attr-defined]


def _bulk_load(
    conn: object,
    provider: FakeProvider,
    user_id: str,
    start: int,
    count: int,
    tokens: int,
    seed: int,
) -> None:
    """COPY ``count`` synthetic rows (ids ``start..start+count``) into ``memories``."""
    rng = random.Random(seed)
    copy_sql = "COPY memories (id, user_id, content, metadata, created_at, embedding) FROM STDIN"
    with conn.cursor() as cur:  # type: ignore[attr-defined]
        with cur.copy(copy_sql) as copy:
            for i in range(start, start + count):
                text = _sentence(rng, tokens)
                vec = provider.embed([text]).vectors[0]
                copy.write_row(
                    (
                        f"m{i:09d}",
                        user_id,
                        text,
                        "{}",
                        "2026-01-01T00:00:00+00:00",
                        _vec_literal(vec),
                    )
                )


def _build_hnsw(conn: object, m: int, ef_construction: int) -> float:
    """Drop + rebuild the HNSW index with the given params; return build seconds.

    Building AFTER bulk load (rather than incrementally) is the recommended pattern at scale and
    lets us sweep ``m`` / ``ef_construction``. Uses the SAME opclass as ``pg_store``:
    ``hnsw (embedding vector_cosine_ops)``.
    """
    t0 = time.perf_counter()
    conn.execute("DROP INDEX IF EXISTS idx_memories_hnsw")  # type: ignore[attr-defined]
    conn.execute(  # type: ignore[attr-defined]
        "CREATE INDEX idx_memories_hnsw ON memories "
        "USING hnsw (embedding vector_cosine_ops) "
        f"WITH (m = {int(m)}, ef_construction = {int(ef_construction)})"
    )
    conn.execute("ANALYZE memories")  # type: ignore[attr-defined]
    return time.perf_counter() - t0


def _dense_ids(cur: object, user_id: str, vlit: str, k: int) -> list[str]:
    """Approximate dense top-k via the HNSW index (honours the session's hnsw.ef_search)."""
    rows = cur.execute(  # type: ignore[attr-defined]
        "SELECT id FROM memories WHERE user_id = %s ORDER BY embedding <=> %s::vector LIMIT %s",
        (user_id, vlit, k),
    ).fetchall()
    return [r[0] for r in rows]


def _lexical_ids(cur: object, user_id: str, query: str, k: int) -> list[str]:
    rows = cur.execute(  # type: ignore[attr-defined]
        "SELECT id FROM memories WHERE user_id = %s "
        "AND content_tsv @@ websearch_to_tsquery('english', %s) "
        "ORDER BY ts_rank_cd(content_tsv, websearch_to_tsquery('english', %s)) DESC LIMIT %s",
        (user_id, query, query, k),
    ).fetchall()
    return [r[0] for r in rows]


def _hybrid_top_ids(conn: object, user_id: str, query: str, vlit: str, top_k: int) -> list[str]:
    """Replicate ``PostgresStore.search`` id-ranking (dense+FTS pool, RRF) to time the real path."""
    pool = max(top_k * 4, top_k)
    with conn.cursor() as cur:  # type: ignore[attr-defined]
        dense = _dense_ids(cur, user_id, vlit, pool)
        lexical = _lexical_ids(cur, user_id, query, pool)
    fused = reciprocal_rank_fusion([dense, lexical], k=RRF_K)
    if not fused:
        return []
    dense_rank = {mid: i for i, mid in enumerate(dense)}
    ranked = sorted(fused, key=lambda mid: (-fused[mid], dense_rank.get(mid, 1 << 30), mid))
    return ranked[:top_k]


def _exact_dense_ids(conn: object, user_id: str, vlit: str, k: int) -> list[str]:
    """Exact dense top-k by forcing a sequential scan (index off) — the recall@k ground truth."""
    with conn.transaction():  # type: ignore[attr-defined]
        conn.execute("SET LOCAL enable_indexscan = off")  # type: ignore[attr-defined]
        conn.execute("SET LOCAL enable_bitmapscan = off")  # type: ignore[attr-defined]
        with conn.cursor() as cur:  # type: ignore[attr-defined]
            return _dense_ids(cur, user_id, vlit, k)


def _measure(conn: object, provider: FakeProvider, args: argparse.Namespace) -> tuple[float, float]:
    """Return ``(p50_ms, p95_ms)`` for the hybrid recall path over ``args.queries`` queries."""
    conn.execute(f"SET hnsw.ef_search = {int(args.ef_search)}")  # type: ignore[attr-defined]
    qrng = random.Random(args.seed + 7)
    latencies: list[float] = []
    for _ in range(args.queries):
        text = _sentence(qrng, max(3, args.tokens // 2))
        vlit = _vec_literal(provider.embed([text]).vectors[0])
        t0 = time.perf_counter()
        _hybrid_top_ids(conn, args.user_id, text, vlit, args.top_k)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return _percentile(latencies, 50), _percentile(latencies, 95)


def _recall_at_k(conn: object, provider: FakeProvider, args: argparse.Namespace) -> float:
    """Mean dense recall@k of HNSW (approx) vs an exact seq scan over ``args.sample`` queries."""
    conn.execute(f"SET hnsw.ef_search = {int(args.ef_search)}")  # type: ignore[attr-defined]
    srng = random.Random(args.seed + 99)
    scores: list[float] = []
    for _ in range(args.sample):
        text = _sentence(srng, max(3, args.tokens // 2))
        vlit = _vec_literal(provider.embed([text]).vectors[0])
        with conn.cursor() as cur:  # type: ignore[attr-defined]
            approx = set(_dense_ids(cur, args.user_id, vlit, args.top_k))
        exact = set(_exact_dense_ids(conn, args.user_id, vlit, args.top_k))
        if exact:
            scores.append(len(approx & exact) / len(exact))
    return sum(scores) / len(scores) if scores else 0.0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else None)
    parser.add_argument(
        "--conninfo",
        default=os.environ.get("CORTEX_PG_CONNINFO", ""),
        help="psycopg conninfo/URL (default: $CORTEX_PG_CONNINFO).",
    )
    parser.add_argument(
        "--sizes",
        default="100000,1000000",
        help="Comma-separated vector counts M (default: 100000,1000000).",
    )
    parser.add_argument("--queries", type=int, default=30, help="Latency queries per M (30).")
    parser.add_argument("--sample", type=int, default=20, help="recall@k sample queries (20).")
    parser.add_argument("--dim", type=int, default=768, help="Embedding width (768).")
    parser.add_argument("--top-k", type=int, default=5, help="recall() top_k (5).")
    parser.add_argument("--tokens", type=int, default=12, help="Tokens per synthetic memory (12).")
    parser.add_argument("--m", type=int, default=16, help="HNSW m (16).")
    parser.add_argument(
        "--ef-construction", type=int, default=64, help="HNSW ef_construction (64)."
    )
    parser.add_argument("--ef-search", type=int, default=100, help="Runtime hnsw.ef_search (100).")
    parser.add_argument("--user-id", default="bench", help="user_id for the synthetic rows.")
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed (1234).")
    parser.add_argument(
        "--no-truncate",
        dest="truncate",
        action="store_false",
        help="Do NOT truncate the memories table first (append instead).",
    )
    parser.set_defaults(truncate=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.conninfo.strip():
        sys.stderr.write(
            "ERROR: no Postgres conninfo. This harness needs a live pgvector instance "
            "(Cloud SQL);\n"
            "it is NOT offline. Set CORTEX_PG_CONNINFO or pass --conninfo, e.g.:\n\n"
            '  export CORTEX_PG_CONNINFO="host=127.0.0.1 port=5432 dbname=cortex '
            'user=postgres password=***"\n'
            "  uv run --with '.[postgres]' python bench/scale/pgvector_latency.py\n\n"
            "Start a tunnel first with the Cloud SQL Auth Proxy (see this file's docstring).\n"
        )
        return 2

    try:
        import psycopg  # lazy: only a real pgvector run needs the driver
    except ModuleNotFoundError:
        sys.stderr.write(
            "ERROR: psycopg not installed. Run with the postgres extra:\n"
            "  uv run --with '.[postgres]' python bench/scale/pgvector_latency.py\n"
        )
        return 2

    sizes = sorted({int(p) for p in args.sizes.split(",") if p.strip()})
    provider = FakeProvider(dim=args.dim)

    print(
        f"# pgvector HNSW recall  (dim={args.dim}, m={args.m}, "
        f"ef_construction={args.ef_construction}, ef_search={args.ef_search}, top_k={args.top_k})"
    )
    header = (
        f"{'M':>10}  {'load_s':>8}  {'index_s':>8}  {'p50_ms':>8}  {'p95_ms':>8}  {'recall@k':>9}"
    )
    print(header)
    print("-" * len(header))

    with psycopg.connect(args.conninfo, autocommit=True) as conn:
        _ensure_schema(conn, args.dim, args.truncate)
        loaded = 0
        for target in sizes:
            t0 = time.perf_counter()
            _bulk_load(
                conn, provider, args.user_id, loaded, target - loaded, args.tokens, args.seed
            )
            load_s = time.perf_counter() - t0
            loaded = target
            index_s = _build_hnsw(conn, args.m, args.ef_construction)
            p50, p95 = _measure(conn, provider, args)
            recall = _recall_at_k(conn, provider, args)
            print(
                f"{target:>10}  {load_s:>8.1f}  {index_s:>8.1f}  {p50:>8.1f}  {p95:>8.1f}  "
                f"{recall:>9.3f}"
            )

    print("-" * len(header))
    print(
        "p50/p95 should stay ~flat as M grows (HNSW is ~O(log n)); raise --ef-search to lift "
        "recall@k toward 1.0 at some latency cost."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
