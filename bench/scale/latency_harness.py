"""Latency characterization of the CURRENT self-host recall path (SQLite + in-memory scan).

OFFLINE + deterministic: uses :memory: SQLite and :class:`FakeProvider` (deterministic fake
vectors, no network, no Gemini, no Postgres). This measures ONLY — it adds nothing to and
changes nothing in the engine.

What it shows
-------------
``CortexMemory.recall`` on the SQLite self-host store has no store-side ``search`` pushdown, so
every recall runs :meth:`SQLiteStore.build_index` (loads ALL of a user's rows into a fresh
``InMemoryStore``) and then ``hybrid_retrieve`` (dense cosine + BM25 over EVERY row). That is
O(n) per query in both work and allocation. As N grows the per-recall latency grows with it,
which is exactly why the decades-scale path must push retrieval into pgvector's HNSW index
(see ``bench/scale/pgvector_latency.py`` and ``docs/decades-scale.md``).

The table reports, per store size N:

- ``build_ms``    — time for one :meth:`SQLiteStore.build_index` (the per-recall full load).
- ``p50_ms``/``p95_ms`` — recall() latency percentiles over ``--queries`` fixed queries.
- ``rows``        — rows materialized + scanned per recall (== N; the O(n) evidence).

Run
---
    uv run python bench/scale/latency_harness.py

    # canonical decades-scale sweep (adds 100k; slow at dim=768 — minutes):
    uv run python bench/scale/latency_harness.py --sizes 1000,5000,20000,50000,100000

    # faster sweep at a truncated embedding width:
    uv run python bench/scale/latency_harness.py --dim 256 --queries 30
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

# Make ``import cortex`` work whether launched via ``uv run`` (installed package) or plain
# ``python bench/scale/latency_harness.py`` from the repo root (server/ not yet on the path).
_SERVER = Path(__file__).resolve().parents[2] / "server"
if _SERVER.is_dir() and str(_SERVER) not in sys.path:
    sys.path.insert(0, str(_SERVER))

from cortex.memory import CortexMemory  # noqa: E402  (path shim must run first)
from cortex.providers.fake import FakeProvider  # noqa: E402
from cortex.store.sqlite_store import SQLiteStore  # noqa: E402

# A small deterministic vocabulary; memories/queries are random word bags over it. Enough lexical
# overlap that BM25 does real work, while keeping tokenization cheap.
_VOCAB = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november "
    "oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu project budget "
    "meeting travel recipe doctor flight invoice password birthday address phone contract launch"
).split()

_DEFAULT_SIZES = (1000, 5000, 20000, 50000)


def _sentence(rng: random.Random, n: int) -> str:
    """A deterministic bag-of-words 'memory' or 'query' of ``n`` tokens."""
    return " ".join(rng.choice(_VOCAB) for _ in range(n))


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile (numpy-free) of ``values`` at ``pct`` in [0, 100]."""
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


def _parse_sizes(raw: str) -> list[int]:
    sizes = sorted({int(part) for part in raw.split(",") if part.strip()})
    if not sizes:
        raise argparse.ArgumentTypeError("--sizes must list at least one positive integer")
    return sizes


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Characterize CortexMemory.recall latency on the SQLite in-memory path.",
    )
    parser.add_argument(
        "--sizes",
        type=_parse_sizes,
        default=list(_DEFAULT_SIZES),
        help="Comma-separated store sizes N (default: 1000,5000,20000,50000; "
        "canonical decades sweep also includes 100000).",
    )
    parser.add_argument(
        "--queries", type=int, default=30, help="Recall queries measured per size (default: 30)."
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=768,
        help="Embedding width (default: 768, the product's Gemini/pgvector width).",
    )
    parser.add_argument("--top-k", type=int, default=5, help="recall() top_k (default: 5).")
    parser.add_argument(
        "--tokens", type=int, default=12, help="Tokens per synthetic memory (default: 12)."
    )
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed (default: 1234).")
    return parser.parse_args(argv)


def _measure_size(
    mem: CortexMemory, store: SQLiteStore, user_id: str, queries: list[str], top_k: int
) -> tuple[float, float, float, int]:
    """Return ``(build_ms, p50_ms, p95_ms, rows)`` for the current store contents."""
    # Isolate one build_index — the full-store load recall repeats on every single query.
    t0 = time.perf_counter()
    index, _by_id = store.build_index(user_id)
    build_ms = (time.perf_counter() - t0) * 1000.0
    rows = len(index)

    latencies: list[float] = []
    for query in queries:
        t0 = time.perf_counter()
        mem.recall(query, limit=top_k)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return build_ms, _percentile(latencies, 50), _percentile(latencies, 95), rows


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    user_id = "bench"
    provider = FakeProvider(dim=args.dim)
    store = SQLiteStore(":memory:")
    mem = CortexMemory(provider, store, user_id=user_id, top_k=args.top_k)

    # A fixed query set, reused at every size so the columns are comparable across N.
    qrng = random.Random(args.seed + 1)
    queries = [_sentence(qrng, max(3, args.tokens // 2)) for _ in range(args.queries)]

    print(
        f"# in-memory recall latency  (dim={args.dim}, queries={args.queries}, "
        f"top_k={args.top_k}, seed={args.seed})"
    )
    print(f"{'N':>8}  {'build_ms':>10}  {'p50_ms':>10}  {'p95_ms':>10}  {'rows':>8}")
    print("-" * 54)

    ingest_rng = random.Random(args.seed)
    inserted = 0
    started = time.perf_counter()
    for target in args.sizes:
        for _ in range(target - inserted):
            mem.memorize(_sentence(ingest_rng, args.tokens))
        inserted = target
        build_ms, p50, p95, rows = _measure_size(mem, store, user_id, queries, args.top_k)
        print(f"{target:>8}  {build_ms:>10.1f}  {p50:>10.1f}  {p95:>10.1f}  {rows:>8}")

    store.close()
    total = time.perf_counter() - started
    print("-" * 54)
    print(
        "rows == N confirms every recall loads + scans the whole store (O(n)); "
        f"p50/p95 grow with N. total wall={total:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
