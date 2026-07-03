"""Hybrid retrieval: fuse dense (semantic) + lexical (BM25) rankings via RRF.

Dense recall and lexical recall fail in different ways — dense misses rare exact
tokens (names, IDs, dates), lexical misses paraphrase. Reciprocal Rank Fusion
(Cormack et al., 2009) combines them WITHOUT score normalization: each list votes
by RANK, so a chunk ranked highly by BOTH retrievers beats one that only one
retriever likes. ``k=60`` is the standard RRF constant from the original paper.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TypeVar

from cortex.store.memory_store import InMemoryStore, MemoryChunk

T = TypeVar("T", bound=Hashable)

RRF_K = 60


def reciprocal_rank_fusion(rankings: Sequence[Sequence[T]], k: int = RRF_K) -> dict[T, float]:
    """Fuse multiple ranked lists of ids into a combined RRF score per id.

    Each ranking contributes ``1 / (k + rank)`` for every id it contains (rank is
    1-based). Ids absent from a list simply contribute nothing from that list. The
    returned mapping is id -> fused score; higher is better.
    """
    scores: dict[T, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


def hybrid_retrieve(
    store: InMemoryStore,
    query: str,
    query_vec: Sequence[float],
    top_k: int,
) -> list[MemoryChunk]:
    """Run dense + lexical search and fuse them with RRF, returning top_k chunks.

    Each retriever is asked for a generous candidate pool (>= top_k) so the fusion
    has room to promote chunks that both retrievers agree on. Chunks are identified
    by their index in ``store.chunks`` (stable for a given store state), which keeps
    fusion correct even when two chunks share identical text.
    """
    if top_k <= 0 or len(store) == 0:
        return []

    pool = max(top_k * 4, top_k)
    chunks = store.chunks
    index_of: dict[int, int] = {id(c): i for i, c in enumerate(chunks)}

    dense = store.dense_search(query_vec, pool)
    lexical = store.lexical_search(query, pool)

    dense_ids = [index_of[id(c)] for c in dense]
    lexical_ids = [index_of[id(c)] for c in lexical]

    fused = reciprocal_rank_fusion([dense_ids, lexical_ids])
    if not fused:
        return []

    # Deterministic tie-break: higher fused score first, then earlier chunk index.
    ranked_ids = sorted(fused, key=lambda i: (-fused[i], i))
    return [chunks[i] for i in ranked_ids[:top_k]]
