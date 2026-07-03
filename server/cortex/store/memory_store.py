"""In-memory per-instance memory store (Phase 2 engine v0).

Each LongMemEval question is independent and its haystack is at most a few hundred
rounds, so an in-process pure-python store is ample — NO external DB, NO sqlite-vec,
NO Docker (pgvector persistence is a later product concern). The store holds
``MemoryChunk`` records and supports two complementary retrievers:

- ``dense_search``: cosine similarity over precomputed embeddings (semantic).
- ``lexical_search``: pure-python Okapi BM25 over chunk texts (exact-keyword).

Both return ranked ``MemoryChunk`` lists; ``cortex.retrieve.hybrid`` fuses them.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

_TOKEN = re.compile(r"[a-z0-9]+")

# Standard Okapi BM25 hyperparameters (Robertson & Zaragoza, 2009).
_BM25_K1 = 1.5
_BM25_B = 0.75


def _tokenize(text: str) -> list[str]:
    """Lowercase + split on word characters (matches the embedder's tokenizer)."""
    return _TOKEN.findall(text.lower())


@dataclass
class MemoryChunk:
    """One retrievable unit of memory: a conversation round and its provenance."""

    text: str
    session_id: str
    date: str
    embedding: list[float] = field(default_factory=list)


class InMemoryStore:
    """Pure-python store with dense (cosine) and lexical (BM25) retrieval.

    BM25 corpus statistics (document frequencies, average length) are recomputed
    lazily on the first lexical search after a mutation, so repeated queries over an
    unchanged store are cheap.
    """

    def __init__(self) -> None:
        self._chunks: list[MemoryChunk] = []
        # BM25 index, rebuilt lazily after add()/reset() via _ensure_index().
        self._doc_tokens: list[list[str]] = []
        self._doc_freqs: list[Counter[str]] = []
        self._df: Counter[str] = Counter()
        self._avgdl: float = 0.0
        self._index_dirty: bool = True

    @property
    def chunks(self) -> list[MemoryChunk]:
        return self._chunks

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: Sequence[MemoryChunk]) -> None:
        """Append chunks to the store and mark the lexical index stale."""
        self._chunks.extend(chunks)
        self._index_dirty = True

    def reset(self) -> None:
        """Clear all memory (each LongMemEval question is independent)."""
        self._chunks = []
        self._doc_tokens = []
        self._doc_freqs = []
        self._df = Counter()
        self._avgdl = 0.0
        self._index_dirty = True

    def dense_search(self, query_vec: Sequence[float], k: int) -> list[MemoryChunk]:
        """Return the top-k chunks by cosine similarity to ``query_vec``.

        Embeddings are assumed L2-normalized (as produced by the providers), so the
        cosine reduces to a dot product; we still divide by norms defensively in case
        a caller supplies raw vectors.
        """
        if k <= 0 or not self._chunks:
            return []
        scored = [
            (self._cosine(query_vec, chunk.embedding), idx, chunk)
            for idx, chunk in enumerate(self._chunks)
        ]
        # Sort by score desc, breaking ties by insertion order for determinism.
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [chunk for _score, _idx, chunk in scored[:k]]

    def lexical_search(self, query: str, k: int) -> list[MemoryChunk]:
        """Return the top-k chunks by Okapi BM25 score for ``query``."""
        if k <= 0 or not self._chunks:
            return []
        self._ensure_index()
        query_terms = _tokenize(query)
        n = len(self._chunks)
        scored: list[tuple[float, int, MemoryChunk]] = []
        for idx, chunk in enumerate(self._chunks):
            freqs = self._doc_freqs[idx]
            dl = len(self._doc_tokens[idx])
            score = 0.0
            for term in query_terms:
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                df = self._df.get(term, 0)
                # Okapi BM25 idf (with +1 to keep it non-negative for common terms).
                idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
                denom = tf + _BM25_K1 * (1.0 - _BM25_B + _BM25_B * dl / self._avgdl)
                score += idf * (tf * (_BM25_K1 + 1.0)) / denom
            scored.append((score, idx, chunk))
        # Keep only chunks with a positive score (at least one query term matched).
        scored = [t for t in scored if t[0] > 0.0]
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [chunk for _score, _idx, chunk in scored[:k]]

    def _ensure_index(self) -> None:
        """(Re)build BM25 corpus statistics if the store changed since last build."""
        if not self._index_dirty:
            return
        self._doc_tokens = [_tokenize(chunk.text) for chunk in self._chunks]
        self._doc_freqs = [Counter(tokens) for tokens in self._doc_tokens]
        self._df = Counter()
        for freqs in self._doc_freqs:
            self._df.update(freqs.keys())
        total = sum(len(tokens) for tokens in self._doc_tokens)
        self._avgdl = (total / len(self._doc_tokens)) if self._doc_tokens else 0.0
        self._index_dirty = False

    @staticmethod
    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
        """Cosine similarity; returns 0.0 if either vector is empty or zero."""
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)
