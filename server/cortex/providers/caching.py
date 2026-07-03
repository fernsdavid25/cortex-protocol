"""On-disk embedding cache wrapper — a cost + speed lever for the benchmark matrix.

Wraps any ``LLMProvider`` so identical ``(embed_model, dim, text)`` embeddings are computed
ONCE and reused across runs and configs. LongMemEval re-embeds the same haystacks for every
config we try, so caching collapses the dominant embedding spend to a one-time cost — which
also mirrors the PRODUCT, where a haystack is embedded once at ingest and amortised over all
future queries.

``generate()`` is passed straight through (reader / extraction / distillation output must
reflect each config and is never cached). Only ``embed()`` is cached. Cache HITS contribute
0 to ``embed_tokens``, so a warm-cache run reports the honest MARGINAL (amortised) cost.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from array import array
from collections.abc import Sequence
from pathlib import Path

from .base import EmbedResult, GenResult, LLMProvider

_VAR_CHUNK = 500  # keep SQL "IN (...)" parameter counts well under SQLite's limit


def _key(model: str, dim: int, text: str) -> str:
    return hashlib.sha256(f"{model}\x1f{dim}\x1f{text}".encode()).hexdigest()


def _to_blob(vec: Sequence[float]) -> bytes:
    return array("f", vec).tobytes()


def _from_blob(blob: bytes) -> list[float]:
    a = array("f")
    a.frombytes(blob)
    return a.tolist()


class CachingProvider(LLMProvider):
    """Embedding-caching decorator around an inner provider (SQLite-backed, thread-safe)."""

    def __init__(self, inner: LLMProvider, cache_path: str) -> None:
        self.inner = inner
        self.cache_path = cache_path
        if cache_path != ":memory:":
            Path(cache_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(cache_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB NOT NULL)")
        self._conn.commit()
        # Surface the embed signature so downstream (model/dim guards) sees through the wrapper.
        self.embed_model = getattr(inner, "embed_model", "unknown")
        self.embed_dim = int(getattr(inner, "embed_dim", getattr(inner, "dim", 0)) or 0)
        self.reader_model = getattr(inner, "reader_model", None)

    def generate(
        self, prompt: str, *, temperature: float = 0.0, max_output_tokens: int = 512
    ) -> GenResult:
        return self.inner.generate(
            prompt, temperature=temperature, max_output_tokens=max_output_tokens
        )

    def embed(self, texts: Sequence[str]) -> EmbedResult:
        items = list(texts)
        if not items:
            return EmbedResult(vectors=[], input_tokens=0)
        model = getattr(self.inner, "embed_model", "unknown")
        dim = int(getattr(self.inner, "embed_dim", getattr(self.inner, "dim", 0)) or 0)
        keys = [_key(model, dim, t) for t in items]

        cached: dict[str, bytes] = {}
        with self._lock:
            for i in range(0, len(keys), _VAR_CHUNK):
                sub = keys[i : i + _VAR_CHUNK]
                placeholders = ",".join("?" * len(sub))
                for k, v in self._conn.execute(
                    f"SELECT k, v FROM emb WHERE k IN ({placeholders})", sub
                ):
                    cached[k] = v

        vectors: list[list[float]] = [[] for _ in items]
        miss_idx = [i for i, k in enumerate(keys) if k not in cached]
        for i, k in enumerate(keys):
            if k in cached:
                vectors[i] = _from_blob(cached[k])

        input_tokens = 0
        if miss_idx:
            res = self.inner.embed([items[i] for i in miss_idx])
            input_tokens = res.input_tokens
            with self._lock:
                for j, i in enumerate(miss_idx):
                    vec = res.vectors[j]
                    vectors[i] = vec
                    self._conn.execute(
                        "INSERT OR REPLACE INTO emb(k, v) VALUES (?, ?)", (keys[i], _to_blob(vec))
                    )
                self._conn.commit()
        return EmbedResult(vectors=vectors, input_tokens=input_tokens)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
