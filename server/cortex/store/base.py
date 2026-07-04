"""The core store contract the engine programs against.

``CortexMemory`` composes a store rather than a concrete class, so the engine can run over the
persistent [`SQLiteStore`][cortex.store.sqlite_store] today and a hosted Postgres/pgvector store
later WITHOUT touching engine code. This module declares that seam as a ``typing.Protocol``:
[`MemoryStore`][cortex.store.base.MemoryStore] lists ONLY the methods the engine calls
unconditionally on every store.

Enrichment/optional capabilities (supersessions, episodic events, the entity graph, a store-side
hybrid ``search`` pushdown) are DELIBERATELY absent here — the engine reaches them behind
``hasattr``/``getattr`` guards, so a store may implement as few or as many of them as it likes.
Keeping the Protocol to the core surface makes the typing additive: any object structurally
satisfying these signatures is a valid store, and adding an optional capability never widens this
contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from cortex.store.memory_store import InMemoryStore
from cortex.store.sqlite_store import Memory


class MemoryStore(Protocol):
    """The core, always-present store surface ``CortexMemory`` depends on.

    Signatures mirror [`SQLiteStore`][cortex.store.sqlite_store.SQLiteStore] exactly. Optional
    enrichment methods are intentionally NOT declared — the engine accesses those behind runtime
    ``hasattr``/``getattr`` guards, so this Protocol stays the minimal contract every store must
    honour.
    """

    def ensure_embedding(self, model: str, dim: int) -> None:
        """Record the embedding (model, dim) on first use; reject a later mismatch."""
        ...

    def add(self, memory: Memory, embedding: Sequence[float]) -> None:
        """Insert one memory and its embedding."""
        ...

    def get(self, memory_id: str, user_id: str) -> Memory | None:
        """Return one memory by exact id (user-scoped), or ``None``."""
        ...

    def delete(self, memory_id: str, user_id: str) -> bool:
        """Delete one memory; return True if a row was removed."""
        ...

    def resolve_id_prefix(self, user_id: str, prefix: str, limit: int = 2) -> list[str]:
        """Return up to ``limit`` ids for this user whose id starts with ``prefix``."""
        ...

    def list_recent(self, user_id: str, limit: int = 20) -> list[Memory]:
        """Return the most recently added memories first."""
        ...

    def count(self, user_id: str) -> int:
        """Total memories stored for this user."""
        ...

    def build_index(self, user_id: str) -> tuple[InMemoryStore, dict[str, Memory]]:
        """Materialise a user's memories into an ``InMemoryStore`` for hybrid retrieval."""
        ...

    def close(self) -> None:
        """Close the underlying store (flush/release handles)."""
        ...
