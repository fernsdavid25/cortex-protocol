"""Persistent, per-user memory store — the PRODUCT store.

The benchmark uses an in-memory per-instance store ([`InMemoryStore`][cortex.store.memory_store])
because each LongMemEval question is independent. The PRODUCT instead needs memory that
**persists across sessions** and is **scoped per user**, with ZERO external services
(Dockerless, no DB server) so self-hosters run at no cost (Goal.md I3/I4). SQLite — in the
Python stdlib — is the right substrate: one file at ``~/.cortex/memory.db``.

Vectors are stored as float32 blobs. Retrieval REUSES the benchmark's tested hybrid pipeline
(dense cosine + Okapi BM25, RRF-fused) by materialising a user's rows into an
``InMemoryStore`` at query time — brute-force cosine is ample at personal scale (thousands of
memories). sqlite-vec / pgvector are a hosted-scale concern (Phase 4 product), deliberately
NOT a dependency here.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from array import array
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from cortex.store.memory_store import InMemoryStore, MemoryChunk

SCHEMA_VERSION = 1


@dataclass
class Memory:
    """One persisted memory record and its provenance.

    ``score`` is populated only by recall results (relevance rank); it is ``None`` for
    stored/listed records.
    """

    id: str
    user_id: str
    content: str
    created_at: str
    metadata: dict[str, object] = field(default_factory=dict)
    score: float | None = None


def _to_blob(vec: Sequence[float]) -> bytes:
    """Pack an embedding into a compact float32 blob for storage."""
    return array("f", vec).tobytes()


def _from_blob(blob: bytes) -> list[float]:
    """Unpack a float32 blob back into a Python float list."""
    a = array("f")
    a.frombytes(blob)
    return a.tolist()


def _now_iso() -> str:
    """Current UTC time as a stable ISO-8601 string (seconds precision)."""
    return datetime.now(UTC).isoformat(timespec="seconds")


# G1 entity-graph vocabulary. The six node types an entity may carry; ``self`` is the synthetic
# ego root ("You"), created lazily per user. An unknown/invalid type falls back to "thing".
VALID_ENTITY_TYPES = frozenset({"self", "person", "place", "org", "project", "thing"})
# A type we may safely upgrade FROM to a more specific one (conservative; never downgrade a self).
_GENERIC_ENTITY_TYPES = frozenset({"", "thing"})
# The two roles a memory→entity link may carry; "subject" (the memory is ABOUT the entity) beats
# "mention" (the entity is merely referenced) on conflict.
VALID_ENTITY_ROLES = frozenset({"subject", "mention"})
# The reserved norm_name of a user's synthetic ``self`` entity — never collides with a real name
# because ``_norm_name`` keeps letters, so a literal "__self__" input would be required to clash.
_SELF_NORM = "__self__"
# The ego/first-person tokens that resolve to the synthetic ``self`` entity ("You") rather than a
# real entity row. SINGLE SOURCE OF TRUTH shared by the write path (memory.py collapses these to
# the self node so a stray "I"/"me"/"my" never mints a spurious node) and the read path
# (``get_entity_by_name`` resolves them to the self root, which is otherwise unreachable by name).
SELF_ALIASES = frozenset({"self", "you", "me", "i", "my", "mine", "myself"})


def _norm_name(name: str) -> str:
    """Normalise an entity name for dedup: lowercase, strip, collapse internal whitespace."""
    return " ".join(name.lower().split())


def _like_escape(text: str) -> str:
    r"""Escape LIKE wildcards so a name is matched literally (used with ``ESCAPE '\'``)."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    id         TEXT NOT NULL UNIQUE,
    user_id    TEXT NOT NULL,
    content    TEXT NOT NULL,
    metadata   TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    embedding  BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
-- L4 episodic companion table (additive; keyed 1:1 to a memory). ISO-8601 TEXT timestamps
-- are lexically sortable, so range/order queries need no date parsing. Written to ONLY when
-- the episodic flag is enabled; the hot recall path never reads it.
CREATE TABLE IF NOT EXISTS events (
    memory_id   TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    user_id     TEXT NOT NULL,
    event_time  TEXT,
    ingest_time TEXT NOT NULL,
    actor       TEXT,
    location    TEXT,
    event_type  TEXT,
    episode_id  TEXT,
    confidence  REAL
);
CREATE INDEX IF NOT EXISTS idx_events_user_time ON events(user_id, event_time);
CREATE INDEX IF NOT EXISTS idx_events_episode ON events(user_id, episode_id);
-- L5 anti-saturation companion table (additive; keyed 1:1 to the SUPERSEDED memory). A row means
-- "``superseded_id`` was replaced by a newer memory ``superseded_by``". Written to ONLY when
-- soft-update is enabled, and recall consults it ONLY when anti-saturation is active — the
-- byte-identical off-path never touches it. ON DELETE CASCADE cleans it when a memory is forgotten.
CREATE TABLE IF NOT EXISTS supersessions (
    superseded_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    superseded_by TEXT NOT NULL,
    at            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_supersessions_by ON supersessions(superseded_by);
-- G1 entity graph (additive; three tables). An ego-centric knowledge graph laid alongside the
-- memories: entity nodes deduped per user by norm_name, labeled directed edges between them, and
-- a memory↔entity link table. Written to ONLY when the write-time graph flag is on; the recall
-- path never reads these, so the byte-identical, graph-off path is unaffected. Edges/links carrying
-- a memory id ON DELETE CASCADE so forgetting a memory prunes its provenance (honesty guardrail).
CREATE TABLE IF NOT EXISTS entities (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    name       TEXT NOT NULL,
    norm_name  TEXT NOT NULL,
    type       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(user_id, norm_name)
);
CREATE INDEX IF NOT EXISTS idx_entities_user ON entities(user_id);
CREATE TABLE IF NOT EXISTS entity_edges (
    id               TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL,
    src_id           TEXT NOT NULL,
    label            TEXT NOT NULL,
    dst_id           TEXT NOT NULL,
    source_memory_id TEXT,
    created_at       TEXT NOT NULL,
    UNIQUE(user_id, src_id, label, dst_id),
    FOREIGN KEY(source_memory_id) REFERENCES memories(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_entity_edges_src ON entity_edges(user_id, src_id);
CREATE INDEX IF NOT EXISTS idx_entity_edges_dst ON entity_edges(user_id, dst_id);
CREATE TABLE IF NOT EXISTS memory_entities (
    memory_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    role      TEXT NOT NULL,
    user_id   TEXT NOT NULL,
    PRIMARY KEY(memory_id, entity_id),
    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY(entity_id) REFERENCES entities(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_memory_entities_entity ON memory_entities(user_id, entity_id);
"""


# Ordered schema migrations. Entry ``i`` (0-based) upgrades a store at ``PRAGMA user_version == i``
# to ``i + 1``; the ladder is applied inside a single transaction on connect and ``user_version`` is
# then stamped to ``SCHEMA_VERSION``. The base ``_SCHEMA`` above already materialises the current
# (v1) shape via ``CREATE TABLE IF NOT EXISTS``, so this list is empty today — but wiring the real
# upgrade path NOW means the first shipped ALTER just appends a callable here instead of scrambling
# to add "detect an old file and hand-patch it" logic after the fact.
_MIGRATIONS: tuple[Callable[[sqlite3.Connection], None], ...] = ()


class SQLiteStore:
    """A persistent, per-user memory store backed by a single SQLite file.

    Thread-safe: one connection guarded by a lock (FastMCP may dispatch tool calls from
    different threads). Pass ``":memory:"`` as ``db_path`` for an ephemeral store (tests).
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            # Normalise (expand ~, collapse ..) so the stored path is unambiguous, then ensure
            # the parent dir exists. CORTEX_DB_PATH is trusted operator config (never agent input).
            path = Path(self.db_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = str(path.resolve())
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        # WAL + a generous busy timeout so concurrent MCP tool calls (and a stray second
        # process / antivirus handle on Windows) don't hit "database is locked".
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        # Enforce the events->memories FK so ON DELETE CASCADE auto-cleans episodic rows when a
        # memory is forgotten. `memories`/`meta` carry no foreign keys, so this is a no-op for
        # every existing operation (the byte-identical, episodic-off path is unaffected).
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        """Bring an existing DB up to ``SCHEMA_VERSION`` via the ordered ``_MIGRATIONS`` ladder.

        Reads ``PRAGMA user_version`` (0 on a fresh file); if it is below ``SCHEMA_VERSION`` the
        pending migrations run inside ONE transaction and ``user_version`` is stamped forward, so an
        interrupted upgrade rolls back cleanly and never leaves a half-migrated file. A fresh DB
        (whose current shape ``_SCHEMA`` just created) simply stamps the version with no migrations
        to run.
        """
        with self._lock:
            version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
            if version >= SCHEMA_VERSION:
                return
            try:
                for migrate in _MIGRATIONS[version:SCHEMA_VERSION]:
                    migrate(self._conn)
                # PRAGMA user_version takes no bound parameters; SCHEMA_VERSION is a trusted int.
                self._conn.execute(f"PRAGMA user_version = {int(SCHEMA_VERSION)}")
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    # -- lifecycle -------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- embedding-signature guard --------------------------------------------

    def ensure_embedding(self, model: str, dim: int) -> None:
        """Record the embedding (model, dim) on first use; reject a later mismatch.

        Stored vectors are only comparable if produced by the same embedder. Switching
        model/dim would silently corrupt cosine similarity, so we fail loudly instead.
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT key, value FROM meta WHERE key IN ('embed_model', 'embed_dim')"
            )
            existing = {k: v for k, v in cur.fetchall()}
            if not existing:
                self._conn.executemany(
                    "INSERT INTO meta(key, value) VALUES (?, ?)",
                    [("embed_model", model), ("embed_dim", str(dim))],
                )
                self._conn.commit()
                return
            old_model = existing.get("embed_model")
            old_dim = existing.get("embed_dim")
            if old_model != model or old_dim != str(dim):
                raise ValueError(
                    f"Embedding mismatch for {self.db_path!r}: store was built with "
                    f"{old_model}/dim={old_dim} but the engine now uses {model}/dim={dim}. "
                    "Use a separate CORTEX_DB_PATH, or re-ingest from scratch."
                )

    # -- writes ----------------------------------------------------------------

    def add(self, memory: Memory, embedding: Sequence[float]) -> None:
        """Insert one memory and its embedding. Raises ``ValueError`` if the id already exists."""
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO memories(id, user_id, content, metadata, created_at, embedding) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        memory.id,
                        memory.user_id,
                        memory.content,
                        json.dumps(memory.metadata),
                        memory.created_at,
                        _to_blob(embedding),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"memory id {memory.id!r} already exists") from exc
            self._conn.commit()

    def delete(self, memory_id: str, user_id: str) -> bool:
        """Delete one memory; return True if a row was removed."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM memories WHERE id = ? AND user_id = ?", (memory_id, user_id)
            )
            self._conn.commit()
            return cur.rowcount > 0

    def resolve_id_prefix(self, user_id: str, prefix: str, limit: int = 2) -> list[str]:
        """Return up to ``limit`` ids for this user whose id starts with ``prefix`` (SQL-side).

        Bounded prefix match — no full-table scan. The caller treats >1 result as ambiguous and
        refuses to delete, preventing a short prefix from silently deleting the wrong memory.
        LIKE wildcards are stripped from ``prefix`` (ids are hex, so this is a no-op for real ids).
        """
        safe = prefix.replace("%", "").replace("_", "")
        with self._lock:
            cur = self._conn.execute(
                "SELECT id FROM memories WHERE user_id = ? AND id LIKE ? ORDER BY id LIMIT ?",
                (user_id, safe + "%", max(1, limit)),
            )
            return [r[0] for r in cur.fetchall()]

    # -- reads -----------------------------------------------------------------

    def get(self, memory_id: str, user_id: str) -> Memory | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, content, created_at, metadata FROM memories "
                "WHERE id = ? AND user_id = ?",
                (memory_id, user_id),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return Memory(
            id=row[0],
            user_id=user_id,
            content=row[1],
            created_at=row[2],
            metadata=json.loads(row[3]),
        )

    def list_recent(self, user_id: str, limit: int = 20) -> list[Memory]:
        """Return the most recently added memories first (insertion order)."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, content, created_at, metadata FROM memories "
                "WHERE user_id = ? ORDER BY seq DESC LIMIT ?",
                (user_id, max(0, limit)),
            )
            rows = cur.fetchall()
        return [
            Memory(
                id=r[0],
                user_id=user_id,
                content=r[1],
                created_at=r[2],
                metadata=json.loads(r[3]),
            )
            for r in rows
        ]

    def count(self, user_id: str) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM memories WHERE user_id = ?", (user_id,))
            return int(cur.fetchone()[0])

    def fetch_embeddings(self, user_id: str, limit: int = 500) -> list[tuple[str, list[float]]]:
        """Return ``(id, embedding)`` for a user's most-recent memories (newest first, capped).

        Read-only. Used ONLY by the dashboard-read path to derive similarity synapses — never by
        recall, so the recall cost invariant is untouched.
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, embedding FROM memories WHERE user_id = ? ORDER BY seq DESC LIMIT ?",
                (user_id, max(0, limit)),
            )
            rows = cur.fetchall()
        return [(r[0], _from_blob(r[1])) for r in rows]

    def build_index(self, user_id: str) -> tuple[InMemoryStore, dict[str, Memory]]:
        """Materialise a user's memories into an ``InMemoryStore`` for hybrid retrieval.

        Returns the populated store (its chunks carry ``session_id = memory_id`` and
        ``date = created_at`` so a retrieved chunk maps straight back to its record) plus a
        ``{memory_id: Memory}`` map for full-record lookup without a second query.
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, content, created_at, metadata, embedding FROM memories "
                "WHERE user_id = ? ORDER BY seq",
                (user_id,),
            )
            rows = cur.fetchall()
        store = InMemoryStore()
        by_id: dict[str, Memory] = {}
        chunks: list[MemoryChunk] = []
        for mem_id, content, created_at, meta_json, blob in rows:
            chunks.append(
                MemoryChunk(
                    text=content,
                    session_id=mem_id,
                    date=created_at,
                    embedding=_from_blob(blob),
                )
            )
            by_id[mem_id] = Memory(
                id=mem_id,
                user_id=user_id,
                content=content,
                created_at=created_at,
                metadata=json.loads(meta_json),
            )
        store.add(chunks)
        return store, by_id

    # -- L4 episodic (additive; only touched when the engine's episodic flag is on) ---------

    def add_event(
        self,
        memory_id: str,
        user_id: str,
        *,
        event_time: str | None,
        ingest_time: str,
        actor: str | None,
        location: str | None,
        event_type: str | None,
        episode_id: str | None = None,
        confidence: float | None = None,
    ) -> None:
        """Insert the episodic ``events`` row for one memory (keyed 1:1 to ``memory_id``)."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO events(memory_id, user_id, event_time, ingest_time, actor, "
                "location, event_type, episode_id, confidence) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    memory_id,
                    user_id,
                    event_time,
                    ingest_time,
                    actor,
                    location,
                    event_type,
                    episode_id,
                    confidence,
                ),
            )
            self._conn.commit()

    def timeline(
        self, user_id: str, since: str | None = None, until: str | None = None, limit: int = 50
    ) -> list[Memory]:
        """Return memories with an episodic event, ordered by ``event_time`` (NULLs last).

        JOINs ``events`` onto ``memories``, filters ``event_time`` to ``[since, until]`` when
        given, and attaches the event fields under ``metadata["event"]``. ISO-8601 TEXT times
        sort lexically, so ordering/range needs no date parsing.
        """
        clauses = ["e.user_id = ?"]
        params: list[object] = [user_id]
        if since is not None:
            clauses.append("e.event_time >= ?")
            params.append(since)
        if until is not None:
            clauses.append("e.event_time <= ?")
            params.append(until)
        params.append(max(0, limit))
        sql = (
            "SELECT m.id, m.user_id, m.content, m.created_at, m.metadata, "
            "e.event_time, e.actor, e.location, e.event_type, e.episode_id "
            "FROM events e JOIN memories m ON m.id = e.memory_id "
            "WHERE " + " AND ".join(clauses) + " "
            "ORDER BY (e.event_time IS NULL), e.event_time LIMIT ?"
        )
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        out: list[Memory] = []
        for r in rows:
            meta = json.loads(r[4])
            meta["event"] = {
                "event_time": r[5],
                "actor": r[6],
                "location": r[7],
                "event_type": r[8],
                "episode_id": r[9],
            }
            out.append(Memory(id=r[0], user_id=r[1], content=r[2], created_at=r[3], metadata=meta))
        return out

    # -- L5 anti-saturation (additive; touched only when a dedup/soft-update flag is on) ----------

    def add_supersession(self, superseded_id: str, superseded_by: str, at: str) -> None:
        """Record that ``superseded_id`` was replaced by the newer memory ``superseded_by``.

        Idempotent per superseded memory (``INSERT OR REPLACE``): re-superseding an old value
        repoints it at the newest replacement, so a value updated N times leaves one live row.
        """
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO supersessions(superseded_id, superseded_by, at) "
                "VALUES (?, ?, ?)",
                (superseded_id, superseded_by, at),
            )
            self._conn.commit()

    def superseded_ids(self, user_id: str) -> set[str]:
        """Return the ids of this user's memories that have been superseded (to drop from recall).

        Joins onto ``memories`` so the result is user-scoped even though the supersession row itself
        is keyed only by memory id. Empty set for a user with no supersessions.
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT s.superseded_id FROM supersessions s "
                "JOIN memories m ON m.id = s.superseded_id WHERE m.user_id = ?",
                (user_id,),
            )
            return {r[0] for r in cur.fetchall()}

    # -- G1 entity graph (additive; touched only when the write-time graph flag is on) ----------

    def ensure_self_entity(self, user_id: str) -> str:
        """Lazily create the synthetic ``self`` entity ("You") for a user; return its id.

        Idempotent: resolves the reserved ``__self__`` norm_name first, minting a row only if
        absent. The ``(user_id, norm_name)`` UNIQUE constraint guards against a duplicate.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM entities WHERE user_id = ? AND norm_name = ?",
                (user_id, _SELF_NORM),
            ).fetchone()
            if row is not None:
                return str(row[0])
            entity_id = uuid.uuid4().hex
            self._conn.execute(
                "INSERT INTO entities(id, user_id, name, norm_name, type, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (entity_id, user_id, "You", _SELF_NORM, "self", _now_iso()),
            )
            self._conn.commit()
            return entity_id

    def upsert_entity(self, user_id: str, name: str, entity_type: str) -> str:
        """Resolve an entity by ``(user_id, norm_name)`` or create it; return its id.

        Dedup is by normalised name, so "Swizel", " swizel " and "SWIZEL" collapse to one node.
        If the row exists we conservatively UPGRADE a generic stored type ("" / "thing") to a more
        specific incoming one, but NEVER downgrade — and never touch a ``self`` node. An unknown
        incoming ``entity_type`` falls back to "thing".
        """
        etype = entity_type if entity_type in VALID_ENTITY_TYPES else "thing"
        norm = _norm_name(name)
        with self._lock:
            row = self._conn.execute(
                "SELECT id, type FROM entities WHERE user_id = ? AND norm_name = ?",
                (user_id, norm),
            ).fetchone()
            if row is not None:
                entity_id, stored = str(row[0]), str(row[1])
                if (
                    stored != "self"
                    and stored in _GENERIC_ENTITY_TYPES
                    and etype not in _GENERIC_ENTITY_TYPES
                    and etype != "self"
                ):
                    self._conn.execute(
                        "UPDATE entities SET type = ? WHERE id = ?", (etype, entity_id)
                    )
                    self._conn.commit()
                return entity_id
            entity_id = uuid.uuid4().hex
            self._conn.execute(
                "INSERT INTO entities(id, user_id, name, norm_name, type, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (entity_id, user_id, name.strip(), norm, etype, _now_iso()),
            )
            self._conn.commit()
            return entity_id

    def add_entity_edge(
        self,
        user_id: str,
        src_id: str,
        label: str,
        dst_id: str,
        source_memory_id: str | None,
    ) -> None:
        """Insert a labeled directed edge ``src -[label]-> dst``; dedup and skip self-loops.

        A UNIQUE ``(user_id, src_id, label, dst_id)`` conflict is ignored (idempotent). An edge
        whose endpoints are the same entity is silently dropped.
        """
        if src_id == dst_id:
            return
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO entity_edges("
                "id, user_id, src_id, label, dst_id, source_memory_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (uuid.uuid4().hex, user_id, src_id, label, dst_id, source_memory_id, _now_iso()),
            )
            self._conn.commit()

    def link_memory_entity(self, user_id: str, memory_id: str, entity_id: str, role: str) -> None:
        """Link a memory to an entity with ``role`` ("subject" or "mention"); upsert by the pair.

        On conflict, "subject" wins: an existing "mention" is promoted to "subject" if the incoming
        role is "subject", but a "subject" is never demoted back to "mention".
        """
        erole = role if role in VALID_ENTITY_ROLES else "mention"
        with self._lock:
            self._conn.execute(
                "INSERT INTO memory_entities(memory_id, entity_id, role, user_id) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(memory_id, entity_id) DO UPDATE SET role = "
                "CASE WHEN excluded.role = 'subject' THEN 'subject' ELSE memory_entities.role END",
                (memory_id, entity_id, erole, user_id),
            )
            self._conn.commit()

    def get_graph(self, user_id: str) -> dict[str, list[dict[str, object]]]:
        """Return the user's raw graph rows for assembly (no LLM/embedding involved).

        ``{"entities": [{id,name,type}], "edges": [{src_id,label,dst_id,source_memory_id}],
        "memory_links": [{memory_id,entity_id,role}]}`` — the read side shapes them into an
        ego-graph payload.
        """
        with self._lock:
            entities = self._conn.execute(
                "SELECT id, name, type FROM entities WHERE user_id = ? ORDER BY created_at, id",
                (user_id,),
            ).fetchall()
            edges = self._conn.execute(
                "SELECT src_id, label, dst_id, source_memory_id FROM entity_edges "
                "WHERE user_id = ? ORDER BY created_at, id",
                (user_id,),
            ).fetchall()
            links = self._conn.execute(
                "SELECT memory_id, entity_id, role FROM memory_entities WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        return {
            "entities": [{"id": r[0], "name": r[1], "type": r[2]} for r in entities],
            "edges": [
                {"src_id": r[0], "label": r[1], "dst_id": r[2], "source_memory_id": r[3]}
                for r in edges
            ],
            "memory_links": [{"memory_id": r[0], "entity_id": r[1], "role": r[2]} for r in links],
        }

    def get_entity_by_name(self, user_id: str, name: str) -> list[dict[str, object]]:
        """Resolve entities by normalised name: exact → prefix → contains, best-match-first.

        A self-alias query ({self, you, me, i, my, mine, myself}) short-circuits to the synthetic
        ``self`` root — the ONLY way to reach it by name, since the ``__self__`` row is otherwise
        EXCLUDED from all matching below (so a stray substring like "elf"/"self" never leaks it).
        Failing that, an exact norm_name hit short-circuits (returned alone); otherwise prefix
        matches come first, then remaining substring matches. Each row is ``{id, name, type}``.
        """
        norm = _norm_name(name)
        if not norm:
            return []
        if norm in SELF_ALIASES:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT id, name, type FROM entities WHERE user_id = ? AND norm_name = ?",
                    (user_id, _SELF_NORM),
                ).fetchall()
            return [{"id": r[0], "name": r[1], "type": r[2]} for r in rows]
        pat = _like_escape(norm)
        with self._lock:
            exact = self._conn.execute(
                "SELECT id, name, type FROM entities "
                "WHERE user_id = ? AND norm_name = ? AND norm_name != ? ORDER BY name",
                (user_id, norm, _SELF_NORM),
            ).fetchall()
            if exact:
                return [{"id": r[0], "name": r[1], "type": r[2]} for r in exact]
            prefix = self._conn.execute(
                "SELECT id, name, type FROM entities "
                "WHERE user_id = ? AND norm_name LIKE ? ESCAPE '\\' AND norm_name != ? "
                "ORDER BY name",
                (user_id, pat + "%", _SELF_NORM),
            ).fetchall()
            contains = self._conn.execute(
                "SELECT id, name, type FROM entities "
                "WHERE user_id = ? AND norm_name LIKE ? ESCAPE '\\' AND norm_name != ? "
                "ORDER BY name",
                (user_id, "%" + pat + "%", _SELF_NORM),
            ).fetchall()
        prefix_ids = {r[0] for r in prefix}
        out = [{"id": r[0], "name": r[1], "type": r[2]} for r in prefix]
        out.extend(
            {"id": r[0], "name": r[1], "type": r[2]} for r in contains if r[0] not in prefix_ids
        )
        return out

    def get_entity_dossier(self, user_id: str, entity_id: str) -> dict[str, object]:
        """Return one entity's dossier: the node, its edges (both directions, other-name
        resolved), and its memories (newest first via ``memory_entities``).

        Edges are ``{src_id, src_name, label, dst_id, dst_name, direction}`` where ``direction`` is
        "out" for edges leaving this entity and "in" for edges arriving. A missing entity yields
        ``{"entity": None, ...}`` (the None sentinel the frontend guards on, never an empty ``{}``
        that would slip past an ``=== null`` check). No LLM/embedding — a keyed DB read only.
        """
        with self._lock:
            ent = self._conn.execute(
                "SELECT id, name, type FROM entities WHERE user_id = ? AND id = ?",
                (user_id, entity_id),
            ).fetchone()
            if ent is None:
                return {"entity": None, "edges": [], "memories": []}
            edges = self._conn.execute(
                "SELECT e.src_id, s.name, e.label, e.dst_id, d.name, "
                "CASE WHEN e.src_id = ? THEN 'out' ELSE 'in' END "
                "FROM entity_edges e "
                "LEFT JOIN entities s ON s.id = e.src_id "
                "LEFT JOIN entities d ON d.id = e.dst_id "
                "WHERE e.user_id = ? AND (e.src_id = ? OR e.dst_id = ?) "
                "ORDER BY e.created_at, e.id",
                (entity_id, user_id, entity_id, entity_id),
            ).fetchall()
            mems = self._conn.execute(
                "SELECT m.id, m.content, m.created_at, m.metadata "
                "FROM memory_entities me JOIN memories m ON m.id = me.memory_id "
                "WHERE me.user_id = ? AND me.entity_id = ? ORDER BY m.seq DESC",
                (user_id, entity_id),
            ).fetchall()
        return {
            "entity": {"id": ent[0], "name": ent[1], "type": ent[2]},
            "edges": [
                {
                    "src_id": r[0],
                    "src_name": r[1],
                    "label": r[2],
                    "dst_id": r[3],
                    "dst_name": r[4],
                    "direction": r[5],
                }
                for r in edges
            ],
            "memories": [
                Memory(
                    id=r[0],
                    user_id=user_id,
                    content=r[1],
                    created_at=r[2],
                    metadata=json.loads(r[3]),
                )
                for r in mems
            ],
        }


# The six semantic kinds an agent may tag a memory with (the dashboard groups memories by these).
# A memory with no/invalid kind defaults to "fact" on the dashboard-read path (web.service._kind).
# Kept here — the shared store module both write paths import — so the local stdio server and the
# hosted web server agree on the vocabulary without the core depending on the optional web package.
VALID_KINDS = frozenset({"preference", "fact", "project", "instruction", "event", "relationship"})


def normalize_kind(kind: str | None) -> str | None:
    """Lower-case ``kind`` and return it iff it is one of the six ``VALID_KINDS``, else ``None``."""
    if not kind:
        return None
    k = kind.strip().lower()
    return k if k in VALID_KINDS else None


def make_metadata(tags: Sequence[str] | None = None, **extra: object) -> dict[str, object]:
    """Build a metadata dict from optional tags plus arbitrary extra fields."""
    meta: dict[str, object] = {}
    if tags:
        meta["tags"] = list(tags)
    for key, value in extra.items():
        if value is not None:
            meta[key] = value
    return meta


def coerce_metadata(metadata: Mapping[str, object] | None) -> dict[str, object]:
    """Normalise an incoming metadata mapping into a plain dict, verified JSON-serializable.

    Metadata is persisted as JSON, so reject non-serializable values up front with a clear
    error rather than letting ``json.dumps`` throw opaquely deep inside ``add()``.
    """
    if not metadata:
        return {}
    data = dict(metadata)
    try:
        # Round-trip through JSON so the in-memory record matches the canonical form that ``add()``
        # persists and ``get()``/``list_recent()`` read back (e.g. a tuple becomes a list, dict keys
        # become strings). Without this, ``memorize(...).metadata`` could differ from the same
        # memory reloaded from disk.
        canonical: dict[str, object] = json.loads(json.dumps(data))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"metadata must be JSON-serializable: {exc}") from exc
    return canonical
