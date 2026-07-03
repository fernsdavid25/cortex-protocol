"""Offline, deterministic tests for the persistent SQLite memory store."""

from __future__ import annotations

import threading

import pytest

from cortex.store.sqlite_store import (
    Memory,
    SQLiteStore,
    _from_blob,
    _to_blob,
    coerce_metadata,
    make_metadata,
)


def _mem(mem_id: str, content: str, user_id: str = "u1") -> Memory:
    return Memory(
        id=mem_id, user_id=user_id, content=content, created_at="2026-06-28T00:00:00+00:00"
    )


def test_blob_roundtrip_is_float32_stable():
    vec = [0.1, -0.5, 1.0, 0.0]
    out = _from_blob(_to_blob(vec))
    assert len(out) == len(vec)
    assert out == pytest.approx(vec, abs=1e-6)


def test_add_get_count_delete(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    store.add(_mem("a" * 32, "hello"), [1.0, 0.0])
    assert store.count("u1") == 1
    got = store.get("a" * 32, "u1")
    assert got is not None and got.content == "hello"
    assert store.get("a" * 32, "other-user") is None  # user scoping
    assert store.delete("a" * 32, "u1") is True
    assert store.delete("a" * 32, "u1") is False
    assert store.count("u1") == 0


def test_metadata_roundtrips(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    m = Memory(
        id="b" * 32,
        user_id="u1",
        content="x",
        created_at="t",
        metadata=make_metadata(["work", "db"]),
    )
    store.add(m, [1.0])
    got = store.get("b" * 32, "u1")
    assert got is not None and got.metadata == {"tags": ["work", "db"]}


def test_list_recent_is_newest_first_and_user_scoped(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    store.add(_mem("1" * 32, "first"), [1.0])
    store.add(_mem("2" * 32, "second"), [1.0])
    store.add(_mem("3" * 32, "other", user_id="u2"), [1.0])
    recent = store.list_recent("u1", limit=10)
    assert [m.content for m in recent] == ["second", "first"]
    assert [m.content for m in store.list_recent("u2", 10)] == ["other"]


def test_persistence_across_reopen(tmp_path):
    path = tmp_path / "m.db"
    store = SQLiteStore(path)
    store.add(_mem("c" * 32, "durable fact"), [0.3, 0.4])
    store.close()
    reopened = SQLiteStore(path)
    got = reopened.get("c" * 32, "u1")
    assert got is not None and got.content == "durable fact"


def test_build_index_maps_chunks_back_to_memories(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    store.add(_mem("d" * 32, "the user lives in Goa"), [1.0, 0.0])
    store.add(_mem("e" * 32, "the user has a dog named Rex"), [0.0, 1.0])
    index, by_id = store.build_index("u1")
    assert len(index) == 2
    assert set(by_id) == {"d" * 32, "e" * 32}
    # Chunk provenance carries the memory id so retrieval maps straight back.
    assert {c.session_id for c in index.chunks} == {"d" * 32, "e" * 32}


def test_ensure_embedding_guards_mismatch(tmp_path):
    path = tmp_path / "m.db"
    store = SQLiteStore(path)
    store.ensure_embedding("gemini-embedding-001", 768)
    store.ensure_embedding("gemini-embedding-001", 768)  # idempotent
    with pytest.raises(ValueError, match="Embedding mismatch"):
        store.ensure_embedding("other-model", 768)
    with pytest.raises(ValueError, match="Embedding mismatch"):
        store.ensure_embedding("gemini-embedding-001", 1536)


def test_add_duplicate_id_raises_value_error(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    store.add(_mem("a" * 32, "first"), [1.0])
    with pytest.raises(ValueError, match="already exists"):
        store.add(_mem("a" * 32, "again"), [1.0])
    assert store.count("u1") == 1


def test_resolve_id_prefix_unique_none_and_ambiguous(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    store.add(_mem("aaaa" + "1" * 28, "one"), [1.0])
    store.add(_mem("aaaa" + "2" * 28, "two"), [1.0])
    store.add(_mem("bbbb" + "3" * 28, "three"), [1.0])
    assert store.resolve_id_prefix("u1", "bbbb") == ["bbbb" + "3" * 28]  # unique
    assert store.resolve_id_prefix("u1", "zzzz") == []  # none
    assert len(store.resolve_id_prefix("u1", "aaaa", limit=2)) == 2  # ambiguous


def test_list_recent_clamps_nonpositive_limit(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    store.add(_mem("a" * 32, "x"), [1.0])
    assert store.list_recent("u1", 0) == []
    assert store.list_recent("u1", -5) == []


def test_wal_mode_enabled(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_concurrent_adds_are_serialized(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")

    def worker(n: int) -> None:
        store.add(_mem(f"{n:032x}", f"mem {n}"), [1.0, 0.0])

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.count("u1") == 20  # lock serialized all writes; no loss, no crash


def test_db_path_expands_and_creates_parent_dirs(tmp_path):
    nested = tmp_path / "deep" / "nested" / "mem.db"
    store = SQLiteStore(nested)
    store.add(_mem("a" * 32, "x"), [1.0])
    assert nested.exists()
    assert store.count("u1") == 1


def test_coerce_metadata_rejects_non_serializable():
    assert coerce_metadata(None) == {}
    assert coerce_metadata({"tags": ["a"]}) == {"tags": ["a"]}
    with pytest.raises(ValueError, match="JSON-serializable"):
        coerce_metadata({"bad": object()})
