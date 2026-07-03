"""Offline, deterministic tests for the CortexMemory product engine (FakeProvider)."""

from __future__ import annotations

import pytest

from cortex.memory import CortexMemory
from cortex.providers.base import EmbedResult
from cortex.providers.fake import FakeProvider
from cortex.store.sqlite_store import SQLiteStore


def _engine(tmp_path, **kw) -> CortexMemory:
    return CortexMemory(FakeProvider(), SQLiteStore(tmp_path / "m.db"), **kw)


class _EmptyEmbedProvider(FakeProvider):
    """Provider whose embed() returns no vectors — simulates a partial/failed embedding."""

    def embed(self, texts):
        return EmbedResult(vectors=[], input_tokens=0)


def test_memorize_returns_record_and_counts(tmp_path):
    eng = _engine(tmp_path)
    mem = eng.memorize("The user lives in Goa.", metadata={"tags": ["bio"]})
    assert mem.id and mem.content == "The user lives in Goa."
    assert mem.metadata == {"tags": ["bio"]}
    assert eng.count() == 1


def test_memorize_rejects_empty(tmp_path):
    eng = _engine(tmp_path)
    with pytest.raises(ValueError, match="empty"):
        eng.memorize("   ")


def test_recall_surfaces_relevant_memory(tmp_path):
    eng = _engine(tmp_path)
    eng.memorize("The user's dog is named Rex.")
    eng.memorize("The user enjoys hiking mountains on weekends.")
    eng.memorize("The user works as a backend engineer.")
    hits = eng.recall("What is the name of the user's dog?", limit=3)
    assert hits, "expected at least one recalled memory"
    assert "Rex" in hits[0].content


def test_recall_empty_store_and_empty_query(tmp_path):
    eng = _engine(tmp_path)
    assert eng.recall("anything") == []
    eng.memorize("a fact")
    assert eng.recall("   ") == []
    assert eng.recall("a fact", limit=0) == []


def test_forget_removes_memory(tmp_path):
    eng = _engine(tmp_path)
    mem = eng.memorize("ephemeral note")
    assert eng.forget(mem.id) is True
    assert eng.count() == 0
    assert eng.forget(mem.id) is False


def test_memory_persists_across_engine_restart(tmp_path):
    path = tmp_path / "m.db"
    first = CortexMemory(FakeProvider(), SQLiteStore(path))
    first.memorize("Durable across restarts.")
    first.store.close()
    second = CortexMemory(FakeProvider(), SQLiteStore(path))
    assert second.count() == 1
    assert second.list_memories()[0].content == "Durable across restarts."


def test_user_scoping_isolates_memories(tmp_path):
    path = tmp_path / "m.db"
    store = SQLiteStore(path)
    alice = CortexMemory(FakeProvider(), store, user_id="alice")
    bob = CortexMemory(FakeProvider(), store, user_id="bob")
    alice.memorize("Alice secret")
    assert alice.count() == 1
    assert bob.count() == 0
    assert bob.recall("secret") == []


def test_memorize_rejects_failed_embedding(tmp_path):
    eng = CortexMemory(_EmptyEmbedProvider(), SQLiteStore(tmp_path / "m.db"))
    with pytest.raises(ValueError, match="embedding failed"):
        eng.memorize("something worth keeping")
    assert eng.count() == 0  # nothing corrupt was stored


def test_recall_rejects_failed_query_embedding(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    CortexMemory(FakeProvider(), store).memorize("a seeded fact")
    eng = CortexMemory(_EmptyEmbedProvider(), store)
    with pytest.raises(ValueError, match="embedding failed"):
        eng.recall("anything")


def test_metadata_must_be_json_serializable(tmp_path):
    eng = _engine(tmp_path)
    with pytest.raises(ValueError, match="JSON-serializable"):
        eng.memorize("note", metadata={"bad": object()})
    assert eng.count() == 0  # fail-fast: rejected before storing


def test_forget_prefix_full_short_and_none(tmp_path):
    eng = _engine(tmp_path)
    mem = eng.memorize("unique fact alpha")
    assert "Forgot" in eng.forget_prefix(mem.id[:8])  # short id
    assert eng.count() == 0
    assert "No memory matched" in eng.forget_prefix("deadbeef")  # no match
    assert "No memory id given" in eng.forget_prefix("   ")  # empty


def test_forget_prefix_refuses_ambiguous_prefix(tmp_path, monkeypatch):
    eng = _engine(tmp_path)
    a = eng.memorize("fact one")
    eng.memorize("fact two")
    # Simulate two ids sharing the prefix -> must refuse, delete nothing.
    monkeypatch.setattr(eng.store, "resolve_id_prefix", lambda u, p, limit=2: [a.id, "other"])
    assert "Ambiguous" in eng.forget_prefix("ab")
    assert eng.count() == 2


def test_embedding_mismatch_blocks_writes_but_not_recovery(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    CortexMemory(FakeProvider(), store).memorize("recoverable fact")  # dim 16 recorded

    mism = CortexMemory(FakeProvider(dim=32), store)  # different embedder -> mismatch
    with pytest.raises(ValueError, match="[Ee]mbedding mismatch"):
        mism.memorize("nope")
    with pytest.raises(ValueError, match="[Ee]mbedding mismatch"):
        mism.recall("anything")
    # The lazy guard's whole point: read-only recovery still works on a mismatched store.
    assert mism.count() == 1
    recovered = mism.list_memories()[0]
    assert recovered.content == "recoverable fact"
    assert "Forgot" in mism.forget_prefix(recovered.id)


def test_recall_uses_default_top_k(tmp_path):
    eng = _engine(tmp_path, top_k=2)
    for i in range(5):
        eng.memorize(f"distinct fact {i} about widget {i}")
    assert len(eng.recall("widget")) <= 2  # no explicit limit -> uses top_k


def test_close_is_safe(tmp_path):
    eng = _engine(tmp_path)
    eng.memorize("x")
    eng.close()  # must not raise
