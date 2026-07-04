"""Offline, deterministic regression tests for the A-plus engine/store hardening pass.

Each test pins an exact failure the fixes close, using only ``FakeProvider`` + an in-file SQLite
store (no network, no live LLM). Covered:

1. A restated-but-currently-true fact whose nearest neighbour is a SUPERSEDED row must be written
   fresh, never dedup-absorbed back into the dead row (the A→B→A "Paris→Berlin→Paris" loss).
2. Recall over-fetches past superseded rows so a supersession filter still yields a FULL ``k``.
3. ``_rerank`` dedupes a misbehaving reranker's repeated ids instead of injecting duplicates.
4. ``coerce_metadata`` round-trips through JSON so the in-memory record matches the reloaded form.
6. ``InMemoryStore.dense_search`` raises on a dimension mismatch instead of silently truncating.

Plus light coverage that the schema version is stamped and ``SQLiteStore`` satisfies the
``MemoryStore`` Protocol.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from cortex.memory import CortexMemory
from cortex.providers.base import EmbedResult
from cortex.providers.fake import FakeProvider
from cortex.store.base import MemoryStore
from cortex.store.memory_store import InMemoryStore, MemoryChunk
from cortex.store.sqlite_store import SCHEMA_VERSION, SQLiteStore, coerce_metadata


def _v(*head: float) -> list[float]:
    """A 16-dim vector with ``head`` in the leading slots (matches FakeProvider's dim=16)."""
    return list(head) + [0.0] * (16 - len(head))


class _KeyedProvider(FakeProvider):
    """FakeProvider whose ``embed`` returns caller-controlled vectors by substring match.

    Keys are tried in insertion order; the first key that is a substring of the text wins, so a
    test can pin exact cosines. Any text matching no key falls back to the deterministic hash
    embedding.
    """

    def __init__(self, table: dict[str, list[float]]) -> None:
        super().__init__(dim=16)
        self.table = table

    def embed(self, texts: Sequence[str]) -> EmbedResult:
        out: list[list[float]] = []
        for t in texts:
            vec: list[float] | None = None
            for key, v in self.table.items():
                if key in t:
                    vec = list(v)
                    break
            out.append(vec if vec is not None else self._embed_one(t))
        return EmbedResult(vectors=out, input_tokens=0)


def _arbiter(verdict_json: str) -> FakeProvider:
    """An arbiter provider that returns ``verdict_json`` on the arbiter prompt, else abstains."""

    def responder(prompt: str) -> str:
        if "contradiction arbiter" in prompt:
            return verdict_json
        return "I don't know."

    return FakeProvider(responder=responder)


# -- (1) restated-true fact must not vanish into a superseded row --------------------------------


def test_restated_true_fact_not_absorbed_into_superseded(tmp_path):
    """A→B→A: after Paris→Berlin (UPDATE), restating Paris must write a FRESH live row.

    The nearest neighbour to the restated "Paris" is the ORIGINAL Paris row — but that row is now
    superseded. Dedup must skip it (not absorb the new fact into the dead row), so recall can still
    surface the currently-true value. Requires BOTH dedup and soft-update to be active.
    """
    provider = _KeyedProvider(
        {"Paris": _v(1.0), "Berlin": _v(1.0, 0.5), "which city": _v(1.0, 0.3)}
    )
    eng = CortexMemory(
        provider,
        SQLiteStore(tmp_path / "m.db"),
        use_dedup=True,
        use_soft_update=True,
        arbiter=_arbiter('{"verdict": "UPDATE", "supersedes_id": null}'),
    )
    paris = eng.memorize("home city is Paris")
    berlin = eng.memorize("home city is Berlin")  # arbiter UPDATE -> supersedes Paris
    paris2 = eng.memorize("home city is Paris")  # restated & true again -> must NOT be absorbed

    assert eng.count() == 3  # all three rows on disk (nothing dedup-absorbed into the dead row)
    assert paris2.id not in (paris.id, berlin.id)  # a genuinely new, live row
    assert eng.store.superseded_ids("local") == {paris.id, berlin.id}  # both old values retired

    hits = eng.recall("which city does the user live in")
    ids = [m.id for m in hits]
    assert paris2.id in ids  # the restated-true value is recallable...
    assert paris.id not in ids and berlin.id not in ids  # ...and the superseded ones are gone
    assert any("Paris" in m.content for m in hits)


# -- (2) recall over-fetches past superseded rows to fill k --------------------------------------


def test_recall_overfetches_past_superseded_to_fill_k(tmp_path):
    """With a supersession filter active, recall must return a FULL ``k`` of LIVE memories.

    The three highest-ranked candidates for the query are superseded; a naive depth==k fetch would
    drop them and return nothing. Over-fetching with headroom must surface the two live rows.
    """
    provider = _KeyedProvider({"alpha": _v(1.0), "bravo": _v(1.0, 0.35), "topic": _v(1.0)})
    eng = CortexMemory(provider, SQLiteStore(tmp_path / "m.db"), use_soft_update=True)

    superseded = [eng.memorize(f"the user likes topic alpha {w}").id for w in ("one", "two", "tri")]
    live = [eng.memorize(f"the user likes topic bravo {w}") for w in ("one", "two")]
    for sid in superseded:  # retire all three high-ranked rows
        eng.store.add_supersession(sid, live[0].id, "t")
    assert eng.store.superseded_ids("local") == set(superseded)

    hits = eng.recall("which topic", limit=2)
    assert len(hits) == 2  # full k despite three superseded rows outranking the live ones
    assert all("bravo" in m.content for m in hits)
    assert all(m.id not in set(superseded) for m in hits)


# -- (3) _rerank dedupes a misbehaving reranker's repeated ids -----------------------------------


class _FixedReranker:
    """A reranker that always returns a fixed id list (here, with a deliberate duplicate)."""

    def __init__(self, ids: Sequence[str]) -> None:
        self.ids = list(ids)

    def rerank(self, query: str, items: Sequence[tuple[str, str]], top_k: int) -> list[str]:
        return self.ids


def test_rerank_dedupes_duplicate_ranked_ids(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    provider = FakeProvider()
    seed = CortexMemory(provider, store)
    a = seed.memorize("a fact about apples")
    b = seed.memorize("a fact about bananas")
    c = seed.memorize("a fact about cherries")

    reranker = _FixedReranker([a.id, a.id, b.id])  # repeats a.id -> must not double-inject
    eng = CortexMemory(provider, store, reranker=reranker, top_k=3)
    hits = eng.recall("fact", limit=3)
    ids = [m.id for m in hits]

    assert len(ids) == len(set(ids))  # no duplicate Memory objects
    assert ids.count(a.id) == 1
    assert set(ids) == {a.id, b.id, c.id}  # deduped, then backfilled to a full k


# -- (4) coerce_metadata round-trips through JSON ------------------------------------------------


def test_coerce_metadata_tuple_round_trips_to_list():
    out = coerce_metadata({"tags": ("x", "y"), "n": 1})
    assert out == {"tags": ["x", "y"], "n": 1}
    assert isinstance(out["tags"], list)  # tuple canonicalised to the stored JSON form


def test_memorize_metadata_matches_reloaded_form(tmp_path):
    path = tmp_path / "m.db"
    eng = CortexMemory(FakeProvider(), SQLiteStore(path))
    mem = eng.memorize("note", metadata={"tags": ("a", "b")})
    assert mem.metadata == {"tags": ["a", "b"]}  # in-memory record is already canonical

    eng.store.close()
    reloaded = CortexMemory(FakeProvider(), SQLiteStore(path)).list_memories()[0]
    assert reloaded.metadata == mem.metadata  # matches the value re-read from disk, byte for byte


# -- (6) dense_search surfaces a dimension mismatch ----------------------------------------------


def test_dense_search_dim_mismatch_raises():
    store = InMemoryStore()
    store.add([MemoryChunk(text="x", session_id="1", date="t", embedding=[1.0, 2.0, 3.0])])
    with pytest.raises(ValueError):
        store.dense_search([1.0, 2.0], 5)  # 2-dim query vs 3-dim stored -> loud, not truncated


# -- store hardening: schema version + Protocol conformance --------------------------------------


def test_schema_version_stamped_on_connect(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    version = store._conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == SCHEMA_VERSION


def test_sqlite_store_satisfies_memory_store_protocol(tmp_path):
    # The annotation is a structural (mypy) conformance check; the calls exercise the core surface.
    store: MemoryStore = SQLiteStore(tmp_path / "m.db")
    store.ensure_embedding("fake", 16)
    assert store.count("nobody") == 0
