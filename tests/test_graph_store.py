"""Offline, deterministic tests for the G1 entity-graph store (SQLite).

Everything here is additive: the three graph tables exist on every store but stay empty unless the
write-time graph path fills them, so the existing memory CRUD is byte-identical when unused. No
network, no live LLM calls.
"""

from __future__ import annotations

from cortex.store.sqlite_store import (
    Memory,
    SQLiteStore,
    _like_escape,
    _norm_name,
)

_CREATED = "2026-07-03T00:00:00+00:00"

_GRAPH_METHODS = frozenset(
    {
        "ensure_self_entity",
        "upsert_entity",
        "add_entity_edge",
        "link_memory_entity",
        "get_graph",
        "get_entity_by_name",
        "get_entity_dossier",
    }
)


def _add_mem(store: SQLiteStore, mem_id: str, content: str, user_id: str = "u1") -> None:
    store.add(Memory(id=mem_id, user_id=user_id, content=content, created_at=_CREATED), [1.0, 0.0])


def _entities_by_id(store: SQLiteStore, user_id: str = "u1") -> dict[str, dict[str, object]]:
    return {e["id"]: e for e in store.get_graph(user_id)["entities"]}


# -- module helpers ----------------------------------------------------------------------------


def test_norm_name_lowercases_strips_and_collapses_whitespace():
    assert _norm_name("SWIZEL") == "swizel"
    assert _norm_name("  swizel  ") == "swizel"
    assert _norm_name("Swizel   Dias\tSapeco") == "swizel dias sapeco"


def test_like_escape_neutralises_wildcards():
    assert _like_escape("a_b%c") == "a\\_b\\%c"
    assert _like_escape("plain") == "plain"


# -- (1) upsert_entity dedups by norm_name -----------------------------------------------------


def test_upsert_entity_dedups_by_norm_name(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    a = store.upsert_entity("u1", "Swizel", "person")
    b = store.upsert_entity("u1", " swizel ", "person")
    c = store.upsert_entity("u1", "SWIZEL", "thing")
    assert a == b == c
    ents = store.get_graph("u1")["entities"]
    assert len(ents) == 1  # one node, no self created
    assert ents[0]["type"] == "person"  # a "thing" re-upsert never downgrades a specific type


def test_upsert_entity_upgrades_generic_but_never_downgrades(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    gid = store.upsert_entity("u1", "Goa", "thing")
    store.upsert_entity("u1", "goa", "place")  # generic thing -> specific place
    assert _entities_by_id(store)[gid]["type"] == "place"
    store.upsert_entity("u1", "Goa", "thing")  # never downgrade back
    assert _entities_by_id(store)[gid]["type"] == "place"


def test_upsert_entity_invalid_type_falls_back_to_thing(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    wid = store.upsert_entity("u1", "Widget", "gizmo")
    assert _entities_by_id(store)[wid]["type"] == "thing"


# -- (2) ensure_self_entity is idempotent ------------------------------------------------------


def test_ensure_self_entity_idempotent_and_per_user(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    s1 = store.ensure_self_entity("u1")
    s2 = store.ensure_self_entity("u1")
    assert s1 == s2
    selves = [e for e in store.get_graph("u1")["entities"] if e["type"] == "self"]
    assert len(selves) == 1
    assert selves[0]["name"] == "You"
    assert store.ensure_self_entity("u2") != s1  # each user gets its own self


# -- (3) add_entity_edge dedups and skips self-loops -------------------------------------------


def test_add_entity_edge_dedups_and_skips_self_loops(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    a = store.upsert_entity("u1", "A", "thing")
    b = store.upsert_entity("u1", "B", "thing")
    store.add_entity_edge("u1", a, "likes", b, None)
    store.add_entity_edge("u1", a, "likes", b, None)  # dup -> ignored
    store.add_entity_edge("u1", a, "knows", a, None)  # self-loop -> skipped
    edges = store.get_graph("u1")["edges"]
    assert len(edges) == 1
    assert edges[0]["label"] == "likes"
    assert edges[0]["source_memory_id"] is None


# -- (4) get_entity_by_name exact/prefix/contains ordering -------------------------------------


def test_get_entity_by_name_exact_prefix_contains_ordering(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    store.upsert_entity("u1", "Goa", "place")
    store.upsert_entity("u1", "Goa Beach", "place")
    store.upsert_entity("u1", "Algoa", "place")

    exact = store.get_entity_by_name("u1", "Goa")
    assert [e["name"] for e in exact] == ["Goa"]  # exact match returned alone

    ranked = store.get_entity_by_name("u1", "Go")
    # prefix matches first (Goa, Goa Beach), then the contains-only match (Algoa) last
    assert [e["name"] for e in ranked] == ["Goa", "Goa Beach", "Algoa"]

    assert store.get_entity_by_name("u1", "") == []
    assert store.get_entity_by_name("u1", "zzz") == []


def test_get_entity_by_name_self_alias_resolves_and_never_leaks(tmp_path):
    # FIX 1: the synthetic self root (name "You", norm_name "__self__") is reachable ONLY via a
    # self-alias, and the "__self__" row NEVER leaks into ordinary substring matching.
    store = SQLiteStore(tmp_path / "m.db")
    self_id = store.ensure_self_entity("u1")
    store.upsert_entity("u1", "Elf", "person")  # a real entity; "elf" ⊂ "__self__" (old leak case)

    # (a) every self-alias — regardless of case — resolves to the self root (the dead-node fix)
    for alias in ("me", "you", "You", "MYSELF", "self", "i", "my", "mine"):
        got = store.get_entity_by_name("u1", alias)
        assert len(got) == 1, alias
        assert got[0]["id"] == self_id and got[0]["type"] == "self", alias

    # (b) a NON-alias substring of "__self__" ("elf") resolves to the real "Elf" entity only — the
    # "__self__" row is excluded from exact/prefix/contains, so it never spuriously matches.
    elf = store.get_entity_by_name("u1", "elf")
    assert [e["name"] for e in elf] == ["Elf"]
    assert all(e["type"] != "self" for e in elf)

    # (b′) a substring with no real entity behind it returns nothing (the pure substring leak is
    # gone: "sel" ⊂ "__self__" used to match the self row via the `%...%` contains clause).
    assert store.get_entity_by_name("u1", "sel") == []


# -- (5) get_entity_dossier: entity + both-direction edges + memories --------------------------


def test_get_entity_dossier_returns_edges_both_directions_and_memories(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    _add_mem(store, "a" * 32, "Swizel is my girlfriend")
    _add_mem(store, "b" * 32, "Swizel likes apples")

    self_id = store.ensure_self_entity("u1")
    swizel = store.upsert_entity("u1", "Swizel", "person")
    apples = store.upsert_entity("u1", "apples", "thing")

    store.add_entity_edge("u1", self_id, "girlfriend", swizel, "a" * 32)  # inbound to Swizel
    store.add_entity_edge("u1", swizel, "likes", apples, "b" * 32)  # outbound from Swizel
    store.link_memory_entity("u1", "a" * 32, swizel, "subject")
    store.link_memory_entity("u1", "b" * 32, swizel, "mention")

    dossier = store.get_entity_dossier("u1", swizel)
    assert dossier["entity"]["name"] == "Swizel"

    edges = dossier["edges"]
    inbound = [e for e in edges if e["direction"] == "in"]
    outbound = [e for e in edges if e["direction"] == "out"]
    assert len(inbound) == 1 and inbound[0]["label"] == "girlfriend"
    assert inbound[0]["src_name"] == "You"  # the OTHER entity's name resolved
    assert len(outbound) == 1 and outbound[0]["label"] == "likes"
    assert outbound[0]["dst_name"] == "apples"

    # both linked memories, newest first (b was added after a)
    mems = dossier["memories"]
    assert [m.id for m in mems] == ["b" * 32, "a" * 32]
    assert all(isinstance(m, Memory) for m in mems)


def test_get_entity_dossier_missing_entity_is_none(tmp_path):
    # A missing entity yields the None sentinel (never an empty {} that would slip past the
    # frontend's `entity === null` guard), with empty edges/memories.
    store = SQLiteStore(tmp_path / "m.db")
    dossier = store.get_entity_dossier("u1", "nope")
    assert dossier == {"entity": None, "edges": [], "memories": []}


def test_link_memory_entity_prefers_subject_over_mention(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    _add_mem(store, "a" * 32, "x")
    e = store.upsert_entity("u1", "E", "thing")
    store.link_memory_entity("u1", "a" * 32, e, "mention")
    store.link_memory_entity("u1", "a" * 32, e, "subject")  # promotes
    assert store.get_graph("u1")["memory_links"][0]["role"] == "subject"
    store.link_memory_entity("u1", "a" * 32, e, "mention")  # never demotes
    assert store.get_graph("u1")["memory_links"][0]["role"] == "subject"


# -- (6) byte-identical guard: graph tables present but unused -> memory CRUD unaffected --------


def test_memory_crud_unaffected_by_graph_rows(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    _add_mem(store, "a" * 32, "first")
    _add_mem(store, "b" * 32, "second")
    before = [m.id for m in store.list_recent("u1", 10)]
    assert store.count("u1") == 2

    # Populate the graph; none of it must perturb the memories table / its reads.
    self_id = store.ensure_self_entity("u1")
    swizel = store.upsert_entity("u1", "Swizel", "person")
    store.add_entity_edge("u1", self_id, "girlfriend", swizel, "a" * 32)
    store.link_memory_entity("u1", "a" * 32, swizel, "subject")

    assert [m.id for m in store.list_recent("u1", 10)] == before
    assert store.count("u1") == 2
    got = store.get("a" * 32, "u1")
    assert got is not None and got.content == "first"
    index, by_id = store.build_index("u1")
    assert set(by_id) == {"a" * 32, "b" * 32}
    assert store.delete("b" * 32, "u1") is True
    assert store.count("u1") == 1


# -- (7) CASCADE: forgetting a memory prunes its links + source-tagged edges --------------------


def test_delete_memory_cascades_links_and_source_edges(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    _add_mem(store, "a" * 32, "m1")
    _add_mem(store, "b" * 32, "m2")
    self_id = store.ensure_self_entity("u1")
    swizel = store.upsert_entity("u1", "Swizel", "person")
    apples = store.upsert_entity("u1", "apples", "thing")

    store.add_entity_edge("u1", self_id, "girlfriend", swizel, "a" * 32)  # tagged to m1
    store.add_entity_edge("u1", swizel, "likes", apples, None)  # untagged -> survives
    store.link_memory_entity("u1", "a" * 32, swizel, "subject")  # link on m1
    store.link_memory_entity("u1", "b" * 32, swizel, "mention")  # link on m2

    assert store.delete("a" * 32, "u1") is True

    graph = store.get_graph("u1")
    assert {e["label"] for e in graph["edges"]} == {"likes"}  # source-tagged edge gone
    assert {link["memory_id"] for link in graph["memory_links"]} == {"b" * 32}  # m1 link gone
    assert len(graph["entities"]) == 3  # entities are not FK'd to memories -> untouched


def test_sqlite_store_exposes_graph_method_surface():
    for method in _GRAPH_METHODS:
        assert callable(getattr(SQLiteStore, method)), method
