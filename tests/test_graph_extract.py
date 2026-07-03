"""Offline, deterministic tests for G2 write-time entity/relationship extraction.

Everything here is additive and OFF by default. The graph is built from the SAME folded
flash-lite extraction call the episodic path already makes (one call feeds both), and the
byte-identical guards prove a default ``CortexMemory`` (graph off) writes NO entity/edge/link
rows. No network, no live LLM calls (FakeProvider + SQLite).
"""

from __future__ import annotations

import json
from collections.abc import Callable

from cortex.memory import CortexMemory
from cortex.providers.fake import FakeProvider
from cortex.reader.reader import build_episodic_extraction_prompt, parse_graph_extraction
from cortex.store.sqlite_store import SQLiteStore

_UID = "u1"


class _CountingProvider(FakeProvider):
    """A FakeProvider that counts ``generate`` calls (to prove ONE shared call feeds both)."""

    def __init__(self, responder: Callable[[str], str]) -> None:
        super().__init__(responder=responder)
        self.calls = 0

    def generate(self, prompt: str, *, temperature: float = 0.0, max_output_tokens: int = 512):
        self.calls += 1
        return super().generate(
            prompt, temperature=temperature, max_output_tokens=max_output_tokens
        )


def _extractor(graph_json: str) -> _CountingProvider:
    """A canned extractor: returns ``graph_json`` on the folded extraction prompt, else abstains."""

    def responder(prompt: str) -> str:
        if "episodic event extractor" in prompt:
            return graph_json
        return "I don't know."

    return _CountingProvider(responder)


def _engine(store: SQLiteStore, extractor: _CountingProvider) -> CortexMemory:
    return CortexMemory(FakeProvider(), store, user_id=_UID, extractor=extractor, use_graph=True)


def _graph_obj(**fields: object) -> str:
    base: dict[str, object] = {
        "event_time": None,
        "actor": None,
        "location": None,
        "event_type": None,
        "entities": [],
        "relations": [],
        "subject": "self",
    }
    base.update(fields)
    return json.dumps(base)


# -- parser --------------------------------------------------------------------------------


def test_parse_graph_extraction_valid():
    text = _graph_obj(
        entities=[{"name": "Swizel", "type": "person"}, {"name": "apples", "type": "thing"}],
        relations=[
            {"src": "self", "label": "girlfriend", "dst": "Swizel"},
            {"src": "Swizel", "label": "likes", "dst": "apples"},
        ],
        subject="Swizel",
    )
    out = parse_graph_extraction(text)
    assert out["entities"] == [
        {"name": "Swizel", "type": "person"},
        {"name": "apples", "type": "thing"},
    ]
    assert out["relations"] == [
        {"src": "self", "label": "girlfriend", "dst": "Swizel"},
        {"src": "Swizel", "label": "likes", "dst": "apples"},
    ]
    assert out["subject"] == "Swizel"


def test_parse_graph_extraction_drops_malformed_and_normalizes():
    text = json.dumps(
        {
            "entities": [
                {"name": "Bob"},  # no type -> "thing"
                {"type": "person"},  # no name -> dropped
                {"name": "  ", "type": "person"},  # blank name -> dropped
                {"name": "Acme", "type": "bogus"},  # invalid type -> "thing"
            ],
            "relations": [
                {"src": "self", "label": "likes"},  # no dst -> dropped
                {"label": "x", "dst": "y"},  # no src -> dropped
                {"src": "a", "label": "KNOWS", "dst": "b"},  # label lowercased
            ],
            "subject": None,
        }
    )
    out = parse_graph_extraction(text)
    assert out["entities"] == [{"name": "Bob", "type": "thing"}, {"name": "Acme", "type": "thing"}]
    assert out["relations"] == [{"src": "a", "label": "knows", "dst": "b"}]
    assert out["subject"] is None


def test_parse_graph_extraction_empty_and_garbage():
    empty = {"entities": [], "relations": [], "subject": None}
    assert parse_graph_extraction("") == empty
    assert parse_graph_extraction("   ") == empty
    assert parse_graph_extraction("this is not json at all") == empty


def test_parse_graph_extraction_caps_each_list_at_20():
    text = json.dumps(
        {
            "entities": [{"name": f"e{i}", "type": "thing"} for i in range(30)],
            "relations": [{"src": "self", "label": "rel", "dst": f"e{i}"} for i in range(30)],
        }
    )
    out = parse_graph_extraction(text)
    assert len(out["entities"]) == 20
    assert len(out["relations"]) == 20


def test_parse_graph_extraction_markdown_fenced_and_prose():
    body = json.dumps(
        {"entities": [{"name": "Goa", "type": "place"}], "relations": [], "subject": "Goa"}
    )
    out = parse_graph_extraction(f"```json\n{body}\n```")
    assert out["entities"] == [{"name": "Goa", "type": "place"}]
    assert out["subject"] == "Goa"


# -- prompt builder (folded, marker preserved) ---------------------------------------------


def test_build_prompt_include_graph_adds_fields_and_keeps_marker():
    p = build_episodic_extraction_prompt("Swizel likes apples", "2026-07-02", include_graph=True)
    assert "episodic event extractor" in p  # FakeProvider responder marker preserved
    assert '"entities"' in p and '"relations"' in p and '"subject"' in p
    # the default (episodic-only) prompt is byte-identical to before: NO graph fields requested
    p0 = build_episodic_extraction_prompt("Swizel likes apples", "2026-07-02")
    assert '"entities"' not in p0 and '"relations"' not in p0 and '"subject"' not in p0


# -- memorize persists the graph -----------------------------------------------------------


def _david_swizel_apples_responder(prompt: str) -> str:
    """Return per-message graph JSON for the David/Swizel/apples scenario.

    Keys on substrings UNIQUE to each message body — NOT "girlfriend"/"likes" alone, which the
    prompt's own instruction examples contain (they would match every message).
    """
    if "likes apples" in prompt:
        return _graph_obj(
            entities=[{"name": "Swizel", "type": "person"}, {"name": "apples", "type": "thing"}],
            relations=[{"src": "Swizel", "label": "likes", "dst": "apples"}],
            subject="Swizel",
        )
    if "girlfriend's" in prompt:
        return _graph_obj(
            entities=[{"name": "Swizel", "type": "person"}],
            relations=[{"src": "self", "label": "girlfriend", "dst": "Swizel"}],
            subject="Swizel",
        )
    return _graph_obj(subject="self")  # "My name is David Fernandes" -> about self, no entities


def test_memorize_persists_entities_edges_and_links(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    extractor = _CountingProvider(_david_swizel_apples_responder)
    eng = _engine(store, extractor)

    m1 = eng.memorize("My name is David Fernandes")
    m2 = eng.memorize("My girlfriend's name is Swizel")
    m3 = eng.memorize("Swizel likes apples")
    assert extractor.calls == 3  # exactly one extraction call per memorize

    graph = store.get_graph(_UID)
    by_name = {e["name"]: e for e in graph["entities"]}
    assert by_name["You"]["type"] == "self"  # the synthetic ego root
    assert by_name["Swizel"]["type"] == "person"
    assert by_name["apples"]["type"] == "thing"
    assert len(graph["entities"]) == 3  # self emitted once; Swizel deduped across the two memories

    self_id = by_name["You"]["id"]
    swizel_id = by_name["Swizel"]["id"]
    apples_id = by_name["apples"]["id"]
    edges = {(e["src_id"], e["label"], e["dst_id"]) for e in graph["edges"]}
    assert (self_id, "girlfriend", swizel_id) in edges
    assert (swizel_id, "likes", apples_id) in edges
    assert len(graph["edges"]) == 2

    links = {(lk["memory_id"], lk["entity_id"], lk["role"]) for lk in graph["memory_links"]}
    assert (m1.id, self_id, "subject") in links  # a self-only memory links to self as subject
    assert (m2.id, swizel_id, "subject") in links
    assert (m3.id, swizel_id, "subject") in links
    assert (m3.id, apples_id, "mention") in links  # non-subject mentioned entity


def test_memorize_dossier_reads_back_edges_and_memories(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    eng = _engine(store, _CountingProvider(_david_swizel_apples_responder))
    eng.memorize("My girlfriend's name is Swizel")
    eng.memorize("Swizel likes apples")

    [swizel] = store.get_entity_by_name(_UID, "Swizel")
    dossier = store.get_entity_dossier(_UID, str(swizel["id"]))
    labels = {(e["label"], e["direction"]) for e in dossier["edges"]}
    assert ("girlfriend", "in") in labels  # self -> girlfriend -> Swizel arrives at Swizel
    assert ("likes", "out") in labels  # Swizel -> likes -> apples leaves Swizel
    contents = {m.content for m in dossier["memories"]}
    assert contents == {"My girlfriend's name is Swizel", "Swizel likes apples"}

    [apples] = store.get_entity_by_name(_UID, "apples")
    apples_doss = store.get_entity_dossier(_UID, str(apples["id"]))
    assert {m.content for m in apples_doss["memories"]} == {"Swizel likes apples"}


def test_first_person_aliases_resolve_to_self(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    canned = _graph_obj(
        entities=[{"name": "car", "type": "thing"}],
        relations=[{"src": "I", "label": "owns", "dst": "car"}],  # "I" -> self at memorize time
        subject="me",  # "me" -> self
    )
    eng = _engine(store, _extractor(canned))
    mem = eng.memorize("I own a car")

    graph = store.get_graph(_UID)
    names = {e["name"] for e in graph["entities"]}
    assert "You" in names and "car" in names
    assert "I" not in names and "me" not in names  # first-person aliases never mint a node
    self_id = next(e["id"] for e in graph["entities"] if e["type"] == "self")
    car_id = next(e["id"] for e in graph["entities"] if e["name"] == "car")
    edges = {(e["src_id"], e["label"], e["dst_id"]) for e in graph["edges"]}
    assert (self_id, "owns", car_id) in edges  # the "I" endpoint collapsed to self
    links = {(lk["memory_id"], lk["entity_id"], lk["role"]) for lk in graph["memory_links"]}
    assert (mem.id, self_id, "subject") in links  # "me" subject collapsed to self


def test_episodic_and_graph_share_one_extraction_call(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    canned = _graph_obj(
        event_time="2026-06-25",
        actor="Swizel",
        event_type="preference",
        entities=[{"name": "Swizel", "type": "person"}, {"name": "apples", "type": "thing"}],
        relations=[{"src": "Swizel", "label": "likes", "dst": "apples"}],
        subject="Swizel",
    )
    extractor = _extractor(canned)
    eng = CortexMemory(
        FakeProvider(),
        store,
        user_id=_UID,
        extractor=extractor,
        use_episodic=True,
        use_graph=True,
    )
    eng.memorize("Swizel likes apples")
    assert extractor.calls == 1  # ONE folded call feeds BOTH episodic + graph
    assert len(eng.timeline()) == 1  # episodic event written
    assert len(store.get_graph(_UID)["edges"]) == 1  # graph edge written


# -- best-effort: a bad extraction never breaks memorize -----------------------------------


def test_graph_extraction_generate_failure_never_breaks_memorize(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")

    class _BoomProvider(FakeProvider):
        def generate(self, prompt, *, temperature=0.0, max_output_tokens=512):
            raise RuntimeError("extractor exploded")

    eng = CortexMemory(
        FakeProvider(), store, user_id=_UID, extractor=_BoomProvider(), use_graph=True
    )
    mem = eng.memorize("A durable fact.")  # must NOT raise despite the extractor blowing up
    assert eng.count() == 1
    assert store.get(mem.id, _UID) is not None
    assert store.get_graph(_UID)["entities"] == []  # generate failed -> no graph at all


def test_graph_garbage_extraction_degrades_to_self_only(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    eng = _engine(store, _extractor("total nonsense, not json"))
    mem = eng.memorize("Some fact")
    graph = store.get_graph(_UID)
    # generate succeeded but parsed empty: self root is created, memory links to self, no edges
    assert [e["type"] for e in graph["entities"]] == ["self"]
    assert graph["edges"] == []
    self_id = graph["entities"][0]["id"]
    links = {(lk["memory_id"], lk["entity_id"], lk["role"]) for lk in graph["memory_links"]}
    assert (mem.id, self_id, "subject") in links


# -- byte-identical-when-off guards --------------------------------------------------------


def test_byte_identical_when_graph_off_writes_no_graph_rows(tmp_path):
    """Defaults (no extractor, graph off): memorize+recall behave exactly as pre-G2."""
    store = SQLiteStore(tmp_path / "m.db")
    eng = CortexMemory(FakeProvider(), store, user_id=_UID)  # use_graph default False, no extractor
    assert eng.use_graph is False

    eng.memorize("Swizel likes apples")
    eng.memorize("My girlfriend's name is Swizel")
    hits = eng.recall("What does Swizel like?", limit=2)
    assert hits  # unchanged recall behaviour

    for table in ("entities", "entity_edges", "memory_entities"):
        assert store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_graph_on_but_no_extractor_is_a_no_op(tmp_path):
    """Safety: use_graph on but NO extractor configured writes nothing (byte-identical)."""
    store = SQLiteStore(tmp_path / "m.db")
    eng = CortexMemory(FakeProvider(), store, user_id=_UID, use_graph=True)  # extractor is None
    eng.memorize("Swizel likes apples")
    for table in ("entities", "entity_edges", "memory_entities"):
        assert store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
