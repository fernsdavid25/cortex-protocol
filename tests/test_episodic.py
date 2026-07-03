"""Offline, deterministic tests for the L4 episodic-memory MVP (FakeProvider + SQLite).

Everything here is additive and OFF by default. The last test proves the byte-identical
guarantee: a default ``CortexMemory`` (no extractor, episodic off) memorizes and recalls
exactly as before and never touches the ``events`` table. No network, no live LLM calls.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from cortex.memory import CortexMemory
from cortex.providers.fake import FakeProvider
from cortex.reader.reader import (
    build_episodic_extraction_prompt,
    parse_episodic_extraction,
)
from cortex.store.sqlite_store import Memory, SQLiteStore

_INGEST = "2026-07-02T00:00:00+00:00"


class _CountingProvider(FakeProvider):
    """A FakeProvider that counts ``generate`` calls (to prove no call happens when off)."""

    def __init__(self, responder: Callable[[str], str]) -> None:
        super().__init__(responder=responder)
        self.calls = 0

    def generate(self, prompt: str, *, temperature: float = 0.0, max_output_tokens: int = 512):
        self.calls += 1
        return super().generate(
            prompt, temperature=temperature, max_output_tokens=max_output_tokens
        )


def _extractor(event_json: str) -> _CountingProvider:
    """A canned extractor: returns ``event_json`` on the episodic prompt, else abstains."""

    def responder(prompt: str) -> str:
        if "episodic event extractor" in prompt:
            return event_json
        return "I don't know."

    return _CountingProvider(responder)


def _add(store: SQLiteStore, mem_id: str, content: str, user_id: str = "u1") -> Memory:
    mem = Memory(id=mem_id, user_id=user_id, content=content, created_at=_INGEST)
    store.add(mem, [1.0, 0.0])
    return mem


# -- store: add_event + timeline -----------------------------------------------------------


def test_add_event_and_timeline_roundtrip_and_ordering(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    _add(store, "a" * 32, "moved to Goa")
    _add(store, "b" * 32, "started job")
    _add(store, "c" * 32, "undated note")
    store.add_event(
        "a" * 32,
        "u1",
        event_time="2026-03-01",
        ingest_time=_INGEST,
        actor="David",
        location="Goa",
        event_type="moved",
    )
    store.add_event(
        "b" * 32,
        "u1",
        event_time="2026-01-15",
        ingest_time=_INGEST,
        actor=None,
        location=None,
        event_type="job",
    )
    store.add_event(
        "c" * 32,
        "u1",
        event_time=None,
        ingest_time=_INGEST,
        actor=None,
        location=None,
        event_type=None,
    )
    timeline = store.timeline("u1")
    # ordered by event_time ascending, NULL event_time last
    assert [m.content for m in timeline] == ["started job", "moved to Goa", "undated note"]
    # the event fields ride along under metadata["event"]
    assert timeline[1].metadata["event"] == {
        "event_time": "2026-03-01",
        "actor": "David",
        "location": "Goa",
        "event_type": "moved",
        "episode_id": None,
    }


def test_timeline_respects_since_until_and_limit(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    for i, date in enumerate(["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"]):
        mid = f"{i:032x}"
        _add(store, mid, f"event {i}")
        store.add_event(
            mid,
            "u1",
            event_time=date,
            ingest_time=_INGEST,
            actor=None,
            location=None,
            event_type=None,
        )
    assert [m.content for m in store.timeline("u1", since="2026-02-15")] == ["event 2", "event 3"]
    assert [m.content for m in store.timeline("u1", until="2026-02-15")] == ["event 0", "event 1"]
    assert [m.content for m in store.timeline("u1", since="2026-01-15", until="2026-03-15")] == [
        "event 1",
        "event 2",
    ]
    assert [m.content for m in store.timeline("u1", limit=2)] == ["event 0", "event 1"]


def test_timeline_is_user_scoped(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    _add(store, "a" * 32, "mine", user_id="alice")
    store.add_event(
        "a" * 32,
        "alice",
        event_time="2026-01-01",
        ingest_time=_INGEST,
        actor=None,
        location=None,
        event_type=None,
    )
    assert len(store.timeline("alice")) == 1
    assert store.timeline("bob") == []


# -- engine: memorize wiring (gated) -------------------------------------------------------


def test_memorize_with_episodic_writes_event(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    canned = (
        '{"event_time": "2026-06-25", "actor": "David", '
        '"location": "Goa", "event_type": "moved apartment"}'
    )
    extractor = _extractor(canned)
    eng = CortexMemory(FakeProvider(), store, extractor=extractor, use_episodic=True)
    mem = eng.memorize("I moved to a new apartment in Goa last week.")
    assert extractor.calls == 1  # exactly one extraction call
    timeline = eng.timeline()
    assert len(timeline) == 1
    assert timeline[0].id == mem.id
    assert timeline[0].metadata["event"] == {
        "event_time": "2026-06-25",
        "actor": "David",
        "location": "Goa",
        "event_type": "moved apartment",
        "episode_id": None,
    }


def test_memorize_without_episodic_makes_no_extractor_call(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    extractor = _extractor('{"event_time": "2026-06-25", "actor": null}')
    # use_episodic defaults False; even WITH an extractor present, nothing episodic runs.
    eng = CortexMemory(FakeProvider(), store, extractor=extractor, use_episodic=False)
    eng.memorize("I moved to a new apartment in Goa last week.")
    assert extractor.calls == 0  # the gate short-circuits before any generate()
    assert eng.timeline() == []  # no event row written
    assert store._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0


def test_relative_date_resolution_stores_absolute_date(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    # The model resolves "last week" (relative to the message date) to an absolute ISO date.
    canned = '{"event_time": "2026-06-25", "actor": null, "location": null, "event_type": "trip"}'
    eng = CortexMemory(FakeProvider(), store, extractor=_extractor(canned), use_episodic=True)
    mem = eng.memorize("We went on a trip last week.")
    stored = eng.timeline()[0].metadata["event"]["event_time"]
    assert stored == "2026-06-25"  # the resolved absolute date, not the ingest date
    assert stored != mem.created_at


def test_episodic_extraction_failure_never_breaks_memorize(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")

    class _BoomProvider(FakeProvider):
        def generate(self, prompt, *, temperature=0.0, max_output_tokens=512):
            raise RuntimeError("extractor exploded")

    eng = CortexMemory(FakeProvider(), store, extractor=_BoomProvider(), use_episodic=True)
    mem = eng.memorize("A durable fact.")  # must NOT raise despite the extractor blowing up
    assert eng.count() == 1
    assert store.get(mem.id, "local") is not None
    assert eng.timeline() == []  # extraction failed -> no event, memory still stored


# -- parser --------------------------------------------------------------------------------


def test_parse_episodic_extraction_valid_json():
    out = parse_episodic_extraction(
        '{"event_time": "2026-06-25", "actor": "David", "location": "Goa", "event_type": "moved"}',
        "2026-07-02",
    )
    assert out == {
        "event_time": "2026-06-25",
        "actor": "David",
        "location": "Goa",
        "event_type": "moved",
    }


def test_parse_episodic_extraction_markdown_fenced():
    text = '```json\n{"event_time": "2026-06-25", "actor": null, "event_type": "trip"}\n```'
    out = parse_episodic_extraction(text, "2026-07-02")
    assert out["event_time"] == "2026-06-25"
    assert out["event_type"] == "trip"
    assert out["actor"] is None


def test_parse_episodic_extraction_prose_wrapped():
    text = 'Here is the event: {"event_time": "2026-05-01", "actor": "Ana"} — hope that helps!'
    out = parse_episodic_extraction(text, "2026-07-02")
    assert out["event_time"] == "2026-05-01"
    assert out["actor"] == "Ana"


def test_parse_episodic_extraction_garbage_falls_back():
    out = parse_episodic_extraction("this is not json at all", "2026-07-02")
    assert out == {
        "event_time": "2026-07-02",
        "actor": None,
        "location": None,
        "event_type": None,
    }


@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
def test_parse_episodic_extraction_empty_falls_back(text):
    assert parse_episodic_extraction(text, "2026-07-02")["event_time"] == "2026-07-02"


def test_parse_episodic_extraction_null_event_time_falls_back_to_ingest():
    out = parse_episodic_extraction(
        '{"event_time": null, "actor": "David", "location": null, "event_type": "note"}',
        "2026-07-02",
    )
    assert out["event_time"] == "2026-07-02"  # null date -> ingest-time fallback
    assert out["actor"] == "David"


def test_build_episodic_extraction_prompt_has_marker_and_context():
    prompt = build_episodic_extraction_prompt("I moved last week.", "2026-07-02")
    assert "episodic event extractor" in prompt  # FakeProvider responder marker
    assert "MESSAGE DATE: 2026-07-02" in prompt
    assert "I moved last week." in prompt


# -- byte-identical-when-off guard ---------------------------------------------------------


def test_byte_identical_when_off_no_events_interaction(tmp_path):
    """Defaults (no extractor, episodic off): memorize+recall behave exactly as pre-L4."""
    store = SQLiteStore(tmp_path / "m.db")
    eng = CortexMemory(FakeProvider(), store)  # no extractor, use_episodic defaults False
    assert eng.use_episodic is False
    assert eng.extractor is None

    eng.memorize("The user's dog is named Rex.")
    eng.memorize("The user enjoys hiking mountains on weekends.")
    eng.memorize("The user works as a backend engineer.")
    hits = eng.recall("What is the name of the user's dog?", limit=3)
    assert hits and "Rex" in hits[0].content  # unchanged recall behavior

    # Zero episodic side effects: no timeline, no rows in the events table.
    assert eng.timeline() == []
    assert store._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
