"""Wiring tests for the opt-in write-time enrichment flags in the stdio MCP server.

These cover the thin ``_build_engine_from_env`` wiring the smoke tests in ``test_mcp_server.py``
don't: that the CORTEX_GRAPH / CORTEX_EPISODIC / CORTEX_DEDUP / CORTEX_SOFT_UPDATE flags default
OFF (engine byte-identical to a bare one), that enabling graph makes ``recall_about`` return a real
dossier after a memorize that mints an entity, and that ``recall_timeline`` degrades to a clear
"enable episodic" notice when episodic is off. All OFFLINE with a FakeProvider — no network, no SDK.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

pytest.importorskip("fastmcp")

from cortex.mcp import server  # noqa: E402
from cortex.memory import CortexMemory  # noqa: E402
from cortex.providers.fake import FakeProvider  # noqa: E402
from cortex.store.sqlite_store import SQLiteStore  # noqa: E402


def _call_tool(name: str, args: dict) -> str:
    """Invoke an MCP tool through FastMCP dispatch and return its text output."""
    res = asyncio.run(server.mcp.call_tool(name, args))
    blocks = getattr(res, "content", res)
    return blocks[0].text if isinstance(blocks, list) and blocks else str(blocks)


def _fake_gemini_factory(
    responder: Callable[[str], str] | None = None,
) -> Callable[..., FakeProvider]:
    """A GeminiProvider stand-in: swallow the Gemini kwargs, return an OFFLINE FakeProvider.

    ``_build_engine_from_env`` constructs a GeminiProvider for the embedder (and, when a flag is on,
    a second one for the extractor). Patching the class with this factory keeps the whole build
    offline while exercising the real wiring/branching.
    """

    def _factory(**_kwargs: object) -> FakeProvider:
        return FakeProvider(responder=responder)

    return _factory


# Canned folded episodic+graph extraction the fake extractor returns for the ONE generate call
# memorize makes when the graph flag is on. It mints a single "Alice" person entity and a
# self→manager→Alice relation so recall_about has a real dossier to render.
_GRAPH_JSON = (
    '{"event_time": null, "actor": null, "location": null, "event_type": null, '
    '"entities": [{"name": "Alice", "type": "person"}], '
    '"relations": [{"src": "self", "label": "manager", "dst": "Alice"}], '
    '"subject": "Alice"}'
)


def _graph_responder(_prompt: str) -> str:
    return _GRAPH_JSON


def _base_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set a fake key + in-memory DB and clear every enrichment flag (start from a clean slate)."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("CORTEX_DB_PATH", ":memory:")
    for flag in ("CORTEX_GRAPH", "CORTEX_EPISODIC", "CORTEX_DEDUP", "CORTEX_SOFT_UPDATE"):
        monkeypatch.delenv(flag, raising=False)


def test_flags_default_off_builds_bare_engine(monkeypatch):
    """With no flags set, the engine has every enrichment off and no extractor (byte-identical)."""
    _base_env(monkeypatch)
    monkeypatch.setattr("cortex.providers.gemini.GeminiProvider", _fake_gemini_factory())

    engine = server._build_engine_from_env()

    assert engine.use_graph is False
    assert engine.use_episodic is False
    assert engine.use_dedup is False
    assert engine.use_soft_update is False
    assert engine.extractor is None
    assert engine.arbiter is None


def test_graph_flag_enables_recall_about_dossier(monkeypatch):
    """CORTEX_GRAPH=1 wires an extractor; a memorize mints an entity recall_about can render."""
    _base_env(monkeypatch)
    monkeypatch.setenv("CORTEX_GRAPH", "1")
    monkeypatch.setattr(
        "cortex.providers.gemini.GeminiProvider", _fake_gemini_factory(_graph_responder)
    )

    engine = server._build_engine_from_env()
    assert engine.use_graph is True
    assert engine.extractor is not None
    monkeypatch.setattr(server, "_ENGINE", engine)

    assert "Stored memory" in _call_tool("memorize", {"content": "Alice is my manager"})

    out = _call_tool("recall_about", {"entity": "Alice"})
    assert "Alice (person)" in out  # the minted entity, header
    assert "manager" in out  # the self→Alice relationship
    assert "Alice is my manager" in out  # the source memory, attached


def test_graph_flag_truthy_variants(monkeypatch):
    """The flag helper accepts the documented truthy spellings and rejects everything else."""
    for raw in ("1", "true", "TRUE", "yes", "on", "  On  "):
        monkeypatch.setenv("CORTEX_GRAPH", raw)
        assert server._env_flag("CORTEX_GRAPH") is True, raw
    for raw in ("0", "false", "no", "off", "", "maybe"):
        monkeypatch.setenv("CORTEX_GRAPH", raw)
        assert server._env_flag("CORTEX_GRAPH") is False, raw
    monkeypatch.delenv("CORTEX_GRAPH", raising=False)
    assert server._env_flag("CORTEX_GRAPH") is False  # unset → off


def test_recall_timeline_off_message(monkeypatch):
    """With episodic off there are no dated events, so recall_timeline returns the enable notice."""
    engine = CortexMemory(FakeProvider(), SQLiteStore(":memory:"), user_id="t")
    monkeypatch.setattr(server, "_ENGINE", engine)
    engine.memorize("An undated fact about cats")

    out = _call_tool("recall_timeline", {})
    assert out == "(no dated events; enable CORTEX_EPISODIC=1)"


def test_recall_timeline_renders_dated_events(monkeypatch):
    """When episodic rows exist, the timeline leads with event_time and appends actor/location."""
    engine = CortexMemory(FakeProvider(), SQLiteStore(":memory:"), user_id="t")
    monkeypatch.setattr(server, "_ENGINE", engine)
    mem = engine.memorize("Moved to Goa")
    engine.store.add_event(
        memory_id=mem.id,
        user_id="t",
        event_time="2025-01-15",
        ingest_time=mem.created_at,
        actor="David",
        location="Goa",
        event_type="moved",
    )

    out = _call_tool("recall_timeline", {})
    assert "2025-01-15" in out
    assert "David" in out and "Goa" in out
    assert "Moved to Goa" in out
