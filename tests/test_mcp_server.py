"""Smoke tests for the MCP server wrapper.

The engine/store have full coverage elsewhere; here we only check the thin MCP layer:
that it imports, registers the expected tools, formats results, and fails cleanly with no
key. Skipped entirely if ``fastmcp`` isn't installed (it's an optional runtime dep).
"""

from __future__ import annotations

import asyncio
import re

import pytest

pytest.importorskip("fastmcp")

from cortex.mcp import server  # noqa: E402
from cortex.memory import CortexMemory  # noqa: E402
from cortex.providers.fake import FakeProvider  # noqa: E402
from cortex.store.sqlite_store import Memory, SQLiteStore  # noqa: E402


def _inject_engine() -> CortexMemory:
    """Replace the lazy singleton with an offline FakeProvider-backed engine for tool tests."""
    eng = CortexMemory(FakeProvider(), SQLiteStore(":memory:"), user_id="t")
    server._ENGINE = eng
    return eng


def _call_tool(name: str, args: dict) -> str:
    """Invoke an MCP tool through FastMCP dispatch and return its text output."""
    res = asyncio.run(server.mcp.call_tool(name, args))
    blocks = getattr(res, "content", res)
    return blocks[0].text if isinstance(blocks, list) and blocks else str(blocks)


def test_build_engine_requires_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="No API key"):
        server._build_engine_from_env()


def test_format_memories_empty_and_populated():
    assert server._format_memories([]) == "(no relevant memories found)"
    mem = Memory(
        id="abcdef0123456789" * 2,
        user_id="local",
        content="The user lives in Goa.",
        created_at="2026-06-28T12:00:00+00:00",
        metadata={"tags": ["bio"]},
    )
    out = server._format_memories([mem])
    assert "1. (2026-06-28" in out
    assert "id=abcdef01" in out
    assert "The user lives in Goa." in out
    assert "[tags: bio]" in out


def test_tools_are_registered():
    lister = getattr(server.mcp, "list_tools", None)
    if lister is None:  # pragma: no cover - guards against a future FastMCP API change
        pytest.skip("FastMCP list_tools() API not available")
    tools = asyncio.run(lister())
    names = set(tools) if isinstance(tools, dict) else {getattr(t, "name", t) for t in tools}
    assert {"memorize", "recall", "list_memories", "forget"} <= names


def test_env_int_rejects_malformed(monkeypatch):
    monkeypatch.setenv("CORTEX_EMBED_DIM", "abc")
    with pytest.raises(RuntimeError, match="must be an integer"):
        server._env_int("CORTEX_EMBED_DIM", 768)
    monkeypatch.delenv("CORTEX_EMBED_DIM", raising=False)
    assert server._env_int("CORTEX_EMBED_DIM", 768) == 768  # default when unset


def test_mcp_tools_roundtrip():
    _inject_engine()
    stored = _call_tool("memorize", {"content": "The dog is Rex.", "tags": ["pet"]})
    assert "Stored memory" in stored
    listed = _call_tool("list_memories", {"limit": 10})
    assert "Rex" in listed
    assert "Rex" in _call_tool("recall", {"query": "what is the dog", "limit": 5})

    match = re.search(r"id=([0-9a-f]{8})", listed)
    assert match is not None, "list output should show a short id"
    assert "Forgot" in _call_tool("forget", {"memory_id": match.group(1)})


def test_recall_limit_is_clamped_no_crash():
    eng = _inject_engine()
    eng.memorize("a fact about cats")
    # A runaway limit must be clamped internally, not blow up.
    assert "cats" in _call_tool("recall", {"query": "cats", "limit": 10_000_000})


def test_recall_about_local_dossier_and_graceful_miss():
    """The local `recall_about` renders an entity dossier (header + relationships + memories) and
    degrades gracefully when nothing resolves. Graph rows are seeded directly (the local engine has
    no extractor) — recall_about is a pure keyed read over them."""
    eng = _inject_engine()
    eng.memorize("Swizel is my girlfriend")
    mem = eng.list_memories()[0]
    self_id = eng.store.ensure_self_entity("t")
    swizel = eng.store.upsert_entity("t", "Swizel", "person")
    eng.store.add_entity_edge("t", self_id, "girlfriend", swizel, mem.id)
    eng.store.link_memory_entity("t", mem.id, swizel, "subject")

    out = _call_tool("recall_about", {"entity": "Swizel"})
    assert "Swizel (person)" in out
    assert "girlfriend" in out and "You" in out  # the relationship to self
    assert "Swizel is my girlfriend" in out  # the attached memory

    miss = _call_tool("recall_about", {"entity": "Nobody"})
    assert "Nobody" in miss and "recall(" in miss  # graceful, no crash
