"""A+ polish tests for the MCP server tools and the Gemini provider (OFFLINE, deterministic).

Covers the four wiring fixes the smoke tests don't:

* ``recall_about`` names CORTEX_GRAPH in its empty-state when the graph layer is off (vs implying
  the entity doesn't exist), and keeps the "not known yet" wording when the graph IS on.
* ``recall`` with no ``limit`` honours the engine's configured ``top_k`` (so CORTEX_TOP_K, which the
  engine reads, is actually wired through the tool) while an explicit limit still overrides it.
* ``_env_int`` rejects a 0 / negative sizing knob with the same clear RuntimeError as a malformed
  one, instead of failing opaquely later.
* ``GeminiProvider.embed`` splits inputs into batches, guards a count mismatch, accounts tokens, and
  resolves its BYOK key from GEMINI_API_KEY / GOOGLE_API_KEY — all driven by a fake genai client
  injected via ``sys.modules`` so nothing touches the SDK or the network.
"""

from __future__ import annotations

import asyncio
import sys
import types as pytypes
from collections.abc import Sequence

import pytest

pytest.importorskip("fastmcp")

from cortex.mcp import server  # noqa: E402
from cortex.memory import CortexMemory  # noqa: E402
from cortex.providers.fake import FakeProvider  # noqa: E402
from cortex.providers.gemini import GeminiProvider  # noqa: E402
from cortex.store.sqlite_store import SQLiteStore  # noqa: E402


def _call_tool(name: str, args: dict) -> str:
    """Invoke an MCP tool through FastMCP dispatch and return its text output."""
    res = asyncio.run(server.mcp.call_tool(name, args))
    blocks = getattr(res, "content", res)
    return blocks[0].text if isinstance(blocks, list) and blocks else str(blocks)


def _inject_engine(monkeypatch: pytest.MonkeyPatch, **kwargs: object) -> CortexMemory:
    """Replace the lazy singleton with an offline FakeProvider-backed engine for tool tests."""
    eng = CortexMemory(FakeProvider(), SQLiteStore(":memory:"), user_id="t", **kwargs)
    monkeypatch.setattr(server, "_ENGINE", eng)
    return eng


# --------------------------------------------------------------------------------------------------
# Fix 1: recall_about off-state names the disabled flag
# --------------------------------------------------------------------------------------------------


def test_recall_about_off_state_names_cortex_graph(monkeypatch):
    """With the graph layer off, a miss says the graph is DISABLED and names CORTEX_GRAPH."""
    _inject_engine(monkeypatch, use_graph=False)

    out = _call_tool("recall_about", {"entity": "Alice"})

    assert "CORTEX_GRAPH" in out
    assert "disabled" in out.lower()
    assert 'recall("Alice")' in out  # still offers the semantic-search redirect


def test_recall_about_on_but_unknown_entity_keeps_not_yet_wording(monkeypatch):
    """With the graph ON, a miss must NOT claim it is disabled; the entity is just unknown."""
    _inject_engine(monkeypatch, use_graph=True)

    out = _call_tool("recall_about", {"entity": "Ghost"})

    assert "in memory yet" in out
    assert "disabled" not in out.lower()
    assert "CORTEX_GRAPH" not in out


# --------------------------------------------------------------------------------------------------
# Fix 2: recall with no limit uses the engine's configured top_k (CORTEX_TOP_K)
# --------------------------------------------------------------------------------------------------


def test_recall_no_limit_uses_engine_top_k(monkeypatch):
    """Omitting `limit` recalls exactly the engine's top_k, so CORTEX_TOP_K is wired through."""
    eng = _inject_engine(monkeypatch, top_k=2)
    for text in ("cats are soft", "cats sleep lots", "cats hunt mice"):
        eng.memorize(text)

    out = _call_tool("recall", {"query": "cats"})

    # Every rendered memory line carries an "id=" tag; top_k=2 caps the result at two of the three.
    assert out.count("id=") == 2


def test_recall_explicit_limit_overrides_top_k(monkeypatch):
    """An explicit limit still overrides the configured top_k for that one call."""
    eng = _inject_engine(monkeypatch, top_k=1)
    for text in ("cats are soft", "cats sleep lots", "cats hunt mice"):
        eng.memorize(text)

    out = _call_tool("recall", {"query": "cats", "limit": 3})

    assert out.count("id=") == 3


# --------------------------------------------------------------------------------------------------
# Fix 3: _env_int range guard
# --------------------------------------------------------------------------------------------------


def test_env_int_rejects_zero(monkeypatch):
    monkeypatch.setenv("CORTEX_TOP_K", "0")
    with pytest.raises(RuntimeError, match="must be >= 1"):
        server._env_int("CORTEX_TOP_K", 5)


def test_env_int_rejects_negative(monkeypatch):
    monkeypatch.setenv("CORTEX_EMBED_DIM", "-1")
    with pytest.raises(RuntimeError, match="must be >= 1"):
        server._env_int("CORTEX_EMBED_DIM", 768)


def test_env_int_accepts_positive_and_custom_minimum(monkeypatch):
    monkeypatch.setenv("CORTEX_TOP_K", "8")
    assert server._env_int("CORTEX_TOP_K", 5) == 8
    monkeypatch.setenv("CORTEX_TOP_K", "0")
    assert server._env_int("CORTEX_TOP_K", 5, minimum=0) == 0  # explicit minimum still honoured


# --------------------------------------------------------------------------------------------------
# Fix 4: GeminiProvider.embed driven by a fake genai client
# --------------------------------------------------------------------------------------------------


class _FakeEmbedding:
    def __init__(self, values: list[float]) -> None:
        self.values = values


class _FakeEmbedResponse:
    def __init__(self, embeddings: list[_FakeEmbedding]) -> None:
        self.embeddings = embeddings


class _FakeModels:
    """Records every embed_content call and returns one vector per input, minus ``drop``."""

    def __init__(self, dim: int, drop: int = 0) -> None:
        self.dim = dim
        self.drop = drop
        self.calls: list[list[str]] = []

    def embed_content(
        self, *, model: str, contents: list[str], config: object
    ) -> _FakeEmbedResponse:
        self.calls.append(list(contents))
        count = len(contents) - self.drop
        return _FakeEmbedResponse([_FakeEmbedding([0.1] * self.dim) for _ in range(count)])


class _FakeClient:
    def __init__(self, dim: int, drop: int = 0) -> None:
        self.models = _FakeModels(dim, drop)


@pytest.fixture
def fake_genai(monkeypatch):
    """Install a minimal fake ``google.genai`` so ``embed`` imports offline (SDK may be absent)."""
    genai_types = pytypes.ModuleType("google.genai.types")
    genai_types.EmbedContentConfig = lambda **kw: dict(kw)  # type: ignore[attr-defined]
    genai_types.GenerateContentConfig = lambda **kw: dict(kw)  # type: ignore[attr-defined]
    genai_types.HttpOptions = lambda **kw: dict(kw)  # type: ignore[attr-defined]
    genai_mod = pytypes.ModuleType("google.genai")
    genai_mod.types = genai_types  # type: ignore[attr-defined]
    google_mod = pytypes.ModuleType("google")
    google_mod.genai = genai_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)
    monkeypatch.setitem(sys.modules, "google.genai.types", genai_types)


def test_embed_splits_into_batches_and_counts_tokens(fake_genai):
    provider = GeminiProvider(embed_dim=4, api_key="k")
    client = _FakeClient(dim=4)
    provider._client = client

    texts: Sequence[str] = ["a b", "c", "d e f", "g", "h"]
    res = provider.embed(texts, batch_size=2)

    assert len(res.vectors) == 5
    assert all(len(v) == 4 for v in res.vectors)
    # 5 inputs at batch_size 2 → three calls sized 2, 2, 1.
    assert [len(c) for c in client.models.calls] == [2, 2, 1]
    # Token accounting approximates by word count across ALL inputs (2+1+3+1+1).
    assert res.input_tokens == 8


def test_embed_count_mismatch_raises(fake_genai):
    provider = GeminiProvider(embed_dim=4, api_key="k")
    provider._client = _FakeClient(dim=4, drop=1)  # returns one fewer vector than requested

    with pytest.raises(RuntimeError, match="count mismatch"):
        provider.embed(["a", "b"], batch_size=10)


def test_embed_passes_configured_model_and_dim(fake_genai):
    provider = GeminiProvider(embed_model="my-embed", embed_dim=7, api_key="k")
    client = _FakeClient(dim=7)
    provider._client = client
    captured: dict[str, object] = {}
    original = client.models.embed_content

    def _spy(*, model: str, contents: list[str], config: object) -> _FakeEmbedResponse:
        captured["model"] = model
        captured["config"] = config
        return original(model=model, contents=contents, config=config)

    client.models.embed_content = _spy  # type: ignore[method-assign]
    res = provider.embed(["hello world"])

    assert captured["model"] == "my-embed"
    assert captured["config"] == {"output_dimensionality": 7}
    assert len(res.vectors[0]) == 7


def test_api_key_fallback_prefers_gemini_then_google(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    # Explicit arg wins over the environment.
    assert GeminiProvider(api_key="explicit")._api_key == "explicit"

    # GEMINI_API_KEY is honoured...
    monkeypatch.setenv("GEMINI_API_KEY", "gem")
    assert GeminiProvider()._api_key == "gem"

    # ...and GOOGLE_API_KEY is the documented fallback when GEMINI_API_KEY is unset.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "goog")
    assert GeminiProvider()._api_key == "goog"

    # GEMINI_API_KEY takes precedence when both are present.
    monkeypatch.setenv("GEMINI_API_KEY", "gem")
    assert GeminiProvider()._api_key == "gem"
