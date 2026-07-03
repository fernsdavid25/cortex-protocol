"""Cortex local MCP server (stdio).

Exposes the Cortex memory engine as MCP tools so ANY MCP client — Claude Code, Cursor,
Claude Desktop — gets persistent, cross-session memory. This is the self-host product
surface (Goal.md Phase 4.1): BYOK (the user's own Gemini key), a local SQLite DB, and
**zero phone-home**. Remote streamable-HTTP + OAuth ("Sign in with Cortex") come later
(Phase 4.3); this stdio server is the stepping stone.

Run:
    uvx cortex-mcp                 # once published
    python -m cortex.mcp.server    # from a checkout

Environment (all optional except the key):
    GEMINI_API_KEY / GOOGLE_API_KEY   BYOK — required to embed (nothing is bundled)
    CORTEX_DB_PATH                     memory file (default: ~/.cortex/memory.db)
    CORTEX_USER_ID                     namespace within the DB (default: "local")
    CORTEX_EMBED_MODEL                 default: gemini-embedding-001
    CORTEX_EMBED_DIM                   default: 768
    CORTEX_TOP_K                       default recall depth (default: 5)

Add to a Claude Code / Cursor MCP config:
    {"mcpServers": {"cortex": {"command": "uvx", "args": ["cortex-mcp"],
      "env": {"GEMINI_API_KEY": "..."}}}}
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from fastmcp import FastMCP

from cortex.memory import CortexMemory
from cortex.store.sqlite_store import Memory, SQLiteStore, make_metadata, normalize_kind

# Lists the SIX kinds so the agent classifies each memory; mirrors the hosted server's tool. An
# omitted/invalid kind is left unset (defaults to "fact" downstream), so kind=None is byte-identical
# to the pre-kind behaviour.
_MEMORIZE_DESC = (
    "Save a durable memory for the user (facts, preferences, decisions, project context worth "
    "remembering across sessions). Optionally set `kind` to the ONE that best classifies it: "
    "preference = how the user likes the agent to behave; fact = stable info about the user or "
    "world; project = ongoing work; instruction = a hard rule to always follow; event = a dated "
    "thing that happened; relationship = a person the user knows. Optionally attach short `tags`. "
    "Do NOT store secrets."
)

# Sharp routing description mirroring the hosted server: the entity-enumeration query that ranked
# `recall` can't answer.
_ABOUT_DESC = (
    "Exhaustive dossier about ONE specific entity — a person, place, project, org, thing, or the "
    "user themselves. Use for 'tell me everything about X', 'what do you know about my ⟨…⟩', or "
    "'who is X'. Returns the entity, its labeled relationships (both directions), and every memory "
    "about it — the full enumeration that ranked `recall` can't give. For a single fact or fuzzy "
    "lookup use `recall` instead."
)

mcp: FastMCP = FastMCP(
    name="cortex",
    instructions=(
        "Cortex is the user's persistent, cross-session memory. Use `memorize` to save "
        "durable facts, preferences, decisions, and project context the user will want "
        "remembered later. Use `recall` at the start of a task to load anything relevant "
        "you may have saved before. Memory persists across sessions and across agents."
    ),
)

_ENGINE: CortexMemory | None = None


def _env_int(name: str, default: int) -> int:
    """Read an int env var, with a clear error on a malformed value (vs an opaque ValueError)."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}.") from None


def _build_engine_from_env() -> CortexMemory:
    """Construct the engine from environment config (BYOK). Raises if no key is set."""
    from cortex.providers.gemini import GeminiProvider

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "No API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY) so Cortex can embed "
            "memories with your own key (BYOK)."
        )
    db_path = os.environ.get("CORTEX_DB_PATH") or str(Path.home() / ".cortex" / "memory.db")
    user_id = os.environ.get("CORTEX_USER_ID", "local")
    embed_model = os.environ.get("CORTEX_EMBED_MODEL", "gemini-embedding-001")
    embed_dim = _env_int("CORTEX_EMBED_DIM", 768)
    top_k = _env_int("CORTEX_TOP_K", 5)

    provider = GeminiProvider(embed_model=embed_model, embed_dim=embed_dim, api_key=key)
    store = SQLiteStore(db_path)
    return CortexMemory(provider, store, user_id=user_id, top_k=top_k)


def _engine() -> CortexMemory:
    """Lazily build a process-wide engine so importing this module needs no API key."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = _build_engine_from_env()
    return _ENGINE


def _format_memories(memories: Sequence[Memory]) -> str:
    """Render memories as a compact, agent-readable numbered list."""
    if not memories:
        return "(no relevant memories found)"
    lines: list[str] = []
    for i, m in enumerate(memories, start=1):
        date = m.created_at.split("T")[0]
        tags = m.metadata.get("tags")
        tag_str = ""
        if isinstance(tags, list) and tags:
            tag_str = f"  [tags: {', '.join(str(t) for t in tags)}]"
        lines.append(f"{i}. ({date} · id={m.id[:8]}) {m.content}{tag_str}")
    return "\n".join(lines)


def _format_dossier(dossier: dict[str, object], note: str, limit: int) -> str:
    """Render an entity dossier (header + relationships + newest-first memories, capped)."""
    entity = dossier["entity"]
    edges = dossier["edges"]
    memories = dossier["memories"]
    assert isinstance(entity, dict) and isinstance(edges, list) and isinstance(memories, list)
    lines: list[str] = []
    if note:
        lines.append(note)
    lines.append(f"{entity.get('name')} ({entity.get('type')})")
    if edges:
        lines.append("")
        lines.append("Relationships:")
        for e in edges:
            if e["direction"] == "out":
                lines.append(f"  → {e['label']}: {e['dst_name'] or '?'}")
            else:
                lines.append(f"  ← {e['label']}: {e['src_name'] or '?'}")
    capped = memories[:limit]
    lines.append("")
    if capped:
        lines.append(f"Memories ({len(capped)}):")
        for i, m in enumerate(capped, start=1):
            date = m.created_at.split("T")[0]
            lines.append(f"  {i}. ({date} · id={m.id[:8]}) {m.content}")
    else:
        lines.append("Memories: (none recorded yet)")
    return "\n".join(lines)


@mcp.tool(description=_MEMORIZE_DESC)
def memorize(content: str, kind: str | None = None, tags: list[str] | None = None) -> str:
    engine = _engine()
    meta = make_metadata(tags)
    # Validate against the six kinds; unknown/None is left unset so it defaults to "fact" later.
    valid_kind = normalize_kind(kind)
    if valid_kind is not None:
        meta["kind"] = valid_kind
    mem = engine.memorize(content, metadata=meta)
    return f"Stored memory {mem.id[:8]} ({engine.count()} total)."


# Upper bound on agent-supplied result sizes — keeps a runaway `limit` from materialising
# the whole store / a giant response.
_MAX_LIMIT = 1000


@mcp.tool()
def recall(query: str, limit: int = 5) -> str:
    """Retrieve the memories most relevant to a query.

    Call this at the start of a task (or whenever you need prior context) to load what the
    user has saved before. Returns ranked raw memories; synthesise your answer from them.
    """
    limit = max(1, min(limit, _MAX_LIMIT))
    memories = _engine().recall(query, limit=limit)
    return _format_memories(memories)


@mcp.tool(description=_ABOUT_DESC)
def recall_about(entity: str, limit: int = 20) -> str:
    engine = _engine()
    limit = max(1, min(limit, _MAX_LIMIT))
    cands = engine.store.get_entity_by_name(engine.user_id, entity)
    if not cands:
        return (
            f'No entity named "{entity}" is in memory yet. '
            f'Try recall("{entity}") for a semantic search instead.'
        )
    best = cands[0]
    note = ""
    if len(cands) > 1:
        others = ", ".join(str(c["name"]) for c in cands[1:])
        note = f'Several matches for "{entity}"; showing {best["name"]}. Did you mean: {others}?'
    dossier = engine.store.get_entity_dossier(engine.user_id, str(best["id"]))
    return _format_dossier(dossier, note, limit)


@mcp.tool()
def list_memories(limit: int = 20) -> str:
    """List the most recently saved memories (newest first)."""
    limit = max(1, min(limit, _MAX_LIMIT))
    return _format_memories(_engine().list_memories(limit=limit))


@mcp.tool()
def forget(memory_id: str) -> str:
    """Delete a memory by its id (the short id shown in recall/list results works too).

    A short id that matches multiple memories is refused (use the full id) so nothing is
    deleted by accident.
    """
    return _engine().forget_prefix(memory_id)


def main() -> None:
    """Console entry point (`cortex-mcp`): load a local .env if present, then serve stdio."""
    import atexit

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # pragma: no cover - dotenv is a declared dependency
        pass
    # Flush/close the SQLite store cleanly on shutdown (only if an engine was built).
    atexit.register(lambda: _ENGINE.close() if _ENGINE is not None else None)
    mcp.run()


if __name__ == "__main__":  # pragma: no cover
    main()
