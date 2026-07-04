"""Cortex quickstart — memorize a few facts, then recall them. Runs fully offline.

Run it with:

    uv run examples/quickstart.py

By default this uses the deterministic, offline ``FakeProvider`` (no API key, no network),
so you can watch the memorize -> recall loop end to end for free. If you export a real key
(``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``), it transparently switches to the real
``GeminiProvider`` and embeds against Gemini instead — same code, same store, real vectors.
"""

from __future__ import annotations

import os

from cortex.memory import CortexMemory
from cortex.providers.base import LLMProvider
from cortex.providers.fake import FakeProvider
from cortex.store.sqlite_store import SQLiteStore


def build_provider() -> LLMProvider:
    """Pick a provider: the real Gemini adapter if a key is set, else the offline FakeProvider.

    Cortex is bring-your-own-key: credentials are read only from the environment, never bundled.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        # Lazy import: google-genai is only needed on the real path (offline runs never import it).
        from cortex.providers.gemini import GeminiProvider

        print("Using GeminiProvider — real embeddings via your key.")
        return GeminiProvider(api_key=api_key)
    print("No GEMINI_API_KEY set — using the offline FakeProvider (deterministic, free).")
    return FakeProvider(dim=256)


def main() -> None:
    provider = build_provider()

    # An in-memory SQLite store keeps this demo self-contained (nothing is written to disk).
    # In real use the store is a file at ~/.cortex/memory.db — pass that path instead of ":memory:".
    store = SQLiteStore(":memory:")
    memory = CortexMemory(provider=provider, store=store, user_id="demo")

    facts = [
        "My name is David and I live in Goa, India.",
        "I'm building Cortex, an open-source memory MCP server for AI agents.",
        "My favourite programming language is Python.",
    ]
    print("\nMemorizing 3 facts:")
    for fact in facts:
        record = memory.memorize(fact)  # one embed + persist; no LLM call
        print(f"  stored {record.id[:8]}  {record.content}")

    query = "What is David building?"
    print(f"\nrecall({query!r}):")
    # recall embeds the query once and runs hybrid dense + BM25 retrieval — no server-side
    # generation. It returns the raw ranked memories; a real agent reasons over them.
    for rank, hit in enumerate(memory.recall(query, limit=3), start=1):
        print(f"  {rank}. {hit.content}")

    print(f"\nTotal memories stored for this user: {memory.count()}")
    memory.close()


if __name__ == "__main__":
    main()
