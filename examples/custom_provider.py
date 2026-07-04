"""Write your own Cortex provider — a minimal, runnable ``LLMProvider`` subclass.

Cortex programs against ONE interface, ``cortex.providers.base.LLMProvider`` (see the API
reference). To support a new model backend — Claude, OpenAI, a local Ollama model, anything —
you implement just two methods:

    generate(prompt, *, temperature, max_output_tokens) -> GenResult   # text out
    embed(texts) -> EmbedResult                                        # vectors out

This example implements a tiny, fully offline provider (deterministic hash embeddings, a canned
``generate``) so it runs with no key and no network:

    uv run examples/custom_provider.py

To ship a real adapter, swap the method bodies for SDK calls — lazy-imported INSIDE the methods,
per the repo idiom, so importing the engine never drags in an optional SDK. Exposing
``embed_model`` / ``embed_dim`` attributes is optional but recommended: the store's
embedding-signature guard records them so a later model/dim change can't silently corrupt recall.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from cortex.memory import CortexMemory
from cortex.providers.base import EmbedResult, GenResult, LLMProvider
from cortex.store.sqlite_store import SQLiteStore


class EchoHashProvider(LLMProvider):
    """The smallest useful provider: a canned ``generate`` + deterministic hash ``embed``."""

    def __init__(self, embed_dim: int = 128) -> None:
        # Read by the store's signature guard (best-effort; both attributes are optional).
        self.embed_model = "echo-hash-v1"
        self.embed_dim = embed_dim

    def generate(
        self, prompt: str, *, temperature: float = 0.0, max_output_tokens: int = 512
    ) -> GenResult:
        # A real adapter calls its model SDK here. GenResult also carries token counts, used
        # for cost accounting — return the real counts when the SDK reports them.
        text = "I don't know."
        return GenResult(
            text=text,
            input_tokens=len(prompt.split()),
            output_tokens=len(text.split()),
        )

    def embed(self, texts: Sequence[str]) -> EmbedResult:
        # A real adapter returns the model's embeddings. Every vector MUST be non-empty and
        # exactly ``embed_dim`` long, or the engine rejects the write rather than corrupt recall.
        vectors = [self._embed_one(text) for text in texts]
        return EmbedResult(vectors=vectors, input_tokens=sum(len(t.split()) for t in texts))

    def _embed_one(self, text: str) -> list[float]:
        # Deterministic L2-normalized bag-of-words hash vector — enough to demo recall offline.
        vec = [0.0] * self.embed_dim
        for token in text.lower().split():
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.embed_dim
            vec[bucket] += 1.0
        norm = sum(value * value for value in vec) ** 0.5
        return [value / norm for value in vec] if norm else vec


def main() -> None:
    # Drop the custom provider straight into the engine — nothing else changes.
    memory = CortexMemory(
        provider=EchoHashProvider(),
        store=SQLiteStore(":memory:"),
        user_id="demo",
    )
    memory.memorize("Cortex is an open-source memory MCP server.")
    memory.memorize("It ships a Gemini adapter and programs against the LLMProvider interface.")

    print("recall('What is Cortex?'):")
    for hit in memory.recall("What is Cortex?", limit=2):
        print(f"  - {hit.content}")

    memory.close()


if __name__ == "__main__":
    main()
