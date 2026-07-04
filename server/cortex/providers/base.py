"""LLM provider abstraction.

Gemini is the only adapter shipped now (Goal.md I5), but everything programs against
this interface so Claude/OpenAI/OpenRouter/Ollama drop in later. Self-hosters supply
their own key (BYOK) — providers read credentials from env, nothing is bundled.

To add a backend, subclass :class:`LLMProvider`: implement ``generate`` (used only on the
write-time enrichment path) and ``embed`` (the single cost on the recall path), and set the
``embed_model`` / ``embed_dim`` attributes so the store can reject a model/dim mismatch before
it corrupts an existing index.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class GenResult:
    """The outcome of one :meth:`LLMProvider.generate` call.

    Attributes:
        text: The model's completion (empty string when the model returned nothing).
        input_tokens: Prompt tokens billed for the call; ``0`` if the provider reports no usage.
        output_tokens: Completion tokens billed for the call; ``0`` if usage is unavailable.
    """

    text: str
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class EmbedResult:
    """The outcome of one :meth:`LLMProvider.embed` batch.

    Attributes:
        vectors: One embedding vector per input text, in the same order as the inputs; each
            vector has ``LLMProvider.embed_dim`` components.
        input_tokens: Tokens billed for the batch; may be an approximation when the provider
            does not report embedding usage.
    """

    vectors: list[list[float]]
    input_tokens: int = 0


class LLMProvider(ABC):
    """The provider contract Cortex programs against — implement it to add a backend (BYOK).

    A provider does exactly two things: it :meth:`generate`\\ s text (write-time enrichment only)
    and it :meth:`embed`\\ s text into vectors (the sole cost on the recall path). Every concrete
    provider MUST also expose its embedding signature so the store can guard against a model/dim
    mismatch silently mixing incompatible vectors into an existing index.

    Attributes:
        embed_model: Identifier of the embedding model (e.g. ``"gemini-embedding-001"``). The store
            records it so a later run with a different model is rejected, not silently blended.
        embed_dim: Output dimensionality of :meth:`embed` vectors (e.g. ``768``). Must equal the
            length of every vector returned by :meth:`embed`.
    """

    embed_model: str
    embed_dim: int

    @abstractmethod
    def generate(
        self, prompt: str, *, temperature: float = 0.0, max_output_tokens: int = 512
    ) -> GenResult:
        """Generate a completion for ``prompt`` and report token usage.

        Args:
            prompt: The full input prompt.
            temperature: Sampling temperature; ``0.0`` for deterministic extraction.
            max_output_tokens: Hard cap on the number of generated tokens.

        Returns:
            A :class:`GenResult` with the completion text and billed token counts.
        """
        raise NotImplementedError

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> EmbedResult:
        """Embed one or more texts into fixed-length vectors.

        Args:
            texts: The texts to embed, in order.

        Returns:
            An :class:`EmbedResult` whose ``vectors`` align 1:1 with ``texts``; each vector has
            ``embed_dim`` components.
        """
        raise NotImplementedError
