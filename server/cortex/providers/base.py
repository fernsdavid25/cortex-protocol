"""LLM provider abstraction.

Gemini is the only adapter shipped now (Goal.md I5), but everything programs against
this interface so Claude/OpenAI/OpenRouter/Ollama drop in later. Self-hosters supply
their own key (BYOK) — providers read credentials from env, nothing is bundled.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class EmbedResult:
    vectors: list[list[float]]
    input_tokens: int = 0


class LLMProvider(ABC):
    @abstractmethod
    def generate(
        self, prompt: str, *, temperature: float = 0.0, max_output_tokens: int = 512
    ) -> GenResult:
        raise NotImplementedError

    @abstractmethod
    def embed(self, texts: list[str]) -> EmbedResult:
        raise NotImplementedError
