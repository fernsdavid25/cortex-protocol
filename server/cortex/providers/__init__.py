"""LLM provider adapters (Goal.md §5.1 — BYOK, program against `LLMProvider`)."""

from __future__ import annotations

from .base import EmbedResult, GenResult, LLMProvider
from .fake import FakeProvider
from .gemini import GeminiProvider

__all__ = [
    "EmbedResult",
    "FakeProvider",
    "GeminiProvider",
    "GenResult",
    "LLMProvider",
]
