"""Deterministic in-process provider for OFFLINE tests (no network, no SDK).

`generate` is driven by a caller-supplied `responder`, and `embed` produces
deterministic L2-normalized bag-of-words hash vectors so the cosine of identical
text is exactly 1.0. Nothing here touches an API — it exists so baselines and the
harness can be exercised end-to-end without spending tokens.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Sequence

from .base import EmbedResult, GenResult, LLMProvider

_TOKEN = re.compile(r"[a-z0-9]+")


def _default_responder(prompt: str) -> str:
    return "I don't know."


def _bucket(token: str, dim: int) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()  # noqa: S324 (non-crypto bucketing)
    return int(digest, 16) % dim


class FakeProvider(LLMProvider):
    def __init__(self, responder: Callable[[str], str] | None = None, dim: int = 16) -> None:
        self.responder = responder or _default_responder
        self.dim = dim
        self.last_prompt: str | None = None

    def generate(
        self, prompt: str, *, temperature: float = 0.0, max_output_tokens: int = 512
    ) -> GenResult:
        self.last_prompt = prompt
        text = self.responder(prompt) or "I don't know."
        return GenResult(
            text=text,
            input_tokens=len(prompt.split()),
            output_tokens=len(text.split()),
        )

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN.findall(text.lower()):
            vec[_bucket(token, self.dim)] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]

    def embed(self, texts: Sequence[str]) -> EmbedResult:
        vectors = [self._embed_one(t) for t in texts]
        approx = sum(len(t.split()) for t in texts)
        return EmbedResult(vectors=vectors, input_tokens=approx)
