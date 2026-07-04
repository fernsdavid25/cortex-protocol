"""Gemini adapter (google-genai). Lazy-imports the SDK so tests/offline code don't need it."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any, TypeVar

from .base import EmbedResult, GenResult, LLMProvider

T = TypeVar("T")

# Transient HTTP statuses worth retrying. 429 = rate limited; 500/502/503/504 = upstream
# server blips (e.g. the "503 UNAVAILABLE" that crashed a live run). Genuine client errors
# like 400 (bad request) and 404 (not found) are NOT retried — they won't fix themselves.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def _is_transient(exc: BaseException) -> bool:
    """True if `exc` is worth retrying: a retryable Gemini HTTP status OR a network error.

    Network-level failures (dropped connection, DNS, read timeout — e.g. ``getaddrinfo
    failed`` or a socket that hangs until the request timeout fires) are transient and
    recover on retry; without this they would kill or hang a live run on a brief blip.
    SDK errors are imported lazily so offline code/tests never import google-genai.
    """
    # Socket / DNS / connection-reset errors surface as OSError (incl. socket.gaierror).
    if isinstance(exc, OSError):
        return True
    # httpx transport errors (connect/read/timeout) — google-genai's HTTP layer.
    try:
        import httpx

        if isinstance(exc, httpx.TransportError):
            return True
    except ImportError:
        pass
    try:
        from google.genai.errors import APIError, ServerError
    except ImportError:
        return False
    if not isinstance(exc, APIError):
        return False
    code = getattr(exc, "code", None)
    # All 5xx ServerErrors are transient; otherwise gate on the explicit status set.
    return isinstance(exc, ServerError) or code in _RETRYABLE_STATUS


def _retry(fn: Callable[[], T], *, tries: int = 5, base: float = 1.0) -> T:
    """Call `fn`, retrying transient Gemini errors with capped exponential backoff.

    Sleeps base*2**i seconds between attempts (e.g. 1, 2, 4, 8 for the default 5 tries),
    re-raising non-transient errors immediately and the last error after `tries` attempts.
    `fn` is a zero-arg callable so this stays unit-testable by injecting a fake.
    """
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — classification is delegated to _is_transient
            if not _is_transient(exc) or attempt == tries - 1:
                raise
            time.sleep(base * (2**attempt))
    raise AssertionError("unreachable")  # pragma: no cover — loop always returns or raises


class GeminiProvider(LLMProvider):
    def __init__(
        self,
        reader_model: str = "gemini-2.5-flash-lite",
        embed_model: str = "gemini-embedding-001",
        embed_dim: int = 768,
        api_key: str | None = None,
    ) -> None:
        self.reader_model = reader_model
        self.embed_model = embed_model
        self.embed_dim = embed_dim
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        # Lazily built in _client_obj(); Any so mypy allows the genai.Client assignment without a
        # module-level SDK import (the SDK is intentionally imported inside methods only).
        self._client: Any = None

    def _client_obj(self):
        if self._client is None:
            from google import genai
            from google.genai import types

            # Per-request timeout (ms) so a dropped/hung connection raises instead of
            # blocking the whole run forever; the raised timeout is retried by _retry.
            self._client = genai.Client(
                api_key=self._api_key,
                http_options=types.HttpOptions(timeout=90_000),
            )
        return self._client

    def generate(
        self, prompt: str, *, temperature: float = 0.0, max_output_tokens: int = 512
    ) -> GenResult:
        from google.genai import types

        def _call():
            return self._client_obj().models.generate_content(
                model=self.reader_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature, max_output_tokens=max_output_tokens
                ),
            )

        resp = _retry(_call)
        um = getattr(resp, "usage_metadata", None)
        return GenResult(
            text=resp.text or "",
            input_tokens=getattr(um, "prompt_token_count", 0) or 0,
            output_tokens=getattr(um, "candidates_token_count", 0) or 0,
        )

    def embed(self, texts: list[str], *, batch_size: int = 100) -> EmbedResult:
        from google.genai import types

        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]

            def _call(b: list[str] = batch):
                return self._client_obj().models.embed_content(
                    model=self.embed_model,
                    contents=b,
                    config=types.EmbedContentConfig(output_dimensionality=self.embed_dim),
                )

            resp = _retry(_call)
            embeddings = list(resp.embeddings or [])
            if len(embeddings) != len(batch):
                raise RuntimeError(
                    f"embedding count mismatch: got {len(embeddings)} for {len(batch)} inputs"
                )
            vectors.extend(list(e.values) for e in embeddings)
        # Embedding token usage isn't always returned; approximate for cost accounting.
        approx = sum(len(t.split()) for t in texts)
        return EmbedResult(vectors=vectors, input_tokens=approx)
