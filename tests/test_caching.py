"""Offline tests for the embedding cache provider (no network)."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from cortex.providers.caching import CachingProvider
from cortex.providers.fake import FakeProvider


class _CountingFake(FakeProvider):
    """FakeProvider that records how many times (and on what) embed() was called."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_calls = 0
        self.embedded: list[str] = []

    def embed(self, texts: Sequence[str]):
        self.embed_calls += 1
        self.embedded.extend(texts)
        return super().embed(texts)


def test_hit_skips_inner_and_reports_zero_tokens():
    inner = _CountingFake()
    cp = CachingProvider(inner, ":memory:")
    miss = cp.embed(["alpha", "beta"])
    assert inner.embed_calls == 1
    assert miss.input_tokens > 0
    hit = cp.embed(["alpha", "beta"])
    assert inner.embed_calls == 1  # served entirely from cache
    assert hit.input_tokens == 0  # warm-cache marginal cost is zero
    for a, b in zip(miss.vectors, hit.vectors, strict=True):
        assert a == pytest.approx(b, abs=1e-6)


def test_partial_hit_only_embeds_missing():
    inner = _CountingFake()
    cp = CachingProvider(inner, ":memory:")
    cp.embed(["alpha"])
    inner.embedded.clear()
    cp.embed(["alpha", "gamma"])  # alpha cached, gamma new
    assert inner.embedded == ["gamma"]


def test_order_preserved_across_mixed_hits():
    inner = _CountingFake()
    cp = CachingProvider(inner, ":memory:")
    cp.embed(["b"])
    out = cp.embed(["a", "b", "c"])
    ref = FakeProvider().embed(["a", "b", "c"])
    for got, exp in zip(out.vectors, ref.vectors, strict=True):
        assert got == pytest.approx(exp, abs=1e-6)


def test_generate_passes_through():
    inner = FakeProvider(responder=lambda p: "passed through")
    cp = CachingProvider(inner, ":memory:")
    assert cp.generate("hi").text == "passed through"


def test_persists_to_disk_across_reopen(tmp_path):
    path = str(tmp_path / "cache.sqlite")
    first = _CountingFake()
    cp1 = CachingProvider(first, path)
    cp1.embed(["durable"])
    cp1.close()

    second = _CountingFake()
    cp2 = CachingProvider(second, path)
    out = cp2.embed(["durable"])
    assert second.embed_calls == 0  # served from the on-disk cache
    assert out.input_tokens == 0


def test_embed_signature_surfaced():
    cp = CachingProvider(FakeProvider(), ":memory:")
    assert cp.embed_dim == 16  # FakeProvider.dim
    assert cp.embed_model == "unknown"
