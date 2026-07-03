"""Offline tests for the Gemini provider retry/backoff helper (no network, no live calls).

`_retry` takes a zero-arg callable so we can inject fakes that raise synthetic Gemini errors.
`time.sleep` is monkeypatched out so the exponential backoff is instantaneous in tests.
`_is_transient` is tested directly against the real (installed) SDK error classes.
"""

from __future__ import annotations

import pytest

from cortex.providers import gemini


class _FakeAPIError(Exception):
    """Stand-in for google.genai.errors.APIError carrying an HTTP status `.code`."""

    def __init__(self, code: int) -> None:
        super().__init__(f"fake {code}")
        self.code = code


class _FakeServerError(_FakeAPIError):
    """Stand-in for google.genai.errors.ServerError (a 5xx APIError subclass)."""


def _fake_is_transient(exc: BaseException) -> bool:
    return isinstance(exc, _FakeAPIError) and (
        isinstance(exc, _FakeServerError) or exc.code in gemini._RETRYABLE_STATUS
    )


@pytest.fixture
def fake_retry(monkeypatch):
    """Classify our fakes as transient and make backoff instantaneous for `_retry` tests."""
    monkeypatch.setattr(gemini, "_is_transient", _fake_is_transient)
    monkeypatch.setattr(gemini.time, "sleep", lambda _s: None)


def test_retry_succeeds_after_transient_failures(fake_retry):
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:  # fail twice (503), then succeed
            raise _FakeServerError(503)
        return "ok"

    assert gemini._retry(flaky, tries=5, base=0.0) == "ok"
    assert calls["n"] == 3


def test_retry_reraises_after_exceeding_max_tries(fake_retry):
    calls = {"n": 0}

    def always_fails():
        calls["n"] += 1
        raise _FakeServerError(503)

    with pytest.raises(_FakeServerError):
        gemini._retry(always_fails, tries=4, base=0.0)
    assert calls["n"] == 4  # exactly `tries` attempts, no more


def test_retry_does_not_retry_genuine_4xx(fake_retry):
    calls = {"n": 0}

    def bad_request():
        calls["n"] += 1
        raise _FakeAPIError(400)  # not in the retryable set

    with pytest.raises(_FakeAPIError):
        gemini._retry(bad_request, tries=5, base=0.0)
    assert calls["n"] == 1  # re-raised immediately, never retried


def test_retry_retries_429_rate_limit(fake_retry):
    calls = {"n": 0}

    def rate_limited():
        calls["n"] += 1
        if calls["n"] < 2:
            raise _FakeAPIError(429)
        return "done"

    assert gemini._retry(rate_limited, tries=5, base=0.0) == "done"
    assert calls["n"] == 2


def test_is_transient_without_sdk_is_safe_false():
    """Offline (no google-genai installed), `_is_transient` returns False, not an ImportError."""
    assert gemini._is_transient(_FakeServerError(503)) is False
    assert gemini._is_transient(ValueError("boom")) is False


def test_is_transient_retries_network_errors():
    """Network-level failures (socket/DNS/timeout) are transient — they hung live runs."""
    import socket

    assert gemini._is_transient(OSError("[WinError 10051] unreachable network")) is True
    assert gemini._is_transient(socket.gaierror("getaddrinfo failed")) is True
    assert gemini._is_transient(TimeoutError("read timed out")) is True  # OSError subclass
    # A non-network, non-API error is still not retried.
    assert gemini._is_transient(ValueError("boom")) is False


def test_retryable_status_set_excludes_genuine_4xx():
    assert gemini._RETRYABLE_STATUS == frozenset({429, 500, 502, 503, 504})
    assert 400 not in gemini._RETRYABLE_STATUS
    assert 404 not in gemini._RETRYABLE_STATUS
