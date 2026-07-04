"""Offline regression tests for two reader.py A+ hardening fixes.

All deterministic, no network / no live LLM. Covers:
1. The transfer line-split fallback strips ONLY a real leading list/ordinal marker — content
   that merely starts with a digit ("401k", "3D", "2024 …") is preserved (the old greedy
   ``lstrip("-*0123456789. ")`` corrupted it), while a genuine "1. text" marker IS stripped.
2. An extracted ``event_time`` that is not a valid ISO-8601 ``YYYY-MM-DD`` date falls back to the
   ingest date, keeping the timeline's lexical-sort invariant.
"""

from __future__ import annotations

from cortex.reader.reader import (
    parse_episodic_extraction,
    parse_transfer_extraction,
)

_INGEST = "2026-07-02"


# --- Fix 1: transfer line-split marker strip -----------------------------------------------------


def test_transfer_fallback_preserves_leading_digit_content() -> None:
    """A digit-leading fact ("401k", "3D", "2024 …") survives the marker strip uncorrupted."""
    # Non-JSON so parse_transfer_extraction falls back to line-splitting.
    raw = "\n".join(
        [
            "401k contributions maxed out this year",
            "3D printer bought for the workshop",
            "2024 was a big travel year",
            "1990s nostalgia is a strong preference",
        ]
    )
    out = [content for content, _kind in parse_transfer_extraction(raw)]
    assert out == [
        "401k contributions maxed out this year",
        "3D printer bought for the workshop",
        "2024 was a big travel year",
        "1990s nostalgia is a strong preference",
    ]


def test_transfer_fallback_strips_real_list_markers() -> None:
    """Genuine leading list/ordinal markers ("- ", "* ", "1. ", "2) ") ARE stripped."""
    raw = "\n".join(
        [
            "1. Prefers dark roast coffee",
            "- Lives in Berlin now",
            "* Works on the Cortex project",
            "2) Owns a golden retriever named Max",
        ]
    )
    out = [content for content, _kind in parse_transfer_extraction(raw)]
    assert out == [
        "Prefers dark roast coffee",
        "Lives in Berlin now",
        "Works on the Cortex project",
        "Owns a golden retriever named Max",
    ]


# --- Fix 2: event_time ISO validation ------------------------------------------------------------


def test_episodic_valid_iso_event_time_kept() -> None:
    """A well-formed ISO date is accepted verbatim."""
    parsed = parse_episodic_extraction('{"event_time": "2024-03-15"}', _INGEST)
    assert parsed["event_time"] == "2024-03-15"


def test_episodic_bad_event_time_falls_back() -> None:
    """A non-ISO ``event_time`` (prose, bare year, timestamp) falls back to the ingest date."""
    for bad in ('"March 2024"', '"2024"', '"2024-3-5"', '"2024-03-15T10:00:00Z"', '"next Friday"'):
        parsed = parse_episodic_extraction(f'{{"event_time": {bad}}}', _INGEST)
        assert parsed["event_time"] == _INGEST, bad
