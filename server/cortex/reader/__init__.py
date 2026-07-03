"""Reader prompt construction (Chain-of-Note + calibrated abstention) — Phase 2."""

from __future__ import annotations

from .reader import ABSTAIN_SENTINEL, build_reader_prompt, format_chunks

__all__ = ["ABSTAIN_SENTINEL", "build_reader_prompt", "format_chunks"]
