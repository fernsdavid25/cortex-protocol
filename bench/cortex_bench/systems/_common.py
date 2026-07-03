"""Shared helpers for the baseline reader systems (prompting, formatting, retrieval).

`READING_INSTRUCTION` is the Chain-of-Note + explicit-abstention prompt from
Goal.md §5.4: note the relevant facts first, answer only from the provided history,
and abstain with the exact sentinel string when the answer is absent (so the
offline judge and the abstention metric both register it).
"""

from __future__ import annotations

from collections.abc import Sequence

from ..memory_system import QAInstance

READING_INSTRUCTION = (
    "You are answering a question using only the conversation history provided below.\n"
    "First, note the relevant facts from the history. Then, answer the question using "
    "only those facts.\n"
    "If the history does not contain the information needed to answer, reply exactly: "
    "I don't know.\n\n"
)


def format_sessions(instance: QAInstance) -> str:
    """Render an instance's haystack as readable text with per-session headers."""
    blocks: list[str] = []
    for sid, date, session in zip(
        instance.haystack_session_ids,
        instance.haystack_dates,
        instance.haystack_sessions,
        strict=False,
    ):
        lines = [f"[Session {sid} | {date}]"]
        for turn in session:
            role = turn.get("role", "")
            content = turn.get("content", "")
            lines.append(f"{role}: {content}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def rounds(session_turns: list[dict]) -> list[str]:
    """Group consecutive turns into round chunks, closing a round after each assistant turn."""
    chunks: list[str] = []
    current: list[str] = []
    for turn in session_turns:
        role = turn.get("role", "")
        content = turn.get("content", "")
        current.append(f"{role}: {content}")
        if role == "assistant":
            chunks.append("\n".join(current))
            current = []
    if current:
        chunks.append("\n".join(current))
    return chunks


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Dot product of two equal-length (already L2-normalized) vectors."""
    return sum(x * y for x, y in zip(a, b, strict=False))
