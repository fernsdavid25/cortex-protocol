"""Core data models + the MemorySystem contract for the LongMemEval harness.

Every system under test (baselines, Cortex, competitors) implements `MemorySystem`,
so the harness can score them identically. Each LongMemEval instance is independent:
the harness calls `reset()` → `ingest(instance)` → `answer(instance)` per question.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Usage:
    """Token/cost/latency accounting for a single operation (ingest or answer).

    ``embed_tokens`` (embedding-model input) is tracked SEPARATELY from
    ``input_tokens``/``output_tokens`` (reader-model in/out) so cost can be priced at
    each model's own rate instead of lumping embeddings in at the reader rate.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    embed_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.latency_ms + other.latency_ms,
            self.embed_tokens + other.embed_tokens,
        )


@dataclass
class QAInstance:
    """One LongMemEval question + its haystack (verified schema, see bench/README.md)."""

    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str = ""
    haystack_session_ids: list[str] = field(default_factory=list)
    haystack_dates: list[str] = field(default_factory=list)
    haystack_sessions: list[list[dict]] = field(default_factory=list)
    answer_session_ids: list[str] = field(default_factory=list)

    @property
    def is_abstention(self) -> bool:
        # Verified: abstention questions are flagged by an `_abs` question_id suffix,
        # overlaid on one of the 6 base question_types (not a 7th disjoint type).
        return self.question_id.endswith("_abs")


class MemorySystem(ABC):
    """Contract every system under test implements."""

    name: str = "base"

    def reset(self) -> None:  # noqa: B027  (optional hook; subclasses override if stateful)
        """Clear all memory between instances (each LongMemEval question is independent)."""

    @abstractmethod
    def ingest(self, instance: QAInstance) -> Usage:
        """Build memory from `instance.haystack_sessions`. Returns usage for cost accounting."""
        raise NotImplementedError

    @abstractmethod
    def answer(self, instance: QAInstance) -> tuple[str, Usage]:
        """Answer `instance.question`. Returns (hypothesis, usage)."""
        raise NotImplementedError

    def retrieved_session_ids(self) -> list[str]:
        """Session ids used to answer the most recent question (for recall@k). Optional."""
        return []
