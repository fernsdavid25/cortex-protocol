"""Full-context baseline: stuff the entire haystack into the reader prompt (Goal.md §6)."""

from __future__ import annotations

from cortex.providers.base import LLMProvider

from ..memory_system import MemorySystem, QAInstance, Usage
from ._common import READING_INSTRUCTION, format_sessions


class FullContextSystem(MemorySystem):
    """No retrieval — the whole history is read every time. The accuracy ceiling baseline."""

    name = "full-context"

    def __init__(self, provider: LLMProvider, max_output_tokens: int = 256) -> None:
        self.provider = provider
        self.max_output_tokens = max_output_tokens
        self._instance: QAInstance | None = None

    def reset(self) -> None:
        self._instance = None

    def ingest(self, instance: QAInstance) -> Usage:
        self._instance = instance
        return Usage()

    def answer(self, instance: QAInstance) -> tuple[str, Usage]:
        prompt = (
            READING_INSTRUCTION
            + f"Question (asked on {instance.question_date}): {instance.question}\n\n"
            + format_sessions(instance)
            + "\n\nAnswer:"
        )
        r = self.provider.generate(prompt, max_output_tokens=self.max_output_tokens)
        return r.text.strip(), Usage(input_tokens=r.input_tokens, output_tokens=r.output_tokens)

    def retrieved_session_ids(self) -> list[str]:
        return list(self._instance.haystack_session_ids) if self._instance else []
