"""Stub systems for harness self-testing (NOT real memory systems)."""

from __future__ import annotations

from ..memory_system import MemorySystem, QAInstance, Usage


class GoldStub(MemorySystem):
    """Returns the gold answer — verifies the harness + judge wiring end-to-end."""

    name = "gold-stub"

    def ingest(self, instance: QAInstance) -> Usage:
        return Usage()

    def answer(self, instance: QAInstance) -> tuple[str, Usage]:
        return instance.answer, Usage(output_tokens=len(instance.answer.split()))


class NullStub(MemorySystem):
    """Always abstains — sanity-checks abstention scoring."""

    name = "null-stub"

    def ingest(self, instance: QAInstance) -> Usage:
        return Usage()

    def answer(self, instance: QAInstance) -> tuple[str, Usage]:
        return "I don't know.", Usage(output_tokens=3)
