"""Naive-RAG baseline: embed round chunks, retrieve top-k by cosine, read (Goal.md §6)."""

from __future__ import annotations

from cortex.providers.base import LLMProvider

from ..memory_system import MemorySystem, QAInstance, Usage
from ._common import READING_INSTRUCTION, cosine, rounds


class NaiveRAGSystem(MemorySystem):
    """Embed-and-retrieve over conversation rounds, then read the top-k chunks."""

    name = "naive-rag"

    def __init__(
        self, provider: LLMProvider, top_k: int = 10, max_output_tokens: int = 256
    ) -> None:
        self.provider = provider
        self.top_k = top_k
        self.max_output_tokens = max_output_tokens
        self._chunks: list[tuple[str, str | None, list[float]]] = []
        self._last_retrieved: list[str] = []

    def reset(self) -> None:
        self._chunks = []
        self._last_retrieved = []

    def ingest(self, instance: QAInstance) -> Usage:
        texts: list[str] = []
        sids: list[str | None] = []
        for sid, session in zip(
            instance.haystack_session_ids, instance.haystack_sessions, strict=False
        ):
            for chunk in rounds(session):
                texts.append(chunk)
                sids.append(sid)
        if not texts:
            return Usage()
        res = self.provider.embed(texts)
        self._chunks = [
            (text, sid, vec) for text, sid, vec in zip(texts, sids, res.vectors, strict=False)
        ]
        return Usage(embed_tokens=res.input_tokens)

    def answer(self, instance: QAInstance) -> tuple[str, Usage]:
        qres = self.provider.embed([instance.question])
        usage = Usage(embed_tokens=qres.input_tokens)
        qvec = qres.vectors[0] if qres.vectors else []

        ranked = sorted(self._chunks, key=lambda c: cosine(qvec, c[2]), reverse=True)[: self.top_k]

        seen: set[str] = set()
        retrieved: list[str] = []
        for _text, sid, _vec in ranked:
            if sid is not None and sid not in seen:
                seen.add(sid)
                retrieved.append(sid)
        self._last_retrieved = retrieved

        context = "\n\n".join(f"[{sid}] {text}" for text, sid, _vec in ranked)
        prompt = (
            READING_INSTRUCTION
            + f"Question (asked on {instance.question_date}): {instance.question}\n\n"
            + context
            + "\n\nAnswer:"
        )
        r = self.provider.generate(prompt, max_output_tokens=self.max_output_tokens)
        usage = usage + Usage(input_tokens=r.input_tokens, output_tokens=r.output_tokens)
        return r.text.strip(), usage

    def retrieved_session_ids(self) -> list[str]:
        return self._last_retrieved
