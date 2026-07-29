"""Naive-RAG baseline: embed round chunks, retrieve top-k by cosine, read (Goal.md §6)."""

from __future__ import annotations

from typing import NamedTuple

from cortex.providers.base import LLMProvider

from ..memory_system import MemorySystem, QAInstance, Usage
from ..turns import turn_ids_for
from ._common import READING_INSTRUCTION, cosine, rounds_with_spans


class _Chunk(NamedTuple):
    """One embedded round chunk and its provenance.

    A NamedTuple (not a bare 3-tuple) so the added turn provenance is read by NAME at every
    use site — an index-shift bug here would silently corrupt the ranking, not fail loudly.
    """

    text: str
    session_id: str | None
    turn_ids: tuple[str, ...]
    embedding: list[float]


class NaiveRAGSystem(MemorySystem):
    """Embed-and-retrieve over conversation rounds, then read the top-k chunks."""

    name = "naive-rag"

    def __init__(
        self, provider: LLMProvider, top_k: int = 10, max_output_tokens: int = 256
    ) -> None:
        self.provider = provider
        self.top_k = top_k
        self.max_output_tokens = max_output_tokens
        self._chunks: list[_Chunk] = []
        self._last_retrieved: list[str] = []
        self._last_retrieved_turns: list[str] = []

    def reset(self) -> None:
        self._chunks = []
        self._last_retrieved = []
        self._last_retrieved_turns = []

    def ingest(self, instance: QAInstance) -> Usage:
        texts: list[str] = []
        sids: list[str | None] = []
        turn_id_lists: list[tuple[str, ...]] = []
        for sid, session in zip(
            instance.haystack_session_ids, instance.haystack_sessions, strict=False
        ):
            for chunk, positions in rounds_with_spans(session):
                texts.append(chunk)
                sids.append(sid)
                turn_id_lists.append(turn_ids_for(sid, positions))
        if not texts:
            return Usage()
        res = self.provider.embed(texts)
        self._chunks = [
            _Chunk(text, sid, tids, vec)
            for text, sid, tids, vec in zip(texts, sids, turn_id_lists, res.vectors, strict=False)
        ]
        return Usage(embed_tokens=res.input_tokens)

    def answer(self, instance: QAInstance) -> tuple[str, Usage]:
        qres = self.provider.embed([instance.question])
        usage = Usage(embed_tokens=qres.input_tokens)
        qvec = qres.vectors[0] if qres.vectors else []

        ranked = sorted(self._chunks, key=lambda c: cosine(qvec, c.embedding), reverse=True)[
            : self.top_k
        ]

        seen: set[str] = set()
        retrieved: list[str] = []
        seen_turns: set[str] = set()
        retrieved_turns: list[str] = []
        for c in ranked:
            if c.session_id is not None and c.session_id not in seen:
                seen.add(c.session_id)
                retrieved.append(c.session_id)
            for tid in c.turn_ids:
                if tid not in seen_turns:
                    seen_turns.add(tid)
                    retrieved_turns.append(tid)
        self._last_retrieved = retrieved
        self._last_retrieved_turns = retrieved_turns

        context = "\n\n".join(f"[{c.session_id}] {c.text}" for c in ranked)
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

    def retrieved_turn_ids(self) -> list[str] | None:
        return self._last_retrieved_turns
