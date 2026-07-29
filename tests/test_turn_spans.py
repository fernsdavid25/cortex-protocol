"""Offline, deterministic tests for turn spans and per-system turn-level provenance.

The load-bearing claim here is that ``rounds()`` is a pure projection of
``rounds_with_spans()`` — if the two ever disagreed, a chunk's text and its recorded turn
provenance would describe different turns and coverage@k would be quietly wrong.
"""

from __future__ import annotations

from cortex.providers.fake import FakeProvider
from cortex_bench.judge import Judge
from cortex_bench.memory_system import MemorySystem, QAInstance, Usage
from cortex_bench.run import run
from cortex_bench.systems._common import rounds, rounds_with_spans
from cortex_bench.systems.cortex_system import CortexSystem
from cortex_bench.systems.full_context import FullContextSystem
from cortex_bench.systems.naive_rag import NaiveRAGSystem
from cortex_bench.systems.stub import GoldStub, NullStub

_SHAPES: list[list[dict]] = [
    # ordinary user/assistant pairs
    [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ],
    # assistant-first (a round closes immediately)
    [{"role": "assistant", "content": "a1"}, {"role": "user", "content": "u1"}],
    # two consecutive users then an assistant (one 3-turn round)
    [
        {"role": "user", "content": "u1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a1"},
    ],
    # trailing user-only remainder (flushed as its own chunk)
    [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ],
]


def test_rounds_output_unchanged():
    for turns in _SHAPES:
        assert rounds(turns) == [text for text, _positions in rounds_with_spans(turns)]


def test_spans_partition_all_turns_exactly_once():
    for turns in _SHAPES:
        flat: list[int] = []
        for _text, positions in rounds_with_spans(turns):
            flat.extend(positions)
        assert flat == list(range(len(turns)))


def test_span_positions_match_the_chunk_text():
    turns = _SHAPES[2]
    for text, positions in rounds_with_spans(turns):
        for pos in positions:
            assert turns[pos]["content"] in text


def _two_session_instance() -> QAInstance:
    return QAInstance(
        question_id="q_dog",
        question_type="single-session-user",
        question="What is the name of the user's dog?",
        answer="Rex",
        question_date="2026-06-27",
        answer_session_ids=["s_answer"],
        haystack_session_ids=["s_noise", "s_answer"],
        haystack_dates=["2026-01-01", "2026-02-01"],
        haystack_sessions=[
            [
                {"role": "user", "content": "I enjoy hiking mountains on weekends."},
                {"role": "assistant", "content": "Hiking is wonderful exercise."},
            ],
            [
                {"role": "user", "content": "The name of my dog is Rex."},
                {"role": "assistant", "content": "Rex is a great name for a dog."},
            ],
        ],
    )


def _all_valid_turn_ids(inst: QAInstance) -> set[str]:
    return {
        f"{sid}#{pos}"
        for sid, session in zip(inst.haystack_session_ids, inst.haystack_sessions, strict=False)
        for pos in range(len(session))
    }


def test_naive_rag_reports_turn_ids():
    system = NaiveRAGSystem(FakeProvider(), top_k=4)
    inst = _two_session_instance()
    system.reset()
    system.ingest(inst)
    system.answer(inst)
    tids = system.retrieved_turn_ids()
    assert tids is not None
    assert tids  # non-empty
    assert len(tids) == len(set(tids))  # deduped
    assert set(tids) <= _all_valid_turn_ids(inst)


def test_cortex_system_reports_turn_ids_for_round_chunks():
    system = CortexSystem(FakeProvider(), top_k=4)
    inst = _two_session_instance()
    system.reset()
    system.ingest(inst)
    system.answer(inst)
    tids = system.retrieved_turn_ids()
    assert tids is not None
    assert tids
    assert set(tids) <= _all_valid_turn_ids(inst)


def test_fact_key_chunks_contribute_no_turn_ids():
    """Distilled facts have SESSION provenance but no TURN provenance, so --fact-keys can
    raise recall@k without raising coverage@k. That asymmetry is the metric working."""
    system = CortexSystem(FakeProvider(responder=lambda p: "a distilled fact"), use_fact_keys=True)
    inst = _two_session_instance()
    system.reset()
    system.ingest(inst)
    fact_chunks = [c for c in system.store.chunks if c.text == "a distilled fact"]
    assert fact_chunks  # the fact-key path really ran
    assert all(c.turn_ids == () for c in fact_chunks)
    round_chunks = [c for c in system.store.chunks if c.text != "a distilled fact"]
    assert all(c.turn_ids for c in round_chunks)


def test_full_context_and_stubs_report_none():
    assert FullContextSystem(FakeProvider()).retrieved_turn_ids() is None
    assert GoldStub().retrieved_turn_ids() is None
    assert NullStub().retrieved_turn_ids() is None


class _TurnReportingSystem(MemorySystem):
    """Gold-answering stub that reports fixed turn provenance."""

    name = "turn-stub"

    def ingest(self, instance: QAInstance) -> Usage:
        return Usage()

    def answer(self, instance: QAInstance) -> tuple[str, Usage]:
        return instance.answer, Usage(input_tokens=1)

    def retrieved_turn_ids(self) -> list[str] | None:
        return ["s1#0"]


class _RaisingSystem(MemorySystem):
    """Raises in answer(), to prove the new hook call cannot turn a recorded failure into a
    crash of the whole run."""

    name = "raiser"

    def ingest(self, instance: QAInstance) -> Usage:
        return Usage()

    def answer(self, instance: QAInstance) -> tuple[str, Usage]:
        raise RuntimeError("boom")


def test_record_gets_turn_ids_through_run(sample_instances):
    records = run(sample_instances, _TurnReportingSystem(), Judge("offline"))
    assert all(r.retrieved_turn_ids == ["s1#0"] for r in records)


def test_failing_system_still_recorded_not_crashed(sample_instances):
    records = run(sample_instances, _RaisingSystem(), Judge("offline"))
    assert len(records) == len(sample_instances)
    assert all(r.correct is False and r.retrieved_turn_ids is None for r in records)
