"""Offline, deterministic tests for the full-context and naive-RAG baselines."""

from cortex.providers.fake import FakeProvider
from cortex_bench.judge import Judge
from cortex_bench.memory_system import QAInstance
from cortex_bench.systems.full_context import FullContextSystem
from cortex_bench.systems.naive_rag import NaiveRAGSystem


def _full_context_instance() -> QAInstance:
    return QAInstance(
        question_id="q_city",
        question_type="single-session-user",
        question="What city does the user live in?",
        answer="Goa",
        question_date="2026-06-27",
        answer_session_ids=["s1"],
        haystack_session_ids=["s1", "s2"],
        haystack_dates=["2026-01-01", "2026-02-01"],
        haystack_sessions=[
            [
                {"role": "user", "content": "I live in Goa near the beach."},
                {"role": "assistant", "content": "Goa is lovely this time of year."},
            ],
            [
                {"role": "user", "content": "My favorite food is pasta."},
                {"role": "assistant", "content": "Pasta is a great choice."},
            ],
        ],
    )


def test_full_context_grades_true_and_includes_session_content():
    provider = FakeProvider(responder=lambda p: "Goa" if "city" in p.lower() else "I don't know.")
    system = FullContextSystem(provider)
    inst = _full_context_instance()

    system.reset()
    system.ingest(inst)
    hyp, _usage = system.answer(inst)

    assert Judge("offline").grade(inst, hyp) is True
    assert provider.last_prompt is not None
    assert "I live in Goa near the beach." in provider.last_prompt
    assert system.retrieved_session_ids() == ["s1", "s2"]


def _naive_rag_instance() -> QAInstance:
    return QAInstance(
        question_id="q_pet",
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


def test_naive_rag_retrieves_the_answer_session_first():
    provider = FakeProvider()
    system = NaiveRAGSystem(provider, top_k=2)
    inst = _naive_rag_instance()

    system.reset()
    system.ingest(inst)
    _hyp, _usage = system.answer(inst)

    retrieved = system.retrieved_session_ids()
    assert retrieved[0] == "s_answer"
