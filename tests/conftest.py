import pytest

from cortex_bench.memory_system import QAInstance


@pytest.fixture
def sample_instances() -> list[QAInstance]:
    return [
        QAInstance(
            question_id="q1",
            question_type="single-session-user",
            question="What city does the user live in?",
            answer="Goa",
            answer_session_ids=["s1"],
            haystack_session_ids=["s1", "s2"],
        ),
        QAInstance(
            question_id="q2",
            question_type="temporal-reasoning",
            question="How many days between the two trips?",
            answer="18",
            answer_session_ids=["s3"],
            haystack_session_ids=["s2", "s3"],
        ),
        QAInstance(
            question_id="q3_abs",
            question_type="multi-session",
            question="What is the user's blood type?",
            answer="The user never mentioned their blood type.",
            answer_session_ids=[],
            haystack_session_ids=["s1", "s2"],
        ),
    ]
