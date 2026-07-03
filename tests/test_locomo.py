"""Offline, deterministic tests for the LoCoMo -> QAInstance adapter + judge mapping."""

from __future__ import annotations

from cortex_bench.judge import get_anscheck_prompt
from cortex_bench.locomo import convert

_SAMPLE = [
    {
        "sample_id": "conv1",
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1_date_time": "1:00 pm on 8 May, 2023",
            "session_1": [
                {"speaker": "Alice", "dia_id": "D1:1", "text": "I adopted a dog named Rex."},
                {"speaker": "Bob", "dia_id": "D1:2", "text": "Nice!"},
            ],
            "session_2_date_time": "2:00 pm on 9 May, 2023",
            "session_2": [
                {"speaker": "Bob", "dia_id": "D2:1", "text": "How is Rex?"},
            ],
        },
        "qa": [
            {"question": "Alice's dog?", "answer": "Rex", "evidence": ["D1:1"], "category": 4},
            {
                "question": "When adopted?",
                "answer": "8 May 2023",
                "evidence": ["D1:1"],
                "category": 2,
            },
            {
                "question": "What realization?",
                "evidence": ["D2:1"],
                "category": 5,
                "adversarial_answer": "x",
            },
        ],
    }
]


def test_convert_basic_shape():
    insts = convert(_SAMPLE)
    assert len(insts) == 3
    i0 = insts[0]
    assert i0.question_id == "conv1_q0"
    assert i0.question_type == "locomo-singlehop"
    assert i0.answer == "Rex"
    assert i0.haystack_session_ids == ["conv1_s1", "conv1_s2"]
    assert i0.answer_session_ids == ["conv1_s1"]  # parsed from evidence D1:1


def test_speaker_role_mapping_keeps_names():
    s1 = convert(_SAMPLE)[0].haystack_sessions[0]
    assert s1[0]["role"] == "user"  # speaker_a -> user
    assert s1[1]["role"] == "assistant"  # speaker_b -> assistant
    assert s1[0]["content"] == "Alice: I adopted a dog named Rex."  # name preserved


def test_temporal_and_adversarial_mapping():
    insts = convert(_SAMPLE)
    assert insts[1].question_type == "locomo-temporal"
    adv = insts[2]
    assert adv.question_type == "locomo-adversarial"
    assert adv.question_id.endswith("_abs")
    assert adv.is_abstention is True


def test_qas_share_session_objects():
    insts = convert(_SAMPLE)
    # All QAs of a conversation share the SAME session list (no per-QA duplication).
    assert insts[0].haystack_sessions is insts[1].haystack_sessions


def test_judge_handles_locomo_categories():
    for t in ("locomo-multihop", "locomo-singlehop", "locomo-opendomain", "locomo-temporal"):
        assert "Question:" in get_anscheck_prompt(t, "q", "a", "r")
    adv = get_anscheck_prompt("locomo-adversarial", "q", "a", "r", abstention=True)
    assert "unanswerable" in adv.lower()
