from cortex_bench.dataset import parse_instances


def test_parse_instances_maps_fields():
    raw = [
        {
            "question_id": "abc",
            "question_type": "knowledge-update",
            "question": "q?",
            "answer": "a",
            "question_date": "2025-01-01",
            "haystack_session_ids": ["s1"],
            "haystack_dates": ["2024-12-01"],
            "haystack_sessions": [[{"role": "user", "content": "hi"}]],
            "answer_session_ids": ["s1"],
        },
        {
            "question_id": "xyz_abs",
            "question_type": "multi-session",
            "question": "q2?",
            "answer": "unanswerable",
        },
    ]
    insts = parse_instances(raw)
    assert len(insts) == 2
    assert insts[0].question_type == "knowledge-update"
    assert insts[0].haystack_sessions[0][0]["content"] == "hi"
    assert insts[0].is_abstention is False
    assert insts[1].is_abstention is True
    # missing optional fields default to empty
    assert insts[1].haystack_sessions == []
