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


def _entry_with_has_answer() -> dict:
    """LongMemEval marks evidence TURNS with a per-turn `has_answer`; non-gold sessions
    omit the key entirely (verified against longmemeval_s_cleaned.json)."""
    return {
        "question_id": "q1",
        "question_type": "multi-session",
        "question": "q?",
        "answer": "a",
        "haystack_session_ids": ["s_a", "s_b"],
        "haystack_dates": ["d1", "d2"],
        "haystack_sessions": [
            [{"role": "user", "content": "noise"}, {"role": "assistant", "content": "noise"}],
            [
                {"role": "user", "content": "gold", "has_answer": True},
                {"role": "assistant", "content": "filler", "has_answer": False},
                {"role": "user", "content": "gold2", "has_answer": True},
            ],
        ],
        "answer_session_ids": ["s_b"],
    }


def test_has_answer_turns_become_answer_turn_ids():
    inst = parse_instances([_entry_with_has_answer()])[0]
    assert inst.answer_turn_ids == ["s_b#0", "s_b#2"]  # 0-based positions, gold session only


def test_missing_has_answer_yields_empty_turn_gold():
    """An entry (or release) with no has_answer anywhere yields NO turn gold, which makes
    coverage@k report None rather than a fabricated 0.0 — this is the safety net for the
    oracle variant, whose per-turn flags we cannot confirm without downloading it."""
    entry = _entry_with_has_answer()
    for session in entry["haystack_sessions"]:
        for turn in session:
            turn.pop("has_answer", None)
    assert parse_instances([entry])[0].answer_turn_ids == []


def test_answer_session_ids_unchanged():
    """recall@k's input is byte-identical alongside the new turn gold."""
    inst = parse_instances([_entry_with_has_answer()])[0]
    assert inst.answer_session_ids == ["s_b"]
