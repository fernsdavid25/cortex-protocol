"""Offline, deterministic tests for the LoCoMo -> QAInstance adapter + judge mapping."""

from __future__ import annotations

import copy

from cortex_bench.judge import get_anscheck_prompt
from cortex_bench.locomo import LOCOMO_EVIDENCE_TURN_BASE, convert

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


# --- turn-level gold evidence (coverage@k) ----------------------------------------------------


def test_evidence_turn_convention_is_one_indexed():
    """PINS THE CONVENTION. In the LoCoMo schema, dialogue turns are numbered from 1, so
    evidence "D1:1" is the FIRST turn of session_1 -> 0-based position 0. If this ever flips,
    coverage@k silently measures the wrong turn, so it is asserted explicitly here."""
    assert LOCOMO_EVIDENCE_TURN_BASE == 1
    insts = convert(_SAMPLE)
    assert insts[0].answer_turn_ids == ["conv1_s1#0"]  # evidence "D1:1" -> first turn


def test_evidence_second_turn_maps_to_position_one():
    sample = copy.deepcopy(_SAMPLE)
    sample[0]["qa"] = [
        {"question": "Bob's reply?", "answer": "Nice!", "evidence": ["D1:2"], "category": 4}
    ]
    assert convert(sample)[0].answer_turn_ids == ["conv1_s1#1"]


def test_evidence_second_session_uses_session_id_not_haystack_index():
    """A session-number GAP (session_1, session_3) must resolve by session NUMBER, never by
    position in the haystack list — otherwise D3:1 would be attributed to session_1's
    neighbour and the whole metric would be off by a session."""
    sample = copy.deepcopy(_SAMPLE)
    conv = sample[0]["conversation"]
    conv["session_3"] = conv.pop("session_2")
    conv["session_3_date_time"] = conv.pop("session_2_date_time")
    for t in conv["session_3"]:
        t["dia_id"] = t["dia_id"].replace("D2:", "D3:")
    sample[0]["qa"] = [
        {"question": "Rex?", "answer": "ok", "evidence": ["D3:1"], "category": 4},
    ]
    inst = convert(sample)[0]
    assert inst.haystack_session_ids == ["conv1_s1", "conv1_s3"]
    assert inst.answer_turn_ids == ["conv1_s3#0"]


def test_answer_session_ids_unchanged_by_turn_gold():
    """recall@k's input is byte-identical to what it was before coverage@k existed."""
    insts = convert(_SAMPLE)
    assert [i.answer_session_ids for i in insts] == [["conv1_s1"], ["conv1_s1"], ["conv1_s2"]]


def test_turn_gold_resolved_by_dia_id_not_position():
    """dia_id is authoritative: when the dia_ids are shuffled relative to list order, gold
    follows the dia_id, not the position. This is what makes an off-by-one impossible on the
    primary path."""
    sample = copy.deepcopy(_SAMPLE)
    s1 = sample[0]["conversation"]["session_1"]
    s1[0]["dia_id"], s1[1]["dia_id"] = "D1:2", "D1:1"
    sample[0]["qa"] = [
        {"question": "?", "answer": "a", "evidence": ["D1:1"], "category": 4},
    ]
    # "D1:1" now lives at list position 1, so positional resolution would say #0.
    assert convert(sample)[0].answer_turn_ids == ["conv1_s1#1"]


def test_positional_fallback_when_dia_id_absent():
    sample = copy.deepcopy(_SAMPLE)
    for key in ("session_1", "session_2"):
        for t in sample[0]["conversation"][key]:
            t.pop("dia_id")
    sample[0]["qa"] = [
        {"question": "?", "answer": "a", "evidence": ["D1:2"], "category": 4},
    ]
    assert convert(sample)[0].answer_turn_ids == ["conv1_s1#1"]  # via LOCOMO_EVIDENCE_TURN_BASE


def test_out_of_range_evidence_is_dropped_and_warned(capsys):
    """A wrong convention must read as a LOUD ZERO, never as a plausible number."""
    sample = copy.deepcopy(_SAMPLE)
    for key in ("session_1", "session_2"):
        for t in sample[0]["conversation"][key]:
            t.pop("dia_id")
    sample[0]["qa"] = [
        {"question": "?", "answer": "a", "evidence": ["D1:99"], "category": 4},
    ]
    insts = convert(sample)
    assert insts[0].answer_turn_ids == []
    assert "WARN: locomo" in capsys.readouterr().out


def test_haystack_sessions_still_shared_and_turn_dicts_unchanged():
    """No per-QA contamination and no prompt-byte change: the turn dicts handed to the reader
    still carry exactly {role, content}, and all QAs still share one session list."""
    insts = convert(_SAMPLE)
    assert insts[0].haystack_sessions is insts[1].haystack_sessions
    for session in insts[0].haystack_sessions:
        for turn in session:
            assert set(turn) == {"role", "content"}


# --- multi-id evidence fields (measured against the real corpus, 2026-07-28) ------------------
#
# Six evidence fields in the released locomo10.json are not a bare "D{n}:{m}". Four pack several
# ids into ONE string. Anchored matching on the raw field keeps only the first and silently drops
# the rest, shrinking the gold set and biasing recall upward. These fixtures reproduce the exact
# malformed shapes found in the corpus.


def _multi_evidence_sample(evidence: list[str]) -> list[dict]:
    """One conversation with 4 known turns, and a single QA carrying `evidence`."""
    sample = copy.deepcopy(_SAMPLE)
    sample[0]["conversation"]["session_1"] = [
        {"speaker": "Alice", "dia_id": f"D1:{i}", "text": f"turn {i}"} for i in range(1, 5)
    ]
    sample[0]["conversation"]["session_2"] = [
        {"speaker": "Bob", "dia_id": f"D2:{i}", "text": f"s2 turn {i}"} for i in range(1, 5)
    ]
    sample[0]["qa"] = [{"question": "q?", "answer": "a", "evidence": evidence, "category": 4}]
    return sample


def test_space_separated_multi_id_field_yields_every_turn():
    """ "D9:1 D4:4 D4:6" in the real corpus holds THREE ids; keeping one loses two."""
    inst = convert(_multi_evidence_sample(["D1:1 D2:3 D2:4"]))[0]
    assert inst.answer_turn_ids == sorted({"conv1_s1#0", "conv1_s2#2", "conv1_s2#3"})


def test_semicolon_separated_multi_id_field_yields_every_turn():
    """The corpus contains "D8:6; D9:17" — a semicolon, not whitespace."""
    inst = convert(_multi_evidence_sample(["D1:2; D2:1"]))[0]
    assert inst.answer_turn_ids == sorted({"conv1_s1#1", "conv1_s2#0"})


def test_zero_padded_turn_normalizes():
    """ "D30:05" appears in the corpus and must resolve to turn 5, not be dropped."""
    inst = convert(_multi_evidence_sample(["D1:03"]))[0]
    assert inst.answer_turn_ids == ["conv1_s1#2"]


def test_corrupt_tokens_are_dropped_not_repaired():
    """ "D" and "D:11:26" are unrecoverable; guessing is what this metric exists to avoid."""
    inst = convert(_multi_evidence_sample(["D", "D:11:26"]))[0]
    assert inst.answer_turn_ids == []


def test_multi_id_field_does_not_change_session_gold():
    """recall@k must keep reporting byte-identical values: session gold takes the FIRST id only."""
    inst = convert(_multi_evidence_sample(["D1:1 D2:3"]))[0]
    assert inst.answer_session_ids == ["conv1_s1"]


def test_turn_base_is_one_so_first_turn_is_position_zero():
    """Fails loudly if LOCOMO_EVIDENCE_TURN_BASE is ever flipped to 0."""
    assert LOCOMO_EVIDENCE_TURN_BASE == 1
    # dia_id "D1:1" is the FIRST turn -> 0-based position 0. Derived from the real corpus,
    # where "D1:3" is session_1[2]; asserted here independently of the parser's own arithmetic.
    sample = _multi_evidence_sample(["D1:1"])
    del sample[0]["conversation"]["session_1"][0]["dia_id"]  # force the positional fallback
    assert convert(sample)[0].answer_turn_ids == ["conv1_s1#0"]
