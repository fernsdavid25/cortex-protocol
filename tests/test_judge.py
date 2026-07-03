import pytest

from cortex_bench.judge import Judge, get_anscheck_prompt, majority_vote, offline_label
from cortex_bench.memory_system import QAInstance


def test_prompt_routing_per_type():
    for t in ("single-session-user", "single-session-assistant", "multi-session"):
        p = get_anscheck_prompt(t, "q", "a", "r")
        assert "contains the correct answer" in p
        assert "off-by-one" not in p

    p_temporal = get_anscheck_prompt("temporal-reasoning", "q", "a", "r")
    assert "off-by-one" in p_temporal

    p_ku = get_anscheck_prompt("knowledge-update", "q", "a", "r")
    assert "updated answer" in p_ku

    p_pref = get_anscheck_prompt("single-session-preference", "q", "a", "r")
    assert "Rubric:" in p_pref

    p_abs = get_anscheck_prompt("multi-session", "q", "a", "r", abstention=True)
    assert "unanswerable" in p_abs


def test_unknown_type_raises():
    with pytest.raises(ValueError):
        get_anscheck_prompt("nonsense-type", "q", "a", "r")


def test_offline_label_answerable():
    inst = QAInstance("q1", "single-session-user", "where?", "Goa")
    assert offline_label(inst, "The user lives in Goa, India.") is True
    assert offline_label(inst, "The user lives in Mumbai.") is False


def test_offline_label_abstention():
    inst = QAInstance("q3_abs", "multi-session", "blood type?", "never mentioned")
    assert inst.is_abstention is True
    assert offline_label(inst, "I don't know — that wasn't mentioned.") is True
    assert offline_label(inst, "Your blood type is O negative.") is False


def test_majority_vote():
    assert majority_vote([True, True, False]) is True
    assert majority_vote([False, False, True]) is False
    assert majority_vote([True]) is True
    assert majority_vote([False]) is False
    assert majority_vote([True, False]) is True  # tie -> True
    with pytest.raises(ValueError):
        majority_vote([])


def test_judge_rejects_zero_votes():
    with pytest.raises(ValueError):
        Judge(backend="gemini", votes=0)


def test_judge_self_consistency_majority(monkeypatch):
    """votes>1 calls the grader `votes` times and takes the majority — no live API calls."""
    inst = QAInstance("q1", "single-session-user", "where?", "Goa")
    fake_labels = iter([True, False, True])  # 2 yes / 1 no -> majority True

    judge = Judge(backend="gemini", votes=3)
    monkeypatch.setattr(judge, "_gemini_grade", lambda prompt, model: next(fake_labels))
    assert judge.grade(inst, "Goa") is True


def test_judge_self_consistency_flips_minority(monkeypatch):
    """A single stray 'yes' is overruled by the majority 'no'."""
    inst = QAInstance("q1", "single-session-user", "where?", "Goa")
    fake_labels = iter([True, False, False])  # 1 yes / 2 no -> majority False

    judge = Judge(backend="gemini", votes=3)
    monkeypatch.setattr(judge, "_gemini_grade", lambda prompt, model: next(fake_labels))
    assert judge.grade(inst, "Mumbai") is False


def test_judge_single_vote_calls_grader_once(monkeypatch):
    inst = QAInstance("q1", "single-session-user", "where?", "Goa")
    calls = {"n": 0}

    def _grade(prompt, model):
        calls["n"] += 1
        return True

    judge = Judge(backend="gemini", votes=1)  # default back-compat path
    monkeypatch.setattr(judge, "_gemini_grade", _grade)
    assert judge.grade(inst, "Goa") is True
    assert calls["n"] == 1
