"""Offline, deterministic tests for turn-level coverage@k and the run-metadata stamp.

These also pin the ADDITIVE contract: recall@k must keep reporting exactly what it did
before coverage@k existed, for the same records.
"""

from __future__ import annotations

from cortex_bench.memory_system import QAInstance, Usage
from cortex_bench.metrics import Record, _coverage, aggregate


def _inst(qid="a", *, sessions=("s1",), turns=()) -> QAInstance:
    return QAInstance(
        qid,
        "single-session-user",
        "q?",
        "x",
        answer_session_ids=list(sessions),
        answer_turn_ids=list(turns),
    )


def _rec(
    *,
    qid="a",
    sessions=("s1",),
    turns=(),
    retrieved=("s1",),
    retrieved_turns=None,
    measured=True,
    correct=True,
) -> Record:
    return Record(
        _inst(qid, sessions=sessions, turns=turns),
        "x",
        correct,
        Usage(input_tokens=10, output_tokens=2),
        list(retrieved),
        measured=measured,
        retrieved_turn_ids=None if retrieved_turns is None else list(retrieved_turns),
    )


# --- the metric itself --------------------------------------------------------------------


def test_coverage_full_and_partial():
    assert _coverage(["s1#0", "s1#1"], ["s1#0"]) == 1.0
    assert _coverage(["s1#0"], ["s1#0", "s1#3"]) == 0.5


def test_coverage_none_when_no_turn_gold():
    assert _coverage(["s1#0"], []) is None


def test_coverage_none_when_system_reports_none():
    # None means "not reported" and must EXCLUDE the record, never score it 0.0.
    assert _coverage(None, ["s1#0"]) is None


def test_coverage_zero_is_a_real_zero():
    # Reported-but-missed is genuinely 0.0, and must not be confused with "not reported".
    assert _coverage(["s2#0"], ["s1#0"]) == 0.0


def test_coverage_empty_retrieval_is_zero_not_none():
    """`[]` (reported, covered nothing) and `None` (not reported) are DIFFERENT.

    A cortex-v0 --fact-keys run can retrieve only distilled fact chunks, whose turn_ids are
    empty, so `retrieved_turn_ids` is `[]` with non-empty gold. That is a real 0.0 — the
    memory layer handed the reader nothing from the gold turns. Collapsing it to `None` would
    EXCLUDE exactly the records coverage@k exists to expose, inflating the mean.
    """
    assert _coverage([], ["s1#0"]) == 0.0
    assert _coverage([], ["s1#0", "s1#1"]) == 0.0
    assert _coverage(None, ["s1#0"]) is None  # the contrast, side by side


# --- report wiring ------------------------------------------------------------------------


def test_report_has_coverage_key_and_is_none_on_legacy_records():
    rep = aggregate([_rec()])
    assert rep["coverage_at_k"] is None  # always present, mirroring recall_at_k
    assert "coverage_at_k_n" not in rep  # no denominator to show


def test_report_coverage_computed_from_turn_gold():
    rep = aggregate([_rec(turns=("s1#0", "s1#1"), retrieved_turns=("s1#0",))])
    assert rep["coverage_at_k"] == 0.5
    assert rep["coverage_at_k_n"] == 1


def test_recall_at_k_byte_identical_with_coverage_added():
    """The exact recall@k values, asserted against literals, for records that now also
    carry turn gold. Adding coverage@k must not perturb recall@k by a single digit."""
    hit = _rec(qid="a", sessions=("s1",), retrieved=("s1",), turns=("s1#0",), retrieved_turns=())
    miss = _rec(qid="b", sessions=("s2",), retrieved=("s9",), turns=("s2#0",), retrieved_turns=())
    assert aggregate([hit])["recall_at_k"] == 1.0
    assert aggregate([hit, miss])["recall_at_k"] == 0.5
    assert aggregate([_rec(sessions=())])["recall_at_k"] is None


def test_report_coverage_zero_is_reported_not_nulled():
    """An all-missed run must report `coverage_at_k == 0.0`, never `None`.

    `None` means "could not be computed" and lets a consumer skip the row; a total retrieval
    failure is the single most important thing this metric has to say, so it must survive the
    `if coverages` guard in aggregate() (a falsy 0.0 in a non-empty list is still a datum).
    """
    empty = _rec(qid="a", turns=("s1#0",), retrieved_turns=())
    wrong = _rec(qid="b", turns=("s1#0",), retrieved_turns=("s9#4",))
    rep = aggregate([empty, wrong])
    assert rep["coverage_at_k"] == 0.0
    assert rep["coverage_at_k"] is not None
    assert rep["coverage_at_k_n"] == 2  # both counted, neither excluded
    assert "coverage_at_k_note" not in rep  # a computed 0.0 needs no excuse


def test_report_coverage_mixes_real_zeros_into_the_mean():
    """One perfect record and one that retrieved nothing average to 0.5, not 1.0."""
    hit = _rec(qid="a", turns=("s1#0",), retrieved_turns=("s1#0",))
    nothing = _rec(qid="b", turns=("s1#0",), retrieved_turns=())
    rep = aggregate([hit, nothing])
    assert rep["coverage_at_k"] == 0.5
    assert rep["coverage_at_k_n"] == 2


def test_coverage_excludes_unmeasured_and_abstention():
    eligible = _rec(qid="a", turns=("s1#0",), retrieved_turns=("s1#0",))
    resumed = _rec(qid="b", turns=("s1#0",), retrieved_turns=(), measured=False)
    abstain = _rec(qid="c_abs", turns=("s1#0",), retrieved_turns=())
    rep = aggregate([eligible, resumed, abstain])
    assert rep["coverage_at_k"] == 1.0  # the two zeros are excluded, not averaged in
    assert rep["coverage_at_k_n"] == 1


# --- the explanatory notes ------------------------------------------------------------------


def test_coverage_note_no_turn_gold():
    rep = aggregate([_rec()])
    assert "no turn-level gold" in rep["coverage_at_k_note"]


def test_coverage_note_system_does_not_report():
    rep = aggregate([_rec(turns=("s1#0",))], system_name="full-context")
    assert "does not report turn-level retrieval provenance" in rep["coverage_at_k_note"]
    assert "full-context" in rep["coverage_at_k_note"]


def test_coverage_note_trivial_for_nonretrieval_system():
    rep = aggregate(
        [_rec(turns=("s1#0",), retrieved_turns=("s1#0",))],
        system_name="full-context",
    )
    assert rep["coverage_at_k"] == 1.0
    assert "trivial" in rep["coverage_at_k_note"]  # mirrors recall_at_k_note


def test_coverage_note_absent_for_real_retrieval_system():
    rep = aggregate(
        [_rec(turns=("s1#0",), retrieved_turns=("s1#0",))],
        system_name="cortex-v0",
    )
    assert "coverage_at_k_note" not in rep


def test_aggregate_empty_still_works():
    rep = aggregate([])
    assert rep["coverage_at_k"] is None
    assert "coverage_at_k_note" not in rep  # nothing eligible -> nothing to explain
    assert rep["recall_at_k"] is None


# --- run metadata ----------------------------------------------------------------------------


def test_run_meta_absent_by_default_and_stamped_when_given():
    assert "run" not in aggregate([_rec()])
    rep = aggregate([_rec()], run_meta={"variant": "s", "reader_model": "m"})
    assert rep["run"] == {"reader_model": "m", "variant": "s"}  # sorted, verbatim
