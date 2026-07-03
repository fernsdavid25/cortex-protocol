"""End-to-end harness smoke test on stub systems with the deterministic offline judge."""

from cortex_bench.judge import Judge
from cortex_bench.memory_system import MemorySystem, QAInstance, Usage
from cortex_bench.metrics import aggregate
from cortex_bench.run import run
from cortex_bench.systems.stub import GoldStub, NullStub


def test_gold_stub_scores_nonabstention_perfect(sample_instances):
    records = run(sample_instances, GoldStub(), Judge("offline"))
    rep = aggregate(records, reader_model="gemini-2.5-flash-lite")
    assert rep["n"] == 3
    assert rep["non_abstention"]["acc"] == 1.0  # gold answer always contained
    assert rep["abstention"]["acc"] == 0.0  # gold stub never abstains
    assert "usd_per_question_reader" in rep


def test_null_stub_only_gets_abstention(sample_instances):
    records = run(sample_instances, NullStub(), Judge("offline"))
    rep = aggregate(records)
    assert rep["non_abstention"]["acc"] == 0.0  # "I don't know" never contains the gold
    assert rep["abstention"]["acc"] == 1.0  # always abstains


def test_run_is_deterministic(sample_instances):
    a = aggregate(run(sample_instances, GoldStub(), Judge("offline")))
    b = aggregate(run(sample_instances, GoldStub(), Judge("offline")))
    assert a["accuracy"] == b["accuracy"]


class _FlakySystem(MemorySystem):
    """Gold-answering stub that raises on one specific question_id (simulates a hard failure)."""

    name = "flaky"

    def __init__(self, fail_qid: str) -> None:
        self.fail_qid = fail_qid

    def ingest(self, instance: QAInstance) -> Usage:
        return Usage()

    def answer(self, instance: QAInstance) -> tuple[str, Usage]:
        if instance.question_id == self.fail_qid:
            raise RuntimeError("provider exhausted retries")
        return instance.answer, Usage()


def test_run_survives_failing_instance(sample_instances):
    records = run(sample_instances, _FlakySystem(fail_qid="q2"), Judge("offline"))
    by_id = {r.instance.question_id: r for r in records}
    assert len(records) == 3  # run completed despite the failure
    assert by_id["q2"].correct is False  # the failed instance counts as wrong
    assert by_id["q2"].hypothesis == ""  # recorded with an empty hypothesis
    # Other instances are unaffected: they were answered normally (non-empty gold hypothesis).
    assert by_id["q1"].correct is True
    assert by_id["q1"].hypothesis == "Goa"
    assert by_id["q3_abs"].hypothesis != ""


def test_run_writes_hypotheses_incrementally(sample_instances, tmp_path):
    out = tmp_path / "hyps.jsonl"
    run(sample_instances, _FlakySystem(fail_qid="q2"), Judge("offline"), hyp_path=out)
    lines = [line for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 3  # one line per instance, including the failed one
    import json

    rows = [json.loads(line) for line in lines]
    assert [r["question_id"] for r in rows] == ["q1", "q2", "q3_abs"]
    assert rows[1]["hypothesis"] == ""  # failed instance still recorded for partial progress
    assert rows[1]["failed"] is True  # ...and flagged as a failure so resume retries it
    assert "failed" not in rows[0]  # successful instances are not flagged


class _TrackingSystem(MemorySystem):
    """Gold-answering stub that records which question_ids it actually answered."""

    name = "track"

    def __init__(self) -> None:
        self.answered: list[str] = []

    def ingest(self, instance: QAInstance) -> Usage:
        return Usage()

    def answer(self, instance: QAInstance) -> tuple[str, Usage]:
        self.answered.append(instance.question_id)
        return instance.answer, Usage(input_tokens=10, output_tokens=2)


class _SpyJudge(Judge):
    """Offline judge that records which question_ids it was asked to grade."""

    def __init__(self) -> None:
        super().__init__("offline")
        self.graded: list[str] = []

    def grade(self, instance: QAInstance, hypothesis: str) -> bool:
        self.graded.append(instance.question_id)
        return super().grade(instance, hypothesis)


class _FailJudge(Judge):
    """Offline judge that raises on one question_id (simulates a judge-backend quota death)."""

    def __init__(self, fail_qid: str) -> None:
        super().__init__("offline")
        self.fail_qid = fail_qid

    def grade(self, instance: QAInstance, hypothesis: str) -> bool:
        if instance.question_id == self.fail_qid:
            raise RuntimeError("judge quota exhausted")
        return super().grade(instance, hypothesis)


def test_run_persists_verdict_in_hypotheses_file(sample_instances, tmp_path):
    """Each written line carries its judged verdict, so a later resume can reuse it."""
    import json

    out = tmp_path / "hyps.jsonl"
    records = run(sample_instances, GoldStub(), Judge("offline"), hyp_path=out)
    by_id = {r.instance.question_id: r for r in records}
    rows = {
        json.loads(line)["question_id"]: json.loads(line)
        for line in out.read_text(encoding="utf-8").splitlines()
        if line
    }
    for qid, row in rows.items():
        assert "correct" in row  # verdict persisted
        assert row["correct"] == by_id[qid].correct


def test_run_resume_skips_completed_and_appends(sample_instances, tmp_path):
    """Resume must NOT re-run the costly reader for already-answered instances, but must
    still score them so accuracy covers the full set."""
    import json

    out = tmp_path / "hyps.jsonl"
    # q1 was answered before the run crashed; its hypothesis is on disk (legacy: no verdict).
    out.write_text(json.dumps({"question_id": "q1", "hypothesis": "Goa"}) + "\n", encoding="utf-8")
    sys = _TrackingSystem()
    records = run(sample_instances, sys, Judge("offline"), hyp_path=out, resume=True)

    # q1 was skipped (no reader call); only the remaining two ran.
    assert sys.answered == ["q2", "q3_abs"]
    # All three are still scored.
    assert len(records) == 3
    by_id = {r.instance.question_id: r for r in records}
    # q1 graded from the saved "Goa" (== gold) -> correct, but marked unmeasured.
    assert by_id["q1"].correct is True
    assert by_id["q1"].hypothesis == "Goa"
    assert by_id["q1"].measured is False
    assert by_id["q2"].measured is True
    # All three instances are present; q1's re-graded verdict is written back (a 2nd q1 row).
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert {r["question_id"] for r in rows} == {"q1", "q2", "q3_abs"}
    assert any(r["question_id"] == "q1" and "correct" in r for r in rows)  # verdict persisted


def test_run_resume_reuses_stored_verdict_without_regrading(sample_instances, tmp_path):
    """A resumed line with a persisted verdict is trusted as-is — the judge is NOT called for
    it (so resume is quota-free for completed work, even if the stored verdict disagrees with
    what a fresh grade would produce)."""
    import json

    out = tmp_path / "hyps.jsonl"
    # Hypothesis does NOT contain the gold ("Goa"), yet the stored verdict says correct=true.
    out.write_text(
        json.dumps({"question_id": "q1", "hypothesis": "nonsense", "correct": True}) + "\n",
        encoding="utf-8",
    )
    spy = _SpyJudge()
    records = run(sample_instances, _TrackingSystem(), spy, hyp_path=out, resume=True)
    by_id = {r.instance.question_id: r for r in records}

    assert "q1" not in spy.graded  # verdict reused, no re-grade
    assert by_id["q1"].correct is True  # the stored verdict, not a fresh "False"


def test_run_resume_regrades_only_legacy_lines(sample_instances, tmp_path):
    """Legacy lines (no stored verdict) are re-graded; everything else freshly graded once."""
    import json

    out = tmp_path / "hyps.jsonl"
    out.write_text(json.dumps({"question_id": "q1", "hypothesis": "Goa"}) + "\n", encoding="utf-8")
    spy = _SpyJudge()
    run(sample_instances, _TrackingSystem(), spy, hyp_path=out, resume=True)
    # q1 legacy -> re-graded; q2/q3 fresh -> graded once. All three hit the judge exactly once.
    assert sorted(spy.graded) == ["q1", "q2", "q3_abs"]


def test_run_resume_retries_failed_lines(sample_instances, tmp_path):
    """A recorded failure (failed=True, e.g. a mid-run reader/quota error) is RE-TRIED on resume,
    not skipped — so a transient/quota failure can never permanently poison the result."""
    import json

    out = tmp_path / "hyps.jsonl"
    out.write_text(
        json.dumps({"question_id": "q1", "hypothesis": "", "correct": False, "failed": True})
        + "\n"
        + json.dumps({"question_id": "q2", "hypothesis": "18", "correct": True})
        + "\n",
        encoding="utf-8",
    )
    sys = _TrackingSystem()
    run(sample_instances, sys, Judge("offline"), hyp_path=out, resume=True)
    assert "q1" in sys.answered  # the failed line is retried
    assert "q2" not in sys.answered  # the completed line is skipped
    assert "q3_abs" in sys.answered  # never-seen instance runs fresh


def test_run_resume_skips_empty_successful_answer(sample_instances, tmp_path):
    """A legitimately-empty answer (recorded without failed=True) is treated as DONE and not
    retried — only true failures (failed=True) are retried. Prevents an empty-answer instance
    from re-running the reader on every resume forever."""
    import json

    out = tmp_path / "hyps.jsonl"
    out.write_text(
        json.dumps({"question_id": "q1", "hypothesis": "", "correct": False}) + "\n",
        encoding="utf-8",
    )
    sys = _TrackingSystem()
    records = run(sample_instances, sys, Judge("offline"), hyp_path=out, resume=True)
    by_id = {r.instance.question_id: r for r in records}
    assert "q1" not in sys.answered  # empty-but-successful answer is NOT re-run
    assert by_id["q1"].correct is False  # its stored verdict is kept
    assert by_id["q1"].measured is False


def test_run_resume_tolerates_torn_last_line(sample_instances, tmp_path):
    """A truncated final line (process killed mid-write) is skipped, not fatal — resume still
    loads the good rows instead of aborting and re-doing everything."""
    import json

    out = tmp_path / "hyps.jsonl"
    out.write_text(
        json.dumps({"question_id": "q2", "hypothesis": "18", "correct": True})
        + "\n"
        + '{"question_id": "q1", "hypothesis": "Go',  # torn line, no closing/newline
        encoding="utf-8",
    )
    sys = _TrackingSystem()
    run(sample_instances, sys, Judge("offline"), hyp_path=out, resume=True)
    assert "q2" not in sys.answered  # good row honored -> skipped
    assert "q1" in sys.answered  # torn row dropped -> re-run


def test_run_resume_writes_back_legacy_verdict(sample_instances, tmp_path):
    """A re-graded legacy line (no stored verdict) has its verdict written back, so a SECOND
    resume reuses it instead of re-grading (judge-free)."""
    import json

    out = tmp_path / "hyps.jsonl"
    out.write_text(json.dumps({"question_id": "q1", "hypothesis": "Goa"}) + "\n", encoding="utf-8")
    # First resume re-grades q1 and should persist its verdict.
    run(sample_instances, _TrackingSystem(), Judge("offline"), hyp_path=out, resume=True)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    q1_rows = [r for r in rows if r["question_id"] == "q1"]
    assert any("correct" in r for r in q1_rows)  # verdict written back
    # Second resume must NOT re-grade q1 (its verdict is now persisted).
    spy = _SpyJudge()
    run(sample_instances, _TrackingSystem(), spy, hyp_path=out, resume=True)
    assert "q1" not in spy.graded


def test_run_resume_repairs_missing_trailing_newline(sample_instances, tmp_path):
    """If the prior last line lacks its trailing newline (crash mid-write), resume must not
    concatenate the next record onto it — every line in the final file stays valid JSON."""
    import json

    out = tmp_path / "hyps.jsonl"
    # Complete object but NO trailing newline.
    out.write_text(
        json.dumps({"question_id": "q2", "hypothesis": "18", "correct": True}), encoding="utf-8"
    )
    run(sample_instances, _TrackingSystem(), Judge("offline"), hyp_path=out, resume=True)
    lines = [line for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line in lines:
        json.loads(line)  # would raise if two objects were concatenated onto one line
    assert {json.loads(line)["question_id"] for line in lines} == {"q1", "q2", "q3_abs"}


def test_run_judge_failure_on_fresh_instance_aborts_without_fabricating(sample_instances, tmp_path):
    """A hard judge failure (after a successful answer) must ABORT the run, not silently record
    the instance as wrong with an empty hypothesis — otherwise the headline accuracy is deflated
    and the good reader answer is lost."""
    import json

    import pytest

    out = tmp_path / "hyps.jsonl"
    with pytest.raises(RuntimeError, match="judge quota exhausted"):
        run(sample_instances, GoldStub(), _FailJudge(fail_qid="q2"), hyp_path=out, resume=False)
    # The judge failed on q2: it must NOT have been written as a fabricated wrong verdict.
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert all(r["question_id"] != "q2" for r in rows)


def test_run_without_resume_truncates(sample_instances, tmp_path):
    """Without --resume, a pre-existing hypotheses file is overwritten (current behavior)."""
    import json

    out = tmp_path / "hyps.jsonl"
    out.write_text(
        json.dumps({"question_id": "stale", "hypothesis": "old"}) + "\n", encoding="utf-8"
    )
    sys = _TrackingSystem()
    run(sample_instances, sys, Judge("offline"), hyp_path=out)
    assert sys.answered == ["q1", "q2", "q3_abs"]  # everything re-run
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert [r["question_id"] for r in rows] == ["q1", "q2", "q3_abs"]  # stale line gone
