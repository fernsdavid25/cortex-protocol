"""Offline, deterministic tests for MUR (memory utility ratio).

Everything here is pure file/arithmetic work — MUR never calls a reader or a judge, which is
exactly what makes it a post-hoc metric over two committed runs.
"""

from __future__ import annotations

import json

import pytest

from cortex_bench.mur import _mcnemar_exact, _paired_bootstrap_ci, compute_mur, main


def _rows(*specs: tuple[str, bool]) -> dict[str, dict]:
    return {qid: {"question_id": qid, "hypothesis": "h", "correct": ok} for qid, ok in specs}


def _write(path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def _hyp(qid: str, ok: bool) -> dict:
    return {"question_id": qid, "hypothesis": "h", "correct": ok}


# --- the ratio ------------------------------------------------------------------------------


def test_mur_basic_ratio():
    system = _rows(("q0", True), ("q1", True), ("q2", True), ("q3", False), ("q4", False))
    oracle = _rows(("q0", True), ("q1", True), ("q2", True), ("q3", True), ("q4", True))
    out = compute_mur(system, oracle, bootstrap=0)
    assert out["n_paired"] == 5
    assert out["system_accuracy"] == 0.6
    assert out["oracle_accuracy"] == 1.0
    assert out["mur"] == 0.6


def test_mur_pairs_on_question_id_intersection_only():
    system = _rows(("q0", True), ("q1", False), ("only_sys", False))
    oracle = _rows(("q0", True), ("q1", True), ("only_ora", False))
    out = compute_mur(system, oracle, bootstrap=0)
    assert out["n_paired"] == 2  # only q0/q1
    assert out["system_accuracy"] == 0.5  # the unpaired rows are not averaged in
    assert out["oracle_accuracy"] == 1.0
    assert out["mur"] == 0.5


def test_mur_last_wins_on_duplicate_question_id(tmp_path):
    """A --resume run legitimately writes a second row for the same question_id; the last
    one wins, matching regrade.load_hypotheses."""
    sys_p = _write(tmp_path / "s.jsonl", [_hyp("q0", False), _hyp("q0", True)])
    ora_p = _write(tmp_path / "o.jsonl", [_hyp("q0", True)])
    out = main(["--system", sys_p, "--oracle", ora_p, "--bootstrap", "0"])
    assert out["system_accuracy"] == 1.0  # the LAST q0 row (True), not the first
    assert out["mur"] == 1.0


def test_mur_excludes_abstention_by_default():
    system = _rows(("q0", True), ("q1_abs", False))
    oracle = _rows(("q0", True), ("q1_abs", True))
    on = compute_mur(system, oracle, bootstrap=0)
    assert on["n_paired"] == 1 and on["excluded_abstention"] == 1 and on["mur"] == 1.0
    off = compute_mur(system, oracle, exclude_abstention=False, bootstrap=0)
    assert off["n_paired"] == 2 and off["excluded_abstention"] == 0 and off["mur"] == 0.5


def test_mur_none_when_oracle_accuracy_zero():
    out = compute_mur(_rows(("q0", True)), _rows(("q0", False)), bootstrap=0)
    assert out["mur"] is None  # not 1.0, not a crash
    assert "undefined" in out["note"]


def test_mur_empty_intersection_raises_systemexit():
    with pytest.raises(SystemExit, match="benchmark/variant"):
        compute_mur(_rows(("a", True)), _rows(("b", True)), bootstrap=0)


def test_mur_all_abstention_pairs_says_so_instead_of_blaming_the_variant():
    """The ids DID match; there is simply nothing eligible left. Sending the reader off to
    check benchmark/variant would be a wild goose chase."""
    rows = _rows(("a_abs", True), ("b_abs", True))
    with pytest.raises(SystemExit, match="all 2 paired question_ids are abstention"):
        compute_mur(rows, dict(rows), bootstrap=0)


def test_mur_below_min_overlap_raises():
    # 10 rows each, but only 3 question_ids in common -> overlap 0.3 < 0.9.
    system = _rows(*[(f"q{i}", True) for i in range(10)])
    oracle = _rows(("q0", True), ("q1", True), ("q2", True), *[(f"x{i}", True) for i in range(7)])
    with pytest.raises(SystemExit, match="different runs"):
        compute_mur(system, oracle, bootstrap=0, min_overlap=0.9)


def test_min_overlap_is_measured_after_abstention_exclusion():
    """A run paired with ITSELF is one identical population, whatever its _abs share.

    The numerator of `overlap` is filtered by `exclude_abstention`, so its denominator must be
    too. Against raw row counts this is 8/12 = 0.667 and the default 0.9 bar aborts a perfect
    pairing — fatal on LoCoMo (~22% adversarial) and flaky on any small --limit subset.
    """
    rows = _rows(*[(f"q{i}", True) for i in range(8)], *[(f"a{i}_abs", True) for i in range(4)])
    out = compute_mur(rows, dict(rows), bootstrap=0, min_overlap=0.9)
    assert out["n_paired"] == 8
    assert out["excluded_abstention"] == 4
    assert out["overlap_fraction"] == 1.0  # the true id overlap, not 0.6667
    assert out["n_system_eligible"] == out["n_oracle_eligible"] == 8
    assert out["n_system_rows"] == out["n_oracle_rows"] == 12  # raw counts still reported


def test_min_overlap_uses_raw_counts_when_abstention_is_kept():
    """With --no-exclude-abstention the _abs rows are in the numerator, so they stay in the
    denominator too — the two sides of the fraction are always the same population."""
    rows = _rows(*[(f"q{i}", True) for i in range(8)], *[(f"a{i}_abs", True) for i in range(4)])
    out = compute_mur(rows, dict(rows), exclude_abstention=False, bootstrap=0, min_overlap=0.9)
    assert out["n_paired"] == 12
    assert out["overlap_fraction"] == 1.0
    assert out["n_system_eligible"] == 12


def test_min_overlap_still_rejects_different_runs_that_share_abstention_ids():
    """The relaxed denominator must not defang the guard: 3 shared of 10 eligible is still 0.3."""
    system = _rows(*[(f"q{i}", True) for i in range(10)], ("z_abs", True))
    oracle = _rows(
        ("q0", True),
        ("q1", True),
        ("q2", True),
        *[(f"x{i}", True) for i in range(7)],
        ("z_abs", True),
    )
    with pytest.raises(SystemExit, match="different runs"):
        compute_mur(system, oracle, bootstrap=0, min_overlap=0.9)


# --- statistics -----------------------------------------------------------------------------


def test_mcnemar_exact_symmetry_and_known_value():
    assert _mcnemar_exact(0, 0) == 1.0
    assert _mcnemar_exact(3, 7) == _mcnemar_exact(7, 3)
    assert _mcnemar_exact(0, 5) == pytest.approx(0.0625)  # 2 * (1/32)
    assert _mcnemar_exact(5, 5) == 1.0  # perfectly concordant split -> clipped


# Both arms vary, so a bootstrap ratio is a quotient of two resampled means and lands on many
# distinct values. A fixture whose oracle arm is constant (e.g. all (x, 1)) would make the CI a
# coarse grid that every seed hits identically — i.e. it would pass even with the RNG unseeded,
# testing nothing. 10/20 over 12/20 -> point estimate 0.8333.
_MIXED_PAIRS = [(1, 1)] * 7 + [(0, 1)] * 5 + [(1, 0)] * 3 + [(0, 0)] * 5


def test_bootstrap_ci_deterministic_under_seed():
    a = _paired_bootstrap_ci(_MIXED_PAIRS, seed=0, iters=500)
    b = _paired_bootstrap_ci(_MIXED_PAIRS, seed=0, iters=500)
    assert a == b
    assert a[0] is not None and a[1] is not None
    assert a[0] <= 0.8333 <= a[1]  # point estimate lies inside


def test_bootstrap_ci_actually_depends_on_the_seed():
    """The teeth of the determinism claim: if `seed` were ignored, this would still be equal.

    Without it, `random.Random(seed)` can be replaced by an unseeded `random.Random()` and the
    suite stays green, so "deterministic under seed" would be an unverified docstring claim.
    """
    a = _paired_bootstrap_ci(_MIXED_PAIRS, seed=0, iters=500)
    b = _paired_bootstrap_ci(_MIXED_PAIRS, seed=1, iters=500)
    assert a != b


def test_cli_threads_seed_through_to_the_bootstrap(tmp_path):
    """--seed must reach _paired_bootstrap_ci: same inputs + different seed -> different CI,
    same seed -> byte-identical CI."""
    sys_rows = [_hyp(f"q{i}", bool(s)) for i, (s, _) in enumerate(_MIXED_PAIRS)]
    ora_rows = [_hyp(f"q{i}", bool(o)) for i, (_, o) in enumerate(_MIXED_PAIRS)]
    sys_p = _write(tmp_path / "s.jsonl", sys_rows)
    ora_p = _write(tmp_path / "o.jsonl", ora_rows)

    def ci(seed: str) -> tuple:
        out = main(["--system", sys_p, "--oracle", ora_p, "--bootstrap", "500", "--seed", seed])
        return (out["mur_ci95_low"], out["mur_ci95_high"])

    # Repeatability first: an unseeded RNG would make two identical invocations disagree.
    assert ci("0") == ci("0")
    # Then dependence: the CI must be a function of --seed. Seeds 0 and 1 happen to collide at
    # 4 dp under the CLI's sorted-question_id pair ordering (q0, q1, q10, ... q2), so this uses
    # 2 — the point is that SOME seed changes the answer, not which one.
    assert ci("0") != ci("2")


def test_bootstrap_ci_none_when_disabled_or_degenerate():
    assert _paired_bootstrap_ci([(1, 1)] * 5, seed=0, iters=0) == (None, None)
    assert _paired_bootstrap_ci([(1, 1)], seed=0, iters=100) == (None, None)


# --- provenance -----------------------------------------------------------------------------


def _results(
    reader: str,
    *,
    variant: str | None = "s",
    benchmark: str = "longmemeval",
    system: str = "cortex-v0",
) -> dict:
    return {
        "accuracy": 0.9,
        "run": {
            "system": system,
            "benchmark": benchmark,
            "variant": variant,
            "reader_model": reader,
            "judge_backend": "gemini",
            "judge_model": "j",
        },
    }


def _prov_argv(tmp_path, sys_results: dict, ora_results: dict) -> list[str]:
    sys_p = _write(tmp_path / "s.jsonl", [_hyp("q0", True)])
    ora_p = _write(tmp_path / "o.jsonl", [_hyp("q0", True)])
    sr, orr = tmp_path / "s.json", tmp_path / "o.json"
    sr.write_text(json.dumps(sys_results), encoding="utf-8")
    orr.write_text(json.dumps(ora_results), encoding="utf-8")
    return [
        "--system", sys_p, "--oracle", ora_p,
        "--system-results", str(sr), "--oracle-results", str(orr),
        "--bootstrap", "0",
    ]  # fmt: skip


def test_provenance_mismatch_refuses_unless_overridden(tmp_path):
    argv = _prov_argv(
        tmp_path, _results("reader-a"), _results("reader-b", variant="oracle", system="oracle")
    )
    with pytest.raises(SystemExit, match="provenance mismatch"):
        main(argv)
    out = main([*argv, "--allow-provenance-mismatch"])
    assert out["provenance"]["provenance_checked"] is True
    assert out["provenance"]["mismatches"]["reader_model"] == ["reader-a", "reader-b"]


def test_provenance_benchmark_mismatch_refuses(tmp_path):
    """A LongMemEval numerator over a LoCoMo denominator is not a ratio of anything. Ids can
    collide across benchmarks, so the overlap guard cannot catch this — provenance must."""
    argv = _prov_argv(
        tmp_path,
        _results("same"),
        _results("same", variant="oracle", benchmark="locomo", system="oracle"),
    )
    with pytest.raises(SystemExit, match="provenance mismatch"):
        main(argv)
    out = main([*argv, "--allow-provenance-mismatch"])
    assert out["provenance"]["mismatches"]["benchmark"] == ["longmemeval", "locomo"]


def test_provenance_matched_runs_record_no_mismatch(tmp_path):
    argv = _prov_argv(
        tmp_path, _results("same"), _results("same", variant="oracle", system="full-context")
    )
    out = main(argv)
    assert out["provenance"]["provenance_checked"] is True
    assert out["provenance"]["mismatches"] == {}
    # `system` differs on purpose and must NOT be flagged: the gold-evidence denominator is
    # normally served by full-context while the numerator is the retrieval system under test.
    assert out["provenance"]["denominator_is_oracle"] is True


def test_denominator_must_be_the_oracle_variant(tmp_path):
    """Two ordinary retrieval runs must not pair into a green-checked 'MUR'.

    Without this check the tool prints the denominator's RETRIEVED-context accuracy under the
    label 'oracle accuracy (gold evidence)', with provenance_checked=True and no warning — a
    number that is not MUR by MUR's own definition.
    """
    argv = _prov_argv(tmp_path, _results("same"), _results("same", variant="s", system="naive-rag"))
    with pytest.raises(SystemExit, match="denominator run is variant 's'"):
        main(argv)


def test_non_oracle_denominator_allowed_explicitly_but_flagged(tmp_path):
    """A LoCoMo gold-evidence system cannot carry the 'oracle' tag, so the escape hatch has to
    exist — but the summary must record that the precondition was waived, not met."""
    argv = _prov_argv(
        tmp_path,
        _results("same", benchmark="locomo", variant=None),
        _results("same", benchmark="locomo", variant=None, system="locomo-oracle"),
    )
    out = main([*argv, "--allow-non-oracle-denominator"])
    assert out["provenance"]["provenance_checked"] is True
    assert out["provenance"]["denominator_is_oracle"] is False


def test_provenance_unchecked_when_results_omit_run(tmp_path):
    sys_p = _write(tmp_path / "s.jsonl", [_hyp("q0", True)])
    ora_p = _write(tmp_path / "o.jsonl", [_hyp("q0", True)])
    out = main(["--system", sys_p, "--oracle", ora_p, "--bootstrap", "0"])
    assert out["provenance"]["provenance_checked"] is False
    assert "NOT verified" in out["provenance"]["note"]


# --- CLI output -----------------------------------------------------------------------------


def test_mur_writes_summary_and_rows(tmp_path):
    sys_p = _write(tmp_path / "s.jsonl", [_hyp("q0", True), _hyp("q1", False)])
    ora_p = _write(tmp_path / "o.jsonl", [_hyp("q0", True), _hyp("q1", True)])
    outdir = tmp_path / "mur"
    main(["--system", sys_p, "--oracle", ora_p, "--bootstrap", "0", "--out", str(outdir)])

    raw = (outdir / "mur_summary.json").read_text(encoding="utf-8")
    assert raw.endswith("\n")
    assert json.loads(raw)["mur"] == 0.5
    rows = [
        json.loads(line)
        for line in (outdir / "mur_rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert rows == [
        {"question_id": "q0", "system_correct": True, "oracle_correct": True},
        {"question_id": "q1", "system_correct": False, "oracle_correct": True},
    ]


def test_mur_per_type_requires_dataset(tmp_path):
    system = _rows(("q0", True), ("q1", False))
    oracle = _rows(("q0", True), ("q1", True))
    assert "per_type" not in compute_mur(system, oracle, bootstrap=0)
    typed = compute_mur(
        system,
        oracle,
        types={"q0": "single-session-user", "q1": "temporal-reasoning"},
        bootstrap=0,
    )
    assert typed["per_type"]["single-session-user"] == {
        "n": 1,
        "system_accuracy": 1.0,
        "oracle_accuracy": 1.0,
        "mur": 1.0,
    }
    assert typed["per_type"]["temporal-reasoning"]["mur"] == 0.0


def test_mur_by_type_loads_dataset_from_disk(tmp_path):
    """--by-type maps question_ids to types via the dataset; supplied explicitly so the test
    is offline (no download)."""
    data = tmp_path / "data.json"
    entries = [
        {"question_id": q, "question_type": "multi-session", "question": "?", "answer": "a"}
        for q in ("q0", "q1")
    ]
    data.write_text(json.dumps(entries), encoding="utf-8")
    sys_p = _write(tmp_path / "s.jsonl", [_hyp("q0", True), _hyp("q1", False)])
    ora_p = _write(tmp_path / "o.jsonl", [_hyp("q0", True), _hyp("q1", True)])
    out = main(
        [
            "--system",
            sys_p,
            "--oracle",
            ora_p,
            "--by-type",
            "--data",
            str(data),
            "--bootstrap",
            "0",
        ]  # fmt: skip
    )
    assert out["per_type"]["multi-session"]["n"] == 2
    assert out["per_type"]["multi-session"]["mur"] == 0.5


def test_mur_missing_file_raises(tmp_path):
    ora_p = _write(tmp_path / "o.jsonl", [_hyp("q0", True)])
    with pytest.raises(SystemExit, match="not found"):
        main(["--system", str(tmp_path / "nope.jsonl"), "--oracle", ora_p])
