"""MUR — Memory Utility Ratio: accuracy(retrieved context) / accuracy(gold-evidence context).

Why this exists
---------------
A raw accuracy number conflates two very different things: how good the MEMORY is at putting
the right evidence in front of the reader, and how good the READER is at using it. Swap in a
stronger reader and accuracy climbs while the memory system is unchanged; swap in a more
lenient judge and it climbs again. Neither move made the memory better.

MUR divides them out. The numerator is a normal run — the system retrieves, the reader
answers. The denominator is the SAME reader, graded by the SAME judge, but handed the gold
evidence context (LongMemEval's ``oracle`` variant, already supported by
``dataset.VARIANT_FILES``). Reader capability and judge leniency appear in both terms and
cancel to first order, so MUR isolates what the memory layer actually delivered. MUR = 1.0
means "retrieval cost the reader nothing"; MUR = 0.6 means the memory layer threw away 40%
of the accuracy the reader could have had.

MUR is computed POST HOC by pairing two already-committed runs. It never invokes a reader or
a judge and it costs nothing to recompute, so it can be re-derived over any two artifacts in
``bench/results/``.

Why it pairs hypotheses, not the two results.json scalars
---------------------------------------------------------
``accuracy`` in a results.json is rounded to 4 dp over whatever ``--limit/--seed/--shuffle``
subset that run happened to use. Dividing two of those silently compares two different
populations. Pairing the ``*_hypotheses.jsonl`` rows per ``question_id`` (intersection only)
instead gives an exact ratio over one common population, plus per-type MUR, a paired McNemar
test, and a paired bootstrap CI. The results.json files are an optional second input, used
only to cross-check the preconditions: that both runs really did share a reader, a judge and
a benchmark, and that the denominator really is the ``oracle`` (gold-evidence) run. Pass them
— without them the tool can only say the preconditions were UNCHECKED.

Usage
-----
::

    python -m cortex_bench.mur \
        --system bench/results/cortex-v0_s_hypotheses.jsonl \
        --oracle bench/results/cortex-v0_oracle_hypotheses.jsonl \
        --system-results bench/results/cortex-v0_s_results.json \
        --oracle-results bench/results/cortex-v0_oracle_results.json \
        --by-type --variant s --out bench/results/mur_s

LoCoMo has no oracle variant, so MUR ships LongMemEval-first. The CLI pairs on
``question_id`` alone, so a future gold-evidence-context LoCoMo system needs no code change
here — only ``--allow-non-oracle-denominator``, since such a run cannot carry the ``oracle``
variant tag that the denominator check looks for.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from .regrade import load_hypotheses

# Below this fraction of the smaller run, a "pair" is really two different populations and
# the ratio is not interpretable. Overridable, but loud by default.
DEFAULT_MIN_OVERLAP: float = 0.9
DEFAULT_BOOTSTRAP_ITERS: int = 2000

# The variant tag a gold-evidence run carries in results.json's ``run`` block. Matches the
# ``"oracle"`` key in ``dataset.VARIANT_FILES``, which is what ``run.py --variant oracle``
# stamps. The denominator of MUR must be this run; anything else makes the ratio meaningless.
ORACLE_VARIANT: str = "oracle"


def _is_abstention(question_id: str) -> bool:
    """Abstention questions are flagged by the ``_abs`` suffix (same rule as QAInstance)."""
    return question_id.endswith("_abs")


def _correct(row: dict) -> bool:
    return bool(row.get("correct", False))


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value for ``b``/``c`` discordant pairs.

    ``b`` = system correct while oracle was wrong, ``c`` = the reverse. Under the null the
    discordant pairs split Binomial(b+c, 0.5), so the exact two-sided p is twice the lower
    tail, clipped to 1.0. Exact (not the chi-square approximation) because the discordant
    count is often small, and stdlib-only (``math.comb``) because no new dependency may be
    added. Symmetric in its arguments, and 1.0 when there are no discordant pairs at all.
    """
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    tail = sum(math.comb(n, i) for i in range(lo + 1)) * (0.5**n)
    return min(1.0, 2.0 * tail)


def _ratio(sys_correct: list[int], ora_correct: list[int]) -> float | None:
    """MUR for one (possibly resampled) paired sample; ``None`` when the oracle scored 0."""
    denom = _mean([float(x) for x in ora_correct])
    if denom == 0.0:
        return None
    return _mean([float(x) for x in sys_correct]) / denom


def _paired_bootstrap_ci(
    pairs: list[tuple[int, int]], seed: int, iters: int
) -> tuple[float | None, float | None]:
    """Percentile 95% CI for MUR by resampling the PAIRS (not the two runs independently).

    Resampling pairs preserves the per-question correlation between the two arms, which is the
    whole point of running the same questions through both. Deterministic under ``seed``
    (``random.Random``), so a committed CI is reproducible. Resamples in which the oracle arm
    scored zero are skipped (the ratio is undefined there, not zero).
    """
    if iters <= 0 or len(pairs) < 2:
        return None, None
    rng = random.Random(seed)
    n = len(pairs)
    ratios: list[float] = []
    for _ in range(iters):
        picked = [pairs[rng.randrange(n)] for _ in range(n)]
        r = _ratio([p[0] for p in picked], [p[1] for p in picked])
        if r is not None:
            ratios.append(r)
    if not ratios:
        return None, None
    ratios.sort()

    def pct(p: float) -> float:
        k = max(0, min(len(ratios) - 1, int(round(p * (len(ratios) - 1)))))
        return round(ratios[k], 4)

    return pct(0.025), pct(0.975)


def compute_mur(
    system_rows: dict[str, dict],
    oracle_rows: dict[str, dict],
    *,
    exclude_abstention: bool = True,
    types: dict[str, str] | None = None,
    seed: int = 0,
    bootstrap: int = DEFAULT_BOOTSTRAP_ITERS,
    min_overlap: float = 0.0,
) -> dict:
    """Pure, I/O-free MUR over two already-loaded hypotheses maps. This is the testable core.

    ``min_overlap`` guards against pairing two runs that only incidentally share question ids
    (different benchmark, variant, or ``--limit`` subset); it is checked here, beside the
    pairing it protects, rather than in the CLI, so a programmatic caller gets the same guard.
    """
    matched = set(system_rows) & set(oracle_rows)
    excluded = 0
    # The overlap fraction must compare like with like. ``exclude_abstention`` shrinks the
    # numerator, so the denominator has to lose the SAME questions — otherwise a run paired
    # with itself scores len(non-abstention) / len(all rows) and gets rejected as "two
    # different runs". LoCoMo is ~22% adversarial (_abs) and LongMemEval ~6%, so an unfiltered
    # denominator makes the default 0.9 bar unreachable on LoCoMo and fragile on subsets.
    if exclude_abstention:
        eligible = {q for q in matched if not _is_abstention(q)}
        excluded = len(matched) - len(eligible)
        system_pop = {q for q in system_rows if not _is_abstention(q)}
        oracle_pop = {q for q in oracle_rows if not _is_abstention(q)}
    else:
        eligible = matched
        system_pop, oracle_pop = set(system_rows), set(oracle_rows)
    qids = sorted(eligible)
    if not qids:
        # Distinguish "the two runs share nothing" from "everything they share is an
        # abstention question" — the second is a real pairing with an empty eligible set, and
        # sending the reader off to check benchmark/variant would be a wild goose chase.
        if matched:
            raise SystemExit(
                f"all {len(matched)} paired question_ids are abstention (_abs) questions; "
                "pass --no-exclude-abstention to score them anyway"
            )
        raise SystemExit("no question_ids matched — are these the same benchmark/variant?")

    smaller = min(len(system_pop), len(oracle_pop))
    overlap = len(qids) / smaller if smaller else 0.0
    if min_overlap and overlap < min_overlap:
        raise SystemExit(
            f"only {len(qids)}/{smaller} eligible question_ids paired (overlap {overlap:.3f} < "
            f"{min_overlap:.3f}) — these look like different runs; pass "
            "--min-overlap to lower the bar deliberately"
        )

    pairs = [(int(_correct(system_rows[q])), int(_correct(oracle_rows[q]))) for q in qids]
    sys_c = [p[0] for p in pairs]
    ora_c = [p[1] for p in pairs]
    system_accuracy = _mean([float(x) for x in sys_c])
    oracle_accuracy = _mean([float(x) for x in ora_c])
    mur = _ratio(sys_c, ora_c)
    b = sum(1 for s, o in pairs if s and not o)
    c = sum(1 for s, o in pairs if o and not s)
    lo, hi = _paired_bootstrap_ci(pairs, seed, bootstrap)

    out: dict = {
        "n_system_rows": len(system_rows),
        "n_oracle_rows": len(oracle_rows),
        # The population overlap_fraction is measured against: raw rows minus whatever
        # ``exclude_abstention`` dropped, so the fraction is a real id-overlap and not a
        # by-product of how adversarial the benchmark is.
        "n_system_eligible": len(system_pop),
        "n_oracle_eligible": len(oracle_pop),
        "n_paired": len(qids),
        "overlap_fraction": round(overlap, 4),
        "excluded_abstention": excluded,
        "system_accuracy": round(system_accuracy, 4),
        "oracle_accuracy": round(oracle_accuracy, 4),
        "mur": round(mur, 4) if mur is not None else None,
        "mur_ci95_low": lo,
        "mur_ci95_high": hi,
        "mcnemar": {"b": b, "c": c, "p_two_sided": round(_mcnemar_exact(b, c), 6)},
    }
    if mur is None:
        # Not 1.0 and not a crash: a zero-accuracy oracle means the denominator run itself
        # failed, so the ratio carries no information about the memory layer.
        out["note"] = "mur undefined: the oracle (gold-evidence) run scored 0.0 accuracy"

    if types:
        per_type: dict[str, dict] = {}
        buckets: dict[str, list[tuple[int, int]]] = {}
        for q, pair in zip(qids, pairs, strict=True):
            buckets.setdefault(types.get(q, "unknown"), []).append(pair)
        for t, bp in sorted(buckets.items()):
            t_mur = _ratio([p[0] for p in bp], [p[1] for p in bp])
            per_type[t] = {
                "n": len(bp),
                "system_accuracy": round(_mean([float(p[0]) for p in bp]), 4),
                "oracle_accuracy": round(_mean([float(p[1]) for p in bp]), 4),
                "mur": round(t_mur, 4) if t_mur is not None else None,
            }
        out["per_type"] = per_type
    return out


def _load_results(path: str | None) -> dict | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _check_provenance(
    system_results: dict | None,
    oracle_results: dict | None,
    allow_mismatch: bool,
    allow_non_oracle: bool = False,
) -> dict:
    """Verify the two runs shared a reader, a judge and a benchmark, AND that the denominator
    really is the gold-evidence run — MUR's whole premise.

    Requires the ``run`` block that ``metrics.aggregate(run_meta=...)`` stamps. Runs committed
    before that stamp existed carry no ``run`` block, so provenance is recorded as UNCHECKED
    (with a caveat) rather than silently assumed good.

    ``variant`` is checked only on the DENOMINATOR, and only for equality with
    ``ORACLE_VARIANT``. The numerator is a retrieval run on some other variant by definition,
    so its variant must differ; and ``system`` is deliberately NOT compared, because the
    oracle-context run is normally served by ``full-context`` while the numerator is
    ``cortex-v0`` — a legitimate difference, not a mispairing.
    """
    sys_run = (system_results or {}).get("run")
    ora_run = (oracle_results or {}).get("run")
    if not sys_run or not ora_run:
        return {
            "provenance_checked": False,
            "note": "results.json missing the 'run' block (pre-dates run-metadata stamping); "
            "MUR's same-reader/same-judge/oracle-denominator preconditions were NOT verified",
        }
    # ``benchmark`` belongs here and not in the id-overlap guard: LongMemEval and LoCoMo can
    # in principle share a question_id string, so a high overlap would not catch the swap.
    fields = ("reader_model", "judge_backend", "judge_model", "benchmark")
    mismatches = {
        f: [sys_run.get(f), ora_run.get(f)] for f in fields if sys_run.get(f) != ora_run.get(f)
    }
    if mismatches and not allow_mismatch:
        detail = "; ".join(f"{f}: {a!r} vs {b!r}" for f, (a, b) in sorted(mismatches.items()))
        raise SystemExit(
            f"provenance mismatch between the two runs ({detail}). MUR assumes the SAME reader, "
            "judge and benchmark on both sides; pass --allow-provenance-mismatch to override "
            "deliberately."
        )
    denominator_is_oracle = ora_run.get("variant") == ORACLE_VARIANT
    if not denominator_is_oracle and not allow_non_oracle:
        raise SystemExit(
            f"the denominator run is variant {ora_run.get('variant')!r}, not "
            f"{ORACLE_VARIANT!r} (system {ora_run.get('system')!r}). MUR's denominator must be "
            "the GOLD-EVIDENCE context run — dividing by another retrieval run yields a ratio "
            "of two retrievers, not a memory utility ratio. Pass "
            "--allow-non-oracle-denominator if the denominator really is a gold-evidence run "
            "that does not carry the 'oracle' variant tag (e.g. a LoCoMo oracle system)."
        )
    return {
        "provenance_checked": True,
        "denominator_is_oracle": denominator_is_oracle,
        "system_run": sys_run,
        "oracle_run": ora_run,
        "mismatches": mismatches,
    }


def _load_types(args: argparse.Namespace) -> dict[str, str]:
    """``{question_id: question_type}`` — loaded ONLY when --by-type is requested."""
    if args.benchmark == "locomo":
        from .locomo import load_locomo

        instances = load_locomo(args.data or f"{args.data_dir}/locomo10.json")
    else:
        from .dataset import download, load_instances

        instances = load_instances(args.data or download(args.variant, args.data_dir))
    return {inst.question_id: inst.question_type for inst in instances}


def _print_table(summary: dict) -> None:
    # ASCII only: this prints to a Windows cp1252 console in the maintainer's environment.
    mur = summary["mur"]
    mur_s = "n/a" if mur is None else f"{mur:.4f}"
    ci = ""
    if summary["mur_ci95_low"] is not None:
        ci = f"  [95% CI {summary['mur_ci95_low']:.4f}, {summary['mur_ci95_high']:.4f}]"
    print(f"\n=== MUR (memory utility ratio) | n={summary['n_paired']} paired ===")
    print(f"system accuracy (retrieved context) : {summary['system_accuracy']:.4f}")
    print(f"oracle accuracy (gold evidence)     : {summary['oracle_accuracy']:.4f}")
    print(f"MUR                                 : {mur_s}{ci}")
    mc = summary["mcnemar"]
    print(
        f"McNemar (paired)                    : b={mc['b']} c={mc['c']} p={mc['p_two_sided']:.4f}"
    )
    if summary.get("excluded_abstention"):
        print(f"excluded abstention questions       : {summary['excluded_abstention']}")
    prov = summary.get("provenance", {})
    if not prov.get("provenance_checked", False):
        print(f"WARN: {prov.get('note', 'provenance unchecked')}")
    elif not prov.get("denominator_is_oracle", False):
        print(
            "WARN: the denominator run is not tagged variant='oracle' "
            "(--allow-non-oracle-denominator was passed); this is only MUR if that run really "
            "was handed the gold evidence context"
        )
    if summary.get("per_type"):
        print(f"\n{'question_type':<30}{'n':>5}{'system':>9}{'oracle':>9}{'MUR':>9}")
        for t, v in summary["per_type"].items():
            t_mur = "n/a" if v["mur"] is None else f"{v['mur']:.3f}"
            print(
                f"{t:<30}{v['n']:>5}{v['system_accuracy']:>9.3f}"
                f"{v['oracle_accuracy']:>9.3f}{t_mur:>9}"
            )


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(
        description="MUR: accuracy(retrieved context) / accuracy(gold-evidence context), "
        "paired post-hoc over two committed hypotheses files."
    )
    ap.add_argument("--system", required=True, help="numerator *_hypotheses.jsonl (retrieved)")
    ap.add_argument("--oracle", required=True, help="denominator *_hypotheses.jsonl (gold context)")
    ap.add_argument("--system-results", default="", help="optional *_results.json for provenance")
    ap.add_argument("--oracle-results", default="", help="optional *_results.json for provenance")
    ap.add_argument(
        "--exclude-abstention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="drop _abs questions (they have no evidence to retrieve, so they measure the "
        "reader's abstention policy, not memory utility). Default: on.",
    )
    ap.add_argument(
        "--by-type",
        action="store_true",
        help="also report MUR per question_type (loads the dataset to map ids to types)",
    )
    ap.add_argument("--benchmark", default="longmemeval", choices=["longmemeval", "locomo"])
    ap.add_argument("--variant", default="s", choices=["s", "m", "oracle"])
    ap.add_argument("--data", default="", help="explicit dataset path (overrides --variant)")
    ap.add_argument("--data-dir", default="bench/data")
    ap.add_argument(
        "--min-overlap",
        type=float,
        default=DEFAULT_MIN_OVERLAP,
        help="refuse to pair runs sharing less than this fraction of the smaller run's ids",
    )
    ap.add_argument(
        "--allow-provenance-mismatch",
        action="store_true",
        help="pair runs with different readers/judges/benchmarks anyway (the ratio no longer "
        "cancels reader capability or judge leniency — say so wherever you quote it)",
    )
    ap.add_argument(
        "--allow-non-oracle-denominator",
        action="store_true",
        help=f"accept a denominator whose run.variant is not {ORACLE_VARIANT!r} (e.g. a LoCoMo "
        "gold-evidence system). Without this, dividing by an ordinary retrieval run is refused.",
    )
    ap.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP_ITERS)
    ap.add_argument("--seed", type=int, default=0, help="seed for the deterministic bootstrap")
    ap.add_argument(
        "--out", default="", help="directory to write mur_summary.json + mur_rows.jsonl"
    )
    args = ap.parse_args(argv)

    sys_path, ora_path = Path(args.system), Path(args.oracle)
    for p in (sys_path, ora_path):
        if not p.exists():
            raise SystemExit(f"hypotheses file not found: {p}")
    system_rows = load_hypotheses(sys_path)
    oracle_rows = load_hypotheses(ora_path)

    provenance = _check_provenance(
        _load_results(args.system_results or None),
        _load_results(args.oracle_results or None),
        args.allow_provenance_mismatch,
        args.allow_non_oracle_denominator,
    )
    types = _load_types(args) if args.by_type else None

    summary = compute_mur(
        system_rows,
        oracle_rows,
        exclude_abstention=args.exclude_abstention,
        types=types,
        seed=args.seed,
        bootstrap=args.bootstrap,
        min_overlap=args.min_overlap,
    )
    summary["provenance"] = provenance
    _print_table(summary)

    if args.out:
        outdir = Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
        paired = sorted(set(system_rows) & set(oracle_rows))
        if args.exclude_abstention:
            paired = [q for q in paired if not _is_abstention(q)]
        with (outdir / "mur_rows.jsonl").open("w", encoding="utf-8") as fh:
            for q in paired:
                fh.write(
                    json.dumps(
                        {
                            "question_id": q,
                            "system_correct": _correct(system_rows[q]),
                            "oracle_correct": _correct(oracle_rows[q]),
                        }
                    )
                    + "\n"
                )
        (outdir / "mur_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {outdir / 'mur_rows.jsonl'} and {outdir / 'mur_summary.json'}")
    return summary


if __name__ == "__main__":  # pragma: no cover
    main()
