"""Scoring + the cost/accuracy diagnostic columns ($/question, recall@k, coverage@k, abstention).

Reports what competitors hide: per-type accuracy, a distinct abstention bucket,
mean context tokens/query, $/question, latency p50/p95, and retrieval recall@k.

Two retrieval metrics, deliberately not one
-------------------------------------------
``recall@k`` is SESSION-level: it asks "did we surface the right CONVERSATION". A system that
drags in a whole 40-turn session because one turn is gold scores 1.0 on it. That is why the
shipped 0.998 recall@k is not evidence of good memory.

``coverage@k`` is TURN-level: "did we surface the right MEMORY". It is fully deterministic —
no LLM, no judge — and it is computed against gold evidence TURNS (LongMemEval's per-turn
``has_answer``, LoCoMo's ``dia_id``-resolved evidence). Both metrics are reported side by
side, because their GAP is the interesting number: high recall with low coverage means the
retriever is finding the right haystack and handing the reader the wrong needle.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .memory_system import QAInstance, Usage

# USD per 1M tokens (input, output). From the research report; update as pricing changes.
PRICES: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-embedding-001": (0.15, 0.0),
    "gpt-4o-2024-08-06": (2.50, 10.0),
    # Confirmed 2026-07-24 against Google's published Gemini pricing (paid tier, standard).
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.6-flash": (1.50, 7.50),
    "gemini-3.1-pro-preview": (2.0, 12.0),  # confirmed for prompts <= 200k tokens
}

# Reader models whose PRICES entry above is a code-labeled ESTIMATE (unreleased/unpublished
# pricing), stamped as ``price_source: "estimated"`` into the report so a downstream $/question
# quote from one of them is never mistaken for published ground truth. Every rate in PRICES is
# currently confirmed against a published rate card, so this set is empty — keep the mechanism
# for the next model that ships before its pricing does.
ESTIMATED_PRICE_MODELS: frozenset[str] = frozenset()

# Systems that actually retrieve a SUBSET of sessions, for which recall@k is a real
# retrieval-quality metric. For any other system (stubs, full-context) recall@k is trivial —
# full-context returns every session (so recall is 1.0 by construction) and the stubs retrieve
# nothing — so the report flags it rather than letting it read as a genuine retrieval score.
RETRIEVAL_SYSTEMS: frozenset[str] = frozenset({"naive-rag", "cortex-v0"})


@dataclass
class Record:
    instance: QAInstance
    hypothesis: str
    correct: bool
    usage: Usage = field(default_factory=Usage)
    retrieved_session_ids: list[str] = field(default_factory=list)
    # False for resumed records: the hypothesis was re-graded from disk (so it counts toward
    # accuracy) but carries no live usage/retrieval, so it is excluded from cost/latency/recall.
    measured: bool = True
    # Turn ids covered by the retrieved memories (coverage@k). ``None`` = the system does not
    # report turn provenance -> the record is EXCLUDED from coverage, not scored 0.0. Appended
    # after ``measured`` so every existing positional Record(...) construction still binds the
    # same fields.
    retrieved_turn_ids: list[str] | None = None


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _pct(xs: Sequence[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, round((p / 100) * (len(s) - 1))))
    return s[k]


def _recall(retrieved: list[str], gold: list[str]) -> float | None:
    if not gold:
        return None
    g = set(gold)
    return len(g & set(retrieved)) / len(g)


def _coverage(retrieved: list[str] | None, gold: list[str]) -> float | None:
    """Fraction of the gold EVIDENCE TURNS contained in the retrieved memories.

    Returns ``None`` (excluded from the metric, NOT 0.0) when the instance carries no
    turn-level gold or the system does not report turn provenance — a fabricated zero would
    silently deflate the mean and make an unmeasurable run look like a bad one.

    A chunk covers every turn in its span, because the reader literally sees that turn's text.
    """
    if retrieved is None or not gold:
        return None
    g = set(gold)
    return len(g & set(retrieved)) / len(g)


def aggregate(
    records: list[Record],
    reader_model: str | None = None,
    embed_model: str | None = None,
    system_name: str | None = None,
    run_meta: dict | None = None,
) -> dict:
    total = len(records)

    def acc(rs: list[Record]) -> float:
        return round(_mean([1.0 if r.correct else 0.0 for r in rs]), 4)

    by_type: dict[str, list[Record]] = {}
    for r in records:
        by_type.setdefault(r.instance.question_type, []).append(r)

    abst = [r for r in records if r.instance.is_abstention]
    nonabst = [r for r in records if not r.instance.is_abstention]
    # Cost / latency / recall come only from records actually run live this pass. Resumed
    # records (measured=False) lack usage + retrieval, so including them would understate
    # cost and dilute recall with spurious zeros.
    measured = [r for r in records if r.measured]
    mtotal = len(measured)
    recalls = [
        x
        for r in nonabst
        if r.measured
        and (x := _recall(r.retrieved_session_ids, r.instance.answer_session_ids)) is not None
    ]
    # coverage@k rides the SAME three gates as recall@k (non-abstention, measured, gold
    # present) so the two are directly comparable; only its third gate is turn gold instead of
    # session gold, which is why its denominator can legitimately differ (coverage_at_k_n).
    coverages = [
        x
        for r in nonabst
        if r.measured
        and (x := _coverage(r.retrieved_turn_ids, r.instance.answer_turn_ids)) is not None
    ]
    has_turn_gold = any(r.instance.answer_turn_ids for r in nonabst)
    in_toks = [r.usage.input_tokens for r in measured]
    out_toks = [r.usage.output_tokens for r in measured]
    embed_toks = [r.usage.embed_tokens for r in measured]
    lat = [r.usage.latency_ms for r in measured]

    report: dict = {
        "n": total,
        "accuracy": acc(records),
        "per_type": {t: {"acc": acc(rs), "n": len(rs)} for t, rs in sorted(by_type.items())},
        "abstention": {"acc": acc(abst), "n": len(abst)},
        "non_abstention": {"acc": acc(nonabst), "n": len(nonabst)},
        "mean_input_tokens": round(_mean(in_toks), 1),
        "mean_output_tokens": round(_mean(out_toks), 1),
        "mean_embed_tokens": round(_mean(embed_toks), 1),
        "latency_ms_p50": round(_pct(lat, 50), 1),
        "latency_ms_p95": round(_pct(lat, 95), 1),
        "recall_at_k": round(_mean(recalls), 4) if recalls else None,
        # Always present, mirroring recall_at_k, so a consumer can tell "coverage was not
        # computed" (null) from "coverage was 0.0" without probing for an optional key.
        "coverage_at_k": round(_mean(coverages), 4) if coverages else None,
    }
    # Coverage's denominator legitimately differs from recall's (turn gold != session gold, and
    # not every system reports turns), so it must be visible whenever a number is reported.
    if report["coverage_at_k"] is not None:
        report["coverage_at_k_n"] = len(coverages)
    # Explain a missing/trivial coverage@k, mirroring recall_at_k_note. Only when there was at
    # least one eligible record, so aggregate([]) keeps its exact legacy shape.
    if nonabst:
        if report["coverage_at_k"] is None and not has_turn_gold:
            report["coverage_at_k_note"] = (
                "unavailable: this dataset/run carries no turn-level gold evidence"
            )
        elif report["coverage_at_k"] is None:
            who = system_name or "this system"
            report["coverage_at_k_note"] = (
                f"unavailable: {who} does not report turn-level retrieval provenance"
            )
        elif system_name is not None and system_name not in RETRIEVAL_SYSTEMS:
            report["coverage_at_k_note"] = (
                f"trivial: {system_name} does not retrieve a subset (reads full context); "
                "coverage@k is not a retrieval-quality metric here"
            )
    # Only surface measured_n on a resumed run, so ordinary runs keep an unchanged report shape.
    if mtotal != total:
        report["measured_n"] = mtotal
    # Flag recall@k as trivial for non-retrieval systems (full-context reads everything ->
    # recall is 1.0 by construction; stubs retrieve nothing) so it isn't read as a real metric.
    # Only stamped when a system_name is supplied AND a recall was computed, so callers that
    # don't pass it (unit tests) keep the legacy report shape.
    if (
        system_name is not None
        and system_name not in RETRIEVAL_SYSTEMS
        and report["recall_at_k"] is not None
    ):
        report["recall_at_k_note"] = (
            f"trivial: {system_name} does not retrieve a subset (reads full context); "
            "recall@k is not a retrieval-quality metric here"
        )
    if reader_model in PRICES and mtotal:
        pin, pout = PRICES[reader_model]
        # Stamp whether the reader's price is published or a code-labeled ESTIMATE, so a
        # downstream $/question is never quoted from preview pricing as if it were ground truth.
        report["price_source"] = (
            "estimated" if reader_model in ESTIMATED_PRICE_MODELS else "published"
        )
        # Reader-only cost (input+output at the reader rate), kept for reference.
        reader_cost = (sum(in_toks) * pin + sum(out_toks) * pout) / 1e6
        report["usd_per_question_reader"] = round(reader_cost / mtotal, 6)
        # TRUE cost: embeddings priced at the embed model's input rate, reader
        # input/output at the reader rate. Only computed when the embed model is
        # also priced (guard for missing models).
        if embed_model in PRICES:
            embed_in = PRICES[embed_model][0]
            total_cost = (
                sum(embed_toks) * embed_in + sum(in_toks) * pin + sum(out_toks) * pout
            ) / 1e6
            report["usd_per_question"] = round(total_cost / mtotal, 6)
    # Stamp the run's identity (reader, judge, benchmark, levers) so MUR's "same reader + same
    # judge on both sides" precondition is CHECKABLE from the committed artifacts instead of
    # taken on trust. Gated exactly like price_source, so callers that pass nothing (every
    # existing unit test) keep the legacy report shape byte for byte.
    if run_meta:
        report["run"] = dict(sorted(run_meta.items()))
    return report
