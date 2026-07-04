"""Scoring + the competitor-beating diagnostic columns (accuracy-per-dollar).

Reports what competitors hide: per-type accuracy, a distinct abstention bucket,
mean context tokens/query, $/question, latency p50/p95, and retrieval recall@k.
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
    # ESTIMATES for newer/preview readers — verify against live Gemini pricing before quoting.
    "gemini-3.5-flash": (0.30, 2.50),
    "gemini-3.1-pro-preview": (2.0, 12.0),
}

# Reader models whose PRICES entry above is a code-labeled ESTIMATE (preview/unreleased pricing),
# stamped as ``price_source: "estimated"`` into the report so a downstream $/question quote from
# one of them is never mistaken for published ground truth.
ESTIMATED_PRICE_MODELS: frozenset[str] = frozenset({"gemini-3.5-flash", "gemini-3.1-pro-preview"})

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


def aggregate(
    records: list[Record],
    reader_model: str | None = None,
    embed_model: str | None = None,
    system_name: str | None = None,
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
    }
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
    return report
