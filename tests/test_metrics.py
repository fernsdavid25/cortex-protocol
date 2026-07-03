from cortex_bench.memory_system import QAInstance, Usage
from cortex_bench.metrics import Record, aggregate


def _inst(qid, qtype, gold_sessions=()):
    return QAInstance(qid, qtype, "q?", "a", answer_session_ids=list(gold_sessions))


def test_aggregate_accuracy_and_buckets():
    records = [
        Record(
            _inst("a", "single-session-user", ["s1"]),
            "a",
            True,
            Usage(input_tokens=100, output_tokens=10, latency_ms=50, embed_tokens=1000),
            retrieved_session_ids=["s1"],
        ),
        Record(
            _inst("b", "single-session-user", ["s2"]),
            "wrong",
            False,
            Usage(input_tokens=200, output_tokens=20, latency_ms=150, embed_tokens=2000),
            retrieved_session_ids=["s9"],
        ),
        Record(
            _inst("c_abs", "multi-session"),
            "I don't know",
            True,
            Usage(input_tokens=50, output_tokens=5, latency_ms=10, embed_tokens=500),
        ),
    ]
    rep = aggregate(
        records,
        reader_model="gemini-2.5-flash-lite",
        embed_model="gemini-embedding-001",
    )

    assert rep["n"] == 3
    assert rep["accuracy"] == round(2 / 3, 4)
    assert rep["per_type"]["single-session-user"] == {"acc": 0.5, "n": 2}
    assert rep["abstention"] == {"acc": 1.0, "n": 1}
    assert rep["non_abstention"]["n"] == 2
    # recall@k over non-abstention with gold sessions: s1 hit (1.0), s2 miss (0.0) -> 0.5
    assert rep["recall_at_k"] == 0.5
    assert rep["mean_input_tokens"] == round((100 + 200 + 50) / 3, 1)
    assert rep["mean_embed_tokens"] == round((1000 + 2000 + 500) / 3, 1)
    # reader-only cost: input 350*0.10/1e6 + output 35*0.40/1e6, per question
    expected_reader = (350 * 0.10 + 35 * 0.40) / 1e6 / 3
    assert rep["usd_per_question_reader"] == round(expected_reader, 6)
    # true cost ALSO prices embeddings at the embed model's input rate (0.15/1M)
    expected_true = (3500 * 0.15 + 350 * 0.10 + 35 * 0.40) / 1e6 / 3
    assert rep["usd_per_question"] == round(expected_true, 6)
    # true cost is strictly higher than reader-only (embeddings are no longer hidden)
    assert rep["usd_per_question"] > rep["usd_per_question_reader"]


def test_aggregate_reader_only_when_embed_model_unpriced():
    records = [
        Record(
            _inst("a", "single-session-user", ["s1"]),
            "a",
            True,
            Usage(input_tokens=100, output_tokens=10, embed_tokens=1000),
        ),
    ]
    # Unknown embed model -> reader-only cost is reported, true cost is omitted.
    rep = aggregate(records, reader_model="gemini-2.5-flash-lite", embed_model="nonexistent-model")
    assert "usd_per_question_reader" in rep
    assert "usd_per_question" not in rep
    # mean_embed_tokens is still always reported regardless of pricing
    assert rep["mean_embed_tokens"] == 1000.0


def test_aggregate_empty():
    rep = aggregate([])
    assert rep["n"] == 0
    assert rep["accuracy"] == 0.0
    assert rep["recall_at_k"] is None
    assert rep["mean_embed_tokens"] == 0.0


def test_aggregate_cost_and_recall_over_measured_subset_only():
    """Resumed records (measured=False) are scored for accuracy but carry no live usage,
    so cost/latency/recall must be computed over the measured subset only — not diluted
    by the resumed zeros."""
    measured = Record(
        _inst("a", "single-session-user", ["s1"]),
        "a",
        True,
        Usage(input_tokens=100, output_tokens=10, latency_ms=50, embed_tokens=1000),
        retrieved_session_ids=["s1"],
    )
    # A correct resumed record with NO usage and an irrelevant gold-session miss.
    resumed = Record(
        _inst("b", "single-session-user", ["s2"]),
        "a-from-disk",
        True,
        measured=False,
    )
    rep = aggregate(
        [measured, resumed],
        reader_model="gemini-2.5-flash",
        embed_model="gemini-embedding-001",
    )
    # Accuracy/buckets span BOTH records.
    assert rep["n"] == 2
    assert rep["accuracy"] == 1.0
    assert rep["per_type"]["single-session-user"] == {"acc": 1.0, "n": 2}
    # Cost/latency/recall span the ONE measured record (and report the subset size).
    assert rep["measured_n"] == 1
    assert rep["mean_input_tokens"] == 100.0  # not (100+0)/2 == 50.0
    assert rep["latency_ms_p50"] == 50.0
    assert rep["recall_at_k"] == 1.0  # the resumed miss is excluded, not a 0.0
    # $/question uses the measured denominator (1), not the total (2).
    assert rep["usd_per_question_reader"] == round((100 * 0.30 + 10 * 2.50) / 1e6, 6)


def test_aggregate_no_measured_key_when_all_measured():
    """Backward compatible: ordinary runs (all measured) omit the measured_n key entirely."""
    rep = aggregate(
        [Record(_inst("a", "single-session-user", ["s1"]), "a", True, Usage(input_tokens=1))]
    )
    assert "measured_n" not in rep
