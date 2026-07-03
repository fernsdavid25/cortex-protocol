"""Deterministic representative sampling (run.select_instances).

The oracle JSON is grouped by question type, so a raw head slice is unrepresentative. We shuffle
(seeded) BEFORE --limit. These tests pin: reproducibility (same seed -> same order), that shuffle
actually reorders vs the sorted input, and that --limit is applied AFTER the shuffle.
"""

from cortex_bench.memory_system import QAInstance
from cortex_bench.run import select_instances


def _make_instances(n: int) -> list[QAInstance]:
    # Mimic the grouped-by-type oracle layout: a temporal block, then a multi-session block.
    out: list[QAInstance] = []
    for i in range(n):
        qtype = "temporal-reasoning" if i < n // 2 else "multi-session"
        out.append(QAInstance(question_id=f"q{i}", question_type=qtype, question="?", answer="a"))
    return out


def test_same_seed_same_order() -> None:
    insts = _make_instances(50)
    a = [x.question_id for x in select_instances(insts, seed=0)]
    b = [x.question_id for x in select_instances(insts, seed=0)]
    assert a == b


def test_different_seed_changes_order() -> None:
    insts = _make_instances(50)
    a = [x.question_id for x in select_instances(insts, seed=0)]
    b = [x.question_id for x in select_instances(insts, seed=1)]
    assert a != b


def test_shuffle_changes_order_vs_sorted() -> None:
    insts = _make_instances(50)
    original = [x.question_id for x in insts]
    shuffled = [x.question_id for x in select_instances(insts, seed=0)]
    assert shuffled != original
    assert sorted(shuffled) == sorted(original)  # same set, just reordered


def test_no_shuffle_preserves_order() -> None:
    insts = _make_instances(10)
    assert [x.question_id for x in select_instances(insts, shuffle=False)] == [
        x.question_id for x in insts
    ]


def test_limit_applied_after_shuffle_is_representative() -> None:
    # With grouping, a no-shuffle head slice is single-type; a shuffled slice spans both types.
    insts = _make_instances(100)
    head = select_instances(insts, limit=20, shuffle=False)
    assert {x.question_type for x in head} == {"temporal-reasoning"}  # unrepresentative

    shuffled = select_instances(insts, limit=20, seed=0)
    assert len(shuffled) == 20
    assert {x.question_type for x in shuffled} == {"temporal-reasoning", "multi-session"}


def test_does_not_mutate_input() -> None:
    insts = _make_instances(20)
    before = [x.question_id for x in insts]
    select_instances(insts, seed=3)
    assert [x.question_id for x in insts] == before
