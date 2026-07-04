"""Property-based tests (Hypothesis) for the load-bearing retrieval/store invariants.

Offline and deterministic: every strategy is seeded via ``@settings(derandomize=True)`` and the
only "engine" used is [`FakeProvider`][cortex.providers.fake] with hand-crafted embeddings — no
network, no live LLM, in the spirit of the rest of the suite. Four invariants are exercised:

1. **RRF** (``reciprocal_rank_fusion``): scores equal ``Σ 1/(k+rank)`` per id, an id present in
   more lists never scores below one present in fewer at the same rank, and fusion is a stable,
   deterministic function of its input.
2. **``resolve_id_prefix``**: over random id sets + a random prefix, resolution returns exactly the
   matching ids (SQL-side ``ORDER BY id LIMIT``) and NEVER an id that fails to start with the
   prefix — so a short prefix can only ever be flagged ambiguous, never delete the wrong memory.
3. **blob round-trip**: ``_from_blob(_to_blob(v))`` preserves float32 values and dimension.
4. **L5 interaction**: over random sequences of ``memorize`` with dedup + soft-update on, recall
   must always surface the LATEST asserted value for a key — the regression guard for the
   *dedup-into-superseded* bug (re-asserting a previously-superseded value must not vanish).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from cortex.memory import CortexMemory
from cortex.providers.base import EmbedResult
from cortex.providers.fake import FakeProvider
from cortex.retrieve.hybrid import RRF_K, reciprocal_rank_fusion
from cortex.store.sqlite_store import Memory, SQLiteStore, _from_blob, _to_blob

# Hex alphabet real memory ids are drawn from (uuid4().hex). Lowercase-only so SQLite's
# case-insensitive ASCII LIKE agrees byte-for-byte with Python's case-sensitive ``str.startswith``.
_HEX = "0123456789abcdef"


# ----------------------------------------------------------------------------------------------
# 1. Reciprocal Rank Fusion invariants
# ----------------------------------------------------------------------------------------------


@settings(deadline=None, derandomize=True, max_examples=200)
@given(
    rankings=st.lists(st.lists(st.integers(0, 12), max_size=8), max_size=6),
    k=st.integers(min_value=1, max_value=100),
)
def test_rrf_scores_match_reciprocal_rank_formula(rankings: list[list[int]], k: int) -> None:
    """Each fused score equals ``Σ 1/(k+rank)`` over every (list, 1-based rank) the id appears at.

    Computed independently from the impl: collect every rank position per id, then sum the
    reciprocals — so this validates the formula, not just mirrors the code.
    """
    positions: dict[int, list[int]] = defaultdict(list)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            positions[item].append(rank)
    expected = {item: sum(1.0 / (k + r) for r in ranks) for item, ranks in positions.items()}

    result = reciprocal_rank_fusion(rankings, k=k)

    assert result == pytest.approx(expected, rel=1e-12, abs=1e-15)


@settings(deadline=None, derandomize=True, max_examples=200)
@given(
    rank=st.integers(min_value=1, max_value=6),
    more=st.integers(min_value=1, max_value=6),
    fewer=st.integers(min_value=1, max_value=6),
)
def test_rrf_more_lists_never_scores_below_fewer_at_same_rank(
    rank: int, more: int, fewer: int
) -> None:
    """An id voted for by MORE lists at a given rank never scores below one voted by FEWER.

    Build ``more`` lists that place id ``X`` at position ``rank`` and ``fewer`` lists that place
    ``Y`` there (distinct fillers pad the earlier positions, contributing only to themselves). With
    ``more >= fewer`` the RRF "voting" property demands ``score(X) >= score(Y)``.
    """
    if more < fewer:
        more, fewer = fewer, more

    def place(item: object, list_idx: int) -> list[object]:
        # Unique fillers so the earlier positions never add to X's or Y's score.
        return [("filler", list_idx, j) for j in range(rank - 1)] + [item]

    rankings: list[list[object]] = [place("X", i) for i in range(more)]
    rankings += [place("Y", more + i) for i in range(fewer)]

    fused = reciprocal_rank_fusion(rankings)

    assert fused["X"] >= fused["Y"]
    assert fused["X"] == pytest.approx(more / (RRF_K + rank))
    assert fused["Y"] == pytest.approx(fewer / (RRF_K + rank))


@settings(deadline=None, derandomize=True, max_examples=150)
@given(rankings=st.lists(st.lists(st.integers(0, 12), max_size=8), max_size=6))
def test_rrf_is_stable_and_deterministic(rankings: list[list[int]]) -> None:
    """Fusion is a pure function: identical input yields identical scores, key order, and ranking.

    The derived ``sorted(fused, key=(-score, id))`` ordering hybrid_retrieve relies on must be
    reproducible across calls (no set/dict-iteration nondeterminism leaking into the result).
    """
    first = reciprocal_rank_fusion(rankings)
    second = reciprocal_rank_fusion(rankings)

    assert first == second
    assert list(first.keys()) == list(second.keys())  # stable insertion order

    order_first = sorted(first, key=lambda i: (-first[i], i))
    order_second = sorted(second, key=lambda i: (-second[i], i))
    assert order_first == order_second


# ----------------------------------------------------------------------------------------------
# 2. resolve_id_prefix — never resolves to the wrong id
# ----------------------------------------------------------------------------------------------


@settings(deadline=None, derandomize=True, max_examples=200)
@given(
    ids=st.lists(
        st.text(alphabet=_HEX, min_size=1, max_size=8), min_size=0, max_size=12, unique=True
    ),
    prefix=st.text(alphabet=_HEX, min_size=0, max_size=6),
)
def test_resolve_id_prefix_returns_only_true_matches(ids: list[str], prefix: str) -> None:
    """Resolution returns exactly the (id-ordered, limit-capped) set of ids starting with prefix.

    - Every returned id genuinely starts with ``prefix`` — a short prefix can only ever be flagged
      AMBIGUOUS (>1 result) and never resolves to an id it does not prefix (no wrong-id deletion).
    - Exactly one match ⇒ the unique full id is returned; zero ⇒ ``[]``.
    """
    store = SQLiteStore(":memory:")
    try:
        for i in ids:
            store.add(Memory(id=i, user_id="u", content="x", created_at="t"), [1.0])

        result = store.resolve_id_prefix("u", prefix, limit=2)
        matching = sorted(i for i in ids if i.startswith(prefix))

        assert result == matching[:2]  # SQL: WHERE id LIKE prefix||'%' ORDER BY id LIMIT 2
        assert all(r.startswith(prefix) for r in result)  # never a wrong id
        if len(matching) == 1:
            assert result == matching  # unambiguous ⇒ the unique full id
        if len(matching) == 0:
            assert result == []
    finally:
        store.close()


# ----------------------------------------------------------------------------------------------
# 3. float32 blob round-trip
# ----------------------------------------------------------------------------------------------


@settings(deadline=None, derandomize=True, max_examples=200)
@given(
    vec=st.lists(
        st.floats(allow_nan=False, allow_infinity=False, width=32),
        min_size=0,
        max_size=64,
    )
)
def test_blob_round_trip_preserves_values_and_dimension(vec: list[float]) -> None:
    """encode→decode preserves dimension and every value (float32-exact for float32 inputs).

    ``width=32`` draws values already representable in float32, so the round-trip is bit-exact and
    equality is strict — proving ``_to_blob``/``_from_blob`` neither truncate nor reorder.
    """
    out = _from_blob(_to_blob(vec))
    assert len(out) == len(vec)
    assert out == vec


# ----------------------------------------------------------------------------------------------
# 4. L5 anti-saturation interaction (dedup + soft-update) — the dedup-into-superseded bug
# ----------------------------------------------------------------------------------------------

# Two keys along orthogonal base axes; a small per-value perturbation along a separate axis. This
# pins the cosines the L5 bands care about: same key + same value → 1.0 (a dedup); same key +
# different value → 1/(1+P²) ≈ 0.891 (inside [0.83, 0.95): a soft-update candidate); different key
# → ≤ 0.11 (unrelated). A query embeds the key axis alone, retrieving that key's memories.
_KEY_AXIS = {"alpha": 0, "beta": 1}
_VAL_AXIS = {"one": 2, "two": 3, "three": 4}
_PERTURB = 0.35


def _content(key: str, val: str) -> str:
    return f"the {key} attribute is {val}"


def _query(key: str) -> str:
    return f"the {key} attribute"


class _SemanticProvider(FakeProvider):
    """Embeds by the (key, value) tokens in the text so L5's cosine bands are exactly reproducible.

    A text with a value token → key-axis 1.0 plus a value-axis perturbation (a stored memory); a
    text with only a key token → the key axis alone (a recall query). Anything else falls back to
    the deterministic hash embedding.
    """

    def embed(self, texts: Sequence[str]) -> EmbedResult:
        out: list[list[float]] = []
        for text in texts:
            low = text.lower()
            key = next((k for k in _KEY_AXIS if k in low), None)
            if key is None:
                out.append(self._embed_one(text))
                continue
            vec = [0.0] * 16
            vec[_KEY_AXIS[key]] = 1.0
            val = next((v for v in _VAL_AXIS if v in low), None)
            if val is not None:
                vec[_VAL_AXIS[val]] = _PERTURB
            out.append(vec)
        return EmbedResult(vectors=out, input_tokens=0)


def _update_arbiter() -> FakeProvider:
    """Arbiter that rules UPDATE — correct here since every band candidate is a same-key rewrite."""

    def responder(prompt: str) -> str:
        if "contradiction arbiter" in prompt:
            return '{"verdict": "UPDATE", "supersedes_id": null}'
        return "I don't know."

    return FakeProvider(responder=responder)


@settings(deadline=None, derandomize=True, max_examples=100)
@given(
    assertions=st.lists(
        st.tuples(st.sampled_from(sorted(_KEY_AXIS)), st.sampled_from(sorted(_VAL_AXIS))),
        min_size=1,
        max_size=6,
    )
)
@example(assertions=[("alpha", "one"), ("alpha", "two"), ("alpha", "one")])
def test_l5_recall_always_surfaces_latest_asserted_value(
    assertions: list[tuple[str, str]],
) -> None:
    """With dedup + soft-update on, recall must return the LATEST value asserted for each key.

    Property: after replaying a sequence of ``memorize`` calls, recalling a key surfaces the value
    from that key's most recent assertion, and never silently drops a currently-true value. The
    ``@example`` (assert one → update to two → re-assert one) deterministically exercises the
    dedup-into-superseded path — the exact regression this guards against.
    """
    eng = CortexMemory(
        _SemanticProvider(),
        SQLiteStore(":memory:"),
        use_dedup=True,
        use_soft_update=True,
        arbiter=_update_arbiter(),
    )
    try:
        latest: dict[str, str] = {}
        for key, val in assertions:
            eng.memorize(_content(key, val))
            latest[key] = val

        for key, val in latest.items():
            contents = [m.content for m in eng.recall(_query(key), limit=10)]
            assert _content(key, val) in contents, (
                f"latest value for {key!r} is {val!r}; recall surfaced {contents!r}"
            )
    finally:
        eng.close()
