"""Offline, deterministic tests for the L5 anti-saturation MVP (FakeProvider + SQLite).

Write-time dedup + contradiction soft-update let a store absorb decades of duplicates and updates
without bloating, while keeping the LATEST value the one recall returns. Everything is additive and
OFF by default; the byte-identical guard below proves that with both flags off, memorize and recall
behave EXACTLY as pre-L5 and never consult the supersession table. No network, no live LLM.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from cortex.memory import CortexMemory
from cortex.providers.base import EmbedResult
from cortex.providers.fake import FakeProvider
from cortex.reader.reader import parse_supersession_verdict
from cortex.store.sqlite_store import Memory, SQLiteStore


def _v(*head: float) -> list[float]:
    """A 16-dim vector with ``head`` in the leading slots (matches FakeProvider's dim=16)."""
    vec = list(head) + [0.0] * (16 - len(head))
    return vec


class _KeyedProvider(FakeProvider):
    """FakeProvider whose ``embed`` returns caller-controlled vectors by substring.

    Lets a test pin exact cosines: e.g. ``_v(1.0)`` vs ``_v(1.0, 0.1)`` -> cosine 0.995 (a dedup),
    ``_v(1.0)`` vs ``_v(1.0, 0.5)`` -> cosine 0.894 (a soft-update candidate). Any text matching no
    key falls back to the deterministic hash embedding.
    """

    def __init__(self, table: dict[str, list[float]]) -> None:
        super().__init__(dim=16)
        self.table = table

    def embed(self, texts: Sequence[str]) -> EmbedResult:
        out: list[list[float]] = []
        for t in texts:
            vec: list[float] | None = None
            for key, v in self.table.items():
                if key in t:
                    vec = list(v)
                    break
            out.append(vec if vec is not None else self._embed_one(t))
        return EmbedResult(vectors=out, input_tokens=0)


def _arbiter(verdict_json: str) -> FakeProvider:
    """An arbiter provider that returns ``verdict_json`` on the arbiter prompt, else abstains."""

    def responder(prompt: str) -> str:
        if "contradiction arbiter" in prompt:
            return verdict_json
        return "I don't know."

    return FakeProvider(responder=responder)


# -- store: supersession roundtrip ---------------------------------------------------------


def test_supersession_add_and_superseded_ids_roundtrip(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    store.add(Memory(id="a" * 32, user_id="u1", content="old", created_at="t"), [1.0, 0.0])
    store.add(Memory(id="b" * 32, user_id="u1", content="new", created_at="t"), [1.0, 0.0])
    store.add(Memory(id="c" * 32, user_id="u2", content="theirs", created_at="t"), [1.0, 0.0])
    assert store.superseded_ids("u1") == set()  # nothing superseded yet

    store.add_supersession("a" * 32, "b" * 32, "2026-07-03T00:00:00+00:00")
    assert store.superseded_ids("u1") == {"a" * 32}
    assert store.superseded_ids("u2") == set()  # user-scoped via the JOIN

    # re-superseding the same old id repoints it (idempotent per superseded memory)
    store.add_supersession("a" * 32, "b" * 32, "2026-07-03T00:00:01+00:00")
    assert store.superseded_ids("u1") == {"a" * 32}


def test_supersession_cascades_on_delete(tmp_path):
    store = SQLiteStore(tmp_path / "m.db")
    store.add(Memory(id="a" * 32, user_id="u1", content="old", created_at="t"), [1.0, 0.0])
    store.add(Memory(id="b" * 32, user_id="u1", content="new", created_at="t"), [1.0, 0.0])
    store.add_supersession("a" * 32, "b" * 32, "t")
    store.delete("a" * 32, "u1")  # ON DELETE CASCADE cleans the companion row
    assert store.superseded_ids("u1") == set()


# -- write-time dedup (embedding-only; no LLM) ---------------------------------------------


def test_dedup_skips_near_identical_row(tmp_path):
    # espresso -> _v(1.0); latte -> _v(1.0, 0.1): cosine 0.995 >= 0.95 threshold -> a duplicate.
    provider = _KeyedProvider({"espresso": _v(1.0), "latte": _v(1.0, 0.1)})
    eng = CortexMemory(provider, SQLiteStore(tmp_path / "m.db"), use_dedup=True)
    first = eng.memorize("my favorite coffee is espresso")
    second = eng.memorize("my favorite coffee is a latte")  # near-identical -> deduped
    assert eng.count() == 1  # NO second row: growth is bounded
    assert second.id == first.id  # dedup returns the existing memory


def test_dedup_off_inserts_every_row(tmp_path):
    provider = _KeyedProvider({"espresso": _v(1.0), "latte": _v(1.0, 0.1)})
    eng = CortexMemory(provider, SQLiteStore(tmp_path / "m.db"))  # dedup default OFF
    eng.memorize("my favorite coffee is espresso")
    eng.memorize("my favorite coffee is a latte")
    assert eng.count() == 2  # both rows kept when dedup is off


# -- contradiction soft-update (LLM arbiter) -----------------------------------------------


def _soft_update_engine(tmp_path, verdict_json: str) -> CortexMemory:
    # Paris -> _v(1.0); Berlin -> _v(1.0, 0.5): cosine 0.894 in [0.83, 0.95) -> arbiter candidate.
    # A "city" query -> _v(1.0, 0.3): retrieves both so the filter can drop the superseded one.
    provider = _KeyedProvider(
        {"Paris": _v(1.0), "Berlin": _v(1.0, 0.5), "which city": _v(1.0, 0.3)}
    )
    return CortexMemory(
        provider,
        SQLiteStore(tmp_path / "m.db"),
        use_soft_update=True,
        arbiter=_arbiter(verdict_json),
    )


def test_soft_update_supersedes_old_and_recall_returns_latest(tmp_path):
    eng = _soft_update_engine(tmp_path, '{"verdict": "UPDATE", "supersedes_id": null}')
    old = eng.memorize("home city is Paris")
    new = eng.memorize("home city is Berlin")  # arbiter: UPDATE -> supersedes Paris
    assert eng.count() == 2  # both rows exist on disk...
    assert eng.store.superseded_ids("local") == {old.id}  # ...but Paris is marked superseded

    hits = eng.recall("which city does the user live in")
    ids = [m.id for m in hits]
    assert new.id in ids  # latest value returned
    assert old.id not in ids  # superseded value filtered out
    assert any("Berlin" in m.content for m in hits)


def test_soft_update_noop_skips_the_write(tmp_path):
    eng = _soft_update_engine(tmp_path, '{"verdict": "NOOP", "supersedes_id": null}')
    first = eng.memorize("home city is Paris")
    second = eng.memorize("home city is Berlin")  # arbiter: NOOP -> redundant, skip
    assert eng.count() == 1  # no new row
    assert second.id == first.id  # returns the existing memory
    assert eng.store.superseded_ids("local") == set()


def test_soft_update_add_verdict_inserts_both(tmp_path):
    eng = _soft_update_engine(tmp_path, '{"verdict": "ADD", "supersedes_id": null}')
    eng.memorize("home city is Paris")
    eng.memorize("home city is Berlin")  # arbiter: ADD -> keep both, no supersession
    assert eng.count() == 2
    assert eng.store.superseded_ids("local") == set()


def test_arbiter_garbage_response_falls_back_to_add(tmp_path):
    eng = _soft_update_engine(tmp_path, "this is not json at all")
    eng.memorize("home city is Paris")
    eng.memorize("home city is Berlin")  # unparseable verdict -> plain ADD
    assert eng.count() == 2  # row inserted despite garbage
    assert eng.store.superseded_ids("local") == set()


def test_arbiter_that_raises_never_breaks_memorize(tmp_path):
    class _BoomArbiter(FakeProvider):
        def generate(self, prompt, *, temperature=0.0, max_output_tokens=512):
            raise RuntimeError("arbiter exploded")

    provider = _KeyedProvider({"Paris": _v(1.0), "Berlin": _v(1.0, 0.5)})
    eng = CortexMemory(
        provider,
        SQLiteStore(tmp_path / "m.db"),
        use_soft_update=True,
        arbiter=_BoomArbiter(),
    )
    eng.memorize("home city is Paris")
    eng.memorize("home city is Berlin")  # arbiter blows up -> falls back to ADD
    assert eng.count() == 2
    assert eng.store.superseded_ids("local") == set()


# -- parser --------------------------------------------------------------------------------


def test_parse_supersession_verdict_valid():
    out = parse_supersession_verdict('{"verdict": "UPDATE", "supersedes_id": "abc123"}')
    assert out == {"verdict": "UPDATE", "supersedes_id": "abc123"}


def test_parse_supersession_verdict_markdown_and_case():
    out = parse_supersession_verdict('```json\n{"verdict": "update", "supersedes_id": null}\n```')
    assert out["verdict"] == "UPDATE"
    assert out["supersedes_id"] is None


@pytest.mark.parametrize("text", ["", "   ", "not json", '{"verdict": "MAYBE"}', "{}"])
def test_parse_supersession_verdict_garbage_defaults_to_add(text):
    assert parse_supersession_verdict(text) == {"verdict": "ADD", "supersedes_id": None}


# -- byte-identical-when-off guard (the key guarantee) -------------------------------------


def test_byte_identical_when_off_never_queries_supersessions(tmp_path):
    """Defaults (both L5 flags off): every row inserted, all matches recalled, no supersession read.

    This is the load-bearing guard: with anti-saturation off, recall must never query the store's
    supersession set, and duplicate writes must all land — exactly the pre-L5 behaviour.
    """
    store = SQLiteStore(tmp_path / "m.db")

    # Spy on the supersession read: recall must NEVER call it when anti-saturation is off.
    calls = {"superseded_ids": 0}
    original = store.superseded_ids

    def spy(user_id: str) -> set[str]:
        calls["superseded_ids"] += 1
        return original(user_id)

    store.superseded_ids = spy  # type: ignore[method-assign]

    eng = CortexMemory(FakeProvider(), store)  # all L5 flags default False
    assert eng.use_dedup is False and eng.use_soft_update is False
    assert eng.arbiter is None

    # Exact duplicates are ALL inserted — dedup never runs on the write path.
    eng.memorize("the user drinks coffee")
    eng.memorize("the user drinks coffee")
    eng.memorize("the user works as an engineer")
    assert eng.count() == 3

    hits = eng.recall("coffee", limit=5)
    assert len([m for m in hits if "coffee" in m.content]) == 2  # both duplicates returned

    assert calls["superseded_ids"] == 0  # recall never consulted supersessions when off


def test_off_path_recall_matches_a_plain_engine(tmp_path):
    """Recall output with L5 available-but-off equals a plain pre-L5 engine's, byte for byte."""
    facts = [
        "the user's dog is named Rex",
        "the user enjoys hiking on weekends",
        "the user works as a backend engineer",
    ]
    plain = CortexMemory(FakeProvider(), SQLiteStore(tmp_path / "plain.db"))
    offed = CortexMemory(FakeProvider(), SQLiteStore(tmp_path / "offed.db"))
    for f in facts:
        plain.memorize(f)
        offed.memorize(f)
    q = "what is the name of the user's dog"
    assert [m.content for m in offed.recall(q)] == [m.content for m in plain.recall(q)]
