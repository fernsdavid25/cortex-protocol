"""Offline, deterministic tests for the in-memory store (dense + BM25 lexical search)."""

from cortex.providers.fake import FakeProvider
from cortex.store.memory_store import InMemoryStore, MemoryChunk


def _chunk(text: str, sid: str, vec: list[float]) -> MemoryChunk:
    return MemoryChunk(text=text, session_id=sid, date="2026-01-01", embedding=vec)


def test_reset_clears_store():
    store = InMemoryStore()
    store.add([_chunk("hello world", "s1", [1.0, 0.0])])
    assert len(store) == 1
    store.reset()
    assert len(store) == 0
    assert store.dense_search([1.0, 0.0], 5) == []
    assert store.lexical_search("hello", 5) == []


def test_dense_search_ranks_by_cosine():
    store = InMemoryStore()
    store.add(
        [
            _chunk("a", "s1", [1.0, 0.0, 0.0]),
            _chunk("b", "s2", [0.0, 1.0, 0.0]),
            _chunk("c", "s3", [0.7, 0.7, 0.0]),
        ]
    )
    ranked = store.dense_search([1.0, 0.0, 0.0], 3)
    assert [c.session_id for c in ranked] == ["s1", "s3", "s2"]


def test_lexical_search_surfaces_exact_keyword_dense_ranks_lower():
    """BM25 surfaces an exact-keyword chunk that the fake bag-of-words dense ranks lower.

    The query token "rex" collides into the fake embedder's small hash space with the
    distractor's vocabulary, so dense ranking is muddy; BM25 keys on the literal token
    and puts the exact-keyword chunk first.
    """
    provider = FakeProvider()
    answer_text = "user: The name of my dog is Rex.\nassistant: Rex is a great name."
    noise_text = "user: I enjoy hiking mountains and climbing on the weekend."
    embeds = provider.embed([answer_text, noise_text])

    store = InMemoryStore()
    store.add(
        [
            _chunk(answer_text, "s_answer", embeds.vectors[0]),
            _chunk(noise_text, "s_noise", embeds.vectors[1]),
        ]
    )

    lexical = store.lexical_search("What is the name of the user's dog Rex?", 2)
    assert lexical[0].session_id == "s_answer"
    # Sanity: the noise chunk shares no query-specific keyword, so it is filtered out.
    assert all(c.text != noise_text for c in store.lexical_search("Rex", 2))


def test_memory_chunk_turn_ids_defaults_empty_and_does_not_affect_ranking():
    """The bench's turn provenance is an optional, immutable-defaulted field: every existing
    construction site keeps working, and it participates in NO scoring — a store built with
    turn_ids ranks identically to one built without."""
    assert MemoryChunk(text="t", session_id="s", date="d").turn_ids == ()
    assert MemoryChunk(text="t", session_id="s", date="d", turn_ids=("s#0",)).turn_ids == ("s#0",)

    texts = [("the cat sat on the mat", "s1"), ("a unique pangolin appeared", "s2")]
    vecs = [[1.0, 0.0], [0.0, 1.0]]
    plain = InMemoryStore()
    plain.add([_chunk(t, s, v) for (t, s), v in zip(texts, vecs, strict=True)])
    tagged = InMemoryStore()
    tagged.add(
        [
            MemoryChunk(text=t, session_id=s, date="2026-01-01", embedding=v, turn_ids=(f"{s}#0",))
            for (t, s), v in zip(texts, vecs, strict=True)
        ]
    )
    assert [c.session_id for c in plain.dense_search([1.0, 0.0], 2)] == [
        c.session_id for c in tagged.dense_search([1.0, 0.0], 2)
    ]
    assert [c.session_id for c in plain.lexical_search("pangolin cat", 2)] == [
        c.session_id for c in tagged.lexical_search("pangolin cat", 2)
    ]


def test_lexical_search_bm25_prefers_rarer_term():
    store = InMemoryStore()
    store.add(
        [
            _chunk("the cat sat on the mat", "s1", [1.0]),
            _chunk("the dog the dog the dog", "s2", [1.0]),
            _chunk("a unique pangolin appeared", "s3", [1.0]),
        ]
    )
    ranked = store.lexical_search("pangolin", 3)
    assert ranked[0].session_id == "s3"
