"""Offline, deterministic tests for RRF fusion and hybrid retrieval."""

from cortex.providers.fake import FakeProvider
from cortex.retrieve.hybrid import hybrid_retrieve, reciprocal_rank_fusion
from cortex.store.memory_store import InMemoryStore, MemoryChunk


def test_rrf_rewards_agreement_across_lists():
    """A chunk ranked high by BOTH lists beats one ranked high by only one list."""
    dense = ["A", "B", "C"]  # A is dense #1
    lexical = ["B", "A", "D"]  # A is lexical #2, B is lexical #1
    fused = reciprocal_rank_fusion([dense, lexical])
    ranked = sorted(fused, key=lambda x: (-fused[x], x))
    # B is #1 lexical and #2 dense -> agreed-upon top; beats A (#1 dense, #2 lexical) by tie,
    # and both beat C and D which appear in only one list.
    assert ranked[0] in {"A", "B"}
    assert fused["A"] > fused["C"]
    assert fused["B"] > fused["D"]
    assert fused["A"] == fused["B"]  # symmetric ranks -> equal fused score


def test_rrf_consensus_beats_single_list_leader():
    # X is mediocre in both lists; Y tops only one list. Consensus should win.
    list1 = ["Y", "X"]
    list2 = ["Z", "X"]
    list3 = ["W", "X"]
    fused = reciprocal_rank_fusion([list1, list2, list3])
    ranked = sorted(fused, key=lambda x: (-fused[x], x))
    assert ranked[0] == "X"  # appears in all three (rank 2 each) -> highest fused score


def test_hybrid_retrieve_fuses_dense_and_lexical():
    provider = FakeProvider()
    a = "user: My dog is named Rex.\nassistant: Rex is a fine dog."
    b = "user: I went hiking in the mountains.\nassistant: Hiking is great."
    c = "user: I cooked pasta for dinner.\nassistant: Pasta sounds delicious."
    embeds = provider.embed([a, b, c])

    store = InMemoryStore()
    store.add(
        [
            MemoryChunk(text=a, session_id="s_a", date="2026-01-01", embedding=embeds.vectors[0]),
            MemoryChunk(text=b, session_id="s_b", date="2026-01-02", embedding=embeds.vectors[1]),
            MemoryChunk(text=c, session_id="s_c", date="2026-01-03", embedding=embeds.vectors[2]),
        ]
    )

    qvec = provider.embed(["What is the name of my dog Rex?"]).vectors[0]
    out = hybrid_retrieve(store, "What is the name of my dog Rex?", qvec, top_k=2)
    assert out
    assert out[0].session_id == "s_a"


def test_hybrid_retrieve_empty_store():
    store = InMemoryStore()
    assert hybrid_retrieve(store, "anything", [1.0, 0.0], top_k=5) == []
