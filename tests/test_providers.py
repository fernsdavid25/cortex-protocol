"""Offline, deterministic tests for FakeProvider (no network, no SDK)."""

from cortex.providers.fake import FakeProvider
from cortex_bench.systems._common import cosine


def test_generate_records_prompt_and_returns_responder_output():
    provider = FakeProvider(responder=lambda p: "the answer is here")
    r = provider.generate("what is the answer please")
    assert provider.last_prompt == "what is the answer please"
    assert r.text == "the answer is here"
    assert r.input_tokens == 5  # words in the prompt
    assert r.output_tokens == 4  # words in the response


def test_generate_default_responder_abstains():
    provider = FakeProvider()
    r = provider.generate("anything at all")
    assert r.text == "I don't know."


def test_generate_empty_responder_falls_back_to_default():
    provider = FakeProvider(responder=lambda p: "")
    r = provider.generate("hello world")
    assert r.text == "I don't know."


def test_embed_identical_text_cosine_is_one():
    provider = FakeProvider()
    res = provider.embed(["I live in Goa", "I live in Goa"])
    assert cosine(res.vectors[0], res.vectors[1]) == 1.0


def test_embed_different_text_cosine_below_one():
    provider = FakeProvider()
    res = provider.embed(["I live in Goa", "the weather is cold today"])
    assert cosine(res.vectors[0], res.vectors[1]) < 1.0


def test_embed_is_deterministic_across_instances():
    a = FakeProvider().embed(["hello world"]).vectors[0]
    b = FakeProvider().embed(["hello world"]).vectors[0]
    assert a == b
