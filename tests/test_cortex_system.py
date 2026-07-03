"""Offline, deterministic end-to-end tests for CortexSystem (engine v0)."""

from cortex.providers.fake import FakeProvider
from cortex.reader.reader import (
    ABSTAIN_SENTINEL,
    build_answerability_gate_prompt,
    build_reader_prompt,
    build_recommendation_prompt,
    build_reflection_prompt,
    build_rerank_prompt,
    gate_says_unanswerable,
    is_recommendation_question,
    parse_rerank_order,
)
from cortex.store.memory_store import MemoryChunk
from cortex_bench.judge import Judge
from cortex_bench.memory_system import QAInstance
from cortex_bench.systems.cortex_system import CortexSystem


def _instance() -> QAInstance:
    return QAInstance(
        question_id="q_dog",
        question_type="single-session-user",
        question="What is the name of the user's dog?",
        answer="Rex",
        question_date="2026-06-27",
        answer_session_ids=["s_answer"],
        haystack_session_ids=["s_noise", "s_answer"],
        haystack_dates=["2026-01-01", "2026-02-01"],
        haystack_sessions=[
            [
                {"role": "user", "content": "I enjoy hiking mountains on weekends."},
                {"role": "assistant", "content": "Hiking is wonderful exercise."},
            ],
            [
                {"role": "user", "content": "The name of my dog is Rex."},
                {"role": "assistant", "content": "Rex is a great name for a dog."},
            ],
        ],
    )


def _gold_responder(gold: str):
    """Return the gold token when it appears in the retrieved memories, else abstain."""

    def respond(prompt: str) -> str:
        if gold.lower() in prompt.lower():
            return f"NOTES: the user's dog is named {gold}.\nANSWER: {gold}"
        return ABSTAIN_SENTINEL

    return respond


def test_cortex_end_to_end_retrieves_answer_session_and_grades_correct():
    provider = FakeProvider(responder=_gold_responder("Rex"))
    system = CortexSystem(provider, top_k=2)
    inst = _instance()

    system.reset()
    ingest_usage = system.ingest(inst)
    hyp, answer_usage = system.answer(inst)

    # The answer session is retrieved first.
    assert system.retrieved_session_ids()[0] == "s_answer"
    # The offline judge marks it correct (gold token present).
    assert Judge("offline").grade(inst, hyp) is True
    # Usage is accounted with embed vs reader split: ingest only embeds rounds
    # (embed_tokens, no reader tokens); answer embeds the question (embed_tokens)
    # and runs the reader (input/output tokens).
    assert ingest_usage.embed_tokens > 0
    assert ingest_usage.input_tokens == 0
    assert answer_usage.embed_tokens > 0
    assert answer_usage.input_tokens > 0
    assert answer_usage.output_tokens > 0


def test_cortex_reset_clears_store():
    provider = FakeProvider(responder=_gold_responder("Rex"))
    system = CortexSystem(provider, top_k=2)
    inst = _instance()
    system.ingest(inst)
    assert len(system.store) > 0
    system.reset()
    assert len(system.store) == 0
    assert system.retrieved_session_ids() == []


def test_cortex_empty_haystack_returns_zero_usage():
    provider = FakeProvider(responder=_gold_responder("Rex"))
    system = CortexSystem(provider)
    inst = QAInstance(
        question_id="q_empty",
        question_type="single-session-user",
        question="anything?",
        answer="x",
    )
    system.reset()
    usage = system.ingest(inst)
    assert usage.input_tokens == 0
    assert usage.embed_tokens == 0


def _fact_key_instance() -> QAInstance:
    """A haystack where the answer fact is phrased very differently from the question.

    The raw round says "I adopted a golden retriever and called him Biscuit"; the
    question asks for "the user's pet". Fact extraction should surface an atomic key
    like "user's pet is named Biscuit" that lexically/semantically matches the query.
    """
    return QAInstance(
        question_id="q_petname",
        question_type="single-session-user",
        question="What is the name of the user's pet?",
        answer="Biscuit",
        question_date="2026-06-27",
        answer_session_ids=["s_answer"],
        haystack_session_ids=["s_noise", "s_answer"],
        haystack_dates=["2026-01-01", "2026-02-01"],
        haystack_sessions=[
            [
                {"role": "user", "content": "I enjoy hiking mountains on weekends."},
                {"role": "assistant", "content": "Hiking is wonderful exercise."},
            ],
            [
                {"role": "user", "content": "I adopted a golden retriever and called him Biscuit."},
                {"role": "assistant", "content": "What a sweet companion."},
            ],
        ],
    )


def _fact_extracting_responder(prompt: str) -> str:
    """Return a numbered fact list for extraction prompts, gold answer for reader prompts."""
    if "Extract the key atomic facts" in prompt:
        if "Biscuit" in prompt:
            return "1. The user's pet is named Biscuit\n2. The user adopted a golden retriever"
        return "1. The user enjoys hiking mountains on weekends"
    if "biscuit" in prompt.lower():
        return "NOTES: the user's pet is Biscuit.\nANSWER: Biscuit"
    return ABSTAIN_SENTINEL


def test_fact_keys_off_by_default_adds_no_extra_chunks():
    provider = FakeProvider(responder=_fact_extracting_responder)
    system = CortexSystem(provider, top_k=4)  # default use_fact_keys=False
    inst = _fact_key_instance()
    system.reset()
    usage = system.ingest(inst)
    # Only the raw rounds are stored (2 sessions x 1 round each = 2 chunks).
    assert len(system.store) == 2
    # No reader/generate tokens are spent during ingest when fact keys are off.
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0


def test_fact_keys_add_fact_chunks_and_count_embed_tokens():
    provider = FakeProvider(responder=_fact_extracting_responder)
    system = CortexSystem(provider, top_k=4, use_fact_keys=True)
    inst = _fact_key_instance()
    system.reset()
    usage = system.ingest(inst)

    # Raw rounds (2) PLUS extracted fact chunks (1 + 2 = 3) are stored.
    assert len(system.store) == 5
    fact_texts = [c.text for c in system.store.chunks]
    assert "The user's pet is named Biscuit" in fact_texts
    # Fact extraction spends reader tokens (one generate per session) and the fact
    # strings are embedded (counted into embed_tokens, NOT input_tokens).
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0
    assert usage.embed_tokens > 0
    # Every fact chunk carries its source session id.
    fact_chunk = next(c for c in system.store.chunks if c.text == "The user's pet is named Biscuit")
    assert fact_chunk.session_id == "s_answer"


def test_fact_keys_help_retrieve_session_for_paraphrased_query():
    # With fact keys ON, an atomic "pet is named Biscuit" key matches the "pet" query.
    provider = FakeProvider(responder=_fact_extracting_responder)
    system = CortexSystem(provider, top_k=4, use_fact_keys=True)
    inst = _fact_key_instance()
    system.reset()
    system.ingest(inst)
    hyp, _usage = system.answer(inst)

    assert system.retrieved_session_ids()[0] == "s_answer"
    assert Judge("offline").grade(inst, hyp) is True


def _distill_responder(prompt: str) -> str:
    """Return distilled facts for the distill prompt, gold answer for the reader prompt."""
    if "Relevant facts:" in prompt:  # the QUERY_DISTILL_PROMPT
        return "- The user's dog is named Rex." if "Rex" in prompt else "NONE"
    if "rex" in prompt.lower():
        return "NOTES: the user's dog is Rex.\nANSWER: Rex"
    return ABSTAIN_SENTINEL


def test_query_distill_off_by_default_makes_one_generate():
    """Default (no distill): answer() makes exactly one generate (the reader)."""
    calls: list[str] = []

    def responder(prompt: str) -> str:
        calls.append(prompt)
        return _distill_responder(prompt)

    system = CortexSystem(FakeProvider(responder=responder), top_k=2)  # distill off
    inst = _instance()
    system.reset()
    system.ingest(inst)
    system.answer(inst)
    assert not any("Relevant facts:" in c for c in calls)  # no distill call
    assert sum("NOTES" in c for c in calls) == 1  # one reader call


def test_query_distill_adds_distill_call_and_focus_chunk():
    calls: list[str] = []

    def responder(prompt: str) -> str:
        calls.append(prompt)
        return _distill_responder(prompt)

    system = CortexSystem(FakeProvider(responder=responder), top_k=2, use_query_distill=True)
    inst = _instance()
    system.reset()
    system.ingest(inst)
    hyp, usage = system.answer(inst)

    # Exactly one distill generate AND one reader generate happened.
    assert sum("Relevant facts:" in c for c in calls) == 1
    reader_prompt = next(c for c in calls if "NOTES:" in c and "Relevant facts:" not in c)
    # The distilled facts were prepended into the reader context.
    assert "The user's dog is named Rex." in reader_prompt
    # Both generate calls are accounted in usage; the answer is graded correct.
    assert usage.input_tokens > 0 and usage.output_tokens > 0
    assert Judge("offline").grade(inst, hyp) is True


def test_query_distill_none_adds_no_focus_chunk():
    """When distillation returns NONE, no synthetic chunk is added to the reader context."""
    seen: list[str] = []

    def responder(prompt: str) -> str:
        seen.append(prompt)
        if "Relevant facts:" in prompt:
            return "NONE"
        return ABSTAIN_SENTINEL

    system = CortexSystem(FakeProvider(responder=responder), top_k=2, use_query_distill=True)
    inst = _instance()
    system.reset()
    system.ingest(inst)
    system.answer(inst)
    reader_prompt = next(c for c in seen if "NOTES:" in c and "Relevant facts:" not in c)
    assert "distilled-facts" not in reader_prompt


def test_aux_provider_handles_extraction_while_reader_uses_main():
    """fact-key extraction runs on the aux (cheap) provider; the reader uses the main one."""
    main_seen: list[str] = []
    aux_seen: list[str] = []

    def main_responder(prompt: str) -> str:
        main_seen.append(prompt)
        if "biscuit" in prompt.lower():
            return "NOTES: pet is Biscuit.\nANSWER: Biscuit"
        return ABSTAIN_SENTINEL

    def aux_responder(prompt: str) -> str:
        aux_seen.append(prompt)
        return "1. The user's pet is named Biscuit"

    system = CortexSystem(
        FakeProvider(responder=main_responder),
        top_k=4,
        use_fact_keys=True,
        aux_provider=FakeProvider(responder=aux_responder),
    )
    inst = _fact_key_instance()
    system.reset()
    system.ingest(inst)
    hyp, _ = system.answer(inst)

    # Extraction (one per session = 2 sessions) went to aux; never to main.
    assert len(aux_seen) == 2
    assert all("Extract the key atomic facts" in p for p in aux_seen)
    assert not any("Extract the key atomic facts" in p for p in main_seen)
    # The reader (NOTES prompt) ran on the main provider, and the answer grades correct.
    assert any("NOTES:" in p for p in main_seen)
    assert Judge("offline").grade(inst, hyp) is True


def test_is_recommendation_question_classifier():
    assert is_recommendation_question("Can you recommend a show for tonight?")
    assert is_recommendation_question("Any tips for keeping my kitchen clean?")
    assert is_recommendation_question("I'm deciding whether to buy a NAS now. What do you think?")
    assert is_recommendation_question("Can you suggest some activities for my commute?")
    assert not is_recommendation_question("What is the name of the user's dog?")
    assert not is_recommendation_question("How many days were between the two trips?")


def test_build_recommendation_prompt_never_abstains_and_grounds_in_preferences():
    chunks = [
        MemoryChunk(text="I love history podcasts, not true crime.", session_id="s1", date="d")
    ]
    prompt = build_recommendation_prompt(
        "Recommend a podcast for my commute?", "2026-06-27", chunks
    )
    assert "recommendation or decision" in prompt
    assert "ALWAYS give a recommendation" in prompt
    assert "history podcasts" in prompt  # the user's stated preference is in context


def _pref_instance() -> QAInstance:
    return QAInstance(
        question_id="q_pref",
        question_type="single-session-preference",
        question="Can you recommend a podcast for my commute?",
        answer="history podcasts",
        question_date="2026-06-27",
        answer_session_ids=["s1"],
        haystack_session_ids=["s1"],
        haystack_dates=["2026-01-01"],
        haystack_sessions=[
            [{"role": "user", "content": "I love history podcasts, not true crime."}]
        ],
    )


def test_preference_mode_routes_recommendation_questions_to_reco_reader():
    seen: list[str] = []

    def responder(prompt: str) -> str:
        seen.append(prompt)
        return "NOTES: likes history.\nANSWER: Try Hardcore History."

    system = CortexSystem(FakeProvider(responder=responder), top_k=2, use_preference_mode=True)
    inst = _pref_instance()
    system.reset()
    system.ingest(inst)
    system.answer(inst)
    assert "recommendation or decision" in seen[-1]  # routed to the preference-aware reader


def test_preference_mode_off_uses_factual_reader():
    seen: list[str] = []

    def responder(prompt: str) -> str:
        seen.append(prompt)
        return ABSTAIN_SENTINEL

    system = CortexSystem(FakeProvider(responder=responder), top_k=2)  # preference mode OFF
    inst = _pref_instance()
    system.reset()
    system.ingest(inst)
    system.answer(inst)
    assert "recommendation or decision" not in seen[-1]  # factual reader used


def test_build_reflection_prompt_has_structured_sections():
    chunks = [MemoryChunk(text="I started a new job at Acme.", session_id="s1", date="2026-01-01")]
    prompt = build_reflection_prompt("Where does the user work?", "2026-06-27", chunks)
    assert "TIMELINE" in prompt and "CURRENT FACTS" in prompt and "TOTALS" in prompt
    assert "I started a new job at Acme." in prompt


def test_reflection_mode_prepends_digest_to_reader():
    seen: list[str] = []

    def responder(prompt: str) -> str:
        seen.append(prompt)
        if "Digest:" in prompt:  # the reflection call
            return "CURRENT FACTS: the user's dog is Rex."
        return "NOTES: dog is Rex.\nANSWER: Rex"

    system = CortexSystem(FakeProvider(responder=responder), top_k=2, use_reflection=True)
    inst = _instance()
    system.reset()
    system.ingest(inst)
    system.answer(inst)

    assert any("Digest:" in p for p in seen)  # a reflection call happened
    reader_prompt = next(p for p in seen if "NOTES:" in p and "Digest:" not in p)
    assert "REFLECTION DIGEST" in reader_prompt
    assert "the user's dog is Rex." in reader_prompt


def test_reflection_none_adds_no_digest():
    seen: list[str] = []

    def responder(prompt: str) -> str:
        seen.append(prompt)
        if "Digest:" in prompt:
            return "NONE"
        return ABSTAIN_SENTINEL

    system = CortexSystem(FakeProvider(responder=responder), top_k=2, use_reflection=True)
    inst = _instance()
    system.reset()
    system.ingest(inst)
    system.answer(inst)
    reader_prompt = next(p for p in seen if "NOTES:" in p and "Digest:" not in p)
    assert "REFLECTION DIGEST" not in reader_prompt


def test_reader_prompt_has_chain_of_note_and_calibrated_abstention():
    chunks = [MemoryChunk(text="My dog is Rex.", session_id="s1", date="2026-01-01")]
    prompt = build_reader_prompt("What is the dog's name?", "2026-06-27", chunks)
    assert "NOTES" in prompt
    assert ABSTAIN_SENTINEL in prompt
    # Calibrated: explicitly instructs NOT to abstain out of mere uncertainty.
    assert "uncertain" in prompt.lower()
    assert "My dog is Rex." in prompt
    assert "2026-06-27" in prompt


# --- A2 detect-then-decline answerability gate -----------------------------------------


def test_gate_says_unanswerable_parsing_is_conservative():
    # Only an exact UNANSWERABLE token (case-insensitive, punctuation-stripped) declines.
    assert gate_says_unanswerable("UNANSWERABLE")
    assert gate_says_unanswerable("Verdict: UNANSWERABLE.")
    assert gate_says_unanswerable("unanswerable")
    # ANSWERABLE (a substring of UNANSWERABLE) must NOT misfire as a decline.
    assert not gate_says_unanswerable("ANSWERABLE")
    assert not gate_says_unanswerable("answerable")
    assert not gate_says_unanswerable("I think this is answerable")
    # Garbled/empty verdicts default to answerable (never suppress a good answer).
    assert not gate_says_unanswerable("")
    assert not gate_says_unanswerable("hmm not sure")


def test_build_answerability_gate_prompt_content():
    chunks = [MemoryChunk(text="My dog is Rex.", session_id="s1", date="2026-01-01")]
    prompt = build_answerability_gate_prompt("What is the dog's name?", "2026-06-27", chunks)
    assert "gatekeeper" in prompt.lower()
    assert "ANSWERABLE" in prompt and "UNANSWERABLE" in prompt
    assert "My dog is Rex." in prompt  # the retrieved memory is in context
    assert prompt.rstrip().endswith("Verdict:")


def test_answerability_gate_declines_returns_sentinel_and_skips_reader():
    """Gate rules UNANSWERABLE → emit the sentinel and never call the reader (also saves cost)."""
    calls: list[str] = []

    def responder(prompt: str) -> str:
        calls.append(prompt)
        if "Verdict:" in prompt:  # the gate call
            return "UNANSWERABLE"
        return "NOTES: the dog is Rex.\nANSWER: Rex"  # would-be reader answer

    system = CortexSystem(FakeProvider(responder=responder), top_k=2, use_answerability_gate=True)
    inst = _instance()
    system.reset()
    system.ingest(inst)
    hyp, _usage = system.answer(inst)

    assert hyp == ABSTAIN_SENTINEL
    assert any("Verdict:" in c for c in calls)  # the gate ran
    # The reader (NOTES/ANSWER prompt, which is not the gate) never ran — short-circuited.
    assert not any("NOTES:" in c and "Verdict:" not in c for c in calls)


def test_answerability_gate_answerable_runs_reader_normally():
    calls: list[str] = []

    def responder(prompt: str) -> str:
        calls.append(prompt)
        if "Verdict:" in prompt:
            return "ANSWERABLE"
        if "rex" in prompt.lower():
            return "NOTES: the dog is Rex.\nANSWER: Rex"
        return ABSTAIN_SENTINEL

    system = CortexSystem(FakeProvider(responder=responder), top_k=2, use_answerability_gate=True)
    inst = _instance()
    system.reset()
    system.ingest(inst)
    hyp, _usage = system.answer(inst)

    assert any("Verdict:" in c for c in calls)  # gate ran
    assert any("NOTES:" in c and "Verdict:" not in c for c in calls)  # reader ran
    assert Judge("offline").grade(inst, hyp) is True


def test_answerability_gate_off_by_default_makes_no_gate_call():
    calls: list[str] = []

    def responder(prompt: str) -> str:
        calls.append(prompt)
        if "rex" in prompt.lower():
            return "NOTES: the dog is Rex.\nANSWER: Rex"
        return ABSTAIN_SENTINEL

    system = CortexSystem(FakeProvider(responder=responder), top_k=2)  # gate OFF
    inst = _instance()
    system.reset()
    system.ingest(inst)
    system.answer(inst)
    assert not any("Verdict:" in c for c in calls)


def test_answerability_gate_runs_on_aux_provider_and_short_circuits_main():
    """The gate runs on the cheap aux provider; on a decline the main reader is never touched."""
    main_seen: list[str] = []
    aux_seen: list[str] = []

    def main_responder(prompt: str) -> str:
        main_seen.append(prompt)
        return "NOTES: the dog is Rex.\nANSWER: Rex"

    def aux_responder(prompt: str) -> str:
        aux_seen.append(prompt)
        return "UNANSWERABLE"

    system = CortexSystem(
        FakeProvider(responder=main_responder),
        top_k=2,
        use_answerability_gate=True,
        aux_provider=FakeProvider(responder=aux_responder),
    )
    inst = _instance()
    system.reset()
    system.ingest(inst)
    hyp, _usage = system.answer(inst)

    assert hyp == ABSTAIN_SENTINEL
    assert any("Verdict:" in p for p in aux_seen)  # gate ran on aux
    assert not any("Verdict:" in p for p in main_seen)  # not on main
    assert main_seen == []  # reader (main) never ran — short-circuited


def test_answerability_gate_skips_recommendation_questions():
    """Preference/recommendation questions never abstain, so the gate must not run for them."""
    calls: list[str] = []

    def responder(prompt: str) -> str:
        calls.append(prompt)
        if "Verdict:" in prompt:
            return "UNANSWERABLE"
        return "NOTES: likes history.\nANSWER: Try Hardcore History."

    system = CortexSystem(
        FakeProvider(responder=responder),
        top_k=2,
        use_preference_mode=True,
        use_answerability_gate=True,
    )
    inst = _pref_instance()  # a recommendation question
    system.reset()
    system.ingest(inst)
    hyp, _usage = system.answer(inst)

    assert not any("Verdict:" in c for c in calls)  # gate skipped
    assert hyp != ABSTAIN_SENTINEL  # answered via the recommendation reader


# --- A2 reader-side strict abstention (--strict-abstain) --------------------------------


def test_strict_abstain_reader_prompt_forbids_fabrication_but_allows_synthesis():
    chunks = [MemoryChunk(text="My dog is Rex.", session_id="s1", date="2026-01-01")]
    default_prompt = build_reader_prompt("What is the dog's name?", "2026-06-27", chunks)
    strict_prompt = build_reader_prompt(
        "What is the dog's name?", "2026-06-27", chunks, strict_abstain=True
    )
    # Still Chain-of-Note with the sentinel and the memory in context.
    assert "NOTES" in strict_prompt
    assert ABSTAIN_SENTINEL in strict_prompt
    assert "My dog is Rex." in strict_prompt
    # Strict clause present: don't fabricate an absent specific, but multi-hop synthesis allowed.
    assert "no memory" in strict_prompt.lower()
    assert "multi-hop" in strict_prompt.lower()
    # The default (non-strict) policy does NOT carry the don't-fabricate clause.
    assert "no memory" not in default_prompt.lower()


def test_strict_abstain_composes_with_answer_first():
    chunks = [MemoryChunk(text="My dog is Rex.", session_id="s1", date="2026-01-01")]
    prompt = build_reader_prompt("q", "2026-06-27", chunks, answer_first=True, strict_abstain=True)
    assert prompt.rstrip().endswith("ANSWER:")  # answer-first order preserved
    assert "no memory" in prompt.lower()  # strict policy present


def test_strict_abstain_threaded_through_system_to_reader():
    seen: list[str] = []

    def responder(prompt: str) -> str:
        seen.append(prompt)
        return "NOTES: the dog is Rex.\nANSWER: Rex"

    system = CortexSystem(FakeProvider(responder=responder), top_k=2, use_strict_abstain=True)
    inst = _instance()
    system.reset()
    system.ingest(inst)
    system.answer(inst)
    assert "no memory" in seen[-1].lower()  # strict policy reached the reader


# --- L2 listwise reranker (--rerank / --rerank-k) --------------------------------------


def test_parse_rerank_order_maps_1based_dedupes_and_bounds():
    # "2, 1, 3" over n=3 -> 0-based [1, 0, 2], in ranked order.
    assert parse_rerank_order("2, 1, 3", 3) == [1, 0, 2]
    # Surrounding prose is tolerated; out-of-range and duplicate ids are dropped.
    assert parse_rerank_order("Memory 3 then 1, then 3 again, and 99", 3) == [2, 0]
    # Garbled / empty -> [] so the caller falls back to RRF order.
    assert parse_rerank_order("none of them", 3) == []
    assert parse_rerank_order("", 3) == []


def test_build_rerank_prompt_numbers_memories_and_asks_for_ranking():
    chunks = [
        MemoryChunk(text="alpha fact", session_id="s1", date="2026-01-01"),
        MemoryChunk(text="beta fact", session_id="s2", date="2026-02-01"),
    ]
    prompt = build_rerank_prompt("which fact?", chunks)
    assert "[1]" in prompt and "[2]" in prompt  # memories are 1-numbered
    assert "alpha fact" in prompt and "beta fact" in prompt
    assert "reranking" in prompt.lower()
    assert prompt.rstrip().endswith("useful first):")


def test_rerank_off_by_default_makes_no_rerank_call():
    calls: list[str] = []

    def responder(prompt: str) -> str:
        calls.append(prompt)
        if "rex" in prompt.lower():
            return "NOTES: the dog is Rex.\nANSWER: Rex"
        return ABSTAIN_SENTINEL

    system = CortexSystem(FakeProvider(responder=responder), top_k=2)  # rerank OFF
    inst = _instance()
    system.reset()
    system.ingest(inst)
    system.answer(inst)
    assert not any("Ranked numbers" in c for c in calls)


def test_rerank_feeds_only_rerank_k_chunks_to_reader():
    calls: list[str] = []

    def responder(prompt: str) -> str:
        calls.append(prompt)
        if "Ranked numbers" in prompt:  # the rerank call
            return "1"  # keep memory 1 (the answer session is retrieved first)
        if "rex" in prompt.lower():
            return "NOTES: the dog is Rex.\nANSWER: Rex"
        return ABSTAIN_SENTINEL

    # pool = top_k 2 candidates, reranked down to rerank_k 1.
    system = CortexSystem(FakeProvider(responder=responder), top_k=2, use_rerank=True, rerank_k=1)
    inst = _instance()
    system.reset()
    system.ingest(inst)
    hyp, _usage = system.answer(inst)

    assert any("Ranked numbers" in c for c in calls)  # rerank ran
    reader_prompt = next(c for c in calls if "NOTES:" in c and "Ranked numbers" not in c)
    assert reader_prompt.count("[Memory ") == 1  # reader saw exactly rerank_k=1 memory
    assert Judge("offline").grade(inst, hyp) is True  # kept the answer-bearing memory


def test_rerank_garbled_response_falls_back_to_rrf_top_k():
    """An unparseable rerank reply must degrade to 'RRF top-k', never drop the answer."""
    calls: list[str] = []

    def responder(prompt: str) -> str:
        calls.append(prompt)
        if "Ranked numbers" in prompt:
            return "sorry, none are relevant"  # parses to [] -> backfill RRF order
        if "rex" in prompt.lower():
            return "NOTES: the dog is Rex.\nANSWER: Rex"
        return ABSTAIN_SENTINEL

    system = CortexSystem(FakeProvider(responder=responder), top_k=2, use_rerank=True, rerank_k=1)
    inst = _instance()
    system.reset()
    system.ingest(inst)
    hyp, _usage = system.answer(inst)

    reader_prompt = next(c for c in calls if "NOTES:" in c and "Ranked numbers" not in c)
    assert reader_prompt.count("[Memory ") == 1  # still exactly rerank_k
    # RRF put the answer session first, so backfill keeps it -> still correct.
    assert Judge("offline").grade(inst, hyp) is True


def test_rerank_skipped_when_pool_within_budget():
    """No rerank call when the retrieved pool already fits rerank_k (nothing to prune)."""
    calls: list[str] = []

    def responder(prompt: str) -> str:
        calls.append(prompt)
        if "rex" in prompt.lower():
            return "NOTES: the dog is Rex.\nANSWER: Rex"
        return ABSTAIN_SENTINEL

    # top_k 2 retrieved, rerank_k 5 -> pool (2) <= rerank_k (5) -> skip rerank.
    system = CortexSystem(FakeProvider(responder=responder), top_k=2, use_rerank=True, rerank_k=5)
    inst = _instance()
    system.reset()
    system.ingest(inst)
    system.answer(inst)
    assert not any("Ranked numbers" in c for c in calls)
