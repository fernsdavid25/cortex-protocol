"""CORTEX_TIER profile resolution — the frozen cheap tier IS the benchmark headline config.

MOAT-CRITICAL: these tests pin the cheap tier so a future edit cannot silently drift the hosted
server or the bench harness away from today's LongMemEval_S 0.932 / LoCoMo 0.813 numbers.
"""

from cortex.profiles import CHEAP, FLAGSHIP, TIER_PROFILES, resolve_tier


def test_default_resolves_to_cheap_with_frozen_values():
    name, prof = resolve_tier({})
    assert name == "cheap"
    # hosted engine knobs (byte-identical to serve.py's historical literals)
    assert prof["embed_model"] == "gemini-embedding-001"
    assert prof["embed_dim"] == 768
    assert prof["top_k"] == 8
    # bench reader knobs (the frozen LME headline)
    assert prof["reader_model"] == "gemini-3.5-flash"
    assert prof["bench_top_k"] == 50
    assert prof["answer_first"] is True
    assert prof["preference_mode"] is True
    assert prof["max_output_tokens"] == 8192


def test_none_env_reads_os_environ_and_defaults_cheap(monkeypatch):
    monkeypatch.delenv("CORTEX_TIER", raising=False)
    name, prof = resolve_tier()
    assert name == "cheap"
    assert prof == CHEAP


def test_unknown_tier_falls_back_to_cheap():
    name, prof = resolve_tier({"CORTEX_TIER": "does-not-exist"})
    assert name == "cheap"
    assert prof == CHEAP


def test_case_and_whitespace_insensitive():
    name, prof = resolve_tier({"CORTEX_TIER": "  FLAGSHIP  "})
    assert name == "flagship"
    assert prof["reader_model"] == "gemini-3.1-pro-preview"


def test_returned_dict_is_a_copy_not_the_global():
    _, prof = resolve_tier({"CORTEX_TIER": "cheap"})
    prof["top_k"] = 999
    prof["reader_model"] = "mutated"
    assert CHEAP["top_k"] == 8  # mutating the returned copy must not touch the module global
    assert CHEAP["reader_model"] == "gemini-3.5-flash"


def test_flagship_contains_every_cheap_key():
    _, flagship = resolve_tier({"CORTEX_TIER": "flagship"})
    for key in CHEAP:
        assert key in flagship, f"flagship is missing cheap key {key!r}"


def test_flagship_overrides_are_recall_time_firepower_only():
    # Flagship = cheap + validated recall-time firepower: pro reader, deep pool, vertex rerank.
    # It must NOT change any cheap WRITE-time / hosted knob (embedder, episodic, top_k, etc.).
    diff = {k for k in CHEAP if CHEAP[k] != FLAGSHIP[k]}
    assert diff == {"reader_model", "bench_top_k", "use_rerank", "rerank_backend"}
    assert FLAGSHIP["reader_model"] == "gemini-3.1-pro-preview"
    assert FLAGSHIP["bench_top_k"] == 100 and FLAGSHIP["use_rerank"] is True
    assert FLAGSHIP["rerank_backend"] == "vertex-ranking"
    # hosted engine knobs (embedder/dim/top_k) + write-time enrichment (episodic, graph) are
    # untouched -> the hosted cheap engine == the hosted flagship engine (recall byte-identical).
    for k in ("embed_model", "embed_dim", "top_k", "use_episodic", "use_graph", "extract_model"):
        assert CHEAP[k] == FLAGSHIP[k]


def test_editing_returned_flagship_cannot_change_cheap():
    _, flagship = resolve_tier({"CORTEX_TIER": "flagship"})
    flagship["embed_dim"] = 3072
    assert CHEAP["embed_dim"] == 768


def test_hosted_engine_knobs_invariant_across_tiers():
    # HOSTED SAFETY: the pgvector column dimension is fixed at deploy; a flagship tier must
    # NEVER change the embedder or it would corrupt every stored vector in Cloud SQL.
    _, cheap = resolve_tier({"CORTEX_TIER": "cheap"})
    _, flagship = resolve_tier({"CORTEX_TIER": "flagship"})
    for key in ("embed_model", "embed_dim", "top_k"):
        assert cheap[key] == flagship[key], f"hosted knob {key!r} drifted across tiers"
    assert (cheap["embed_model"], cheap["embed_dim"], cheap["top_k"]) == (
        "gemini-embedding-001",
        768,
        8,
    )


def test_tier_profiles_registry_keys():
    assert set(TIER_PROFILES) == {"cheap", "flagship"}


def test_cheap_enables_write_time_graph():
    # G2 write-time entity graph is a hosted write-time enrichment: ON in cheap (folded into the
    # episodic flash-lite call, so recall cost is unchanged) and inherited unchanged by flagship.
    _, cheap = resolve_tier({"CORTEX_TIER": "cheap"})
    assert cheap["use_graph"] is True
    assert FLAGSHIP["use_graph"] is True
