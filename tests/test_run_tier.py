"""Bench-side tier resolution: resolve_run_config maps the CORTEX_TIER profile (+ locomo delta)
and explicit CLI overrides onto the concrete reader/retrieval levers main() feeds the system.

MOAT-CRITICAL: these tests pin that `--tier cheap` reproduces the frozen LME 0.932 lever values
and `--tier cheap --locomo` reproduces the LoCoMo 0.813 lever values.
"""

from argparse import Namespace

from cortex_bench.run import resolve_run_config


def _args(**overrides) -> Namespace:
    """A Namespace with every tier lever at its None sentinel (i.e. 'inherit the profile')."""
    base = dict(
        tier="cheap",
        locomo=False,
        reader_model=None,
        top_k=None,
        answer_first=None,
        preference_mode=None,
        max_output_tokens=None,
        reflect=None,
        rerank=None,
        rerank_k=None,
        rerank_backend=None,
    )
    base.update(overrides)
    return Namespace(**base)


def test_cheap_tier_reproduces_lme_headline():
    r = resolve_run_config(_args(tier="cheap"))
    assert r["reader_model"] == "gemini-3.5-flash"
    assert r["top_k"] == 50  # LME retrieval depth (bench_top_k)
    assert r["answer_first"] is True
    assert r["preference_mode"] is True
    assert r["max_output_tokens"] == 8192


def test_cheap_locomo_reproduces_locomo_headline():
    r = resolve_run_config(_args(tier="cheap", locomo=True))
    assert r["reader_model"] == "gemini-2.5-flash"
    assert r["top_k"] == 100
    assert r["preference_mode"] is False
    assert r["answer_first"] is True  # inherited from cheap; locomo delta doesn't touch it


def test_flagship_pro_reader_deep_pool_and_vertex_rerank():
    r = resolve_run_config(_args(tier="flagship"))
    assert r["reader_model"] == "gemini-3.1-pro-preview"
    assert r["top_k"] == 100  # deep pool for the cross-encoder to select from
    assert r["rerank"] is True
    assert r["rerank_backend"] == "vertex-ranking"  # the validated cross-encoder, not listwise
    assert r["preference_mode"] is True
    assert r["answer_first"] is True


def test_flagship_locomo_keeps_pro_reader_pool_and_vertex_rerank():
    # On LoCoMo the flagship KEEPS its premium reader (the LOCOMO_DELTA reader downgrade is a
    # cheap-tier-only choice); preference-mode still drops off, and the deep pool + rerank persist.
    r = resolve_run_config(_args(tier="flagship", locomo=True))
    assert r["reader_model"] == "gemini-3.1-pro-preview"  # NOT downgraded to gemini-2.5-flash
    assert r["preference_mode"] is False  # LOCOMO_DELTA still drops preference-mode
    assert r["top_k"] == 100
    assert r["rerank"] is True
    assert r["rerank_backend"] == "vertex-ranking"


def test_explicit_top_k_overrides_profile():
    r = resolve_run_config(_args(tier="cheap", top_k=7))
    assert r["top_k"] == 7


def test_explicit_boolean_false_overrides_profile_true():
    # BooleanOptionalAction(default=None) preserves the 3-state: an explicit --no-answer-first
    # (False) must beat the profile's True, not be confused with "unset".
    r = resolve_run_config(_args(tier="cheap", answer_first=False, preference_mode=False))
    assert r["answer_first"] is False
    assert r["preference_mode"] is False


def test_explicit_reader_model_overrides_profile():
    r = resolve_run_config(_args(tier="cheap", reader_model="gemini-x"))
    assert r["reader_model"] == "gemini-x"


def test_int_levers_are_always_concrete_ints():
    for args in (
        _args(tier="cheap"),
        _args(tier="cheap", locomo=True),
        _args(tier="flagship"),
    ):
        r = resolve_run_config(args)
        for key in ("top_k", "max_output_tokens", "rerank_k"):
            assert isinstance(r[key], int), f"{key} must be a concrete int, got {r[key]!r}"


def test_additive_stages_off_in_cheap():
    r = resolve_run_config(_args(tier="cheap"))
    assert r["reflection"] is False
    assert r["rerank"] is False
    assert r["rerank_k"] == 25
