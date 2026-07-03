"""Offline, deterministic tests for the L7 agent-uplift eval (FakeProvider, no network).

The FakeProvider responder (``make_fact_responder``) answers a task correctly IFF the
scenario's fact cues are in the prompt, so it mimics an agent that can only answer when the
load-bearing fact actually reaches it. That lets us prove the core thesis deterministically:
memory beats no-memory on cross-session tasks, and cortex matches full-context at a fraction
of the input-token cost.
"""

from __future__ import annotations

from agent_uplift.harness import (
    ARM_ORDER,
    make_fact_responder,
    run_cortex,
    run_eval,
    run_full_context,
    run_memoryless,
)
from agent_uplift.scenarios import SCENARIOS, SCENARIOS_BY_ID

from cortex.memory import CortexMemory
from cortex.providers.fake import FakeProvider
from cortex.store.sqlite_store import SQLiteStore


def _provider() -> FakeProvider:
    """A fact-aware fake provider over the full suite, with wide vectors for clean retrieval."""
    return FakeProvider(responder=make_fact_responder(SCENARIOS), dim=64)


def test_memoryless_fails_but_full_context_passes_on_non_final_fact():
    # secret-store hides its answer fact in a middle session (index 4 of 12), so the answer
    # genuinely depends on recalling an EARLIER session, not the final one.
    scenario = SCENARIOS_BY_ID["secret-store"]
    non_final = " ".join(scenario.sessions[:-1]).lower()
    assert any(cue.lower() in non_final for cue in scenario.fact_cues)
    assert scenario.fact_cues and all(
        cue.lower() not in scenario.task.lower() for cue in scenario.fact_cues
    )

    provider = _provider()
    mem_answer, _ = run_memoryless(provider, scenario)
    full_answer, _ = run_full_context(provider, scenario)

    assert scenario.check(mem_answer) is False  # no history -> cannot answer
    assert scenario.check(full_answer) is True  # whole history -> answers


def test_cortex_passes_when_recall_surfaces_the_fact(tmp_path):
    scenario = SCENARIOS_BY_ID["secret-store"]
    provider = _provider()

    # Retrieval must SELECT the one fact session out of twelve: prove the recalled set
    # actually contains it (not just that the arm happened to pass).
    store = SQLiteStore(tmp_path / "probe.db")
    memory = CortexMemory(provider, store, top_k=5)
    for session in scenario.sessions:
        memory.memorize(session)
    hits = memory.recall(scenario.task)
    store.close()
    assert len(hits) <= 5
    assert any("vault" in hit.content.lower() for hit in hits), "fact session must be recalled"

    answer, _ = run_cortex(provider, scenario, db_path=tmp_path / "cortex.db", top_k=5)
    assert scenario.check(answer) is True


def test_checks_grade_right_and_wrong_answers():
    rename = SCENARIOS_BY_ID["db-table-rename"]
    assert rename.check("It's the accounts table now.") is True
    assert rename.check("The users table.") is False

    port = SCENARIOS_BY_ID["dev-server-port"]
    assert port.check("It runs on port 4100.") is True
    assert port.check("It runs on port 3000.") is False

    helper = SCENARIOS_BY_ID["helper-prefix"]
    assert helper.check("def _impl_validate_email(email: str) -> bool:") is True
    assert helper.check("def validateEmail(email):") is False  # camelCase + no type hints

    const = SCENARIOS_BY_ID["config-constant-prefix"]
    assert const.check("CFG_MAX_RETRY_COUNT = 5") is True
    assert const.check("max_retry_count = 5") is False  # missing CFG_ + not upper-snake


def test_every_gold_answer_satisfies_its_own_check():
    for scenario in SCENARIOS:
        assert scenario.check(scenario.answer) is True, scenario.id


def test_runner_aggregates_across_all_arms(tmp_path):
    provider = _provider()
    results = run_eval(provider, SCENARIOS, top_k=5, db_dir=tmp_path)

    assert set(results) == set(ARM_ORDER)
    for arm in ARM_ORDER:
        r = results[arm]
        assert r.n == len(SCENARIOS)
        assert 0.0 <= r.pass_rate <= 1.0

    # The thesis, made deterministic: no-memory fails every cross-session task; both
    # full-context and cortex answer them all.
    assert results["memoryless"].pass_rate == 0.0
    assert results["full_context"].pass_rate == 1.0
    assert results["cortex"].pass_rate == 1.0


def test_cortex_prompt_is_materially_shorter_than_full_context(tmp_path):
    # On a 12-session scenario, cortex feeds the reader only top-k recalled memories while
    # full-context feeds all twelve, so cortex's input tokens are a fraction of full-context's.
    scenario = SCENARIOS_BY_ID["secret-store"]
    provider = _provider()

    _, cortex_usage = run_cortex(provider, scenario, db_path=tmp_path / "cx.db", top_k=5)
    _, full_usage = run_full_context(provider, scenario)

    assert cortex_usage.input_tokens < full_usage.input_tokens
    assert cortex_usage.input_tokens <= 0.75 * full_usage.input_tokens
