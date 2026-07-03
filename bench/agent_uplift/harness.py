"""Agent-uplift harness: three arms answer each cross-session task, then we score them.

The arms:

- ``memoryless`` — the agent sees ONLY the final task (no history). It should FAIL any task
  whose answer lives in an earlier session (the cross-session floor).
- ``full_context`` — the agent sees ALL prior sessions concatenated + the task. It should
  PASS, but pays for every historical token on every query (does not scale to decades).
- ``cortex`` — each session is ``memorize()``d into a fresh ``CortexMemory`` (a per-scenario
  SQLite file); at task time ``recall(task)`` fetches the top-k relevant memories, which go
  into the prompt. It should PASS at a FRACTION of full_context's input tokens.

The harness loops scenarios x arms, grades each answer with the scenario's ``check``, and
aggregates per arm: pass-rate, mean input tokens (the cost proxy), and mean latency. Run it
offline with ``--provider fake`` (a fact-aware responder, below) or live with
``--provider gemini`` (the real BYOK key, loaded from ``.env`` exactly like ``cortex_bench``).
"""

from __future__ import annotations

import argparse
import random
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_uplift.scenarios import Scenario
    from cortex.providers.base import LLMProvider
    from cortex_bench.memory_system import Usage

# This file is runnable BOTH as a module (``agent_uplift.harness``, e.g. under pytest where
# pythonpath already has server/ + bench/) AND as a bare script (``python
# bench/agent_uplift/harness.py``, where only this dir is on sys.path). Put server/ and bench/
# on the path once, up front, so ``import cortex`` and ``import agent_uplift`` both resolve in
# script mode; it is a harmless no-op when they are already importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_import_paths() -> None:
    for sub in ("server", "bench"):
        p = str(_REPO_ROOT / sub)
        if p not in sys.path:
            sys.path.insert(0, p)


_ensure_import_paths()

# The default per-scenario recall depth for the cortex arm.
DEFAULT_TOP_K = 5

# The three arms, in table/report order.
ARM_ORDER: tuple[str, ...] = ("memoryless", "full_context", "cortex")

_INSTRUCTION = (
    "You are a software engineering assistant answering with the help of memory from "
    "earlier work sessions on this project.\n"
    "Use ONLY the context provided below to answer the task. Be concise and direct.\n"
    "If the context does not contain the information needed, reply exactly: I don't know.\n\n"
)


def make_fact_responder(scenarios: Sequence[Scenario]) -> Callable[[str], str]:
    """Build a deterministic ``FakeProvider`` responder for OFFLINE runs.

    It answers a task correctly IFF every one of that scenario's ``fact_cues`` is present in
    the prompt (i.e. the load-bearing fact was actually put in context) — so the memoryless
    arm fails, and the full_context/cortex arms pass exactly when the fact reaches the reader.
    The scenario is identified by its verbatim task text in the prompt (longest match wins, a
    safe tiebreak since a prompt only ever carries one scenario's task).
    """
    ordered = sorted(scenarios, key=lambda s: len(s.task), reverse=True)

    def responder(prompt: str) -> str:
        low = prompt.lower()
        for scenario in ordered:
            if scenario.task.lower() in low:
                if all(cue.lower() in low for cue in scenario.fact_cues):
                    return scenario.answer
                return "I don't know."
        return "I don't know."

    return responder


def _format_sessions(sessions: Sequence[str]) -> str:
    """Render every session as a numbered line for the full_context arm."""
    return "\n".join(f"[session {i}] {text}" for i, text in enumerate(sessions))


def _build_prompt(task: str, context: str, *, label: str) -> str:
    """Assemble the reader prompt shared by all three arms (only the context differs)."""
    return f"{_INSTRUCTION}Earlier {label}:\n{context}\n\nTask: {task}\n\nAnswer:"


def _timed_generate(
    provider: LLMProvider, prompt: str, *, max_output_tokens: int
) -> tuple[str, Usage]:
    """Run one reader ``generate`` and package its answer + token/latency usage."""
    from cortex_bench.memory_system import Usage

    t0 = time.perf_counter()
    result = provider.generate(prompt, max_output_tokens=max_output_tokens)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    usage = Usage(
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=elapsed_ms,
    )
    return result.text.strip(), usage


def run_memoryless(
    provider: LLMProvider, scenario: Scenario, *, max_output_tokens: int = 256
) -> tuple[str, Usage]:
    """Arm 1: the agent sees the task with NO history — the cross-session floor."""
    prompt = _build_prompt(scenario.task, "(no earlier sessions are available)", label="context")
    return _timed_generate(provider, prompt, max_output_tokens=max_output_tokens)


def run_full_context(
    provider: LLMProvider, scenario: Scenario, *, max_output_tokens: int = 256
) -> tuple[str, Usage]:
    """Arm 2: the agent sees EVERY prior session concatenated — the accuracy ceiling."""
    context = _format_sessions(scenario.sessions)
    prompt = _build_prompt(scenario.task, context, label="sessions (full history)")
    return _timed_generate(provider, prompt, max_output_tokens=max_output_tokens)


def run_cortex(
    provider: LLMProvider,
    scenario: Scenario,
    *,
    db_path: str | Path,
    top_k: int = DEFAULT_TOP_K,
    max_output_tokens: int = 256,
) -> tuple[str, Usage]:
    """Arm 3: memorize each session into a fresh store, then recall the top-k for the task.

    Only the recalled memories reach the reader, so the prompt is a fraction of the full
    history while still carrying the load-bearing fact (when retrieval surfaces it).
    """
    from cortex.memory import CortexMemory
    from cortex.store.sqlite_store import SQLiteStore

    store = SQLiteStore(db_path)
    try:
        memory = CortexMemory(provider, store, top_k=top_k)
        for i, session in enumerate(scenario.sessions):
            memory.memorize(session, metadata={"session_index": i})
        hits = memory.recall(scenario.task)
        context = "\n".join(f"- {hit.content}" for hit in hits) or "(no memories recalled)"
        prompt = _build_prompt(scenario.task, context, label="memories recalled for this task")
        return _timed_generate(provider, prompt, max_output_tokens=max_output_tokens)
    finally:
        store.close()


@dataclass
class ArmResult:
    """Per-arm aggregate over the scenario suite."""

    arm: str
    n: int
    passes: int
    mean_input_tokens: float
    mean_latency_ms: float

    @property
    def pass_rate(self) -> float:
        return self.passes / self.n if self.n else 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_eval(
    provider: LLMProvider,
    scenarios: Sequence[Scenario],
    *,
    top_k: int = DEFAULT_TOP_K,
    db_dir: str | Path,
) -> dict[str, ArmResult]:
    """Run all three arms over every scenario and aggregate per arm.

    Each scenario gets its OWN fresh SQLite file under ``db_dir`` for the cortex arm, so
    scenarios never share memory. Grading is the scenario's own deterministic ``check``.
    """
    db_root = Path(db_dir)
    passes: dict[str, int] = {arm: 0 for arm in ARM_ORDER}
    in_tokens: dict[str, list[float]] = {arm: [] for arm in ARM_ORDER}
    latencies: dict[str, list[float]] = {arm: [] for arm in ARM_ORDER}

    for scenario in scenarios:
        answers: dict[str, tuple[str, Usage]] = {
            "memoryless": run_memoryless(provider, scenario),
            "full_context": run_full_context(provider, scenario),
            "cortex": run_cortex(
                provider, scenario, db_path=db_root / f"{scenario.id}.db", top_k=top_k
            ),
        }
        for arm in ARM_ORDER:
            answer, usage = answers[arm]
            if scenario.check(answer):
                passes[arm] += 1
            in_tokens[arm].append(usage.input_tokens)
            latencies[arm].append(usage.latency_ms)

    n = len(scenarios)
    return {
        arm: ArmResult(
            arm=arm,
            n=n,
            passes=passes[arm],
            mean_input_tokens=_mean(in_tokens[arm]),
            mean_latency_ms=_mean(latencies[arm]),
        )
        for arm in ARM_ORDER
    }


def format_table(results: dict[str, ArmResult]) -> str:
    """Render the per-arm aggregates as a compact fixed-width table."""
    header = f"{'arm':<14}{'pass_rate':>11}{'mean_input_tok':>17}{'mean_latency_ms':>17}"
    lines = [header, "-" * len(header)]
    for arm in ARM_ORDER:
        r = results[arm]
        lines.append(
            f"{r.arm:<14}{r.pass_rate:>11.2f}{r.mean_input_tokens:>17.1f}{r.mean_latency_ms:>17.2f}"
        )
    return "\n".join(lines)


def select_scenarios(
    scenarios: Sequence[Scenario], *, limit: int = 0, seed: int = 0
) -> list[Scenario]:
    """Deterministically shuffle (by ``seed``) then take ``limit`` — a representative subset.

    Mirrors ``cortex_bench.run.select_instances``: shuffling BEFORE the slice keeps a
    ``--limit`` subset mixed across scenario kinds instead of a single-kind block.
    """
    out = list(scenarios)
    random.Random(seed).shuffle(out)
    if limit:
        out = out[:limit]
    return out


def build_provider(kind: str, scenarios: Sequence[Scenario]) -> LLMProvider:
    """Build the provider for an arm run — offline fake (fact-aware) or the live Gemini key."""
    if kind == "fake":
        from cortex.providers.fake import FakeProvider

        # Wider vectors than the dim-16 default so bag-of-words retrieval separates the buried
        # fact session from the noise cleanly and deterministically.
        return FakeProvider(responder=make_fact_responder(scenarios), dim=64)
    if kind == "gemini":
        from cortex.providers.gemini import GeminiProvider

        return GeminiProvider()
    raise ValueError(f"unknown provider {kind!r}")


def main(argv: list[str] | None = None) -> dict[str, ArmResult]:
    from dotenv import load_dotenv

    # Load THIS repo's .env by absolute path (cwd-independent) so --provider gemini works with
    # the real key regardless of where the script is launched from — exactly like cortex_bench.
    load_dotenv(_REPO_ROOT / ".env", override=True)

    from agent_uplift.scenarios import SCENARIOS

    ap = argparse.ArgumentParser(
        description="Agent-uplift eval: memory vs no-memory vs full-context."
    )
    ap.add_argument("--provider", choices=["gemini", "fake"], default="fake")
    ap.add_argument("--limit", type=int, default=0, help="cap the number of scenarios (0 = all)")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="cortex recall depth")
    ap.add_argument("--seed", type=int, default=0, help="seed for the deterministic shuffle")
    args = ap.parse_args(argv)

    scenarios = select_scenarios(SCENARIOS, limit=args.limit, seed=args.seed)
    provider = build_provider(args.provider, scenarios)

    with tempfile.TemporaryDirectory(prefix="agent_uplift_") as tmp:
        results = run_eval(provider, scenarios, top_k=args.top_k, db_dir=tmp)

    print(
        f"agent-uplift eval | provider={args.provider} | "
        f"scenarios={len(scenarios)} | top_k={args.top_k}"
    )
    print(format_table(results))
    return results


if __name__ == "__main__":
    main()
