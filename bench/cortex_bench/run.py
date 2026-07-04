"""LongMemEval harness runner: load → reset/ingest/answer per instance → judge → metrics."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from cortex.profiles import LOCOMO_DELTA, TIER_PROFILES, resolve_tier
from cortex.providers.base import LLMProvider

from .dataset import download, load_instances
from .judge import Judge
from .memory_system import MemorySystem, QAInstance
from .metrics import Record, aggregate
from .systems.cortex_system import CortexSystem
from .systems.full_context import FullContextSystem
from .systems.naive_rag import NaiveRAGSystem
from .systems.stub import GoldStub, NullStub

SYSTEMS = ("gold-stub", "null-stub", "full-context", "naive-rag", "cortex-v0")


def build_provider(kind: str, reader_model: str | None, embed_model: str | None) -> LLMProvider:
    if kind == "fake":
        from cortex.providers.fake import FakeProvider

        return FakeProvider()
    if kind == "gemini":
        from cortex.providers.gemini import GeminiProvider

        provider = GeminiProvider()
        if reader_model is not None:
            provider.reader_model = reader_model
        if embed_model is not None:
            provider.embed_model = embed_model
        return provider
    raise ValueError(f"unknown provider {kind!r}")


def build_system(
    name: str,
    provider: LLMProvider,
    *,
    fact_keys: bool = False,
    query_distill: bool = False,
    preference_mode: bool = False,
    reflection: bool = False,
    answer_first: bool = False,
    answerability_gate: bool = False,
    strict_abstain: bool = False,
    rerank: bool = False,
    rerank_k: int = 25,
    rerank_backend: str = "listwise",
    max_output_tokens: int = 256,
    aux_provider: LLMProvider | None = None,
    top_k: int = 10,
) -> MemorySystem:
    if name == "gold-stub":
        return GoldStub()
    if name == "null-stub":
        return NullStub()
    if name == "full-context":
        return FullContextSystem(provider)
    if name == "naive-rag":
        return NaiveRAGSystem(provider)
    if name == "cortex-v0":
        return CortexSystem(
            provider,
            top_k=top_k,
            use_fact_keys=fact_keys,
            use_query_distill=query_distill,
            use_preference_mode=preference_mode,
            use_reflection=reflection,
            use_answer_first=answer_first,
            use_answerability_gate=answerability_gate,
            use_strict_abstain=strict_abstain,
            use_rerank=rerank,
            rerank_k=rerank_k,
            rerank_backend=rerank_backend,
            max_output_tokens=max_output_tokens,
            aux_provider=aux_provider,
        )
    raise ValueError(f"unknown system {name!r}")


def resolve_run_config(args: argparse.Namespace) -> dict:
    """Resolve the reader/retrieval levers from the CORTEX_TIER profile + explicit CLI overrides.

    Precedence (highest first): an explicit CLI flag (value is not None) > the ``--locomo`` delta >
    the resolved tier profile. This is the SINGLE place the bench decides its knobs, so ``--tier
    cheap`` reproduces the frozen headline (LME: reader gemini-3.5-flash, top_k 50, answer_first,
    preference_mode, 8192 out-tokens) and ``--tier cheap --locomo`` reproduces the LoCoMo headline
    (reader gemini-2.5-flash, top_k 100, preference_mode off). The int levers (top_k,
    max_output_tokens, rerank_k) are guaranteed concrete ints, never the None sentinel.
    """
    name, prof = resolve_tier({"CORTEX_TIER": args.tier})
    if getattr(args, "locomo", False):
        delta = dict(LOCOMO_DELTA)
        # LOCOMO_DELTA's reader override (gemini-2.5-flash) is a CHEAP-tier choice — the flagship
        # keeps its premium reader on LoCoMo too, so don't let the delta downgrade it.
        if name == "flagship":
            delta.pop("reader_model", None)
        prof = {**prof, **delta}

    def pick(cli_value: Any, key: str) -> Any:
        return cli_value if cli_value is not None else prof[key]

    return {
        "tier": name,
        "reader_model": pick(args.reader_model, "reader_model"),
        "top_k": int(pick(args.top_k, "bench_top_k")),
        "answer_first": bool(pick(args.answer_first, "answer_first")),
        "preference_mode": bool(pick(args.preference_mode, "preference_mode")),
        "max_output_tokens": int(pick(args.max_output_tokens, "max_output_tokens")),
        "reflection": bool(pick(args.reflect, "use_reflection")),
        "rerank": bool(pick(args.rerank, "use_rerank")),
        "rerank_k": int(pick(args.rerank_k, "rerank_k")),
        "rerank_backend": str(pick(args.rerank_backend, "rerank_backend")),
    }


def select_instances(
    instances: Sequence[QAInstance],
    *,
    limit: int = 0,
    shuffle: bool = True,
    seed: int = 0,
) -> list[QAInstance]:
    """Pick the instances to run.

    The oracle JSON is grouped by question type, so a raw ``instances[:limit]`` slice is an
    unrepresentative single-type block (all temporal, zero abstention). To make a ``--limit``
    subset representative across question types AND abstention, we deterministically shuffle with
    ``random.Random(seed)`` BEFORE applying ``--limit``. Same seed -> same order (reproducible).
    """
    out = list(instances)
    if shuffle:
        random.Random(seed).shuffle(out)
    if limit:
        out = out[:limit]
    return out


def _load_done(hyp_path: Path) -> dict[str, dict]:
    """Read the prior COMPLETED `{question_id: row}` from a hypotheses jsonl (for --resume).

    A row counts as done (skip + don't re-run the reader) when it is NOT a recorded failure
    (``failed`` unset) AND carries a real result (a non-empty ``hypothesis`` or a stored
    ``correct``). Recorded failures (``failed: true``, written when an instance errors) are
    RETRIED, so a transient/quota death can never permanently poison the result. A torn/partial
    last line (process killed mid-write) is skipped, not fatal, so resume still works. Rows
    written since verdict-persistence carry ``correct``; legacy rows without it are re-graded
    (and the verdict written back) on resume. If a question_id repeats, the last done row wins.
    """
    done: dict[str, dict] = {}
    for line in hyp_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn line from a crash mid-write -> skip; that instance simply re-runs
        if not row.get("failed") and (row.get("hypothesis") or "correct" in row):
            done[row["question_id"]] = row
    return done


def _ensure_trailing_newline(hyp_path: Path) -> None:
    """Guarantee the file ends with a newline before appending, so a prior newline-less last
    line (process killed mid-write) can't be concatenated with the next record."""
    with hyp_path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        if fh.tell() == 0:
            return
        fh.seek(-1, os.SEEK_END)
        if fh.read(1) != b"\n":
            with hyp_path.open("a", encoding="utf-8") as afh:
                afh.write("\n")


def run(
    instances: list[QAInstance],
    system: MemorySystem,
    judge: Judge,
    *,
    hyp_path: Path | None = None,
    resume: bool = False,
) -> list[Record]:
    """Run each instance through the system and grade it.

    Resilient by design: provider transient errors are already retried inside the provider, so a
    SYSTEM (reader) exception here means the instance genuinely failed — we record it as a
    retriable failure (hypothesis="", correct=False, failed=True) and continue, so one bad
    instance never aborts a multi-minute run. The judge is called OUTSIDE that guard: a hard
    judge failure (e.g. judge quota) is NOT downgraded to a fabricated wrong verdict — it
    propagates and aborts the run, leaving the jsonl intact, so the headline accuracy can never
    be silently deflated. When `hyp_path` is given, each result is appended to that jsonl AS IT
    IS PRODUCED (flushed per line, with its judged verdict) so a crash preserves partial progress.

    With `resume=True`, any COMPLETED question_id in `hyp_path` is NOT re-run: the costly reader
    call is skipped and the saved result is recorded as `measured=False` (counts toward accuracy,
    excluded from cost/latency/recall). Its verdict is reused if persisted, so resume is
    judge-free for completed work; legacy lines without a verdict are re-graded once and written
    back. Recorded failures are retried (see `_load_done`).
    """
    records: list[Record] = []
    done: dict[str, dict] = {}
    mode = "w"
    if hyp_path is not None and resume and hyp_path.exists():
        done = _load_done(hyp_path)
        _ensure_trailing_newline(hyp_path)  # repair a torn last line before we append
        mode = "a"  # keep the already-written lines; append the rest
    fh = None
    if hyp_path is not None:
        hyp_path.parent.mkdir(parents=True, exist_ok=True)
        fh = hyp_path.open(mode, encoding="utf-8")

    def _persist(qid: str, hyp: str, correct: bool, *, failed: bool = False) -> None:
        if fh is None:
            return
        row: dict = {"question_id": qid, "hypothesis": hyp, "correct": correct}
        if failed:
            row["failed"] = True
        fh.write(json.dumps(row) + "\n")
        fh.flush()

    try:
        for inst in instances:
            if inst.question_id in done:
                # Already completed by a prior run: skip the costly reader call.
                row = done[inst.question_id]
                # A done row can carry a verdict but no hypothesis key (hand-edited / older
                # format): `_load_done` accepts it via ``"correct" in row``, so default the key
                # rather than KeyError and abort the resumed run.
                hyp = row.get("hypothesis", "")
                stored = row.get("correct")
                if stored is not None:
                    correct = stored  # reuse persisted verdict -> judge-free
                else:
                    # legacy line: re-grade ONCE and write the verdict back so later resumes
                    # reuse it. A hard judge failure here propagates (aborts, file intact).
                    correct = judge.grade(inst, hyp)
                    _persist(inst.question_id, hyp, correct)
                records.append(Record(inst, hyp, correct, measured=False))
                continue
            # Fresh instance. Catch SYSTEM (reader) failures only -> retriable failure.
            try:
                system.reset()
                u_ingest = system.ingest(inst)
                t0 = time.perf_counter()
                hyp, u_answer = system.answer(inst)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                usage = u_ingest + u_answer
                usage.latency_ms = max(usage.latency_ms, elapsed_ms)
                retrieved = system.retrieved_session_ids()
            except Exception as exc:  # noqa: BLE001 — keep the run alive past any one instance
                print(f"WARN: instance {inst.question_id} failed, recording as wrong: {exc}")
                records.append(Record(inst, "", False))
                _persist(inst.question_id, "", False, failed=True)
                continue
            # Judge OUTSIDE the system guard: a judge failure aborts cleanly (file intact)
            # instead of fabricating a wrong verdict over a good answer.
            correct = judge.grade(inst, hyp)
            records.append(Record(inst, hyp, correct, usage, retrieved))
            _persist(inst.question_id, hyp, correct)
    finally:
        if fh is not None:
            fh.close()
    return records


def main(argv: list[str] | None = None) -> dict:
    from dotenv import load_dotenv

    # Load THIS repo's .env by absolute path (cwd-independent) so a stale/suspended OS-env
    # GOOGLE_API_KEY can never shadow it.
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env", override=True)
    # GOOGLE_APPLICATION_CREDENTIALS in .env is repo-relative (service-account.json), but the
    # Vertex Ranking client resolves it via ADC relative to the CWD — so a run from any other
    # directory would fail to find it. Rewrite a relative value to an absolute path (once, here)
    # so the service account is found regardless of cwd.
    cred = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if cred and not os.path.isabs(cred):
        abs_cred = (repo_root / cred).resolve()
        if abs_cred.exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(abs_cred)
    ap = argparse.ArgumentParser(description="Run a system through the LongMemEval harness.")
    ap.add_argument("--data", help="path to a longmemeval json (overrides --variant)")
    ap.add_argument("--variant", default="oracle", choices=["s", "m", "oracle"])
    ap.add_argument(
        "--locomo",
        action="store_true",
        help="load the LoCoMo dataset (--data path or <data-dir>/locomo10.json) instead of "
        "LongMemEval; QA categories map to locomo-* question types.",
    )
    ap.add_argument("--data-dir", default="bench/data")
    ap.add_argument("--system", default="gold-stub", choices=list(SYSTEMS))
    # CORTEX_TIER profile = the single source of truth for the reader/retrieval levers below.
    # Any lever left unset (its default None sentinel) inherits the profile; an explicit flag
    # overrides it. Default 'cheap' = the frozen benchmark headline config.
    ap.add_argument(
        "--tier",
        default=os.environ.get("CORTEX_TIER", "cheap"),
        choices=list(TIER_PROFILES),
        help="tier profile supplying reader/retrieval lever defaults (env: CORTEX_TIER). "
        "Explicit lever flags below still override it.",
    )
    # Per-role model selection is env-configurable (CORTEX_* in .env) with CLI override, so the
    # judge/reader/embed/extract model can each be swapped (e.g. judge -> gpt-4o later) without
    # touching code. CLI flag > env var > built-in default.
    ap.add_argument(
        "--judge",
        default=os.environ.get("CORTEX_JUDGE_BACKEND", "offline"),
        choices=["offline", "gemini", "openai"],
        help="judge backend (env: CORTEX_JUDGE_BACKEND)",
    )
    ap.add_argument(
        "--judge-model",
        default=os.environ.get("CORTEX_JUDGE_MODEL"),
        help="judge model name; defaults per backend (gemini-3.5-flash / gpt-4o-2024-08-06). "
        "Env: CORTEX_JUDGE_MODEL. Set a non-reader model to avoid self-preference bias.",
    )
    ap.add_argument(
        "--judge-votes",
        type=int,
        default=int(os.environ.get("CORTEX_JUDGE_VOTES", "1")),
        help="self-consistency votes for the gemini judge (majority label); cuts non-determinism",
    )
    ap.add_argument("--provider", default="gemini", choices=["gemini", "fake"])
    ap.add_argument(
        "--reader-model",
        default=os.environ.get("CORTEX_READER_MODEL"),
        help="reader model name for $/question (env: CORTEX_READER_MODEL)",
    )
    ap.add_argument(
        "--embed-model",
        default=os.environ.get("CORTEX_EMBED_MODEL"),
        help="embedding model name (env: CORTEX_EMBED_MODEL)",
    )
    ap.add_argument(
        "--fact-keys",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="cortex-v0: distil each session into atomic fact keys as extra retrieval chunks",
    )
    ap.add_argument(
        "--query-distill",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="cortex-v0: one query-time generate() distils retrieved memories for the reader "
        "(cheap alternative to --fact-keys)",
    )
    ap.add_argument(
        "--preference-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="cortex-v0: route recommendation/preference questions to a preference-aware "
        "reader (never abstains; grounds advice in the user's stated preferences). "
        "Unset -> inherit --tier profile.",
    )
    ap.add_argument(
        "--reflect",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="cortex-v0: one cheap query-time reflection digests retrieved memories into a "
        "dated timeline / current-facts / totals digest prepended to the reader. "
        "Unset -> inherit --tier profile.",
    )
    ap.add_argument(
        "--answer-first",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="cortex-v0: reader emits ANSWER before NOTES so output-budget truncation cannot "
        "eat the answer (A1: 69/500 headline answers were truncated mid-NOTES at k=50). "
        "Unset -> inherit --tier profile.",
    )
    ap.add_argument(
        "--answerability-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="cortex-v0: A2 detect-then-decline — a conservative gate (aux model) rules each "
        "question ANSWERABLE/UNANSWERABLE over the retrieved memories BEFORE the reader; on "
        "UNANSWERABLE it emits the abstention sentinel without a reader call. EXPERIMENTAL, off "
        "by default: the n=100 LoCoMo probe showed NO adversarial recovery and +67% $/q — the "
        "hard adversarial cases look answer-bearing and evade the gate (see LOCOMO_results.md).",
    )
    ap.add_argument(
        "--strict-abstain",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="cortex-v0: A2 reader-side — the reader's answering policy forbids emitting a "
        "specific detail that NO memory states (still allowing multi-hop synthesis). Free (no "
        "extra model call), unlike --answerability-gate. EXPERIMENTAL, off by default: the n=100 "
        "LoCoMo probe showed NO adversarial recovery and hurt temporal (−0.25) — the reader "
        "declines the wrong questions (see LOCOMO_results.md).",
    )
    ap.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="cortex-v0: L2 listwise reranker — over-retrieve the deep --top-k pool, then a cheap "
        "aux call reorders it and the reader sees only the --rerank-k most useful memories. "
        "EXPERIMENTAL, off by default: the n=100 probe FAILED the +1pt gate — a cheap Gemini "
        "listwise reranker is not a cross-encoder and demotes weak multi-hop bridge chunks "
        "(LoCoMo multi-hop −20pt, recall-on-25 0.78; LME temporal −8pt, +22% $/q). Kept as "
        "reusable plumbing for a cross-encoder backend (Vertex Ranking); see LOCOMO_results.md.",
    )
    ap.add_argument(
        "--rerank-k",
        type=int,
        default=None,
        help="cortex-v0: how many reranked memories to feed the reader when --rerank is on "
        "(the deep pool is --top-k). Unset -> inherit --tier profile (cheap: 25).",
    )
    ap.add_argument(
        "--rerank-backend",
        default=None,
        choices=["listwise", "vertex-ranking"],
        help="cortex-v0: which reranker backend --rerank uses. Unset -> inherit the --tier profile "
        "(cheap: 'listwise'; flagship: 'vertex-ranking'). 'listwise' = the cheap aux-LLM listwise "
        "reranker (failed the +1pt gate). 'vertex-ranking' = the Vertex AI / Discovery Engine "
        "semantic-ranker cross-encoder (validated +1pt LoCoMo / ~3x cheaper; needs the [vertex] "
        "extra + GOOGLE_APPLICATION_CREDENTIALS; falls back to RRF top-k if creds are absent).",
    )
    ap.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="cortex-v0 generate budget (reader/distill/extract). Bump (e.g. 2048) for "
        "thinking readers (gemini-3.x) so reasoning tokens don't starve the answer. "
        "Unset -> inherit --tier profile (cheap: 8192).",
    )
    ap.add_argument(
        "--extract-model",
        default=os.environ.get("CORTEX_EXTRACT_MODEL"),
        help="cheaper model for fact-key extraction + query distillation (defaults to the "
        "reader model); pairs a strong reader with a cheap extractor. Env: CORTEX_EXTRACT_MODEL.",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="cortex-v0 retrieval depth (memories fed to the reader). Raise for multi-session "
        "questions that need many sessions in context. Unset -> inherit --tier profile "
        "(cheap LME: 50, locomo: 100).",
    )
    ap.add_argument(
        "--embed-cache",
        default="bench/.cache/embed.sqlite",
        help="on-disk embedding cache path (reuse embeddings across configs). "
        "Set to '' or 'none' to disable.",
    )
    ap.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="resume an interrupted run: skip question_ids already in the output hypotheses "
        "jsonl (re-grading their saved answers) and append the rest, instead of starting over. "
        "Cost/latency/recall are then reported over the freshly-run subset (measured_n).",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0, help="seed for the deterministic shuffle")
    ap.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="shuffle (with --seed) BEFORE --limit so a subset is representative (default: on)",
    )
    ap.add_argument("--out", default="bench/results")
    args = ap.parse_args(argv)

    if args.locomo:
        from .locomo import load_locomo

        loaded = load_locomo(args.data or f"{args.data_dir}/locomo10.json")
    else:
        loaded = load_instances(args.data or download(args.variant, args.data_dir))
    instances = select_instances(
        loaded,
        limit=args.limit,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    # Resolve reader/retrieval levers from the CORTEX_TIER profile (+ locomo delta) with CLI
    # overrides, so every call site below uses the resolved values (NOT args.* levers directly).
    resolved = resolve_run_config(args)
    provider = build_provider(args.provider, resolved["reader_model"], args.embed_model)
    if args.embed_cache and args.embed_cache.lower() != "none":
        from cortex.providers.caching import CachingProvider

        provider = CachingProvider(provider, args.embed_cache)
    # Optional cheaper model for extraction/distillation (no embedding -> no cache wrapper).
    aux_provider = None
    if args.extract_model:
        aux_provider = build_provider(args.provider, args.extract_model, args.embed_model)
    system = build_system(
        args.system,
        provider,
        fact_keys=args.fact_keys,
        query_distill=args.query_distill,
        preference_mode=resolved["preference_mode"],
        reflection=resolved["reflection"],
        answer_first=resolved["answer_first"],
        answerability_gate=args.answerability_gate,
        strict_abstain=args.strict_abstain,
        rerank=resolved["rerank"],
        rerank_k=resolved["rerank_k"],
        rerank_backend=resolved["rerank_backend"],
        max_output_tokens=resolved["max_output_tokens"],
        aux_provider=aux_provider,
        top_k=resolved["top_k"],
    )
    judge = Judge(backend=args.judge, model=args.judge_model, votes=args.judge_votes)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{system.name}_{'locomo' if args.locomo else args.variant}"
    # Hypotheses are written incrementally by run() so a crash mid-run keeps partial progress;
    # --resume continues from those saved lines instead of restarting.
    records = run(
        instances,
        system,
        judge,
        hyp_path=outdir / f"{tag}_hypotheses.jsonl",
        resume=args.resume,
    )
    embed_model = args.embed_model or "gemini-embedding-001"
    report = aggregate(
        records,
        reader_model=resolved["reader_model"],
        embed_model=embed_model,
        system_name=system.name,
    )

    (outdir / f"{tag}_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
