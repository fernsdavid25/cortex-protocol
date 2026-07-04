# Contributing to Cortex Protocol

Thanks for your interest! Cortex is an open-source, user-owned memory MCP server: you own one
portable memory, and any agent reads it with your consent. This guide covers local development.
(Project direction lives in [ROADMAP.md](ROADMAP.md); notable changes are recorded in
[CHANGELOG.md](CHANGELOG.md); please also read the [Code of Conduct](CODE_OF_CONDUCT.md).)

## Layout

- `server/cortex/` — the engine + MCP server (the published `cortex-protocol` package).
  - `mcp/server.py` — the stdio MCP server (`uvx --from cortex-protocol cortex-mcp`); exposes the six memory tools
    (`recall_about`/`recall_timeline` are live only with `CORTEX_GRAPH=1`/`CORTEX_EPISODIC=1`).
  - `memory.py` — the `CortexMemory` engine (memorize / recall / list / forget, plus the opt-in
    episodic, entity-graph, and anti-saturation layers).
  - `store/` — `sqlite_store.py` (the persistent per-user product store) and `memory_store.py`
    (the in-memory dense + BM25 store the retriever runs over).
  - `retrieve/hybrid.py` — dense + BM25 fused with Reciprocal Rank Fusion (RRF).
  - `reader/reader.py` — Chain-of-Note reader prompts + the write-time extraction/arbiter prompts.
  - `providers/` — the `LLMProvider` abstraction (BYOK): `gemini.py`, `caching.py`, `fake.py`.
- `bench/cortex_bench/` — the LongMemEval / LoCoMo evaluation harness.
- `tests/` — offline, deterministic tests (use `FakeProvider`; no live LLM calls).
- `docs/ARCHITECTURE.md` — a contributor-facing tour of the pipeline; read it first.

## Setup

[uv](https://docs.astral.sh/uv/) is the toolchain — it is the only prerequisite. The lint, type,
and test tools live in the `dev` optional-dependency group, so pass `--extra dev` to pull them
from the locked toolchain. These are the exact commands CI runs:

```bash
uv run --extra dev ruff check .                            # lint
uv run --extra dev ruff format --check .                   # format
uv run --extra dev mypy server/cortex bench/cortex_bench   # types
uv run --extra dev pytest                                  # tests (offline, deterministic)
```

`pythonpath` (`server`, `bench`) and `testpaths` are set in `pyproject.toml`, so a bare
`uv run --extra dev pytest` finds the packages and the suite. All four must pass before a PR — CI
runs them on every push and pull request to `main`, plus a gitleaks secret scan, a lockfile check,
and a (non-blocking) dependency audit.

## Conventions

- `from __future__ import annotations`; full type hints; prefer `collections.abc.Sequence` over
  `list` in parameters (avoids invariance errors).
- Lazy-import heavy/optional SDKs (`google-genai`, `fastmcp`) **inside functions** so offline code
  and tests never need them imported at module load.
- **Tests are offline + deterministic.** Use `FakeProvider` — never call a live model in a test.
- **BYOK, no secrets.** Read API keys from the environment only; never bundle, default, or hardcode
  a key. Do not commit `.env` or `service-account*.json` (both are gitignored; a gitleaks CI scan
  backs this up).
- Ruff line length is 100. Commit messages follow
  [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `test:`,
  `refactor:`, `chore:` …).
- **Keep it simple.** Add a mechanism only when a benchmark shows it earns its cost (see ROADMAP) —
  this principle already saved the project from a reflection layer that *hurt* accuracy.

## Pull request flow

1. Fork and branch from `main` (`feat/…`, `fix/…`).
2. Make the change with a test that covers it (offline, `FakeProvider`).
3. Run the four checks above locally until green.
4. Open a PR with a Conventional-Commit title and a short description of *what* and *why*. Link any
   related issue. If behaviour changed, add a `CHANGELOG.md` entry under `[Unreleased]`.
5. If your change affects accuracy or cost, include benchmark numbers (see below) — claims are
   gated on the suite, not on intuition.
6. Keep PRs focused and small; a reviewer will respond. CI must be green to merge.

## Running the benchmark

The harness needs a BYOK Gemini key in `.env` (see [`.env.example`](.env.example)):

```bash
# LongMemEval_S:
python -m cortex_bench.run --system cortex-v0 --variant s --limit 100 \
  --reader-model gemini-3.5-flash --top-k 50 --preference-mode --answer-first --judge gemini
# LoCoMo (place locomo10.json in bench/data/ first):
python -m cortex_bench.run --system cortex-v0 --locomo --reader-model gemini-2.5-flash \
  --top-k 100 --answer-first --judge gemini
```

Per-role models are env-configurable (`CORTEX_READER_MODEL`, `CORTEX_JUDGE_MODEL`,
`CORTEX_EMBED_MODEL`, …) — see [`.env.example`](.env.example) and [`bench/README.md`](bench/README.md).

## Security

Never commit secrets. Report vulnerabilities **privately** — see [SECURITY.md](SECURITY.md) — rather
than in a public issue.
