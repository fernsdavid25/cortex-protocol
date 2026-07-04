# Changelog

All notable changes to Cortex Protocol are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/); this project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-07-04

Elite open-source hardening pass: documentation, quality gates, governance, and correctness.

### Added
- **Documentation site** (MkDocs Material + mkdocstrings) with API reference generated from
  docstrings; new `docs` extra (`pip install cortex-protocol[docs]`).
- **Coverage gate**: `pytest-cov` wired into CI with `[tool.coverage.run]`/`[tool.coverage.report]`
  configuration and `show_missing` reporting.
- **CI test matrix** across supported Python versions (3.11, 3.12).
- **Governance automation**: issue/PR templates, `CONTRIBUTING`/`CODE_OF_CONDUCT`/`SECURITY`
  policies, and automated dependency/release workflows.
- **Property-based tests** via Hypothesis (added to the `dev` extra).
- **Store `Protocol`**: an explicit typed interface for the persistence layer, decoupling the
  engine from the concrete SQLite store.
- **README badges** (CI, coverage, PyPI, license) and a single-sourced version.

### Changed
- **Version single-sourced** from `server/cortex/__init__.py` (`0.2.0`) via Hatch dynamic version;
  `pyproject.toml` and the `.mcpb` manifest now track it.
- **Stricter typing**: mypy `disallow_untyped_defs` enabled.
- **Stricter linting**: ruff `flake8-builtins` (`A`) enabled to forbid shadowing builtins.

### Fixed
- Correctness fixes surfaced by the stricter type/lint gates and expanded test coverage.

## [0.1.0] - 2026-07-04

Published to PyPI as [`cortex-protocol`](https://pypi.org/project/cortex-protocol/); install/run with `uvx --from cortex-protocol cortex-mcp`.

### Added
- **Local memory MCP server** (`uvx --from cortex-protocol cortex-mcp`): a stdio MCP server exposing six tools —
  `memorize`, `recall`, `list_memories`, `forget`, plus `recall_about` and `recall_timeline`
  (gated on the opt-in `CORTEX_GRAPH` / `CORTEX_EPISODIC` layers) — to coding agents (Claude Code,
  Cursor, …). BYOK (Gemini), local SQLite storage, zero phone-home.
- **Hybrid retrieval engine**: dense (cosine) + Okapi BM25, fused with Reciprocal Rank Fusion;
  Chain-of-Note reader with calibrated abstention; preference-aware reader mode.
- **Persistent per-user store** (SQLite, WAL, float32-blob vectors) with an embedding-signature
  guard, bounded short-id deletion, and graceful shutdown.
- **Per-role model selection via env** (`CORTEX_JUDGE_BACKEND/JUDGE_MODEL/READER_MODEL/
  EMBED_MODEL/EXTRACT_MODEL`); the GPT-4o judge is one switch away (`CORTEX_JUDGE_BACKEND=openai`).
- **LongMemEval harness** (`cortex_bench`) with per-type accuracy, abstention, recall@k, and
  per-question cost accounting; on-disk embedding cache; provider retry/backoff + request timeout.
- **CI**: ruff + mypy + pytest + gitleaks secret scan.
- **`.mcpb` bundle** ([`packaging/mcpb/`](packaging/mcpb/)) for one-click Claude Desktop install
  (launches `uvx --from cortex-protocol cortex-mcp`; key entered at install time).
- **LoCoMo evaluation** support in the harness (`--locomo`) for multi-hop / temporal / adversarial
  question categories.

### Benchmark
- **0.932 on the full LongMemEval_S (500 questions)** (~$0.008/query) and **0.813 on the full
  LoCoMo (1986 questions)** (~$0.0034/query), both with a cheap Gemini reader under a Gemini judge
  (LongMemEval cross-validated by an independent Claude-Opus panel). Positioning: **#1 on
  accuracy-per-dollar** — raw #2 on LongMemEval (behind Mastra 0.949), raw #3 on LoCoMo; **not raw
  SOTA**. The 0.894 → 0.932 LongMemEval gain came from the **A1 fix** (`--answer-first` +
  `--max-output-tokens 8192`). See
  [`bench/results/PHASE3_authoritative.md`](bench/results/PHASE3_authoritative.md),
  [`bench/results/LOCOMO_results.md`](bench/results/LOCOMO_results.md), and
  [`bench/results/leaderboard_research.md`](bench/results/leaderboard_research.md).

### Notes
- Versioned `0.1.0` (first public release). Storage format and APIs may change before `1.0.0`.

[Unreleased]: https://github.com/fernsdavid25/cortex-protocol/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/fernsdavid25/cortex-protocol/releases/tag/v0.2.0
[0.1.0]: https://github.com/fernsdavid25/cortex-protocol/releases/tag/v0.1.0
