# Changelog

All notable changes to Cortex Protocol are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/); this project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Local memory MCP server** (`uvx cortex-mcp`): a stdio MCP server exposing `memorize`,
  `recall`, `list_memories`, and `forget` to coding agents (Claude Code, Cursor, …). BYOK
  (Gemini), local SQLite storage, zero phone-home.
- **Hybrid retrieval engine**: dense (cosine) + Okapi BM25, fused with Reciprocal Rank Fusion;
  Chain-of-Note reader with calibrated abstention; preference-aware reader mode.
- **Persistent per-user store** (SQLite, WAL, float32-blob vectors) with an embedding-signature
  guard, bounded short-id deletion, and graceful shutdown.
- **Per-role model selection via env** (`CORTEX_JUDGE_BACKEND/JUDGE_MODEL/READER_MODEL/
  EMBED_MODEL/EXTRACT_MODEL`); the GPT-4o judge is one switch away (`CORTEX_JUDGE_BACKEND=openai`).
- **LongMemEval harness** (`cortex_bench`) with per-type accuracy, abstention, recall@k, and
  per-question cost accounting; on-disk embedding cache; provider retry/backoff + request timeout.
- **CI**: ruff + mypy + pytest, gitleaks secret scan, dependency audit, and lockfile check.
- **`.mcpb` bundle** ([`packaging/mcpb/`](packaging/mcpb/)) for one-click Claude Desktop install
  (launches `uvx cortex-mcp`; key entered at install time).
- **LoCoMo evaluation** support in the harness (`--locomo`) for multi-hop / temporal / adversarial
  question categories.

### Benchmark
- **0.894 on the full LongMemEval_S (500 questions)** with a cheap Gemini reader (~$0.008/query),
  under a Gemini judge (cross-validated by an independent Claude-Opus panel). See
  [`bench/results/PHASE3_authoritative.md`](bench/results/PHASE3_authoritative.md) and
  [`bench/results/leaderboard_research.md`](bench/results/leaderboard_research.md).

### Notes
- Versioned `0.0.1` (pre-release). Storage format and APIs may change before `0.1.0`.
