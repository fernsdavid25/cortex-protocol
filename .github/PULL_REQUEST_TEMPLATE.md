<!--
Thanks for contributing to Cortex Protocol! Please give the PR a Conventional Commit
title (e.g. `feat: ...`, `fix: ...`, `docs: ...`) and fill in the sections below.
See CONTRIBUTING.md for the full flow.
-->

## What & why

<!-- What does this change do, and why? Link any related issue (e.g. Closes #123). -->

## Checklist

- [ ] PR title follows [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:` …).
- [ ] Added/updated a test that covers this change (offline, deterministic, `FakeProvider` — no live LLM calls).
- [ ] `uv run --extra dev ruff check .` passes (lint).
- [ ] `uv run --extra dev ruff format --check .` passes (format).
- [ ] `uv run --extra dev mypy server/cortex bench/cortex_bench` passes (types).
- [ ] `uv run --extra dev pytest` passes (tests).
- [ ] Updated `CHANGELOG.md` under `[Unreleased]` if behaviour changed.
- [ ] If accuracy or cost is affected, benchmark numbers are included (claims are gated on the suite).

## Notes for reviewers

<!-- Anything reviewers should focus on, trade-offs, follow-ups, or screenshots/benchmarks. -->
