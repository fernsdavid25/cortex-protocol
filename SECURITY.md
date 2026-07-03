# Security Policy

## Supported versions

Cortex is pre-`1.0` (currently `0.0.x`). Only the **latest commit on `main`** (and the most recent
release) is supported with security fixes. Storage format and APIs may change before `0.1.0`.

| Version | Supported |
|---|---|
| latest `main` / latest release | ✅ |
| any older tag/commit | ❌ |

## Reporting a vulnerability

**Please report security issues privately. Do NOT open a public GitHub issue for a vulnerability** —
public disclosure before a fix puts users at risk.

Use either private channel:

- **Preferred — GitHub private vulnerability reporting:** on this repository, go to
  **Security → Advisories → "Report a vulnerability"**. This opens a private advisory only the
  maintainers can see.
- **Email:** <info@wafer.ee>.

Include repro steps, affected version/commit, and impact. We will acknowledge, investigate, and
coordinate a fix and disclosure with you; please give us a reasonable window before any public
write-up.

## Security model

Cortex is a **local, bring-your-own-key** memory server. What that means for your security:

- **No phone-home.** The local server has no backend of its own. The only outbound network call is
  to **Google's embedding API, using your own key** — memory text is embedded under your key
  (subject to Google's data terms), so choose a key/project accordingly.
- **BYOK.** Keys are read only from the environment (`GEMINI_API_KEY` / `GOOGLE_API_KEY`); none are
  bundled, defaulted, or hardcoded.
- **Local storage is a plain SQLite file** (`~/.cortex/memory.db` by default), **unencrypted** —
  protect it like any local data, and **do not store secrets** as memories.
- **Trusted config.** `CORTEX_DB_PATH` is operator config; never set it from untrusted/agent input.
- **Hardening in place:** all SQL is parameterized; agent-supplied inputs are bounded (recall/list
  limits are clamped, short-id deletion refuses ambiguous matches); embeddings are validated
  (non-empty, correct dimension) before storage; CI runs a gitleaks secret scan and a dependency
  audit on every push and PR.

## Scope

In scope: the engine, store, providers, and MCP server under `server/cortex/` (and the
`bench/cortex_bench/` harness). Out of scope: how you manage your own Gemini key/account, and your
operating-system / file-system security.
