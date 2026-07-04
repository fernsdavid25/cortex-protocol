# Security Policy

## Supported versions

Cortex is pre-`1.0`. Security fixes land on the **latest release line** (the most recent published
release and the current `main`); older tags/commits are not patched. Storage format and APIs may
still change before `1.0`, so track the latest release.

| Version | Supported |
|---|---|
| latest release line (`main` + latest release) | ✅ |
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

### Stored input / memory poisoning

Recalled memory content is **untrusted data, not instructions.** Anything Cortex returns from a
`recall`/`recall_about`/`recall_timeline` (or the episodic/graph layers) is text that was written at
some earlier point — potentially by a different agent, a shared or multi-tenant memory, or an
imported/transferred memory file. A hostile writer can plant text crafted to read as an instruction
("ignore your previous rules and…", a fake tool call, an exfiltration prompt) so that it fires later
when some other agent recalls it. The **transfer/import path** (moving or merging a `memory.db` or an
exported memory bundle between users or agents) is the primary poisoning vector, because it lets
content authored under one trust boundary surface inside another.

Cortex stores and retrieves this text faithfully; it does **not** and cannot sanitize meaning. The
**consuming agent is responsible** for treating recalled memories as quoted, untrusted data — never
concatenating them into a system/instruction context, never auto-executing tool calls or links they
contain, and applying the same prompt-injection defenses it would to any third-party document. Only
import or transfer memory from sources you trust.

## Scope

In scope: the engine, store, providers, and MCP server under `server/cortex/` (and the
`bench/cortex_bench/` harness). Out of scope: how you manage your own Gemini key/account, and your
operating-system / file-system security.
