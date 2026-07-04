# Cortex `.mcpb` bundle (Claude Desktop one-click install)

[MCPB](https://github.com/anthropics/mcpb) packages an MCP server so a user installs **one file**.
This bundle is a *command* bundle: it launches `uvx --from cortex-protocol cortex-mcp`, so it needs
[`uv`](https://docs.astral.sh/uv/) on the machine and the `cortex-protocol` package available
(PyPI once published). It does **not** vendor Python (google-genai/grpc/pydantic native wheels
aren't portably bundleable) — the trade-off is the small `uv` dependency.

## Build

```bash
npx @anthropic-ai/mcpb validate packaging/mcpb/manifest.json
npx @anthropic-ai/mcpb pack packaging/mcpb dist/cortex-protocol.mcpb
# optional, for distribution:
npx @anthropic-ai/mcpb sign dist/cortex-protocol.mcpb
```

## Install
Drag `dist/cortex-protocol.mcpb` onto **Claude Desktop** (Settings → Extensions). Desktop prompts
for your **Gemini API key** (stored in the OS keychain) and optional DB path / namespace, then runs
the six tools (`memorize`, `recall`, `list_memories`, `forget`, plus `recall_about` and
`recall_timeline`, which require the opt-in `CORTEX_GRAPH` / `CORTEX_EPISODIC` layers).

The `.mcpb` itself is a build artifact (in `dist/`, gitignored); only the `manifest.json` source is
committed so the bundle is reproducible.
