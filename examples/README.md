# Connecting Cortex to your agent

The local Cortex server is a **stdio MCP server**. Every major MCP client uses the same
`mcpServers` config shape ([`mcp.json`](mcp.json) here) — only the file location differs.
Copy the `cortex` block into the right file and restart the client.

| Client | Config file |
|---|---|
| **Claude Code** | `~/.claude.json` (global), or `.mcp.json` in a project root |
| **Cursor** | `~/.cursor/mcp.json` (global), or `.cursor/mcp.json` in a project |
| **Claude Desktop** | `claude_desktop_config.json` (Settings → Developer → Edit Config) |
| **VS Code (MCP)** | `.vscode/mcp.json` |

```json
{
  "mcpServers": {
    "cortex": {
      "command": "uvx",
      "args": ["cortex-mcp"],
      "env": { "GEMINI_API_KEY": "your-gemini-api-key-here" }
    }
  }
}
```

## Notes

- **BYOK.** `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) is *your* key — get one at
  <https://aistudio.google.com/apikey>. Nothing is bundled and nothing phones home.
- **Storage.** Memories live in a local SQLite file at `~/.cortex/memory.db`. Override with
  `CORTEX_DB_PATH`. Other optional env vars: `CORTEX_USER_ID`, `CORTEX_EMBED_MODEL`,
  `CORTEX_EMBED_DIM`, `CORTEX_TOP_K` (see [`.env.example`](../.env.example)).
- **From a checkout** (no published package yet): replace the command with
  `"command": "uv", "args": ["run", "--with", "fastmcp", "python", "-m", "cortex.mcp.server"]`
  run from the repo, or build the wheel and `uvx --from <wheel> cortex-mcp`.
- **Tools exposed:** `memorize`, `recall`, `list_memories`, `forget`.
