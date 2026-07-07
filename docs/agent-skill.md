# The Cortex agent skill

Connecting the MCP server gives your agent the *tools*; this skill gives it the *craft* — how to
phrase memories so the knowledge graph and timeline actually build, which tool answers which kind
of question, and when to recall before asking you things you've already said. Agents with the
skill get meaningfully more out of the same memory.

## Install

**Claude Code:** copy the skill folder into your skills directory —

```bash
mkdir -p ~/.claude/skills/using-cortex-memory
curl -fsSL https://raw.githubusercontent.com/fernsdavid25/cortex-protocol/main/skills/using-cortex-memory/SKILL.md \
  -o ~/.claude/skills/using-cortex-memory/SKILL.md
```

It auto-loads whenever Cortex tools are in play, or invoke it with `/using-cortex-memory`.

**Any other agent** (Cursor rules, ChatGPT custom instructions, a system prompt): paste the
skill body below — the guidance is client-agnostic.

The canonical file lives at
[`skills/using-cortex-memory/SKILL.md`](https://github.com/fernsdavid25/cortex-protocol/blob/main/skills/using-cortex-memory/SKILL.md);
this page embeds it verbatim.

---

--8<-- "skills/using-cortex-memory/SKILL.md"
