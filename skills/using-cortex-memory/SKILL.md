---
name: using-cortex-memory
description: Use when connected to a Cortex memory MCP server (tools named memorize, recall, recall_about, recall_timeline, list_memories, forget) — at session start, before asking the user for facts they may have shared before, after learning durable facts/preferences/decisions, when the user says "remember/recall/forget", or when deciding which memory tool fits a question.
---

# Using Cortex Memory — get maximum value from the user's memory

Cortex is the USER'S portable, cross-agent memory — you are one of several AIs reading and
writing the same store, with their consent. Write for a reader that isn't you; recall before
you assume; the user's memory outlives this session and this agent.

## How Cortex works (why the rules below exist)

- **Recall is hybrid search with NO LLM:** dense embeddings + keyword search fused by rank.
  Your query text is the entire retrieval signal — names, distinctive words, and paraphrases of
  meaning all help; no model on the server interprets vague queries. Recall is one embedding
  call: fast and cheap, so use it liberally.
- **Enrichment happens at WRITE time:** when you memorize, Cortex extracts entities and
  relationships into a knowledge graph, resolves dates onto an event timeline, deduplicates
  near-identical facts, and supersedes contradicted ones (old versions are kept, marked
  outdated — never silently lost). The quality of all of that depends on how you phrase what
  you store.
- **The tools are specialized reads over one store** — routing correctly is most of this skill.

## Routing — pick the right tool for the question

| Situation | Tool |
|---|---|
| Specific question ("what stack does she prefer?") | `recall` — phrase with names + meaning |
| "Tell me everything about X" (person/project/place) | `recall_about` — exhaustive entity dossier (relations + attached memories); top-k `recall` structurally can't enumerate |
| "What happened last month / between dates / when did…" | `recall_timeline` — queries true event time, not storage time |
| "What's recently stored?" / orientation | `list_memories` |
| User states something durable / corrects a fact | `memorize` (supersession is automatic) |
| User asks to delete / data is wrong or sensitive | `forget` — immediately, no debate |

## Writing memories that retrieve well (the craft)

1. **One fact per memorize.** Bundled paragraphs dedupe badly, embed vaguely, and can't be
   individually superseded or forgotten. Split compound facts.
2. **Self-contained phrasing:** resolve every pronoun and implicit reference — "David prefers
   pnpm over npm for all projects", never "he prefers it". Another agent with zero context is
   the reader.
3. **Name the entities explicitly** (people, projects, places, orgs): names power the graph
   extraction, the entity dossiers, AND the keyword half of retrieval. "Swizel (David's
   girlfriend) likes apples" builds graph edges; "she likes apples" builds nothing.
4. **Anchor time absolutely:** "on 2026-07-05" or "in March 2026". Relative phrases
   ("yesterday") are resolved against the write date, so they work — but absolute dates are
   safer and survive re-telling.
5. **Store the durable, skip the transient:** preferences, profile facts, decisions and their
   reasons, relationships, recurring context, important events. Do NOT store: this-session task
   state, secrets/credentials/API keys (never — memory is not a vault), huge pasted content
   (store the fact plus where the content lives), or trivia the user wouldn't expect remembered.
6. **Updates are new writes:** when a fact changes ("I switched to pnpm"), memorize the new
   state — Cortex supersedes the old version automatically. Use `forget` only for deletion
   requests and wrong/sensitive data, not for routine updates.
7. **Classify with `kind`** — one of `preference`, `fact`, `project`, `instruction`, `event`,
   `relationship` (unset defaults to `fact`). It improves organization and the user's dashboard
   view; add `tags` when a natural grouping exists.

## Session protocol

- **Start:** one broad `recall` for who the user is plus context for the current topic (e.g.
  "David preferences current projects"). Cheap insurance against re-asking known things.
- **Before asking the user anything factual about themselves or their history: recall first.**
  Asking what memory already knows wastes their time and makes the memory look decorative.
- **During:** when the user shares something durable, memorize it at that moment, phrased per
  the craft rules — don't batch for session end; sessions get cut off.
- **When answering FROM memory, say so** naturally ("you mentioned before that…"). If recall
  returns nothing relevant, say you don't have it rather than inventing — absence of memory is
  a valid, honest answer.
- **Corrections are gold:** when the user contradicts a recalled memory, memorize the
  correction immediately. A memory that self-corrects earns trust; one that repeats stale
  facts loses it.

## Query craft (when the first recall misses)

Retry once with different words before concluding absence: add the entity's name; swap
abstraction level ("code editor" ↔ "VS Code"); try the user's likely original phrasing (keyword
matching rewards their exact words). Two misses with good queries = genuinely not stored — ask
the user, then memorize the answer so the next agent doesn't have to ask.

## Self-hosted Cortex note

On self-hosted (OSS) servers, `recall_about` and `recall_timeline` depend on opt-in enrichment
flags (`CORTEX_GRAPH=1`, `CORTEX_EPISODIC=1`). Check which tools your connection exposes; if
they're absent or return an "extraction disabled" notice, route those questions through plain
`recall` and suggest the user enable the flags. Hosted Cortex has everything enabled.

## Respect the ownership model

This memory belongs to the human, not to you or your app: don't store things they'd be
surprised to find; honor forget requests instantly; when they ask "what do you know about me?",
show them plainly via `list_memories`/`recall_about`. Trust in the memory IS the product.
