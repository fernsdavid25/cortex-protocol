# API reference

Auto-generated from the source docstrings. Cortex is a small, embeddable library: you compose
a [provider](#llmprovider) (BYOK embeddings) with a store and get the
[`CortexMemory`](#cortexmemory) engine. To plug in your own model backend, implement the
`LLMProvider` contract below — the engine only ever programs against this interface.

## CortexMemory

The persistent memory engine the MCP server wraps: `memorize` / `recall` / `list_memories` /
`timeline` / `forget` / `count` for one user.

::: cortex.memory.CortexMemory

## LLMProvider

The provider abstraction (BYOK). Subclass this to add a new model backend — implement
`generate` and `embed`, returning the result types below. See
[`examples/custom_provider.py`](https://github.com/fernsdavid25/cortex-protocol/blob/main/examples/custom_provider.py)
for a minimal, runnable subclass.

::: cortex.providers.base.LLMProvider

## Result types

The value objects a provider returns. Both carry token counts for cost accounting.

::: cortex.providers.base.GenResult

::: cortex.providers.base.EmbedResult

## Memory

One persisted memory record and its provenance — the record `memorize` returns and `recall`
ranks.

::: cortex.store.sqlite_store.Memory
