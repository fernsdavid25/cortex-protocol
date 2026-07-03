"""CortexMemory — the product memory engine (persistent, cross-session, per-user).

This is the engine the MCP server wraps. It is deliberately thin: it composes the
provider abstraction (BYOK embeddings) with the persistent [`SQLiteStore`] and the
benchmark-proven hybrid retrieval (dense cosine + Okapi BM25, RRF-fused).

Design choices that matter:

- **recall does NO generation.** It embeds the query once and returns the most relevant
  *raw* memories. The CLIENT agent (Claude/Cursor/…) synthesises the answer from them.
  This keeps per-call cost to a single embed (negligible) — there is no server-side
  reader/LLM cost, which is what makes self-hosting effectively free (Goal.md I3).
- **memorize does ONE embed** then persists. No LLM call.

So the only runtime cost a self-hoster pays is their own embedding tokens against their
own key. Nothing phones home.
"""

from __future__ import annotations

import logging
import math
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Protocol

from cortex.providers.base import LLMProvider
from cortex.reader.reader import (
    SUPERSEDE_NOOP,
    SUPERSEDE_UPDATE,
    build_episodic_extraction_prompt,
    build_supersession_arbiter_prompt,
    parse_episodic_extraction,
    parse_graph_extraction,
    parse_supersession_verdict,
)
from cortex.retrieve.hybrid import hybrid_retrieve
from cortex.store.sqlite_store import SELF_ALIASES, Memory, SQLiteStore, coerce_metadata

log = logging.getLogger(__name__)


class Reranker(Protocol):
    """Structural type for an injected cross-encoder reranker (duck-typed ``VertexRanker``).

    Declared as a Protocol so ``memory.py`` NEVER imports the Vertex / Discovery Engine SDK: any
    object exposing this ``rerank`` (the real ``VertexRanker``, or a deterministic test fake)
    satisfies it. Keeps the engine offline-safe and the reranker a fully optional, injected dep.
    """

    def rerank(self, query: str, items: Sequence[tuple[str, str]], top_k: int) -> list[str]: ...


# L5 anti-saturation: the lower cosine bound of the "related but not a duplicate" band. Below this,
# a new memory is treated as unrelated (a plain ADD, no arbiter call); between this and
# ``dedup_threshold`` it is a contradiction candidate worth one cheap arbiter call.
_SOFT_UPDATE_FLOOR = 0.83
# How many nearest neighbours to consider when picking a live soft-update candidate.
_NEIGHBOR_POOL = 10

# G2 write-time graph: the ego/first-person tokens that resolve to the synthetic ``self`` entity
# rather than a real entity row. The extractor is instructed to emit the literal ``self``; these
# aliases catch the common variants defensively (case-insensitive) so a stray "I"/"me"/"my" never
# mints a spurious node. Defined in the store module (SINGLE SOURCE OF TRUTH) so the write path here
# and the read path (``get_entity_by_name``) agree on exactly which tokens are the self root.
_SELF_ALIASES = SELF_ALIASES


def _norm_entity_name(name: str) -> str:
    """Normalise an entity name the way the store dedups it (lowercase, collapse whitespace)."""
    return " ".join(name.lower().split())


def _now_iso() -> str:
    """Current UTC time as a stable ISO-8601 string (seconds precision)."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; 0.0 if either vector is empty or zero (local copy, no store internals)."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _embed_signature(provider: LLMProvider) -> tuple[str, int]:
    """Best-effort (model, dim) for a provider, so the store can guard mismatches."""
    model = str(getattr(provider, "embed_model", None) or getattr(provider, "name", "unknown"))
    dim = int(getattr(provider, "embed_dim", None) or getattr(provider, "dim", 0) or 0)
    return model, dim


class CortexMemory:
    """Persistent memory engine: memorize / recall / list / forget for one user."""

    def __init__(
        self,
        provider: LLMProvider,
        store: SQLiteStore,
        user_id: str = "local",
        top_k: int = 5,
        extractor: LLMProvider | None = None,
        use_episodic: bool = False,
        use_graph: bool = False,
        use_dedup: bool = False,
        dedup_threshold: float = 0.95,
        use_soft_update: bool = False,
        arbiter: LLMProvider | None = None,
        reranker: Reranker | None = None,
        rerank_pool: int = 60,
    ) -> None:
        self.provider = provider
        self.store = store
        self.user_id = user_id
        self.top_k = top_k
        # L4 episodic (additive, OFF by default): with ``use_episodic`` false or no ``extractor``,
        # memorize/recall run the exact pre-episodic path — zero extra calls, byte-identical.
        self.extractor = extractor
        self.use_episodic = use_episodic
        # G2 write-time entity graph (additive, OFF by default). With ``use_graph`` false (or no
        # ``extractor``) memorize writes NO entity/edge/link rows and is byte-identical to today. It
        # reuses the SAME folded extraction the episodic path already makes, so enabling both spends
        # a single flash-lite call per memorize. Recall never reads these tables.
        self.use_graph = use_graph
        # L5 anti-saturation (additive, OFF by default). With BOTH ``use_dedup`` and
        # ``use_soft_update`` false, memorize AND recall run the exact pre-L5 path — no neighbour
        # scan, no arbiter call, and recall never queries ``superseded_ids`` (byte-identical). Dedup
        # is embedding-only (no LLM); soft-update spends ONE cheap ``arbiter`` call only inside the
        # "related but not duplicate" band and only when an ``arbiter`` is configured.
        self.use_dedup = use_dedup
        self.dedup_threshold = dedup_threshold
        self.use_soft_update = use_soft_update
        self.arbiter = arbiter
        # Hosted cross-encoder reranker (additive, OFF by default). With ``reranker`` None, recall
        # runs the EXACT pre-rerank path — same retrieval depth (``k``), no reorder, byte-identical.
        # When set, recall over-retrieves a POOL of ``max(rerank_pool, k)`` candidates and asks the
        # reranker to select the top ``k``; ANY reranker failure degrades to the RRF top-``k`` pool,
        # so recall can never fail because of reranking.
        self.reranker = reranker
        self.rerank_pool = rerank_pool
        # The embedding-signature guard is enforced LAZILY (on memorize/recall), NOT here, so a
        # model/dim mismatch can never brick the whole server on startup — list_memories and
        # forget stay usable to inspect and recover the store.
        self._embed_model, self._embed_dim = _embed_signature(provider)

    def _embed(self, text: str, what: str) -> list[float]:
        """Verify the embedding signature, embed one text, and reject an empty/wrong-dim result.

        The signature guard runs here (not in __init__) so a model/dim mismatch only blocks
        embed-dependent ops, leaving list/forget usable to recover. A partial provider failure
        can return no vector; storing/querying an empty/short vector silently corrupts
        retrieval, so we fail loudly instead.
        """
        self.store.ensure_embedding(self._embed_model, self._embed_dim)
        vectors = self.provider.embed([text]).vectors
        vec = list(vectors[0]) if vectors else []
        if not vec or (self._embed_dim and len(vec) != self._embed_dim):
            raise ValueError(
                f"{what} embedding failed: expected a non-empty "
                f"{self._embed_dim or '?'}-dim vector, got {len(vec)} values."
            )
        return vec

    def memorize(self, content: str, metadata: Mapping[str, object] | None = None) -> Memory:
        """Embed and persist a single memory; return the stored record.

        Raises ``ValueError`` on empty content, an embedding-signature mismatch, or a failed
        (empty/wrong-dim) embedding — never stores a corrupt vector.
        """
        content = content.strip()
        if not content:
            raise ValueError("cannot memorize empty content")
        meta = coerce_metadata(metadata)  # validate JSON-safety before spending an embed call
        vec = self._embed(content, "memory")
        memory = Memory(
            id=uuid.uuid4().hex,
            user_id=self.user_id,
            content=content,
            created_at=_now_iso(),
            metadata=meta,
        )
        # L5 anti-saturation — additive and OFF by default. Only when a flag is on AND the store
        # supports supersessions do we scan the user's neighbours; otherwise this whole block is
        # skipped and the path below is byte-identical to pre-L5. ``absorbed`` is a pre-existing
        # memory the write folds into (dedup/NOOP → no new row); ``supersede_old_id`` is the id the
        # new memory replaces (soft-update UPDATE → insert new row + record the supersession).
        supersede_old_id: str | None = None
        if (self.use_dedup or self.use_soft_update) and hasattr(self.store, "add_supersession"):
            absorbed, supersede_old_id = self._antisaturation(memory, vec)
            if absorbed is not None:
                return absorbed
        self.store.add(memory, vec)
        if supersede_old_id is not None:
            # UPDATE verdict: the new memory supersedes the old value so recall returns only the
            # latest. Best-effort — a failure here (e.g. a stale/bad id) must never break memorize.
            try:
                self.store.add_supersession(supersede_old_id, memory.id, memory.created_at)
            except Exception:
                log.warning("supersession record failed for memory %s; continuing", memory.id[:8])
        # L4 episodic + G2 graph write-time enrichment — additive and OFF by default. BOTH features
        # derive from ONE shared ``extractor.generate`` call (the folded flash-lite extraction), so
        # a memorize with both on still spends a single aux call. When neither flag is on (or no
        # extractor is configured) this whole block is skipped and the path above is byte-identical
        # to the pre-enrichment engine. Any failure is swallowed: enrichment must NEVER make
        # memorize fail or change the stored memory.
        self._enrich(memory)
        return memory

    def _antisaturation(
        self, memory: Memory, vec: Sequence[float]
    ) -> tuple[Memory | None, str | None]:
        """Decide how L5 should absorb ``memory`` before it is inserted.

        Returns ``(absorbed, supersede_old_id)``:

        - ``(existing, None)`` — a duplicate (dedup) or redundant restatement (arbiter NOOP): the
          caller returns ``existing`` and inserts NOTHING, bounding store growth.
        - ``(None, old_id)`` — a contradiction the arbiter ruled UPDATE: the caller inserts the new
          row and records that ``old_id`` is now superseded (so recall returns only the new value).
        - ``(None, None)`` — a plain ADD (unrelated, or no arbiter): the caller inserts normally.
        """
        related = self._related_memories(vec)
        if not related:
            return None, None
        top_mem, top_cos = related[0]
        if self.use_dedup and top_cos >= self.dedup_threshold:
            return top_mem, None  # near-identical to an existing memory → skip the write
        if self.use_soft_update and self.arbiter is not None:
            superseded = self.store.superseded_ids(self.user_id)
            candidate = next(
                (
                    m
                    for m, c in related
                    if _SOFT_UPDATE_FLOOR <= c < self.dedup_threshold and m.id not in superseded
                ),
                None,
            )
            if candidate is not None:
                verdict, old_id = self._arbitrate(memory, candidate)
                if verdict == SUPERSEDE_NOOP:
                    return candidate, None
                if verdict == SUPERSEDE_UPDATE:
                    return None, old_id
        return None, None

    def _related_memories(self, vec: Sequence[float]) -> list[tuple[Memory, float]]:
        """Return this user's nearest existing memories to ``vec`` as ``(Memory, cosine)``, desc.

        Reuses the store's existing ``build_index`` + dense search (works for both SQLite and
        Postgres). At personal scale this brute-force scan is the same one recall already runs.
        """
        index, by_id = self.store.build_index(self.user_id)
        if len(index) == 0:
            return []
        out: list[tuple[Memory, float]] = []
        for chunk in index.dense_search(vec, _NEIGHBOR_POOL):
            mem = by_id.get(chunk.session_id)
            if mem is not None:
                out.append((mem, _cosine(vec, chunk.embedding)))
        return out

    def _arbitrate(self, memory: Memory, candidate: Memory) -> tuple[str, str | None]:
        """Run ONE cheap arbiter call to classify ``memory`` vs ``candidate``. NEVER raises.

        Returns ``(verdict, supersede_old_id)``. On UPDATE the id is bound to the ``candidate`` we
        actually showed the arbiter (never a free-form id it invents) — so a supersession can only
        ever target a real, user-scoped memory. Any failure falls back to a plain ADD.
        """
        arbiter = self.arbiter
        if arbiter is None:  # pragma: no cover — guarded by the caller; narrows for typing
            return "ADD", None
        try:
            prompt = build_supersession_arbiter_prompt(
                memory.content, candidate.id, candidate.content, memory.created_at
            )
            result = arbiter.generate(prompt, temperature=0.0, max_output_tokens=128)
            parsed = parse_supersession_verdict(result.text)
            verdict = parsed["verdict"] or "ADD"
            # Bind the supersession to the candidate we actually showed the arbiter — never a
            # free-form id it might invent — so a supersession can only target a real, user-scoped
            # memory. (The parsed ``supersedes_id`` confirms intent; the write uses the candidate.)
            return verdict, candidate.id
        except Exception:
            log.warning("soft-update arbiter failed for memory %s; adding plainly", memory.id[:8])
            return "ADD", None

    def _enrich(self, memory: Memory) -> None:
        """Run the ONE shared write-time extraction and fan out to episodic + graph (best-effort).

        A single ``extractor.generate`` call feeds BOTH features. The block is skipped entirely when
        no extractor is configured or neither feature is enabled/supported by the store — so
        memorize is byte-identical to the pre-enrichment path. The generate call and each downstream
        store are individually guarded: an extraction or persistence failure NEVER makes memorize
        fail or change the stored memory. Graph fields are requested only when ``use_graph`` is on,
        so the episodic-only path (``use_graph`` off) keeps its exact prompt + token budget.
        """
        extractor = self.extractor
        if extractor is None:
            return
        want_episodic = self.use_episodic and hasattr(self.store, "add_event")
        want_graph = self.use_graph and hasattr(self.store, "ensure_self_entity")
        if not (want_episodic or want_graph):
            return
        try:
            prompt = build_episodic_extraction_prompt(
                memory.content, memory.created_at, include_graph=want_graph
            )
            # 256 tokens (episodic-only) keeps the graph-off budget byte-identical; the graph
            # payload (entities + relations) needs more room, so widen ONLY when graph is requested.
            max_tokens = 512 if want_graph else 256
            text = extractor.generate(prompt, temperature=0.0, max_output_tokens=max_tokens).text
        except Exception:
            log.warning("write-time extraction failed for memory %s; continuing", memory.id[:8])
            return
        if want_episodic:
            self._store_event(memory, text)
        if want_graph:
            self._store_graph(memory, text)

    def _store_event(self, memory: Memory, text: str) -> None:
        """Parse the episodic fields from the shared extraction and write the ``events`` row.

        Best-effort: any parse/write error is logged and swallowed so episodic enrichment can never
        fail memorize. Byte-identical to the pre-graph episodic write (same parser, same row).
        """
        try:
            parsed = parse_episodic_extraction(text, fallback_event_time=memory.created_at)
            self.store.add_event(
                memory_id=memory.id,
                user_id=memory.user_id,
                event_time=parsed["event_time"],
                ingest_time=memory.created_at,
                actor=parsed["actor"],
                location=parsed["location"],
                event_type=parsed["event_type"],
            )
        except Exception:
            log.warning("episodic extraction failed for memory %s; continuing", memory.id[:8])

    def _store_graph(self, memory: Memory, text: str) -> None:
        """Parse the graph half of the extraction and persist it for ``memory`` (best-effort).

        Resolves the ego ``self`` root, upserts every extracted entity (self aliases collapse to the
        self node and are NEVER a row), adds each labeled directed relation tagged with the source
        ``memory.id``, then links the memory to its subject entity (role=subject) and every other
        mentioned entity (role=mention). Wrapped whole in try/except: a bad extraction must NEVER
        fail memorize or touch the stored memory.
        """
        store = self.store
        try:
            parsed = parse_graph_extraction(text)
            self_id = store.ensure_self_entity(self.user_id)
            # norm_name -> entity_id cache (self aliases collapse to self_id, never a real row)
            resolved: dict[str, str] = {}
            # norm_name -> declared type, so a relation endpoint upserts with its stated type
            declared: dict[str, str] = {}
            for ent in parsed["entities"]:
                norm = _norm_entity_name(ent["name"])
                if norm and norm not in _SELF_ALIASES:
                    declared[norm] = ent["type"]

            def resolve(raw: str) -> str | None:
                norm = _norm_entity_name(raw)
                if not norm:
                    return None
                if norm in _SELF_ALIASES:
                    return self_id
                eid = resolved.get(norm)
                if eid is None:
                    etype = declared.get(norm, "thing")
                    eid = store.upsert_entity(self.user_id, raw.strip(), etype)
                    resolved[norm] = eid
                return eid

            mentioned: set[str] = set()
            # upsert every declared entity (linked as a mention even without an explicit relation)
            for ent in parsed["entities"]:
                eid = resolve(ent["name"])
                if eid is not None and eid != self_id:
                    mentioned.add(eid)
            for rel in parsed["relations"]:
                src_id, dst_id = resolve(rel["src"]), resolve(rel["dst"])
                if src_id is None or dst_id is None:
                    continue
                store.add_entity_edge(self.user_id, src_id, rel["label"], dst_id, memory.id)
                mentioned.update(e for e in (src_id, dst_id) if e != self_id)
            subject = parsed["subject"]
            subject_id = self_id
            if subject and _norm_entity_name(subject) not in _SELF_ALIASES:
                subject_id = resolve(subject) or self_id
            store.link_memory_entity(self.user_id, memory.id, subject_id, "subject")
            for eid in mentioned:
                if eid != subject_id:
                    store.link_memory_entity(self.user_id, memory.id, eid, "mention")
        except Exception:
            log.warning("graph extraction failed for memory %s; continuing", memory.id[:8])

    def recall(self, query: str, limit: int | None = None) -> list[Memory]:
        """Return the memories most relevant to ``query`` (hybrid dense+lexical, RRF).

        Embeds the query once; no generation. Returns ``[]`` for an empty query, an empty
        store, or ``limit <= 0``.
        """
        query = query.strip()
        if not query:
            return []
        k = self.top_k if limit is None else limit
        if k <= 0:
            return []
        qvec = self._embed(query, "query")
        # With a reranker configured, over-retrieve a deeper POOL so the cross-encoder has real
        # candidates to reorder; with no reranker ``depth == k`` and every line below is exactly the
        # pre-rerank path (same search/hybrid call, same superseded filter, same return) — the
        # ``_rerank`` call is then a no-op that returns the pool unchanged, so recall is
        # byte-identical to today.
        depth = k if self.reranker is None else max(self.rerank_pool, k)
        # Prefer a store-side hybrid pushdown (Postgres/pgvector: dense <=> + FTS, fused in-DB) —
        # lower latency + no full-store load. Stores without it (SQLite) use the in-memory path.
        search = getattr(self.store, "search", None)
        if callable(search):
            pool = self._filter_superseded(search(self.user_id, query, qvec, depth))
            return self._rerank(query, pool, k)
        index, by_id = self.store.build_index(self.user_id)
        if len(index) == 0:
            return []
        chunks = hybrid_retrieve(index, query, qvec, depth)
        out: list[Memory] = []
        for chunk in chunks:
            mem = by_id.get(chunk.session_id)
            if mem is not None:
                out.append(mem)
        return self._rerank(query, self._filter_superseded(out), k)

    def _rerank(self, query: str, pool: list[Memory], k: int) -> list[Memory]:
        """Reorder an RRF-ordered candidate ``pool`` with the cross-encoder, keeping the top ``k``.

        With no reranker configured this returns ``pool`` UNCHANGED — the load-bearing guard for the
        byte-identical no-reranker path (recall neither over-fetches nor reorders). When a reranker
        is set, it scores every ``(id, content)`` candidate and returns the top ``k`` in its order;
        any ids it drops are backfilled in RRF order so a short result still yields a full ``k``. On
        ``VertexRerankError`` or ANY exception we log and fall back to the pool's RRF top-``k``
        (``pool[:k]``) — recall must never fail because of reranking.
        """
        if self.reranker is None or not pool:
            return pool
        try:
            ranked_ids = self.reranker.rerank(query, [(m.id, m.content) for m in pool], top_k=k)
        except Exception:  # noqa: BLE001 — VertexRerankError or anything else → graceful fallback
            log.warning("rerank failed for user %s; falling back to RRF top-%d", self.user_id, k)
            return pool[:k]
        by_id = {m.id: m for m in pool}
        ranked = [by_id[i] for i in ranked_ids if i in by_id]
        if len(ranked) < k:  # backfill dropped/unknown ids in the original RRF order (no dupes)
            seen = {m.id for m in ranked}
            ranked.extend(m for m in pool if m.id not in seen)
        return ranked[:k]

    def _filter_superseded(self, memories: list[Memory]) -> list[Memory]:
        """Drop superseded memories so recall returns the LATEST value — GATED on anti-saturation.

        The FIRST guard is load-bearing for the byte-identical guarantee: with BOTH L5 flags off we
        return the input list UNCHANGED and NEVER query ``superseded_ids`` — recall is exactly the
        pre-L5 result. Only when dedup/soft-update is active (and the store supports it) do we
        consult the supersession set and filter.
        """
        if not (self.use_dedup or self.use_soft_update):
            return memories
        if not hasattr(self.store, "superseded_ids"):
            return memories
        superseded = self.store.superseded_ids(self.user_id)
        if not superseded:
            return memories
        return [m for m in memories if m.id not in superseded]

    def list_memories(self, limit: int = 20) -> list[Memory]:
        """Return the most recently added memories first."""
        return self.store.list_recent(self.user_id, limit)

    def timeline(
        self, since: str | None = None, until: str | None = None, limit: int = 50
    ) -> list[Memory]:
        """Return this user's episodic memories ordered by event_time (L4; additive).

        Delegates to ``store.timeline``. Returns ``[]`` for a store that predates episodic
        support, so callers never break on the pre-L4 store surface.
        """
        timeline_fn = getattr(self.store, "timeline", None)
        if not callable(timeline_fn):
            return []
        results: list[Memory] = timeline_fn(self.user_id, since=since, until=until, limit=limit)
        return results

    def forget(self, memory_id: str) -> bool:
        """Delete one memory by exact id; return True if it existed."""
        return self.store.delete(memory_id, self.user_id)

    def forget_prefix(self, memory_id: str) -> str:
        """Delete by full id OR an unambiguous short-id prefix; return a status message.

        Refuses to delete when a short prefix matches MULTIPLE memories — prevents a short id
        from silently deleting the wrong memory (data loss). Read-only resolution is SQL-side
        and bounded (no full-table scan).
        """
        target = memory_id.strip()
        if not target:
            return "No memory id given."
        if len(target) < 32:  # a short id (full uuid4 hex is 32 chars)
            matches = self.store.resolve_id_prefix(self.user_id, target, limit=2)
            if not matches:
                return f"No memory matched {memory_id!r}."
            if len(matches) > 1:
                return (
                    f"Ambiguous id {memory_id!r}: it matches multiple memories — "
                    "use the full id from list_memories."
                )
            target = matches[0]
        ok = self.forget(target)
        return f"Forgot {target[:8]}." if ok else f"No memory matched {memory_id!r}."

    def count(self) -> int:
        """Total memories stored for this user."""
        return self.store.count(self.user_id)

    def close(self) -> None:
        """Close the underlying store (flush SQLite); safe to call once at shutdown."""
        self.store.close()
