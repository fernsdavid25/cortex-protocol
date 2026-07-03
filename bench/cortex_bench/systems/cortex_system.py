"""Cortex engine v0: in-memory store + hybrid (dense+lexical RRF) retrieval + CoN reader.

The cheap-by-default Cortex system. Each haystack session is chunked into ROUNDS,
all rounds are embedded in ONE provider batch, and stored in an in-memory store.
At query time we embed the question once, fuse dense + lexical retrieval with RRF,
and read the top_k memories with a Chain-of-Note, calibrated-abstention prompt.

When ``use_fact_keys`` is on, ingest also distils EACH session into a few atomic
user-fact / keyphrase strings (one generate() per session) and stores them as EXTRA
memory chunks beside the raw rounds. These fact keys are the LongMemEval
"session-userfact" key-expansion lever: they give retrieval additional, denser keys
to match a question against (~+5% QA) while the raw rounds are kept intact.

When ``use_query_distill`` is on, the answer step makes ONE extra generate() that distils
the question-relevant facts from the *already-retrieved* memories and prepends them to the
reader context. This is the cheap, query-time counterpart to fact-keys: it targets the same
"clean facts help the reader" benefit but at ~1 generate per QUERY instead of one per
SESSION at ingest (≈40× fewer calls on a typical _S haystack) — the accuracy-per-dollar
lever. The two flags are independent and may be combined.

Temporal indexing, a query classifier, and reranking are LEFT for later phases.
"""

from __future__ import annotations

import re

from cortex.providers.base import LLMProvider
from cortex.reader.reader import (
    ABSTAIN_SENTINEL,
    REFLECTION_SENTINEL,
    build_answerability_gate_prompt,
    build_reader_prompt,
    build_recommendation_prompt,
    build_reflection_prompt,
    build_rerank_prompt,
    format_chunks,
    gate_says_unanswerable,
    is_recommendation_question,
    parse_rerank_order,
)
from cortex.retrieve.hybrid import hybrid_retrieve
from cortex.store.memory_store import InMemoryStore, MemoryChunk

from ..memory_system import MemorySystem, QAInstance, Usage
from ._common import rounds

# Cap facts per session so the extra key chunks stay cheap and on-topic.
MAX_FACTS_PER_SESSION = 6

FACT_EXTRACTION_PROMPT = (
    "Extract the key atomic facts about the user and the most important keyphrases "
    "from this conversation session. Return a short plain list, one fact per line, "
    f"at most {MAX_FACTS_PER_SESSION} items, no numbering, no extra prose.\n\n"
    "Session:\n"
)

QUERY_DISTILL_PROMPT = (
    "From the retrieved memories below, extract ONLY the facts relevant to answering the "
    "question. Quote dates, names, and numbers exactly as written. Reply with a short "
    "bullet list and nothing else; if no memory is relevant, reply exactly 'NONE'.\n\n"
    "Question: {question}\n\n"
    "Retrieved memories:\n{memories}\n\nRelevant facts:"
)


def _parse_facts(text: str, cap: int = MAX_FACTS_PER_SESSION) -> list[str]:
    """Robustly parse a model fact list (newline / numbered / bulleted) into strings.

    Strips list markers (``1.``, ``-``, ``*``, ``•``) and surrounding whitespace,
    drops empties, de-duplicates while preserving order, and caps the count.
    """
    facts: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Strip a leading list marker: "1.", "1)", "-", "*", "•".
        line = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", line).strip()
        if not line:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        facts.append(line)
        if len(facts) >= cap:
            break
    return facts


class CortexSystem(MemorySystem):
    """Hybrid-retrieval memory system designed to beat the naive baselines."""

    name = "cortex-v0"

    def __init__(
        self,
        provider: LLMProvider,
        top_k: int = 10,
        max_output_tokens: int = 256,
        use_fact_keys: bool = False,
        use_query_distill: bool = False,
        use_preference_mode: bool = False,
        use_reflection: bool = False,
        use_answer_first: bool = False,
        use_answerability_gate: bool = False,
        use_strict_abstain: bool = False,
        use_rerank: bool = False,
        rerank_k: int = 25,
        rerank_backend: str = "listwise",
        aux_provider: LLMProvider | None = None,
    ) -> None:
        self.provider = provider
        # Fact extraction + query distillation are cheap prep work, not reasoning — run them on
        # a cheaper model (aux_provider) so a strong/expensive reader isn't invoked dozens of
        # times per question at ingest. Falls back to the main provider when not supplied.
        self.aux = aux_provider or provider
        self.top_k = top_k
        self.max_output_tokens = max_output_tokens
        self.use_fact_keys = use_fact_keys
        self.use_query_distill = use_query_distill
        self.use_preference_mode = use_preference_mode
        self.use_reflection = use_reflection
        self.use_answer_first = use_answer_first
        self.use_answerability_gate = use_answerability_gate
        self.use_strict_abstain = use_strict_abstain
        self.use_rerank = use_rerank
        self.rerank_k = rerank_k
        # Reranker backend selector (only consulted when use_rerank is on): "listwise" = the cheap
        # aux-LLM prompt/parse reranker that FAILED the +1pt gate. The "vertex"/"vertex-ranking"
        # cross-encoder backend is a flagship-only path and is NOT bundled with the open-source
        # build; when selected here it degrades to the RRF top-k order (see ``_rerank``). Byte-
        # identical to the off behaviour when use_rerank is False (the reranker is never invoked).
        self.rerank_backend = rerank_backend
        self.store = InMemoryStore()
        self._last_retrieved: list[MemoryChunk] = []

    def reset(self) -> None:
        self.store.reset()
        self._last_retrieved = []

    def ingest(self, instance: QAInstance) -> Usage:
        texts: list[str] = []
        sids: list[str] = []
        dates: list[str] = []
        for sid, date, session in zip(
            instance.haystack_session_ids,
            instance.haystack_dates,
            instance.haystack_sessions,
            strict=False,
        ):
            for chunk_text in rounds(session):
                texts.append(chunk_text)
                sids.append(sid)
                dates.append(date)
        if not texts:
            return Usage()

        usage = Usage()
        res = self.provider.embed(texts)  # ONE batched embed call for all rounds
        usage.embed_tokens += res.input_tokens
        chunks = [
            MemoryChunk(text=t, session_id=s, date=d, embedding=vec)
            for t, s, d, vec in zip(texts, sids, dates, res.vectors, strict=False)
        ]
        self.store.add(chunks)

        if self.use_fact_keys:
            usage = usage + self._ingest_fact_keys(instance)

        return usage

    def _ingest_fact_keys(self, instance: QAInstance) -> Usage:
        """Distil each session into atomic fact keys and store them as extra chunks.

        One generate() call per session extracts the facts; the facts are embedded
        (counted into ``embed_tokens``) and stored as additional MemoryChunks carrying
        that session_id, giving retrieval extra fact-based keys alongside raw rounds.
        """
        usage = Usage()
        fact_texts: list[str] = []
        fact_sids: list[str] = []
        fact_dates: list[str] = []
        for sid, date, session in zip(
            instance.haystack_session_ids,
            instance.haystack_dates,
            instance.haystack_sessions,
            strict=False,
        ):
            session_text = "\n".join(
                f"{turn.get('role', '')}: {turn.get('content', '')}" for turn in session
            )
            r = self.aux.generate(
                FACT_EXTRACTION_PROMPT + session_text,
                max_output_tokens=self.max_output_tokens,
            )
            usage.input_tokens += r.input_tokens
            usage.output_tokens += r.output_tokens
            for fact in _parse_facts(r.text):
                fact_texts.append(fact)
                fact_sids.append(sid)
                fact_dates.append(date)

        if not fact_texts:
            return usage

        fres = self.provider.embed(fact_texts)
        usage.embed_tokens += fres.input_tokens
        fact_chunks = [
            MemoryChunk(text=t, session_id=s, date=d, embedding=vec)
            for t, s, d, vec in zip(fact_texts, fact_sids, fact_dates, fres.vectors, strict=False)
        ]
        self.store.add(fact_chunks)
        return usage

    def _distill_query(self, question: str, chunks: list[MemoryChunk]) -> tuple[str, Usage]:
        """One generate() that distils question-relevant facts from the retrieved chunks.

        The cheap, query-time counterpart to fact-keys: extract once per query from only the
        already-retrieved memories (not once per session at ingest), then hand the clean
        facts to the reader. Returns ``("", usage)`` when nothing is relevant ("NONE").
        """
        usage = Usage()
        prompt = QUERY_DISTILL_PROMPT.format(question=question, memories=format_chunks(chunks))
        r = self.aux.generate(prompt, max_output_tokens=self.max_output_tokens)
        usage.input_tokens += r.input_tokens
        usage.output_tokens += r.output_tokens
        text = r.text.strip()
        if not text or text.upper().startswith("NONE"):
            return "", usage
        return text, usage

    def _reflect(
        self, question: str, question_date: str, chunks: list[MemoryChunk]
    ) -> tuple[str, Usage]:
        """One cheap aux call that digests retrieved memories into a structured reflection.

        Produces a dated TIMELINE, update-resolved CURRENT FACTS, and cross-session TOTALS —
        targeting the residual reader failures (temporal ordering, knowledge-update recency,
        multi-session aggregation). Returns ``("", usage)`` when nothing is relevant.
        """
        usage = Usage()
        prompt = build_reflection_prompt(question, question_date, chunks)
        r = self.aux.generate(prompt, max_output_tokens=self.max_output_tokens)
        usage.input_tokens += r.input_tokens
        usage.output_tokens += r.output_tokens
        text = r.text.strip()
        if not text or text.upper().startswith(REFLECTION_SENTINEL):
            return "", usage
        return text, usage

    def _gate_unanswerable(
        self, question: str, question_date: str, chunks: list[MemoryChunk]
    ) -> tuple[bool, Usage]:
        """One cheap aux call ruling the question ANSWERABLE / UNANSWERABLE from the memories (A2).

        Detect-then-decline: a conservative second opinion, biased toward ANSWERABLE, that owns
        the unanswerable verdict — it declines only when the asked-about fact is genuinely absent,
        recovering the adversarial accuracy a deep-context, answer-first reader loses by
        over-answering. Runs on the aux (cheap) provider. Returns ``(True, usage)`` when it rules
        the question unanswerable (the caller then emits the abstention sentinel without a reader
        call — which also saves the reader cost on those questions).
        """
        usage = Usage()
        prompt = build_answerability_gate_prompt(question, question_date, chunks)
        r = self.aux.generate(prompt, max_output_tokens=self.max_output_tokens)
        usage.input_tokens += r.input_tokens
        usage.output_tokens += r.output_tokens
        return gate_says_unanswerable(r.text), usage

    def _rerank(
        self, question: str, chunks: list[MemoryChunk], top_k: int
    ) -> tuple[list[MemoryChunk], Usage]:
        """Rerank a deep candidate pool down to the ``top_k`` most useful (L2), by backend.

        Dispatches on ``rerank_backend``: the default ("listwise") uses the cheap aux-LLM listwise
        reranker. The ``vertex``/``vertex-ranking`` cross-encoder backend is a flagship-only path
        not bundled with the open-source build, so it degrades to the RRF ``top_k`` order here
        (never dropping the answer). A pool already within budget is returned untouched.
        """
        usage = Usage()
        if len(chunks) <= top_k:
            return list(chunks), usage  # pool already within budget — nothing to prune
        if self.rerank_backend in ("vertex", "vertex-ranking"):
            # Vertex cross-encoder is not part of the OSS build — keep the RRF top-k order.
            return list(chunks[:top_k]), usage
        return self._rerank_listwise(question, chunks, top_k)

    def _rerank_listwise(
        self, question: str, chunks: list[MemoryChunk], top_k: int
    ) -> tuple[list[MemoryChunk], Usage]:
        """Listwise-rerank via one cheap aux call reordering the pool by usefulness (L2).

        We keep the model's picks first, then backfill any it omitted in the original RRF order, and
        truncate to ``top_k``. The backfill guarantees ``top_k`` chunks and makes a garbled/empty
        rerank degrade gracefully to "RRF top-k" rather than dropping the answer. Runs on the aux
        (cheap) provider; shrinking the reader context from the pool to ``top_k`` offsets the call.
        """
        usage = Usage()
        prompt = build_rerank_prompt(question, chunks)
        r = self.aux.generate(prompt, max_output_tokens=self.max_output_tokens)
        usage.input_tokens += r.input_tokens
        usage.output_tokens += r.output_tokens
        order = parse_rerank_order(r.text, len(chunks))
        picked_idx = set(order)
        picked = [chunks[i] for i in order]
        backfill = [c for j, c in enumerate(chunks) if j not in picked_idx]  # RRF order preserved
        return (picked + backfill)[:top_k], usage

    def answer(self, instance: QAInstance) -> tuple[str, Usage]:
        qres = self.provider.embed([instance.question])
        usage = Usage(embed_tokens=qres.input_tokens)
        qvec = qres.vectors[0] if qres.vectors else []

        retrieved = hybrid_retrieve(self.store, instance.question, qvec, self.top_k)
        # L2: over-retrieve the deep pool, then rerank down to rerank_k for the reader (precision).
        if self.use_rerank and len(retrieved) > self.rerank_k:
            retrieved, rrusage = self._rerank(instance.question, retrieved, self.rerank_k)
            usage = usage + rrusage
        self._last_retrieved = retrieved

        reader_chunks: list[MemoryChunk] = list(retrieved)
        if self.use_query_distill and retrieved:
            distilled, dusage = self._distill_query(instance.question, retrieved)
            usage = usage + dusage
            if distilled:
                # Prepend a synthetic "distilled facts" block so the reader sees clean,
                # question-focused facts first, with the raw rounds kept as backup context.
                focus = MemoryChunk(
                    text=distilled,
                    session_id="distilled-facts",
                    date=instance.question_date,
                )
                reader_chunks = [focus, *retrieved]

        if self.use_reflection and retrieved:
            digest, rusage = self._reflect(instance.question, instance.question_date, retrieved)
            usage = usage + rusage
            if digest:
                # Prepend the structured reflection (timeline / current facts / totals) so the
                # reader sees a pre-digested view first, with the raw rounds kept as backup.
                focus = MemoryChunk(
                    text="[REFLECTION DIGEST]\n" + digest,
                    session_id="reflection",
                    date=instance.question_date,
                )
                reader_chunks = [focus, *reader_chunks]

        is_reco = self.use_preference_mode and is_recommendation_question(instance.question)

        # A2 detect-then-decline: a conservative gate owns the unanswerable verdict. Skip it for
        # recommendation questions (those never abstain). When it declines, short-circuit to the
        # abstention sentinel WITHOUT the reader call (also saving the reader cost on those Qs).
        if self.use_answerability_gate and reader_chunks and not is_reco:
            unanswerable, gusage = self._gate_unanswerable(
                instance.question, instance.question_date, reader_chunks
            )
            usage = usage + gusage
            if unanswerable:
                return ABSTAIN_SENTINEL, usage

        if is_reco:
            prompt = build_recommendation_prompt(
                instance.question,
                instance.question_date,
                reader_chunks,
                answer_first=self.use_answer_first,
            )
        else:
            prompt = build_reader_prompt(
                instance.question,
                instance.question_date,
                reader_chunks,
                answer_first=self.use_answer_first,
                strict_abstain=self.use_strict_abstain,
            )
        r = self.provider.generate(prompt, max_output_tokens=self.max_output_tokens)
        usage = usage + Usage(input_tokens=r.input_tokens, output_tokens=r.output_tokens)
        return r.text.strip(), usage

    def retrieved_session_ids(self) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for chunk in self._last_retrieved:
            if chunk.session_id not in seen:
                seen.add(chunk.session_id)
                ordered.append(chunk.session_id)
        return ordered
