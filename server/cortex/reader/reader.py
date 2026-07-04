"""Reader prompt builder for Cortex engine v0 (Phase 2).

Chain-of-Note (Yu et al., 2023): the model first NOTES the relevant facts from the
retrieved memories, then answers from those notes — this curbs hallucination and
makes the reasoning auditable.

CALIBRATED abstention: the Phase 1 diagnostic found flash-lite OVER-abstains (it
says "I don't know" when the answer is actually present but phrased differently).
So the instruction here is deliberately asymmetric: answer from the memories, and
reply with the exact sentinel ONLY when the memories genuinely lack the answer —
do NOT abstain merely out of uncertainty.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import TypedDict

from cortex.store.memory_store import MemoryChunk
from cortex.store.sqlite_store import VALID_ENTITY_TYPES, normalize_kind

ABSTAIN_SENTINEL = "I don't know"

# Recommendation / preference questions ("recommend a show", "any tips?", "what should I
# serve?") are NOT factual lookups — the user wants advice grounded in their OWN stated
# preferences. The factual reader wrongly abstains or gives generic advice on these, so we
# detect them by phrasing and route to a preference-aware reader.
_RECO_PATTERNS = re.compile(
    r"\b(recommend|recommendation|suggest|suggestion|any tips|tips (?:for|on)|"
    r"what should i|what do you think|any ideas|ideas for|advice|"
    r"help me (?:decide|choose)|can you (?:recommend|suggest)|what would you)\b",
    re.IGNORECASE,
)


def is_recommendation_question(question: str) -> bool:
    """Heuristically detect a recommendation/preference question (vs a factual lookup)."""
    return bool(_RECO_PATTERNS.search(question))


_ANSWER_POLICY = (
    "Answering policy (read carefully):\n"
    f'- Answer directly from the memories. Reply with exactly "{ABSTAIN_SENTINEL}" ONLY '
    "when the memories genuinely do not contain the answer.\n"
    "- Do NOT abstain merely because you are uncertain, because the wording differs from "
    "the question, or because the answer is implied rather than stated verbatim. If the "
    "needed fact is present in any form, ANSWER it.\n"
    "- Keep the final answer concise.\n"
)

# STRICT-abstain policy (A2, reader-side): same calibration as above, PLUS an explicit
# don't-fabricate clause for the deep-context case. A deep, answer-first reader over-answers
# genuinely-unanswerable (adversarial) questions — with 100 loosely-related memories in view it
# invents a plausible specific detail instead of declining (LoCoMo adversarial −4.1 at k=100).
# This keeps multi-hop synthesis (combining facts that ARE present) but forbids emitting a
# specific name/number/date/place that NO memory states — recovering abstention without a second
# model call (unlike a separate gate, so it preserves accuracy-per-dollar). Toggled by
# ``strict_abstain``.
# EXPERIMENTAL (off by default): the n=100 LoCoMo probe (at the correct 8192 budget) recovered NO
# adversarial questions and hurt temporal (−0.25) — the reader declines answerable questions it
# is unsure of rather than the truly-unanswerable ones. See bench/results/LOCOMO_results.md.
_ANSWER_POLICY_STRICT = (
    "Answering policy (read carefully):\n"
    "- Answer directly from the memories. You MAY combine and reason over multiple memories to "
    "derive the answer — multi-hop chains, counts, and date arithmetic over facts that ARE "
    "present are all valid and encouraged.\n"
    "- Do NOT abstain merely because the wording differs from the question or because the answer "
    "is implied rather than stated verbatim. If the needed fact is present in any form, answer "
    "it.\n"
    "- BUT do NOT answer with a specific name, number, date, place, or detail that NO memory "
    "actually states. If the memories only discuss the general topic and never state the specific "
    "thing the question asks about, that question is unanswerable — reply with exactly "
    f'"{ABSTAIN_SENTINEL}" instead of guessing a plausible-sounding detail.\n'
    "- Keep the final answer concise.\n"
)

# Chain-of-Note head (NOTES → ANSWER) and its answer-first inverse (ANSWER → NOTES). The
# answer-first order keeps a truncated output-budget from ever eating the answer (thinking-capable
# readers reason internally, then lead with the answer) — A1: 69/500 headline answers were
# truncated mid-NOTES before reaching ANSWER at k=50. The policy (calibrated vs strict) is
# appended per-call so ``answer_first`` × ``strict_abstain`` compose freely.
_COT_HEAD = (
    "You are Cortex, answering a question using ONLY the retrieved memories below.\n"
    "\n"
    "Work in two steps:\n"
    "1. NOTES: briefly note the specific facts in the memories that bear on the "
    "question (quote dates, names, and numbers exactly as written).\n"
    "2. ANSWER: give the final answer on its own line, derived only from those notes.\n"
    "\n"
)
_ANSWER_FIRST_HEAD = (
    "You are Cortex, answering a question using ONLY the retrieved memories below.\n"
    "\n"
    "Output in THIS order — the answer FIRST so it is never cut off:\n"
    "1. ANSWER: on the first line, the final concise answer, derived only from the memories.\n"
    "2. NOTES: then cite the specific supporting facts (quote dates, names, and numbers "
    "exactly as written).\n"
    "\n"
)


def format_chunks(chunks: Sequence[MemoryChunk]) -> str:
    """Render retrieved memories as structured, dated, attributed context blocks."""
    if not chunks:
        return "(no memories retrieved)"
    blocks: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"[Memory {i} | session {chunk.session_id} | {chunk.date}]"
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)


def build_reader_prompt(
    question: str,
    question_date: str,
    chunks: Sequence[MemoryChunk],
    answer_first: bool = False,
    strict_abstain: bool = False,
) -> str:
    """Build the Chain-of-Note reader prompt with calibrated abstention.

    The question date is surfaced so the reader can resolve relative-time phrasing
    ("last month") against the dated memory blocks. When ``answer_first`` is set, the reader
    emits ANSWER before NOTES so budget truncation cannot eat the answer (A1). When
    ``strict_abstain`` is set, the answering policy adds a don't-fabricate clause so a
    deep-context reader declines on genuinely-unanswerable questions instead of inventing a
    specific detail (A2, reader-side) — while still allowing multi-hop synthesis.
    """
    head = _ANSWER_FIRST_HEAD if answer_first else _COT_HEAD
    policy = _ANSWER_POLICY_STRICT if strict_abstain else _ANSWER_POLICY
    return (
        head
        + policy
        + "\n=== RETRIEVED MEMORIES ===\n"
        + format_chunks(chunks)
        + "\n=== END MEMORIES ===\n\n"
        + f"Question (asked on {question_date}): {question}\n\n"
        + ("ANSWER:" if answer_first else "NOTES:")
    )


_RECO_INSTRUCTION = (
    "You are Cortex, helping the user with a recommendation or decision using the retrieved "
    "memories about them.\n"
    "\n"
    "The user is asking for advice/a recommendation, NOT a single stored fact. Work in two "
    "steps:\n"
    "1. NOTES: list the user's OWN relevant preferences, habits, goals, constraints, and past "
    "statements found in the memories (quote them exactly).\n"
    "2. ANSWER: give a concrete, specific recommendation that explicitly builds on those "
    "stated preferences.\n"
    "\n"
    "Policy (read carefully):\n"
    "- ALWAYS give a recommendation grounded in what the memories reveal about the user. NEVER "
    f'reply "{ABSTAIN_SENTINEL}" and never refuse — even sparse context is enough to tailor '
    "advice.\n"
    "- Do NOT give generic advice that ignores the user's stated preferences; if the user said "
    "they dislike or want to avoid something, respect that.\n"
    "- Tie the recommendation back to the user's specific situation; keep it concise.\n"
)

_RECO_POLICY = _RECO_INSTRUCTION[_RECO_INSTRUCTION.index("Policy") :]

_RECO_INSTRUCTION_ANSWER_FIRST = (
    "You are Cortex, helping the user with a recommendation or decision using the retrieved "
    "memories about them.\n"
    "\n"
    "The user is asking for advice/a recommendation, NOT a single stored fact. Output in THIS "
    "order — the recommendation FIRST so it is never cut off:\n"
    "1. ANSWER: a concrete, specific recommendation that explicitly builds on the user's stated "
    "preferences.\n"
    "2. NOTES: then cite the user's OWN relevant preferences, habits, goals, and constraints "
    "found in the memories (quote them exactly).\n"
    "\n" + _RECO_POLICY
)


def build_recommendation_prompt(
    question: str,
    question_date: str,
    chunks: Sequence[MemoryChunk],
    answer_first: bool = False,
) -> str:
    """Build a preference-aware reader prompt for recommendation/preference questions.

    Unlike the factual reader, this never abstains and explicitly grounds the recommendation
    in the user's own stated preferences — the failure mode on LongMemEval's
    single-session-preference questions (the factual reader either abstained or gave generic
    advice that ignored the user's preferences). ``answer_first`` leads with the recommendation
    so budget truncation cannot eat it (A1).
    """
    return (
        (_RECO_INSTRUCTION_ANSWER_FIRST if answer_first else _RECO_INSTRUCTION)
        + "\n=== RETRIEVED MEMORIES ===\n"
        + format_chunks(chunks)
        + "\n=== END MEMORIES ===\n\n"
        + f"Question (asked on {question_date}): {question}\n\n"
        + ("ANSWER:" if answer_first else "NOTES:")
    )


GATE_ANSWERABLE = "ANSWERABLE"
GATE_UNANSWERABLE = "UNANSWERABLE"

# A2 detect-then-decline: a conservative gatekeeper that OWNS the unanswerable verdict. A deep,
# answer-first reader over-answers genuinely-unanswerable (adversarial) questions — it finds some
# loosely-related memory and answers instead of declining (LoCoMo adversarial −4.1 at k=100). This
# gate runs a cheap second opinion over the SAME retrieved memories BEFORE the reader and declines
# only when the asked-about fact is truly absent. It is deliberately biased toward ANSWERABLE so it
# never suppresses a real answer (multi-hop chains, counts, and date arithmetic all count as
# answerable), targeting only the over-answering failure.
# EXPERIMENTAL (off by default): the n=100 LoCoMo probe (at the correct 8192 budget) recovered NO
# adversarial questions and added +67% $/q — the few hard adversarial cases retrieve memories that
# look answer-bearing, so the gate rules them ANSWERABLE just as the reader does. See
# bench/results/LOCOMO_results.md ("A2 detect-then-decline"). Kept as an available lever.
_GATE_INSTRUCTION = (
    "You are a strict gatekeeper. Decide whether the question below can be answered from the "
    "retrieved memories. A reader will produce an answer ONLY if you rule it ANSWERABLE; if you "
    "rule it UNANSWERABLE the system correctly declines to answer.\n"
    "\n"
    "- ANSWERABLE: the memories state the specific information the question asks for, OR it can "
    "be derived by combining/reasoning over facts that ARE present (multi-hop chains, counts, "
    "and date arithmetic all count as ANSWERABLE).\n"
    "- UNANSWERABLE: the specific person, event, object, or fact the question asks about is "
    "simply NOT present in the memories, so any answer would be a guess or need outside "
    "knowledge.\n"
    "\n"
    "Be biased toward ANSWERABLE — rule UNANSWERABLE ONLY when the asked-about information is "
    "genuinely absent. When in doubt, rule ANSWERABLE.\n"
    "\n"
    f"Reply with exactly one word: {GATE_ANSWERABLE} or {GATE_UNANSWERABLE}."
)


def build_answerability_gate_prompt(
    question: str,
    question_date: str,
    chunks: Sequence[MemoryChunk],
) -> str:
    """Build the detect-then-decline gate prompt (A2).

    Run before the reader over the SAME retrieved memories: a conservative second opinion that
    rules a question ANSWERABLE / UNANSWERABLE. When it rules UNANSWERABLE the system emits the
    abstention sentinel without invoking the reader — recovering the adversarial/abstention
    accuracy a deep-context, answer-first reader loses by over-answering unanswerable questions.
    """
    return (
        _GATE_INSTRUCTION
        + "\n=== RETRIEVED MEMORIES ===\n"
        + format_chunks(chunks)
        + "\n=== END MEMORIES ===\n\n"
        + f"Question (asked on {question_date}): {question}\n\n"
        + "Verdict:"
    )


def gate_says_unanswerable(text: str) -> bool:
    """Parse a gate verdict → True iff it ruled the question UNANSWERABLE.

    Conservative by construction: returns True only when an exact ``UNANSWERABLE`` token is
    present (matched token-wise so the ``ANSWERABLE`` substring of ``UNANSWERABLE`` never
    misfires). A garbled/empty verdict therefore defaults to answerable and can never wrongly
    suppress a good answer.
    """
    tokens = {tok.strip(".:,;!?\"'()[]").upper() for tok in text.split()}
    return GATE_UNANSWERABLE in tokens


# L2 listwise reranker: over-retrieve a deep pool (RRF top-100) then let a cheap model reorder it
# by usefulness so the reader sees a SMALL, high-precision top-k instead of the noisy full pool.
# Depth (k=100) recovered the retrieval-miss multi-hop bucket (+10.6) but dragged in loosely-related
# noise (hurt adversarial, "broke 12" answerable). A reranker aims to keep depth's recall while
# restoring top-k precision — and, by shrinking the reader context, is cost-neutral or cheaper.
# Reuses the existing google-genai (AI Studio) key on the aux provider — no Vertex, no new auth.
# EXPERIMENTAL (off by default): the n=100 probe FAILED the +1pt gate. A cheap listwise LLM is not
# a cross-encoder — it demotes weak multi-hop bridge chunks (LoCoMo multi-hop −20pt, recall-on-25
# 0.78) and, on large-context LME, the extra 100-chunk rerank call raises $/q (+22%). It DOES help
# single-hop (less noise, +5.6pt) — the noise-reduction idea has merit IF recall is preserved, which
# needs a real cross-encoder (Vertex Ranking API, human-gated). Plumbing kept for that backend.
_RERANK_INSTRUCTION = (
    "You are reranking retrieved memories by how useful each is for answering the question.\n"
    "Below are numbered memories. Return the numbers of the memories MOST useful for answering, "
    "most useful first, as a comma-separated list (e.g. '3, 1, 7'). Judge usefulness for THIS "
    "question specifically — a memory that supplies a needed fact, name, date, or a link in a "
    "multi-hop chain is useful even if it only partially matches the wording. Include every "
    "memory that could plausibly help; omit only clearly-irrelevant ones. Return ONLY the numbers."
)


def build_rerank_prompt(question: str, chunks: Sequence[MemoryChunk]) -> str:
    """Build the listwise rerank prompt: numbered memories + the question, ending at 'Ranked:'."""
    body = "\n".join(
        f"[{i}] ({chunk.date} | {chunk.session_id}) {chunk.text}"
        for i, chunk in enumerate(chunks, start=1)
    )
    return (
        _RERANK_INSTRUCTION
        + "\n\n=== MEMORIES ===\n"
        + body
        + "\n=== END MEMORIES ===\n\n"
        + f"Question: {question}\n\n"
        + "Ranked numbers (most useful first):"
    )


def parse_rerank_order(text: str, n: int) -> list[int]:
    """Parse a listwise rerank response into 0-based indices, in ranked order, deduped, in-range.

    Robust to any surrounding prose ("Memory 3, then 1") — it extracts integers in order, maps the
    model's 1-based numbering to 0-based, drops out-of-range/duplicate ids. An empty/garbled
    response yields ``[]`` so the caller can fall back to the original RRF order.
    """
    seen: set[int] = set()
    order: list[int] = []
    for tok in re.findall(r"\d+", text):
        i = int(tok) - 1  # the prompt numbers memories from 1
        if 0 <= i < n and i not in seen:
            seen.add(i)
            order.append(i)
    return order


REFLECTION_SENTINEL = "NONE"

_REFLECTION_INSTRUCTION = (
    "You are condensing retrieved chat memories into a COMPACT structured digest that helps "
    "another model answer the question. Read the dated memories; extract ONLY what is relevant.\n"
    "\n"
    "Output these sections, omitting any that do not apply:\n"
    "- TIMELINE: relevant events in chronological order, each prefixed with its date (resolve "
    "relative dates against the memory dates). This anchors temporal reasoning.\n"
    "- CURRENT FACTS: for anything the user changed/updated over time, state the LATEST value and "
    "note what it changed from (most recent date wins).\n"
    "- TOTALS: any counts or aggregations across multiple sessions the question may require.\n"
    "\n"
    "Quote dates, names, and numbers EXACTLY as written. Be terse — facts, not prose. "
    f'If no memory is relevant, reply exactly "{REFLECTION_SENTINEL}".'
)


def build_reflection_prompt(
    question: str,
    question_date: str,
    chunks: Sequence[MemoryChunk],
) -> str:
    """Build the prompt for a cheap query-time reflection over the retrieved memories.

    The reflection (one cheap aux-model call) digests retrieved rounds into a dated TIMELINE,
    update-resolved CURRENT FACTS, and cross-session TOTALS — directly targeting the residual
    reader failures (temporal ordering, knowledge-update recency, multi-session aggregation).
    The digest is prepended to the reader context.
    """
    return (
        _REFLECTION_INSTRUCTION
        + "\n\n=== RETRIEVED MEMORIES ===\n"
        + format_chunks(chunks)
        + "\n=== END MEMORIES ===\n\n"
        + f"Question (asked on {question_date}): {question}\n\n"
        + "Digest:"
    )


# L4 episodic extraction (write-time): ONE cheap flash-lite call per memorize that pulls the
# structured "episode" out of a memory — when it happened (an ABSOLUTE date, resolving relative
# phrasing like "last week" against the message date), who it was about, where, and what. Stored in
# the additive `events` table; nothing here runs unless the engine's episodic flag is on. The
# "episodic event extractor" marker below lets an offline FakeProvider responder recognise the
# prompt and return canned JSON (CLAUDE.md: deterministic, no live LLM calls in tests).
_EPISODIC_EXTRACTION_INSTRUCTION = (
    "You are an episodic event extractor. Read the MESSAGE below and extract the single "
    "real-world event it records, as structured fields for a personal timeline.\n"
    "\n"
    "Return ONE single-line JSON object with EXACTLY these keys and nothing else:\n"
    '{"event_time": <ISO-8601 date or null>, "actor": <str or null>, '
    '"location": <str or null>, "event_type": <str or null>}\n'
    "\n"
    "Rules:\n"
    "- event_time: the date the event actually happened, as an absolute ISO-8601 date "
    '(YYYY-MM-DD). Resolve relative dates ("yesterday", "last week", "next Friday") against '
    "the MESSAGE DATE below. Use null if no date can be determined.\n"
    "- actor: who the event is about (a name or role), else null.\n"
    "- location: where it happened, else null.\n"
    '- event_type: a short phrase for WHAT happened (e.g. "moved apartment", "started a new '
    'job"), else null.\n'
    "- Use null (not an empty string) for anything the message does not state.\n"
    "- Output ONLY the JSON object on a single line — no prose, no markdown fences."
)


# G2 write-time entity/relationship extraction, FOLDED into the episodic call. When the graph flag
# is on, ``build_episodic_extraction_prompt(include_graph=True)`` swaps in THIS instruction so a
# SINGLE flash-lite call yields BOTH the episodic event AND the ego-graph (entities, labeled
# directed relations, and the memory's primary subject) — never a second model call. The "episodic
# event extractor" marker is preserved so existing FakeProvider responders still fire; the graph
# fields are parsed SEPARATELY by ``parse_graph_extraction`` (episodic parsing is byte-identical).
_EPISODIC_GRAPH_EXTRACTION_INSTRUCTION = (
    "You are an episodic event extractor and personal knowledge-graph builder. The USER is the "
    "speaker. Read the MESSAGE below and extract BOTH the single real-world event it records AND "
    "the entities and relationships it states, for a personal timeline and ego-graph.\n"
    "\n"
    "Return ONE single-line JSON object with EXACTLY these keys and nothing else:\n"
    '{"event_time": <ISO-8601 date or null>, "actor": <str or null>, '
    '"location": <str or null>, "event_type": <str or null>, '
    '"entities": [{"name": <str>, "type": "person"|"place"|"org"|"project"|"thing"}], '
    '"relations": [{"src": <name or "self">, "label": <str>, "dst": <name or "self">}], '
    '"subject": <entity name or "self">}\n'
    "\n"
    "Rules:\n"
    "- event_time: the date the event actually happened, as an absolute ISO-8601 date "
    '(YYYY-MM-DD). Resolve relative dates ("yesterday", "last week", "next Friday") against '
    "the MESSAGE DATE below. Use null if no date can be determined.\n"
    "- actor: who the event is about (a name or role), else null.\n"
    "- location: where it happened, else null.\n"
    '- event_type: a short phrase for WHAT happened (e.g. "moved apartment", "started a new '
    'job"), else null.\n'
    "- entities: the distinct people, places, orgs, projects, and things NAMED in the message. "
    'Each is {"name", "type"} with type one of person, place, org, project, thing. NEVER emit the '
    "user themselves as an entity — the user is implicit.\n"
    '- The user is the speaker: first-person subjects ("I", "me", "my", "mine") are the literal '
    'string "self", never a named entity.\n'
    '- relations: labeled directed facts {"src", "label", "dst"} where src/dst are an entity name '
    'or the literal "self". The label is a SHORT lowercase verb or role phrase (e.g. "girlfriend", '
    '"likes", "lives_in", "works_on", "employer").\n'
    "- subject: the single most specific non-self entity the message is ABOUT, by name; use "
    '"self" when the message is primarily about the user.\n'
    "- Use null (not an empty string) for anything the message does not state, and [] when there "
    "are no entities or relations.\n"
    "- Output ONLY the JSON object on a single line — no prose, no markdown fences."
)


def build_episodic_extraction_prompt(
    content: str, message_date: str, include_graph: bool = False
) -> str:
    """Build the write-time episodic event extraction prompt (single-line JSON output).

    The model returns ``{event_time, actor, location, event_type}`` for ``content``, resolving
    relative dates against ``message_date`` to an absolute ISO date (null when unknown). The
    "episodic event extractor" marker lets an offline FakeProvider responder detect the prompt.

    When ``include_graph`` is set (the G2 graph write path), the folded instruction ALSO requests
    ``entities``/``relations``/``subject`` in the SAME JSON object, so one call feeds both features;
    with ``include_graph`` false the prompt is byte-identical to the episodic-only prompt.
    """
    instruction = (
        _EPISODIC_GRAPH_EXTRACTION_INSTRUCTION
        if include_graph
        else _EPISODIC_EXTRACTION_INSTRUCTION
    )
    return (
        instruction
        + f"\n\nMESSAGE DATE: {message_date}\n\n"
        + "=== MESSAGE ===\n"
        + content
        + "\n=== END MESSAGE ===\n\n"
        + "JSON:"
    )


def _extract_json_object(text: str) -> dict[str, object] | None:
    """Best-effort pull of the first JSON object from ``text`` (tolerates prose + ``` fences)."""
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s[:4].lower() == "json":
            s = s[4:]
        s = s.strip()
    candidates = [s]
    start, end = s.find("{"), s.rfind("}")
    if 0 <= start < end:
        candidates.append(s[start : end + 1])
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _opt_str(value: object) -> str | None:
    """Normalise a JSON field to a non-blank string, else None (nulls, blanks, non-strings)."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


# An extracted event_time must be an absolute ISO-8601 calendar date (YYYY-MM-DD) — the timeline
# is ordered by a plain lexical sort on this string, so any other shape ("2024", "March 3rd",
# "2024-3-3", a timestamp) would sort wrongly. Anything that fails this anchored match falls back
# to the ingest date, keeping the sort invariant intact.
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _valid_iso_date(value: object) -> str | None:
    """Return a normalised ``YYYY-MM-DD`` string iff ``value`` is a valid ISO date, else None."""
    s = _opt_str(value)
    return s if s and _ISO_DATE_RE.match(s) else None


def parse_episodic_extraction(text: str, fallback_event_time: str) -> dict[str, str | None]:
    """Parse an episodic-extraction response into ``{event_time, actor, location, event_type}``.

    Robust to surrounding prose and markdown fences. Never raises: on empty input, a parse
    failure, or a missing/blank/null ``event_time``, ``event_time`` falls back to
    ``fallback_event_time`` (the ingest date) so every extracted event still lands on the
    timeline; the structured fields default to ``None``.
    """
    fallback: dict[str, str | None] = {
        "event_time": fallback_event_time,
        "actor": None,
        "location": None,
        "event_type": None,
    }
    if not text or not text.strip():
        return fallback
    obj = _extract_json_object(text)
    if obj is None:
        return fallback
    return {
        "event_time": _valid_iso_date(obj.get("event_time")) or fallback_event_time,
        "actor": _opt_str(obj.get("actor")),
        "location": _opt_str(obj.get("location")),
        "event_type": _opt_str(obj.get("event_type")),
    }


# G2 graph extraction — a SEPARATE tolerant parser reading the SAME single-line JSON the folded
# episodic+graph prompt emits. Kept apart from ``parse_episodic_extraction`` so the episodic return
# shape/behaviour is byte-identical; ``memory.py`` calls both over the one shared response text.
# Cap each list so a runaway extraction can't balloon the write.
_MAX_GRAPH_ITEMS = 20
# The entity types a WRITE-TIME extraction may carry (``self`` is implicit — never emitted as an
# entity, so it is excluded here; an unknown/invalid type falls back to "thing").
_EXTRACT_ENTITY_TYPES = VALID_ENTITY_TYPES - {"self"}


class GraphExtraction(TypedDict):
    """The graph half of a folded extraction: named entities, labeled relations, primary subject."""

    entities: list[dict[str, str]]
    relations: list[dict[str, str]]
    subject: str | None


def _parse_entity_items(raw: object) -> list[dict[str, str]]:
    """Coerce the ``entities`` field into ``[{"name", "type"}]``; drop nameless/malformed, cap.

    ``type`` is lowercased and kept only if it is a valid non-self entity type, else "thing".
    """
    out: list[dict[str, str]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = _opt_str(item.get("name"))
        if not name:
            continue
        etype = (_opt_str(item.get("type")) or "").lower()
        if etype not in _EXTRACT_ENTITY_TYPES:
            etype = "thing"
        out.append({"name": name, "type": etype})
        if len(out) >= _MAX_GRAPH_ITEMS:
            break
    return out


def _parse_relation_items(raw: object) -> list[dict[str, str]]:
    """Coerce ``relations`` into ``[{"src", "label", "dst"}]``; drop items missing any field, cap.

    ``label`` is lowercased to a short role/verb phrase (matching the prompt's contract).
    """
    out: list[dict[str, str]] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        src = _opt_str(item.get("src"))
        label = _opt_str(item.get("label"))
        dst = _opt_str(item.get("dst"))
        if not (src and label and dst):
            continue
        out.append({"src": src, "label": label.lower(), "dst": dst})
        if len(out) >= _MAX_GRAPH_ITEMS:
            break
    return out


def parse_graph_extraction(text: str) -> GraphExtraction:
    """Parse a folded extraction's graph half → ``{entities, relations, subject}``. Never raises.

    Tolerant like the episodic/transfer parsers: reads the same single-line JSON object (ignoring
    surrounding prose / markdown fences), drops malformed entities (no name) and relations (missing
    src/label/dst), and caps each list at ``_MAX_GRAPH_ITEMS``. On empty input or a parse failure it
    returns empty lists and a null subject — so a garbage extraction degrades to "no graph" and can
    never break the write path. First-person "self" resolution is done at the memorize layer, not
    here (the raw subject/relation strings are returned verbatim).
    """
    if not text or not text.strip():
        return {"entities": [], "relations": [], "subject": None}
    obj = _extract_json_object(text)
    if obj is None:
        return {"entities": [], "relations": [], "subject": None}
    return {
        "entities": _parse_entity_items(obj.get("entities")),
        "relations": _parse_relation_items(obj.get("relations")),
        "subject": _opt_str(obj.get("subject")),
    }


# L5 anti-saturation contradiction arbiter (write-time): ONE cheap call per memorize, made ONLY
# when a NEW memory falls in the "related but not duplicate" band against an existing memory (the
# embedding-only dedup already absorbs near-identical restatements at ~zero cost). It decides if the
# new memory UPDATEs (supersedes) the old value, is redundant (NOOP), or is an independent ADD —
# so a store can absorb decades of updates while keeping the LATEST value the one recall returns.
# The "contradiction arbiter" marker below lets an offline FakeProvider responder recognise the
# prompt and return canned JSON (CLAUDE.md: deterministic, no live LLM calls in tests).
SUPERSEDE_ADD = "ADD"
SUPERSEDE_UPDATE = "UPDATE"
SUPERSEDE_NOOP = "NOOP"
_SUPERSEDE_VERDICTS = frozenset({SUPERSEDE_ADD, SUPERSEDE_UPDATE, SUPERSEDE_NOOP})

_ARBITER_INSTRUCTION = (
    "You are a contradiction arbiter for a personal memory store. Compare the NEW memory against "
    "the EXISTING memory and decide how the store should absorb the NEW one so it never holds a "
    "stale value.\n"
    "\n"
    "Return ONE single-line JSON object with EXACTLY these keys and nothing else:\n"
    '{"verdict": "ADD"|"UPDATE"|"NOOP", "supersedes_id": <existing id or null>}\n'
    "\n"
    "Rules:\n"
    "- UPDATE: the NEW memory changes/overrides the SAME attribute of the SAME subject (e.g. a new "
    'city, job, phone, or preference that replaces the old one). Set "supersedes_id" to the '
    "EXISTING memory's id — it will be marked as superseded so recall returns only the NEW value.\n"
    "- NOOP: the NEW memory just restates the EXISTING one, adding nothing. It will be dropped.\n"
    "- ADD: the two are about different things (no conflict). The NEW memory is stored alongside "
    'the old. Set "supersedes_id" to null.\n'
    "- When unsure, prefer ADD — never discard information you are not confident is superseded.\n"
    "- Output ONLY the JSON object on a single line — no prose, no markdown fences."
)


def build_supersession_arbiter_prompt(
    new_content: str,
    existing_id: str,
    existing_content: str,
    message_date: str,
) -> str:
    """Build the write-time contradiction-arbiter prompt (single-line JSON verdict output).

    The model compares ``new_content`` against ONE related ``existing_content`` (identified by
    ``existing_id``) and returns ``{verdict, supersedes_id}`` deciding UPDATE / NOOP / ADD. The
    "contradiction arbiter" marker lets an offline FakeProvider responder detect the prompt.
    """
    return (
        _ARBITER_INSTRUCTION
        + f"\n\nMESSAGE DATE: {message_date}\n\n"
        + f"=== EXISTING MEMORY (id: {existing_id}) ===\n"
        + existing_content
        + "\n=== NEW MEMORY ===\n"
        + new_content
        + "\n=== END ===\n\n"
        + "JSON:"
    )


def parse_supersession_verdict(text: str) -> dict[str, str | None]:
    """Parse an arbiter response into ``{verdict, supersedes_id}``. NEVER raises.

    Robust to surrounding prose and markdown fences. On empty input, a parse failure, or an
    unrecognised verdict, it falls back to ``{"verdict": "ADD", "supersedes_id": None}`` — so any
    arbiter garbage degrades to a plain, information-preserving insert (soft-update must never lose
    data or break memorize).
    """
    fallback: dict[str, str | None] = {"verdict": SUPERSEDE_ADD, "supersedes_id": None}
    if not text or not text.strip():
        return fallback
    obj = _extract_json_object(text)
    if obj is None:
        return fallback
    raw_verdict = _opt_str(obj.get("verdict"))
    verdict = raw_verdict.upper() if raw_verdict else None
    if verdict not in _SUPERSEDE_VERDICTS:
        return fallback
    return {"verdict": verdict, "supersedes_id": _opt_str(obj.get("supersedes_id"))}


# Bulk transfer / import (write-time): ONE generate call turns a raw external dump (an exported
# chat history, a profile paste, notes) into discrete standalone memories. This is a WRITE-TIME
# import — it is never on the recall path, so it cannot touch the recall cost invariant. The "memory
# extraction system" marker below lets an offline FakeProvider responder recognise the prompt and
# return canned JSON (CLAUDE.md: deterministic, no live LLM calls in tests).
_TRANSFER_INSTRUCTION = (
    "You are a memory extraction system. The user pasted a raw dump from a previous AI "
    "conversation or their own notes. Extract the distinct, standalone facts, preferences, project "
    "details, instructions, events, and relationships worth remembering long-term.\n"
    "\n"
    "Return ONLY a valid JSON array — no prose, no markdown fences. Each element is an object "
    '{"content": <a concise, self-contained fact as a string>, "kind": <one of: preference, fact, '
    "project, instruction, event, relationship>}. Make each `content` stand alone without the "
    "dump. Omit conversational filler and anything transient. Pick the single kind that best fits."
)


def build_transfer_extraction_prompt(raw_data: str) -> str:
    """Build the bulk-import extraction prompt (a JSON array of ``{content, kind}`` objects).

    Write-time import only, never on the recall path. The "memory extraction system" marker lets an
    offline FakeProvider responder detect the prompt and return canned JSON.
    """
    return (
        _TRANSFER_INSTRUCTION
        + "\n\n=== RAW DATA ===\n"
        + raw_data
        + "\n=== END RAW DATA ===\n\n"
        + "JSON array:"
    )


def _extract_json_array(text: str) -> list[object] | None:
    """Best-effort pull of the first JSON array from ``text`` (tolerates prose + ``` fences).

    Also unwraps a ``{"memories": [...]}`` object, mirroring the old server's tolerant parsing.
    """
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s[:4].lower() == "json":
            s = s[4:]
        s = s.strip()
    candidates = [s]
    start, end = s.find("["), s.rfind("]")
    if 0 <= start < end:
        candidates.append(s[start : end + 1])
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and isinstance(obj.get("memories"), list):
            return obj["memories"]
    return None


_REFUSAL_PREFIXES = (
    "i'm sorry",
    "i am sorry",
    "sorry",
    "i cannot",
    "i can't",
    "i can not",
    "as an ai",
    "unfortunately",
    "there are no",
    "i don't",
    "i do not",
    "no memories",
)


def _looks_like_refusal(line: str) -> bool:
    """A model refusal / meta-comment, not data — so the line-split fallback never imports it."""
    low = line.lower()
    return any(low.startswith(p) for p in _REFUSAL_PREFIXES)


def parse_transfer_extraction(text: str, max_facts: int = 200) -> list[tuple[str, str | None]]:
    """Parse a transfer-extraction response into ``[(content, kind_or_None), ...]``. NEVER raises.

    Accepts a JSON array of strings OR of ``{"content", "kind"}`` objects (also a
    ``{"memories": [...]}`` wrapper). ``kind`` is kept only when it is one of the six valid kinds,
    else ``None``. On unparseable output it falls back to line-splitting so a best-effort import
    still lands. Blank contents are dropped; the result is capped at ``max_facts``.
    """
    out: list[tuple[str, str | None]] = []
    arr = _extract_json_array(text)
    if arr is not None:
        for item in arr:
            if isinstance(item, str):
                content, kind = item.strip(), None
            elif isinstance(item, dict):
                content = _opt_str(item.get("content")) or _opt_str(item.get("text")) or ""
                kind = normalize_kind(_opt_str(item.get("kind")))
            else:
                continue
            if content:
                out.append((content, kind))
    else:  # fallback: treat each substantial line as a standalone memory (skip model meta-prose)
        for line in text.splitlines():
            # Strip ONLY a real leading list/ordinal marker ("- ", "* ", "• ", "1. ", "2) ") — an
            # anchored match so content that merely starts with a digit ("401k", "3D", "2024 …") is
            # preserved, unlike a greedy ``lstrip`` of the marker characters.
            stripped = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line).strip()
            if len(stripped) > 10 and not _looks_like_refusal(stripped):
                out.append((stripped, None))
    return out[: max(0, max_facts)]
