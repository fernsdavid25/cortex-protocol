"""LoCoMo dataset adapter — maps LoCoMo into the harness's ``QAInstance`` format.

LoCoMo (Maharana et al., 2024, "Evaluating Very Long-Term Conversational Memory of LLM
Agents"): 10 very-long multi-session conversations between two speakers, each with QA pairs
categorised 1=multi-hop, 2=temporal, 3=open-domain, 4=single-hop, 5=adversarial (unanswerable).

Mapping:
- Each conversation's sessions become the haystack (speaker_a -> "user", speaker_b ->
  "assistant" so the round-level chunker works; the speaker NAME is kept in the content because
  LoCoMo questions reference speakers by name).
- Each QA pair -> one ``QAInstance``; all QAs of a conversation SHARE the same session list
  objects (no duplication in memory).
- category -> ``locomo-{multihop,temporal,opendomain,singlehop,adversarial}`` so metrics bucket
  per category; the judge (``get_anscheck_prompt``) handles these labels.
- Adversarial (cat 5) is unanswerable: ``question_id`` gets the ``_abs`` suffix (so
  ``is_abstention`` is True and the abstention judge prompt is used).
- Evidence ``"D{n}:{m}"`` -> answer session id ``"{sample}_s{n}"`` for recall@k.
- Evidence ``"D{n}:{m}"`` -> answer TURN id ``"{sample}_s{n}#{pos}"`` for coverage@k.

The evidence-turn convention (read this before changing anything below)
-----------------------------------------------------------------------
recall@k throws the ``:{m}`` component away, so it can only ever say "we surfaced the right
conversation". coverage@k needs the turn, and getting the turn wrong produces a
plausible-looking but WRONG number — the exact failure this metric exists to fix. So the
resolution is deliberately two-tier:

1. **Primary (no arithmetic, no off-by-one possible).** Real LoCoMo turns carry a ``dia_id``
   whose value is byte-identical to the strings in ``qa[].evidence`` (``"D1:1"``, ``"D2:7"``).
   ``convert()`` builds a ``dia_id -> turn id`` map while it walks each session and resolves
   evidence by EXACT STRING EQUALITY. Indexing never enters into it.
2. **Fallback (only for turns with no ``dia_id``).** Positional resolution using
   ``LOCOMO_EVIDENCE_TURN_BASE``, bounds-checked; see that constant for the justification of
   its value and for what happens when it does not resolve.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .memory_system import QAInstance
from .turns import turn_id

_CATEGORY_TYPE = {
    1: "locomo-multihop",
    2: "locomo-temporal",
    3: "locomo-opendomain",
    4: "locomo-singlehop",
    5: "locomo-adversarial",
}
_SESSION_RE = re.compile(r"session_(\d+)$")
_EVIDENCE_RE = re.compile(r"D(\d+)")
# Full evidence id: session number AND turn number. Anchored at the end so a malformed
# trailing suffix falls through to the (session-only) recall path rather than resolving
# to a confidently wrong turn.
_EVIDENCE_TURN_RE = re.compile(r"D(\d+):(\d+)$")

# LoCoMo evidence ``"D{n}:{m}"``: ``m`` is 1-INDEXED into that session's raw turn list, so
# the 0-based position is ``m - LOCOMO_EVIDENCE_TURN_BASE``.
#
# This constant governs ONLY the fallback path — real LoCoMo turns carry ``dia_id`` equal to
# the evidence string, and those are resolved by exact string equality (see the module
# docstring), where no off-by-one is possible.
#
# MEASURED, not inferred (2026-07-28, against the real ``locomo10.json`` from
# snap-research/locomo: 10 samples, 1,986 QA pairs, 5,882 turns): ``dia_id`` is present on
# 5,882/5,882 turns (100%), and ``"D1:3"`` is ``session_1[2]`` — so the turn component is
# 1-indexed and the exact-match path covers every well-formed id in the corpus.
#
# A fallback id that lands outside the session's turn range yields NO gold turn and is
# counted; ``convert()`` prints one warning per file. So if this convention is ever wrong for
# a future release, coverage@k reads as a loud zero with a warning, never as a plausible
# number that quietly misstates retrieval quality.
LOCOMO_EVIDENCE_TURN_BASE: int = 1

# Six evidence fields in the released corpus are not a bare ``"D{n}:{m}"`` — four of them pack
# SEVERAL ids into one string, separated by a space or ``"; "``:
#
#     "D8:6; D9:17"                    "D9:1 D4:4 D4:6"
#     "D22:1 D22:2 D9:10 D9:11"        "D21:18 D21:22 D11:15 D11:19"
#
# Matching the raw field anchored (as the session-level ``_EVIDENCE_RE`` still does) keeps only
# the FIRST id and silently discards the rest, which shrinks the gold set and therefore biases
# recall UPWARD. Tokenizing first recovers 13 of the 14 real references those six fields hold.
#
# The two genuinely corrupt fields, ``"D"`` and ``"D:11:26"``, are left unresolved on purpose:
# ``"D:11:26"`` is *probably* ``D11:26``, but "probably" is exactly the guess this metric
# exists to eliminate, so it is dropped and counted rather than repaired.
_EVIDENCE_SPLIT_RE = re.compile(r"[;\s]+")


def _ordered_sessions(conv: dict) -> list[tuple[int, str, list[dict]]]:
    """Return [(n, date, turns), ...] for each ``session_N`` key, in numeric order."""
    nums = sorted(int(m.group(1)) for k in conv if (m := _SESSION_RE.match(k)))
    return [(n, conv.get(f"session_{n}_date_time", ""), conv[f"session_{n}"]) for n in nums]


def _resolve_one(
    base: str,
    token: str,
    dia_index: dict[str, str],
    session_len: dict[str, int],
) -> str | None:
    """Resolve a SINGLE well-formed evidence token to a canonical turn id, or ``None``.

    Primary: exact ``dia_id`` match (no arithmetic). Fallback: positional, using
    ``LOCOMO_EVIDENCE_TURN_BASE`` and bounds-checked against the session's real length so an
    out-of-range id is DROPPED (and counted by the caller) instead of inventing a turn.
    """
    hit = dia_index.get(token)
    if hit is not None:
        return hit
    m = _EVIDENCE_TURN_RE.match(token)
    if m is None:
        return None
    sid = f"{base}_s{m.group(1)}"
    # int() also normalizes a zero-padded turn ("D30:05" -> turn 5), which exact dia_id
    # matching cannot do because "D30:05" != "D30:5" as strings.
    pos = int(m.group(2)) - LOCOMO_EVIDENCE_TURN_BASE
    if pos < 0 or pos >= session_len.get(sid, 0):
        return None
    return turn_id(sid, pos)


def _resolve_evidence_turns(
    base: str,
    evidence: object,
    dia_index: dict[str, str],
    session_len: dict[str, int],
) -> tuple[list[str], int]:
    """Resolve one raw evidence FIELD to every turn id it names.

    Returns ``(turn_ids, unresolved_token_count)``. A field may hold several ids (see
    ``_EVIDENCE_SPLIT_RE``), so this returns a list, not a scalar — keeping only the first
    token is what silently shrinks the gold set today.
    """
    raw = str(evidence).strip()
    if not raw:
        return [], 0
    resolved: list[str] = []
    unresolved = 0
    for token in _EVIDENCE_SPLIT_RE.split(raw):
        if not token:
            continue
        tid = _resolve_one(base, token, dia_index, session_len)
        if tid is None:
            unresolved += 1
        else:
            resolved.append(tid)
    return resolved, unresolved


def convert(raw: list[dict]) -> list[QAInstance]:
    """Convert loaded LoCoMo JSON (list of samples) into a flat list of ``QAInstance``."""
    instances: list[QAInstance] = []
    unmatched = 0
    for idx, sample in enumerate(raw):
        conv = sample["conversation"]
        base = str(sample.get("sample_id", idx))
        speaker_a = conv.get("speaker_a")

        session_ids: list[str] = []
        dates: list[str] = []
        sessions: list[list[dict]] = []
        # dia_id -> turn id, and session id -> turn count: the two lookups the evidence
        # resolver needs. Built here, while we already walk every turn, so no second pass.
        dia_index: dict[str, str] = {}
        session_len: dict[str, int] = {}
        for n, date, turns in _ordered_sessions(conv):
            sid = f"{base}_s{n}"
            session_ids.append(sid)
            dates.append(date)
            session_len[sid] = len(turns)
            for pos, t in enumerate(turns):
                dia = t.get("dia_id")
                if dia is not None:
                    dia_index[str(dia)] = turn_id(sid, pos)
            # The harness turn dicts stay exactly {role, content} — nothing about turn
            # identity leaks into the reader prompt, so no prompt byte moves.
            sessions.append(
                [
                    {
                        "role": "user" if t.get("speaker") == speaker_a else "assistant",
                        "content": f"{t.get('speaker', '')}: {t.get('text', '')}",
                    }
                    for t in turns
                ]
            )
        last_date = dates[-1] if dates else ""

        for qi, qa in enumerate(sample.get("qa", [])):
            category = qa.get("category")
            qid = f"{base}_q{qi}"
            if category == 5:  # adversarial / unanswerable
                qid += "_abs"
                answer = "This question is not answerable from the conversation."
            else:
                answer = str(qa.get("answer", ""))
            evidence = qa.get("evidence", [])
            evidence_sids = sorted(
                {f"{base}_s{m.group(1)}" for e in evidence if (m := _EVIDENCE_RE.match(str(e)))}
            )
            # Computed in PARALLEL with (never instead of) the session gold above, so
            # recall@k keeps reporting byte-identical values for every existing input.
            evidence_tids: set[str] = set()
            for e in evidence:
                tids, bad = _resolve_evidence_turns(base, e, dia_index, session_len)
                unmatched += bad
                evidence_tids.update(tids)
            instances.append(
                QAInstance(
                    question_id=qid,
                    question_type=_CATEGORY_TYPE.get(category, "locomo-other"),
                    question=str(qa.get("question", "")),
                    answer=answer,
                    question_date=last_date,
                    haystack_session_ids=session_ids,
                    haystack_dates=dates,
                    haystack_sessions=sessions,
                    answer_session_ids=evidence_sids,
                    answer_turn_ids=sorted(evidence_tids),
                )
            )
    if unmatched:
        # ASCII only: this prints to a Windows cp1252 console in the maintainer's environment.
        print(
            f"WARN: locomo: {unmatched} evidence id(s) matched no turn; coverage@k understated "
            "(check LOCOMO_EVIDENCE_TURN_BASE against this release's schema)"
        )
    return instances


def load_locomo(path: str | Path) -> list[QAInstance]:
    """Load + convert a LoCoMo JSON file into ``QAInstance`` objects."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return convert(raw)
