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
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .memory_system import QAInstance

_CATEGORY_TYPE = {
    1: "locomo-multihop",
    2: "locomo-temporal",
    3: "locomo-opendomain",
    4: "locomo-singlehop",
    5: "locomo-adversarial",
}
_SESSION_RE = re.compile(r"session_(\d+)$")
_EVIDENCE_RE = re.compile(r"D(\d+)")


def _ordered_sessions(conv: dict) -> list[tuple[int, str, list[dict]]]:
    """Return [(n, date, turns), ...] for each ``session_N`` key, in numeric order."""
    nums = sorted(int(m.group(1)) for k in conv if (m := _SESSION_RE.match(k)))
    return [(n, conv.get(f"session_{n}_date_time", ""), conv[f"session_{n}"]) for n in nums]


def convert(raw: list[dict]) -> list[QAInstance]:
    """Convert loaded LoCoMo JSON (list of samples) into a flat list of ``QAInstance``."""
    instances: list[QAInstance] = []
    for idx, sample in enumerate(raw):
        conv = sample["conversation"]
        base = str(sample.get("sample_id", idx))
        speaker_a = conv.get("speaker_a")

        session_ids: list[str] = []
        dates: list[str] = []
        sessions: list[list[dict]] = []
        for n, date, turns in _ordered_sessions(conv):
            session_ids.append(f"{base}_s{n}")
            dates.append(date)
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
            evidence_sids = sorted(
                {
                    f"{base}_s{m.group(1)}"
                    for e in qa.get("evidence", [])
                    if (m := _EVIDENCE_RE.match(str(e)))
                }
            )
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
                )
            )
    return instances


def load_locomo(path: str | Path) -> list[QAInstance]:
    """Load + convert a LoCoMo JSON file into ``QAInstance`` objects."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return convert(raw)
