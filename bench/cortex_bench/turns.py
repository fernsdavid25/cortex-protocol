"""The single definition of what one *memory unit* is: a conversational TURN.

Why this module exists
----------------------
``recall@k`` answers "did we surface the right CONVERSATION" — it compares retrieved
session ids against ``QAInstance.answer_session_ids``, so a system that returns a 40-turn
session because one of its turns is gold scores a perfect 1.0. ``coverage@k`` answers the
harder and more honest question, "did we surface the right MEMORY", and to ask it at all
both the dataset adapters and the systems under test need ONE shared, dataset-independent
identity for a single turn. That identity is defined here and nowhere else, so LoCoMo,
LongMemEval, the chunkers and the metric can never drift apart about what a turn is.

The identity is a string, ``"{session_id}#{position}"``:

- ``session_id`` is already the universal join key in this harness (recall@k,
  ``MemoryChunk.session_id``, prompt headers, LoCoMo's ``{sample}_s{n}``), so suffixing it
  makes turn ids globally unique with zero extra plumbing.
- A string is JSON-serializable, set-hashable, sortable, and readable in a failure dump
  (``conv1_s2#0``). A ``(sid, pos)`` tuple would need re-normalizing at every JSON boundary
  and buys nothing; a flat index into a concatenated turn list is unreadable and becomes
  silently wrong the moment a haystack is filtered or reordered — which is exactly what the
  LongMemEval *oracle* variant is.
- ``position`` is **0-based**, because every producer and consumer in this repo indexes a
  Python list. The only 1-based number in this system is LoCoMo's ``"D{n}:{m}"`` evidence
  string, and it is converted exactly once, at parse time, in ``locomo.py``.
"""

from __future__ import annotations

from collections.abc import Iterable

# Separator between the session id and the 0-based turn position. Chosen because '#' cannot
# appear in a LongMemEval session id (they are hex/ultrachat slugs) nor in a LoCoMo one
# (``{sample_id}_s{n}``), so ``turn_id`` is injective and a turn id can be split back apart.
TURN_ID_SEP: str = "#"


def turn_id(session_id: str, position: int) -> str:
    """Canonical id for one turn: ``"{session_id}#{position}"``, ``position`` 0-based."""
    return f"{session_id}{TURN_ID_SEP}{position}"


def turn_ids_for(session_id: str, positions: Iterable[int]) -> tuple[str, ...]:
    """Canonical ids for several turns of the SAME session, order preserved."""
    return tuple(turn_id(session_id, p) for p in positions)
