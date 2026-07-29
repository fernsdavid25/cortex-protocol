"""LongMemEval dataset loading (verified schema — see bench/README.md)."""

from __future__ import annotations

import json
import os
import tempfile
import urllib.request
from pathlib import Path

from .memory_system import QAInstance
from .turns import turn_id

HF_BASE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"
VARIANT_FILES = {
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
    "oracle": "longmemeval_oracle.json",
}


def _gold_turn_ids(entry: dict) -> list[str]:
    """Turn ids of every haystack turn flagged ``has_answer: true`` (turn-level gold).

    LongMemEval marks the individual evidence TURNS inside a gold session with a per-turn
    ``has_answer`` boolean; non-gold sessions omit the key entirely. That is the turn-level
    counterpart of ``answer_session_ids``, and until now zero lines of this harness read it.

    A release that omits ``has_answer`` (possible for the *oracle* variant, which we cannot
    confirm without downloading it) yields ``[]`` here — which makes coverage@k degrade to
    ``None`` ("not reported") rather than to a fabricated 0.0.
    """
    ids: list[str] = []
    for sid, session in zip(
        entry.get("haystack_session_ids", []),
        entry.get("haystack_sessions", []),
        strict=False,
    ):
        for pos, turn in enumerate(session):
            if isinstance(turn, dict) and turn.get("has_answer"):
                ids.append(turn_id(str(sid), pos))
    return ids


def parse_instances(raw: list[dict]) -> list[QAInstance]:
    return [
        QAInstance(
            question_id=str(e["question_id"]),
            question_type=e["question_type"],
            question=str(e["question"]),
            # `answer` is occasionally an int (e.g. temporal counts) in the real data — coerce.
            answer=str(e["answer"]),
            question_date=str(e.get("question_date", "")),
            haystack_session_ids=e.get("haystack_session_ids", []),
            haystack_dates=e.get("haystack_dates", []),
            haystack_sessions=e.get("haystack_sessions", []),
            answer_session_ids=e.get("answer_session_ids", []),
            answer_turn_ids=_gold_turn_ids(e),
        )
        for e in raw
    ]


def load_instances(path: str | Path) -> list[QAInstance]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_instances(raw)


def download(variant: str, dest_dir: str | Path) -> Path:
    """Download a LongMemEval variant ('s' | 'm' | 'oracle') to dest_dir if missing.

    Downloaded ATOMICALLY: fetch to a temp file in the destination dir, verify it parses as
    JSON, then ``os.replace()`` it into the final path only on success. An interrupted or
    corrupt download can therefore never leave a truncated file at ``target`` that would poison
    every subsequent run (a later run finds the "existing" file and skips the re-download, then
    ``load_instances`` chokes on it). On any failure the temp file is removed so a re-run cleanly
    re-downloads.
    """
    if variant not in VARIANT_FILES:
        raise ValueError(f"unknown variant {variant!r}; choose from {list(VARIANT_FILES)}")
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / VARIANT_FILES[variant]
    if target.exists():
        return target
    fd, tmp_name = tempfile.mkstemp(dir=dest, prefix=f".{VARIANT_FILES[variant]}.", suffix=".part")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        urllib.request.urlretrieve(f"{HF_BASE}/{VARIANT_FILES[variant]}", tmp)  # noqa: S310
        # Validate the payload parses BEFORE publishing it — a truncated download must never be
        # os.replace()'d into place, where it would be treated as a cached-good file forever.
        json.loads(tmp.read_text(encoding="utf-8"))
        os.replace(tmp, target)
    except BaseException:  # incl. KeyboardInterrupt — a killed run must not leave a .part behind
        tmp.unlink(missing_ok=True)
        raise
    return target
