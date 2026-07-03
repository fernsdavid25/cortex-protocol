"""LongMemEval dataset loading (verified schema — see bench/README.md)."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from .memory_system import QAInstance

HF_BASE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"
VARIANT_FILES = {
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
    "oracle": "longmemeval_oracle.json",
}


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
        )
        for e in raw
    ]


def load_instances(path: str | Path) -> list[QAInstance]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_instances(raw)


def download(variant: str, dest_dir: str | Path) -> Path:
    """Download a LongMemEval variant ('s' | 'm' | 'oracle') to dest_dir if missing."""
    if variant not in VARIANT_FILES:
        raise ValueError(f"unknown variant {variant!r}; choose from {list(VARIANT_FILES)}")
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / VARIANT_FILES[variant]
    if not target.exists():
        urllib.request.urlretrieve(f"{HF_BASE}/{VARIANT_FILES[variant]}", target)  # noqa: S310
    return target
