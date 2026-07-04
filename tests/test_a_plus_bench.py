"""A+ harness-robustness fixes: atomic dataset download, resume tolerating a missing hypothesis
key, recall@k trivial-flagging for non-retrieval systems, and estimated-price stamping.

All offline/deterministic — the download is exercised with a monkeypatched urlretrieve so no
network or live LLM is touched.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from cortex_bench import dataset
from cortex_bench.judge import Judge
from cortex_bench.memory_system import QAInstance, Usage
from cortex_bench.metrics import Record, aggregate
from cortex_bench.run import run
from cortex_bench.systems.stub import GoldStub


def _fake_urlretrieve(content: str) -> Callable[[str, object], tuple[str, None]]:
    """Stand-in for urllib.request.urlretrieve that writes `content` to the requested path."""

    def _inner(url: str, filename: object) -> tuple[str, None]:
        Path(filename).write_text(content, encoding="utf-8")
        return str(filename), None

    return _inner


# --- Fix #1: atomic download ------------------------------------------------------------------


def test_download_atomic_replace_on_valid_json(tmp_path, monkeypatch):
    """A well-formed download is validated then os.replace()'d into the final path, leaving no
    temp/.part file behind."""
    payload = [
        {"question_id": "a", "question_type": "single-session-user", "question": "q", "answer": "x"}
    ]
    monkeypatch.setattr(
        dataset.urllib.request, "urlretrieve", _fake_urlretrieve(json.dumps(payload))
    )
    target = dataset.download("oracle", tmp_path)
    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8")) == payload
    # No stray temp artifacts — only the published file remains.
    assert [p.name for p in tmp_path.iterdir()] == [target.name]


def test_download_truncated_never_published(tmp_path, monkeypatch):
    """A truncated/corrupt download must raise and NOT leave a file at the target path, so a
    later run cleanly re-downloads instead of finding a poisoned 'cached' file."""
    monkeypatch.setattr(
        dataset.urllib.request, "urlretrieve", _fake_urlretrieve('[{"question_id": "a"')
    )
    with pytest.raises(json.JSONDecodeError):
        dataset.download("oracle", tmp_path)
    assert not (tmp_path / dataset.VARIANT_FILES["oracle"]).exists()
    assert list(tmp_path.iterdir()) == []  # temp file cleaned up too


def test_download_interrupt_cleans_up_and_skips_target(tmp_path, monkeypatch):
    """A process killed mid-download (KeyboardInterrupt) leaves neither a target nor a .part."""

    def _boom(url: str, filename: object) -> None:
        Path(filename).write_text("partial-bytes", encoding="utf-8")  # partial write
        raise KeyboardInterrupt

    monkeypatch.setattr(dataset.urllib.request, "urlretrieve", _boom)
    with pytest.raises(KeyboardInterrupt):
        dataset.download("oracle", tmp_path)
    assert not (tmp_path / dataset.VARIANT_FILES["oracle"]).exists()
    assert list(tmp_path.iterdir()) == []


def test_download_short_circuits_existing_file(tmp_path, monkeypatch):
    """An already-present file is returned without touching the network."""
    target = tmp_path / dataset.VARIANT_FILES["oracle"]
    target.write_text("[]", encoding="utf-8")

    def _explode(url: str, filename: object) -> None:  # pragma: no cover - must not be called
        raise AssertionError("urlretrieve must not be called when the file already exists")

    monkeypatch.setattr(dataset.urllib.request, "urlretrieve", _explode)
    assert dataset.download("oracle", tmp_path) == target


# --- Fix #2: resume tolerates a done row missing the hypothesis key ----------------------------


def test_resume_tolerates_row_missing_hypothesis_key(tmp_path):
    """A resumed 'done' row carrying a verdict but no 'hypothesis' key (hand-edited / older
    format) must be treated as done with an empty hypothesis, not KeyError and abort the run."""
    insts = [QAInstance("q1", "single-session-user", "q?", "gold")]
    out = tmp_path / "hyps.jsonl"
    out.write_text(json.dumps({"question_id": "q1", "correct": True}) + "\n", encoding="utf-8")
    records = run(insts, GoldStub(), Judge("offline"), hyp_path=out, resume=True)
    by_id = {r.instance.question_id: r for r in records}
    assert by_id["q1"].correct is True  # stored verdict reused
    assert by_id["q1"].hypothesis == ""  # missing key defaulted, no KeyError
    assert by_id["q1"].measured is False  # skipped the reader -> unmeasured


# --- Fixes #3 & #4: recall trivial-flag + price-source stamp -----------------------------------


def _rec() -> Record:
    return Record(
        QAInstance("a", "single-session-user", "q", "x", answer_session_ids=["s1"]),
        "x",
        True,
        Usage(input_tokens=10, output_tokens=2),
        retrieved_session_ids=["s1"],
    )


def test_recall_flagged_trivial_for_nonretrieval_system():
    rep = aggregate([_rec()], system_name="full-context")
    assert rep["recall_at_k"] == 1.0
    assert "recall_at_k_note" in rep  # flagged so the trivial 1.0 isn't read as a real metric


def test_recall_not_flagged_for_retrieval_system_or_default():
    # A real retrieval system's recall carries no caveat note.
    assert "recall_at_k_note" not in aggregate([_rec()], system_name="cortex-v0")
    # No system_name supplied -> legacy report shape (backward compatible).
    assert "recall_at_k_note" not in aggregate([_rec()])


def test_price_source_stamped_estimated_vs_published():
    assert aggregate([_rec()], reader_model="gemini-3.5-flash")["price_source"] == "estimated"
    assert aggregate([_rec()], reader_model="gemini-2.5-flash")["price_source"] == "published"
    # No priced reader -> no price_source key (nothing to caveat).
    assert "price_source" not in aggregate([_rec()])
