"""Single source of truth for CORTEX_TIER profiles (hosted engine + bench reader knobs).

The ``cheap`` tier is the #1 accuracy-per-dollar system. Its invariant is **cost per RECALL**: the
hosted recall path and the benchmark HEADLINE config (LongMemEval_S 0.932 / LoCoMo 0.813) stay
byte-identical. Cheap MAY gain write-time ("store") enrichments that don't touch recall cost —
per the tier-cost policy, episodic extraction is one such: a cheap flash-lite call per ``memorize``
that structures event_time/actor/location/what (recall is unchanged). It is ON by default in
``cheap`` and opt-out via ``CORTEX_EPISODIC=0`` (or a self-hoster leaving no extract model). Only
recall-time cost-adders (pro reader, rerank, reflection) live behind the ``flagship`` switch.

Note the bench harness reads only reader/retrieval knobs via :func:`resolve_run_config` — NOT
``use_episodic``/``extract_model`` — so episodic never perturbs the 0.932/0.813 numbers; it is a
hosted-product feature.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping

log = logging.getLogger(__name__)

# FROZEN cheap tier = today's benchmark HEADLINE config (LongMemEval_S 0.932).
CHEAP: dict = {
    # hosted engine knobs (the only server-side tunables)
    "embed_model": "gemini-embedding-001",
    "embed_dim": 768,
    "top_k": 8,  # hosted recall depth (serve.py CORTEX_TOP_K default)
    # HOSTED-product recall knob (distinct from the bench ``use_rerank`` below): rerank recalled
    # memories with the Vertex cross-encoder for better agent-facing precision. Needs GCP Discovery
    # Engine creds; a pure self-host (or CORTEX_RERANK=0) falls back to RRF top-k, byte-identical.
    "hosted_rerank": True,
    # bench reader knobs (the frozen headline)
    "reader_model": "gemini-3.5-flash",
    "answer_first": True,
    "max_output_tokens": 8192,
    "preference_mode": True,
    "bench_top_k": 50,  # LME retrieval depth
    # write-time enrichment — ON in cheap (recall cost unchanged; opt-out via CORTEX_EPISODIC=0)
    "use_episodic": True,
    # G2 write-time entity/relationship graph — ON in cheap (folded into the episodic flash-lite
    # call, so ZERO extra recall cost; opt-out via CORTEX_GRAPH=0). The bench never reads it, so the
    # 0.932/0.813 headline numbers are unaffected.
    "use_graph": True,
    "extract_model": "gemini-2.5-flash-lite",  # cheap event extractor (episodic; hosted only)
    # recall-time cost-adders — OFF in cheap, flagship-only
    "use_reflection": False,
    "use_rerank": False,
    "rerank_k": 25,
    "rerank_backend": "listwise",  # only when use_rerank; flagship = cross-encoder
}
# Applied ON TOP of the resolved tier when the harness runs --locomo (LoCoMo 0.813 headline).
LOCOMO_DELTA: dict = {
    "reader_model": "gemini-2.5-flash",
    "preference_mode": False,
    "bench_top_k": 100,
}
# Flagship = cheap + validated recall-time firepower: the pro reader, a deeper retrieval pool, and
# the Vertex cross-encoder reranker (n=100: +1pt LoCoMo / held LME, and ~3x cheaper by feeding the
# reader 25 reranked chunks instead of 100 — bench/results/vertex_rerank_results.md). The reranker
# needs GCP Discovery Engine creds; without them it falls back gracefully to RRF top-k. The killed
# cheap listwise reranker (rerank_backend="listwise") is never used here.
FLAGSHIP: dict = {
    **CHEAP,
    "reader_model": "gemini-3.1-pro-preview",
    "bench_top_k": 100,  # deep pool for the reranker to select from
    "use_rerank": True,
    "rerank_backend": "vertex-ranking",
}
TIER_PROFILES: dict = {"cheap": CHEAP, "flagship": FLAGSHIP}


def resolve_tier(env: Mapping[str, str] | None = None) -> tuple[str, dict]:
    """Return (tier_name, a COPY of its profile dict). Default + unknown -> cheap."""
    env = os.environ if env is None else env
    raw = (env.get("CORTEX_TIER") or "cheap").strip().lower()
    if raw not in TIER_PROFILES:
        log.warning("unknown CORTEX_TIER=%r; falling back to 'cheap'", raw)
        raw = "cheap"
    return raw, dict(TIER_PROFILES[raw])  # copy so callers can't mutate the module globals
