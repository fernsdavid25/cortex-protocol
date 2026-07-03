"""L7 agent-uplift eval: does Cortex memory improve an agent on multi-session tasks?

A small, fully deterministic harness that pits three arms against the same cross-session
tasks — ``memoryless`` (task only), ``full_context`` (all history stuffed into the prompt),
and ``cortex`` (each session memorized, top-k recalled at task time). The thesis: cortex
matches full_context's accuracy at a FRACTION of its input-token cost, and both beat the
memoryless arm on tasks whose answer lives in an earlier session.
"""

from __future__ import annotations
