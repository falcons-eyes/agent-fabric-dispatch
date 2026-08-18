"""Metering recorder — appends dispatch and tool execution events to JSONL.

File: ~/.fabric/metering.jsonl
Format: one JSON object per line with standardized fields.
"""

import json
import os
import time
import uuid
from pathlib import Path

METERING_PATH = Path.home() / ".fabric" / "metering.jsonl"

# Session ID persists for the lifetime of the process
_session_id: str | None = None


def _get_session_id() -> str:
    global _session_id
    if _session_id is None:
        _session_id = str(uuid.uuid4())[:8]
    return _session_id


def _ensure_dir():
    METERING_PATH.parent.mkdir(parents=True, exist_ok=True)


def _append(record: dict):
    """Append a record to the metering JSONL file."""
    _ensure_dir()
    with open(METERING_PATH, "a") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")


def record_dispatch(
    tool: str,
    decision: str,
    reason: str,
    tokens_est: int = 0,
    worker_id: str = "local",
):
    """Record a routing dispatch decision."""
    _append({
        "type": "dispatch",
        "ts": time.time(),
        "session": _get_session_id(),
        "tool": tool,
        "decision": decision,
        "reason": reason,
        "tokens_est": tokens_est,
        "worker_id": worker_id,
    })


def record_tool_execution(
    tool: str,
    duration_ms: int = 0,
    tokens_input: int = 0,
    tokens_output: int = 0,
    was_local: bool = False,
):
    """Record a tool execution completion."""
    _append({
        "type": "execution",
        "ts": time.time(),
        "session": _get_session_id(),
        "tool": tool,
        "duration_ms": duration_ms,
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
        "was_local": was_local,
    })
