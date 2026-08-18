"""Metering summary — session statistics and savings report.

Reads ~/.fabric/metering.jsonl and produces human-readable summaries.
Can be run as a module: python -m metering.summary
"""

import json
import sys
from collections import Counter
from pathlib import Path

METERING_PATH = Path.home() / ".fabric" / "metering.jsonl"


def load_records(session_id: str | None = None) -> list[dict]:
    """Load metering records, optionally filtered by session."""
    if not METERING_PATH.exists():
        return []

    records = []
    with open(METERING_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if session_id and record.get("session") != session_id:
                    continue
                records.append(record)
            except json.JSONDecodeError:
                continue
    return records


def compute_summary(records: list[dict]) -> dict:
    """Compute summary statistics from metering records."""
    dispatches = [r for r in records if r.get("type") == "dispatch"]
    executions = [r for r in records if r.get("type") == "execution"]

    total_dispatches = len(dispatches)
    local_dispatches = sum(1 for d in dispatches if d["decision"] in ("LOCAL_WORKER", "LOCAL_ONLY", "LOCAL"))
    frontier_dispatches = sum(1 for d in dispatches if d["decision"] == "FRONTIER")

    tokens_saved = sum(d.get("tokens_est", 0) for d in dispatches if d["decision"] != "FRONTIER")
    sensitive_egress = sum(
        1 for d in dispatches
        if d["decision"] == "FRONTIER" and "LOCAL_ONLY" in d.get("reason", "")
    )

    # Decision breakdown
    decision_counts = Counter(d["decision"] for d in dispatches)

    # Tool breakdown
    tool_counts = Counter(d["tool"] for d in dispatches)

    # Execution stats
    total_duration_ms = sum(e.get("duration_ms", 0) for e in executions)
    local_executions = sum(1 for e in executions if e.get("was_local", False))

    return {
        "total_dispatches": total_dispatches,
        "local_dispatches": local_dispatches,
        "frontier_dispatches": frontier_dispatches,
        "tokens_saved_est": tokens_saved,
        "sensitive_egress": sensitive_egress,
        "decision_counts": dict(decision_counts),
        "tool_counts": dict(tool_counts),
        "total_executions": len(executions),
        "local_executions": local_executions,
        "total_duration_ms": total_duration_ms,
    }


def format_summary(summary: dict) -> str:
    """Format summary as a human-readable string."""
    tokens = summary["tokens_saved_est"]
    if tokens >= 1_000_000:
        tokens_str = f"{tokens / 1_000_000:.1f}M"
    elif tokens >= 1_000:
        tokens_str = f"{tokens / 1_000:.1f}K"
    else:
        tokens_str = str(tokens)

    total = summary["total_dispatches"]
    local = summary["local_dispatches"]
    frontier = summary["frontier_dispatches"]
    egress = summary["sensitive_egress"]

    lines = [
        f"session: {total} dispatches · {local} local · {frontier} frontier · est. saved {tokens_str} tokens · sensitive egress {egress}",
    ]

    if summary["decision_counts"]:
        lines.append("")
        lines.append("Routing breakdown:")
        for decision, count in sorted(summary["decision_counts"].items()):
            pct = (count / total * 100) if total > 0 else 0
            lines.append(f"  {decision}: {count} ({pct:.0f}%)")

    if summary["tool_counts"]:
        lines.append("")
        lines.append("Tool breakdown:")
        for tool, count in sorted(summary["tool_counts"].items()):
            lines.append(f"  {tool}: {count}")

    return "\n".join(lines)


def print_session_summary(session_id: str | None = None):
    """Load records and print a session summary."""
    records = load_records(session_id if session_id != "current" else None)
    if not records:
        print("No metering data recorded for this session.")
        return

    summary = compute_summary(records)
    print(format_summary(summary))


if __name__ == "__main__":
    session_id = sys.argv[1] if len(sys.argv) > 1 else None
    print_session_summary(session_id)
