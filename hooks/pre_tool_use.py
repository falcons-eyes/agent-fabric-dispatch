#!/usr/bin/env python3
"""PreToolUse hook — Policy enforcement for routing decisions.

Called by Claude Code before each tool invocation. Evaluates policy rules
and enforces routing:
  - FRONTIER: exit 0 (no action, Claude proceeds normally)
  - LOCAL:    exit 2 + stderr guidance (blocks tool, suggests fabric-worker MCP)
  - LOCAL_ONLY: exit 2 + stderr hard-block (blocks tool, sensitive data stays local)

Input (stdin): JSON with session_id, hook_event_name, tool_name, tool_input, cwd.
Output: exit 0 (allow) or exit 2 + stderr (block with feedback to Claude).
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router.engine import evaluate_policy
from metering.recorder import record_dispatch


def main():
    try:
        hook_input = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, EOFError):
        # If no input or bad JSON, let Claude proceed normally
        sys.exit(0)

    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})

    # Never intercept fabric-worker MCP tools — that's where we're routing TO
    if tool_name.startswith("mcp__fabric"):
        sys.exit(0)

    cwd = hook_input.get("cwd", "")

    # Build context for policy evaluation
    context = {
        "tool_name": tool_name,
        "paths": _extract_paths(tool_input),
        "task_type": _classify_task(tool_name, tool_input),
        "cwd": cwd,
    }

    # Evaluate policy
    decision = evaluate_policy(context)
    route = decision.get("route", "FRONTIER")

    if route == "FRONTIER":
        # No action — Claude proceeds as usual
        sys.exit(0)

    if route == "LOCAL_ONLY":
        # Hard wall: block and redirect to fabric-worker
        record_dispatch(
            tool=tool_name,
            decision="LOCAL_ONLY",
            reason=decision.get("reason", "unknown"),
            tokens_est=_estimate_tokens(tool_input),
        )
        msg = (
            f"FABRIC POLICY BLOCK: This operation touches sensitive files matching "
            f"a LOCAL_ONLY rule ({decision.get('reason', '')}). "
            f"Data must stay on this machine. "
            f"Use the fabric-worker MCP server tools (refactor_bulk, summarize_files, "
            f"classify) to process this data locally instead."
        )
        print(msg, file=sys.stderr)
        sys.exit(2)

    if route in ("LOCAL", "LOCAL_WORKER"):
        # Soft redirect: block and guide Claude to use fabric-worker MCP
        record_dispatch(
            tool=tool_name,
            decision=route,
            reason=decision.get("reason", "unknown"),
            tokens_est=_estimate_tokens(tool_input),
        )
        task_type = context["task_type"]
        msg = (
            f"FABRIC DISPATCH: This task ({task_type}) matches a LOCAL routing rule "
            f"({decision.get('reason', '')}). "
            f"Use the fabric-worker MCP server's '{task_type}' tool to handle this "
            f"locally and save frontier tokens."
        )
        print(msg, file=sys.stderr)
        sys.exit(2)

    # Default: proceed normally
    sys.exit(0)


def _extract_paths(tool_input: dict) -> list[str]:
    """Extract file paths from tool input."""
    paths = []
    if "file_path" in tool_input:
        paths.append(tool_input["file_path"])
    if "paths" in tool_input:
        paths.extend(tool_input["paths"])
    if "path" in tool_input:
        paths.append(tool_input["path"])
    return paths


def _classify_task(tool_name: str, tool_input: dict) -> str:
    """Classify the task type based on tool name and input."""
    task_map = {
        "refactor": "refactor_bulk",
        "summarize": "summarize_files",
        "classify": "classify",
        "test": "test_scaffold",
    }
    tool_lower = tool_name.lower()
    for keyword, task_type in task_map.items():
        if keyword in tool_lower:
            return task_type
    return tool_name


def _estimate_tokens(tool_input: dict) -> int:
    """Rough token estimate based on input size."""
    input_str = json.dumps(tool_input)
    # ~4 chars per token as rough estimate
    return len(input_str) // 4


if __name__ == "__main__":
    main()
