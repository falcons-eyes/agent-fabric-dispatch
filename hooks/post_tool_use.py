#!/usr/bin/env python3
"""PostToolUse hook — Metering logger.

Called by Claude Code after each tool invocation completes.
Logs the tool execution details for metering and analysis.

Input (stdin): JSON with tool_name, tool_input, tool_output, session_id, cwd.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metering.recorder import record_tool_execution


def main():
    try:
        hook_input = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    tool_name = hook_input.get("tool_name", "unknown")
    tool_input = hook_input.get("tool_input", {})
    tool_output = hook_input.get("tool_output", "")

    # Estimate tokens from available data (~4 chars per token)
    tokens_input = len(json.dumps(tool_input)) // 4
    tokens_output = len(str(tool_output)) // 4

    # Detect if this was a local worker call (fabric-worker MCP tools)
    was_local = (
        tool_name.startswith("mcp__fabric")
        or tool_name in ("refactor_bulk", "summarize_files", "classify")
    )

    record_tool_execution(
        tool=tool_name,
        duration_ms=0,  # not available from protocol
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        was_local=was_local,
    )

    sys.exit(0)


if __name__ == "__main__":
    main()
