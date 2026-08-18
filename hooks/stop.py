#!/usr/bin/env python3
"""Stop hook — Session summary.

Called by Claude Code when a session ends.
Prints a summary of dispatches, savings, and sensitive egress stats.

Input (stdin): JSON with session_id, hook_event_name, stop_hook_active, assistant_message.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metering.summary import print_session_summary


def main():
    try:
        hook_input = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, EOFError):
        hook_input = {}

    # Summarize all records for this process (Phase 1: no session filtering)
    _ = hook_input.get("session_id", "")
    print_session_summary(None)
    sys.exit(0)


if __name__ == "__main__":
    main()
