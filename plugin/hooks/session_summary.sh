#!/usr/bin/env bash
# Stop hook: print a one-line session summary from metering.jsonl (Phase 0: counts only; Phase 1 adds savings).
set -euo pipefail
F="${HOME}/.falcon/metering.jsonl"
[ -f "$F" ] || exit 0
if command -v jq >/dev/null 2>&1; then
  N=$(wc -l < "$F" | tr -d ' ')
  LOCAL=$(jq -r 'select(.decision=="LOCAL")|1' "$F" 2>/dev/null | wc -l | tr -d ' ')
  echo "agent-fabric-dispatch · logged $N tool calls · local dispatches: $LOCAL · log: ~/.falcon/metering.jsonl" >&2
fi
exit 0
