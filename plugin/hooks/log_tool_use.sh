#!/usr/bin/env bash
# Phase 0 instrumentation: append one JSON line per tool call. Records metadata only — never prompt or file contents.
set -euo pipefail
DIR="${HOME}/.falcon"; mkdir -p "$DIR"
INPUT="$(cat)"
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if command -v jq >/dev/null 2>&1; then
  echo "$INPUT" | jq -c --arg ts "$TS" '{ts:$ts, session:(.session_id//""), tool:(.tool_name//""), path:(.tool_input.file_path//.tool_input.path//""), decision:"OBSERVE"}' >> "$DIR/metering.jsonl"
else
  echo "{\"ts\":\"$TS\",\"raw\":\"tool_use\"}" >> "$DIR/metering.jsonl"
fi
exit 0
