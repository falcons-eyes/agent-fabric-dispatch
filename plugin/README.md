# agent-fabric-dispatch (Claude Code plugin)

Route bulk and sensitive sub-tasks from your Claude Code session to a local model on your own machines. Keep using Claude Code exactly as before; this rides along via hooks and MCP and tells you how many tokens it saved.

## Install
```
/plugin marketplace add falcons-eyes/agent-fabric-dispatch
/plugin install agent-fabric-dispatch@falcons-eyes
```

## What it does (v0.1 — Phase 0)
- Logs every tool call's metadata to `~/.falcon/metering.jsonl` (never prompts or file contents)
- Prints a session summary on exit
- `/agent-fabric-dispatch:dispatch-status` — see what was logged and what looks local-eligible
- `/agent-fabric-dispatch:dispatch-policy` — view your routing policy

Phase 1 adds the PreToolUse router and a local Ollama worker (MCP). See the repo README for the roadmap.

## Privacy
Metadata only. Your credentials never leave your terminal. Nothing is sent anywhere.
