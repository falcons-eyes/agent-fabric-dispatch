---
description: Show agent-fabric-dispatch session stats — tool calls logged, local vs frontier dispatches, estimated tokens saved. Use when the user asks how much was saved or what was routed locally.
disable-model-invocation: true
---

Read `~/.falcon/metering.jsonl` and report:
1. total tool calls logged this session
2. how many were dispatched LOCAL vs FRONTIER (Phase 0 shows OBSERVE only)
3. the share that looks local-eligible (refactor / summarize / classify / test scaffolding by tool + path)
Keep it to five lines. Never print file contents or prompts — metadata only.
