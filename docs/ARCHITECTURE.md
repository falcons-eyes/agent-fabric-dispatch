# Architecture

## Components
| Component | Runs where | Responsibility |
|---|---|---|
| Claude Code | user's terminal | unchanged; the cockpit. Emits hook events; calls MCP tools. |
| `hooks/` | user's terminal | thin scripts wired into `~/.claude/settings.json`; call router / metering. |
| `router/` | user's terminal | evaluate policy → `LOCAL_WORKER | CLOUD_WORKER | FRONTIER`. Stateless per call. |
| `worker/` | any node | MCP server wrapping Ollama/vLLM; exposes tool set v1. Location-agnostic. |
| `metering/` | user's terminal | jsonl recorder, session summary, savings estimator. |
| Agent Fabric (external) | control plane + nodes | identity, L3, zones. Absent in Phase 0–1 by design. |

## Data flow (Phase 1)
1. Claude decides to spawn a subagent / call a task tool → `PreToolUse` fires with tool name + input.
2. Router loads policy, tags the request (path, task, budget), returns decision.
   - `FRONTIER` → exit 0, no change; Claude proceeds normally.
   - `LOCAL_WORKER` → rewrite spawn to a routed local subagent / point Claude to the `falcon-worker` MCP tool.
3. Worker executes on a local model, returns text; Claude reviews/integrates as with any tool result.
4. Metering appends `{ts, session, tool, decision, reason, tokens_est, worker_id}`.
5. `Stop` hook prints session summary.

## Trust boundary
- Credentials: never read, stored, or forwarded by any component. Frontier calls happen only inside the user's own Claude session.
- Router sees tool names, paths, tags, sizes — **not** file contents or prompts, unless a rule needs a path match (path only).
- Worker sees the content it is asked to process — on the user's own machine or a node the user's zone owns.
- Metering records routing metadata and token estimates; never prompt or output text.

## Non-goals (v1)
Screen scraping, credential proxying, content inspection/DLP, a custom TUI.
