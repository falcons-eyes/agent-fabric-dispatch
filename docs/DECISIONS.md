# Architecture Decision Records

## ADR-001 — Enter through official surfaces, not scraping
Context: original idea was to intercept Claude Code I/O via terminal multiplexing.
Decision: use hooks (PreToolUse/PostToolUse/SubagentStart/Stop), MCP, subagents, Agent SDK.
Why: scraping is brittle and touches credentials/ToS; official surfaces already expose the exact interception points, and OSS precedents (Ollama-delegate MCP servers, model-router hooks) prove the path.
Consequence: terminal multiplexing (tmux) becomes a *visibility* layer (panes) in Phase 2, not the dispatch mechanism.

## ADR-002 — Deterministic rules before any classifier
Why: predictability = trust; users must be able to explain every route. ML routing only after rules are shown insufficient (Phase 3).

## ADR-003 — Measure first (Phase 0 gate at 30%)
Why: if the local-eligible share of real workloads is small, savings claims are marketing. The tool must be able to say "not for you".

## ADR-004 — Savings and quality always reported together
Why: token savings offset by rework is a loss. Quality pass rate per task type is a first-class output.

## ADR-005 — No custom TUI in v1
Why: the promise is "no new tool". A TUI fork is a gated Phase 3 option only.

## ADR-006 — Orchestrator-neutral router
Why: single-vendor dependence (hook/SDK surface changes) is the top strategic risk. Router core is CLI-agnostic; adapters per CLI.

## ADR-007 — Bring your own subscription; one identity per zone
Why: no token reselling (COGS 0, no ToS exposure). Workers' frontier calls only in the user's own session context. Positioned as the legitimate alternative to account-sharing.

## ADR-008 — Go CLI + Python components (hybrid stack)
Context: need a fast, distributable CLI for session management and tmux orchestration, plus flexible scripting for hooks/router/worker/metering.
Decision: Go for the `fabric` CLI binary; Python for all dispatch logic.
Why: Go compiles to a single static binary (no runtime dependency), starts in milliseconds, and has excellent process/exec support for tmux. Python provides rapid iteration and a rich ecosystem for LLM tooling.

## ADR-009 — Use Ollama HTTP API, not CLI subprocesses
Context: initially considered `ollama run` (CLI) for inference.
Decision: use Ollama HTTP API (`/api/generate`, `stream: false`) directly.
Why: `ollama run` injects ANSI escape codes that corrupt structured output. The HTTP API returns clean JSON with token counts and timing metadata.

## ADR-010 — tmux for terminal multiplexing
Context: needed multi-pane terminal visibility for Claude Code + worker streams.
Decision: use tmux as the terminal multiplexer.
Why: ubiquitous on Linux/macOS, scriptable, supports named sessions/windows/panes, integrates well with Go process management.

## ADR-011 — Inverted architecture (local-first, not MCP dispatch)
Context: MCP dispatch (hooks redirect tool calls to MCP server) was tested end-to-end.
Decision: invert the architecture. Ollama is the default; `claude -p` is called as a subprocess only when needed.
Why: MCP dispatch **increases** cost by 78% due to (1) output token tax — Claude generates tool arguments including full code payloads as output tokens, averaging 2.1x the direct answer cost, and (2) system prompt inflation — CLAUDE.md + MCP tool definitions add ~$0.15/call in input tokens. The inverted approach eliminates both overheads entirely.
Consequence: MCP server code is retained for potential future integration but is not the primary dispatch path. The `agent/agent_fabric.py` module is the main entry point.

## ADR-012 — Pre-flight interception is not possible with Claude Code
Context: explored intercepting prompts before they reach Claude (ANTHROPIC_BASE_URL override, pre-prompt hooks, third-party CLIs).
Decision: accept that Claude Code does not support pre-flight interception.
Why: (1) `ANTHROPIC_BASE_URL` is ignored — Claude Code uses its own OAuth channel, (2) no pre-prompt hook exists in the hook protocol, (3) third-party CLIs are blocked by Anthropic's OAuth-only authentication policy. The inverted architecture (ADR-011) sidesteps this by making the local model the default entry point.
