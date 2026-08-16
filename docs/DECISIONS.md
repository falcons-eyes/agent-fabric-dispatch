# Architecture Decision Records

## ADR-001 — Enter through official surfaces, not scraping
Context: original idea was to intercept Claude Code I/O via terminal multiplexing.
Decision: use hooks (PreToolUse/PostToolUse/SubagentStart/Stop), MCP, subagents, Agent SDK.
Why: scraping is brittle and touches credentials/ToS; official surfaces already expose the exact interception points, and OSS precedents (Ollama-delegate MCP servers, model-router hooks) prove the path.
Consequence: terminal multiplexing becomes a *visibility* layer (panes) in Phase 2, not the dispatch mechanism.

## ADR-002 — Deterministic rules before any classifier
Why: predictability = trust; users must be able to explain every route. ML routing only after rules are shown insufficient (Phase 3).

## ADR-003 — Measure first (Phase 0 gate at 30%)
Why: if the local-eligible share of real workloads is small, savings claims are marketing. The tool must be able to say "not for you".

## ADR-004 — Savings and quality always reported together
Why: token savings offset by rework is a loss. Quality pass rate per task type is a first-class output.

## ADR-005 — No custom TUI in v1
Why: the promise is "no new tool". An opencode/OpenClaude fork is a gated Phase 3 option only.

## ADR-006 — Orchestrator-neutral router
Why: single-vendor dependence (hook/SDK surface changes) is the top strategic risk. Router core is CLI-agnostic; adapters per CLI.

## ADR-007 — Bring your own subscription; one identity per zone
Why: no token reselling (COGS 0, no ToS exposure). Workers' frontier calls only in the user's own session context. Positioned as the legitimate alternative to account-sharing.
