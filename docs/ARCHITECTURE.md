# Architecture

## Architecture Evolution

Three dispatch architectures were evaluated. The inverted (local-first) approach
proved viable; MCP dispatch was disproven.

| Architecture | Cost vs Frontier | Why |
|---|---|---|
| MCP Dispatch (hooks redirect to MCP tools) | **+78%** | Tool call output token tax + system prompt inflation |
| Inverted (Ollama default, `claude -p` fallback) | **-84%** | Zero overhead for local work; frontier only when needed |
| Frontier-only (baseline) | 0% | Full quality, full cost |

## Current Architecture: Inverted (Local-First)

```
User Prompt
    │
    ▼
Ollama (local model) ──── always runs first, $0.00
    │
    ▼
Confidence check (self-consistency + self-verify)
    │
    ├── HIGH ──► return local answer ($0.00)
    │
    ├── MEDIUM ──► Shepherding: frontier hint (~$0.03)
    │                 │
    │                 └──► re-run local with hint ──► done
    │
    └── LOW ──► try shepherding first
                  │
                  ├── hint works ──► done (~$0.03)
                  │
                  └── hint fails ──► full frontier fallback (~$0.19)
```

### Why MCP Dispatch Failed

When Claude Code calls an MCP tool, it generates the tool arguments as **output
tokens** — including the full code payload. This output token tax averaged 2.1x the
cost of answering directly. Combined with system prompt inflation from CLAUDE.md and
MCP tool definitions (~$0.15/call input overhead), MCP dispatch costs **more** than
doing nothing.

## Components

| Component | Language | Responsibility |
|---|---|---|
| `agent/fabric_agent.py` | Python | Local-first agent: Ollama default, `claude -p` escalation |
| `router/engine.py` | Python | Policy evaluation (deterministic rules) |
| `worker/server.py` | Python | MCP server wrapping Ollama (retained for integration) |
| `hooks/` | Python | Claude Code hook scripts (PreToolUse, PostToolUse, Stop) |
| `metering/` | Python | JSONL recorder, session summary, savings estimator |
| `fabric` CLI | Go | Command-line interface, session management |
| `benchmarks/` | Python | Benchmark suite (15 runners) + fabric-agent comparison |

## Data Flow

1. Task arrives (prompt or file reference)
2. `classify_task()` routes: keyword match → local or frontier
3. Local path: `ollama_generate()` via HTTP API (`/api/generate`, `stream: false`)
4. Confidence estimation: self-consistency (N samples) + self-verification (AutoMix)
5. HIGH confidence: return local answer directly
6. MEDIUM/LOW confidence: `_try_shepherding()` — ask frontier for a hint (~50 tokens),
   re-run local with the hint. ~85% cheaper than full frontier calls.
7. If shepherding fails on LOW: full `frontier_query()` via `claude -p --output-format json`
8. Metering records: routing decision, tokens, cost, timing

## Trust Boundary

- Credentials: never read, stored, or forwarded. Frontier calls happen only via the
  user's own `claude` CLI session.
- Router sees tool names, paths, tags — **not** file contents, unless a rule needs a
  path match (path only).
- Worker sees content it processes — on the user's own machine.
- Metering records routing metadata and token estimates; never prompt or output text.
