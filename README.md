# agent-fabric-dispatch

> **Keep using Claude Code. Spend fewer tokens. Keep your code where it lives.**
>
> A policy router that sits *beside* your Claude Code session and dispatches sub-tasks to workers on your own machines, your other machines, or a shared Agent Fabric node — so the frontier model does the thinking and cheap local models do the grunt work.

`agent-fabric-dispatch` is the first open component of **Agent Fabric** (FalconEyes): the operations layer for customer-owned AI agents. This repo is deliberately narrow — it proves one loop end to end and measures it.

```
Claude Code (your session, untouched)
      │  hooks: PreToolUse · SubagentStart · Stop
      ▼
falcon-router  ── policy: sensitivity · cost · difficulty ──►  LOCAL_WORKER | CLOUD_WORKER | FRONTIER
      ├─► worker @ this PC        (Ollama / vLLM, exposed as MCP tools)
      ├─► worker @ another PC     (same address over Agent Fabric L3)
      ├─► worker @ AWS node       (shared base model + per-zone LoRA)
      └─► FRONTIER               (hook does nothing; Claude proceeds as usual)
      │
metering ── every dispatch: where it went · tokens · estimated savings ──► "saved 1.2M tokens · 0 sensitive egress"
```

## Why

- **Agent workloads are 80% grunt work.** Bulk refactors, summaries, classification, test scaffolding — verifiable, repetitive tasks that a 32B local model handles fine (NVIDIA, *Small Language Models are the Future of Agentic AI*, 2025). Only ~20% needs a frontier model.
- **Subscriptions cap you; APIs bill you.** Always-on agents hit weekly limits or explode API bills. Local workers cost electricity.
- **Some code must not leave the building.** A policy tag on `repo:corp` or `secrets/` means that work is *physically* routed to a local worker — not a prompt instruction, a route.
- **You should not have to learn a new tool.** No new IDE, no new account. Claude Code stays your cockpit; this rides along via hooks and MCP.

## Design principles

1. **Do not touch the commercial CLI.** No screen scraping, no credential handling. We use Claude Code's official surfaces: hooks, MCP, subagents, Agent SDK.
2. **Deterministic policy first.** v1 routes on rules you wrote (path tags, task tags, budgets). No ML classifier until rules are proven insufficient.
3. **Measure before you save.** Phase 0 is instrumentation only. If less than 30% of your workload is local-eligible, this tool is not for you — and it will tell you.
4. **Orchestrator-neutral by design.** Claude Code first; Codex / Gemini CLI / OpenClaw adapters follow. Router logic never depends on one CLI.
5. **Quality reported next to savings, always.** A dispatch that saves tokens but fails review is a loss. Both numbers ship together.

## Status

| Phase | Goal | Status |
|---|---|---|
| 0 — Instrument | Log real workload distribution via hooks (1 week) | ⬜ |
| 1 — Minimal loop | hook → router → local worker → result → savings report | ⬜ |
| 2 — Multi-node & zones | Workers on other PCs / AWS node over Fabric; zone isolation; multi-LoRA | ⬜ |
| 3 — Intelligence layer | Verification voting, KV/context sharing, cost-objective router, session fan-out | ⬜ |

See [docs/ROADMAP.md](docs/ROADMAP.md) for gates and acceptance criteria.

## Quick start (Phase 1 target — not yet functional)

```bash
# 1. local worker (Ollama + a 32B coding model)
ollama pull qwen2.5-coder:32b
falcon worker up --local

# 2. register hooks + MCP with Claude Code
falcon install-hooks        # writes PreToolUse/PostToolUse/Stop entries into ~/.claude/settings.json
falcon install-mcp          # registers falcon-worker as an MCP server

# 3. write a policy
falcon policy set path:secrets/  local-only
falcon policy set task:refactor_bulk,summarize_files,classify  local

# 4. use Claude Code exactly as before
claude
# on exit:  "session: 14 dispatches · 9 local · 5 frontier · est. saved 1.2M tokens · sensitive egress 0"
```


## Install as a Claude Code plugin

```
/plugin marketplace add falcons-eyes/agent-fabric-dispatch
/plugin install agent-fabric-dispatch@falcons-eyes
```

The plugin lives in [`plugin/`](plugin/) and this repo doubles as its marketplace via [`.claude-plugin/marketplace.json`](.claude-plugin/marketplace.json). v0.1 ships Phase 0 (metering hooks + two skills); the router and local worker land in Phase 1.

## Repository layout

```
agent-fabric-dispatch/
├── README.md
├── .claude-plugin/marketplace.json   this repo IS the marketplace
├── plugin/                   the installable Claude Code plugin (manifest, hooks, skills)
├── docs/
│   ├── ROADMAP.md            phases, gates, acceptance tests
│   ├── ARCHITECTURE.md       components, data flow, trust boundary
│   ├── POLICY.md             policy language v1 (rules, tags, precedence)
│   └── DECISIONS.md          ADRs — why hooks not scraping, why rules not ML, etc.
├── hooks/                    Claude Code hook scripts (PreToolUse router, PostToolUse/Stop loggers)
├── router/                   policy evaluation → dispatch decision
├── worker/                   MCP server wrapping Ollama/vLLM; tool set v1 (refactor_bulk, summarize_files, classify)
├── metering/                 jsonl recorder + session summary + savings estimator
├── examples/                 sample policies, sample sessions, ON/OFF benchmark scripts
└── tests/                    functional · policy-egress · savings A/B · quality scoring
```

## What this repo is NOT

- Not a new coding agent or TUI (see DECISIONS.md — an interface fork of opencode/OpenClaude is a Phase 3 *option*, gated on hooks+MCP proving insufficient).
- Not a token reseller. Bring your own subscription; your credentials never leave your terminal.
- Not a DLP. Policy blocks routes, not intent. Pair it with your existing controls.

## Relationship to Agent Fabric

Agent Fabric provides identity, L3 connectivity, zones, and metering across machines. This repo consumes those to place workers anywhere and keep addresses stable — but Phase 0–1 run on a single machine with none of it, on purpose.

## License

Apache-2.0 (proposed).
