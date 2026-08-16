# Roadmap — phases, gates, acceptance tests

Rule: **no phase starts until the previous phase's gate number exists.** Numbers, not intentions.

## Phase 0 — Instrument (1 week)
**Goal:** know the workload before optimizing it.
- Install PostToolUse / SubagentStop / Stop hook loggers only. No router.
- Record per session: tool calls, subagent spawns, task shapes, token usage.
- **Deliverable:** `examples/phase0-report.md` — task distribution + share that is *local-eligible* (verifiable, repetitive: refactor / summarize / classify / test-scaffold).
- **Gate:** local-eligible share ≥ 30% → Phase 1. Below → revisit value hypothesis (document why).

## Phase 1 — Minimal dispatch loop (2 weeks)
**Goal:** hook → router → local worker → result back into Claude → savings report, on one machine.
- `worker/`: MCP server on Ollama (fork an existing OSS delegate server; do not reinvent). Tools v1: `refactor_bulk`, `summarize_files`, `classify`.
- `router/`: PreToolUse hook. **Deterministic rules only:** path tags (`secrets/`, `repo:corp` → LOCAL forced), task tags (v1 tools → LOCAL), else FRONTIER (hook is a no-op).
- `metering/`: append `~/.falcon/metering.jsonl` per dispatch; print session summary on Stop.
- **Acceptance tests (all four must pass):**
  1. *Functional* — refactor request routes LOCAL, worker output re-enters Claude context, final answer is correct.
  2. *Policy* — a `secrets/` task never reaches FRONTIER (blocked-egress log entry present).
  3. *Savings* — 10 identical tasks × 3 task types, router ON vs OFF: frontier tokens **−40% or better**.
  4. *Quality* — local outputs scored by a frontier review pass; pass rate recorded per task type (this starts the "peer-grade task list").
- **Gate:** savings ≥ 40% AND quality pass rate ≥ 85% on at least two task types.

## Phase 2 — Multi-node & zones (3 weeks)
**Goal:** same loop with the worker on another PC or an AWS node; router unaware of location.
- Place worker on remote node over Agent Fabric L3; same address, same identity.
- AWS node: vLLM multi-LoRA (one base, per-zone adapters). Reproduce Punica-class throughput on our hardware.
- Terminal multiplexer integration: worker streams visible as panes (visibility only; no session fan-out yet).
- Zone v0: `falcon zone init / add-node`; per-zone policy + metering.
- **Tests:** remote-worker latency overhead vs local · zone isolation (zone A worker cannot read zone B data) · multi-LoRA throughput · savings dashboard first render.
- **Gate:** remote overhead < 15% on p50 latency; isolation test passes; dashboard renders real data.

## Phase 3 — Intelligence layer (4+ weeks)
- Verification loops / voting across 3 workers; measure *effective N* (Ringelmann-style), publish "peer-grade task list".
- KV / context sharing across workers (LMCache); measure TTFT and cost.
- Cost-objective router v2 (minimize frontier share under quality constraint).
- Session fan-out via Agent SDK / headless, in the user's session context (ToS review runs in parallel).
- **Interface gate:** only if hooks+MCP UX proves insufficient → evaluate a `falcon` TUI forked from opencode/OpenClaude. Not before.

## First 10 days (checklist)
```
D1   confirm hook events + JSON schema; install one PostToolUse logger        (Phase 0 starts)
D2   Ollama + 32B coder running; fork one OSS delegate MCP server; run it
D3   register MCP with Claude Code; "summarize these files" delegates by hand
D5   PreToolUse router v0 (3 rules); rewrite Task/Agent spawn to local subagent
D7   metering.jsonl + Stop-hook session summary
D8-10 ON/OFF experiment: 10 runs × 3 task types → savings % and quality table
D10  Phase 0 distribution report + Phase 1 numbers → gate decision
```
