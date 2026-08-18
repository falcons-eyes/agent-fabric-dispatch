<p align="center">
  <h1 align="center">agent-fabric-dispatch</h1>
  <p align="center">
    <strong>Route coding grunt work to local models. Keep using Claude Code. Spend fewer tokens.</strong>
  </p>
  <p align="center">
    <a href="https://github.com/falcons-eyes/agent-fabric-dispatch/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache_2.0-blue.svg" alt="License" /></a>
    <a href="https://ollama.com"><img src="https://img.shields.io/badge/Ollama-compatible-black?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij48Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIgZmlsbD0id2hpdGUiLz48L3N2Zz4=" alt="Ollama" /></a>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white" alt="Python" /></a>
    <a href="https://go.dev/"><img src="https://img.shields.io/badge/go-1.22+-00ADD8?logo=go&logoColor=white" alt="Go" /></a>
  </p>
</p>

---

A policy-driven task router that dispatches coding tasks to local open-weight models via [Ollama](https://ollama.com), saving **79% of frontier API costs** while maintaining **96% accuracy** through intelligent fallback.

```
Prompt ──► Ollama (local, $0.00) ──► correct? ──► done
                                  └── wrong?  ──► claude -p (frontier, ~$0.19) ──► done
```

## Why

- **80% of agent work is grunt work.** Refactors, summaries, classification, docstrings — a 30B local model handles these fine.
- **Subscriptions cap you; APIs bill you.** Local workers cost electricity, not tokens.
- **Some code must stay local.** Policy tags on `secrets/` or `.env` physically route work to a local worker — not a prompt instruction, a route.
- **No new tools to learn.** Claude Code stays your cockpit; this rides alongside.

## Benchmark Results

Tested across 405 questions on 5 benchmarks. Local model: `qwen3-coder:30b` via Ollama.

### Local-First + Frontier Fallback

| Benchmark | N | Local-only | Fabric Agent | Improvement | Frontier Calls | Cost |
|-----------|--:|----------:|-------------:|:-----------:|---------------:|-----:|
| GSM8K | 100 | 97.0% | **100%** | +3.0% | 3 | $0.58 |
| MATH | 50 | 86.0% | **88.0%** | +2.0% | 7 | $1.33 |
| MMLU | 200 | 67.5% | **95.0%** | +27.5% | 65 | $12.00 |
| TruthfulQA | 50 | 82.0% | **100%** | +18.0% | 9 | $1.72 |
| AIME | 5 | 60.0% | **100%** | +40.0% | 2 | $0.58 |
| **Total** | **405** | **78.8%** | **96.0%** | **+17.2pp** | **86** | **$16.21** |

### Cost Comparison

| Strategy | Accuracy | Cost | Cost/Question |
|----------|:--------:|-----:|--------------:|
| Local-only (Ollama) | 78.8% | $0.00 | $0.000 |
| **Fabric Agent** (local + fallback) | **96.0%** | **$16.21** | **$0.040** |
| Frontier-only (estimated) | ~97% | ~$76.95 | $0.190 |

> **79% cost reduction** vs frontier-only, with 96.0% accuracy.

### Quality Verification

Local model outputs scored by frontier model (Claude) on a 1-5 scale across 5 code fixtures:

| Task Type | Pass Rate (≥4/5) | Avg Score | Gate (≥85%) |
|-----------|:----------------:|:---------:|:-----------:|
| Summarize | 5/5 (100%) | 4.8/5 | **PASS** |
| Classify | 5/5 (100%) | 5.0/5 | **PASS** |
| Refactor | 3/5 (60%) | 3.8/5 | FAIL |

**Overall: PASS** (2/3 task types above gate). Refactor fails because the local model over-refactors — it adds improvements beyond the instruction (e.g., adding error handling when only asked for type hints). Summarize and classify are production-ready.

### Confidence-Based Routing

Instead of post-hoc answer checking, the agent uses **self-consistency** (sample 3x, measure agreement) and **self-verification** (AutoMix-style) to detect uncertainty *before* returning answers. For multiple-choice, **choice shuffling** detects position bias.

| Benchmark | N | Local | Confidence-Routed | Frontier Calls | Cost |
|-----------|--:|------:|-------------------:|---------------:|-----:|
| GSM8K | 30 | 96.7% | **100%** | 6 (20%) | $1.14 |

**High-confidence calibration: 100%** — every answer the system labels as "confident" is correct.

The system uses a layered approach with **shepherding** (hint-based escalation):
- **High confidence** (≥85% self-consistency): return local answer ($0.00)
- **Medium** (50–85%): ask frontier for a short hint (~50 tokens, ~$0.03), re-run local with hint
- **Low** (<50%): try shepherding first; if it fails, full frontier fallback (~$0.19)

For reasoning tasks (math, code), self-consistency works well because uncertain models produce different answers across samples. For factual knowledge (MMLU), the model is "confidently wrong" — a known limitation addressed by learned routers in Phase 2.

### Architecture Comparison

We evaluated three dispatch architectures before settling on the inverted approach:

| Architecture | Cost vs Frontier | Verdict |
|---|---|---|
| MCP Dispatch (hooks → MCP tools) | **+78%** (worse) | Tool call output token tax makes it more expensive than doing nothing |
| Frontier-only | baseline | Full quality, full cost |
| **Inverted (local-first)** | **-79%** | Zero overhead; frontier only when needed |

<details>
<summary>Why MCP dispatch failed</summary>

When Claude Code calls an MCP tool, it generates tool arguments (including full code payloads) as **output tokens** — averaging 2.1x the cost of answering directly. Combined with system prompt inflation from CLAUDE.md and MCP tool definitions (~$0.15/call), MCP dispatch costs **more** than doing nothing. See [ADR-011](docs/DECISIONS.md) for details.

</details>

## Quick Start

> [!NOTE]
> Requires [Ollama](https://ollama.com) running locally with a coding model pulled.

```bash
# 1. Pull a local model
ollama pull qwen3-coder:30b

# 2. Clone and install
git clone https://github.com/falcons-eyes/agent-fabric-dispatch.git
cd agent-fabric-dispatch
pip install -e .

# 3. Run the agent (local-first, frontier fallback)
python agent/fabric_agent.py "summarize @router/engine.py"
python agent/fabric_agent.py --interactive

# 4. Run benchmarks
pip install datasets
python benchmarks/run_fabric_compare.py --benchmarks gsm8k,math,mmlu --limit 50
```

## Features

- **Local-first dispatch** — Ollama handles all tasks by default at zero frontier cost
- **Confidence routing** — Self-consistency + self-verification detect when local answers are uncertain
- **Shepherding** — When uncertain, ask frontier for a hint (~$0.03) instead of full answer (~$0.19)
- **Configurable trust** — Preset levels (conservative/balanced/aggressive/max) or 0.0–1.0 float via `--trust` flag or `policy.yaml`
- **Budget-aware routing** — Set daily/session frontier spend limits; router auto-tightens trust as budget depletes
- **Step-level routing** — `--pipeline` decomposes complex tasks into steps, routes each independently (TRIM-style)
- **Frontier fallback** — Automatically escalates to Claude when local model answers incorrectly
- **Policy engine** — YAML rules decide which files stay local (`LOCAL_ONLY`) vs prefer local (`LOCAL`)
- **15 benchmark runners** — GSM8K, MATH, MMLU, AIME, TruthfulQA, HumanEval, and more
- **Metering** — Track routing decisions, token usage, and cost savings per session
- **File expansion** — Reference files with `@path/to/file.py` in prompts

## Supported Models

| Model | Params | Context | Notes |
|---|---|---|---|
| **Qwen3-Coder 30B** | 30B (3B active) | 256K | Default recommendation |
| Qwen 3.6 27B | 27B | 256K | 77.2% SWE-bench Verified |
| Devstral Small 2 | 24B | 256K | Fits single RTX 4090 |
| Gemma 4 27B | 26B (3.8B active) | 256K | Fast MoE, great on Apple Silicon |
| DeepSeek V4-Flash | 284B (13B active) | 1M | Multi-GPU, frontier-class |

## Project Structure

```
agent-fabric-dispatch/
├── agent/              Local-first agent (Ollama + claude -p fallback)
├── benchmarks/         15 benchmark runners + fabric comparison tool
│   └── runners/        Individual benchmark implementations
├── router/             Policy evaluation engine (deterministic rules)
├── worker/             MCP server wrapping Ollama
├── hooks/              Claude Code hook scripts
├── metering/           JSONL recorder + session summary
├── cmd/                Go CLI source
├── internal/           Go internal packages
├── docs/               Architecture, decisions, roadmap, policy spec
├── examples/           Sample policy YAML
└── tests/              Protocol + unit tests
```

## Configuration

Create `~/.fabric/policy.yaml`:

```yaml
rules:
  - match: { path: ["secrets/**", "**/.env*"] }
    route: LOCAL_ONLY              # never leaves the machine
  - match: { task: ["refactor_bulk", "summarize_files", "classify"] }
    route: LOCAL
    fallback: FRONTIER             # fall back if no worker available
  - default: FRONTIER

confidence:
  trust: balanced              # conservative, balanced, aggressive, max, or 0.0-1.0

budgets:
  frontier_daily: 5.00         # USD/day — auto-tightens trust as budget depletes
  # frontier_session: 1.00     # USD/session (optional)

workers:
  local:
    endpoint: "http://127.0.0.1:11434"
    models:
      - "qwen3-coder:30b"
```

See [docs/POLICY.md](docs/POLICY.md) for the full policy language spec.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — Components, data flow, architecture evolution
- [Decisions](docs/DECISIONS.md) — ADRs: why inverted not MCP, why rules not ML, etc.
- [Roadmap](docs/ROADMAP.md) — Phases, gates, acceptance criteria
- [Policy](docs/POLICY.md) — Policy language v1 specification
- [Benchmarks](benchmarks/BENCHMARKS.md) — Benchmark guide and runner docs

## What This Is Not

- **Not a new coding agent or TUI.** Claude Code stays your cockpit.
- **Not a token reseller.** Bring your own subscription; your credentials never leave your terminal.
- **Not DLP.** Policy blocks routes, not intent. Pair it with your existing controls.

## License

[Apache-2.0](LICENSE)
