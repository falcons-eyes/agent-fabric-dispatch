# Agent Fabric Dispatch

Policy-driven task router for Claude Code that dispatches grunt work to local
open-weight models via Ollama.

## Development

- Go CLI: `make build` (binary at `dist/fabric`)
- Python tests: `pytest tests/ -v`
- Go tests: `go test ./...`
- Benchmarks: `python benchmarks/runner.py --list`

## Architecture

The project uses an **inverted architecture**: Ollama handles all tasks by default
(zero frontier cost), and `claude -p` is called as a subprocess only when frontier
quality is needed. This avoids MCP tool call overhead entirely.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## Policy Rules

Policy is defined in `~/.fabric/policy.yaml`:
- **LOCAL_ONLY**: Sensitive files (secrets/**, .env*) must never leave the machine
- **LOCAL**: Grunt work (refactor, summarize, classify) prefers local execution
- **FRONTIER**: Everything else proceeds normally
