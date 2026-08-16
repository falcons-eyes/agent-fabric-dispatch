# Policy language v1 (deterministic)

Rules are evaluated top-down; first match wins; default = FRONTIER.

```yaml
# ~/.falcon/policy.yaml
rules:
  - match: { path: ["secrets/**", "**/.env*"] }        # sensitivity
    route: LOCAL_ONLY          # never FRONTIER, never CLOUD_WORKER
  - match: { repo_tag: "corp" }
    route: LOCAL_ONLY
  - match: { task: ["refactor_bulk", "summarize_files", "classify"] }   # cost
    route: LOCAL
    fallback: FRONTIER          # if no worker healthy
  - match: { budget_exceeded: "frontier_daily" }        # budget
    route: LOCAL
    fallback: QUEUE
  - default: FRONTIER

budgets:
  frontier_daily: 500k        # tokens
workers:
  local:  { endpoint: "http://127.0.0.1:11434", models: ["qwen2.5-coder:32b"] }
  cloud:  { fabric_service: "worker.aws-node-1", zone: "james" }    # Phase 2
```

Semantics
- `LOCAL_ONLY` is a hard wall (egress blocked, logged). `LOCAL` is a preference with fallback.
- Difficulty tagging is **not** in v1 — Claude itself decides what to delegate; we constrain *where* it may go.
- Every decision logs `reason` = the matched rule id.
