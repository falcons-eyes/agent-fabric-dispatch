---
description: Show or explain the current dispatch policy (~/.falcon/policy.yaml) — which paths are local-only, which task types route to a local worker, and the frontier daily budget.
disable-model-invocation: true
---

Read `~/.falcon/policy.yaml` if present (else say no policy is set and show the example from the plugin README).
Explain each rule in one line: what it matches, where it routes, and the fallback.
Remind the user that LOCAL_ONLY is a hard wall (egress blocked and logged) and LOCAL is a preference with fallback.
