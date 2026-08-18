"""Policy evaluation engine — deterministic, stateless, per-call.

Loads policy from ~/.fabric/policy.yaml and evaluates rules top-down.
First matching rule wins. Default is FRONTIER (no-op).
"""

import fnmatch
import os
from pathlib import Path

import yaml


def _default_policy_path() -> Path:
    return Path(os.environ.get("FABRIC_POLICY_PATH", str(Path.home() / ".fabric" / "policy.yaml")))


def load_policy(path: Path | None = None) -> dict:
    """Load the policy YAML file."""
    policy_path = path or _default_policy_path()
    if not policy_path.exists():
        return {"rules": [{"default": "FRONTIER"}]}

    with open(policy_path) as f:
        return yaml.safe_load(f) or {"rules": [{"default": "FRONTIER"}]}


def evaluate_policy(context: dict, policy: dict | None = None) -> dict:
    """Evaluate policy rules against the given context.

    Args:
        context: Dict with keys: tool_name, paths (list[str]), task_type (str).
        policy: Optional pre-loaded policy dict. If None, loads from disk.

    Returns:
        Dict with keys: route, reason, worker, fallback.
    """
    if policy is None:
        policy = load_policy()

    rules = policy.get("rules", [])
    budgets = policy.get("budgets", {})
    workers = policy.get("workers", {})

    cwd = context.get("cwd", "")

    for i, rule in enumerate(rules):
        # Default rule (catch-all)
        if "default" in rule:
            return {
                "route": rule["default"],
                "reason": f"rule:{i}:default",
                "worker": None,
                "fallback": None,
            }

        match = rule.get("match", {})
        if _matches(match, context, budgets, cwd):
            route = rule.get("route", "FRONTIER")
            fallback = rule.get("fallback")

            # Determine worker target
            worker = None
            if route in ("LOCAL", "LOCAL_ONLY"):
                worker = "local"
                # LOCAL_ONLY is a hard wall — never falls back
                # LOCAL checks worker availability and may fallback
                if route == "LOCAL" and fallback:
                    if not _worker_available(workers.get("local", {})):
                        return {
                            "route": fallback,
                            "reason": f"rule:{i}:fallback(worker_unavailable)",
                            "worker": None,
                            "fallback": None,
                        }

            return {
                "route": route,
                "reason": f"rule:{i}:match",
                "worker": worker,
                "fallback": fallback,
            }

    # No rule matched — default FRONTIER
    return {
        "route": "FRONTIER",
        "reason": "no_match:default",
        "worker": None,
        "fallback": None,
    }


def _matches(match: dict, context: dict, budgets: dict, cwd: str = "") -> bool:
    """Check if a rule's match criteria apply to the context."""
    for key, pattern in match.items():
        if key == "path":
            paths = context.get("paths", [])
            if not _match_paths(pattern, paths, cwd):
                return False

        elif key == "task":
            task_type = context.get("task_type", "")
            task_patterns = pattern if isinstance(pattern, list) else [pattern]
            if task_type not in task_patterns:
                return False

        elif key == "repo_tag":
            repo_tag = context.get("repo_tag", "")
            if repo_tag != pattern:
                return False

        elif key == "budget_exceeded":
            # Budget checking requires metering data
            if not _budget_exceeded(pattern, budgets):
                return False

    return True


def _match_paths(patterns: list[str], paths: list[str], cwd: str = "") -> bool:
    """Check if any path matches any pattern (glob).

    Tries both the original path and a cwd-relative version so absolute paths
    from Claude Code match relative policy patterns like 'secrets/**'.
    """
    if not paths:
        return False
    for path in paths:
        candidates = [path]
        if cwd and os.path.isabs(path):
            try:
                candidates.append(os.path.relpath(path, cwd))
            except ValueError:
                pass
        for candidate in candidates:
            for pattern in patterns:
                if fnmatch.fnmatch(candidate, pattern):
                    return True
    return False


def _budget_exceeded(budget_name: str, budgets: dict) -> bool:
    """Check if a named budget has been exceeded."""
    try:
        from agent.budget import get_tracker
        tracker = get_tracker()
        return not tracker.can_escalate()
    except ImportError:
        return False


def _worker_available(worker_config: dict) -> bool:
    """Check if a local worker is available.

    TODO: Phase 1 — health-check the Ollama endpoint.
    """
    endpoint = worker_config.get("endpoint", "")
    if not endpoint:
        return False

    try:
        import urllib.request
        req = urllib.request.Request(f"{endpoint}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False
