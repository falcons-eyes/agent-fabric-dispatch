#!/usr/bin/env python3
"""fabric-agent: Local-first coding agent with frontier escalation.

Architecture (inverted from MCP approach):
  - Ollama (local model) handles all tasks by default — zero frontier cost
  - When frontier quality is needed, calls `claude -p` as a subprocess
  - Python script orchestrates routing, no MCP overhead

Usage:
    python3 agent/fabric_agent.py "summarize router/engine.py"
    python3 agent/fabric_agent.py --interactive
    python3 agent/fabric_agent.py --compare "refactor this code: ..."
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

OLLAMA_API_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("FABRIC_MODEL", "qwen3-coder:30b")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def set_model(model: str):
    global DEFAULT_MODEL
    DEFAULT_MODEL = model


def ollama_generate(prompt: str, model: str = None) -> dict:
    """Call Ollama /api/generate and return result with timing."""
    model = model or DEFAULT_MODEL
    url = f"{OLLAMA_API_URL}/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }).encode("utf-8")

    t0 = time.time()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            dt = time.time() - t0
            return {
                "result": data.get("response", "").strip(),
                "model": model,
                "eval_count": data.get("eval_count", 0),
                "prompt_eval_count": data.get("prompt_eval_count", 0),
                "duration_ms": round(dt * 1000),
                "cost_usd": 0.0,
                "source": "local",
            }
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        return {"result": "", "cost_usd": 0.0, "source": "local", "error": str(e)}


def ollama_chat(messages: list, model: str = None, tools: list = None) -> dict:
    """Call Ollama /api/chat for conversational mode."""
    model = model or DEFAULT_MODEL
    url = f"{OLLAMA_API_URL}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    if tools:
        payload["tools"] = tools

    t0 = time.time()
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            dt = time.time() - t0
            msg = data.get("message", {})
            return {
                "result": msg.get("content", "").strip(),
                "tool_calls": msg.get("tool_calls", []),
                "model": model,
                "eval_count": data.get("eval_count", 0),
                "duration_ms": round(dt * 1000),
                "cost_usd": 0.0,
                "source": "local",
            }
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        return {"result": "", "cost_usd": 0.0, "source": "local", "error": str(e)}


def frontier_query(prompt: str, max_turns: int = 1) -> dict:
    """Call claude -p for frontier-quality answers."""
    cmd = [
        "claude", "-p", prompt,
        "--output-format", "json",
        "--max-turns", str(max_turns),
    ]

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd="/tmp",
            stdin=subprocess.DEVNULL,
        )
        dt = time.time() - t0

        if result.returncode != 0:
            return {
                "result": f"Error: {result.stderr[:500]}",
                "cost_usd": 0.0,
                "source": "frontier",
                "error": result.stderr[:500],
            }

        data = json.loads(result.stdout)
        usage = data.get("usage", {})
        return {
            "result": data.get("result", ""),
            "output_tokens": usage.get("output_tokens", 0),
            "input_tokens": usage.get("input_tokens", 0),
            "cost_usd": data.get("total_cost_usd", 0),
            "duration_ms": round(dt * 1000),
            "num_turns": data.get("num_turns", 0),
            "source": "frontier",
        }
    except subprocess.TimeoutExpired:
        return {"result": "Error: timeout", "cost_usd": 0.0, "source": "frontier", "error": "timeout"}
    except json.JSONDecodeError as e:
        return {"result": f"Error: {e}", "cost_usd": 0.0, "source": "frontier", "error": str(e)}


def classify_task(prompt: str) -> str:
    """Classify task type from the prompt text.

    Returns: 'local' for grunt work, 'frontier' for complex tasks.
    """
    prompt_lower = prompt.lower()

    # Tasks the local model handles well
    local_keywords = [
        "summarize", "summary",
        "refactor", "rename",
        "classify", "categorize",
        "format", "lint",
        "translate",
        "convert",
        "docstring", "comment",
        "type hint",
    ]

    frontier_keywords = [
        "debug", "fix bug",
        "architect", "design",
        "security", "vulnerability",
        "review",
        "explain why",
        "optimize", "performance",
        "test", "write test",
    ]

    local_score = sum(1 for kw in local_keywords if kw in prompt_lower)
    frontier_score = sum(1 for kw in frontier_keywords if kw in prompt_lower)

    if local_score > frontier_score:
        return "local"
    if frontier_score > local_score:
        return "frontier"

    # Default: local for short prompts, frontier for complex ones
    return "local" if len(prompt) < 500 else "frontier"


def read_file_content(path: str) -> str:
    """Read a file and return its content, resolving relative to PROJECT_ROOT."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        return f"Error: file not found: {p}"
    return p.read_text()


def expand_prompt(prompt: str) -> str:
    """Expand file references in the prompt (e.g., @file.py → file contents)."""
    import re
    def replace_file_ref(match):
        path = match.group(1)
        content = read_file_content(path)
        if content.startswith("Error:"):
            return f"[{content}]"
        return f"\n```\n{content}\n```\n"

    return re.sub(r'@([\w/.\\-]+\.\w+)', replace_file_ref, prompt)


def ollama_generate_with_temp(prompt: str, temperature: float = 0.0) -> dict:
    """Wrapper for self-consistency sampling: generate with a specific temperature."""
    model = DEFAULT_MODEL
    url = f"{OLLAMA_API_URL}/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }).encode("utf-8")

    t0 = time.time()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            dt = time.time() - t0
            return {
                "result": data.get("response", "").strip(),
                "model": model,
                "eval_count": data.get("eval_count", 0),
                "duration_ms": round(dt * 1000),
                "cost_usd": 0.0,
                "source": "local",
            }
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        return {"result": "", "cost_usd": 0.0, "source": "local", "error": str(e)}


def run_single(prompt: str, force_route: str = None, use_confidence: bool = True) -> dict:
    """Run a single task with confidence-aware routing.

    When use_confidence is True (default), local results go through a
    self-consistency + self-verification check before being returned.
    Low-confidence answers escalate to frontier automatically.

    Trust level is loaded from ~/.fabric/policy.yaml (confidence.trust key).
    """
    from agent.confidence import Confidence, ConfidenceConfig, estimate_confidence, get_config

    expanded = expand_prompt(prompt)
    route = force_route or classify_task(expanded)

    if route == "frontier":
        return frontier_query(expanded)

    result = ollama_generate(expanded)
    if result.get("error") and force_route != "local":
        result = frontier_query(expanded)
        result["fallback"] = True
        return result

    cfg = get_config()
    if not use_confidence or force_route == "local" or cfg.skip_confidence:
        return result

    conf = estimate_confidence(expanded, result, ollama_generate_with_temp, cfg)

    result["confidence"] = conf["confidence"].value
    result["agreement"] = conf["agreement"]
    result["verify_score"] = conf.get("verify_score")

    if conf["confidence"] == Confidence.HIGH:
        if conf["best_answer"]:
            result["result"] = conf["best_answer"]
        return result

    if conf["confidence"] == Confidence.LOW:
        frontier_result = frontier_query(expanded)
        frontier_result["fallback"] = True
        frontier_result["fallback_reason"] = "low_confidence"
        frontier_result["local_agreement"] = conf["agreement"]
        frontier_result["local_verify_score"] = conf.get("verify_score")
        return frontier_result

    # MEDIUM confidence: return local but flag uncertainty
    if conf["best_answer"]:
        result["result"] = conf["best_answer"]
    result["uncertain"] = True
    return result


def run_compare(prompt: str) -> dict:
    """Run the same task on both local and frontier, compare results."""
    expanded = expand_prompt(prompt)

    print("  Running LOCAL...", flush=True)
    local = ollama_generate(expanded)

    print("  Running FRONTIER...", flush=True)
    frontier = frontier_query(expanded)

    return {
        "local": local,
        "frontier": frontier,
        "savings_usd": frontier.get("cost_usd", 0),
    }


def run_interactive():
    """Interactive REPL — local-first with /frontier escape hatch."""
    print("fabric-agent — local-first coding agent")
    print(f"  Model: {DEFAULT_MODEL}")
    print(f"  Commands: /frontier <prompt>  /compare <prompt>  /quit")
    print()

    messages = []
    total_local_tokens = 0
    total_frontier_cost = 0.0

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input in ("/quit", "/exit", "/q"):
            break

        force_route = None
        if user_input.startswith("/frontier "):
            force_route = "frontier"
            user_input = user_input[10:]
        elif user_input.startswith("/compare "):
            user_input = user_input[9:]
            expanded = expand_prompt(user_input)
            result = run_compare(expanded)
            print(f"\n--- LOCAL ({result['local'].get('duration_ms', 0)}ms, free) ---")
            print(result["local"]["result"][:2000])
            print(f"\n--- FRONTIER ({result['frontier'].get('duration_ms', 0)}ms, ${result['frontier'].get('cost_usd', 0):.4f}) ---")
            print(result["frontier"]["result"][:2000])
            print()
            total_frontier_cost += result["frontier"].get("cost_usd", 0)
            continue

        expanded = expand_prompt(user_input)
        route = force_route or classify_task(expanded)

        if route == "frontier":
            print(f"  [routing: frontier]", flush=True)
            result = frontier_query(expanded)
            total_frontier_cost += result.get("cost_usd", 0)
            tag = f"frontier, ${result.get('cost_usd', 0):.4f}"
        else:
            result = ollama_generate(expanded)
            total_local_tokens += result.get("eval_count", 0)
            tag = f"local, {result.get('eval_count', 0)} tok"

        print(f"\n[{tag}, {result.get('duration_ms', 0)}ms]")
        print(result["result"])
        print()

    print(f"\nSession: {total_local_tokens} local tokens (free), ${total_frontier_cost:.4f} frontier cost")


def main():
    parser = argparse.ArgumentParser(description="fabric-agent: local-first coding agent")
    parser.add_argument("prompt", nargs="?", help="Single-shot prompt (use @file.py to include files)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive REPL mode")
    parser.add_argument("--compare", "-c", action="store_true", help="Run on both local and frontier")
    parser.add_argument("--frontier", "-f", action="store_true", help="Force frontier routing")
    parser.add_argument("--local", "-l", action="store_true", help="Force local routing")
    parser.add_argument("--model", "-m", help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--trust", "-t",
                        help="Trust level: conservative, balanced, aggressive, max, or 0.0-1.0")
    args = parser.parse_args()

    if args.model:
        set_model(args.model)

    from agent.confidence import ConfidenceConfig, set_config
    if args.trust:
        set_config(ConfidenceConfig.from_preset(args.trust))
    else:
        set_config(ConfidenceConfig.from_policy())

    if args.interactive:
        run_interactive()
        return

    if not args.prompt:
        parser.print_help()
        return

    if args.compare:
        result = run_compare(args.prompt)
        print(f"\n--- LOCAL ({result['local'].get('duration_ms', 0)}ms, $0.00) ---")
        print(result["local"]["result"])
        print(f"\n--- FRONTIER ({result['frontier'].get('duration_ms', 0)}ms, ${result['frontier'].get('cost_usd', 0):.4f}) ---")
        print(result["frontier"]["result"])
        print(f"\nSavings: ${result['savings_usd']:.4f} (100% if local-only)")
        return

    force_route = None
    if args.frontier:
        force_route = "frontier"
    elif args.local:
        force_route = "local"

    result = run_single(args.prompt, force_route)
    print(result["result"])

    if sys.stderr.isatty():
        print(f"\n[{result['source']}, {result.get('duration_ms', 0)}ms, ${result.get('cost_usd', 0):.4f}]",
              file=sys.stderr)


if __name__ == "__main__":
    main()
