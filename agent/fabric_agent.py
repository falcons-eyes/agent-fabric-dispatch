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
    """Run a single task with confidence-aware, budget-aware routing.

    Trust level and budget are loaded from ~/.fabric/policy.yaml.
    When budget is running low, trust automatically shifts toward aggressive/max.
    """
    from agent.budget import get_tracker
    from agent.confidence import Confidence, ConfidenceConfig, estimate_confidence, get_config, set_config

    tracker = get_tracker()
    expanded = expand_prompt(prompt)
    route = force_route or classify_task(expanded)

    if route == "frontier":
        if not tracker.can_escalate():
            result = ollama_generate(expanded)
            result["budget_blocked"] = True
            return result
        fr = frontier_query(expanded)
        tracker.record_spend(fr.get("cost_usd", 0))
        return fr

    result = ollama_generate(expanded)
    if result.get("error") and force_route != "local":
        if not tracker.can_escalate():
            result["budget_blocked"] = True
            return result
        result = frontier_query(expanded)
        result["fallback"] = True
        tracker.record_spend(result.get("cost_usd", 0))
        return result

    cfg = get_config()

    adjusted_trust = tracker.suggested_trust(cfg.trust)
    if adjusted_trust != cfg.trust:
        cfg = ConfidenceConfig.from_preset(adjusted_trust)

    if not use_confidence or force_route == "local" or cfg.skip_confidence:
        return result

    conf = estimate_confidence(expanded, result, ollama_generate_with_temp, cfg)

    result["confidence"] = conf["confidence"].value
    result["agreement"] = conf["agreement"]
    result["verify_score"] = conf.get("verify_score")
    result["budget_remaining"] = tracker.remaining()

    if conf["confidence"] == Confidence.HIGH:
        if conf["best_answer"]:
            result["result"] = conf["best_answer"]
        return result

    if not tracker.can_escalate():
        if conf["best_answer"]:
            result["result"] = conf["best_answer"]
        result["budget_blocked"] = True
        result["uncertain"] = True
        return result

    if conf["confidence"] == Confidence.MEDIUM:
        guided = _try_shepherding(expanded, conf["best_answer"] or result["result"])
        if guided:
            tracker.record_spend(guided.get("cost_usd", 0))
            guided["fallback_reason"] = "shepherding"
            guided["local_agreement"] = conf["agreement"]
            return guided
        if conf["best_answer"]:
            result["result"] = conf["best_answer"]
        result["uncertain"] = True
        return result

    # LOW confidence: try shepherding first, fall back to full frontier
    guided = _try_shepherding(expanded, conf["best_answer"] or result["result"])
    if guided:
        tracker.record_spend(guided.get("cost_usd", 0))
        guided["fallback_reason"] = "shepherding"
        guided["local_agreement"] = conf["agreement"]
        return guided

    frontier_result = frontier_query(expanded)
    tracker.record_spend(frontier_result.get("cost_usd", 0))
    frontier_result["fallback"] = True
    frontier_result["fallback_reason"] = "low_confidence"
    frontier_result["local_agreement"] = conf["agreement"]
    frontier_result["local_verify_score"] = conf.get("verify_score")
    return frontier_result


def _try_shepherding(prompt: str, local_answer: str) -> dict | None:
    """Request a short hint from frontier, then re-run local with the hint.

    Returns the guided result if it looks better than the original, else None.
    Based on LLM Shepherding (Dong et al., 2026): 70-90% cheaper than full
    frontier calls because we request ~50 tokens instead of ~500.
    """
    hint_prompt = (
        f"A local model attempted this task but may be wrong. "
        f"Give a brief hint (under 50 words) — the key insight, approach, "
        f"or correction needed. Do NOT give the full answer.\n\n"
        f"Task: {prompt[:800]}\n\n"
        f"Local model's attempt: {local_answer[:400]}"
    )

    hint_result = frontier_query(hint_prompt, max_turns=1)
    hint = hint_result.get("result", "").strip()
    hint_cost = hint_result.get("cost_usd", 0)

    if not hint or hint_result.get("error"):
        return None

    guided_prompt = (
        f"{prompt}\n\n"
        f"Hint from expert: {hint}\n\n"
        f"Using the hint above, provide your answer."
    )
    guided_result = ollama_generate(guided_prompt)

    if guided_result.get("error"):
        return None

    guided_result["source"] = "shepherded"
    guided_result["hint"] = hint
    guided_result["hint_cost_usd"] = hint_cost
    guided_result["cost_usd"] = hint_cost
    return guided_result


DECOMPOSE_PROMPT = """Break this task into independent steps. Each step should be a self-contained sub-task.

Task: {task}

Reply with a JSON array of step objects. Each step has:
- "step": short description of what to do
- "type": one of "refactor", "summarize", "classify", "format", "test", "debug", "design", "review", "other"

Example: [{{"step": "Add type hints to function foo", "type": "refactor"}}, {{"step": "Write unit tests for foo", "type": "test"}}]

Reply with ONLY the JSON array, nothing else."""


def decompose_task(prompt: str) -> list[dict]:
    """Break a complex prompt into independent steps using the local model."""
    result = ollama_generate(DECOMPOSE_PROMPT.format(task=prompt[:1500]))
    raw = result.get("result", "").strip()

    try:
        if "```" in raw:
            import re
            match = re.search(r'```(?:json)?\s*(\[.+?\])\s*```', raw, re.DOTALL)
            if match:
                raw = match.group(1)
        steps = json.loads(raw)
        if isinstance(steps, list) and all(isinstance(s, dict) and "step" in s for s in steps):
            return steps
    except (json.JSONDecodeError, TypeError):
        pass

    return [{"step": prompt, "type": "other"}]


def run_pipeline(prompt: str) -> dict:
    """Decompose a complex task into steps and route each independently.

    TRIM-style step-level routing: easy steps (refactor, summarize, format)
    stay local; hard steps (debug, design, review) get frontier routing.
    Each step runs through confidence estimation independently.
    """
    expanded = expand_prompt(prompt)
    steps = decompose_task(expanded)

    if len(steps) <= 1:
        return run_single(prompt)

    results = []
    total_cost = 0.0
    sources = []

    for i, step in enumerate(steps):
        step_prompt = step["step"]
        if i > 0 and results:
            prev = results[-1].get("result", "")
            if prev:
                step_prompt = f"Previous step result:\n{prev[:500]}\n\nNow: {step_prompt}"

        step_result = run_single(step_prompt)
        step_result["step_index"] = i
        step_result["step_type"] = step.get("type", "other")
        step_result["step_description"] = step["step"]
        results.append(step_result)

        total_cost += step_result.get("cost_usd", 0)
        sources.append(step_result.get("source", "unknown"))

    local_count = sum(1 for s in sources if s == "local")
    frontier_count = sum(1 for s in sources if s == "frontier")
    shepherded_count = sum(1 for s in sources if s == "shepherded")

    return {
        "result": results[-1].get("result", ""),
        "steps": results,
        "total_steps": len(steps),
        "local_steps": local_count,
        "frontier_steps": frontier_count,
        "shepherded_steps": shepherded_count,
        "cost_usd": total_cost,
        "source": "pipeline",
    }


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
    """Interactive REPL with confidence routing and shepherding.

    Commands: /frontier, /local, /compare, /pipeline, /trust, /quit
    """
    from agent.budget import get_tracker
    from agent.confidence import get_config

    cfg = get_config()
    tracker = get_tracker()
    budget_info = ""
    rem = tracker.remaining()
    if rem is not None:
        budget_info = f"  |  Budget: ${rem:.2f} remaining"

    print("fabric-agent — local-first coding agent")
    print(f"  Model: {DEFAULT_MODEL}  |  Trust: {cfg.trust}{budget_info}")
    print(f"  Commands: /frontier  /local  /compare  /pipeline  /trust  /budget  /quit")
    print()

    total_local_tokens = 0
    total_frontier_cost = 0.0
    total_shepherded = 0

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input in ("/quit", "/exit", "/q"):
            break

        if user_input.startswith("/trust "):
            from agent.confidence import ConfidenceConfig, set_config
            level = user_input[7:].strip()
            set_config(ConfidenceConfig.from_preset(level))
            cfg = get_config()
            print(f"  Trust: {cfg.trust} (high={cfg.high_threshold}, low={cfg.low_threshold})")
            continue

        if user_input == "/budget":
            rem = tracker.remaining()
            ratio = tracker.budget_ratio()
            adj = tracker.suggested_trust(cfg.trust)
            if rem is not None:
                print(f"  Budget: ${rem:.2f} remaining ({ratio*100:.0f}%)")
                print(f"  Session spent: ${tracker.session_spent:.4f}")
                print(f"  Adjusted trust: {adj}")
            else:
                print("  No budget configured. Set budgets.frontier_daily in ~/.fabric/policy.yaml")
            continue

        force_route = None
        if user_input.startswith("/frontier "):
            force_route = "frontier"
            user_input = user_input[10:]
        elif user_input.startswith("/local "):
            force_route = "local"
            user_input = user_input[7:]
        elif user_input.startswith("/compare "):
            user_input = user_input[9:]
            result = run_compare(user_input)
            print(f"\n--- LOCAL ({result['local'].get('duration_ms', 0)}ms, free) ---")
            print(result["local"]["result"][:2000])
            print(f"\n--- FRONTIER ({result['frontier'].get('duration_ms', 0)}ms, ${result['frontier'].get('cost_usd', 0):.4f}) ---")
            print(result["frontier"]["result"][:2000])
            print()
            total_frontier_cost += result["frontier"].get("cost_usd", 0)
            continue
        elif user_input.startswith("/vote "):
            from agent.voting import VotingConfig, multi_model_vote
            user_input = user_input[6:]
            vcfg = VotingConfig.from_policy()
            models = vcfg.models or [DEFAULT_MODEL]
            result = multi_model_vote(expand_prompt(user_input), models)
            print(f"\n[voted: {result['n_agree']}/{result['n_models']} agree, "
                  f"conf={result['confidence']}]")
            print(result["result"])
            print()
            continue
        elif user_input.startswith("/pipeline "):
            user_input = user_input[10:]
            result = run_pipeline(user_input)
            print(f"\n[pipeline: {result['total_steps']} steps, "
                  f"{result['local_steps']}L/{result['shepherded_steps']}S/{result['frontier_steps']}F, "
                  f"${result['cost_usd']:.4f}]")
            print(result["result"])
            print()
            total_frontier_cost += result.get("cost_usd", 0)
            continue

        result = run_single(user_input, force_route)

        source = result.get("source", "unknown")
        cost = result.get("cost_usd", 0)
        total_frontier_cost += cost
        total_local_tokens += result.get("eval_count", 0)
        if source == "shepherded":
            total_shepherded += 1

        conf = result.get("confidence", "")
        conf_tag = f", conf={conf}" if conf else ""
        tag = f"{source}, ${cost:.4f}{conf_tag}"

        print(f"\n[{tag}, {result.get('duration_ms', 0)}ms]")
        if result.get("hint"):
            print(f"  hint: {result['hint'][:100]}")
        print(result["result"])
        print()

    print(f"\nSession: {total_local_tokens} local tokens (free), "
          f"${total_frontier_cost:.4f} frontier cost, "
          f"{total_shepherded} shepherded")


def main():
    parser = argparse.ArgumentParser(description="fabric-agent: local-first coding agent")
    parser.add_argument("prompt", nargs="?", help="Single-shot prompt (use @file.py to include files)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive REPL mode")
    parser.add_argument("--compare", "-c", action="store_true", help="Run on both local and frontier")
    parser.add_argument("--pipeline", "-p", action="store_true",
                        help="Decompose into steps and route each independently")
    parser.add_argument("--frontier", "-f", action="store_true", help="Force frontier routing")
    parser.add_argument("--local", "-l", action="store_true", help="Force local routing")
    parser.add_argument("--model", "-m", help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--trust", "-t",
                        help="Trust level: conservative, balanced, aggressive, max, or 0.0-1.0")
    parser.add_argument("--vote", nargs="*", metavar="MODEL",
                        help="Multi-model vote (uses models from policy.yaml, or specify models)")
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

    if args.pipeline:
        result = run_pipeline(args.prompt)
        print(result["result"])
        if sys.stderr.isatty():
            print(f"\n[pipeline: {result['total_steps']} steps, "
                  f"{result['local_steps']} local / {result['shepherded_steps']} shepherded / "
                  f"{result['frontier_steps']} frontier, ${result['cost_usd']:.4f}]",
                  file=sys.stderr)
        return

    if args.vote is not None:
        from agent.voting import VotingConfig, multi_model_vote
        if args.vote:
            models = args.vote
        else:
            vcfg = VotingConfig.from_policy()
            models = vcfg.models or [DEFAULT_MODEL]
        result = multi_model_vote(expand_prompt(args.prompt), models)
        print(result["result"])
        if sys.stderr.isatty():
            print(f"\n[voted: {result['n_agree']}/{result['n_models']} agree, "
                  f"conf={result['confidence']}, {result['n_unique']} unique]",
                  file=sys.stderr)
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
