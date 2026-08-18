"""Unified benchmark runner for agent-fabric-dispatch.

Runs LLM benchmarks against local models (via Ollama) or API endpoints.
Supports selective benchmark execution, result caching, and summary reporting.

Usage:
    python benchmarks/runner.py --model qwen3-coder:30b --benchmarks mmlu,gsm8k
    python benchmarks/runner.py --model qwen3-coder:30b --benchmarks all --limit 100
    python benchmarks/runner.py --list  # show available benchmarks
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

BENCHMARKS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARKS_DIR.parent

# Ensure project root is on sys.path so `benchmarks.*` imports work
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = BENCHMARKS_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class BenchmarkResult:
    benchmark: str
    model: str
    total: int
    correct: int
    score: float
    metric_name: str
    details: dict
    timestamp: str
    duration_s: float


OLLAMA_API_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


def ollama_generate(model: str, prompt: str, temperature: float = 0.0) -> str:
    """Generate a response using Ollama HTTP API.

    Uses /api/generate with stream=false for clean JSON output.
    The CLI (`ollama run`) emits ANSI escape sequences that corrupt
    subprocess captured output, so we use the HTTP API instead.
    """
    url = f"{OLLAMA_API_URL}/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip()
    except urllib.error.URLError as e:
        return f"ERROR: Cannot connect to Ollama at {OLLAMA_API_URL}: {e}"
    except Exception as e:
        return f"ERROR: {e}"


def list_benchmarks():
    """List all available benchmarks with descriptions."""
    from benchmarks.registry import BENCHMARK_REGISTRY

    print("\n=== Available Benchmarks ===\n")
    print(f"{'Category':<18} {'Benchmark':<18} {'HF Dataset':<35} {'Metric':<15} {'Local?'}")
    print("-" * 110)
    for cat, benches in BENCHMARK_REGISTRY.items():
        for b in benches:
            print(f"{cat:<18} {b['name']:<18} {b['hf_path']:<35} {b['metric']:<15} {b['local_eval']}")
    print()


def run_benchmark(name: str, model: str, limit: int = 0, generate_fn=None):
    """Run a single benchmark and return results."""
    from benchmarks.registry import get_runner

    runner_fn = get_runner(name)
    if runner_fn is None:
        print(f"Unknown benchmark: {name}")
        return None

    if generate_fn is None:
        generate_fn = lambda prompt: ollama_generate(model, prompt)

    start = time.time()
    result = runner_fn(model=model, generate_fn=generate_fn, limit=limit)
    elapsed = time.time() - start

    result.duration_s = elapsed
    result.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Save result
    result_path = RESULTS_DIR / f"{name}_{model.replace(':', '_')}_{int(time.time())}.json"
    with open(result_path, "w") as f:
        json.dump(asdict(result), f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Fabric Benchmark Runner")
    parser.add_argument("--model", type=str, default="qwen3-coder:30b", help="Model name (Ollama)")
    parser.add_argument("--benchmarks", type=str, default="all", help="Comma-separated benchmark names or 'all'")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of examples (0=all)")
    parser.add_argument("--list", action="store_true", help="List available benchmarks")
    args = parser.parse_args()

    if args.list:
        list_benchmarks()
        return

    from benchmarks.registry import BENCHMARK_REGISTRY, get_all_names

    if args.benchmarks == "all":
        names = get_all_names()
    else:
        names = [n.strip() for n in args.benchmarks.split(",")]

    print(f"\n=== Fabric Benchmark Runner ===")
    print(f"Model: {args.model}")
    print(f"Benchmarks: {', '.join(names)}")
    print(f"Limit: {args.limit or 'all'}")
    print()

    results = []
    for name in names:
        print(f"--- Running: {name} ---")
        result = run_benchmark(name, args.model, args.limit)
        if result:
            results.append(result)
            print(f"  Score: {result.score:.1%} ({result.correct}/{result.total}) [{result.metric_name}]")
            print(f"  Time: {result.duration_s:.1f}s")
        print()

    # Print summary table
    if results:
        print("\n=== Summary ===\n")
        print(f"{'Benchmark':<18} {'Score':<10} {'Correct/Total':<15} {'Metric':<15} {'Time'}")
        print("-" * 75)
        for r in results:
            print(f"{r.benchmark:<18} {r.score:<10.1%} {r.correct}/{r.total:<13} {r.metric_name:<15} {r.duration_s:.1f}s")


if __name__ == "__main__":
    main()
