#!/usr/bin/env python3
"""Session fan-out: parallel task execution across local models.

Runs multiple independent tasks concurrently against Ollama, with each
task going through the full confidence/shepherding/budget pipeline.

Ollama handles concurrent requests natively (queues them per model,
parallelizes across different models). Fan-out is most effective when:
  - Multiple independent sub-tasks need execution
  - Multiple models are available (true parallelism)
  - Tasks have different complexity (fast tasks don't wait for slow ones)

Usage:
    from agent.fanout import fan_out, fan_out_models

    # Run same prompt on multiple models in parallel
    results = fan_out_models("summarize this code", ["qwen3-coder:30b", "gemma4:27b"])

    # Run multiple prompts in parallel (each through run_single)
    results = fan_out(["summarize @file1.py", "refactor @file2.py", "classify @file3.py"])
"""

import concurrent.futures
import time
from dataclasses import dataclass, field


@dataclass
class FanOutResult:
    results: list[dict] = field(default_factory=list)
    total_cost: float = 0.0
    total_duration_ms: int = 0
    wall_clock_ms: int = 0
    n_tasks: int = 0
    n_succeeded: int = 0
    n_failed: int = 0
    sources: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "results": self.results,
            "total_cost": self.total_cost,
            "total_duration_ms": self.total_duration_ms,
            "wall_clock_ms": self.wall_clock_ms,
            "n_tasks": self.n_tasks,
            "n_succeeded": self.n_succeeded,
            "n_failed": self.n_failed,
            "sources": self.sources,
            "speedup": (self.total_duration_ms / self.wall_clock_ms
                        if self.wall_clock_ms > 0 else 1.0),
        }


def fan_out(prompts: list[str], max_workers: int = 4,
            force_route: str = None) -> FanOutResult:
    """Run multiple prompts in parallel, each through run_single().

    Each task gets independent confidence estimation and routing.
    """
    from agent.agent_fabric import run_single

    result = FanOutResult(n_tasks=len(prompts))
    t0 = time.time()

    def run_task(idx_prompt):
        idx, prompt = idx_prompt
        try:
            r = run_single(prompt, force_route=force_route)
            r["task_index"] = idx
            return r
        except Exception as e:
            return {"task_index": idx, "error": str(e), "result": "",
                    "cost_usd": 0, "source": "error"}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = executor.map(run_task, enumerate(prompts))
        result.results = list(futures)

    result.wall_clock_ms = round((time.time() - t0) * 1000)

    for r in result.results:
        cost = r.get("cost_usd", 0)
        result.total_cost += cost
        result.total_duration_ms += r.get("duration_ms", 0)

        source = r.get("source", "unknown")
        result.sources[source] = result.sources.get(source, 0) + 1

        if r.get("error"):
            result.n_failed += 1
        else:
            result.n_succeeded += 1

    return result


def fan_out_models(prompt: str, models: list[str],
                   max_workers: int = 4) -> FanOutResult:
    """Run the same prompt on multiple models in parallel.

    Unlike voting (which picks majority), this returns all results
    for comparison or aggregation.
    """
    from agent.agent_fabric import ollama_generate

    result = FanOutResult(n_tasks=len(models))
    t0 = time.time()

    def run_model(model):
        try:
            r = ollama_generate(prompt, model=model)
            r["model"] = model
            return r
        except Exception as e:
            return {"model": model, "error": str(e), "result": "",
                    "cost_usd": 0, "source": "error"}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = executor.map(run_model, models)
        result.results = list(futures)

    result.wall_clock_ms = round((time.time() - t0) * 1000)

    for r in result.results:
        result.total_duration_ms += r.get("duration_ms", 0)
        source = r.get("source", "unknown")
        result.sources[source] = result.sources.get(source, 0) + 1

        if r.get("error"):
            result.n_failed += 1
        else:
            result.n_succeeded += 1

    return result


def fan_out_batch(prompts: list[str], batch_size: int = 4,
                  force_route: str = None) -> FanOutResult:
    """Process prompts in batches to avoid overwhelming Ollama.

    Useful when you have many tasks but limited GPU memory — processes
    batch_size tasks at a time.
    """
    all_results = []
    total_cost = 0.0
    total_duration = 0
    t0 = time.time()

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        batch_result = fan_out(batch, max_workers=batch_size,
                               force_route=force_route)
        for r in batch_result.results:
            r["task_index"] = i + r.get("task_index", 0)
        all_results.extend(batch_result.results)
        total_cost += batch_result.total_cost
        total_duration += batch_result.total_duration_ms

    wall_ms = round((time.time() - t0) * 1000)

    result = FanOutResult(
        results=all_results,
        total_cost=total_cost,
        total_duration_ms=total_duration,
        wall_clock_ms=wall_ms,
        n_tasks=len(prompts),
        n_succeeded=sum(1 for r in all_results if not r.get("error")),
        n_failed=sum(1 for r in all_results if r.get("error")),
    )

    for r in all_results:
        source = r.get("source", "unknown")
        result.sources[source] = result.sources.get(source, 0) + 1

    return result
