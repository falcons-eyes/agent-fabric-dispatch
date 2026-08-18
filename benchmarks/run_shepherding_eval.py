#!/usr/bin/env python3
"""Evaluate shepherding vs full frontier fallback.

Compares three strategies on the same questions:
  1. Confidence + full frontier fallback (baseline)
  2. Confidence + shepherding (hint ~$0.03, then local re-run)
  3. Always-frontier (cost ceiling)

Measures: accuracy, cost, frontier token usage, shepherd success rate.

Usage:
    python3 benchmarks/run_shepherding_eval.py --benchmark gsm8k --limit 30
    python3 benchmarks/run_shepherding_eval.py --benchmark gsm8k --limit 30 --trust conservative
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.confidence import (
    Confidence,
    ConfidenceConfig,
    estimate_confidence,
    set_config,
)
from agent.agent_fabric import (
    _try_shepherding,
    frontier_query,
    ollama_generate,
    ollama_generate_with_temp,
)
from benchmarks.run_confidence_eval import check_answer, load_gsm8k, load_mmlu


def run_eval(benchmark: str, limit: int, trust: str = "balanced"):
    cfg = ConfidenceConfig.from_preset(trust)
    set_config(cfg)

    print(f"\n{'='*70}")
    print(f"Shepherding Evaluation")
    print(f"  Benchmark: {benchmark}  |  N: {limit}  |  Trust: {trust}")
    print(f"  Thresholds: high={cfg.high_threshold}, low={cfg.low_threshold}")
    print(f"{'='*70}\n")

    if benchmark == "gsm8k":
        items = load_gsm8k(limit)
    elif benchmark == "mmlu":
        items = load_mmlu(limit)
    else:
        print(f"Unknown benchmark: {benchmark}")
        return

    if not items:
        print("No items loaded.")
        return

    results = []
    stats = {
        "total": 0,
        "local_only_correct": 0,
        "full_fallback_correct": 0,
        "shepherded_correct": 0,
        "full_fallback_cost": 0.0,
        "shepherded_cost": 0.0,
        "frontier_baseline_cost": 0.0,
        "full_fallback_frontier_calls": 0,
        "shepherd_hint_calls": 0,
        "shepherd_success": 0,
        "shepherd_fail_to_full": 0,
        "confidence_dist": {"high": 0, "medium": 0, "low": 0},
    }

    for i, item in enumerate(items):
        stats["total"] += 1
        print(f"[{i+1}/{len(items)}] ", end="", flush=True)

        local_result = ollama_generate(item["prompt"])
        local_answer = local_result.get("result", "")
        local_correct = check_answer(local_answer, item["answer"])
        if local_correct:
            stats["local_only_correct"] += 1

        conf = estimate_confidence(item["prompt"], local_result, ollama_generate_with_temp)
        confidence = conf["confidence"]
        stats["confidence_dist"][confidence.value] += 1

        best = conf["best_answer"] or local_answer

        if confidence == Confidence.HIGH:
            full_correct = check_answer(best, item["answer"])
            shep_correct = full_correct
            source = "high"
            if full_correct:
                stats["full_fallback_correct"] += 1
                stats["shepherded_correct"] += 1
        else:
            # Strategy 1: full frontier fallback
            frontier_result = frontier_query(item["prompt"])
            full_cost = frontier_result.get("cost_usd", 0)
            stats["full_fallback_cost"] += full_cost
            stats["full_fallback_frontier_calls"] += 1
            full_correct = check_answer(frontier_result.get("result", ""), item["answer"])
            if full_correct:
                stats["full_fallback_correct"] += 1

            # Strategy 2: shepherding
            stats["shepherd_hint_calls"] += 1
            guided = _try_shepherding(item["prompt"], best)
            if guided and not guided.get("error"):
                shep_answer = guided.get("result", "")
                shep_correct = check_answer(shep_answer, item["answer"])
                shep_cost = guided.get("cost_usd", 0)
                stats["shepherded_cost"] += shep_cost

                if shep_correct:
                    stats["shepherded_correct"] += 1
                    stats["shepherd_success"] += 1
                    source = f"shepherded(${shep_cost:.3f})"
                else:
                    # Shepherding failed — would fall back to full frontier
                    stats["shepherd_fail_to_full"] += 1
                    stats["shepherded_cost"] += full_cost
                    if full_correct:
                        stats["shepherded_correct"] += 1
                    source = f"shep→full(${shep_cost + full_cost:.3f})"
            else:
                # Hint failed — fall back to full frontier
                stats["shepherd_fail_to_full"] += 1
                stats["shepherded_cost"] += full_cost
                if full_correct:
                    stats["shepherded_correct"] += 1
                source = f"hint_fail→full(${full_cost:.3f})"

        # Frontier baseline cost estimate
        if i < 5:
            baseline = frontier_query(item["prompt"])
            stats["frontier_baseline_cost"] += baseline.get("cost_usd", 0)
        else:
            stats["frontier_baseline_cost"] += stats["frontier_baseline_cost"] / max(i, 1)

        agr = f"agr={conf['agreement']:.2f}"
        status_full = "OK" if full_correct else "FAIL"
        status_shep = "OK" if shep_correct else "FAIL"
        print(f"{confidence.value:6s} {agr} | full={status_full} shep={status_shep} [{source}]")

        results.append({
            "question": item["question"][:80],
            "expected": item["answer"],
            "local_correct": local_correct,
            "confidence": confidence.value,
            "agreement": conf["agreement"],
            "full_correct": full_correct,
            "shep_correct": shep_correct,
            "source": source,
        })

    n = stats["total"]
    print(f"\n{'='*70}")
    print(f"RESULTS ({benchmark}, N={n}, trust={trust})")
    print(f"{'='*70}")

    lo_acc = stats["local_only_correct"] / n * 100
    ff_acc = stats["full_fallback_correct"] / n * 100
    sh_acc = stats["shepherded_correct"] / n * 100

    print(f"\n  Accuracy:")
    print(f"    Local-only:             {stats['local_only_correct']}/{n} ({lo_acc:.1f}%)")
    print(f"    Full frontier fallback: {stats['full_fallback_correct']}/{n} ({ff_acc:.1f}%)")
    print(f"    Shepherded:             {stats['shepherded_correct']}/{n} ({sh_acc:.1f}%)")

    print(f"\n  Cost:")
    print(f"    Local-only:             $0.00")
    print(f"    Full frontier fallback: ${stats['full_fallback_cost']:.4f}")
    print(f"    Shepherded:             ${stats['shepherded_cost']:.4f}")
    est_frontier = stats["frontier_baseline_cost"]
    print(f"    Always-frontier (est):  ${est_frontier:.4f}")

    if stats["full_fallback_cost"] > 0:
        savings = (1 - stats["shepherded_cost"] / stats["full_fallback_cost"]) * 100
        print(f"\n  Shepherding savings vs full fallback: {savings:.0f}%")

    print(f"\n  Confidence distribution:")
    for level in ["high", "medium", "low"]:
        count = stats["confidence_dist"][level]
        pct = count / n * 100
        print(f"    {level:8s}: {count:3d} ({pct:.0f}%)")

    print(f"\n  Shepherding details:")
    print(f"    Hint calls:            {stats['shepherd_hint_calls']}")
    print(f"    Shepherd success:      {stats['shepherd_success']}")
    print(f"    Shepherd→full fallback: {stats['shepherd_fail_to_full']}")
    if stats["shepherd_hint_calls"]:
        sr = stats["shepherd_success"] / stats["shepherd_hint_calls"] * 100
        print(f"    Shepherd success rate: {sr:.0f}%")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"shepherding_{benchmark}_{n}_{trust}.json"
    out_path.write_text(json.dumps({
        "benchmark": benchmark,
        "n": n,
        "trust": trust,
        "stats": stats,
        "results": results,
    }, indent=2, default=str))
    print(f"\n  Results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Shepherding evaluation")
    parser.add_argument("--benchmark", "-b", default="gsm8k", choices=["gsm8k", "mmlu"])
    parser.add_argument("--limit", "-n", type=int, default=30)
    parser.add_argument("--trust", "-t", default="balanced",
                        help="Trust level: conservative, balanced, aggressive, max, or 0.0-1.0")
    args = parser.parse_args()

    run_eval(args.benchmark, args.limit, args.trust)


if __name__ == "__main__":
    main()
