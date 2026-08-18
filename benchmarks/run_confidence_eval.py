#!/usr/bin/env python3
"""Evaluate confidence-based routing on benchmark questions.

Compares three strategies:
  1. Local-only (no fallback)
  2. Confidence-based (self-consistency + self-verification → fallback)
  3. Always-frontier (baseline)

Measures: accuracy, frontier calls, cost, confidence calibration.

Usage:
    python3 benchmarks/run_confidence_eval.py --benchmark gsm8k --limit 20
    python3 benchmarks/run_confidence_eval.py --benchmark mmlu --limit 50
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.confidence import Confidence, estimate_confidence
from agent.fabric_agent import (
    frontier_query,
    ollama_generate,
    ollama_generate_with_temp,
)


def load_gsm8k(limit: int) -> list:
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
    except Exception as e:
        print(f"Failed to load gsm8k: {e}")
        return []

    items = []
    for row in ds:
        if len(items) >= limit:
            break
        answer_text = row["answer"].split("####")[-1].strip()
        items.append({
            "question": row["question"],
            "answer": answer_text,
            "prompt": (
                f"Solve this math problem step by step, then give the final answer.\n"
                f"Put your final numerical answer after ####.\n\n{row['question']}"
            ),
        })
    return items


def load_mmlu(limit: int) -> list:
    try:
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", "all", split="test")
    except Exception as e:
        print(f"Failed to load mmlu: {e}")
        return []

    items = []
    choices_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    for row in ds:
        if len(items) >= limit:
            break
        choices = row["choices"]
        correct = choices_map[row["answer"]]
        prompt = (
            f"/no_think Answer this multiple choice question. Reply with ONLY the letter (A, B, C, or D).\n\n"
            f"Question: {row['question']}\n"
            f"A) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}"
        )
        items.append({
            "question": row["question"],
            "answer": correct,
            "prompt": prompt,
        })
    return items


def extract_final_answer(output: str) -> str:
    """Extract the final answer from model output, handling #### delimiter."""
    import re
    if "####" in output:
        after = output.split("####")[-1].strip()
        numbers = re.findall(r'-?[\d,]+\.?\d*', after)
        if numbers:
            return numbers[0].replace(",", "")

    lines = output.strip().split("\n")
    for line in reversed(lines):
        numbers = re.findall(r'-?[\d,]+\.?\d*', line)
        if numbers:
            return numbers[-1].replace(",", "")

    return output.strip()


def check_answer(output: str, expected: str) -> bool:
    import re
    output = output.strip()
    expected = expected.strip().lower().replace(",", "")

    extracted = extract_final_answer(output).lower().replace(",", "")
    if extracted == expected:
        return True

    try:
        if abs(float(extracted) - float(expected)) < 0.01:
            return True
    except (ValueError, OverflowError):
        pass

    if expected in output.lower():
        return True

    letters = re.findall(r'\b([a-d])\b', output.lower())
    if letters and expected in letters:
        return True

    return False


def run_eval(benchmark: str, limit: int, skip_frontier: bool = False):
    print(f"\n{'='*70}")
    print(f"Confidence-Based Routing Evaluation")
    print(f"  Benchmark: {benchmark}")
    print(f"  Limit: {limit}")
    print(f"  Strategy: self-consistency (N=3) + self-verification")
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
        "local_correct": 0,
        "confidence_correct": 0,
        "frontier_calls": 0,
        "frontier_cost": 0.0,
        "confidence_dist": {"high": 0, "medium": 0, "low": 0},
        "escalation_correct": 0,
        "escalation_total": 0,
        "high_correct": 0,
        "high_total": 0,
        "medium_correct": 0,
        "medium_total": 0,
        "low_correct_local": 0,
        "low_total": 0,
    }

    for i, item in enumerate(items):
        stats["total"] += 1
        print(f"[{i+1}/{len(items)}] ", end="", flush=True)

        # Step 1: Local generation
        local_result = ollama_generate(item["prompt"])
        local_answer = local_result.get("result", "")
        local_correct = check_answer(local_answer, item["answer"])
        if local_correct:
            stats["local_correct"] += 1

        # Step 2: Confidence estimation
        conf = estimate_confidence(item["prompt"], local_result, ollama_generate_with_temp)
        confidence = conf["confidence"]
        stats["confidence_dist"][confidence.value] += 1

        # Step 3: Route based on confidence
        if confidence == Confidence.HIGH:
            final_answer = conf["best_answer"]
            final_correct = check_answer(final_answer, item["answer"])
            stats["high_total"] += 1
            if final_correct:
                stats["high_correct"] += 1
                stats["confidence_correct"] += 1
            source = "local"

        elif confidence == Confidence.LOW:
            stats["low_total"] += 1
            if local_correct:
                stats["low_correct_local"] += 1

            if skip_frontier:
                final_answer = conf["best_answer"]
                final_correct = check_answer(final_answer, item["answer"])
                source = "local(low)"
            else:
                frontier_result = frontier_query(item["prompt"])
                stats["frontier_calls"] += 1
                stats["frontier_cost"] += frontier_result.get("cost_usd", 0)
                stats["escalation_total"] += 1
                final_answer = frontier_result.get("result", "")
                final_correct = check_answer(final_answer, item["answer"])
                if final_correct:
                    stats["escalation_correct"] += 1
                    stats["confidence_correct"] += 1
                source = "frontier"

        else:  # MEDIUM
            final_answer = conf["best_answer"]
            final_correct = check_answer(final_answer, item["answer"])
            stats["medium_total"] += 1
            if final_correct:
                stats["medium_correct"] += 1
                stats["confidence_correct"] += 1
            source = "local(med)"

        status = "OK" if final_correct else "FAIL"
        agr = f"agr={conf['agreement']:.2f}"
        vscore = f"v={conf.get('verify_score', '-')}"
        print(f"{status} [{source}] {confidence.value} {agr} {vscore} | expected={item['answer']}")

        results.append({
            "question": item["question"][:80],
            "expected": item["answer"],
            "local_correct": local_correct,
            "confidence": confidence.value,
            "agreement": conf["agreement"],
            "verify_score": conf.get("verify_score"),
            "final_correct": final_correct,
            "source": source,
        })

    # Report
    n = stats["total"]
    print(f"\n{'='*70}")
    print(f"RESULTS ({benchmark}, N={n})")
    print(f"{'='*70}")

    local_acc = stats["local_correct"] / n * 100 if n else 0
    conf_acc = stats["confidence_correct"] / n * 100 if n else 0

    print(f"\n  Local-only accuracy:      {stats['local_correct']}/{n} ({local_acc:.1f}%)")
    print(f"  Confidence-routed accuracy: {stats['confidence_correct']}/{n} ({conf_acc:.1f}%)")
    print(f"  Improvement:              +{conf_acc - local_acc:.1f}pp")

    print(f"\n  Confidence distribution:")
    for level in ["high", "medium", "low"]:
        count = stats["confidence_dist"][level]
        pct = count / n * 100 if n else 0
        print(f"    {level:8s}: {count:3d} ({pct:.0f}%)")

    print(f"\n  Frontier calls:           {stats['frontier_calls']} ({stats['frontier_calls']/n*100:.0f}% of total)")
    print(f"  Frontier cost:            ${stats['frontier_cost']:.4f}")

    if stats["high_total"]:
        print(f"\n  High-confidence accuracy: {stats['high_correct']}/{stats['high_total']} ({stats['high_correct']/stats['high_total']*100:.0f}%)")
    if stats["medium_total"]:
        print(f"  Medium-confidence accuracy: {stats['medium_correct']}/{stats['medium_total']} ({stats['medium_correct']/stats['medium_total']*100:.0f}%)")
    if stats["escalation_total"]:
        print(f"  Escalation success:       {stats['escalation_correct']}/{stats['escalation_total']} ({stats['escalation_correct']/stats['escalation_total']*100:.0f}%)")
    if stats["low_total"]:
        print(f"  Low-conf local accuracy:  {stats['low_correct_local']}/{stats['low_total']} ({stats['low_correct_local']/stats['low_total']*100:.0f}%) [would-have-been-wrong rate]")

    # Calibration check
    print(f"\n  Calibration:")
    if stats["high_total"]:
        high_acc = stats["high_correct"] / stats["high_total"] * 100
        calibration = "GOOD" if high_acc >= 90 else "NEEDS TUNING"
        print(f"    High-confidence should be ~95%+ correct: {high_acc:.0f}% [{calibration}]")
    if stats["low_total"]:
        low_local_acc = stats["low_correct_local"] / stats["low_total"] * 100
        calibration = "GOOD" if low_local_acc <= 50 else "OVER-ESCALATING"
        print(f"    Low-confidence should be ~50% wrong locally: {low_local_acc:.0f}% [{calibration}]")

    # Save results
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"confidence_{benchmark}_{n}.json"
    out_path.write_text(json.dumps({
        "benchmark": benchmark,
        "n": n,
        "stats": stats,
        "results": results,
    }, indent=2, default=str))
    print(f"\n  Results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Confidence routing evaluation")
    parser.add_argument("--benchmark", "-b", default="gsm8k", choices=["gsm8k", "mmlu"])
    parser.add_argument("--limit", "-n", type=int, default=20)
    parser.add_argument("--skip-frontier", action="store_true", help="Don't call frontier (measure confidence only)")
    args = parser.parse_args()

    run_eval(args.benchmark, args.limit, args.skip_frontier)


if __name__ == "__main__":
    main()
