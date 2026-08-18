#!/usr/bin/env python3
"""Benchmark comparison: local-only vs agent-fabric (local + frontier fallback).

For each question:
  1. Try local model (Ollama) first
  2. If wrong, retry with frontier (claude -p)
  3. Track accuracy, cost, and time for both strategies

Usage:
    python3 benchmarks/run_fabric_compare.py --benchmarks gsm8k,math,mmlu --limit 50
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARKS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.runner import BenchmarkResult

OLLAMA_API_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("FABRIC_MODEL", "qwen3-coder:30b")

RESULTS_DIR = BENCHMARKS_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def ollama_generate(prompt: str) -> dict:
    url = f"{OLLAMA_API_URL}/api/generate"
    payload = json.dumps({
        "model": DEFAULT_MODEL, "prompt": prompt,
        "stream": False, "options": {"temperature": 0.0},
    }).encode("utf-8")

    t0 = time.time()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return {
                "text": data.get("response", "").strip(),
                "eval_count": data.get("eval_count", 0),
                "duration_ms": round((time.time() - t0) * 1000),
                "cost": 0.0,
            }
    except Exception as e:
        return {"text": "", "eval_count": 0, "duration_ms": round((time.time() - t0) * 1000), "cost": 0.0, "error": str(e)}


def frontier_generate(prompt: str) -> dict:
    cmd = ["claude", "-p", prompt, "--output-format", "json", "--max-turns", "1"]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                            cwd="/tmp", stdin=subprocess.DEVNULL)
    dt = time.time() - t0

    if result.returncode != 0:
        return {"text": "", "output_tokens": 0, "duration_ms": round(dt * 1000), "cost": 0.0, "error": True}

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"text": result.stdout.strip(), "output_tokens": 0, "duration_ms": round(dt * 1000), "cost": 0.0, "error": True}

    return {
        "text": data.get("result", ""),
        "output_tokens": data.get("usage", {}).get("output_tokens", 0),
        "duration_ms": round(dt * 1000),
        "cost": data.get("total_cost_usd", 0),
    }


# ─── Answer extraction (shared with benchmark runners) ───

def extract_number(text):
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m: return normalize_number(m.group(1))
    m = re.search(r"(?:answer|result)\s*(?:is|=|:)\s*(-?[\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if m: return normalize_number(m.group(1))
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if numbers: return normalize_number(numbers[-1])
    return ""

def normalize_number(s):
    s = s.replace(",", "").strip()
    try:
        val = float(s)
        return str(int(val)) if val == int(val) else str(val)
    except ValueError:
        return s

def extract_boxed(text):
    m = re.search(r"\\boxed\{", text)
    if not m: return text.strip()
    start, depth, i = m.end(), 1, m.end()
    while i < len(text) and depth > 0:
        if text[i] == "{": depth += 1
        elif text[i] == "}": depth -= 1
        i += 1
    return text[start:i - 1].strip()

def math_equiv(a, b):
    if not a or not b: return False
    a, b = a.strip().replace(" ", ""), b.strip().replace(" ", "")
    if a == b: return True
    try: return abs(float(a) - float(b)) < 1e-6
    except ValueError: pass
    def frac_to_float(s):
        m = re.match(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
        return int(m.group(1)) / int(m.group(2)) if m else None
    fa, fb = frac_to_float(a), frac_to_float(b)
    if fa is not None and fb is not None: return abs(fa - fb) < 1e-6
    return False

def extract_choice(response, num_choices=4):
    letters = "ABCDEFGHIJ"[:num_choices]
    pattern = rf"[{letters}]"
    for line in reversed(response.strip().split("\n")):
        line = line.strip()
        if len(line) == 1 and line.upper() in letters: return line.upper()
        m = re.search(rf"(?:answer|choice)\s*(?:is|:)\s*[(\[]?\s*({pattern})", line, re.IGNORECASE)
        if m: return m.group(1).upper()
    m = re.search(pattern, response, re.IGNORECASE)
    return m.group(0).upper() if m else ""

def extract_integer(text):
    m = re.search(r"(?:answer|result)\s*(?:is|=|:)\s*(\d+)", text, re.IGNORECASE)
    if m: return str(int(m.group(1)))
    numbers = re.findall(r"\b(\d+)\b", text)
    return str(int(numbers[-1])) if numbers else ""


# ─── Benchmark definitions ───

BENCHMARKS = {
    "gsm8k": {
        "load": lambda limit: _load_ds("openai/gsm8k", "main", "test", limit),
        "prompt": lambda item: f"Solve this math problem step by step. At the end, write your final answer as a single number after \"#### \".\n\nProblem: {item['question']}\n\nSolution:",
        "gold": lambda item: normalize_number(item["answer"].split("####")[-1].strip()),
        "check": lambda pred, gold: extract_number(pred) == gold,
        "extract": lambda pred: extract_number(pred),
    },
    "math": {
        "load": lambda limit: _load_ds_fallback(
            [("hendrycks/competition_math", None, "test"), ("DigitalLearningGmbH/MATH-lighteval", None, "test")], limit),
        "prompt": lambda item: f"Solve this math problem. Put your final answer in \\boxed{{}}.\n\nProblem: {item['problem']}\n\nSolution:",
        "gold": lambda item: extract_boxed(item["solution"]),
        "check": lambda pred, gold: math_equiv(extract_boxed(pred), gold),
        "extract": lambda pred: extract_boxed(pred),
    },
    "mmlu": {
        "load": lambda limit: _load_ds("cais/mmlu", "all", "test", limit),
        "prompt": lambda item: f"The following is a multiple choice question. Answer with ONLY the letter (A, B, C, or D).\n\nQuestion: {item['question']}\nA. {item['choices'][0]}\nB. {item['choices'][1]}\nC. {item['choices'][2]}\nD. {item['choices'][3]}\n\nAnswer:",
        "gold": lambda item: {0: "A", 1: "B", 2: "C", 3: "D"}[item["answer"]],
        "check": lambda pred, gold: extract_choice(pred) == gold,
        "extract": lambda pred: extract_choice(pred),
    },
    "truthfulqa": {
        "load": lambda limit: _load_truthfulqa(limit),
        "prompt": lambda item: _truthfulqa_prompt(item),
        "gold": lambda item: _truthfulqa_gold(item),
        "check": lambda pred, gold: extract_choice(pred, num_choices=len(gold.split("|"))) == gold.split("|")[0] if "|" in gold else extract_choice(pred) == gold,
        "extract": lambda pred: extract_choice(pred),
    },
    "aime": {
        "load": lambda limit: _load_ds_fallback(
            [("MathArena/aime_2025", None, "train"), ("HuggingFaceH4/aime_2024", None, "train")], limit),
        "prompt": lambda item: f"This is an AIME problem. The answer is always an integer between 000 and 999.\n\nProblem: {item.get('problem', item.get('question', ''))}\n\nSolve step by step. At the end, state your answer as a single integer:",
        "gold": lambda item: str(item.get("answer", item.get("expected_answer", ""))),
        "check": lambda pred, gold: extract_integer(pred) == gold,
        "extract": lambda pred: extract_integer(pred),
    },
}


def _load_ds(path, name, split, limit):
    from datasets import load_dataset
    ds = load_dataset(path, name, split=split)
    return ds.select(range(min(limit, len(ds)))) if limit > 0 else ds

def _load_ds_fallback(options, limit):
    from datasets import load_dataset
    for path, name, split in options:
        try:
            ds = load_dataset(path, name, split=split)
            return ds.select(range(min(limit, len(ds)))) if limit > 0 else ds
        except Exception:
            continue
    raise RuntimeError(f"Could not load any of: {options}")

def _load_truthfulqa(limit):
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    return ds.select(range(min(limit, len(ds)))) if limit > 0 else ds

def _truthfulqa_prompt(item):
    choices = item["mc1_targets"]["choices"]
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    choices_text = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices) if i < len(letters))
    return f"Answer this question by selecting the correct letter.\n\nQuestion: {item['question']}\n{choices_text}\n\nAnswer:"

def _truthfulqa_gold(item):
    labels = item["mc1_targets"]["labels"]
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    correct_idx = labels.index(1)
    return letters[correct_idx] if correct_idx < len(letters) else "A"


def run_benchmark_compare(name: str, limit: int) -> dict:
    bench = BENCHMARKS[name]
    print(f"\n{'=' * 60}")
    print(f"  {name.upper()} — limit {limit}")
    print(f"{'=' * 60}")

    ds = bench["load"](limit)
    results = []

    for i, item in enumerate(ds):
        prompt = bench["prompt"](item)
        gold = bench["gold"](item)

        # 1) Local attempt
        t0 = time.time()
        local_resp = ollama_generate(prompt)
        local_time = time.time() - t0
        local_text = local_resp["text"]
        local_correct = bench["check"](local_text, gold)
        local_answer = bench["extract"](local_text)

        # 2) Frontier attempt (only if local is wrong)
        frontier_text = ""
        frontier_correct = False
        frontier_answer = ""
        frontier_cost = 0.0
        frontier_time = 0.0
        frontier_used = False

        if not local_correct:
            t0 = time.time()
            frontier_resp = frontier_generate(prompt)
            frontier_time = time.time() - t0
            frontier_text = frontier_resp["text"]
            frontier_correct = bench["check"](frontier_text, gold)
            frontier_answer = bench["extract"](frontier_text)
            frontier_cost = frontier_resp.get("cost", 0)
            frontier_used = True

        # Final answer for agent-fabric
        fabric_correct = local_correct or frontier_correct

        status = "LOCAL ✓" if local_correct else ("FRONTIER ✓" if frontier_correct else "BOTH ✗")
        print(f"  [{i+1}/{len(ds)}] gold={gold} local={local_answer} {'✓' if local_correct else '✗'}", end="")
        if frontier_used:
            print(f" → frontier={frontier_answer} {'✓' if frontier_correct else '✗'} ${frontier_cost:.4f}", end="")
        print(f"  [{status}]")

        results.append({
            "gold": gold,
            "local_answer": local_answer,
            "local_correct": local_correct,
            "local_time": round(local_time, 2),
            "local_tokens": local_resp.get("eval_count", 0),
            "frontier_used": frontier_used,
            "frontier_answer": frontier_answer,
            "frontier_correct": frontier_correct,
            "frontier_cost": frontier_cost,
            "frontier_time": round(frontier_time, 2),
            "fabric_correct": fabric_correct,
        })

    # Compute stats
    total = len(results)
    local_only_correct = sum(1 for r in results if r["local_correct"])
    fabric_correct = sum(1 for r in results if r["fabric_correct"])
    frontier_calls = sum(1 for r in results if r["frontier_used"])
    frontier_wins = sum(1 for r in results if r["frontier_used"] and r["frontier_correct"])
    total_frontier_cost = sum(r["frontier_cost"] for r in results)
    total_local_time = sum(r["local_time"] for r in results)
    total_frontier_time = sum(r["frontier_time"] for r in results)
    total_local_tokens = sum(r["local_tokens"] for r in results)

    return {
        "benchmark": name,
        "total": total,
        "local_only": {
            "correct": local_only_correct,
            "accuracy": local_only_correct / total if total else 0,
            "time_s": round(total_local_time, 1),
            "tokens": total_local_tokens,
            "cost": 0.0,
        },
        "agent_fabric": {
            "correct": fabric_correct,
            "accuracy": fabric_correct / total if total else 0,
            "frontier_calls": frontier_calls,
            "frontier_wins": frontier_wins,
            "time_s": round(total_local_time + total_frontier_time, 1),
            "tokens_local": total_local_tokens,
            "cost": round(total_frontier_cost, 4),
        },
        "per_question": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Fabric benchmark comparison")
    parser.add_argument("--benchmarks", default="gsm8k,math,mmlu", help="Comma-separated benchmarks")
    parser.add_argument("--limit", type=int, default=50, help="Questions per benchmark")
    args = parser.parse_args()

    names = [n.strip() for n in args.benchmarks.split(",")]
    all_results = {}

    print(f"\nAgent Fabric Benchmark Comparison")
    print(f"  Local model: {DEFAULT_MODEL}")
    print(f"  Frontier: claude -p")
    print(f"  Benchmarks: {', '.join(names)}")
    print(f"  Limit: {args.limit} per benchmark")

    for name in names:
        if name not in BENCHMARKS:
            print(f"\n  Skipping unknown benchmark: {name}")
            continue
        result = run_benchmark_compare(name, args.limit)
        all_results[name] = result

    # ─── Summary ───
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  {'Benchmark':<12} {'LOCAL acc':>10} {'FABRIC acc':>11} {'Δ':>6} {'Frontier$':>10} {'F calls':>8} {'F wins':>7}")
    print(f"  {'-'*66}")

    total_local_correct = 0
    total_fabric_correct = 0
    total_questions = 0
    total_cost = 0.0
    total_frontier_calls = 0
    total_frontier_wins = 0
    total_local_time = 0.0
    total_fabric_time = 0.0

    for name, r in all_results.items():
        lo = r["local_only"]
        fa = r["agent_fabric"]
        delta = fa["accuracy"] - lo["accuracy"]
        print(f"  {name:<12} {lo['accuracy']:>9.1%} {fa['accuracy']:>10.1%} {delta:>+5.1%} ${fa['cost']:>8.4f} {fa['frontier_calls']:>7} {fa['frontier_wins']:>6}")

        total_local_correct += lo["correct"]
        total_fabric_correct += fa["correct"]
        total_questions += r["total"]
        total_cost += fa["cost"]
        total_frontier_calls += fa["frontier_calls"]
        total_frontier_wins += fa["frontier_wins"]
        total_local_time += lo["time_s"]
        total_fabric_time += fa["time_s"]

    if total_questions > 0:
        lo_acc = total_local_correct / total_questions
        fa_acc = total_fabric_correct / total_questions
        delta = fa_acc - lo_acc
        print(f"  {'-'*66}")
        print(f"  {'TOTAL':<12} {lo_acc:>9.1%} {fa_acc:>10.1%} {delta:>+5.1%} ${total_cost:>8.4f} {total_frontier_calls:>7} {total_frontier_wins:>6}")

    print(f"\n  Time:   LOCAL {total_local_time:.0f}s vs FABRIC {total_fabric_time:.0f}s")
    print(f"  Cost:   LOCAL $0.00 vs FABRIC ${total_cost:.4f}")
    print(f"  Frontier efficiency: {total_frontier_wins}/{total_frontier_calls} calls recovered answers ({total_frontier_wins/total_frontier_calls*100:.0f}%)" if total_frontier_calls > 0 else "")

    # Save
    out_path = RESULTS_DIR / f"fabric_compare_{int(time.time())}.json"
    save_data = {k: {kk: vv for kk, vv in v.items() if kk != "per_question"} for k, v in all_results.items()}
    save_data["_summary"] = {
        "total_questions": total_questions,
        "local_accuracy": lo_acc if total_questions else 0,
        "fabric_accuracy": fa_acc if total_questions else 0,
        "total_cost": total_cost,
        "frontier_calls": total_frontier_calls,
        "frontier_wins": total_frontier_wins,
    }
    out_path.write_text(json.dumps(save_data, indent=2))
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
