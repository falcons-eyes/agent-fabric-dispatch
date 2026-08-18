#!/usr/bin/env python3
"""Quality verification: frontier model scores local model outputs.

For each task type (summarize, refactor, classify), runs the local model,
then asks the frontier model to score the output on a 1-5 scale.

Gate: >= 85% pass rate (score >= 4) on at least 2 task types.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.fabric_agent import ollama_generate

FIXTURES = [
    {
        "name": "sort_utils",
        "code": """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr""",
        "refactor_instruction": "Add type hints, add early return for empty/single-element arrays",
        "expected_category": "algorithm",
        "categories": "algorithm,utility,data-structure,io,network,math",
    },
    {
        "name": "string_helpers",
        "code": """def reverseString(s):
    return s[::-1]

def isPalindrome(s):
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]

def countWords(text):
    return len(text.split())

def capitalizeFirst(text):
    return ' '.join(word.capitalize() for word in text.split())""",
        "refactor_instruction": "Convert camelCase to snake_case, add type hints",
        "expected_category": "string-processing",
        "categories": "string-processing,algorithm,data-structure,io,network,math",
    },
    {
        "name": "file_processor",
        "code": """import os
import json

def read_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def write_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def list_files(directory, extension=None):
    files = []
    for f in os.listdir(directory):
        if extension is None or f.endswith(extension):
            files.append(os.path.join(directory, f))
    return files""",
        "refactor_instruction": "Use pathlib instead of os.path, add type hints",
        "expected_category": "io",
        "categories": "io,algorithm,utility,data-structure,network,math",
    },
    {
        "name": "math_utils",
        "code": """def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n):
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True""",
        "refactor_instruction": "Add type hints and add docstrings to each function",
        "expected_category": "math",
        "alt_categories": ["algorithm"],
        "categories": "math,algorithm,utility,data-structure,io,network",
    },
    {
        "name": "http_client",
        "code": """import urllib.request
import json

def get_json(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode('utf-8'))

def post_json(url, data):
    payload = json.dumps(data).encode('utf-8')
    req = urllib.request.Request(url, data=payload,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode('utf-8'))

def download_file(url, path):
    urllib.request.urlretrieve(url, path)""",
        "refactor_instruction": "Add error handling with try/except, add type hints, add timeout parameter",
        "expected_category": "network",
        "categories": "network,io,algorithm,utility,data-structure,math",
    },
]

TASK_TYPES = {
    "summarize": {
        "prompt": lambda f: (
            f"/no_think Summarize this Python code concisely. List each function with "
            f"a one-line description. Keep the total summary under 150 words.\n\n"
            f"```python\n{f['code']}\n```"
        ),
    },
    "refactor": {
        "prompt": lambda f: (
            f"/no_think Refactor this Python code: {f['refactor_instruction']}. "
            f"Return only the refactored code, no explanation.\n\n```python\n{f['code']}\n```"
        ),
    },
    "classify": {
        "prompt": lambda f: (
            f"/no_think Classify this Python code into exactly one of these categories: "
            f"{f['categories']}\n\n"
            f"Rules:\n"
            f"- Reply with ONLY the category name, nothing else\n"
            f"- The category must be one from the list above\n"
            f"- Do not add explanation or punctuation\n\n"
            f"```python\n{f['code']}\n```"
        ),
    },
}


def classify_score(local_output: str, expected: str, categories: str, alt_categories: list = None) -> dict:
    """Score classify output by deterministic extraction — no frontier call needed."""
    valid_cats = [c.strip() for c in categories.split(",")]
    output_lower = local_output.strip().lower()
    acceptable = {expected.lower()} | {a.lower() for a in (alt_categories or [])}

    extracted = None
    for cat in valid_cats:
        if cat.lower() in output_lower:
            extracted = cat
            break

    if extracted and extracted.lower() in acceptable:
        return {"score": 5, "reason": f"Correct: '{extracted}'", "cost": 0}
    elif extracted:
        return {"score": 2, "reason": f"Wrong category: '{extracted}' (expected '{expected}')", "cost": 0}
    else:
        return {"score": 1, "reason": f"No valid category found in output (expected '{expected}')", "cost": 0}


def frontier_score(task_type: str, original_code: str, instruction: str, local_output: str,
                   expected_category: str = None, categories: str = None,
                   alt_categories: list = None) -> dict:
    if task_type == "classify":
        return classify_score(local_output, expected_category, categories, alt_categories)

    if task_type == "summarize":
        scoring_prompt = (
            f"Rate this code summary on a scale of 1-5.\n\n"
            f"Original code:\n```python\n{original_code}\n```\n\n"
            f"Summary produced:\n{local_output}\n\n"
            f"Criteria: accuracy (mentions key functions), completeness (covers purpose), "
            f"conciseness (not overly verbose).\n\n"
            f"Reply with ONLY a JSON object: {{\"score\": <1-5>, \"reason\": \"<brief>\"}}"
        )
    elif task_type == "refactor":
        scoring_prompt = (
            f"Rate this code refactoring on a scale of 1-5.\n\n"
            f"Original code:\n```python\n{original_code}\n```\n\n"
            f"Instruction: {instruction}\n\n"
            f"Refactored code:\n{local_output}\n\n"
            f"Criteria: follows the instruction, preserves functionality, "
            f"code quality (no bugs introduced).\n\n"
            f"Reply with ONLY a JSON object: {{\"score\": <1-5>, \"reason\": \"<brief>\"}}"
        )
    else:
        return {"score": 0, "reason": "unknown task type"}

    import re
    cmd = ["claude", "-p", scoring_prompt, "--output-format", "json", "--max-turns", "1"]
    for attempt in range(2):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=90,
                                    cwd="/tmp", stdin=subprocess.DEVNULL)
            if result.returncode != 0:
                if attempt == 0:
                    time.sleep(3)
                    continue
                return {"score": 0, "reason": f"frontier error: {result.stderr[:200]}", "cost": 0}

            data = json.loads(result.stdout)
            text = data.get("result", "")
            cost = data.get("total_cost_usd", 0)

            m = re.search(r'"score"\s*:\s*(\d)', text)
            score = int(m.group(1)) if m else 0

            m2 = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
            reason = m2.group(1) if m2 else text[:200]

            return {"score": score, "reason": reason, "cost": cost}
        except Exception as e:
            if attempt == 0:
                time.sleep(3)
                continue
            return {"score": 0, "reason": str(e), "cost": 0}
    return {"score": 0, "reason": "all retries failed", "cost": 0}


def main():
    n = len(FIXTURES)
    print(f"Quality Scoring Test")
    print(f"  {n} fixtures x 3 task types = {n * 3} evaluations")
    print(f"  Local: Ollama qwen3-coder:30b")
    print(f"  Scorer: claude -p (frontier)")
    print(f"  Gate: >= 85% pass rate (score >= 4) on >= 2 task types")
    print("=" * 70)

    results = {}
    total_cost = 0.0

    for task_name, task_config in TASK_TYPES.items():
        print(f"\n--- {task_name.upper()} ({n} fixtures) ---")
        scores = []

        for i, fixture in enumerate(FIXTURES):
            prompt = task_config["prompt"](fixture)

            # Run local model
            print(f"  [{i+1}/{n}] {fixture['name']}...", end=" ", flush=True)
            local_resp = ollama_generate(prompt)
            local_output = local_resp.get("result", "")

            if local_resp.get("error"):
                print(f"LOCAL ERROR: {local_resp['error'][:100]}")
                scores.append({"fixture": fixture["name"], "score": 0, "reason": "local error"})
                continue

            # Score with frontier (or deterministic for classify)
            instruction = fixture.get("refactor_instruction", fixture.get("categories", ""))
            score_result = frontier_score(
                task_name, fixture["code"], instruction, local_output,
                expected_category=fixture.get("expected_category"),
                categories=fixture.get("categories"),
                alt_categories=fixture.get("alt_categories"),
            )
            total_cost += score_result.get("cost", 0)

            s = score_result["score"]
            status = "PASS" if s >= 4 else "FAIL"
            print(f"score={s}/5 [{status}] — {score_result['reason'][:80]}")

            scores.append({
                "fixture": fixture["name"],
                "score": s,
                "reason": score_result["reason"],
                "local_output_preview": local_output[:200],
            })

        pass_count = sum(1 for s in scores if s["score"] >= 4)
        pass_rate = pass_count / len(scores) if scores else 0
        results[task_name] = {
            "scores": scores,
            "pass_count": pass_count,
            "total": len(scores),
            "pass_rate": pass_rate,
        }

    # Summary
    print(f"\n{'=' * 70}")
    print(f"QUALITY SCORING REPORT")
    print(f"{'=' * 70}")

    passing_types = 0
    for task_name in ["summarize", "refactor", "classify"]:
        r = results[task_name]
        avg_score = sum(s["score"] for s in r["scores"]) / len(r["scores"]) if r["scores"] else 0
        gate = "PASS" if r["pass_rate"] >= 0.85 else "FAIL"
        if r["pass_rate"] >= 0.85:
            passing_types += 1
        print(f"\n  {task_name.upper()}:")
        print(f"    Pass rate: {r['pass_count']}/{r['total']} ({r['pass_rate']:.0%})")
        print(f"    Avg score: {avg_score:.1f}/5")
        print(f"    Gate (>=85%): {gate}")
        for s in r["scores"]:
            icon = "+" if s["score"] >= 4 else "-"
            print(f"      [{icon}] {s['fixture']}: {s['score']}/5 — {s['reason'][:60]}")

    overall_gate = "PASS" if passing_types >= 2 else "FAIL"
    print(f"\n  OVERALL GATE (>=2 types passing): {overall_gate} ({passing_types}/3 types)")
    print(f"  Total scoring cost: ${total_cost:.4f}")

    # Save
    out_path = Path(__file__).parent / "quality_results.json"
    out_path.write_text(json.dumps({
        "results": {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
        "passing_types": passing_types,
        "total_cost": total_cost,
        "gate": overall_gate,
    }, indent=2))
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
