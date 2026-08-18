"""Coding benchmarks: HumanEval, MBPP.

HumanEval: Python function completion → execute against unit tests.
MBPP: Python program generation → execute against test assertions.

Both use pass@k metric (functional correctness via execution).
IMPORTANT: Code execution in a sandboxed environment is recommended.
"""

import re
import sys
import tempfile
import subprocess
from datasets import load_dataset
from tqdm import tqdm
from benchmarks.runner import BenchmarkResult


# ──────────────────────────────────────────────
# HumanEval — OpenAI Human Evaluation
# ──────────────────────────────────────────────
# Input:  Python function signature + docstring
# Output: Function body (code completion)
# Metric: pass@1 (% of problems where generated code passes ALL unit tests)
# Eval:   Requires code execution (sandbox recommended)
#
# Example input:
#   from typing import List
#   def has_close_elements(numbers: List[float], threshold: float) -> bool:
#       """Check if in given list of numbers, are any two numbers closer
#       to each other than given threshold.
#       >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
#       False
#       """
#
# Example output:
#       for i, n1 in enumerate(numbers):
#           for j, n2 in enumerate(numbers):
#               if i != j and abs(n1 - n2) < threshold:
#                   return True
#       return False
# ──────────────────────────────────────────────

HUMANEVAL_PROMPT = """Complete the following Python function. Return ONLY the function body (the code that goes inside the function), nothing else.

{prompt}"""


def run_humaneval(model: str, generate_fn, limit: int = 0) -> BenchmarkResult:
    """Run HumanEval benchmark.

    Dataset: openai/openai_humaneval
    Format: Function signature + docstring → function body
    Size: 164 problems
    """
    ds = load_dataset("openai/openai_humaneval", split="test")
    if limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    total = 0
    errors = []

    for item in tqdm(ds, desc="HumanEval", unit="q"):
        prompt = HUMANEVAL_PROMPT.format(prompt=item["prompt"])
        response = generate_fn(prompt)

        # Combine function signature with generated body
        full_code = item["prompt"] + _clean_code_response(response)

        # Add test cases
        test_code = full_code + "\n\n" + item["test"] + f"\n\ncheck({item['entry_point']})"

        passed = _execute_code_safely(test_code, timeout=10)
        if passed:
            correct += 1
        else:
            errors.append(item["task_id"])
        total += 1

    return BenchmarkResult(
        benchmark="humaneval",
        model=model,
        total=total,
        correct=correct,
        score=correct / total if total > 0 else 0,
        metric_name="pass@1",
        details={"failed_tasks": errors[:20]},
        timestamp="",
        duration_s=0,
    )


# ──────────────────────────────────────────────
# MBPP — Mostly Basic Python Programming
# ──────────────────────────────────────────────
# Input:  Task description + 3 test assertions
# Output: Complete Python function/program
# Metric: pass@1 (functional correctness)
#
# Example input:
#   Task: Write a function to find the similar elements from the given two tuple lists.
#   Tests:
#     assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)
#
# Example output:
#   def similar_elements(test_tup1, test_tup2):
#       return tuple(set(test_tup1) & set(test_tup2))
# ──────────────────────────────────────────────

MBPP_PROMPT = """Write a Python function to solve this task. Return ONLY the code, no explanations.

Task: {text}

Test cases for reference:
{tests}

Code:"""


def run_mbpp(model: str, generate_fn, limit: int = 0) -> BenchmarkResult:
    """Run MBPP benchmark.

    Dataset: google-research-datasets/mbpp (sanitized subset)
    Format: Task description → complete Python function
    Size: 427 sanitized test problems
    """
    ds = load_dataset("google-research-datasets/mbpp", split="test")
    if limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    total = 0

    for item in tqdm(ds, desc="MBPP", unit="q"):
        tests_str = "\n".join(item["test_list"])
        prompt = MBPP_PROMPT.format(
            text=item["text"],
            tests=tests_str,
        )

        response = generate_fn(prompt)
        code = _clean_code_response(response)

        # Build test program
        setup = item.get("test_setup_code", "")
        test_code = f"{setup}\n{code}\n\n{tests_str}"

        passed = _execute_code_safely(test_code, timeout=10)
        if passed:
            correct += 1
        total += 1

    return BenchmarkResult(
        benchmark="mbpp",
        model=model,
        total=total,
        correct=correct,
        score=correct / total if total > 0 else 0,
        metric_name="pass@1",
        details={},
        timestamp="",
        duration_s=0,
    )


# ──── Utility functions ────

def _clean_code_response(response: str) -> str:
    """Extract code from model response, stripping markdown fences."""
    # Remove markdown code fences
    response = re.sub(r"```python\s*\n?", "", response)
    response = re.sub(r"```\s*\n?", "", response)
    return response.strip()


def _execute_code_safely(code: str, timeout: int = 10) -> bool:
    """Execute Python code in a subprocess and check if it passes.

    Returns True if code executes without errors (all assertions pass).
    Uses subprocess for basic isolation.
    """
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as f:
            f.write(code)
            f.flush()
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False
