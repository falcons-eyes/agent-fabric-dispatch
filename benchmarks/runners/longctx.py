"""Long Context benchmarks: LongBench v2.

LongBench v2: Multi-task long-context evaluation (8K-128K+ tokens), MCQ format.

Note: Original THUDM/LongBench uses deprecated loading scripts.
LongBench-v2 (503 examples, MCQ) is used instead.
"""

import re
from datasets import load_dataset
from tqdm import tqdm
from benchmarks.runner import BenchmarkResult


# ──────────────────────────────────────────────
# LongBench v2 — Long Context Understanding
# ──────────────────────────────────────────────
# Input:  Long context + question + 4 choices (A/B/C/D)
# Output: Single letter (A, B, C, or D)
# Metric: Accuracy (exact match on selected answer)
# Eval:   Fully local, requires model with long context window
#
# Domains: Single-doc QA, Multi-doc QA, Summarization,
#          Few-shot, Code, In-context Learning
# Difficulty: easy / hard
# Length: short / medium / long
# ──────────────────────────────────────────────

LONGBENCH_PROMPT = """Read the following context carefully and answer the question by selecting the correct option.

{context}

Question: {question}
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Answer with ONLY the letter (A, B, C, or D):"""


def run_longbench(model: str, generate_fn, limit: int = 0) -> BenchmarkResult:
    """Run LongBench v2 benchmark.

    Dataset: THUDM/LongBench-v2
    Format: Long context + 4-choice MCQ
    Size: 503 examples across multiple domains
    """
    try:
        ds = load_dataset("THUDM/LongBench-v2", split="train")
    except Exception as e:
        return BenchmarkResult(
            benchmark="longbench",
            model=model,
            total=0, correct=0, score=0,
            metric_name="accuracy",
            details={"error": f"Could not load LongBench-v2: {str(e)[:200]}"},
            timestamp="", duration_s=0,
        )

    if limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    total = 0
    per_domain = {}

    for item in tqdm(ds, desc="LongBench-v2", unit="q"):
        context = item.get("context", "")
        question = item.get("question", "")
        expected = item.get("answer", "")

        if not question or not expected:
            continue

        # Truncate context if extremely long (model context window safety)
        if len(context) > 100000:
            context = context[:50000] + "\n...[truncated]...\n" + context[-50000:]

        prompt = LONGBENCH_PROMPT.format(
            context=context,
            question=question,
            choice_a=item.get("choice_A", ""),
            choice_b=item.get("choice_B", ""),
            choice_c=item.get("choice_C", ""),
            choice_d=item.get("choice_D", ""),
        )

        response = generate_fn(prompt)
        predicted = _extract_letter(response)
        is_correct = predicted == expected.upper()

        if is_correct:
            correct += 1
        total += 1

        domain = item.get("domain", "unknown")
        if domain not in per_domain:
            per_domain[domain] = {"correct": 0, "total": 0}
        per_domain[domain]["total"] += 1
        if is_correct:
            per_domain[domain]["correct"] += 1

    return BenchmarkResult(
        benchmark="longbench",
        model=model,
        total=total,
        correct=correct,
        score=correct / total if total > 0 else 0,
        metric_name="accuracy",
        details={"per_domain": per_domain},
        timestamp="",
        duration_s=0,
    )


def _extract_letter(response: str) -> str:
    """Extract a single letter answer from response."""
    response = response.strip()
    if len(response) == 1 and response.upper() in "ABCD":
        return response.upper()

    m = re.search(r"(?:answer|choice)\s*(?:is|:)\s*[(\[]?\s*([A-D])", response, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Check last line
    for line in reversed(response.strip().split("\n")):
        line = line.strip()
        if len(line) == 1 and line.upper() in "ABCD":
            return line.upper()

    m = re.search(r"\b([A-D])\b", response)
    if m:
        return m.group(1).upper()

    return ""
