"""General Knowledge benchmarks: MMLU, MMLU-Pro.

MMLU: 57-subject multiple-choice (A/B/C/D), 5-shot prompting.
MMLU-Pro: 10-choice (A-J), harder, Chain-of-Thought recommended.
"""

import re
from datasets import load_dataset
from tqdm import tqdm
from benchmarks.runner import BenchmarkResult


# ──────────────────────────────────────────────
# MMLU — Massive Multitask Language Understanding
# ──────────────────────────────────────────────
# Input:  Question text + 4 choices (A/B/C/D)
# Output: Single letter (A, B, C, or D)
# Metric: Exact match accuracy (% correct)
# Eval:   Fully local, no special infra needed
# ──────────────────────────────────────────────

MMLU_PROMPT = """The following is a multiple choice question. Answer with ONLY the letter (A, B, C, or D).

Question: {question}
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Answer:"""


def run_mmlu(model: str, generate_fn, limit: int = 0) -> BenchmarkResult:
    """Run MMLU benchmark.

    Dataset: cais/mmlu (all subjects)
    Format: 4-choice MCQ across 57 subjects (STEM, humanities, social sciences, etc.)
    Size: ~14,042 test questions
    """
    ds = load_dataset("cais/mmlu", "all", split="test")
    if limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    correct = 0
    total = 0
    per_subject = {}

    for item in tqdm(ds, desc="MMLU", unit="q"):
        choices = item["choices"]
        prompt = MMLU_PROMPT.format(
            question=item["question"],
            choice_a=choices[0],
            choice_b=choices[1],
            choice_c=choices[2],
            choice_d=choices[3],
        )

        response = generate_fn(prompt)
        predicted = _extract_choice(response, num_choices=4)
        expected = answer_map[item["answer"]]

        is_correct = predicted == expected
        if is_correct:
            correct += 1
        total += 1

        subject = item.get("subject", "unknown")
        if subject not in per_subject:
            per_subject[subject] = {"correct": 0, "total": 0}
        per_subject[subject]["total"] += 1
        if is_correct:
            per_subject[subject]["correct"] += 1

    return BenchmarkResult(
        benchmark="mmlu",
        model=model,
        total=total,
        correct=correct,
        score=correct / total if total > 0 else 0,
        metric_name="accuracy",
        details={"per_subject": per_subject},
        timestamp="",
        duration_s=0,
    )


# ──────────────────────────────────────────────
# MMLU-Pro — Harder MMLU with 10 choices
# ──────────────────────────────────────────────
# Input:  Question text + 10 choices (A-J)
# Output: Single letter (A through J)
# Metric: Accuracy with 5-shot CoT recommended
# Eval:   Fully local
# ──────────────────────────────────────────────

MMLU_PRO_PROMPT = """The following is a hard multiple choice question. Think step by step, then give your answer as a single letter (A through J).

Question: {question}
{choices_text}

Think step by step, then answer with ONLY the letter:"""


def run_mmlu_pro(model: str, generate_fn, limit: int = 0) -> BenchmarkResult:
    """Run MMLU-Pro benchmark.

    Dataset: TIGER-Lab/MMLU-Pro
    Format: 10-choice MCQ, reasoning-focused
    Size: ~12,032 questions
    """
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    if limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    letters = "ABCDEFGHIJ"
    correct = 0
    total = 0

    for item in tqdm(ds, desc="MMLU-Pro", unit="q"):
        options = item["options"]
        choices_text = "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options))

        prompt = MMLU_PRO_PROMPT.format(
            question=item["question"],
            choices_text=choices_text,
        )

        response = generate_fn(prompt)
        predicted = _extract_choice(response, num_choices=len(options))
        expected = item["answer"]

        if predicted == expected:
            correct += 1
        total += 1

    return BenchmarkResult(
        benchmark="mmlu_pro",
        model=model,
        total=total,
        correct=correct,
        score=correct / total if total > 0 else 0,
        metric_name="accuracy",
        details={},
        timestamp="",
        duration_s=0,
    )


def _extract_choice(response: str, num_choices: int = 4) -> str:
    """Extract answer letter from model response."""
    response = response.strip()
    letters = "ABCDEFGHIJ"[:num_choices]
    pattern = rf"[({letters})]"

    # Try to find a standalone letter
    # Check last line first (model often puts answer at end)
    lines = response.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if len(line) == 1 and line.upper() in letters:
            return line.upper()
        # Match "Answer: X" or "The answer is X"
        m = re.search(rf"(?:answer|choice)\s*(?:is|:)\s*[(\[]?\s*({pattern})", line, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # Fallback: first occurrence of a valid letter
    m = re.search(pattern, response, re.IGNORECASE)
    if m:
        return m.group(0).upper()

    return ""
