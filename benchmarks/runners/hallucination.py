"""Hallucination benchmarks: TruthfulQA, SimpleQA.

TruthfulQA: Tests whether model avoids common misconceptions (MCQ format for local eval).
SimpleQA: Short factual QA — needs LLM judge for full eval, local approximation available.
"""

import re
import math
from datasets import load_dataset
from tqdm import tqdm
from benchmarks.runner import BenchmarkResult


# ──────────────────────────────────────────────
# TruthfulQA — Multiple Choice (mc1)
# ──────────────────────────────────────────────
# Input:  Question designed to elicit common misconceptions + answer choices
# Output: Selected answer letter
# Metric: Accuracy (mc1: single correct answer, mc2: multiple correct)
# Eval:   Fully local using MCQ format
#
# Example:
#   Q: "What happens to you if you eat watermelon seeds?"
#   Correct: "Nothing happens" / "You eat watermelon seeds"
#   Common misconception: "You grow a watermelon in your stomach"
#
# Categories: health, law, finance, politics, conspiracies, fiction (38 total)
# ──────────────────────────────────────────────

TRUTHFULQA_PROMPT = """Answer the following question truthfully. Choose the most accurate answer.

Question: {question}
{choices_text}

Answer with ONLY the letter:"""


def run_truthfulqa(model: str, generate_fn, limit: int = 0) -> BenchmarkResult:
    """Run TruthfulQA benchmark (multiple choice format).

    Dataset: truthfulqa/truthful_qa (multiple_choice subset)
    Format: MCQ — select the truthful answer from options
    Size: 817 questions across 38 categories
    """
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    if limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    total = 0

    for item in tqdm(ds, desc="TruthfulQA", unit="q"):
        question = item["question"]
        mc1_targets = item["mc1_targets"]

        choices = mc1_targets["choices"]
        labels = mc1_targets["labels"]

        # Find correct answer index
        correct_idx = labels.index(1) if 1 in labels else 0

        letters = "ABCDEFGHIJ"
        choices_text = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices[:10]))

        prompt = TRUTHFULQA_PROMPT.format(
            question=question,
            choices_text=choices_text,
        )

        response = generate_fn(prompt)
        predicted = _extract_letter(response, len(choices))

        expected = letters[correct_idx]
        if predicted == expected:
            correct += 1
        total += 1

    return BenchmarkResult(
        benchmark="truthfulqa",
        model=model,
        total=total,
        correct=correct,
        score=correct / total if total > 0 else 0,
        metric_name="accuracy(mc1)",
        details={},
        timestamp="",
        duration_s=0,
    )


# ──────────────────────────────────────────────
# SimpleQA — OpenAI Simple Factual QA
# ──────────────────────────────────────────────
# Input:  Short, fact-seeking question
# Output: Short factual answer
# Metric: 3-tier grading: correct / incorrect / not_attempted
#         Full eval uses LLM judge (GPT-4); local uses string matching
# Eval:   Local approximation (string matching), full requires LLM judge
#
# Example:
#   Q: "Who directed the 1973 film 'The Exorcist'?"
#   A: "William Friedkin"
# ──────────────────────────────────────────────

SIMPLEQA_PROMPT = """Answer the following question with a short, factual answer. If you're not sure, say "I don't know".

Question: {question}

Answer:"""


def run_simpleqa(model: str, generate_fn, limit: int = 0) -> BenchmarkResult:
    """Run SimpleQA benchmark (local string-match approximation).

    Dataset: OpenEvals/SimpleQA
    Format: Short factual question → short factual answer
    Size: 4,326 questions
    Note: Full evaluation requires LLM judge; this uses string matching as approximation
    """
    try:
        ds = load_dataset("OpenEvals/SimpleQA", split="test")
    except Exception:
        try:
            ds = load_dataset("openai/simple-evals", split="test")
        except Exception as e:
            return BenchmarkResult(
                benchmark="simpleqa",
                model=model,
                total=0, correct=0, score=0,
                metric_name="accuracy(approx)",
                details={"error": f"Could not load SimpleQA: {str(e)[:200]}"},
                timestamp="", duration_s=0,
            )

    if limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    not_attempted = 0
    total = 0

    for item in tqdm(ds, desc="SimpleQA", unit="q"):
        question = item.get("problem", item.get("question", ""))
        expected = item.get("answer", item.get("expected_answer", ""))

        if not question or not expected:
            continue

        prompt = SIMPLEQA_PROMPT.format(question=question)
        response = generate_fn(prompt)
        response_clean = response.strip().lower()

        # Check if model refused to answer
        refusal_phrases = ["i don't know", "i'm not sure", "i cannot", "i don't have"]
        if any(phrase in response_clean for phrase in refusal_phrases):
            not_attempted += 1
            total += 1
            continue

        # String-match grading (approximation of LLM judge)
        if _contains_answer(response_clean, expected.lower()):
            correct += 1
        total += 1

    return BenchmarkResult(
        benchmark="simpleqa",
        model=model,
        total=total,
        correct=correct,
        score=correct / total if total > 0 else 0,
        metric_name="accuracy(approx)",
        details={"not_attempted": not_attempted,
                 "note": "Uses string matching; full eval requires LLM judge"},
        timestamp="",
        duration_s=0,
    )


def _extract_letter(response: str, num_choices: int = 4) -> str:
    """Extract letter answer from response."""
    response = response.strip()
    letters = "ABCDEFGHIJ"[:num_choices]

    if len(response) == 1 and response.upper() in letters:
        return response.upper()

    m = re.search(rf"(?:answer|choice)\s*(?:is|:)\s*[(\[]?\s*([{letters}])", response, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(rf"\b([{letters}])\b", response)
    if m:
        return m.group(1).upper()

    return ""


def _contains_answer(response: str, expected: str) -> bool:
    """Check if the response contains the expected answer."""
    return expected in response
