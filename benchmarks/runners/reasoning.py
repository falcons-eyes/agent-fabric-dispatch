"""Reasoning benchmarks: GPQA, HLE.

GPQA: Graduate-level multiple-choice (biology, physics, chemistry).
HLE: Humanity's Last Exam — expert-level MCQ + short-answer.

Note: ARC-AGI requires custom grid handling and is listed as infrastructure-heavy.
"""

import random
import re
from datasets import load_dataset
from tqdm import tqdm
from benchmarks.runner import BenchmarkResult


# ──────────────────────────────────────────────
# GPQA — Graduate-Level Google-Proof Q&A
# ──────────────────────────────────────────────
# Input:  4-choice MCQ in graduate-level science
# Output: Single letter (A/B/C/D)
# Metric: Exact match accuracy
# Eval:   Local (but dataset is GATED — requires HF access request)
#
# Expert human accuracy: ~65%, non-expert: ~34%
# ──────────────────────────────────────────────

GPQA_PROMPT = """This is a graduate-level science question. Think carefully and answer with ONLY the letter (A, B, C, or D).

Question: {question}
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Answer:"""


def run_gpqa(model: str, generate_fn, limit: int = 0) -> BenchmarkResult:
    """Run GPQA Diamond benchmark.

    Dataset: Idavidrein/gpqa (GATED — requires access approval on HuggingFace)
    Format: 4-choice MCQ in biology, physics, chemistry
    Size: 198 questions (Diamond subset)
    """
    try:
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    except Exception as e:
        return BenchmarkResult(
            benchmark="gpqa",
            model=model,
            total=0, correct=0, score=0,
            metric_name="accuracy",
            details={"error": f"GPQA is a gated dataset. Request access at huggingface.co/datasets/Idavidrein/gpqa. Error: {str(e)[:200]}"},
            timestamp="", duration_s=0,
        )

    if limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    total = 0

    rng = random.Random(42)
    letters = ["A", "B", "C", "D"]

    for item in tqdm(ds, desc="GPQA", unit="q"):
        choices = [
            item["Correct Answer"],
            item["Incorrect Answer 1"],
            item["Incorrect Answer 2"],
            item["Incorrect Answer 3"],
        ]
        indices = list(range(4))
        rng.shuffle(indices)
        correct_pos = indices.index(0)

        prompt = GPQA_PROMPT.format(
            question=item["Question"],
            choice_a=choices[indices[0]],
            choice_b=choices[indices[1]],
            choice_c=choices[indices[2]],
            choice_d=choices[indices[3]],
        )

        response = generate_fn(prompt)
        predicted = _extract_letter(response)

        if predicted == letters[correct_pos]:
            correct += 1
        total += 1

    return BenchmarkResult(
        benchmark="gpqa",
        model=model,
        total=total,
        correct=correct,
        score=correct / total if total > 0 else 0,
        metric_name="accuracy",
        details={},
        timestamp="",
        duration_s=0,
    )


# ──────────────────────────────────────────────
# HLE — Humanity's Last Exam
# ──────────────────────────────────────────────
# Input:  MCQ or short-answer question (some with images)
# Output: Selected answer (MCQ) or short text
# Metric: Accuracy (automated string matching)
# Eval:   Local for text-only questions
#
# Note: ~30% of biology/chemistry answers may have issues per FutureHouse audit
# ──────────────────────────────────────────────

HLE_MCQ_PROMPT = """This is an expert-level question. Think step by step, then give your final answer.

Question: {question}
{choices_text}

Answer with ONLY the letter:"""

HLE_SHORT_PROMPT = """This is an expert-level question. Give a short, precise answer.

Question: {question}

Answer:"""


def run_hle(model: str, generate_fn, limit: int = 0) -> BenchmarkResult:
    """Run Humanity's Last Exam benchmark.

    Dataset: cais/hle
    Format: MCQ + short-answer, multi-subject, some multimodal
    Size: ~3,000 questions
    """
    try:
        ds = load_dataset("cais/hle", split="test")
    except Exception:
        try:
            ds = load_dataset("cais/hle", split="validation")
        except Exception as e:
            return BenchmarkResult(
                benchmark="hle",
                model=model,
                total=0, correct=0, score=0,
                metric_name="accuracy",
                details={"error": f"Could not load HLE dataset: {str(e)[:200]}"},
                timestamp="", duration_s=0,
            )

    if limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    total = 0

    for item in tqdm(ds, desc="HLE", unit="q"):
        # Skip multimodal questions (need vision model)
        image = item.get("image")
        if image and str(image).strip():
            continue

        question = item.get("question", "")
        answer = item.get("answer", "")
        answer_type = item.get("answer_type", "")

        if not question or not answer:
            continue

        if answer_type == "multiple_choice" or len(answer) == 1:
            prompt = HLE_MCQ_PROMPT.format(
                question=question,
                choices_text=item.get("choices", ""),
            )
            response = generate_fn(prompt)
            predicted = _extract_letter(response)
            is_correct = predicted.upper() == answer.upper()
        else:
            prompt = HLE_SHORT_PROMPT.format(question=question)
            response = generate_fn(prompt)
            predicted = response.strip()
            is_correct = _fuzzy_match(predicted, answer)

        if is_correct:
            correct += 1
        total += 1

    return BenchmarkResult(
        benchmark="hle",
        model=model,
        total=total,
        correct=correct,
        score=correct / total if total > 0 else 0,
        metric_name="accuracy",
        details={},
        timestamp="",
        duration_s=0,
    )


def _extract_letter(response: str) -> str:
    """Extract a single letter answer from response."""
    response = response.strip()
    if len(response) == 1 and response.upper() in "ABCDEFGHIJ":
        return response.upper()

    m = re.search(r"(?:answer|choice)\s*(?:is|:)\s*[(\[]?\s*([A-J])", response, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"\b([A-D])\b", response)
    if m:
        return m.group(1).upper()

    return ""


def _fuzzy_match(predicted: str, expected: str) -> bool:
    """Fuzzy string matching for short answers."""
    p = predicted.lower().strip()
    e = expected.lower().strip()
    return p == e or e in p or p in e
