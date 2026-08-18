#!/usr/bin/env python3
"""Confidence estimation for local model outputs.

Combines two training-free signals to decide whether a local answer
should be returned or escalated to the frontier model:

1. Self-consistency: sample N times with temperature > 0, measure agreement.
   High agreement = the model "knows" the answer. Low agreement = uncertain.

2. Self-verification (AutoMix-style): ask the model to critique its own answer.
   Models are better at judging answers than generating them.

References:
  - Self-Consistency (Wang et al., 2022; EMNLP 2024)
  - AutoMix (Aggarwal & Madaan et al., NeurIPS 2024)
"""

import random
import re
from collections import Counter
from enum import Enum


class Confidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


SELF_VERIFY_PROMPT = """You just answered a question. Now verify your answer.

Question/Task: {question}

Your answer: {answer}

Is your answer correct and complete? Rate your confidence from 1-5:
  5 = Definitely correct, I'm very sure
  4 = Likely correct, minor uncertainty
  3 = Unsure, could go either way
  2 = Probably wrong or incomplete
  1 = Almost certainly wrong

Reply with ONLY a number 1-5, nothing else."""

HIGH_THRESHOLD = 0.85
LOW_THRESHOLD = 0.50
VERIFY_PASS_SCORE = 4
N_SAMPLES = 3
SAMPLE_TEMPERATURE = 0.7


def normalize_answer(text: str) -> str:
    """Normalize an answer for comparison: strip whitespace, lowercase, collapse spacing."""
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


def extract_core_answer(text: str) -> str:
    """Extract the core answer from a response, ignoring preamble/explanation.

    For structured answers (code, numbers, categories), extract the payload.
    For free-form text, normalize and use first 200 chars as fingerprint.
    """
    text = text.strip()
    if not text:
        return ""

    code_match = re.search(r'```(?:\w+)?\n(.+?)```', text, re.DOTALL)
    if code_match:
        return normalize_answer(code_match.group(1))

    lines = text.strip().split('\n')
    if len(lines) == 1 and len(lines[0]) < 100:
        return normalize_answer(lines[0])

    return normalize_answer(text[:200])


def is_multiple_choice(prompt: str) -> bool:
    """Detect if a prompt is a multiple-choice question."""
    return bool(re.search(r'\b[A-D]\)', prompt) or re.search(r'\b[A-D]\.\s', prompt))


def shuffle_choices(prompt: str, seed: int) -> tuple:
    """Shuffle multiple-choice options and return (new_prompt, mapping).

    The mapping maps original letters to new positions so we can
    translate the model's answer back to the original ordering.
    """
    pattern = r'([A-D])\)\s*(.+?)(?=\n[A-D]\)|\n\n|$)'
    matches = list(re.finditer(pattern, prompt, re.DOTALL))
    if len(matches) < 3:
        pattern = r'([A-D])\.\s*(.+?)(?=\n[A-D]\.|\n\n|$)'
        matches = list(re.finditer(pattern, prompt, re.DOTALL))

    if len(matches) < 3:
        return prompt, {}

    seen = set()
    deduped = []
    for m in matches:
        label = m.group(1)
        if label not in seen:
            seen.add(label)
            deduped.append(m)
    matches = deduped[:4]

    original_labels = [m.group(1) for m in matches]
    original_texts = [m.group(2).strip() for m in matches]

    rng = random.Random(seed)
    indices = list(range(len(original_texts)))
    rng.shuffle(indices)

    new_labels = ["A", "B", "C", "D"][:len(original_texts)]
    new_to_original = {}
    for new_idx, orig_idx in enumerate(indices):
        new_to_original[new_labels[new_idx]] = original_labels[orig_idx]

    new_prompt = prompt
    for i, match in enumerate(reversed(matches)):
        new_idx = indices.index(len(matches) - 1 - i)
        pass

    new_choices = ""
    sep = ") " if ")" in prompt else ". "
    for new_idx, orig_idx in enumerate(indices):
        new_choices += f"{new_labels[new_idx]}{sep}{original_texts[orig_idx]}\n"

    first_match = matches[0]
    last_match = matches[-1]
    start = first_match.start()
    end = last_match.end()
    new_prompt = prompt[:start] + new_choices.rstrip() + prompt[end:]

    return new_prompt, new_to_original


def choice_shuffle_consistency(prompt: str, generate_fn,
                               n: int = N_SAMPLES) -> dict:
    """For multiple-choice: shuffle answer order and check if model answer stays consistent.

    If the model picks the same *content* regardless of ordering, it knows the answer.
    If it follows position bias, confidence is low.
    """
    answers_original = []
    raw_results = []

    for i in range(n):
        if i == 0:
            shuffled, mapping = prompt, {}
        else:
            shuffled, mapping = shuffle_choices(prompt, seed=i * 42)

        result = generate_fn(shuffled, 0.0)
        raw = result.get("result", "").strip()
        raw_results.append(raw)

        letter_match = re.search(r'\b([A-Da-d])\b', raw)
        if letter_match:
            chosen = letter_match.group(1).upper()
            original_letter = mapping.get(chosen, chosen)
            answers_original.append(original_letter)
        else:
            answers_original.append(raw[:20])

    if not answers_original:
        return {"agreement": 0.0, "answers": raw_results, "majority_answer": ""}

    counts = Counter(answers_original)
    most_common, most_common_count = counts.most_common(1)[0]
    agreement = most_common_count / len(answers_original)

    return {
        "agreement": agreement,
        "answers": raw_results,
        "majority_answer": raw_results[0] if answers_original[0] == most_common else "",
        "n_unique": len(counts),
        "method": "choice_shuffle",
    }


def self_consistency(prompt: str, generate_fn, n: int = N_SAMPLES,
                     temperature: float = SAMPLE_TEMPERATURE) -> dict:
    """Sample the model N times and measure answer agreement.

    Args:
        prompt: The original prompt.
        generate_fn: Callable(prompt, temperature) -> dict with 'result' key.
        n: Number of samples.
        temperature: Sampling temperature (>0 for diversity).

    Returns:
        dict with 'agreement' (0-1), 'answers' (list), 'majority_answer' (str).
    """
    answers = []
    raw_results = []

    for _ in range(n):
        result = generate_fn(prompt, temperature)
        raw = result.get("result", "")
        raw_results.append(raw)
        answers.append(extract_core_answer(raw))

    if not answers or all(a == "" for a in answers):
        return {"agreement": 0.0, "answers": raw_results, "majority_answer": ""}

    counts = Counter(answers)
    most_common_answer, most_common_count = counts.most_common(1)[0]
    agreement = most_common_count / len(answers)

    majority_idx = answers.index(most_common_answer)
    majority_raw = raw_results[majority_idx]

    return {
        "agreement": agreement,
        "answers": raw_results,
        "majority_answer": majority_raw,
        "n_unique": len(counts),
    }


def self_verify(question: str, answer: str, generate_fn) -> dict:
    """Ask the model to verify its own answer (AutoMix-style).

    Args:
        question: The original prompt/question.
        answer: The model's answer to verify.
        generate_fn: Callable(prompt, temperature) -> dict with 'result' key.

    Returns:
        dict with 'score' (1-5), 'raw' (str).
    """
    verify_prompt = SELF_VERIFY_PROMPT.format(question=question[:500], answer=answer[:1000])
    result = generate_fn(verify_prompt, 0.0)
    raw = result.get("result", "").strip()

    match = re.search(r'[1-5]', raw)
    score = int(match.group()) if match else 3

    return {"score": score, "raw": raw}


def estimate_confidence(prompt: str, initial_result: dict,
                        generate_fn) -> dict:
    """Estimate confidence in a local model answer using layered signals.

    Layer 1: If the initial result is empty or an error, confidence is LOW.
    Layer 2: Self-consistency (N samples). High agreement → HIGH, low → LOW.
    Layer 3: For borderline cases, self-verification breaks the tie.

    Args:
        prompt: The original prompt.
        initial_result: The first generation result dict.
        generate_fn: Callable(prompt, temperature) -> dict with 'result' key.

    Returns:
        dict with 'confidence' (Confidence enum), 'agreement', 'verify_score',
        'best_answer' (str), 'signals' (dict of raw signal data).
    """
    answer = initial_result.get("result", "")
    if not answer or initial_result.get("error"):
        return {
            "confidence": Confidence.LOW,
            "agreement": 0.0,
            "verify_score": None,
            "best_answer": answer,
            "signals": {"reason": "empty_or_error"},
        }

    mc = is_multiple_choice(prompt)
    if mc:
        sc = choice_shuffle_consistency(prompt, generate_fn, n=5)
    else:
        sc = self_consistency(prompt, generate_fn, n=N_SAMPLES, temperature=SAMPLE_TEMPERATURE)
    agreement = sc["agreement"]

    if mc:
        # MC: use only shuffle agreement (self-verification is unreliable for
        # knowledge questions — models self-verify wrong answers as correct).
        if agreement >= 0.95:
            confidence = Confidence.HIGH
        elif agreement < 0.55:
            confidence = Confidence.LOW
        else:
            confidence = Confidence.MEDIUM
        return {
            "confidence": confidence,
            "agreement": agreement,
            "verify_score": None,
            "best_answer": sc["majority_answer"] or answer,
            "signals": {"self_consistency": sc},
        }

    # Non-MC: use self-consistency + self-verification for borderline cases
    if agreement >= HIGH_THRESHOLD:
        return {
            "confidence": Confidence.HIGH,
            "agreement": agreement,
            "verify_score": None,
            "best_answer": sc["majority_answer"] or answer,
            "signals": {"self_consistency": sc},
        }

    if agreement < LOW_THRESHOLD:
        return {
            "confidence": Confidence.LOW,
            "agreement": agreement,
            "verify_score": None,
            "best_answer": sc["majority_answer"] or answer,
            "signals": {"self_consistency": sc},
        }

    sv = self_verify(prompt, sc["majority_answer"] or answer, generate_fn)

    if sv["score"] >= VERIFY_PASS_SCORE:
        confidence = Confidence.HIGH
    elif sv["score"] <= 2:
        confidence = Confidence.LOW
    else:
        confidence = Confidence.MEDIUM

    return {
        "confidence": confidence,
        "agreement": agreement,
        "verify_score": sv["score"],
        "best_answer": sc["majority_answer"] or answer,
        "signals": {"self_consistency": sc, "self_verification": sv},
    }
