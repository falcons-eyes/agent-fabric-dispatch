"""Math benchmarks: GSM8K, MATH, AIME, Omni-MATH.

GSM8K: Grade-school word problems → integer answer.
MATH: Competition math → LaTeX-boxed answer.
AIME: AMC competition → integer 0-999.
Omni-MATH: Olympiad-level → needs LLM judge for full eval.
"""

import re
from datasets import load_dataset
from tqdm import tqdm
from benchmarks.runner import BenchmarkResult


# ──────────────────────────────────────────────
# GSM8K — Grade School Math 8K
# ──────────────────────────────────────────────
# Input:  Natural language word problem
# Output: Final numerical answer (integer)
# Metric: Exact match on extracted number
# Eval:   Fully local, simplest math benchmark
# ──────────────────────────────────────────────

GSM8K_PROMPT = """Solve this math problem step by step. At the end, write your final answer as a single number after "#### ".

Problem: {question}

Solution:"""


def run_gsm8k(model: str, generate_fn, limit: int = 0) -> BenchmarkResult:
    """Run GSM8K benchmark.

    Dataset: openai/gsm8k
    Format: Grade-school word problem → step-by-step solution ending with #### <number>
    Size: 1,319 test problems
    """
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    total = 0

    for item in tqdm(ds, desc="GSM8K", unit="q"):
        prompt = GSM8K_PROMPT.format(question=item["question"])
        response = generate_fn(prompt)

        predicted = _extract_number(response)
        # Extract gold answer after ####
        gold_answer = item["answer"].split("####")[-1].strip()
        expected = _normalize_number(gold_answer)

        if predicted == expected:
            correct += 1
        total += 1
        tqdm.write(f"  [{total}] pred={predicted} gold={expected} {'✓' if predicted == expected else '✗'}")

    return BenchmarkResult(
        benchmark="gsm8k",
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
# MATH — Competition Mathematics
# ──────────────────────────────────────────────
# Input:  Competition math problem text
# Output: Answer in LaTeX \boxed{...} format
# Metric: Exact match after LaTeX normalization
# Eval:   Local, needs LaTeX normalization
# ──────────────────────────────────────────────

MATH_PROMPT = """Solve this math problem. Put your final answer in \\boxed{{}}.

Problem: {problem}

Solution:"""


def run_math(model: str, generate_fn, limit: int = 0) -> BenchmarkResult:
    """Run MATH benchmark.

    Dataset: DigitalLearningGmbH/MATH-lighteval (mirror of hendrycks/competition_math)
    Format: Competition problem → LaTeX-boxed answer
    Size: 5,000 test problems, difficulty levels 1-5
    """
    try:
        ds = load_dataset("hendrycks/competition_math", split="test")
    except Exception:
        ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="test")
    if limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    total = 0
    per_level = {}

    for item in tqdm(ds, desc="MATH", unit="q"):
        prompt = MATH_PROMPT.format(problem=item["problem"])
        response = generate_fn(prompt)

        predicted = _extract_boxed(response)
        expected = _extract_boxed(item["solution"])

        is_correct = _math_equiv(predicted, expected)
        if is_correct:
            correct += 1
        total += 1

        level = item.get("level", "unknown")
        if level not in per_level:
            per_level[level] = {"correct": 0, "total": 0}
        per_level[level]["total"] += 1
        if is_correct:
            per_level[level]["correct"] += 1

    return BenchmarkResult(
        benchmark="math",
        model=model,
        total=total,
        correct=correct,
        score=correct / total if total > 0 else 0,
        metric_name="accuracy",
        details={"per_level": per_level},
        timestamp="",
        duration_s=0,
    )


# ──────────────────────────────────────────────
# AIME — American Invitational Math Examination
# ──────────────────────────────────────────────
# Input:  Competition math problem text
# Output: Integer between 0 and 999
# Metric: Exact integer match
# Eval:   Fully local, very small dataset (30 per year)
# ──────────────────────────────────────────────

AIME_PROMPT = """This is an AIME (American Invitational Mathematics Examination) problem.
The answer is always an integer between 000 and 999 inclusive.

Problem: {problem}

Solve step by step. At the end, state your answer as a single integer:"""


def run_aime(model: str, generate_fn, limit: int = 0) -> BenchmarkResult:
    """Run AIME benchmark.

    Dataset: MathArena/aime_2025 (or other years)
    Format: Competition problem → integer 0-999
    Size: 30 problems per year
    """
    try:
        ds = load_dataset("MathArena/aime_2025", split="train")
    except Exception:
        try:
            ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
        except Exception:
            return BenchmarkResult(
                benchmark="aime",
                model=model,
                total=0, correct=0, score=0,
                metric_name="accuracy",
                details={"error": "Could not load AIME dataset"},
                timestamp="", duration_s=0,
            )

    if limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    correct = 0
    total = 0

    for item in tqdm(ds, desc="AIME", unit="q"):
        problem = item.get("problem", item.get("question", ""))
        prompt = AIME_PROMPT.format(problem=problem)
        response = generate_fn(prompt)

        predicted = _extract_integer(response)
        expected = str(item.get("answer", item.get("expected_answer", "")))

        if predicted == expected:
            correct += 1
        total += 1

    return BenchmarkResult(
        benchmark="aime",
        model=model,
        total=total,
        correct=correct,
        score=correct / total if total > 0 else 0,
        metric_name="accuracy",
        details={},
        timestamp="",
        duration_s=0,
    )


# ──── Utility functions ────

def _extract_number(text: str) -> str:
    """Extract the final number from model output."""
    # Look for #### pattern first
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return _normalize_number(m.group(1))

    # Look for "answer is X" pattern
    m = re.search(r"(?:answer|result)\s*(?:is|=|:)\s*(-?[\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        return _normalize_number(m.group(1))

    # Take the last number in the response
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if numbers:
        return _normalize_number(numbers[-1])

    return ""


def _normalize_number(s: str) -> str:
    """Normalize a number string (remove commas, trailing .0)."""
    s = s.replace(",", "").strip()
    try:
        val = float(s)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return s


def _extract_boxed(text: str) -> str:
    """Extract content from \\boxed{...}."""
    # Handle nested braces
    m = re.search(r"\\boxed\{", text)
    if not m:
        return text.strip()

    start = m.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    return text[start:i - 1].strip()


def _math_equiv(a: str, b: str) -> bool:
    """Check if two math answers are equivalent."""
    if not a or not b:
        return False
    # Normalize both
    a = a.strip().replace(" ", "")
    b = b.strip().replace(" ", "")
    if a == b:
        return True

    # Try numeric comparison
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        pass

    # Normalize LaTeX fractions
    def frac_to_float(s):
        m = re.match(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
        if m:
            return int(m.group(1)) / int(m.group(2))
        return None

    fa, fb = frac_to_float(a), frac_to_float(b)
    if fa is not None and fb is not None:
        return abs(fa - fb) < 1e-6

    return False


def _extract_integer(text: str) -> str:
    """Extract a final integer answer (for AIME)."""
    # Look for explicit answer statement
    m = re.search(r"(?:answer|result)\s*(?:is|=|:)\s*(\d+)", text, re.IGNORECASE)
    if m:
        return m.group(1)

    # Last standalone integer
    numbers = re.findall(r"\b(\d+)\b", text)
    if numbers:
        return numbers[-1]

    return ""
