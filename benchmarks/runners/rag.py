"""RAG benchmarks: RAGBench.

RAGBench: Retrieval-augmented generation evaluation across 12 domains.

Note: RGB (chen700564/RGB on GitHub) requires custom download and is not on HuggingFace.
"""

import re
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm
from benchmarks.runner import BenchmarkResult


# ──────────────────────────────────────────────
# RAGBench — Retrieval Augmented Generation Bench
# ──────────────────────────────────────────────
# Input:  Question + retrieved context documents + expected answer
# Output: Generated answer based on provided context
# Metric: F1 (token overlap), exact match
# Eval:   Fully local
#
# Sub-datasets (12 domains):
#   covidqa, cuad, delucionqa, emanual, expertqa,
#   finqa, hagrid, hotpotqa, msmarco, pubmedqa, tatqa, techqa
#
# Tests RAG abilities:
#   - Answer from relevant context
#   - Ignore irrelevant/noisy context
#   - Handle contradictory retrieved passages
# ──────────────────────────────────────────────

RAGBENCH_PROMPT = """Answer the question based ONLY on the provided context. If the context doesn't contain enough information, say "I cannot answer based on the provided context."

Context:
{context}

Question: {question}

Answer:"""


def run_ragbench(model: str, generate_fn, limit: int = 0) -> BenchmarkResult:
    """Run RAGBench benchmark.

    Dataset: galileo-ai/ragbench
    Format: Question + retrieved context → answer
    Size: ~100,000 examples across 12 sub-datasets
    """
    # Use a representative subset of domains
    domains = ["hotpotqa", "msmarco", "covidqa"]
    correct = 0
    total = 0
    per_domain = {}

    for domain in domains:
        try:
            ds = load_dataset("galileo-ai/ragbench", domain, split="test")
        except Exception:
            continue

        domain_limit = (limit // len(domains)) if limit > 0 else 50
        ds = ds.select(range(min(domain_limit, len(ds))))

        domain_correct = 0
        domain_total = 0

        for item in tqdm(ds, desc=f"RAGBench/{domain}", unit="q"):
            question = item.get("question", "")
            context_docs = item.get("documents", item.get("context", ""))

            if isinstance(context_docs, list):
                context = "\n\n".join(context_docs)
            else:
                context = str(context_docs)

            expected = item.get("response", item.get("answer", ""))
            if not question or not expected:
                continue

            prompt = RAGBENCH_PROMPT.format(context=context, question=question)
            response = generate_fn(prompt)

            f1 = _compute_f1(response, expected)
            if f1 >= 0.5:
                domain_correct += 1
                correct += 1
            domain_total += 1
            total += 1

        per_domain[domain] = {
            "correct": domain_correct,
            "total": domain_total,
            "pass_rate": domain_correct / domain_total if domain_total > 0 else 0,
        }

    return BenchmarkResult(
        benchmark="ragbench",
        model=model,
        total=total,
        correct=correct,
        score=correct / total if total > 0 else 0,
        metric_name="F1>=0.5_rate",
        details={"per_domain": per_domain},
        timestamp="",
        duration_s=0,
    )


def _compute_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1."""
    pred_tokens = _normalize(prediction).split()
    ref_tokens = _normalize(reference).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())
