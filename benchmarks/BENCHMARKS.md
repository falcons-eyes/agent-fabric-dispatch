# LLM Benchmark Guide

## Running Benchmarks

```bash
# List available benchmarks
python benchmarks/runner.py --list

# Run a specific benchmark
python benchmarks/runner.py --model qwen3-coder:30b --benchmarks gsm8k --limit 100

# Run multiple benchmarks
python benchmarks/runner.py --model qwen3-coder:30b --benchmarks mmlu,gsm8k,truthfulqa --limit 50

# Run agent-fabric comparison (local-first + frontier fallback)
python benchmarks/run_fabric_compare.py --benchmarks gsm8k,math,mmlu --limit 50
```

## Benchmark Categories

### Tier 1: Fully Local (no external infra needed)

| Benchmark | Dataset | Size | Input | Output | Metric |
|-----------|---------|------|-------|--------|--------|
| MMLU | `cais/mmlu` | 14K (57 subjects) | 4-choice MCQ | Letter (A-D) | Accuracy |
| GSM8K | `openai/gsm8k` | 1,319 | Grade-school math | Number after `####` | Exact match |
| MATH | `hendrycks/competition_math` | 5,000 | Competition math | LaTeX `\boxed{}` | LaTeX equiv |
| AIME | `MathArena/aime_2025` | ~30/year | AMC competition | Integer 0-999 | Exact match |
| TruthfulQA | `truthfulqa/truthful_qa` | 817 | Misleading questions | MCQ letter | Accuracy |
| LongBench | `THUDM/LongBench` | 4,750 | Long docs + questions | Free text | F1/ROUGE-L |
| RAGBench | `galileo-ai/ragbench` | ~100K | Question + context | Generated answer | F1/AUROC |
| HLE | `cais/hle` | ~3,000 | Expert-level MCQ | Letter or text | Accuracy |

### Tier 2: Requires Code Execution or LLM Judge

| Benchmark | Dataset | Evaluation |
|-----------|---------|-----------|
| HumanEval | `openai/openai_humaneval` | pass@k (sandbox needed) |
| MBPP | `google-research-datasets/mbpp` | pass@k (sandbox needed) |
| HarmBench | `walledai/HarmBench` | Refusal rate |
| StrongREJECT | `walledai/StrongREJECT` | Refusal score |

### Tier 3+: Heavy Infrastructure

SWE-bench (Docker), GAIA (Agent), WebArena (Browser), MMMU (Multimodal)

## Implemented Runners

All Tier 1 benchmarks are implemented in `benchmarks/runners/`:
- `math_bench.py` — GSM8K, MATH, AIME
- `knowledge.py` — MMLU, MMLU-Pro
- `coding.py` — HumanEval, MBPP
- `hallucination.py` — TruthfulQA, SimpleQA
- `safety.py` — HarmBench, StrongREJECT
- `longctx.py` — LongBench
- `rag.py` — RAGBench
- `reasoning.py` — GPQA, HLE
