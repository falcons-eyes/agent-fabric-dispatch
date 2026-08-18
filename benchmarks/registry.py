"""Benchmark registry — maps names to runner functions and metadata."""

from typing import Callable

# Lazy imports to avoid loading all runners at startup
BENCHMARK_REGISTRY = {
    "General Knowledge": [
        {"name": "mmlu", "hf_path": "cais/mmlu", "metric": "accuracy", "local_eval": "Yes"},
        {"name": "mmlu_pro", "hf_path": "TIGER-Lab/MMLU-Pro", "metric": "accuracy", "local_eval": "Yes"},
    ],
    "Reasoning": [
        {"name": "gpqa", "hf_path": "Idavidrein/gpqa", "metric": "accuracy", "local_eval": "Yes (gated)"},
        {"name": "arc_agi", "hf_path": "fchollet/ARC-AGI (GitHub)", "metric": "exact_grid", "local_eval": "Custom"},
        {"name": "hle", "hf_path": "cais/hle", "metric": "accuracy", "local_eval": "Yes"},
    ],
    "Math": [
        {"name": "gsm8k", "hf_path": "openai/gsm8k", "metric": "accuracy", "local_eval": "Yes"},
        {"name": "math", "hf_path": "hendrycks/competition_math", "metric": "accuracy", "local_eval": "Yes"},
        {"name": "aime", "hf_path": "MathArena/aime_2025", "metric": "accuracy", "local_eval": "Yes"},
        {"name": "omni_math", "hf_path": "KbsdJames/Omni-MATH", "metric": "llm_judge", "local_eval": "Partial"},
    ],
    "Coding": [
        {"name": "humaneval", "hf_path": "openai/openai_humaneval", "metric": "pass@1", "local_eval": "Sandbox"},
        {"name": "mbpp", "hf_path": "google-research-datasets/mbpp", "metric": "pass@1", "local_eval": "Sandbox"},
        {"name": "swe_bench", "hf_path": "princeton-nlp/SWE-bench_Verified", "metric": "resolve_rate", "local_eval": "Docker"},
    ],
    "Agent": [
        {"name": "gaia", "hf_path": "gaia-benchmark/GAIA", "metric": "exact_match", "local_eval": "Agent infra"},
        {"name": "browserarena", "hf_path": "N/A (live platform)", "metric": "task_success", "local_eval": "Browser"},
        {"name": "webarena", "hf_path": "N/A (Docker)", "metric": "task_success", "local_eval": "Docker+Browser"},
    ],
    "Long Context": [
        {"name": "longbench", "hf_path": "THUDM/LongBench", "metric": "F1/ROUGE", "local_eval": "Yes"},
        {"name": "infinitebench", "hf_path": "xinrongzhang2022/InfiniteBench", "metric": "F1/accuracy", "local_eval": "Yes (100K+)"},
        {"name": "ruler", "hf_path": "NVIDIA/RULER (GitHub)", "metric": "accuracy", "local_eval": "Generate"},
    ],
    "RAG": [
        {"name": "ragbench", "hf_path": "galileo-ai/ragbench", "metric": "F1/AUROC", "local_eval": "Yes"},
        {"name": "rgb", "hf_path": "chen700564/RGB (GitHub)", "metric": "accuracy", "local_eval": "Custom"},
    ],
    "Dialogue": [
        {"name": "mt_bench", "hf_path": "lmsys/mt_bench_human_judgments", "metric": "llm_judge_1-10", "local_eval": "LLM judge"},
        {"name": "arena_hard", "hf_path": "lmarena-ai/arena-hard-auto-v0.1", "metric": "win_rate", "local_eval": "LLM judge"},
    ],
    "Multimodal": [
        {"name": "mmmu", "hf_path": "MMMU/MMMU", "metric": "accuracy", "local_eval": "Multimodal"},
        {"name": "mmmu_pro", "hf_path": "MMMU/MMMU_Pro", "metric": "accuracy", "local_eval": "Multimodal"},
        {"name": "mathvista", "hf_path": "AI4Math/MathVista", "metric": "accuracy", "local_eval": "Multimodal"},
    ],
    "Safety": [
        {"name": "harmbench", "hf_path": "walledai/HarmBench", "metric": "ASR", "local_eval": "Classifier"},
        {"name": "strongreject", "hf_path": "walledai/StrongREJECT", "metric": "reject_score", "local_eval": "Evaluator"},
    ],
    "Hallucination": [
        {"name": "truthfulqa", "hf_path": "truthfulqa/truthful_qa", "metric": "accuracy", "local_eval": "Yes (MCQ)"},
        {"name": "simpleqa", "hf_path": "OpenEvals/SimpleQA", "metric": "accuracy", "local_eval": "LLM judge"},
    ],
}


def get_all_names() -> list[str]:
    """Get all benchmark names."""
    names = []
    for benches in BENCHMARK_REGISTRY.values():
        for b in benches:
            names.append(b["name"])
    return names


def get_runner(name: str):
    """Get the runner function for a benchmark (lazy import)."""
    runners = {
        # Tier 1: Fully local
        "mmlu": lambda **kw: _import_run("benchmarks.runners.knowledge", "run_mmlu", **kw),
        "mmlu_pro": lambda **kw: _import_run("benchmarks.runners.knowledge", "run_mmlu_pro", **kw),
        "gsm8k": lambda **kw: _import_run("benchmarks.runners.math_bench", "run_gsm8k", **kw),
        "math": lambda **kw: _import_run("benchmarks.runners.math_bench", "run_math", **kw),
        "aime": lambda **kw: _import_run("benchmarks.runners.math_bench", "run_aime", **kw),
        "truthfulqa": lambda **kw: _import_run("benchmarks.runners.hallucination", "run_truthfulqa", **kw),
        "humaneval": lambda **kw: _import_run("benchmarks.runners.coding", "run_humaneval", **kw),
        "mbpp": lambda **kw: _import_run("benchmarks.runners.coding", "run_mbpp", **kw),
        "longbench": lambda **kw: _import_run("benchmarks.runners.longctx", "run_longbench", **kw),
        "ragbench": lambda **kw: _import_run("benchmarks.runners.rag", "run_ragbench", **kw),
        "harmbench": lambda **kw: _import_run("benchmarks.runners.safety", "run_harmbench", **kw),
        "strongreject": lambda **kw: _import_run("benchmarks.runners.safety", "run_strongreject", **kw),
        "gpqa": lambda **kw: _import_run("benchmarks.runners.reasoning", "run_gpqa", **kw),
        "hle": lambda **kw: _import_run("benchmarks.runners.reasoning", "run_hle", **kw),
        "simpleqa": lambda **kw: _import_run("benchmarks.runners.hallucination", "run_simpleqa", **kw),
    }
    return runners.get(name)


def _import_run(module_path: str, func_name: str, **kwargs):
    """Lazy import and run a benchmark function."""
    import importlib
    mod = importlib.import_module(module_path)
    fn = getattr(mod, func_name)
    return fn(**kwargs)
