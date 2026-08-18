#!/usr/bin/env python3
"""Multi-model voting for verification loops.

Runs the same prompt across multiple local models and votes on the answer.
Different models make different mistakes — majority agreement across diverse
models is a stronger confidence signal than self-consistency on one model.

Voting strategies:
  - majority: pick the most common answer
  - weighted: weight by model-specific reliability scores
  - unanimous: require all models to agree (high precision, low recall)

Configuration in ~/.fabric/policy.yaml:
    voting:
      enabled: true
      models: ["qwen3-coder:30b", "gemma4:27b-a4b", "devstral-small:24b"]
      strategy: majority          # majority, weighted, unanimous
      min_agreement: 0.67         # fraction required for majority
"""

import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from agent.confidence import extract_core_answer

OLLAMA_API_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


@dataclass
class VotingConfig:
    enabled: bool = False
    models: list = field(default_factory=list)
    strategy: str = "majority"
    min_agreement: float = 0.67

    @classmethod
    def from_policy(cls, policy_path: str = None) -> "VotingConfig":
        if policy_path is None:
            policy_path = Path.home() / ".fabric" / "policy.yaml"
        else:
            policy_path = Path(policy_path)

        if not policy_path.exists():
            return cls()

        try:
            import yaml
            with open(policy_path) as f:
                policy = yaml.safe_load(f) or {}
        except Exception:
            return cls()

        voting = policy.get("voting", {})
        if not voting:
            return cls()

        return cls(
            enabled=voting.get("enabled", False),
            models=voting.get("models", []),
            strategy=voting.get("strategy", "majority"),
            min_agreement=voting.get("min_agreement", 0.67),
        )


def generate_with_model(prompt: str, model: str, temperature: float = 0.0) -> dict:
    url = f"{OLLAMA_API_URL}/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }).encode("utf-8")

    t0 = time.time()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            dt = time.time() - t0
            return {
                "result": data.get("response", "").strip(),
                "model": model,
                "duration_ms": round(dt * 1000),
                "cost_usd": 0.0,
                "source": "local",
            }
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        return {"result": "", "model": model, "cost_usd": 0.0, "source": "local", "error": str(e)}


def multi_model_vote(prompt: str, models: list[str],
                     strategy: str = "majority",
                     min_agreement: float = 0.67) -> dict:
    """Run prompt on multiple models and vote on the answer."""
    results = []
    answers = []
    raw_answers = []

    for model in models:
        result = generate_with_model(prompt, model)
        results.append(result)
        raw = result.get("result", "")
        raw_answers.append(raw)
        if raw and not result.get("error"):
            answers.append(extract_core_answer(raw))
        else:
            answers.append("")

    valid_answers = [a for a in answers if a]
    if not valid_answers:
        return {
            "result": "",
            "confidence": "low",
            "agreement": 0.0,
            "votes": results,
            "strategy": strategy,
        }

    counts = Counter(valid_answers)
    best_answer, best_count = counts.most_common(1)[0]
    agreement = best_count / len(models)

    best_idx = answers.index(best_answer)
    best_raw = raw_answers[best_idx]

    if strategy == "unanimous":
        confident = agreement == 1.0
    elif strategy == "majority":
        confident = agreement >= min_agreement
    else:
        confident = agreement >= min_agreement

    return {
        "result": best_raw,
        "confidence": "high" if confident else "low",
        "agreement": agreement,
        "n_models": len(models),
        "n_agree": best_count,
        "n_unique": len(counts),
        "winning_model": results[best_idx].get("model", ""),
        "votes": results,
        "strategy": strategy,
        "cost_usd": 0.0,
        "source": "voted",
    }


def verify_answer(prompt: str, answer: str, models: list[str]) -> dict:
    """Ask multiple models to verify an answer and vote on correctness.

    Each model rates the answer 1-5, and the median score determines
    the verdict.
    """
    verify_prompt = (
        f"Rate this answer's correctness from 1-5.\n"
        f"5=definitely correct, 1=definitely wrong.\n\n"
        f"Question: {prompt[:500]}\n"
        f"Answer: {answer[:1000]}\n\n"
        f"Reply with ONLY a number 1-5."
    )

    import re
    scores = []
    results = []

    for model in models:
        result = generate_with_model(verify_prompt, model, temperature=0.0)
        results.append(result)
        raw = result.get("result", "").strip()
        match = re.search(r'[1-5]', raw)
        if match:
            scores.append(int(match.group()))

    if not scores:
        return {"verdict": "unknown", "scores": [], "results": results}

    scores.sort()
    median = scores[len(scores) // 2]
    avg = sum(scores) / len(scores)

    return {
        "verdict": "correct" if median >= 4 else ("uncertain" if median >= 3 else "wrong"),
        "median_score": median,
        "avg_score": round(avg, 1),
        "scores": scores,
        "n_models": len(models),
        "results": results,
    }
