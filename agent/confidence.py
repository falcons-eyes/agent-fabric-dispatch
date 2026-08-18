#!/usr/bin/env python3
"""Confidence estimation for local model outputs.

Combines two training-free signals to decide whether a local answer
should be returned or escalated to the frontier model:

1. Self-consistency: sample N times with temperature > 0, measure agreement.
   High agreement = the model "knows" the answer. Low agreement = uncertain.

2. Self-verification (AutoMix-style): ask the model to critique its own answer.
   Models are better at judging answers than generating them.

3. Choice shuffling: for multiple-choice, permute answer order to detect
   position bias vs real knowledge.

All thresholds are configurable via ConfidenceConfig, which can be loaded
from ~/.fabric/policy.yaml under the `confidence` key.

References:
  - Self-Consistency (Wang et al., 2022; EMNLP 2024)
  - AutoMix (Aggarwal & Madaan et al., NeurIPS 2024)
"""

import random
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Confidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


TRUST_PRESETS = {
    "conservative": {
        "n_samples": 5,
        "temperature": 0.8,
        "high_threshold": 0.92,
        "low_threshold": 0.65,
        "verify_pass_score": 5,
        "mc_high_threshold": 0.98,
        "mc_low_threshold": 0.60,
        "mc_samples": 7,
    },
    "balanced": {
        "n_samples": 3,
        "temperature": 0.7,
        "high_threshold": 0.85,
        "low_threshold": 0.50,
        "verify_pass_score": 4,
        "mc_high_threshold": 0.95,
        "mc_low_threshold": 0.55,
        "mc_samples": 5,
    },
    "aggressive": {
        "n_samples": 2,
        "temperature": 0.5,
        "high_threshold": 0.70,
        "low_threshold": 0.35,
        "verify_pass_score": 3,
        "mc_high_threshold": 0.75,
        "mc_low_threshold": 0.40,
        "mc_samples": 3,
    },
    "max": {
        "n_samples": 0,
        "temperature": 0.0,
        "high_threshold": 0.0,
        "low_threshold": 0.0,
        "verify_pass_score": 1,
        "mc_high_threshold": 0.0,
        "mc_low_threshold": 0.0,
        "mc_samples": 0,
    },
}


@dataclass
class ConfidenceConfig:
    trust: str = "balanced"
    n_samples: int = 3
    temperature: float = 0.7
    high_threshold: float = 0.85
    low_threshold: float = 0.50
    verify_pass_score: int = 4
    mc_high_threshold: float = 0.95
    mc_low_threshold: float = 0.55
    mc_samples: int = 5

    @classmethod
    def from_preset(cls, trust: str) -> "ConfidenceConfig":
        if trust in TRUST_PRESETS:
            return cls(trust=trust, **TRUST_PRESETS[trust])
        try:
            level = float(trust)
            return cls._from_float(level)
        except (ValueError, TypeError):
            return cls()

    @classmethod
    def _from_float(cls, level: float) -> "ConfidenceConfig":
        """Interpolate between conservative (0.0) and aggressive (1.0)."""
        level = max(0.0, min(1.0, level))
        if level >= 1.0:
            return cls.from_preset("max")

        c = TRUST_PRESETS["conservative"]
        a = TRUST_PRESETS["aggressive"]

        def lerp(key):
            return c[key] + (a[key] - c[key]) * level

        return cls(
            trust=str(level),
            n_samples=max(1, round(lerp("n_samples"))),
            temperature=round(lerp("temperature"), 2),
            high_threshold=round(lerp("high_threshold"), 2),
            low_threshold=round(lerp("low_threshold"), 2),
            verify_pass_score=max(1, round(lerp("verify_pass_score"))),
            mc_high_threshold=round(lerp("mc_high_threshold"), 2),
            mc_low_threshold=round(lerp("mc_low_threshold"), 2),
            mc_samples=max(1, round(lerp("mc_samples"))),
        )

    @classmethod
    def from_policy(cls, policy_path: str = None) -> "ConfidenceConfig":
        """Load from ~/.fabric/policy.yaml confidence section."""
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

        conf = policy.get("confidence", {})
        if not conf:
            return cls()

        trust = conf.get("trust", "balanced")
        base = cls.from_preset(trust)

        for key in ("n_samples", "temperature", "high_threshold",
                     "low_threshold", "verify_pass_score",
                     "mc_high_threshold", "mc_low_threshold", "mc_samples"):
            if key in conf:
                setattr(base, key, conf[key])

        return base

    @property
    def skip_confidence(self) -> bool:
        return self.trust == "max" or self.n_samples == 0


_default_config = ConfidenceConfig()


def set_config(config: ConfidenceConfig):
    global _default_config
    _default_config = config


def get_config() -> ConfidenceConfig:
    return _default_config


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


def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


def extract_core_answer(text: str) -> str:
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
    return bool(re.search(r'\b[A-D]\)', prompt) or re.search(r'\b[A-D]\.\s', prompt))


def shuffle_choices(prompt: str, seed: int) -> tuple:
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
                               n: int = None) -> dict:
    cfg = get_config()
    n = n if n is not None else cfg.mc_samples

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


def self_consistency(prompt: str, generate_fn, n: int = None,
                     temperature: float = None) -> dict:
    cfg = get_config()
    n = n if n is not None else cfg.n_samples
    temperature = temperature if temperature is not None else cfg.temperature

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
    verify_prompt = SELF_VERIFY_PROMPT.format(question=question[:500], answer=answer[:1000])
    result = generate_fn(verify_prompt, 0.0)
    raw = result.get("result", "").strip()

    match = re.search(r'[1-5]', raw)
    score = int(match.group()) if match else 3

    return {"score": score, "raw": raw}


def estimate_confidence(prompt: str, initial_result: dict,
                        generate_fn, config: ConfidenceConfig = None) -> dict:
    cfg = config or get_config()

    answer = initial_result.get("result", "")
    if not answer or initial_result.get("error"):
        return {
            "confidence": Confidence.LOW,
            "agreement": 0.0,
            "verify_score": None,
            "best_answer": answer,
            "signals": {"reason": "empty_or_error"},
        }

    if cfg.skip_confidence:
        return {
            "confidence": Confidence.HIGH,
            "agreement": 1.0,
            "verify_score": None,
            "best_answer": answer,
            "signals": {"reason": "trust_max"},
        }

    mc = is_multiple_choice(prompt)
    if mc:
        sc = choice_shuffle_consistency(prompt, generate_fn, n=cfg.mc_samples)
    else:
        sc = self_consistency(prompt, generate_fn, n=cfg.n_samples,
                              temperature=cfg.temperature)
    agreement = sc["agreement"]

    if mc:
        if agreement >= cfg.mc_high_threshold:
            confidence = Confidence.HIGH
        elif agreement < cfg.mc_low_threshold:
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

    if agreement >= cfg.high_threshold:
        return {
            "confidence": Confidence.HIGH,
            "agreement": agreement,
            "verify_score": None,
            "best_answer": sc["majority_answer"] or answer,
            "signals": {"self_consistency": sc},
        }

    if agreement < cfg.low_threshold:
        return {
            "confidence": Confidence.LOW,
            "agreement": agreement,
            "verify_score": None,
            "best_answer": sc["majority_answer"] or answer,
            "signals": {"self_consistency": sc},
        }

    sv = self_verify(prompt, sc["majority_answer"] or answer, generate_fn)

    if sv["score"] >= cfg.verify_pass_score:
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
