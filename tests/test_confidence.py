#!/usr/bin/env python3
"""Tests for confidence estimation module.

Tests the self-consistency, self-verification, and combined confidence
estimation logic using mock generate functions (no Ollama required).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.confidence import (
    Confidence,
    choice_shuffle_consistency,
    estimate_confidence,
    extract_core_answer,
    is_multiple_choice,
    normalize_answer,
    self_consistency,
    self_verify,
    shuffle_choices,
)


def make_generate_fn(responses):
    """Create a mock generate function that cycles through responses."""
    idx = [0]
    def fn(prompt, temperature=0.0):
        result = responses[idx[0] % len(responses)]
        idx[0] += 1
        if isinstance(result, str):
            return {"result": result, "cost_usd": 0.0, "source": "local"}
        return result
    return fn


class TestNormalizeAnswer:
    def test_basic(self):
        assert normalize_answer("  Hello World  ") == "hello world"

    def test_whitespace_collapse(self):
        assert normalize_answer("a  b\n\nc") == "a b c"

    def test_punctuation_removal(self):
        assert normalize_answer("Hello, World!") == "hello world"

    def test_empty(self):
        assert normalize_answer("") == ""


class TestExtractCoreAnswer:
    def test_code_block(self):
        text = "Here is the code:\n```python\ndef foo():\n    pass\n```\nDone."
        assert "def foo" in extract_core_answer(text)

    def test_short_single_line(self):
        assert extract_core_answer("algorithm") == "algorithm"

    def test_long_text_uses_prefix(self):
        long_text = "word " * 100
        result = extract_core_answer(long_text)
        assert len(result) <= 200

    def test_empty(self):
        assert extract_core_answer("") == ""


class TestSelfConsistency:
    def test_full_agreement(self):
        gen = make_generate_fn(["42"])
        result = self_consistency("What is 6*7?", gen, n=3)
        assert result["agreement"] == 1.0
        assert result["majority_answer"] == "42"
        assert result["n_unique"] == 1

    def test_partial_agreement(self):
        gen = make_generate_fn(["42", "42", "48"])
        result = self_consistency("What is 6*7?", gen, n=3)
        assert abs(result["agreement"] - 2/3) < 0.01
        assert result["n_unique"] == 2

    def test_no_agreement(self):
        gen = make_generate_fn(["42", "48", "36"])
        result = self_consistency("What is 6*7?", gen, n=3)
        assert abs(result["agreement"] - 1/3) < 0.01
        assert result["n_unique"] == 3

    def test_empty_responses(self):
        gen = make_generate_fn(["", "", ""])
        result = self_consistency("test", gen, n=3)
        assert result["agreement"] == 0.0

    def test_error_response(self):
        gen = make_generate_fn([{"result": "", "error": "timeout", "cost_usd": 0}])
        result = self_consistency("test", gen, n=3)
        assert result["agreement"] == 0.0


class TestMultipleChoice:
    def test_detect_mc_parenthesis(self):
        prompt = "Question:\nA) foo\nB) bar\nC) baz\nD) qux"
        assert is_multiple_choice(prompt) is True

    def test_detect_mc_period(self):
        prompt = "Question:\nA. foo\nB. bar\nC. baz\nD. qux"
        assert is_multiple_choice(prompt) is True

    def test_detect_non_mc(self):
        prompt = "Solve this math problem: 2 + 2 = ?"
        assert is_multiple_choice(prompt) is False

    def test_shuffle_choices_preserves_content(self):
        prompt = "Q: What?\nA) alpha\nB) beta\nC) gamma\nD) delta"
        shuffled, mapping = shuffle_choices(prompt, seed=42)
        assert "alpha" in shuffled
        assert "beta" in shuffled
        assert "gamma" in shuffled
        assert "delta" in shuffled
        assert len(mapping) == 4

    def test_shuffle_choices_mapping_correct(self):
        prompt = "Q: What?\nA) alpha\nB) beta\nC) gamma\nD) delta"
        _, mapping = shuffle_choices(prompt, seed=42)
        originals = set(mapping.values())
        assert originals == {"A", "B", "C", "D"}

    def test_choice_shuffle_full_agreement(self):
        # Model always picks "alpha" content regardless of position
        call_count = [0]
        def gen(prompt, temp):
            call_count[0] += 1
            # Find which letter "alpha" is at in this shuffled prompt
            import re
            for m in re.finditer(r'([A-D])\)\s*(\w+)', prompt):
                if m.group(2) == "alpha":
                    return {"result": m.group(1)}
            return {"result": "A"}

        prompt = "Q: What?\nA) alpha\nB) beta\nC) gamma\nD) delta"
        result = choice_shuffle_consistency(prompt, gen, n=3)
        assert result["agreement"] == 1.0

    def test_choice_shuffle_position_bias(self):
        # Model always picks "A" regardless of content
        def gen(prompt, temp):
            return {"result": "A"}

        prompt = "Q: What?\nA) alpha\nB) beta\nC) gamma\nD) delta"
        result = choice_shuffle_consistency(prompt, gen, n=3)
        # When shuffled, "A" maps to different original letters each time
        assert result["agreement"] < 1.0


class TestSelfVerify:
    def test_high_confidence(self):
        gen = make_generate_fn(["5"])
        result = self_verify("What is 2+2?", "4", gen)
        assert result["score"] == 5

    def test_low_confidence(self):
        gen = make_generate_fn(["1"])
        result = self_verify("What is sqrt(-1)?", "no solution", gen)
        assert result["score"] == 1

    def test_verbose_response_extracts_score(self):
        gen = make_generate_fn(["I'd rate this a 4 out of 5 because..."])
        result = self_verify("test", "answer", gen)
        assert result["score"] == 4

    def test_no_number_defaults_to_3(self):
        gen = make_generate_fn(["I'm not sure about this answer"])
        result = self_verify("test", "answer", gen)
        assert result["score"] == 3


class TestEstimateConfidence:
    def test_empty_result_is_low(self):
        gen = make_generate_fn(["anything"])
        result = estimate_confidence("test", {"result": ""}, gen)
        assert result["confidence"] == Confidence.LOW

    def test_error_result_is_low(self):
        gen = make_generate_fn(["anything"])
        result = estimate_confidence("test", {"result": "x", "error": "fail"}, gen)
        assert result["confidence"] == Confidence.LOW

    def test_high_agreement_is_high(self):
        gen = make_generate_fn(["42"])
        result = estimate_confidence("What is 6*7?", {"result": "42"}, gen)
        assert result["confidence"] == Confidence.HIGH
        assert result["agreement"] == 1.0

    def test_low_agreement_is_low(self):
        gen = make_generate_fn(["42", "48", "36"])
        result = estimate_confidence("test", {"result": "42"}, gen)
        assert result["confidence"] == Confidence.LOW
        assert result["agreement"] < 0.5

    def test_medium_agreement_with_high_verify_is_high(self):
        # 2/3 agreement (borderline) + verify score 5 → HIGH
        responses = ["42", "42", "48", "5"]  # 3 SC samples + 1 verify
        gen = make_generate_fn(responses)
        result = estimate_confidence("test", {"result": "42"}, gen)
        assert result["confidence"] == Confidence.HIGH
        assert result["verify_score"] == 5

    def test_medium_agreement_with_low_verify_is_low(self):
        # 2/3 agreement (borderline) + verify score 1 → LOW
        responses = ["42", "42", "48", "1"]
        gen = make_generate_fn(responses)
        result = estimate_confidence("test", {"result": "42"}, gen)
        assert result["confidence"] == Confidence.LOW
        assert result["verify_score"] == 1

    def test_medium_agreement_with_medium_verify_is_medium(self):
        # 2/3 agreement (borderline) + verify score 3 → MEDIUM
        responses = ["42", "42", "48", "3"]
        gen = make_generate_fn(responses)
        result = estimate_confidence("test", {"result": "42"}, gen)
        assert result["confidence"] == Confidence.MEDIUM
        assert result["verify_score"] == 3

    def test_best_answer_uses_majority(self):
        gen = make_generate_fn(["correct answer"])
        result = estimate_confidence("test", {"result": "original"}, gen)
        assert result["best_answer"] == "correct answer"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
