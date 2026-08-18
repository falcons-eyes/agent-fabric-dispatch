#!/usr/bin/env python3
"""Tests for multi-model voting and verification."""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.voting import VotingConfig, multi_model_vote, verify_answer


def _mock_generate(responses_by_model):
    """Mock generate_with_model that returns per-model responses."""
    call_idx = {}

    def fn(prompt, model, temperature=0.0):
        if model not in call_idx:
            call_idx[model] = 0
        idx = call_idx[model]
        call_idx[model] += 1

        if model in responses_by_model:
            resp = responses_by_model[model]
            if isinstance(resp, list):
                text = resp[idx % len(resp)]
            else:
                text = resp
        else:
            text = ""

        return {"result": text, "model": model, "cost_usd": 0.0, "source": "local"}

    return fn


class TestVotingConfig:
    def test_default_disabled(self):
        cfg = VotingConfig()
        assert cfg.enabled is False
        assert cfg.models == []

    def test_from_policy_missing_file(self):
        cfg = VotingConfig.from_policy("/nonexistent/policy.yaml")
        assert cfg.enabled is False


class TestMultiModelVote:
    def test_unanimous_agreement(self):
        mock = _mock_generate({
            "model-a": "42",
            "model-b": "42",
            "model-c": "42",
        })
        with patch("agent.voting.generate_with_model", mock):
            result = multi_model_vote("What is 6*7?", ["model-a", "model-b", "model-c"])

        assert result["agreement"] == 1.0
        assert result["confidence"] == "high"
        assert result["n_agree"] == 3
        assert result["n_unique"] == 1
        assert result["source"] == "voted"

    def test_majority_agreement(self):
        mock = _mock_generate({
            "model-a": "42",
            "model-b": "42",
            "model-c": "48",
        })
        with patch("agent.voting.generate_with_model", mock):
            result = multi_model_vote("What is 6*7?", ["model-a", "model-b", "model-c"],
                                      min_agreement=0.6)

        assert abs(result["agreement"] - 2/3) < 0.01
        assert result["confidence"] == "high"
        assert result["n_agree"] == 2

    def test_no_agreement(self):
        mock = _mock_generate({
            "model-a": "42",
            "model-b": "48",
            "model-c": "36",
        })
        with patch("agent.voting.generate_with_model", mock):
            result = multi_model_vote("What is 6*7?", ["model-a", "model-b", "model-c"])

        assert abs(result["agreement"] - 1/3) < 0.01
        assert result["confidence"] == "low"

    def test_unanimous_strategy(self):
        mock = _mock_generate({
            "model-a": "42",
            "model-b": "42",
            "model-c": "48",
        })
        with patch("agent.voting.generate_with_model", mock):
            result = multi_model_vote("test", ["model-a", "model-b", "model-c"],
                                      strategy="unanimous")

        assert result["confidence"] == "low"

    def test_all_errors(self):
        def error_gen(prompt, model, temperature=0.0):
            return {"result": "", "model": model, "error": "timeout", "cost_usd": 0}

        with patch("agent.voting.generate_with_model", error_gen):
            result = multi_model_vote("test", ["model-a", "model-b"])

        assert result["confidence"] == "low"
        assert result["agreement"] == 0.0

    def test_partial_errors(self):
        mock = _mock_generate({
            "model-a": "42",
            "model-b": "",
            "model-c": "42",
        })
        with patch("agent.voting.generate_with_model", mock):
            result = multi_model_vote("test", ["model-a", "model-b", "model-c"],
                                      min_agreement=0.6)

        assert result["n_agree"] == 2
        assert result["confidence"] == "high"

    def test_cost_is_zero(self):
        mock = _mock_generate({"model-a": "42", "model-b": "42"})
        with patch("agent.voting.generate_with_model", mock):
            result = multi_model_vote("test", ["model-a", "model-b"])
        assert result["cost_usd"] == 0.0

    def test_custom_min_agreement(self):
        mock = _mock_generate({
            "model-a": "42",
            "model-b": "42",
            "model-c": "48",
            "model-d": "36",
        })
        with patch("agent.voting.generate_with_model", mock):
            result = multi_model_vote("test", ["model-a", "model-b", "model-c", "model-d"],
                                      min_agreement=0.75)

        assert result["agreement"] == 0.5
        assert result["confidence"] == "low"


class TestVerifyAnswer:
    def test_correct_verdict(self):
        mock = _mock_generate({
            "model-a": "5",
            "model-b": "4",
            "model-c": "5",
        })
        with patch("agent.voting.generate_with_model", mock):
            result = verify_answer("What is 2+2?", "4",
                                   ["model-a", "model-b", "model-c"])

        assert result["verdict"] == "correct"
        assert result["median_score"] == 5
        assert result["n_models"] == 3

    def test_wrong_verdict(self):
        mock = _mock_generate({
            "model-a": "1",
            "model-b": "2",
            "model-c": "1",
        })
        with patch("agent.voting.generate_with_model", mock):
            result = verify_answer("What is sqrt(-1)?", "no solution",
                                   ["model-a", "model-b", "model-c"])

        assert result["verdict"] == "wrong"
        assert result["median_score"] <= 2

    def test_uncertain_verdict(self):
        mock = _mock_generate({
            "model-a": "3",
            "model-b": "3",
            "model-c": "4",
        })
        with patch("agent.voting.generate_with_model", mock):
            result = verify_answer("test", "maybe",
                                   ["model-a", "model-b", "model-c"])

        assert result["verdict"] == "uncertain"

    def test_all_errors(self):
        def error_gen(prompt, model, temperature=0.0):
            return {"result": "", "model": model, "error": "fail", "cost_usd": 0}

        with patch("agent.voting.generate_with_model", error_gen):
            result = verify_answer("test", "answer", ["model-a"])

        assert result["verdict"] == "unknown"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
