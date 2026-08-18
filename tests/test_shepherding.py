#!/usr/bin/env python3
"""Tests for shepherding (hint-based escalation).

Tests _try_shepherding and its integration with the confidence routing
in run_single(). Uses monkeypatching to mock frontier_query and
ollama_generate — no network required.
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.confidence import Confidence, ConfidenceConfig, set_config
from agent.agent_fabric import _try_shepherding, run_single


def _mock_frontier(hint_text="Try using the quadratic formula.", cost=0.03):
    def frontier_query(prompt, max_turns=1):
        return {
            "result": hint_text,
            "cost_usd": cost,
            "source": "frontier",
        }
    return frontier_query


def _mock_frontier_error():
    def frontier_query(prompt, max_turns=1):
        return {"result": "", "cost_usd": 0, "source": "frontier", "error": "timeout"}
    return frontier_query


def _mock_ollama(answer="42"):
    def ollama_generate(prompt, model=None):
        return {"result": answer, "cost_usd": 0.0, "source": "local"}
    return ollama_generate


class TestTryShepherding:
    def test_returns_guided_result(self):
        with patch("agent.agent_fabric.frontier_query", _mock_frontier("Use chain rule.")):
            with patch("agent.agent_fabric.ollama_generate", _mock_ollama("derivative is 2x")):
                result = _try_shepherding("differentiate x^2", "x squared")
        assert result is not None
        assert result["source"] == "shepherded"
        assert result["hint"] == "Use chain rule."
        assert result["result"] == "derivative is 2x"
        assert result["hint_cost_usd"] == 0.03
        assert result["cost_usd"] == 0.03

    def test_returns_none_on_frontier_error(self):
        with patch("agent.agent_fabric.frontier_query", _mock_frontier_error()):
            result = _try_shepherding("test prompt", "test answer")
        assert result is None

    def test_returns_none_on_empty_hint(self):
        with patch("agent.agent_fabric.frontier_query", _mock_frontier("")):
            result = _try_shepherding("test prompt", "test answer")
        assert result is None

    def test_returns_none_on_local_error(self):
        def bad_ollama(prompt, model=None):
            return {"result": "", "error": "connection refused", "cost_usd": 0}
        with patch("agent.agent_fabric.frontier_query", _mock_frontier("hint")):
            with patch("agent.agent_fabric.ollama_generate", bad_ollama):
                result = _try_shepherding("test", "answer")
        assert result is None

    def test_hint_prompt_truncates_long_input(self):
        long_prompt = "x" * 2000
        long_answer = "y" * 1000
        captured = {}

        def capturing_frontier(prompt, max_turns=1):
            captured["prompt"] = prompt
            return {"result": "hint", "cost_usd": 0.03, "source": "frontier"}

        with patch("agent.agent_fabric.frontier_query", capturing_frontier):
            with patch("agent.agent_fabric.ollama_generate", _mock_ollama()):
                _try_shepherding(long_prompt, long_answer)

        assert len(captured["prompt"]) < len(long_prompt) + len(long_answer)

    def test_guided_prompt_includes_hint(self):
        captured = {}

        def capturing_ollama(prompt, model=None):
            captured["prompt"] = prompt
            return {"result": "guided answer", "cost_usd": 0.0, "source": "local"}

        with patch("agent.agent_fabric.frontier_query", _mock_frontier("Use modular arithmetic.")):
            with patch("agent.agent_fabric.ollama_generate", capturing_ollama):
                _try_shepherding("compute 17 mod 5", "wrong answer")

        assert "Use modular arithmetic." in captured["prompt"]
        assert "compute 17 mod 5" in captured["prompt"]


class TestShepherdingIntegration:
    """Test shepherding in the run_single routing flow."""

    def setup_method(self):
        set_config(ConfidenceConfig.from_preset("balanced"))

    def test_medium_confidence_tries_shepherding(self):
        responses = ["42", "42", "48", "3"]  # 2/3 agreement + verify=3 → MEDIUM
        call_idx = [0]

        def mock_gen_temp(prompt, temperature=0.0):
            r = responses[call_idx[0] % len(responses)]
            call_idx[0] += 1
            return {"result": r, "cost_usd": 0.0, "source": "local"}

        with patch("agent.agent_fabric.ollama_generate", _mock_ollama("original")):
            with patch("agent.agent_fabric.ollama_generate_with_temp", mock_gen_temp):
                with patch("agent.agent_fabric.frontier_query", _mock_frontier("hint")):
                    with patch("agent.agent_fabric.classify_task", return_value="local"):
                        result = run_single("test prompt")

        assert result["source"] == "shepherded"
        assert result["fallback_reason"] == "shepherding"

    def test_low_confidence_tries_shepherding_before_frontier(self):
        responses = ["a", "b", "c"]  # all different → LOW
        call_idx = [0]

        def mock_gen_temp(prompt, temperature=0.0):
            r = responses[call_idx[0] % len(responses)]
            call_idx[0] += 1
            return {"result": r, "cost_usd": 0.0, "source": "local"}

        with patch("agent.agent_fabric.ollama_generate", _mock_ollama("original")):
            with patch("agent.agent_fabric.ollama_generate_with_temp", mock_gen_temp):
                with patch("agent.agent_fabric.frontier_query", _mock_frontier("useful hint")):
                    with patch("agent.agent_fabric.classify_task", return_value="local"):
                        result = run_single("test prompt")

        assert result["source"] == "shepherded"
        assert result["fallback_reason"] == "shepherding"

    def test_low_confidence_falls_back_to_frontier_when_shepherding_fails(self):
        responses = ["a", "b", "c"]
        call_idx = [0]

        def mock_gen_temp(prompt, temperature=0.0):
            r = responses[call_idx[0] % len(responses)]
            call_idx[0] += 1
            return {"result": r, "cost_usd": 0.0, "source": "local"}

        frontier_calls = [0]

        def mock_frontier(prompt, max_turns=1):
            frontier_calls[0] += 1
            if frontier_calls[0] == 1:
                # First call: shepherding hint fails
                return {"result": "", "cost_usd": 0, "source": "frontier", "error": "timeout"}
            # Second call: full frontier fallback
            return {"result": "frontier answer", "cost_usd": 0.19, "source": "frontier"}

        with patch("agent.agent_fabric.ollama_generate", _mock_ollama("original")):
            with patch("agent.agent_fabric.ollama_generate_with_temp", mock_gen_temp):
                with patch("agent.agent_fabric.frontier_query", mock_frontier):
                    with patch("agent.agent_fabric.classify_task", return_value="local"):
                        result = run_single("test prompt")

        assert result["source"] == "frontier"
        assert result["fallback"] is True
        assert result["fallback_reason"] == "low_confidence"

    def test_high_confidence_skips_shepherding(self):
        def mock_gen_temp(prompt, temperature=0.0):
            return {"result": "42", "cost_usd": 0.0, "source": "local"}

        frontier_called = [False]
        original_frontier = _mock_frontier()

        def tracking_frontier(prompt, max_turns=1):
            frontier_called[0] = True
            return original_frontier(prompt, max_turns)

        with patch("agent.agent_fabric.ollama_generate", _mock_ollama("42")):
            with patch("agent.agent_fabric.ollama_generate_with_temp", mock_gen_temp):
                with patch("agent.agent_fabric.frontier_query", tracking_frontier):
                    with patch("agent.agent_fabric.classify_task", return_value="local"):
                        result = run_single("test prompt")

        assert result["source"] == "local"
        assert not frontier_called[0]

    def test_shepherded_result_has_cost_metadata(self):
        def mock_gen_temp(prompt, temperature=0.0):
            return {"result": "42", "cost_usd": 0.0, "source": "local"}

        # Force MEDIUM confidence: set config so 100% agreement (1 sample) is still medium
        cfg = ConfidenceConfig(
            trust="test", n_samples=1, temperature=0.7,
            high_threshold=1.1, low_threshold=0.0,
            verify_pass_score=5, mc_high_threshold=0.95,
            mc_low_threshold=0.55, mc_samples=5,
        )
        set_config(cfg)

        with patch("agent.agent_fabric.ollama_generate", _mock_ollama("42")):
            with patch("agent.agent_fabric.ollama_generate_with_temp", mock_gen_temp):
                with patch("agent.agent_fabric.frontier_query", _mock_frontier("hint", 0.03)):
                    with patch("agent.agent_fabric.classify_task", return_value="local"):
                        result = run_single("test", use_confidence=True)

        assert result["cost_usd"] == 0.03
        assert "hint" in result


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
