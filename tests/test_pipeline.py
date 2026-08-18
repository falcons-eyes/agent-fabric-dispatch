#!/usr/bin/env python3
"""Tests for step-level routing (pipeline mode).

Tests decompose_task and run_pipeline using mocks — no network required.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.confidence import ConfidenceConfig, set_config
from agent.fabric_agent import decompose_task, run_pipeline


def _mock_ollama_with_decompose(decompose_response, step_answers):
    """Mock that returns decompose JSON first, then step answers in order."""
    call_idx = [0]

    def ollama_generate(prompt, model=None):
        idx = call_idx[0]
        call_idx[0] += 1
        if idx == 0:
            return {"result": decompose_response, "cost_usd": 0.0, "source": "local"}
        answer_idx = min(idx - 1, len(step_answers) - 1)
        return {"result": step_answers[answer_idx], "cost_usd": 0.0, "source": "local"}

    return ollama_generate


class TestDecomposeTask:
    def test_returns_steps_from_json(self):
        steps_json = json.dumps([
            {"step": "Add type hints", "type": "refactor"},
            {"step": "Write tests", "type": "test"},
        ])
        with patch("agent.fabric_agent.ollama_generate",
                    return_value={"result": steps_json, "cost_usd": 0}):
            steps = decompose_task("refactor and test foo.py")

        assert len(steps) == 2
        assert steps[0]["step"] == "Add type hints"
        assert steps[1]["type"] == "test"

    def test_handles_code_block_wrapped_json(self):
        wrapped = '```json\n[{"step": "summarize", "type": "summarize"}]\n```'
        with patch("agent.fabric_agent.ollama_generate",
                    return_value={"result": wrapped, "cost_usd": 0}):
            steps = decompose_task("summarize everything")

        assert len(steps) == 1
        assert steps[0]["step"] == "summarize"

    def test_falls_back_to_single_step_on_bad_json(self):
        with patch("agent.fabric_agent.ollama_generate",
                    return_value={"result": "I can't parse that", "cost_usd": 0}):
            steps = decompose_task("do something complex")

        assert len(steps) == 1
        assert steps[0]["step"] == "do something complex"
        assert steps[0]["type"] == "other"

    def test_falls_back_on_wrong_structure(self):
        with patch("agent.fabric_agent.ollama_generate",
                    return_value={"result": '{"not": "a list"}', "cost_usd": 0}):
            steps = decompose_task("test task")

        assert len(steps) == 1

    def test_falls_back_on_missing_step_key(self):
        bad = json.dumps([{"description": "no step key", "type": "other"}])
        with patch("agent.fabric_agent.ollama_generate",
                    return_value={"result": bad, "cost_usd": 0}):
            steps = decompose_task("test task")

        assert len(steps) == 1


class TestRunPipeline:
    def setup_method(self):
        cfg = ConfidenceConfig.from_preset("max")
        set_config(cfg)

    def test_single_step_delegates_to_run_single(self):
        single = json.dumps([{"step": "just one thing", "type": "other"}])
        mock = _mock_ollama_with_decompose(single, ["done"])

        with patch("agent.fabric_agent.ollama_generate", mock):
            with patch("agent.fabric_agent.classify_task", return_value="local"):
                result = run_pipeline("just one thing")

        assert "steps" not in result or result.get("source") != "pipeline"

    def test_multi_step_runs_all_steps(self):
        steps = json.dumps([
            {"step": "Add type hints to foo", "type": "refactor"},
            {"step": "Write tests for foo", "type": "test"},
        ])
        mock = _mock_ollama_with_decompose(steps, ["typed code", "test code"])

        with patch("agent.fabric_agent.ollama_generate", mock):
            with patch("agent.fabric_agent.classify_task", return_value="local"):
                result = run_pipeline("refactor and test foo")

        assert result["source"] == "pipeline"
        assert result["total_steps"] == 2
        assert len(result["steps"]) == 2
        assert result["steps"][0]["step_type"] == "refactor"
        assert result["steps"][1]["step_type"] == "test"

    def test_pipeline_passes_previous_context(self):
        steps = json.dumps([
            {"step": "Summarize the code", "type": "summarize"},
            {"step": "Classify complexity", "type": "classify"},
        ])
        captured_prompts = []

        call_idx = [0]
        def mock(prompt, model=None):
            idx = call_idx[0]
            call_idx[0] += 1
            if idx == 0:
                return {"result": steps, "cost_usd": 0, "source": "local"}
            captured_prompts.append(prompt)
            return {"result": f"step {idx} result", "cost_usd": 0, "source": "local"}

        with patch("agent.fabric_agent.ollama_generate", mock):
            with patch("agent.fabric_agent.classify_task", return_value="local"):
                run_pipeline("summarize then classify")

        assert len(captured_prompts) == 2
        assert "Previous step result" in captured_prompts[1]

    def test_pipeline_aggregates_cost(self):
        steps = json.dumps([
            {"step": "step 1", "type": "refactor"},
            {"step": "step 2", "type": "test"},
        ])

        call_idx = [0]
        def mock(prompt, model=None):
            idx = call_idx[0]
            call_idx[0] += 1
            if idx == 0:
                return {"result": steps, "cost_usd": 0, "source": "local"}
            return {"result": "answer", "cost_usd": 0.0, "source": "local"}

        with patch("agent.fabric_agent.ollama_generate", mock):
            with patch("agent.fabric_agent.classify_task", return_value="local"):
                result = run_pipeline("multi-step task")

        assert result["cost_usd"] == 0.0
        assert result["local_steps"] == 2
        assert result["frontier_steps"] == 0

    def test_pipeline_counts_sources(self):
        steps = json.dumps([
            {"step": "easy step", "type": "summarize"},
            {"step": "hard step", "type": "debug"},
            {"step": "medium step", "type": "refactor"},
        ])

        call_idx = [0]
        def mock(prompt, model=None):
            idx = call_idx[0]
            call_idx[0] += 1
            if idx == 0:
                return {"result": steps, "cost_usd": 0, "source": "local"}
            sources = ["local", "frontier", "shepherded"]
            return {"result": "answer", "cost_usd": 0.05, "source": sources[(idx - 1) % 3]}

        # Skip confidence by using max trust
        with patch("agent.fabric_agent.ollama_generate", mock):
            with patch("agent.fabric_agent.classify_task", return_value="local"):
                result = run_pipeline("complex task")

        assert result["total_steps"] == 3
        assert result["source"] == "pipeline"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
