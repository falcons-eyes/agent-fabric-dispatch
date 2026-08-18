#!/usr/bin/env python3
"""Tests for session fan-out (parallel task execution)."""

import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.budget import BudgetConfig, BudgetTracker, set_tracker
from agent.confidence import ConfidenceConfig, set_config
from agent.fanout import FanOutResult, fan_out, fan_out_batch, fan_out_models


class TestFanOutResult:
    def test_to_dict(self):
        r = FanOutResult(n_tasks=2, n_succeeded=2, wall_clock_ms=100,
                         total_duration_ms=200)
        d = r.to_dict()
        assert d["n_tasks"] == 2
        assert d["speedup"] == 2.0

    def test_speedup_with_zero_wall(self):
        r = FanOutResult(wall_clock_ms=0, total_duration_ms=100)
        d = r.to_dict()
        assert d["speedup"] == 1.0


class TestFanOut:
    def setup_method(self):
        set_config(ConfidenceConfig.from_preset("max"))
        set_tracker(BudgetTracker(config=BudgetConfig()))

    def teardown_method(self):
        set_config(ConfidenceConfig())
        set_tracker(BudgetTracker(config=BudgetConfig()))

    def test_parallel_execution(self):
        call_count = [0]

        def mock_ollama(prompt, model=None):
            call_count[0] += 1
            return {"result": f"answer for: {prompt[:20]}",
                    "cost_usd": 0, "source": "local", "duration_ms": 10}

        with patch("agent.fabric_agent.ollama_generate", mock_ollama):
            with patch("agent.fabric_agent.classify_task", return_value="local"):
                result = fan_out(["task 1", "task 2", "task 3"])

        assert result.n_tasks == 3
        assert result.n_succeeded == 3
        assert result.n_failed == 0
        assert len(result.results) == 3

    def test_results_preserve_order(self):
        def mock_ollama(prompt, model=None):
            return {"result": prompt, "cost_usd": 0, "source": "local"}

        with patch("agent.fabric_agent.ollama_generate", mock_ollama):
            with patch("agent.fabric_agent.classify_task", return_value="local"):
                result = fan_out(["first", "second", "third"])

        indices = [r["task_index"] for r in result.results]
        assert indices == [0, 1, 2]

    def test_handles_errors(self):
        call_count = [0]

        def mock_ollama(prompt, model=None):
            call_count[0] += 1
            if call_count[0] == 2:
                return {"result": "", "error": "timeout", "cost_usd": 0, "source": "local"}
            return {"result": "ok", "cost_usd": 0, "source": "local"}

        with patch("agent.fabric_agent.ollama_generate", mock_ollama):
            with patch("agent.fabric_agent.classify_task", return_value="local"):
                result = fan_out(["a", "b", "c"], force_route="local")

        assert result.n_tasks == 3

    def test_aggregates_cost(self):
        def mock_ollama(prompt, model=None):
            return {"result": "ok", "cost_usd": 0.0, "source": "local",
                    "duration_ms": 50}

        with patch("agent.fabric_agent.ollama_generate", mock_ollama):
            with patch("agent.fabric_agent.classify_task", return_value="local"):
                result = fan_out(["a", "b"])

        assert result.total_cost == 0.0
        assert result.total_duration_ms == 100

    def test_source_tracking(self):
        def mock_ollama(prompt, model=None):
            return {"result": "ok", "cost_usd": 0.0, "source": "local"}

        with patch("agent.fabric_agent.ollama_generate", mock_ollama):
            with patch("agent.fabric_agent.classify_task", return_value="local"):
                result = fan_out(["a", "b", "c"])

        assert result.sources.get("local", 0) == 3


class TestFanOutModels:
    def test_runs_on_multiple_models(self):
        captured = []

        def mock_ollama(prompt, model=None):
            captured.append(model)
            return {"result": f"from {model}", "model": model,
                    "cost_usd": 0, "source": "local", "duration_ms": 10}

        with patch("agent.fabric_agent.ollama_generate", mock_ollama):
            result = fan_out_models("test prompt",
                                    ["model-a", "model-b", "model-c"])

        assert result.n_tasks == 3
        assert result.n_succeeded == 3
        models_used = set(captured)
        assert models_used == {"model-a", "model-b", "model-c"}


class TestFanOutBatch:
    def setup_method(self):
        set_config(ConfidenceConfig.from_preset("max"))
        set_tracker(BudgetTracker(config=BudgetConfig()))

    def teardown_method(self):
        set_config(ConfidenceConfig())
        set_tracker(BudgetTracker(config=BudgetConfig()))

    def test_processes_in_batches(self):
        call_count = [0]

        def mock_ollama(prompt, model=None):
            call_count[0] += 1
            return {"result": f"answer {call_count[0]}",
                    "cost_usd": 0, "source": "local", "duration_ms": 5}

        with patch("agent.fabric_agent.ollama_generate", mock_ollama):
            with patch("agent.fabric_agent.classify_task", return_value="local"):
                result = fan_out_batch(
                    ["a", "b", "c", "d", "e"], batch_size=2)

        assert result.n_tasks == 5
        assert result.n_succeeded == 5
        assert len(result.results) == 5

    def test_preserves_global_indices(self):
        def mock_ollama(prompt, model=None):
            return {"result": "ok", "cost_usd": 0, "source": "local"}

        with patch("agent.fabric_agent.ollama_generate", mock_ollama):
            with patch("agent.fabric_agent.classify_task", return_value="local"):
                result = fan_out_batch(["a", "b", "c", "d"], batch_size=2)

        indices = sorted(r["task_index"] for r in result.results)
        assert indices == [0, 1, 2, 3]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
