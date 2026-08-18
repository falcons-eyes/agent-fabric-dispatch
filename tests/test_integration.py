#!/usr/bin/env python3
"""Integration tests against real Ollama and Claude CLI.

These tests require:
  - Ollama running at http://127.0.0.1:11434
  - qwen3-coder:30b model pulled
  - claude CLI available in PATH

Run: pytest tests/test_integration.py -v -m integration
Skip: pytest tests/ -v -m "not integration"
"""

import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


def ollama_available() -> bool:
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def claude_available() -> bool:
    try:
        result = subprocess.run(["claude", "--version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def model_available(model: str) -> bool:
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            return any(m["name"] == model for m in data.get("models", []))
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not ollama_available(), reason="Ollama not running")
requires_claude = pytest.mark.skipif(
    not claude_available(), reason="Claude CLI not available")
requires_model = pytest.mark.skipif(
    not model_available("qwen3-coder:30b"), reason="qwen3-coder:30b not pulled")

pytestmark = pytest.mark.integration


@requires_ollama
@requires_model
class TestOllamaAPI:
    def test_ollama_generate_returns_result(self):
        from agent.agent_fabric import ollama_generate, set_model
        set_model("qwen3-coder:30b")
        result = ollama_generate("What is 2+2? Answer with just the number.")
        assert result["result"].strip() == "4"
        assert result["source"] == "local"
        assert result["cost_usd"] == 0.0
        assert result["eval_count"] > 0

    def test_ollama_generate_with_temp(self):
        from agent.agent_fabric import ollama_generate_with_temp, set_model
        set_model("qwen3-coder:30b")
        result = ollama_generate_with_temp(
            "What is 3+3? Answer with just the number.", temperature=0.7)
        assert "6" in result.get("result", "")

    def test_voting_generate_with_model(self):
        from agent.voting import generate_with_model
        result = generate_with_model(
            "What is 5+5? Answer with just the number.", "qwen3-coder:30b")
        assert "10" in result["result"]
        assert result["source"] == "local"


@requires_claude
class TestFrontierQuery:
    def test_frontier_returns_result(self):
        from agent.agent_fabric import frontier_query
        result = frontier_query("What is 2+2? Answer with just the number.")
        assert "4" in result["result"]
        assert result["source"] == "frontier"
        assert result["cost_usd"] > 0
        assert result["output_tokens"] > 0


@requires_ollama
@requires_model
class TestRunSingle:
    def setup_method(self):
        from agent.agent_fabric import set_model
        from agent.confidence import ConfidenceConfig, set_config
        from agent.budget import BudgetConfig, BudgetTracker, set_tracker
        set_model("qwen3-coder:30b")
        set_config(ConfidenceConfig.from_preset("max"))
        set_tracker(BudgetTracker(config=BudgetConfig()))

    def teardown_method(self):
        from agent.confidence import ConfidenceConfig, set_config
        from agent.budget import BudgetConfig, BudgetTracker, set_tracker
        set_config(ConfidenceConfig())
        set_tracker(BudgetTracker(config=BudgetConfig()))

    def test_simple_math_local(self):
        from agent.agent_fabric import run_single
        r = run_single("What is 15 * 17? Answer with just the number.")
        assert r["result"].strip() == "255"
        assert r["source"] == "local"
        assert r["cost_usd"] == 0.0

    def test_file_reference_expansion(self):
        from agent.agent_fabric import run_single
        from agent.context_cache import set_cache, ContextCache
        set_cache(ContextCache())
        r = run_single("How many rules does @examples/policy.example.yaml define?")
        assert r["source"] == "local"
        assert r["cost_usd"] == 0.0

    def test_classify_routes_locally(self):
        from agent.agent_fabric import classify_task
        assert classify_task("summarize this code") == "local"
        assert classify_task("refactor the function") == "local"
        assert classify_task("classify these items") == "local"

    def test_classify_does_not_use_expanded(self):
        from agent.agent_fabric import classify_task
        prompt = "List functions in this file"
        assert classify_task(prompt) == "local"


@requires_ollama
@requires_model
class TestConfidenceEstimation:
    def test_high_confidence_on_math(self):
        from agent.agent_fabric import ollama_generate, ollama_generate_with_temp, set_model
        from agent.confidence import ConfidenceConfig, set_config, estimate_confidence
        set_model("qwen3-coder:30b")
        set_config(ConfidenceConfig.from_preset("balanced"))

        result = ollama_generate("What is 7*8? Answer with just the number.")
        conf = estimate_confidence(
            "What is 7*8? Answer with just the number.",
            result, ollama_generate_with_temp)
        assert conf["confidence"].value == "high"
        assert conf["agreement"] >= 0.8


@requires_ollama
@requires_model
class TestMultiModelVoting:
    def test_voting_agreement(self):
        from agent.voting import multi_model_vote
        available = []
        for m in ["qwen3-coder:30b", "qwen3:30b"]:
            if model_available(m):
                available.append(m)
        if len(available) < 2:
            pytest.skip("Need at least 2 models for voting")

        result = multi_model_vote(
            "What is 2+2? Answer with just the number.",
            models=available, min_agreement=0.5)
        assert "4" in result["result"]
        assert result["agreement"] >= 0.5


@requires_ollama
@requires_model
class TestFanOut:
    def setup_method(self):
        from agent.agent_fabric import set_model
        from agent.confidence import ConfidenceConfig, set_config
        from agent.budget import BudgetConfig, BudgetTracker, set_tracker
        set_model("qwen3-coder:30b")
        set_config(ConfidenceConfig.from_preset("max"))
        set_tracker(BudgetTracker(config=BudgetConfig()))

    def teardown_method(self):
        from agent.confidence import ConfidenceConfig, set_config
        from agent.budget import BudgetConfig, BudgetTracker, set_tracker
        set_config(ConfidenceConfig())
        set_tracker(BudgetTracker(config=BudgetConfig()))

    def test_parallel_execution(self):
        from agent.fanout import fan_out
        result = fan_out([
            "What is 2+2? Just the number.",
            "What is 3+3? Just the number.",
        ], max_workers=2)
        assert result.n_tasks == 2
        assert result.n_succeeded == 2
        assert result.total_cost == 0.0
        assert result.to_dict()["speedup"] >= 1.0


@requires_ollama
@requires_model
class TestContextCache:
    def test_kv_cache_speedup(self):
        from agent.agent_fabric import ollama_generate, set_model
        from agent.context_cache import ContextCache
        import time

        set_model("qwen3-coder:30b")
        cache = ContextCache()
        cache.load_file("router/engine.py")

        p1 = cache.build_prompt("Count the functions.")
        t0 = time.time()
        ollama_generate(p1)
        cold_ms = (time.time() - t0) * 1000

        p2 = cache.build_prompt("List the imports.")
        t0 = time.time()
        ollama_generate(p2)
        warm_ms = (time.time() - t0) * 1000

        assert warm_ms < cold_ms * 1.5


class TestMCPServer:
    def test_stdio_roundtrip(self):
        init_req = json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                       "clientInfo": {"name": "test", "version": "1.0"}}
        })
        ping_req = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"})
        tools_req = json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})

        stdin_data = ""
        for msg in [init_req, ping_req, tools_req]:
            stdin_data += f"Content-Length: {len(msg)}\r\n\r\n{msg}"

        proc = subprocess.run(
            [sys.executable, "worker/server.py", "--mcp"],
            input=stdin_data, capture_output=True, text=True, timeout=10)

        responses = []
        pos = 0
        stdout = proc.stdout
        while pos < len(stdout):
            match = re.match(r"Content-Length:\s*(\d+)\r?\n\r?\n", stdout[pos:])
            if match:
                length = int(match.group(1))
                body_start = pos + match.end()
                body = stdout[body_start:body_start + length]
                responses.append(json.loads(body))
                pos = body_start + length
            else:
                pos += 1

        assert len(responses) == 3
        assert responses[0]["result"]["protocolVersion"] == "2024-11-05"
        assert responses[1]["result"] == {}
        assert "tools" in responses[2]["result"]


class TestHooks:
    def test_pre_tool_use_frontier_passes(self):
        hook_input = json.dumps({
            "session_id": "test", "hook_event_name": "PreToolUse",
            "tool_name": "Read", "tool_input": {"file_path": "README.md"},
            "cwd": "/tmp"})
        proc = subprocess.run(
            [sys.executable, "hooks/pre_tool_use.py"],
            input=hook_input, capture_output=True, text=True, timeout=5,
            env={**os.environ, "FABRIC_POLICY_PATH": "/dev/null"})
        assert proc.returncode == 0

    def test_pre_tool_use_local_only_blocks(self):
        hook_input = json.dumps({
            "session_id": "test", "hook_event_name": "PreToolUse",
            "tool_name": "Read",
            "tool_input": {"file_path": "secrets/api_key.txt"},
            "cwd": str(Path(__file__).resolve().parent.parent)})
        proc = subprocess.run(
            [sys.executable, "hooks/pre_tool_use.py"],
            input=hook_input, capture_output=True, text=True, timeout=5)
        assert proc.returncode == 2
        assert "LOCAL_ONLY" in proc.stderr

    def test_pre_tool_use_fabric_mcp_passes(self):
        hook_input = json.dumps({
            "session_id": "test", "hook_event_name": "PreToolUse",
            "tool_name": "mcp__fabric_refactor_bulk",
            "tool_input": {"code": "pass"}, "cwd": "/tmp"})
        proc = subprocess.run(
            [sys.executable, "hooks/pre_tool_use.py"],
            input=hook_input, capture_output=True, text=True, timeout=5)
        assert proc.returncode == 0

    def test_post_tool_use_records(self):
        hook_input = json.dumps({
            "session_id": "test", "hook_event_name": "PostToolUse",
            "tool_name": "Read",
            "tool_input": {"file_path": "README.md"},
            "tool_output": "file content", "cwd": "/tmp"})
        proc = subprocess.run(
            [sys.executable, "hooks/post_tool_use.py"],
            input=hook_input, capture_output=True, text=True, timeout=5)
        assert proc.returncode == 0

    def test_stop_hook_prints_summary(self):
        hook_input = json.dumps({
            "session_id": "test", "hook_event_name": "Stop", "cwd": "/tmp"})
        proc = subprocess.run(
            [sys.executable, "hooks/stop.py"],
            input=hook_input, capture_output=True, text=True, timeout=5)
        assert proc.returncode == 0
        assert "session" in proc.stdout.lower() or "dispatch" in proc.stdout.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
