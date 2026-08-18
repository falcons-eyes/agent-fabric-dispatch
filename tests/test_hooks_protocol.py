"""Tests for hook scripts against the real Claude Code protocol."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)

# Standard policy for testing (matches the example policy)
TEST_POLICY = """\
rules:
  - match:
      path: ["secrets/**", "**/.env*"]
    route: LOCAL_ONLY
  - match:
      task: ["refactor_bulk", "summarize_files", "classify"]
    route: LOCAL
    fallback: FRONTIER
  - default: FRONTIER

workers:
  local:
    endpoint: "http://127.0.0.1:11434"
    models: ["qwen3-coder:30b"]
"""


@pytest.fixture
def policy_file(tmp_path):
    """Create a temporary policy file and return its path."""
    p = tmp_path / "policy.yaml"
    p.write_text(TEST_POLICY)
    return str(p)


def run_hook(script_name: str, stdin_data: dict, policy_path: str | None = None) -> tuple[int, str, str]:
    """Run a hook script with JSON input, return (exit_code, stdout, stderr)."""
    script_path = Path(PROJECT_ROOT) / "hooks" / script_name
    env = {
        "PATH": "/usr/bin:/usr/local/bin",
        "HOME": str(Path.home()),
        "PYTHONPATH": PROJECT_ROOT,
    }
    if policy_path:
        env["FABRIC_POLICY_PATH"] = policy_path
    result = subprocess.run(
        [sys.executable, str(script_path)],
        input=json.dumps(stdin_data),
        capture_output=True,
        text=True,
        timeout=10,
        env=env,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


class TestPreToolUseProtocol:
    """Validate PreToolUse hook against Claude Code hook protocol."""

    def test_frontier_task_exits_0_silently(self, policy_file):
        """FRONTIER decision: exit 0, no stdout."""
        code, stdout, stderr = run_hook("pre_tool_use.py", {
            "session_id": "test-session",
            "hook_event_name": "PreToolUse",
            "tool_name": "Read",
            "tool_input": {"file_path": "src/main.py"},
            "cwd": "/tmp",
        }, policy_path=policy_file)
        assert code == 0
        assert stdout == ""

    def test_local_only_path_exits_2(self, policy_file):
        """LOCAL_ONLY (secrets path): exit 2, stderr has guidance."""
        code, stdout, stderr = run_hook("pre_tool_use.py", {
            "session_id": "test-session",
            "hook_event_name": "PreToolUse",
            "tool_name": "Read",
            "tool_input": {"file_path": "secrets/api_key.txt"},
            "cwd": "/tmp",
        }, policy_path=policy_file)
        assert code == 2
        assert "FABRIC POLICY BLOCK" in stderr
        assert "LOCAL_ONLY" in stderr
        assert "fabric-worker" in stderr

    def test_local_task_exits_2(self, policy_file):
        """LOCAL (refactor task): exit 2, stderr has dispatch guidance."""
        code, stdout, stderr = run_hook("pre_tool_use.py", {
            "session_id": "test-session",
            "hook_event_name": "PreToolUse",
            "tool_name": "refactor_tool",
            "tool_input": {"code": "def foo(): pass"},
            "cwd": "/tmp",
        }, policy_path=policy_file)
        assert code == 2
        assert "FABRIC DISPATCH" in stderr
        assert "fabric-worker" in stderr

    def test_empty_input_exits_0(self):
        """Bad/empty input: exit 0 (fail open)."""
        script_path = Path(PROJECT_ROOT) / "hooks" / "pre_tool_use.py"
        result = subprocess.run(
            [sys.executable, str(script_path)],
            input="",
            capture_output=True,
            text=True,
            timeout=10,
            env={
                "PATH": "/usr/bin:/usr/local/bin",
                "HOME": str(Path.home()),
                "PYTHONPATH": PROJECT_ROOT,
            },
        )
        assert result.returncode == 0

    def test_unknown_tool_exits_0(self, policy_file):
        """Unknown tool (no policy match): exit 0, no interference."""
        code, stdout, stderr = run_hook("pre_tool_use.py", {
            "session_id": "test-session",
            "hook_event_name": "PreToolUse",
            "tool_name": "Glob",
            "tool_input": {"pattern": "*.py"},
            "cwd": "/tmp",
        }, policy_path=policy_file)
        assert code == 0
        assert stdout == ""

    def test_env_file_exits_2(self, policy_file):
        """LOCAL_ONLY (.env file): exit 2, blocked."""
        code, stdout, stderr = run_hook("pre_tool_use.py", {
            "session_id": "test-session",
            "hook_event_name": "PreToolUse",
            "tool_name": "Read",
            "tool_input": {"file_path": "project/.env.local"},
            "cwd": "/tmp",
        }, policy_path=policy_file)
        assert code == 2
        assert "FABRIC POLICY BLOCK" in stderr


class TestPostToolUseProtocol:
    """Validate PostToolUse hook parses real protocol fields."""

    def test_parses_real_fields(self):
        """PostToolUse processes real protocol fields without error."""
        code, stdout, stderr = run_hook("post_tool_use.py", {
            "session_id": "test-session",
            "hook_event_name": "PostToolUse",
            "tool_name": "Read",
            "tool_input": {"file_path": "src/main.py"},
            "tool_output": "file contents here...",
            "cwd": "/tmp",
        })
        assert code == 0

    def test_detects_fabric_worker_as_local(self):
        """MCP fabric-worker tool detected as local."""
        code, stdout, stderr = run_hook("post_tool_use.py", {
            "session_id": "test-session",
            "hook_event_name": "PostToolUse",
            "tool_name": "mcp__fabric-worker__refactor_bulk",
            "tool_input": {"code": "def foo(): pass"},
            "tool_output": "def bar(): pass",
            "cwd": "/tmp",
        })
        assert code == 0

    def test_handles_empty_input(self):
        """Empty input: exits 0 gracefully."""
        script_path = Path(PROJECT_ROOT) / "hooks" / "post_tool_use.py"
        result = subprocess.run(
            [sys.executable, str(script_path)],
            input="",
            capture_output=True,
            text=True,
            timeout=10,
            env={
                "PATH": "/usr/bin:/usr/local/bin",
                "HOME": str(Path.home()),
                "PYTHONPATH": PROJECT_ROOT,
            },
        )
        assert result.returncode == 0


class TestStopProtocol:
    """Validate Stop hook handles real protocol input."""

    def test_handles_real_stop_input(self):
        """Stop hook processes session_id without error."""
        code, stdout, stderr = run_hook("stop.py", {
            "session_id": "test-session",
            "hook_event_name": "Stop",
            "stop_hook_active": False,
            "assistant_message": "Task completed.",
        })
        assert code == 0
