"""Tests for the policy evaluation router engine."""

import pytest
from router.engine import evaluate_policy


def make_policy(rules, budgets=None, workers=None):
    return {
        "rules": rules,
        "budgets": budgets or {},
        "workers": workers or {"local": {"endpoint": "http://127.0.0.1:11434", "models": ["qwen3-coder:30b"]}},
    }


class TestPolicyEvaluation:
    def test_default_frontier(self):
        """No rules → FRONTIER."""
        policy = make_policy([{"default": "FRONTIER"}])
        result = evaluate_policy({"tool_name": "anything", "paths": [], "task_type": ""}, policy)
        assert result["route"] == "FRONTIER"

    def test_path_match_local_only(self):
        """Secrets path → LOCAL_ONLY."""
        policy = make_policy([
            {"match": {"path": ["secrets/**"]}, "route": "LOCAL_ONLY"},
            {"default": "FRONTIER"},
        ])
        result = evaluate_policy(
            {"tool_name": "read", "paths": ["secrets/api_key.txt"], "task_type": ""},
            policy,
        )
        assert result["route"] == "LOCAL_ONLY"

    def test_env_file_match(self):
        """Env files → LOCAL_ONLY."""
        policy = make_policy([
            {"match": {"path": ["**/.env*"]}, "route": "LOCAL_ONLY"},
            {"default": "FRONTIER"},
        ])
        result = evaluate_policy(
            {"tool_name": "read", "paths": ["project/.env.local"], "task_type": ""},
            policy,
        )
        assert result["route"] == "LOCAL_ONLY"

    def test_task_match_local(self):
        """Refactor task → LOCAL with FRONTIER fallback."""
        policy = make_policy([
            {"match": {"task": ["refactor_bulk", "summarize_files"]}, "route": "LOCAL", "fallback": "FRONTIER"},
            {"default": "FRONTIER"},
        ])
        result = evaluate_policy(
            {"tool_name": "tool", "paths": [], "task_type": "refactor_bulk"},
            policy,
        )
        assert result["route"] == "LOCAL"
        assert result["fallback"] == "FRONTIER"

    def test_no_match_falls_to_default(self):
        """Unknown task type → FRONTIER (default)."""
        policy = make_policy([
            {"match": {"task": ["refactor_bulk"]}, "route": "LOCAL"},
            {"default": "FRONTIER"},
        ])
        result = evaluate_policy(
            {"tool_name": "tool", "paths": [], "task_type": "unknown_task"},
            policy,
        )
        assert result["route"] == "FRONTIER"

    def test_first_match_wins(self):
        """First matching rule takes precedence."""
        policy = make_policy([
            {"match": {"path": ["secrets/**"]}, "route": "LOCAL_ONLY"},
            {"match": {"task": ["refactor_bulk"]}, "route": "LOCAL"},
            {"default": "FRONTIER"},
        ])
        # Path in secrets/ AND task is refactor_bulk → LOCAL_ONLY wins (first match)
        result = evaluate_policy(
            {"tool_name": "tool", "paths": ["secrets/config.py"], "task_type": "refactor_bulk"},
            policy,
        )
        assert result["route"] == "LOCAL_ONLY"

    def test_repo_tag_match(self):
        """Corp repo tag → LOCAL_ONLY."""
        policy = make_policy([
            {"match": {"repo_tag": "corp"}, "route": "LOCAL_ONLY"},
            {"default": "FRONTIER"},
        ])
        result = evaluate_policy(
            {"tool_name": "tool", "paths": [], "task_type": "", "repo_tag": "corp"},
            policy,
        )
        assert result["route"] == "LOCAL_ONLY"

    def test_empty_context(self):
        """Empty context → falls through to default."""
        policy = make_policy([
            {"match": {"task": ["refactor_bulk"]}, "route": "LOCAL"},
            {"default": "FRONTIER"},
        ])
        result = evaluate_policy({}, policy)
        assert result["route"] == "FRONTIER"

    def test_empty_policy(self):
        """No policy file → FRONTIER."""
        result = evaluate_policy(
            {"tool_name": "anything", "paths": [], "task_type": ""},
            {"rules": [{"default": "FRONTIER"}]},
        )
        assert result["route"] == "FRONTIER"


class TestToolClassification:
    def test_classify_task_local(self):
        """Classify task → LOCAL."""
        policy = make_policy([
            {"match": {"task": ["classify"]}, "route": "LOCAL"},
            {"default": "FRONTIER"},
        ])
        result = evaluate_policy(
            {"tool_name": "classify", "paths": [], "task_type": "classify"},
            policy,
        )
        assert result["route"] == "LOCAL"

    def test_summarize_task_local(self):
        """Summarize task → LOCAL."""
        policy = make_policy([
            {"match": {"task": ["summarize_files"]}, "route": "LOCAL"},
            {"default": "FRONTIER"},
        ])
        result = evaluate_policy(
            {"tool_name": "summarize", "paths": [], "task_type": "summarize_files"},
            policy,
        )
        assert result["route"] == "LOCAL"
