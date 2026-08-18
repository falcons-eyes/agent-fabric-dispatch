"""Tests for the worker tool definitions."""

import pytest
from worker.tools import TOOLS, execute_tool


class TestToolDefinitions:
    def test_all_tools_exist(self):
        assert "refactor_bulk" in TOOLS
        assert "summarize_files" in TOOLS
        assert "classify" in TOOLS

    def test_refactor_bulk_schema(self):
        tool = TOOLS["refactor_bulk"]
        schema = tool.to_mcp_schema()
        assert schema["name"] == "refactor_bulk"
        assert "code" in schema["inputSchema"]["properties"]
        assert "instructions" in schema["inputSchema"]["properties"]
        assert "language" in schema["inputSchema"]["properties"]

    def test_summarize_files_schema(self):
        tool = TOOLS["summarize_files"]
        schema = tool.to_mcp_schema()
        assert schema["name"] == "summarize_files"
        assert "code" in schema["inputSchema"]["properties"]
        assert "file_path" in schema["inputSchema"]["properties"]

    def test_classify_schema(self):
        tool = TOOLS["classify"]
        schema = tool.to_mcp_schema()
        assert schema["name"] == "classify"
        assert "content" in schema["inputSchema"]["properties"]
        assert "categories" in schema["inputSchema"]["properties"]


class TestToolExecution:
    def test_refactor_bulk_prompt(self):
        tool = TOOLS["refactor_bulk"]
        prompt = tool.build_prompt({
            "code": "def foo(): pass",
            "instructions": "rename to bar",
            "language": "python",
        })
        assert "rename to bar" in prompt
        assert "def foo(): pass" in prompt
        assert "python" in prompt

    def test_execute_with_mock_generate(self):
        def mock_generate(model, prompt):
            return "def bar(): pass"

        result = execute_tool(
            "refactor_bulk",
            {"code": "def foo(): pass", "instructions": "rename to bar", "language": "python"},
            "test-model",
            mock_generate,
        )
        assert result == "def bar(): pass"

    def test_execute_unknown_tool(self):
        def mock_generate(model, prompt):
            return ""

        result = execute_tool("nonexistent", {}, "test-model", mock_generate)
        assert "Error" in result

    def test_execute_missing_args(self):
        def mock_generate(model, prompt):
            return ""

        result = execute_tool("refactor_bulk", {"code": "x"}, "test-model", mock_generate)
        assert "Error" in result
        assert "missing" in result.lower()
