"""Tests for MCP server protocol compliance."""

import io
import json
import pytest
from worker.server import handle_mcp_request, _read_mcp_message, _write_mcp_message


class TestMCPInitialize:
    def test_initialize_returns_result(self):
        """initialize returns protocol version and capabilities."""
        result = handle_mcp_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
        })
        assert result is not None
        assert "protocolVersion" in result
        assert result["protocolVersion"] == "2024-11-05"
        assert "capabilities" in result
        assert "serverInfo" in result
        assert result["serverInfo"]["name"] == "fabric-worker"

    def test_notifications_initialized_returns_none(self):
        """notifications/initialized is a notification — no response."""
        result = handle_mcp_request({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        })
        assert result is None

    def test_other_notifications_return_none(self):
        """Any notification (method starting with notifications/) returns None."""
        result = handle_mcp_request({
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
        })
        assert result is None


class TestMCPPing:
    def test_ping_returns_empty_result(self):
        """ping returns empty dict."""
        result = handle_mcp_request({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "ping",
        })
        assert result == {}


class TestMCPToolsList:
    def test_tools_list_returns_tools(self):
        """tools/list returns all three tools."""
        result = handle_mcp_request({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/list",
        })
        assert "tools" in result
        tool_names = [t["name"] for t in result["tools"]]
        assert "refactor_bulk" in tool_names
        assert "summarize_files" in tool_names
        assert "classify" in tool_names

    def test_tools_have_input_schema(self):
        """Each tool has name, description, and inputSchema."""
        result = handle_mcp_request({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/list",
        })
        for tool in result["tools"]:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool


class TestMCPToolsCall:
    def test_call_with_mock_returns_content(self):
        """tools/call returns result with content array."""
        # Patch ollama_generate for testing
        import worker.server as ws
        original = ws.ollama_generate

        def mock_generate(model, prompt):
            return "mocked output"

        ws.ollama_generate = mock_generate
        try:
            result = handle_mcp_request({
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {
                    "name": "refactor_bulk",
                    "arguments": {
                        "code": "def foo(): pass",
                        "instructions": "rename to bar",
                        "language": "python",
                    },
                },
            })
            assert "content" in result
            assert len(result["content"]) == 1
            assert result["content"][0]["type"] == "text"
            assert result["content"][0]["text"] == "mocked output"
        finally:
            ws.ollama_generate = original

    def test_call_unknown_tool_returns_error(self):
        """tools/call with unknown tool returns error."""
        result = handle_mcp_request({
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "nonexistent_tool",
                "arguments": {},
            },
        })
        assert "_error" in result
        assert result["_error"]["code"] == -32601

    def test_unknown_method_returns_error(self):
        """Unknown method returns error with code -32601."""
        result = handle_mcp_request({
            "jsonrpc": "2.0",
            "id": 7,
            "method": "unknown/method",
        })
        assert "_error" in result
        assert result["_error"]["code"] == -32601


class TestMCPResponseWrapping:
    """Test that run_mcp_server would wrap responses correctly."""

    def test_result_wrapping_for_initialize(self):
        """Result from handle_mcp_request should be wrapped in 'result' key."""
        result = handle_mcp_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
        })
        # result should NOT have _error
        assert "_error" not in result
        # This would go under response["result"] in run_mcp_server
        assert "protocolVersion" in result

    def test_error_wrapping_for_unknown_method(self):
        """Error from handle_mcp_request has _error key."""
        result = handle_mcp_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "bad/method",
        })
        assert "_error" in result
        # This would go under response["error"] in run_mcp_server


class TestMCPTransportFraming:
    """Test Content-Length header framing for stdio transport."""

    def test_read_message(self):
        body = '{"jsonrpc":"2.0","id":1,"method":"ping"}'
        raw = f"Content-Length: {len(body)}\r\n\r\n{body}"
        stream = io.StringIO(raw)
        msg = _read_mcp_message(stream)
        assert msg == body

    def test_read_message_eof(self):
        stream = io.StringIO("")
        msg = _read_mcp_message(stream)
        assert msg is None

    def test_read_message_no_content_length(self):
        stream = io.StringIO("\r\n")
        msg = _read_mcp_message(stream)
        assert msg is None

    def test_read_multiple_headers(self):
        body = '{"test":true}'
        raw = f"Content-Length: {len(body)}\r\nContent-Type: application/json\r\n\r\n{body}"
        stream = io.StringIO(raw)
        msg = _read_mcp_message(stream)
        assert msg == body

    def test_roundtrip(self):
        import sys
        body = {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}}
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            _write_mcp_message(body)
        finally:
            sys.stdout = old_stdout
        buf.seek(0)
        msg = _read_mcp_message(buf)
        assert msg is not None
        assert json.loads(msg) == body
