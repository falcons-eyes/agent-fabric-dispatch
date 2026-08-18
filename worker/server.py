"""MCP server wrapping Ollama/vLLM via HTTP APIs.

Exposes tool set v1: refactor_bulk, summarize_files, classify.
Uses Ollama HTTP API (/api/generate) for inference.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from worker.tools import TOOLS, execute_tool

# Default model — can be overridden via policy config
DEFAULT_MODEL = "qwen3-coder:30b"

# Ollama API endpoint — overridable via OLLAMA_HOST env var
OLLAMA_API_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


def get_model_from_policy() -> str:
    """Read the preferred model from policy config."""
    policy_path = Path.home() / ".fabric" / "policy.yaml"
    if policy_path.exists():
        try:
            import yaml
            with open(policy_path) as f:
                policy = yaml.safe_load(f) or {}
            models = (
                policy.get("workers", {})
                .get("local", {})
                .get("models", [])
            )
            if models:
                return models[0]
        except Exception:
            pass
    return DEFAULT_MODEL


def ollama_generate(model: str, prompt: str) -> str:
    """Run inference via Ollama HTTP API (/api/generate, stream=false)."""
    url = f"{OLLAMA_API_URL}/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip()
    except urllib.error.URLError as e:
        return f"Error: Cannot connect to Ollama at {OLLAMA_API_URL}: {e}"
    except Exception as e:
        return f"Error: {e}"


def vllm_generate(model: str, prompt: str, endpoint: str = "http://127.0.0.1:8000") -> str:
    """Run inference via vLLM OpenAI-compatible API."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
    })

    req = urllib.request.Request(
        f"{endpoint}/v1/chat/completions",
        data=payload.encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: vLLM request failed: {e}"


def handle_mcp_request(request: dict) -> dict | None:
    """Handle an MCP protocol request. Returns result dict or None for notifications."""
    method = request.get("method", "")

    # Notifications (no id) — no response required
    if method.startswith("notifications/"):
        return None

    if method == "initialize":
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {
                "name": "fabric-worker",
                "version": "0.1.0",
            },
        }

    if method == "ping":
        return {}

    if method == "tools/list":
        return {"tools": [tool.to_mcp_schema() for tool in TOOLS.values()]}

    if method == "tools/call":
        tool_name = request.get("params", {}).get("name", "")
        tool_args = request.get("params", {}).get("arguments", {})

        if tool_name not in TOOLS:
            return {"_error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}}

        model = get_model_from_policy()
        result_text = execute_tool(tool_name, tool_args, model, ollama_generate)

        return {"content": [{"type": "text", "text": result_text}]}

    return {"_error": {"code": -32601, "message": f"Unknown method: {method}"}}


def run_mcp_server():
    """Run as an MCP stdio server (JSON-RPC 2.0 over stdin/stdout)."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        result = handle_mcp_request(request)

        # Notifications produce no response
        if result is None:
            continue

        response = {"jsonrpc": "2.0", "id": request.get("id")}

        # Check for error responses (marked with _error key)
        if "_error" in result:
            response["error"] = result["_error"]
        else:
            response["result"] = result

        print(json.dumps(response), flush=True)


def run_standalone():
    """Run as a standalone worker for testing."""
    model = get_model_from_policy()
    print(f"Fabric Worker (standalone mode)")
    print(f"Model: {model}")
    print(f"Tools: {', '.join(TOOLS.keys())}")
    print(f"Enter prompts (Ctrl+D to exit):\n")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        result = ollama_generate(model, line)
        print(result)
        print()


def main():
    parser = argparse.ArgumentParser(description="Fabric Worker — MCP server for local model inference")
    parser.add_argument("--mcp", action="store_true", help="Run as MCP stdio server")
    parser.add_argument("--model", type=str, help="Override model name")
    args = parser.parse_args()

    if args.model:
        global DEFAULT_MODEL
        DEFAULT_MODEL = args.model

    if args.mcp:
        run_mcp_server()
    else:
        run_standalone()


if __name__ == "__main__":
    main()
