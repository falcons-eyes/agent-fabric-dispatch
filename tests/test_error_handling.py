"""Test graceful error handling when Ollama is unreachable."""

import os
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestOllamaDown(unittest.TestCase):
    """Verify graceful behavior when Ollama server is unreachable."""

    def test_ollama_generate_returns_error_on_connection_refused(self):
        from agent.agent_fabric import ollama_generate
        with patch.dict(os.environ, {"OLLAMA_HOST": "http://127.0.0.1:1"}):
            import agent.agent_fabric as af
            old_url = af.OLLAMA_API_URL
            af.OLLAMA_API_URL = "http://127.0.0.1:1"
            try:
                result = af.ollama_generate("test prompt")
                self.assertIn("error", result)
                self.assertEqual(result["cost_usd"], 0.0)
                self.assertEqual(result["source"], "local")
                self.assertEqual(result["result"], "")
            finally:
                af.OLLAMA_API_URL = old_url

    def test_ollama_chat_returns_error_on_connection_refused(self):
        from agent import agent_fabric as af
        old_url = af.OLLAMA_API_URL
        af.OLLAMA_API_URL = "http://127.0.0.1:1"
        try:
            result = af.ollama_chat([{"role": "user", "content": "test"}])
            self.assertIn("error", result)
            self.assertEqual(result["cost_usd"], 0.0)
        finally:
            af.OLLAMA_API_URL = old_url

    def test_run_single_falls_back_to_frontier_on_local_error(self):
        from agent import agent_fabric as af

        original_ollama = af.ollama_generate
        original_frontier = af.frontier_query

        af.ollama_generate = lambda prompt, model=None: {
            "result": "", "cost_usd": 0.0, "source": "local", "error": "connection refused"
        }
        af.frontier_query = lambda prompt, max_turns=1: {
            "result": "frontier answer", "cost_usd": 0.1, "source": "frontier"
        }

        try:
            result = af.run_single("summarize this code")
            self.assertEqual(result["source"], "frontier")
            self.assertEqual(result["result"], "frontier answer")
            self.assertTrue(result.get("fallback"))
        finally:
            af.ollama_generate = original_ollama
            af.frontier_query = original_frontier

    def test_run_single_forced_local_does_not_fallback(self):
        from agent import agent_fabric as af

        original_ollama = af.ollama_generate
        af.ollama_generate = lambda prompt, model=None: {
            "result": "", "cost_usd": 0.0, "source": "local", "error": "connection refused"
        }

        try:
            result = af.run_single("summarize this code", force_route="local")
            self.assertEqual(result["source"], "local")
            self.assertIn("error", result)
        finally:
            af.ollama_generate = original_ollama

    def test_classify_task_returns_valid_route(self):
        from agent.agent_fabric import classify_task
        self.assertEqual(classify_task("summarize this code"), "local")
        self.assertEqual(classify_task("debug this function"), "frontier")
        self.assertIn(classify_task("hello world"), ["local", "frontier"])


if __name__ == "__main__":
    unittest.main()
