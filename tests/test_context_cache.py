#!/usr/bin/env python3
"""Tests for context cache (KV cache optimization)."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.context_cache import ContextCache


class TestContextCache:
    def _make_temp_file(self, content="hello world"):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        f.write(content)
        f.close()
        return f.name

    def test_load_file(self):
        path = self._make_temp_file("def foo(): pass")
        cache = ContextCache()
        entry = cache.load_file(path)

        assert entry is not None
        assert entry.content == "def foo(): pass"
        assert entry.size == 15
        assert entry.hash

    def test_load_file_not_found(self):
        cache = ContextCache()
        entry = cache.load_file("/nonexistent/file.py")
        assert entry is None

    def test_cache_deduplication(self):
        path = self._make_temp_file("content")
        cache = ContextCache()
        e1 = cache.load_file(path)
        e2 = cache.load_file(path)

        assert e1.hash == e2.hash
        assert len(cache.files) == 1

    def test_invalidate(self):
        path = self._make_temp_file("content")
        cache = ContextCache()
        cache.load_file(path)
        assert path in cache.files

        cache.invalidate(path)
        assert path not in cache.files

    def test_clear(self):
        cache = ContextCache()
        cache.load_file(self._make_temp_file("a"))
        cache.load_file(self._make_temp_file("b"))
        assert len(cache.files) == 2

        cache.clear()
        assert len(cache.files) == 0

    def test_total_size(self):
        cache = ContextCache()
        cache.load_file(self._make_temp_file("12345"))
        cache.load_file(self._make_temp_file("67890"))
        assert cache.total_size() == 10

    def test_total_size_with_system_context(self):
        cache = ContextCache()
        cache.set_system_context("system prompt")
        cache.load_file(self._make_temp_file("code"))
        assert cache.total_size() == len("system prompt") + 4


class TestBuildPrompt:
    def _cache_with_files(self):
        cache = ContextCache()
        f1 = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        f1.write("def foo(): pass")
        f1.close()
        f2 = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        f2.write("def bar(): pass")
        f2.close()
        cache.load_file(f1.name)
        cache.load_file(f2.name)
        return cache, f1.name, f2.name

    def test_includes_all_files(self):
        cache, f1, f2 = self._cache_with_files()
        prompt = cache.build_prompt("summarize these")

        assert "def foo(): pass" in prompt
        assert "def bar(): pass" in prompt
        assert "summarize these" in prompt

    def test_includes_specific_files(self):
        cache, f1, f2 = self._cache_with_files()
        prompt = cache.build_prompt("summarize", files=[f1])

        assert "def foo(): pass" in prompt
        assert "def bar(): pass" not in prompt

    def test_includes_system_context(self):
        cache = ContextCache()
        cache.set_system_context("You are a coding assistant.")
        prompt = cache.build_prompt("help me")

        assert "You are a coding assistant." in prompt
        assert "help me" in prompt

    def test_instruction_at_end(self):
        cache, f1, f2 = self._cache_with_files()
        prompt = cache.build_prompt("do the thing")
        lines = prompt.strip().split("\n")
        assert lines[-1] == "do the thing"

    def test_deterministic_file_order(self):
        cache, f1, f2 = self._cache_with_files()
        p1 = cache.build_prompt("task A")
        p2 = cache.build_prompt("task B")

        prefix1 = p1[:p1.rfind("task A")]
        prefix2 = p2[:p2.rfind("task B")]
        assert prefix1 == prefix2

    def test_truncates_large_files(self):
        cache = ContextCache(max_context_chars=100)
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        f.write("x" * 5000)
        f.close()
        cache.load_file(f.name)

        prompt = cache.build_prompt("summarize")
        assert "truncated" in prompt
        assert len(prompt) < 5000


class TestBatchPrompts:
    def test_shared_prefix(self):
        cache = ContextCache()
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        f.write("shared content")
        f.close()
        cache.load_file(f.name)

        prompts = cache.build_prompts_batch(["task 1", "task 2", "task 3"])

        assert len(prompts) == 3
        prefix_len = len("shared content") + 20  # approximate prefix
        # All prompts should share the same prefix
        p1_prefix = prompts[0][:prompts[0].rfind("task 1")]
        p2_prefix = prompts[1][:prompts[1].rfind("task 2")]
        assert p1_prefix == p2_prefix


class TestStats:
    def test_stats(self):
        cache = ContextCache()
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        f.write("code")
        f.close()
        cache.load_file(f.name)

        stats = cache.stats()
        assert stats["n_files"] == 1
        assert stats["total_chars"] == 4
        assert f.name in stats["files"]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
