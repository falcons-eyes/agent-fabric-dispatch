#!/usr/bin/env python3
"""Context cache for KV cache optimization.

Ollama automatically caches KV state for matching prompt prefixes.
This module structures prompts to maximize cache hits:

1. Shared prefix: file contents and system context go first (cached)
2. Task-specific suffix: the actual instruction goes last (not cached)

When the same file appears in multiple prompts within a session, the
KV cache for that file's content is reused — reducing TTFT from seconds
to milliseconds for the repeated prefix.

Usage:
    cache = ContextCache()
    cache.load_file("router/engine.py")
    cache.load_file("worker/server.py")

    # Both prompts share the same prefix (file contents) → KV cache hit
    prompt1 = cache.build_prompt("summarize the router")
    prompt2 = cache.build_prompt("find bugs in the worker")
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CachedFile:
    path: str
    content: str
    hash: str
    size: int


@dataclass
class ContextCache:
    """Session-level context cache for prompt prefix sharing."""
    files: dict[str, CachedFile] = field(default_factory=dict)
    system_context: str = ""
    max_context_chars: int = 100_000

    def load_file(self, path: str, base_dir: str = None) -> CachedFile | None:
        """Load a file into the cache. Returns cached entry or None on error."""
        p = Path(path)
        if base_dir and not p.is_absolute():
            p = Path(base_dir) / p

        try:
            p = p.resolve()
        except OSError:
            return None

        if not p.exists():
            return None

        key = str(p)

        try:
            content = p.read_text()
        except (OSError, UnicodeDecodeError):
            return None

        h = hashlib.md5(content.encode()).hexdigest()[:12]

        if key in self.files and self.files[key].hash == h:
            return self.files[key]

        cached = CachedFile(path=key, content=content, hash=h, size=len(content))
        self.files[key] = cached
        return cached

    def invalidate(self, path: str):
        """Remove a file from the cache (e.g., after editing)."""
        try:
            key = str(Path(path).resolve())
        except OSError:
            key = path
        self.files.pop(key, None)

    def clear(self):
        """Clear all cached files."""
        self.files.clear()

    def set_system_context(self, context: str):
        """Set a system-level prefix (e.g., project description)."""
        self.system_context = context

    def total_size(self) -> int:
        """Total cached content size in characters."""
        return sum(f.size for f in self.files.values()) + len(self.system_context)

    def build_prompt(self, instruction: str,
                     files: list[str] = None) -> str:
        """Build a prompt with cached file context as a shared prefix.

        The prefix (system context + file contents) should match across
        prompts that reference the same files — maximizing KV cache hits.

        Args:
            instruction: The task-specific instruction (goes last)
            files: Specific files to include (default: all cached files)
        """
        parts = []

        if self.system_context:
            parts.append(self.system_context)

        included = files or list(self.files.keys())
        included.sort()

        total_chars = len(self.system_context)
        for path in included:
            if path not in self.files:
                continue
            cached = self.files[path]
            if total_chars + cached.size > self.max_context_chars:
                parts.append(f"\n--- {path} (truncated, {cached.size} chars) ---\n"
                             f"{cached.content[:2000]}...\n")
                total_chars += 2000
            else:
                parts.append(f"\n--- {path} ---\n{cached.content}\n")
                total_chars += cached.size

        parts.append(f"\n{instruction}")

        return "\n".join(parts)

    def build_prompts_batch(self, instructions: list[str],
                            files: list[str] = None) -> list[str]:
        """Build multiple prompts sharing the same file prefix.

        All returned prompts have identical prefixes — Ollama's KV cache
        will reuse the cached prefix computation for the 2nd+ prompt.
        """
        return [self.build_prompt(inst, files) for inst in instructions]

    def stats(self) -> dict:
        return {
            "n_files": len(self.files),
            "total_chars": self.total_size(),
            "files": {
                path: {"size": f.size, "hash": f.hash}
                for path, f in sorted(self.files.items())
            },
        }


_session_cache: ContextCache | None = None


def get_cache() -> ContextCache:
    global _session_cache
    if _session_cache is None:
        _session_cache = ContextCache()
    return _session_cache


def set_cache(cache: ContextCache):
    global _session_cache
    _session_cache = cache
