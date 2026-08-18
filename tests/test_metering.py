"""Tests for the metering recorder and summary modules."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from metering.recorder import record_dispatch, record_tool_execution, _append
from metering.summary import compute_summary, format_summary, load_records


@pytest.fixture
def tmp_metering(tmp_path):
    """Create a temporary metering file."""
    metering_file = tmp_path / "metering.jsonl"
    with patch("metering.recorder.METERING_PATH", metering_file):
        with patch("metering.summary.METERING_PATH", metering_file):
            yield metering_file


class TestRecorder:
    def test_record_dispatch(self, tmp_metering):
        record_dispatch(
            tool="refactor_bulk",
            decision="LOCAL",
            reason="rule:1:match",
            tokens_est=500,
        )

        lines = tmp_metering.read_text().strip().split("\n")
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["type"] == "dispatch"
        assert record["tool"] == "refactor_bulk"
        assert record["decision"] == "LOCAL"
        assert record["tokens_est"] == 500

    def test_record_tool_execution(self, tmp_metering):
        record_tool_execution(
            tool="summarize_files",
            duration_ms=1500,
            tokens_input=200,
            tokens_output=100,
            was_local=True,
        )

        lines = tmp_metering.read_text().strip().split("\n")
        record = json.loads(lines[0])
        assert record["type"] == "execution"
        assert record["tool"] == "summarize_files"
        assert record["was_local"] is True


class TestSummary:
    def test_compute_summary_empty(self):
        summary = compute_summary([])
        assert summary["total_dispatches"] == 0
        assert summary["tokens_saved_est"] == 0

    def test_compute_summary_mixed(self):
        records = [
            {"type": "dispatch", "decision": "LOCAL", "tokens_est": 1000, "tool": "refactor_bulk", "reason": "rule:0"},
            {"type": "dispatch", "decision": "LOCAL", "tokens_est": 500, "tool": "summarize_files", "reason": "rule:0"},
            {"type": "dispatch", "decision": "FRONTIER", "tokens_est": 2000, "tool": "complex_task", "reason": "default"},
            {"type": "execution", "was_local": True, "duration_ms": 1200, "tool": "refactor_bulk"},
            {"type": "execution", "was_local": False, "duration_ms": 800, "tool": "complex_task"},
        ]

        summary = compute_summary(records)
        assert summary["total_dispatches"] == 3
        assert summary["local_dispatches"] == 2
        assert summary["frontier_dispatches"] == 1
        assert summary["tokens_saved_est"] == 1500
        assert summary["local_executions"] == 1

    def test_format_summary(self):
        summary = {
            "total_dispatches": 14,
            "local_dispatches": 9,
            "frontier_dispatches": 5,
            "tokens_saved_est": 1_200_000,
            "sensitive_egress": 0,
            "decision_counts": {"LOCAL": 9, "FRONTIER": 5},
            "tool_counts": {"refactor_bulk": 6, "summarize_files": 5, "classify": 3},
            "total_executions": 14,
            "local_executions": 9,
            "total_duration_ms": 45000,
        }

        output = format_summary(summary)
        assert "14 dispatches" in output
        assert "9 local" in output
        assert "5 frontier" in output
        assert "1.2M tokens" in output
        assert "sensitive egress 0" in output
