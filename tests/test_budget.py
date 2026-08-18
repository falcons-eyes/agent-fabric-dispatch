#!/usr/bin/env python3
"""Tests for budget-aware frontier cost management."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.budget import BudgetConfig, BudgetTracker, _parse_budget_value


class TestParseBudgetValue:
    def test_plain_number(self):
        assert _parse_budget_value("5.00") == 5.0

    def test_dollar_prefix(self):
        assert _parse_budget_value("$10.00") == 10.0

    def test_k_suffix(self):
        assert _parse_budget_value("500k") == 500_000

    def test_dollar_k(self):
        assert _parse_budget_value("$2k") == 2000


class TestBudgetConfig:
    def test_default_no_budget(self):
        cfg = BudgetConfig()
        assert cfg.daily_usd == 0.0
        assert cfg.session_usd == 0.0

    def test_from_policy_missing_file(self):
        cfg = BudgetConfig.from_policy("/nonexistent/path/policy.yaml")
        assert cfg.daily_usd == 0.0


class TestBudgetTracker:
    def _make_tracker(self, daily=5.0, session=1.0):
        cfg = BudgetConfig(daily_usd=daily, session_usd=session)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            log_path = Path(f.name)
        return BudgetTracker(config=cfg, _daily_log_path=log_path)

    def test_initial_state(self):
        t = self._make_tracker()
        assert t.session_spent == 0.0
        assert t.remaining_session() == 1.0
        assert t.budget_ratio() == 1.0

    def test_record_spend(self):
        t = self._make_tracker(session=2.0)
        t.record_spend(0.5)
        assert t.session_spent == 0.5
        assert t.remaining_session() == 1.5

    def test_remaining_uses_min(self):
        t = self._make_tracker(daily=10.0, session=1.0)
        t.record_spend(0.8)
        assert abs(t.remaining() - 0.2) < 0.001  # session limit is binding

    def test_budget_ratio(self):
        t = self._make_tracker(session=2.0)
        t.record_spend(1.0)
        assert t.budget_ratio() == 0.5

    def test_budget_ratio_no_budget(self):
        t = self._make_tracker(daily=0, session=0)
        assert t.budget_ratio() == 1.0

    def test_can_escalate_true(self):
        t = self._make_tracker(session=1.0)
        assert t.can_escalate() is True

    def test_can_escalate_false_when_exhausted(self):
        t = self._make_tracker(session=0.10)
        t.record_spend(0.10)
        assert t.can_escalate() is False

    def test_can_escalate_no_budget(self):
        t = self._make_tracker(daily=0, session=0)
        assert t.can_escalate() is True

    def test_suggested_trust_full_budget(self):
        t = self._make_tracker(session=10.0)
        assert t.suggested_trust("balanced") == "balanced"

    def test_suggested_trust_half_budget(self):
        t = self._make_tracker(session=2.0)
        t.record_spend(1.0)  # 50% remaining
        assert t.suggested_trust("balanced") == "aggressive"

    def test_suggested_trust_low_budget(self):
        t = self._make_tracker(session=1.0)
        t.record_spend(0.80)  # 20% remaining
        assert t.suggested_trust("balanced") == "max"

    def test_suggested_trust_exhausted(self):
        t = self._make_tracker(session=1.0)
        t.record_spend(1.0)
        assert t.suggested_trust("conservative") == "max"

    def test_record_spend_zero_ignored(self):
        t = self._make_tracker()
        t.record_spend(0.0)
        assert t.session_spent == 0.0

    def test_remaining_none_when_no_budget(self):
        t = self._make_tracker(daily=0, session=0)
        assert t.remaining() is None
        assert t.remaining_session() is None
        assert t.remaining_daily() is None

    def test_daily_log_persistence(self):
        t = self._make_tracker(daily=5.0, session=0)
        t.record_spend(1.50)
        t.record_spend(0.50)

        lines = t._daily_log_path.read_text().strip().split("\n")
        assert len(lines) == 2
        entries = [json.loads(l) for l in lines]
        assert entries[0]["cost_usd"] == 1.50
        assert entries[1]["cost_usd"] == 0.50


class TestBudgetIntegration:
    """Test budget integration with run_single."""

    def test_budget_blocks_frontier_when_exhausted(self):
        from unittest.mock import patch
        from agent.budget import BudgetConfig, BudgetTracker, set_tracker
        from agent.confidence import ConfidenceConfig, set_config
        from agent.agent_fabric import run_single

        set_config(ConfidenceConfig.from_preset("max"))
        cfg = BudgetConfig(session_usd=0.01)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            log_path = Path(f.name)
        tracker = BudgetTracker(config=cfg, session_spent=0.01, _daily_log_path=log_path)
        set_tracker(tracker)

        def mock_ollama(prompt, model=None):
            return {"result": "local answer", "cost_usd": 0, "source": "local"}

        frontier_called = [False]
        def mock_frontier(prompt, max_turns=1):
            frontier_called[0] = True
            return {"result": "frontier", "cost_usd": 0.19, "source": "frontier"}

        with patch("agent.agent_fabric.ollama_generate", mock_ollama):
            with patch("agent.agent_fabric.frontier_query", mock_frontier):
                with patch("agent.agent_fabric.classify_task", return_value="frontier"):
                    result = run_single("test")

        assert result.get("budget_blocked") is True
        assert not frontier_called[0]

        set_tracker(BudgetTracker(config=BudgetConfig()))
        set_config(ConfidenceConfig())


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
