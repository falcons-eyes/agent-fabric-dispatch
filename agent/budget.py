#!/usr/bin/env python3
"""Budget-aware frontier cost management.

Tracks frontier spending and dynamically adjusts the confidence trust
level as the budget gets consumed. When the budget runs out, the system
stops escalating entirely (equivalent to trust=max).

Budget is configured in ~/.fabric/policy.yaml under `budgets`:
    budgets:
      frontier_daily: 5.00    # USD per day
      frontier_session: 1.00  # USD per session (optional)

The router checks remaining budget before each escalation decision and
adjusts the confidence thresholds accordingly:
  - >75% remaining → use configured trust level
  - 25-75% remaining → shift trust toward aggressive
  - <25% remaining → aggressive
  - 0% remaining → max (never escalate)
"""

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BudgetConfig:
    daily_usd: float = 0.0
    session_usd: float = 0.0

    @classmethod
    def from_policy(cls, policy_path: str = None) -> "BudgetConfig":
        if policy_path is None:
            policy_path = Path.home() / ".fabric" / "policy.yaml"
        else:
            policy_path = Path(policy_path)

        if not policy_path.exists():
            return cls()

        try:
            import yaml
            with open(policy_path) as f:
                policy = yaml.safe_load(f) or {}
        except Exception:
            return cls()

        budgets = policy.get("budgets", {})
        daily = budgets.get("frontier_daily", 0)
        session = budgets.get("frontier_session", 0)

        if isinstance(daily, str):
            daily = _parse_budget_value(daily)
        if isinstance(session, str):
            session = _parse_budget_value(session)

        return cls(daily_usd=float(daily), session_usd=float(session))


def _parse_budget_value(value: str) -> float:
    value = value.strip().lstrip("$")
    multipliers = {"k": 1000, "m": 1_000_000}
    for suffix, mult in multipliers.items():
        if value.lower().endswith(suffix):
            return float(value[:-1]) * mult
    return float(value)


@dataclass
class BudgetTracker:
    config: BudgetConfig
    session_spent: float = 0.0
    _daily_log_path: Path = field(default_factory=lambda: Path.home() / ".fabric" / "budget_log.jsonl")
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_spend(self, cost_usd: float):
        if cost_usd <= 0:
            return
        with self._lock:
            self.session_spent += cost_usd
            self._append_log(cost_usd)

    def remaining_session(self) -> float | None:
        if self.config.session_usd <= 0:
            return None
        return max(0.0, self.config.session_usd - self.session_spent)

    def remaining_daily(self) -> float | None:
        if self.config.daily_usd <= 0:
            return None
        spent_today = self._daily_spent()
        return max(0.0, self.config.daily_usd - spent_today)

    def remaining(self) -> float | None:
        session_rem = self.remaining_session()
        daily_rem = self.remaining_daily()

        if session_rem is None and daily_rem is None:
            return None
        if session_rem is None:
            return daily_rem
        if daily_rem is None:
            return session_rem
        return min(session_rem, daily_rem)

    def budget_ratio(self) -> float:
        """Fraction of budget remaining (1.0 = full, 0.0 = exhausted).

        Returns 1.0 if no budget is configured.
        """
        session_rem = self.remaining_session()
        daily_rem = self.remaining_daily()

        ratios = []
        if session_rem is not None and self.config.session_usd > 0:
            ratios.append(session_rem / self.config.session_usd)
        if daily_rem is not None and self.config.daily_usd > 0:
            ratios.append(daily_rem / self.config.daily_usd)

        if not ratios:
            return 1.0
        return min(ratios)

    def suggested_trust(self, base_trust: str) -> str:
        """Adjust trust level based on remaining budget.

        >75% remaining → base trust
        25-75%         → aggressive
        <25%           → max (skip confidence, never escalate)
        0%             → max
        """
        ratio = self.budget_ratio()

        if ratio >= 0.75:
            return base_trust
        if ratio >= 0.25:
            return "aggressive"
        return "max"

    def can_escalate(self) -> bool:
        remaining = self.remaining()
        if remaining is None:
            return True
        return remaining > 0

    def _daily_spent(self) -> float:
        if not self._daily_log_path.exists():
            return 0.0

        today = time.strftime("%Y-%m-%d")
        total = 0.0
        try:
            with open(self._daily_log_path) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("date") == today:
                            total += entry.get("cost_usd", 0)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass
        return total

    def _append_log(self, cost_usd: float):
        entry = {
            "date": time.strftime("%Y-%m-%d"),
            "timestamp": time.time(),
            "cost_usd": cost_usd,
        }
        try:
            self._daily_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._daily_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass


_tracker: BudgetTracker | None = None


def get_tracker() -> BudgetTracker:
    global _tracker
    if _tracker is None:
        _tracker = BudgetTracker(config=BudgetConfig.from_policy())
    return _tracker


def set_tracker(tracker: BudgetTracker):
    global _tracker
    _tracker = tracker
