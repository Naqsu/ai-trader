"""Perform the final signal validation before execution."""

from __future__ import annotations

from bot.core.config import RiskConfig
from bot.core.models import StrategySignal


class DoubleCheckAgent:
    """Perform the final signal validation before execution."""

    def __init__(self, config: RiskConfig) -> None:
        self.config = config

    def validate(self, signal: StrategySignal) -> tuple[bool, str | None]:
        """Check structural validity of a signal."""
        if signal.risk_per_unit <= 0:
            return False, "invalid_stop_distance"
        if signal.risk_reward < self.config.min_risk_reward:
            return False, "risk_reward_below_minimum"
        if signal.confidence < self.config.min_signal_confidence:
            return False, "signal_confidence_too_low"
        return True, None
